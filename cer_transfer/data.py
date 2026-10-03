"""Config-driven dataset and samplers.

No machine-specific literals here. All channel selection and preprocessing
parameters come from MachineConfig, so DIII-D and NSTX data go through the
identical code path.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from joblib import load
from torch.utils.data import Dataset, Sampler, get_worker_info

from .configs import MachineConfig


class Preprocessor:
    """Shared spectrogram preprocessing, matching the original pipeline:

      1. offset removal: per-channel minimum over the WHOLE subsequence
         (axes T and W together) — i.e. `x -= x.min(axis=(1, 2)) - 1`.
         NOTE: this makes preprocessing depend on the subsequence a
         timestep lands in (train chunks vs full-shot validation see
         slightly different offsets); kept deliberately to match the
         original, empirically fast/validated pipeline.
      2. log10 in torch on CPU (torch.log10 is faster than np.log10 in
         dataloader workers).

    The wavelength axis is passed through untouched; variable W between
    shots is handled by per-shot batching + the encoder's adaptive pooling.
    """

    def __init__(
        self,
        machine: MachineConfig,
        resample_w: Optional[int] = None,
        w_scale: float = 1.0,
        flip_w: bool = False,
    ):
        self.baseline_region = machine.baseline_region
        self.resample_w = resample_w
        # axis-convention probes for zero-shot transfer: stretch the
        # wavelength axis by w_scale (resample to w_scale * resample_w bins,
        # then centre-crop or edge-pad back to resample_w) and/or reverse it
        # combined with the machine's convention relative to the NSTX reference
        # (flip, dispersion ratio); without resample_w nothing is applied, so
        # the default pipeline is unchanged
        self.w_scale = w_scale * (machine.wavelength_scale if resample_w else 1.0)
        self.flip_w = flip_w != (bool(machine.wavelength_flip) if resample_w else False)

    def __call__(self, spec: np.ndarray) -> torch.Tensor:
        """spec: (C, T, W) float array -> log10 spectrogram tensor."""
        t = torch.from_numpy(spec)
        if self.baseline_region is not None:
            lo, hi = self.baseline_region
            offset = torch.amin(t[..., lo:hi], dim=(1, 2), keepdim=True)
        else:
            offset = torch.amin(t, dim=(1, 2), keepdim=True)  # per channel
        t = t - (offset - 1.0)
        t = torch.log10(t)
        if self.flip_w:
            t = torch.flip(t, dims=(-1,))
        if self.resample_w is not None:
            c, n, w = t.shape
            target = max(int(round(self.resample_w * self.w_scale)), 2)
            if target != w:
                # plain linear interpolation along the wavelength axis
                t = torch.nn.functional.interpolate(
                    t.reshape(c * n, 1, w),
                    size=target,
                    mode="linear",
                    align_corners=False,
                ).reshape(c, n, target)
            if target > self.resample_w:  # centre crop
                a = (target - self.resample_w) // 2
                t = t[..., a : a + self.resample_w]
            elif target < self.resample_w:  # edge pad
                pad = self.resample_w - target
                t = torch.nn.functional.pad(
                    t, (pad // 2, pad - pad // 2), mode="replicate"
                )
        return t


class ShotDataset(Dataset):
    """One joblib file per shot with keys 'input' (C, T, W), 'target'
    (C, T, n_targets), 'target_error' (same), 'end_index' (valid T).

    Splits each shot into non-overlapping subsequences of length subseq_len
    (-1 = whole shot). Files are mmap-opened lazily per worker.
    """

    def __init__(
        self,
        file_paths: Sequence[Path],
        machine: MachineConfig,
        subseq_len: int,
        mmap: bool = True,
        max_label_relerr: float = 0.0,
        zero_dark_frames: bool = False,
        resample_w: Optional[int] = None,
        w_scale: float = 1.0,
        flip_w: bool = False,
    ):
        """mmap=True: memory-map files, materialize slices via page faults
        (random small reads — latency-bound on GPFS). mmap=False: read each
        shot's file sequentially in full on first access (the pattern GPFS
        prefers), keep only the most recent shot per worker (batches are
        per-shot, so an LRU of size 1 gets all hits within a batch)."""
        super().__init__()
        self.file_paths = [Path(p) for p in file_paths]
        self.machine = machine
        self.subseq_len = subseq_len
        self.mmap = mmap
        self.preprocess = Preprocessor(machine, resample_w, w_scale, flip_w)

        self._input_idx = (
            None
            if machine.input_channel_indices is None
            else np.asarray(machine.input_channel_indices)
        )
        self._chord_idx = (
            None if machine.chord_indices is None else np.asarray(machine.chord_indices)
        )
        # unit conversion into canonical units (ti: eV, vtor: km/s),
        # broadcast over (C, T, n_targets); identity when all ones
        ts = np.asarray(machine.target_scale, dtype=np.float32)
        self._target_scale = None if np.all(ts == 1.0) else ts
        # optional training label QC: mask (-> NaN) labels whose fit
        # uncertainty is huge in relative terms (end-of-shot/failed fits);
        # masked loss ignores NaNs.
        self.max_label_relerr = max_label_relerr
        self.zero_dark_frames = zero_dark_frames

        self._opened_files: Optional[dict] = None

        if not self.file_paths:
            raise ValueError(
                "ShotDataset got an empty file list — check the "
                "data directory / glob pattern"
            )

        # Build the subsequence index by opening each file once to read
        # end_index (handle dropped immediately; same reads as the original
        # serial loop). joblib.load is latency-bound on parallel filesystems
        # (unpickling touches scattered offsets -> several round-trips per
        # file), so the opens run in a thread pool; IO releases the GIL.
        _t0 = time.time()

        def _end_index(fp):
            dd = load(fp, mmap_mode="r")
            n = int(dd["end_index"])
            del dd
            return n

        with ThreadPoolExecutor(max_workers=32) as ex:
            n_samples_all = list(ex.map(_end_index, self.file_paths))

        self.subseq_info: list[tuple[int, int]] = []
        for f_idx, n_samples in enumerate(n_samples_all):
            if self.subseq_len == -1:
                self.subseq_info.append((f_idx, 0))
            else:
                for chunk in range(n_samples // self.subseq_len):
                    self.subseq_info.append((f_idx, chunk * self.subseq_len))
        _dt = time.time() - _t0
        print(
            f"indexed {len(self.file_paths)} files in {_dt:.1f}s "
            f"({1e3 * _dt / len(self.file_paths):.1f} ms/file effective)",
            flush=True,
        )
        if self.subseq_len != -1:
            contributing = {f for f, _ in self.subseq_info}
            n_empty = len(self.file_paths) - len(contributing)
            if n_empty:
                print(
                    f"WARNING: {n_empty}/{len(self.file_paths)} shots "
                    f"yield NO subsequences (end_index < subseq_len="
                    f"{self.subseq_len}) and are effectively excluded — "
                    f"consider a smaller --subseq-len",
                    flush=True,
                )

    def __len__(self) -> int:
        return len(self.subseq_info)

    def worker_init(self) -> None:
        # Lazy strategy: no upfront opens. Each worker opens a file on first
        # access and caches the handle. Spreads ~N_files opens over the first
        # epoch's batches instead of a multi-minute silent wall per worker —
        # and per epoch, if workers are not persistent.
        """Reset per-worker file caches (called from ``worker_init_fn``)."""
        self._opened_files = {}

    def _get_file(self, f_idx: int):
        if self.mmap:
            d = self._opened_files.get(f_idx)
            if d is None:
                d = load(self.file_paths[f_idx], mmap_mode="r")
                # joblib silently ignores mmap for compressed files; the
                # handle cache would then hoard fully-loaded dicts and grow
                # without bound over an epoch. Refuse rather than OOM.
                arr = d["input"]
                if not (
                    isinstance(arr, np.memmap)
                    or getattr(arr, "base", None) is not None
                    and isinstance(arr.base, np.memmap)
                ):
                    raise RuntimeError(
                        f"{self.file_paths[f_idx]} appears to be compressed "
                        "(mmap not honored). Use --no-mmap for compressed "
                        "datasets, or re-save uncompressed."
                    )
                self._opened_files[f_idx] = d
            return d
        # full sequential read; cache ONLY the latest shot (caching more
        # would accumulate whole files in RAM)
        d = self._opened_files.get(f_idx)
        if d is None:
            self._opened_files.clear()
            d = load(self.file_paths[f_idx])  # no mmap: sequential read
            self._opened_files[f_idx] = d
        return d

    def __getitem__(self, idx: int):
        if self._opened_files is None:
            # covers num_workers=0 too
            self.worker_init()

        f_idx, start = self.subseq_info[idx]
        d = self._get_file(f_idx)
        end = None if self.subseq_len == -1 else start + self.subseq_len

        spec = np.array(d["input"][:, start:end, :], dtype=np.float32)
        target = np.array(d["target"][:, start:end, :], dtype=np.float32)
        error = np.array(d["target_error"][:, start:end, :], dtype=np.float32)

        if self._target_scale is not None:
            target = target * self._target_scale
            error = error * self._target_scale
        if self.max_label_relerr > 0:
            bad = ~(
                np.abs(error)
                <= self.max_label_relerr * np.maximum(np.abs(target), 1e-12)
            )
            if bad.any():
                target = np.where(bad, np.nan, target)

        if self.zero_dark_frames:
            # Physics pseudo-labels: on DARK frames -- no emission above
            # the pedestal on ANY chord -- the true ion temperature and
            # rotation are zero (no plasma). Frames that merely lack CX
            # light (beam off, plasma present) do NOT qualify: their
            # per-chord amplitudes stay well above the dark level.
            from cer_transfer.dark import dark_mask

            # shared criterion (cer_transfer.dark): outside the labeled
            # span (labels imply plasma, plasma persists) AND below a
            # noise-anchored amplitude floor; beam-off-with-plasma frames
            # lie inside the span and can never qualify
            dark = dark_mask(spec, target)
            if dark.any():
                unl = np.isnan(target)
                zsig = np.full_like(error, np.nan)
                # modest confidence: sigma = machine-configured dark
                # sigmas, broadcast over targets
                for ti_, s_ in enumerate(self.machine.dark_sigma):
                    zsig[:, :, ti_] = s_ * (
                        self._target_scale[ti_]
                        if self._target_scale is not None
                        else 1.0
                    )
                m = unl & dark[None, :, None]
                target = np.where(m, 0.0, target)
                error = np.where(m, zsig, error)

        # classical per-(chord, frame) spectral moments on RAW counts at
        # native resolution — the information the pooled trunk degrades.
        med = np.median(spec, axis=-1, keepdims=True)
        wts = np.clip(spec - med, 0.0, None)
        m0 = wts.sum(axis=-1) + 1e-6
        W_ = spec.shape[-1]
        xg = (np.arange(W_, dtype=np.float32) - W_ / 2.0) / W_
        m1 = (wts * xg).sum(axis=-1) / m0
        var = (wts * (xg[None, None, :] - m1[..., None]) ** 2).sum(axis=-1) / m0
        m2 = np.sqrt(np.clip(var, 0.0, None))
        # rough O(1) scaling: log-amplitude/4, centroid*4, width*20
        moments = np.stack(
            [np.log10(m0 + 1.0) / 4.0, m1 * 4.0, m2 * 20.0], axis=-1
        ).astype(np.float32)

        spec_t = self.preprocess(spec)

        if self._input_idx is not None:
            spec_t = spec_t[self._input_idx]
        if self._chord_idx is not None:
            target = target[self._chord_idx]
            error = error[self._chord_idx]
            moments = moments[self._chord_idx]

        return (
            spec_t,
            torch.from_numpy(target),
            torch.from_numpy(error),
            torch.from_numpy(moments),
        )


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker initializer: resets the dataset's per-worker caches.

    Parameters
    ----------
    worker_id : int
        Worker index assigned by the DataLoader.
    """
    get_worker_info().dataset.worker_init()


def compute_tail_weights(dataset, gamma: float, quantile: float = 0.90) -> dict:
    """{f_idx: weight} with weight = (q_shot / median_q)^gamma, where q_shot
    is the per-shot `quantile` of |vtor| labels (canonical km/s). gamma=0 ->
    uniform. Cheap pass: reads target arrays only."""
    from joblib import load as _jload

    scale = float(np.asarray(dataset.machine.target_scale)[1])
    chord_idx = (
        None
        if dataset.machine.chord_indices is None
        else np.asarray(dataset.machine.chord_indices)
    )
    q = {}
    for f_idx, fp in enumerate(dataset.file_paths):
        d = _jload(fp, mmap_mode="r" if dataset.mmap else None)
        end = int(d["end_index"])
        if end <= 0:
            q[f_idx] = 0.0
            continue
        v = np.asarray(d["target"][:, :end, 1], dtype=np.float64)
        if chord_idx is not None:
            v = v[chord_idx]
        v = np.abs(v * scale)
        q[f_idx] = float(np.nanquantile(v, quantile)) if np.isfinite(v).any() else 0.0
    med = np.median([x for x in q.values() if x > 0]) or 1.0
    return {f: max((x / med), 1e-3) ** gamma for f, x in q.items()}


class PerShotBatchSampler(Sampler):
    """Batches of subsequences drawn from a single shot; shot order and
    within-shot order shuffled each epoch (seedable via a torch.Generator
    for reproducible HPO)."""

    def __init__(
        self,
        dataset: ShotDataset,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        shot_weights: Optional[dict] = None,
    ):
        """shot_weights: optional {f_idx: weight}; when given, each epoch
        draws len(file_list) shots WITH replacement, probability
        proportional to weight (tail oversampling for imbalanced
        regression). Epoch length is preserved in expectation."""
        self.batch_size = batch_size
        self.generator = generator
        self.file_to_indices: dict[int, list[int]] = {}
        for g_idx, (f_idx, _) in enumerate(dataset.subseq_info):
            self.file_to_indices.setdefault(f_idx, []).append(g_idx)
        self.file_list = list(self.file_to_indices)
        self.shot_weights = None
        if shot_weights:
            w = torch.tensor(
                [float(shot_weights.get(f, 1.0)) for f in self.file_list],
                dtype=torch.float64,
            )
            self.shot_weights = torch.clamp(w, min=0) + 1e-12

    def _shuffled(self, seq):
        seq = list(seq)
        if self.generator is not None:
            perm = torch.randperm(len(seq), generator=self.generator).tolist()
            return [seq[i] for i in perm]
        random.shuffle(seq)
        return seq

    def __iter__(self):
        if self.shot_weights is not None:
            order = torch.multinomial(
                self.shot_weights,
                len(self.file_list),
                replacement=True,
                generator=self.generator,
            ).tolist()
            shots = [self.file_list[i] for i in order]
        else:
            shots = self._shuffled(self.file_list)
        for f_idx in shots:
            idxs = self._shuffled(self.file_to_indices[f_idx])
            for s in range(0, len(idxs), self.batch_size):
                yield idxs[s : s + self.batch_size]

    def __len__(self) -> int:
        return sum(
            (len(v) + self.batch_size - 1) // self.batch_size
            for v in self.file_to_indices.values()
        )


class FullShotSampler(Sampler):
    """One batch per shot containing all its subsequences (validation)."""

    def __init__(self, dataset: ShotDataset):
        self.file_to_indices: dict[int, list[int]] = {}
        for g_idx, (f_idx, _) in enumerate(dataset.subseq_info):
            self.file_to_indices.setdefault(f_idx, []).append(g_idx)

    def __iter__(self):
        for idxs in self.file_to_indices.values():
            yield idxs

    def __len__(self) -> int:
        return len(self.file_to_indices)


def split_files(files: Sequence[Path], val_fraction: float, seed: int):
    """Deterministic shot-level split (never mixes one shot across splits)."""
    files = sorted(Path(p) for p in files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_fraction))
    return files[n_val:], files[:n_val]
