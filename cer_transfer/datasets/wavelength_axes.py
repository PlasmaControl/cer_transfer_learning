"""Rebuild the wavelength axis (nm) of preprocessed discharge files.

The DIII-D joblib files store spectra on a per-shot master wavelength axis
but not the axis itself. The preprocessing (preprocess_cer_spectrograms.ipynb)
built it from the raw per-chord axes in ``<shot>_cer_spectra.h5``: from the
largest to the smallest wavelength over all chords, at the finest step of
any chord, in descending order. This script repeats that construction for
every listed joblib file and writes one row per file:

    file, shot, lam_first, lam_last, n_bins, n_bins_joblib, match

``lam_first``/``lam_last`` are the wavelengths of the first/last bin (nm),
so the orientation follows from their order. ``match`` is False when the
rebuilt bin count differs from the joblib's (e.g. a shot processed with a
different axis construction); such rows must not be used.

    pixi run python -m cer_transfer.datasets.wavelength_axes \\
        --list splits/d3d_val.txt --raw-dir /scratch/gpfs/EKOLEMEN/ps9551/cer_spectra \\
        --out data/d3d_wavelength_axes.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from joblib import load

from cer_transfer.configs import data_path


def chord_axes(h5_path, chord_filter, with_names=False):
    """Per-chord wavelength axes (nm) from a raw CER spectra HDF5 file."""
    import h5py

    axes, names = [], []
    with h5py.File(str(h5_path), "r") as h5:
        for name in sorted(h5["chords"].keys()):
            if not chord_filter(name):
                continue
            ax = np.asarray(h5["chords"][name]["feature_axis"][()], dtype=np.float64)
            if ax.size >= 2 and not np.all(np.mod(ax, 1) == 0):
                axes.append(ax)
                names.append(name)
    return (axes, names) if with_names else axes


def notebook_chord_filter(name):
    """Chords used in the preprocessing: t01-t48 and v01-v32, no 'm' chords."""
    if name.startswith("m"):
        return False
    if name.startswith("t") and int(name[1:]) > 48:
        return False
    if name.startswith("v") and int(name[1:]) > 32:
        return False
    return True


def master_axis(axes):
    """Union of the chord ranges at the finest step, descending (notebook)."""
    starts = [max(a[0], a[-1]) for a in axes]
    ends = [min(a[0], a[-1]) for a in axes]
    steps = [abs(a[-1] - a[0]) / (len(a) - 1) for a in axes]
    gmax, gmin, step = max(starts), min(ends), min(steps)
    n = max(2, int(round((gmax - gmin) / step)) + 1)
    return np.linspace(gmax, gmin, n)


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="file list (CER_DATA_ROOT-relative)")
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--summary",
        type=int,
        default=0,
        help="print the per-chord axes (first/last nm, pixels, step) of this many "
        "shots and exit",
    )
    args = ap.parse_args()
    entries = [
        ln.strip()
        for ln in open(args.list).read().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if args.limit:
        entries = entries[: args.limit]
    if args.summary:
        for e in entries[: args.summary]:
            shot = re.findall(r"\d{6}", Path(e).stem)
            raw = args.raw_dir / f"{shot[-1]}_cer_spectra.h5"
            axes, names = chord_axes(raw, notebook_chord_filter, with_names=True)
            m = master_axis(axes)
            print(f"== {shot[-1]}: master {m[0]:.3f} -> {m[-1]:.3f} nm, {len(m)} bins")
            for n_, a in zip(names, axes):
                step = (a[-1] - a[0]) / (len(a) - 1)
                print(
                    f"  {n_}: {a[0]:8.3f} -> {a[-1]:8.3f} nm  {len(a):4d} px  "
                    f"step {step:+.5f} nm"
                )
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_bad = n_missing = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "file",
                "shot",
                "lam_first",
                "lam_last",
                "n_bins",
                "n_bins_joblib",
                "match",
            ]
        )
        for e in entries:
            shot = re.findall(r"\d{6}", Path(e).stem)
            raw = args.raw_dir / f"{shot[-1]}_cer_spectra.h5" if shot else None
            if raw is None or not raw.exists():
                n_missing += 1
                continue
            axes = chord_axes(raw, notebook_chord_filter)
            if not axes:
                n_missing += 1
                continue
            m = master_axis(axes)
            n_job = load(data_path(e), mmap_mode="r")["input"].shape[-1]
            ok = len(m) == n_job
            n_ok += ok
            n_bad += not ok
            w.writerow([e, shot[-1], f"{m[0]:.4f}", f"{m[-1]:.4f}", len(m), n_job, ok])
    print(
        f"wrote {args.out}: {n_ok} matching, {n_bad} bin-count mismatches, "
        f"{n_missing} without raw file"
    )


if __name__ == "__main__":
    main()


def _norm(v):
    v = v - np.median(v)
    s = np.abs(v).max()
    return v / s if s > 0 else v


def estimate_axis(job_path, raw_path, channels=range(24), n_best=8):
    """Fit the wavelength axis of a joblib file to the raw spectra.

    For each of the strongest chords among ``channels`` (channel i is chord
    t{i+1:02d}, as in the preprocessing), the time-averaged joblib spectrum
    is compared with the time-averaged raw spectrum on its nm axis. The joblib
    grid is modelled as lam_i = first + i * step; ``first`` and ``step`` are
    found by a coarse-to-fine search maximizing the mean correlation over
    the chords, using only bins inside each chord's own window.

    Returns
    -------
    (first, last, corr)
        Wavelength of the first and last bin (nm) and the mean correlation
        at the optimum (close to 1 for a correct axis).
    """
    import h5py

    z = load(job_path, mmap_mode="r")
    x = z["input"]
    end = int(z["end_index"]) if "end_index" in z else -1
    t_end = x.shape[1] if end <= 0 else min(end, x.shape[1])
    W = x.shape[-1]
    pairs = []
    with h5py.File(str(raw_path), "r") as h5:
        for c in channels:
            name = f"t{c + 1:02d}"
            if name not in h5["chords"]:
                continue
            ax = np.asarray(h5["chords"][name]["feature_axis"][()], np.float64)
            dat = np.asarray(h5["chords"][name]["data"][()], np.float64)
            if ax.size < 10 or np.all(np.mod(ax, 1) == 0):
                continue
            o = np.argsort(ax)
            r = _norm(np.nanmean(dat, axis=0)[o])
            s = _norm(np.nanmean(np.asarray(x[c, :t_end], np.float64), axis=0))
            contrast = np.max(s) - np.median(s)
            pairs.append((contrast, ax[o], r, s))
    if not pairs:
        return None
    pairs = sorted(pairs, key=lambda p: -p[0])[:n_best]
    lo = min(p[1][0] for p in pairs) - 2.0
    hi = max(p[1][-1] for p in pairs) + 2.0
    idx = np.arange(W)

    def score(firsts, step):
        # the preprocessing interpolated each chord onto the common grid and
        # filled bins outside the chord's window with its edge values
        # (np.interp); reproduce exactly that and compare all bins
        lam = firsts[:, None] + idx[None, :] * step  # (n_first, W)
        tot = np.zeros(len(firsts))
        for _, axr, r, s in pairs:
            inside = (lam >= axr[0]) & (lam <= axr[-1])
            ok = inside.sum(1) > 0.2 * W
            pred = np.interp(lam, axr, r)
            pc = pred - pred.mean(1, keepdims=True)
            sc = s[None, :] - s.mean()
            corr = (pc * sc).sum(1) / np.sqrt((pc**2).sum(1) * (sc**2).sum() + 1e-12)
            tot += np.where(ok, corr, -1.0)
        return tot / len(pairs)

    coarse = []  # best first per trial step
    for step in np.r_[np.linspace(-0.04, -0.008, 129), np.linspace(0.008, 0.04, 129)]:
        span = step * (W - 1)
        f0, f1 = (lo - span, hi) if step < 0 else (lo, hi - span)
        firsts = np.arange(min(f0, f1), max(f0, f1), 0.01)
        if firsts.size == 0:
            continue
        sc = score(firsts, step)
        k = int(np.argmax(sc))
        coarse.append((sc[k], firsts[k], step))
    coarse.sort(key=lambda c: -c[0])
    results = []
    for _, first, step in coarse[:10]:  # refine the ten best candidates
        for rel, dfirst in ((0.10, 0.6), (0.02, 0.12), (0.004, 0.025), (0.001, 0.005)):
            cand = []
            for st in step * np.linspace(1 - rel, 1 + rel, 41):
                firsts = first + np.linspace(-dfirst, dfirst, 61)
                sc = score(firsts, st)
                k = int(np.argmax(sc))
                cand.append((sc[k], firsts[k], st))
            corr, first, step = max(cand)
        results.append((corr, first, step))
    corr, first, step = max(results)
    return first, first + step * (W - 1), corr
