"""Per-chord instrument width of a CER/CHERS spectrometer, estimated from spectra.

Two estimators, both per chord:

1. Lower envelope (label-free, works on any machine): a low percentile of the
   per-frame line width over many frames and discharges. It bounds the
   instrument width from above by the thermal width of the coldest frames the
   channel records: for C VI 529.05 nm, sigma_th = 0.035 nm (1.4 NSTX px) at
   50 eV and 0.158 nm at 1 keV, so the envelope is sigma_inst only where
   frames at a few tens of eV exist (edge chords, beam-off, start/end of the
   discharge); otherwise it is biased high by up to the coldest sigma_th.
2. Label-anchored (source machine, where conventional fits exist): with the
   Doppler relation sigma_obs^2 = sigma_inst^2 + T_i / K, K = m c^2 / lambda_0^2,
   the median of sigma_obs^2 - T_i / K over labeled frames is sigma_inst^2,
   independent of how hot the frames are. Printed whenever the files carry
   labels; this is the number to use for the training machine, and the check
   of estimator 1 wherever both exist.

Per frame: background = 10th percentile of the spectrum; width = FWHM of the
background-subtracted line with linear sub-pixel interpolation at half
maximum, as a Gaussian sigma (FWHM / 2.3548). Frames with peak S/N below
``--snr-min`` or a line touching the window edge are dropped. Blended frames
(active + passive above half maximum) come out wider and never set the
envelope; for estimator 2 they bias sigma_inst^2 upward, hence the median.

Units: nm (``--window-nm`` = wavelength span of the files' window, 2.08 nm
NSTX, 6.0 nm DIII-D), native pixels, and bins of the common 256-bin NSTX
reference grid (2.08 nm / 256), the unit in which the model sees widths.

    pixi run --frozen python -u -m cer_transfer.analysis.instrument_width \\
        --machine nstx --list splits/nstx_train.txt --max-files 200 \\
        --csv gallery/instrument_width_nstx.csv
    pixi run --frozen python -u -m cer_transfer.analysis.instrument_width \\
        --machine d3d_tangential --max-files 200 --csv gallery/instrument_width_d3d.csv
    # (DIII-D has no list files: the machine's train_dir is scanned; --dirs overrides)

Sanity: widths should be similar across the chords of one spectrometer and
above ~0.5 px; a value below that is set by noise (raise --snr-min).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from joblib import load

from cer_transfer.configs import data_path, get_machine

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # 1 / 2.3548
AMU_EV = 931.494_102e6  # eV / c^2
LAMBDA0_NM, MASS_AMU = 529.05, 12.0  # C VI
K_EV_PER_NM2 = MASS_AMU * AMU_EV / LAMBDA0_NM**2  # T_i = K * sigma_th^2
REF_WINDOW_NM, REF_BINS = 2.08, 256  # NSTX CHERS window on the common grid
DEFAULT_WINDOW_NM = {"nstx": 2.08, "d3d": 6.0}  # by machine-name prefix


def read_list(paths):
    """Entries of one or more split lists (comments and blanks skipped)."""
    out = []
    for p in paths:
        for ln in Path(p).read_text().splitlines():
            s = ln.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def fwhm_interp(line):
    """Interpolated FWHM (pixels) of each row of non-negative line profiles.

    Returns (fwhm, valid): valid is False where the half-maximum region
    touches the window edge.
    """
    n, w = line.shape
    peak = line.max(-1)
    half = 0.5 * peak
    above = line >= half[:, None]
    first = np.argmax(above, -1)
    last = w - 1 - np.argmax(above[:, ::-1], -1)
    valid = (first > 0) & (last < w - 1) & (peak > 0)
    first_c = np.clip(first, 1, w - 1)
    last_c = np.clip(last, 0, w - 2)
    rows = np.arange(n)
    y0, y1 = line[rows, first_c - 1], line[rows, first_c]
    frac_l = np.where(y1 > y0, (half - y0) / np.maximum(y1 - y0, 1e-12), 0.0)
    x_left = first_c - 1 + np.clip(frac_l, 0.0, 1.0)
    z0, z1 = line[rows, last_c], line[rows, last_c + 1]
    frac_r = np.where(z0 > z1, (z0 - half) / np.maximum(z0 - z1, 1e-12), 0.0)
    x_right = last_c + np.clip(frac_r, 0.0, 1.0)
    return x_right - x_left, valid


def frame_widths(spec, snr_min):
    """Per-row Gaussian sigma in pixels and S/N for (N, W) spectra."""
    spec = np.asarray(spec, dtype=np.float64)
    bg = np.percentile(spec, 10, axis=-1, keepdims=True)
    line = np.clip(spec - bg, 0.0, None)
    # noise: robust spread of the lowest half of the background-subtracted bins
    low = np.sort(line, axis=-1)[:, : max(line.shape[-1] // 2, 2)]
    noise = 1.4826 * np.median(np.abs(low - np.median(low, -1, keepdims=True)), -1)
    noise = np.maximum(noise, 1e-9)
    snr = line.max(-1) / noise
    fw, valid = fwhm_interp(line)
    ok = valid & (snr >= snr_min) & np.isfinite(fw) & (fw > 0)
    return fw * FWHM_TO_SIGMA, snr, ok


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--machine", required=True)
    ap.add_argument("--list", nargs="+", default=None, help="split list(s)")
    ap.add_argument(
        "--dirs",
        nargs="+",
        default=None,
        help="folders of discharge files under CER_DATA_ROOT (default for machines "
        "with a train_dir, e.g. d3d_tangential -> training_set_30)",
    )
    ap.add_argument("--max-files", type=int, default=200, help="evenly spaced files read")
    ap.add_argument("--stride", type=int, default=4, help="use every k-th frame")
    ap.add_argument("--snr-min", type=float, default=20.0, help="peak S/N threshold")
    ap.add_argument(
        "--percentile", type=float, default=2.0, help="lower envelope percentile"
    )
    ap.add_argument(
        "--window-nm",
        type=float,
        default=None,
        help="wavelength span of the files' window (default by machine: "
        + ", ".join(f"{k} {v}" for k, v in DEFAULT_WINDOW_NM.items()),
    )
    ap.add_argument(
        "--bins", type=int, default=None,
        help="use only files with this native bin count (e.g. 390 or 536 on DIII-D)",
    )
    ap.add_argument("--csv", type=Path, default=None, help="per-chord table")
    args = ap.parse_args()

    machine = get_machine(args.machine)
    window_nm = args.window_nm
    if window_nm is None:
        for k, v in DEFAULT_WINDOW_NM.items():
            if machine.name.startswith(k):
                window_nm = v
        if window_nm is None:
            raise SystemExit("--window-nm required for this machine")
    chords = (
        np.arange(machine.n_chords)
        if machine.chord_indices is None
        else np.asarray(machine.chord_indices)
    )

    if args.list:
        entries = read_list(args.list)
    else:
        dirs = args.dirs or ([str(machine.train_dir)] if machine.train_dir else None)
        if not dirs:
            raise SystemExit("--list or --dirs required for this machine")
        entries = []
        for d in dirs:
            pd = data_path(d)
            entries += [str(Path(d) / f) for f in sorted(os.listdir(pd)) if f.endswith(".joblib")]
        print(f"{len(entries)} files in {dirs}")
    if args.bins is not None:
        keep = []
        for e in entries:
            try:
                if load(data_path(e), mmap_mode="r")["input"].shape[-1] == args.bins:
                    keep.append(e)
            except Exception:
                pass
        print(f"{len(keep)}/{len(entries)} files with {args.bins} bins")
        entries = keep
    if len(entries) > args.max_files:
        idx = np.linspace(0, len(entries) - 1, args.max_files).astype(int)
        entries = [entries[i] for i in idx]

    per_chord = [[] for _ in chords]
    anchored = [[] for _ in chords]  # sigma_obs^2 - T_i/K on labeled frames [nm^2]
    scale = np.asarray(machine.target_scale, dtype=np.float64)
    n_frames = n_kept = 0
    widths_px = {}
    for i, e in enumerate(entries):
        fp = data_path(e)
        try:
            dd = load(fp, mmap_mode="r")
        except Exception as exc:  # unreadable file: report and continue
            print(f"skip {e}: {exc}")
            continue
        T = int(dd["end_index"])
        x = np.asarray(dd["input"][:, : T : args.stride, :], dtype=np.float32)[chords]
        W = x.shape[-1]
        widths_px.setdefault(W, 0)
        widths_px[W] += 1
        C, Tn, _ = x.shape
        sig, snr, ok = frame_widths(x.reshape(C * Tn, W), args.snr_min)
        sig, ok = sig.reshape(C, Tn), ok.reshape(C, Tn)
        n_frames += C * Tn
        n_kept += int(ok.sum())
        ti = None
        if "target" in dd:
            ti = np.asarray(dd["target"][:, : T : args.stride, 0], np.float64)[chords] * scale[0]
        for c in range(C):
            s = sig[c][ok[c]] * (window_nm / W)  # nm; this file's bin count
            if s.size:
                per_chord[c].append(s)
            if ti is not None:
                lab = np.isfinite(ti[c]) & ok[c]
                if lab.any():
                    s_lab = sig[c][lab] * (window_nm / W)
                    anchored[c].append(s_lab**2 - ti[c][lab] / K_EV_PER_NM2)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(entries)} files, {n_kept}/{n_frames} frames kept")

    ref_nm_per_bin = REF_WINDOW_NM / REF_BINS
    main_w = max(widths_px, key=widths_px.get) if widths_px else float("nan")
    print(
        f"\n{machine.name}: {len(entries)} files, native bins {dict(widths_px)}, "
        f"window {window_nm} nm ({window_nm / main_w:.5f} nm/px for {main_w} bins); "
        f"{n_kept}/{n_frames} chord-frames with S/N >= {args.snr_min}"
    )
    print(
        f"{'chord':>5} {'n':>7} {'p'+str(args.percentile):>9} {'p10':>8} {'median':>8}"
        f"  | sigma_inst  [nm]   [px@{main_w}]  [ref bins]"
    )
    rows = []
    for c, lst, anc in zip(chords, per_chord, anchored):
        if not lst:
            print(f"{c:>5} {0:>7}      -")
            rows.append((int(c), 0, np.nan, np.nan, np.nan, np.nan, 0))
            continue
        v = np.concatenate(lst)
        p_env = float(np.percentile(v, args.percentile))
        p10, med = float(np.percentile(v, 10)), float(np.median(v))
        s_anc, n_anc = np.nan, 0
        if anc:
            r = np.concatenate(anc)
            n_anc = r.size
            s_anc = float(np.sqrt(max(np.median(r), 0.0)))
        rows.append((int(c), v.size, p_env, p10, med, s_anc, n_anc))
        print(
            f"{c:>5} {v.size:>7} {p_env:9.5f} {p10:8.5f} {med:8.5f}  | "
            f"{p_env:.5f} nm  {p_env / (window_nm / main_w):6.2f} px  "
            f"{p_env / ref_nm_per_bin:6.2f}"
            + (f"  | anchored {s_anc:.5f} nm ({n_anc} labeled)" if n_anc else "")
        )
    env = np.array([r[2] for r in rows])
    good = np.isfinite(env)
    if good.any():
        print(
            f"\nmedian over chords: sigma_inst = {np.median(env[good]):.5f} nm = "
            f"{np.median(env[good]) / ref_nm_per_bin:.2f} reference bins; "
            f"chord spread (p10-p90) {np.percentile(env[good], 10):.5f}-"
            f"{np.percentile(env[good], 90):.5f} nm"
        )
        if (env[good] / (window_nm / main_w) < 0.5).any():
            print("WARNING: sigma below 0.5 px on some chords -> noise-limited; raise --snr-min")
        anc = np.array([r[5] for r in rows])
        ga = np.isfinite(anc)
        if ga.any():
            print(
                f"label-anchored (median sigma_obs^2 - T_i/K): sigma_inst = "
                f"{np.median(anc[ga]):.5f} nm = {np.median(anc[ga]) / ref_nm_per_bin:.2f} "
                f"reference bins; envelope/anchored = {np.median(env[good]) / np.median(anc[ga]):.2f} "
                f"(>1: the coldest frames are not cold enough for the envelope)"
            )
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["machine", "chord", "n_frames", "sigma_inst_nm", "sigma_p10_nm",
                 "sigma_median_nm", "sigma_inst_refbins", "sigma_anchored_nm",
                 "n_labeled", "percentile", "snr_min", "window_nm", "native_bins"]
            )
            for c, n, p_env, p10, med, s_anc, n_anc in rows:
                w.writerow(
                    [machine.name, c, n, p_env, p10, med,
                     p_env / ref_nm_per_bin if np.isfinite(p_env) else np.nan,
                     s_anc, n_anc, args.percentile, args.snr_min, window_nm, main_w]
                )
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
