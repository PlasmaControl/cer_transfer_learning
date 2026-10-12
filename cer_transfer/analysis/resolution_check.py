"""Do files with different numbers of wavelength bins share one pixel scale?

Groups the discharge files of a machine by their native bin count and, per
group, measures the line FWHM in native pixels on labeled tangential
chord-frames within fixed T_i bands. If the FWHM at equal T_i is the same in
all groups, the files differ only by cropping (same dispersion per pixel);
if it scales with the bin count, they cover the same wavelength range with
different binning. The answer decides how spectra must be resampled.

    pixi run python -u -m cer_transfer.analysis.resolution_check \\
        --dirs training_set_30 test_set_30 --machine d3d_tangential --per-group 15
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
from joblib import load

from cer_transfer.configs import data_path, get_machine

BANDS = ((300, 600), (600, 1000), (1000, 1500), (1500, 2500), (2500, 4000))


def fwhm(lines):
    """FWHM in pixels of each row of non-negative line profiles."""
    m = lines.max(-1, keepdims=True)
    above = lines >= 0.5 * m
    idx = np.arange(lines.shape[-1])
    first = np.where(above.any(-1), np.argmax(above, -1), 0)
    last = lines.shape[-1] - 1 - np.argmax(above[:, ::-1], -1)
    return (last - first + 1).astype(float), idx


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dirs", nargs="+", required=True, help="folders under CER_DATA_ROOT"
    )
    ap.add_argument("--machine", default="d3d_tangential")
    ap.add_argument("--per-group", type=int, default=15, help="files read per group")
    ap.add_argument("--min-group", type=int, default=20, help="ignore smaller groups")
    args = ap.parse_args()
    machine = get_machine(args.machine)
    chords = (
        np.arange(machine.n_chords)
        if machine.chord_indices is None
        else np.asarray(machine.chord_indices)
    )
    scale = np.asarray(machine.target_scale, dtype=np.float32)

    groups = defaultdict(list)
    for d in args.dirs:
        p = data_path(d)
        for f in sorted(os.listdir(p)):
            if f.endswith(".joblib"):
                fp = os.path.join(p, f)
                groups[load(fp, mmap_mode="r")["input"].shape[-1]].append(fp)
    rows = []
    for w, files in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(files) < args.min_group:
            continue
        pick = [
            files[i] for i in np.linspace(0, len(files) - 1, args.per_group).astype(int)
        ]
        per_band = defaultdict(list)
        peaks = []
        for fp in pick:
            dd = load(fp, mmap_mode="r")
            y = np.asarray(dd["target"], np.float32)[chords] * scale
            ok = np.isfinite(y).all(-1)
            if not ok.any():
                continue
            c_idx, t_idx = np.where(ok)
            x = np.asarray(dd["input"], np.float32)[chords][c_idx, t_idx]
            line = np.clip(x - np.percentile(x, 10, axis=-1, keepdims=True), 0, None)
            strong = line.max(-1) > np.percentile(line.max(-1), 50)
            fw, _ = fwhm(line[strong])
            ti = y[c_idx, t_idx, 0][strong]
            peaks.extend(line[strong].argmax(-1) / w)
            for lo, hi in BANDS:
                sel = (ti >= lo) & (ti < hi)
                per_band[(lo, hi)].extend(fw[sel])
        cells = []
        for lo, hi in BANDS:
            v = per_band[(lo, hi)]
            cells.append(
                f"{np.median(v):6.1f} ({len(v):>5})"
                if len(v) >= 50
                else "     -        "
            )
        rows.append((w, len(files), cells, np.median(peaks) if peaks else np.nan))
    head = " | ".join(f"Ti {lo}-{hi} eV   " for lo, hi in BANDS)
    print(
        f"{'bins':>5} {'files':>6} | median FWHM in native px (n frames): {head} | line pos/W"
    )
    for w, n, cells, pk in rows:
        print(f"{w:>5} {n:>6} | " + " | ".join(cells) + f" | {pk:.3f}")
    print(
        "\nsame FWHM at equal Ti across bin counts -> cropping (same pixel scale);"
        "\nFWHM proportional to bin count -> same wavelength range, different binning"
    )


if __name__ == "__main__":
    main()
