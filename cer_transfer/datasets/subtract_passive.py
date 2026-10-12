"""Background-subtracted NSTX active spectra: active - r * passive, per shot and chord.

The CHERS dual-array files view the same flux surfaces with an active (beam-
intersecting) and a passive array, frame-aligned and chord-index-aligned, with
labels from the active fits. The active array collects roughly 3x the counts of
the passive one (instrumental), so the passive spectrum must be scaled before
subtraction. The scale is measured on the shot itself: in frames without any
conventional fit on a chord (beam off) the active array sees only passive
light, so r = median(active line amplitude / passive line amplitude) over those
frames is the instrumental ratio. Chords without usable beam-off frames fall
back to the dataset median of that chord (two passes).

Per bin, saturated passive samples (>= --rail) are not subtracted (the active
value is kept). The result is clipped at zero; the loader's log preprocessing
handles the rest. Output: joblib files with the same keys as the active files
(input, target, target_error, end_index) plus 'ratio' (per chord) in a new
folder, same file names, so a list of relative paths addresses them.

    pixi run --frozen python -u -m cer_transfer.datasets.subtract_passive \\
        --active-list splits/nstx_active_val.txt --passive-list splits/nstx_passive_val.txt \\
        --out-dir chers_nstx_active_minus_passive --list-out splits/nstx_subtracted_val.txt
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from joblib import dump, load

from cer_transfer.configs import data_path


def read_list(p):
    """Entries of a split list."""
    return [ln.strip() for ln in Path(p).read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]


def line_amplitude(x):
    """Continuum-subtracted clipped sum per (chord, frame) of (C, T, W) spectra."""
    cont = np.percentile(x, 20, axis=-1, keepdims=True)
    return np.clip(x - cont, 0.0, None).sum(-1)


def pair_lists(active, passive):
    """Pair entries by file stem."""
    p = {Path(e).stem: e for e in passive}
    out = []
    for a in active:
        s = Path(a).stem
        if s in p:
            out.append((a, p[s]))
    return out


def shot_ratio(xa, xp, labeled, rail, min_amp):
    """Per-chord instrumental ratio from beam-off frames (NaN where none)."""
    C = xa.shape[0]
    aa, ap = line_amplitude(xa), line_amplitude(xp)
    sat = (xp >= rail).any(-1)
    r = np.full(C, np.nan)
    for c in range(C):
        off = (~labeled[c]) & (~sat[c]) & (ap[c] > min_amp) & (aa[c] > min_amp)
        if off.sum() >= 20:
            r[c] = float(np.median(aa[c][off] / ap[c][off]))
    return r


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--active-list", required=True)
    ap.add_argument("--passive-list", required=True)
    ap.add_argument("--out-dir", required=True, help="folder under CER_DATA_ROOT for the new files")
    ap.add_argument("--list-out", type=Path, required=True, help="list of the new files (relative paths)")
    ap.add_argument("--rail", type=float, default=4095.0, help="saturation level of the passive array")
    ap.add_argument("--min-amp", type=float, default=50.0, help="minimum line amplitude for the ratio")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--csv", type=Path, default=None, help="per-shot/chord ratios")
    args = ap.parse_args()

    pairs = pair_lists(read_list(args.active_list), read_list(args.passive_list))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} paired shots")
    out_root = data_path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # pass 1: ratios
    ratios = {}
    for a, p in pairs:
        da, dp = load(data_path(a), mmap_mode="r"), load(data_path(p), mmap_mode="r")
        T = min(int(da["end_index"]), int(dp["end_index"]))
        xa = np.asarray(da["input"][:, :T], np.float64)
        xp = np.asarray(dp["input"][:, :T], np.float64)
        if xa.shape != xp.shape:
            print(f"skip {a}: shapes {xa.shape} vs {xp.shape}")
            continue
        labeled = np.isfinite(np.asarray(da["target"][:, :T, 0]))
        ratios[a] = shot_ratio(xa, xp, labeled, args.rail, args.min_amp)
    allr = np.array(list(ratios.values()))
    chord_med = np.nanmedian(allr, axis=0)
    print("dataset-median ratio per chord:", np.round(chord_med, 2))
    print(f"shots with a ratio on all chords: {int(np.isfinite(allr).all(1).sum())}/{len(allr)}")

    # pass 2: subtract and write
    rows, written = [], []
    for a, p in pairs:
        if a not in ratios:
            continue
        da, dp = load(data_path(a), mmap_mode="r"), load(data_path(p), mmap_mode="r")
        T = da["input"].shape[1]
        xa = np.asarray(da["input"][:, :T], np.float64)
        xp = np.asarray(dp["input"][:, :T], np.float64) if dp["input"].shape[1] >= T else None
        if xp is None:
            print(f"skip {a}: passive shorter than active")
            continue
        r = ratios[a].copy()
        used_fallback = ~np.isfinite(r)
        r[used_fallback] = chord_med[used_fallback]
        r = np.where(np.isfinite(r), r, 3.0)
        sub = xa - r[:, None, None] * xp
        sub = np.where(xp >= args.rail, xa, sub)  # saturated passive: keep active
        sub = np.clip(sub, 0.0, None).astype(np.float32)
        out = out_root / Path(a).name
        payload = {k: np.asarray(da[k]) for k in da.keys() if k != "input"}
        payload["input"] = sub
        payload["ratio"] = r.astype(np.float32)
        payload["source"] = f"active - r*passive; active {a}, passive {p}"
        tmp = out.with_name(out.name + ".tmp")
        dump(payload, tmp)
        os.replace(tmp, out)
        rel = f"{args.out_dir}/{Path(a).name}"
        written.append(rel)
        for c in range(len(r)):
            rows.append((Path(a).stem, c, r[c], int(used_fallback[c])))
    args.list_out.parent.mkdir(parents=True, exist_ok=True)
    args.list_out.write_text("\n".join(written) + "\n")
    print(f"wrote {len(written)} files to {out_root}; list {args.list_out}")
    if args.csv is not None:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["shot", "chord", "ratio", "fallback"])
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
