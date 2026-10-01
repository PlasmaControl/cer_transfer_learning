"""Confidence check for --zero-dark-frames BEFORE retraining.

Scans shots and reports, per shot and pooled:
  * fraction of frames classified dark (amp < 2% of shot max, the
    exact criterion of ShotDataset.zero_dark_frames)
  * overlap dark AND labeled  -- MUST be ~0: a fitted label on a
    'dark' frame means the criterion mislabels real plasma as dark
  * amplitude separation: dark frames vs unlabeled-but-bright frames
    (beam-off plasma) vs labeled frames -- the margin between the
    dark population and everything else is the safety margin

    pixi run python -u -m cer_transfer.analysis.check_dark_frames --list splits/nstx_val.txt \
        [--limit 200] [--out dark_check.csv]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from joblib import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--guard", type=int, default=60,
                   help="frames around the labeled span never dark")
    p.add_argument("--k", type=float, default=3.0,
                   help="dark threshold as multiple of the noise floor")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    files = [Path(l.strip()) for l in args.list.read_text().splitlines()
             if l.strip() and not l.strip().startswith("#")]
    if args.limit:
        files = files[: args.limit]

    rows = []
    gaps = []
    pool = dict(frames=0, dark=0, dark_labeled=0)
    amps = {"dark": [], "beam_off_plasma": [], "outside_not_dark": [],
            "labeled": []}
    for fp in files:
        d = load(fp, mmap_mode="r")
        end = int(d["end_index"])
        spec = np.asarray(d["input"][:, :end, :], dtype=np.float32)
        tgt = np.asarray(d["target"][:, :end, :], dtype=np.float32)
        base = np.median(spec, axis=2, keepdims=True)
        amp_t = np.clip(spec - base, 0, None).mean(axis=(0, 2))
        from cer_transfer.dark import dark_mask
        dark = dark_mask(spec, tgt, guard=args.guard, k=args.k)
        labeled_t = np.isfinite(tgt[..., 0]).any(axis=0)
        n_dl = int((dark & labeled_t).sum())
        gap = np.nan
        if dark.any() and labeled_t.any():
            gap = int(np.flatnonzero(labeled_t).min()
                      - np.flatnonzero(dark).max())
        gaps.append(gap)
        rows.append((fp.stem, end, int(dark.sum()), n_dl,
                     float(np.median(amp_t[dark])) if dark.any() else np.nan,
                     float(np.median(amp_t[~dark & ~labeled_t]))
                     if (~dark & ~labeled_t).any() else np.nan,
                     float(np.median(amp_t[labeled_t]))
                     if labeled_t.any() else np.nan))
        pool["frames"] += end
        pool["dark"] += int(dark.sum())
        pool["dark_labeled"] += n_dl
        lab_idx = np.flatnonzero(labeled_t)
        inside = np.zeros(end, bool)
        if lab_idx.size:
            inside[lab_idx.min(): lab_idx.max() + 1] = True
        amps["dark"].extend(amp_t[dark].tolist())
        amps["beam_off_plasma"].extend(
            amp_t[inside & ~labeled_t].tolist())
        amps["outside_not_dark"].extend(
            amp_t[~inside & ~dark].tolist())
        amps["labeled"].extend(amp_t[labeled_t].tolist())

    print(f"shots: {len(files)} | frames: {pool['frames']}")
    print(f"dark frames: {pool['dark']} "
          f"({pool['dark']/max(pool['frames'],1):.1%})")
    print(f"dark AND labeled (MUST be ~0): {pool['dark_labeled']}")
    for k, v in amps.items():
        v = np.array(v)
        if v.size:
            print(f"amp[{k}]: median {np.median(v):.4g} | "
                  f"p95 {np.percentile(v, 95):.4g} | n {v.size}")
    g = np.array([x for x in gaps if np.isfinite(x)])
    if g.size:
        print(f"ramp buffer (first label - last dark frame): "
              f"median {np.median(g):.0f} frames "
              f"({np.median(g)/200*1e3:.0f} ms) | p5 {np.percentile(g,5):.0f}"
              f" | negative (label before dark end -- impossible): "
              f"{(g <= 0).sum()}")
        at_edge = (g == np.min(g)).mean() if g.size else 0.0
        print(f"shots where the guard (not the threshold) binds: "
              f"{(g <= args.guard + 1).mean():.0%}")
    if amps["dark"] and amps["beam_off_plasma"]:
        margin = (np.percentile(amps['beam_off_plasma'], 5)
                  / max(np.percentile(amps['dark'], 95), 1e-9))
        print(f"PHYSICAL margin (beam-off-plasma p5 / dark p95): "
              f"{margin:.1f}x  <-- the number that must be >> 1")
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["shot", "frames", "dark", "dark_labeled",
                        "amp_dark_med", "amp_unlabeled_bright_med",
                        "amp_labeled_med"])
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
