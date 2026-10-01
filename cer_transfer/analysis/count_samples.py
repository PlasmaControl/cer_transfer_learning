"""Dataset size table: shots, frames, labeled points, subseqs per list.

    pixi run python -u -m cer_transfer.analysis.count_samples --lists splits/*.txt [--subseq-len 64]

Counts per list file: shots, total frames (sum end_index), labeled
(chord, frame) points (finite target AND finite positive sigma, either
target), and non-overlapping training subsequences of --subseq-len.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from joblib import load

from cer_transfer.configs import data_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lists", nargs="+", type=Path, required=True)
    p.add_argument("--subseq-len", type=int, default=64)
    p.add_argument("--out", type=Path, default=None,
                   help="optional CSV of the table")
    args = p.parse_args()
    rows = []

    print(f"{'list':>28} {'shots':>7} {'frames':>10} {'labeled pts':>12} "
          f"{'subseqs':>8}")
    for lst in args.lists:
        files = [Path(l.strip()) for l in lst.read_text().splitlines()
                 if l.strip().endswith(".joblib")]
        if not files:
            continue          # not a shot list (e.g. a split report)
        shots = frames = pts = subs = 0
        for fp in files:
            try:
                d = load(data_path(fp), mmap_mode="r")
                end = int(d["end_index"])
                if end <= 0:
                    continue
                tgt = np.asarray(d["target"][:, :end, :])
                err = np.asarray(d["target_error"][:, :end, :])
                ok = (np.isfinite(tgt) & np.isfinite(err) & (err > 0))
                shots += 1
                frames += end
                pts += int(ok.any(axis=-1).sum())
                subs += end // args.subseq_len
            except Exception as e:
                print(f"  ! {fp.name}: {e}")
        print(f"{lst.name:>28} {shots:>7,} {frames:>10,} {pts:>12,} "
              f"{subs:>8,}")
        rows.append((lst.name, shots, frames, pts, subs))
    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["list", "shots", "frames", "labeled_points",
                        f"subseqs_{args.subseq_len}"])
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
