"""Per-chord line-shift response of one machine, from its own labels.

For every chord, the line centroid (windowed moments around the peak, in the
model's resampled pixels with the machine's stored convention) is regressed
on the rotation label over all labeled frames: pixels per km/s, and the
centroid at zero rotation. If the response varies systematically across
chords, the line-of-sight projection (viewing geometry) enters; if it is
constant, the machine has one shift scale.

    pixi run python -u -m cer_transfer.analysis.shift_response \\
        --machine nstx --list splits/nstx_val.txt --limit 50 \\
        [--coord figs/chords_Rtan_137711.csv]
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from cer_transfer.analysis.domain_check import line_moments
from cer_transfer.configs import data_path, get_machine
from cer_transfer.data import ShotDataset


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--resample-w", type=int, default=256)
    ap.add_argument("--coord", default=None, help="chord,x CSV (e.g. tangency radius)")
    ap.add_argument("--min-n", type=int, default=200, help="labeled frames per chord")
    args = ap.parse_args()
    machine = get_machine(args.machine)
    files = [
        data_path(ln.strip())
        for ln in open(args.list).read().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ][: args.limit]
    ds = ShotDataset(
        files, machine, subseq_len=-1, mmap=False, resample_w=args.resample_w
    )
    cen, vt, ch = [], [], []
    for i in range(len(ds)):
        x, tgt, _, _ = ds[i]
        c, _, amp = line_moments(x.numpy())
        y = tgt.numpy().reshape(c.shape[0], c.shape[1], -1)
        ok = np.isfinite(y[..., 1]) & np.isfinite(c) & (amp > np.nanpercentile(amp, 30))
        ci, ti = np.where(ok)
        cen.append(c[ci, ti]), vt.append(y[ci, ti, 1]), ch.append(ci)
    cen, vt, ch = map(np.concatenate, (cen, vt, ch))
    coord = {}
    if args.coord:
        coord = {
            int(r["chord"]): float(r["x"]) for r in csv.DictReader(open(args.coord))
        }
    print(f"{args.machine}: {len(files)} discharges, {cen.size:,} labeled chord-frames")
    print(
        f"{'chord':>5} {'coord':>7} {'n':>7} {'px/(km/s)':>10} {'px at v=0':>10} {'R2':>5} {'v range':>15}"
    )
    slopes = []
    for c in np.unique(ch):
        s = ch == c
        if s.sum() < args.min_n or np.ptp(vt[s]) < 20:
            continue
        x_, y_ = vt[s].astype(np.float64), cen[s].astype(np.float64)
        xm, ym = x_.mean(), y_.mean()
        b = ((x_ - xm) * (y_ - ym)).sum() / ((x_ - xm) ** 2).sum()
        a = ym - b * xm
        r2 = 1 - ((y_ - a - b * x_) ** 2).mean() / y_.var()
        slopes.append(b)
        lo, hi = np.percentile(x_, [5, 95])
        cs = f"{coord[c]:.3f}" if c in coord else "-"
        print(
            f"{c:>5} {cs:>7} {s.sum():>7} {b:10.4f} {a:10.1f} {r2:5.2f} {lo:7.0f}..{hi:<6.0f}"
        )
    if slopes:
        sl = np.array(slopes)
        print(
            f"\nshift response across chords: median {np.median(sl):.4f} px/(km/s), "
            f"5-95 % {np.percentile(sl, 5):.4f} .. {np.percentile(sl, 95):.4f} "
            f"(spread {np.percentile(sl, 95) / np.percentile(sl, 5):.2f}x)"
        )


if __name__ == "__main__":
    main()
