"""Two-shot comparison of the background-array reconstruction: e.g.
locked-mode ramp shot vs reference shot. Background model only, both
shots overlaid on the same axes, one quantity per panel, with a z-score
of the difference in a window after the marker (difference / combined
predicted sigma, median over chords and frames).

    pixi run python -u -m cer_transfer.figures.shot_compare \
        --a gallery/rq_117193_bg.npz:"117193 (n=1 ramp)" \
        --b gallery/rq_117192_bg.npz:"117192 (reference)" \
        --chords 10,15,20 --target vtor --t0 0.10 --t1 0.24 \
        --t-offset -0.235 --mark 0.168 --zwin 0.168,0.20 \
        --out figs --out-name lockedmode_compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
from cer_transfer.figures.common import FRAME_HZ, load_predictions

FS = FRAME_HZ  # overridden by --fs


def load(spec):
    """Parse ``path[:label]`` and return (label, pred, pred_sigma)."""
    path, _, label = spec.partition(":")
    dmp = load_predictions(path)
    return label or Path(path).stem, dmp.pred, dmp.pred_sigma


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--fs", type=float, default=FRAME_HZ, help="frame rate (Hz)")
    p.add_argument("--a", required=True, help="npz:label")
    p.add_argument("--b", required=True, help="npz:label")
    p.add_argument("--chords", type=str, default="10,15,20")
    p.add_argument("--target", choices=("ti", "vtor"), default="vtor")
    p.add_argument("--t0", type=float, required=True)
    p.add_argument("--t1", type=float, required=True)
    p.add_argument("--t-offset", type=float, default=0.0)
    p.add_argument("--mark", type=float, default=None)
    p.add_argument(
        "--zwin",
        type=str,
        default=None,
        help="t_start,t_end for the difference z-score",
    )
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", type=str, default="shot_compare")
    args = p.parse_args()
    global FS
    FS = args.fs
    args.out.mkdir(parents=True, exist_ok=True)

    la, pa, sa = load(args.a)
    lb, pb, sb = load(args.b)
    T = min(pa.shape[1], pb.shape[1])
    k = 1 if args.target == "vtor" else 0
    unit = "$v_{tor}$ (km/s)" if k else "$T_i$ (eV)"
    chords = [int(c) for c in args.chords.split(",")]
    t = np.arange(T) / FS + args.t_offset
    f0 = max(int(round((args.t0 - args.t_offset) * FS)), 0)
    f1 = min(int(round((args.t1 - args.t_offset) * FS)), T - 1)
    w = np.arange(f0, f1 + 1)

    if args.zwin:
        z0, z1 = (float(x) for x in args.zwin.split(","))
        g0 = int(round((z0 - args.t_offset) * FS))
        g1 = int(round((z1 - args.t_offset) * FS))
        zz = np.arange(max(g0, 0), min(g1, T - 1) + 1)
        dd = pa[chords][:, zz, k] - pb[chords][:, zz, k]
        ss = np.sqrt(sa[chords][:, zz, k] ** 2 + sb[chords][:, zz, k] ** 2)
        print(
            f"{la} - {lb}, {args.target}, {z0:.3f}-{z1:.3f} s, chords "
            f"{chords}: median difference {np.median(dd):.3g}, median "
            f"combined sigma {np.median(ss):.3g}, median |z| "
            f"{np.median(np.abs(dd / ss)):.2f} (|z| < 1: not resolved)"
        )

    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            n = len(chords)
            size = (5.0, 1.8 * n + 0.6) if ctx == "paper" else (9, 3 * n + 1)
            fig, axes = plt.subplots(n, 1, figsize=size, sharex=True, squeeze=False)
            for i, c in enumerate(chords):
                ax = axes[i, 0]
                for lab, pr, sg, col in ((la, pa, sa, "C3"), (lb, pb, sb, "C0")):
                    m_, s_ = pr[c, w, k], sg[c, w, k]
                    ax.fill_between(t[w], m_ - s_, m_ + s_, color=col, alpha=0.18, lw=0)
                    ax.plot(
                        t[w], m_, "-", color=col, lw=1.6, label=lab if i == 0 else None
                    )
                if args.mark is not None:
                    ax.axvline(args.mark, ls="--", color="k", lw=0.9)
                ax.set_ylabel(f"chord {c}\n{unit}", fontsize="small")
            axes[0, 0].legend(fontsize="x-small", loc="best")
            axes[-1, 0].set_xlabel("time (s)")
            axes[-1, 0].set_xlim(t[f0], t[f1])
            fig.tight_layout()
            fig.savefig(
                args.out / f"{args.out_name}.{ext}", dpi=300, bbox_inches="tight"
            )
            plt.close(fig)
    print(f"wrote {args.out}/{args.out_name}.png/.pdf")


if __name__ == "__main__":
    main()
