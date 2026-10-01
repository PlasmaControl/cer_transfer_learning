"""Event figure as small multiples: one row of panels, each the profile
at one fitted time inside the event window. Same visual language as the
reconstruction figure: model line with its uncertainty band, conventional
fits as black points, grey where no conventional fit exists; shared y-axis
across the row so the evolution reads left to right.

    pixi run python -u -m cer_transfer.figures.event_panels --preds gallery/e116939.npz \
        --t0 0.340 --t1 0.435 --t-offset -0.235 --n-times 4 --target vtor \
        --profile-x figs/chords_Rtan.csv --profile-xlabel '$R_\\mathrm{tan}$ (m)' \
        --k 0.992 1.171 --out figs --out-name event_ntv_116939
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sns.set_style("whitegrid")
from cer_transfer.figures.common import (FRAME_HZ, chord_coordinate, load_predictions,
                                         sorted_chords)

FS = FRAME_HZ  # overridden by --fs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", type=float, default=FRAME_HZ, help="frame rate (Hz)")
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--t0", type=float, required=True)
    p.add_argument("--t1", type=float, required=True)
    p.add_argument("--t-offset", type=float, default=0.0)
    p.add_argument("--n-times", type=int, default=4)
    p.add_argument("--times", type=float, nargs="*", default=None,
                   help="explicit times (s); the nearest fitted frames are used")
    p.add_argument("--target", choices=("ti", "vtor"), default="vtor")
    p.add_argument("--profile-x", type=Path, default=None)
    p.add_argument("--profile-xlabel", type=str, default=None)
    p.add_argument("--k", type=float, nargs=2, default=None)
    p.add_argument("--no-reference", action="store_true",
                   help="omit the first profile as a grey reference in later panels")
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", type=str, default="event_panels")
    args = p.parse_args()
    global FS
    FS = args.fs
    args.out.mkdir(parents=True, exist_ok=True)

    dmp = load_predictions(args.preds)
    C, T = dmp.C, dmp.T
    k = 1 if args.target == "vtor" else 0
    y = dmp.y[..., k]
    sl = dmp.sigma[..., k]
    mu = dmp.pred[..., k]
    kk = 1.0 if args.k is None else args.k[k]
    ps = dmp.pred_sigma[..., k] * kk if dmp.pred_sigma is not None else None

    xm, xlab = chord_coordinate(C, args.profile_x, args.profile_xlabel)
    order, xs = sorted_chords(xm)

    f0 = int(round((args.t0 - args.t_offset) * FS))
    f1 = int(round((args.t1 - args.t_offset) * FS))
    fitted = [f for f in range(max(f0, 0), min(f1, T - 1) + 1)
              if np.isfinite(y[:, f]).sum() >= 5]
    if not fitted:
        raise SystemExit("no fitted frames in the window")
    if args.times:
        frames = [min(fitted, key=lambda f: abs(f / FS + args.t_offset - tt))
                  for tt in args.times]
    else:
        idx = np.linspace(0, len(fitted) - 1, args.n_times).round().astype(int)
        frames = [fitted[i] for i in idx]

    # shared y-limits from curves and fits (not from the bands)
    vals = np.concatenate([np.r_[mu[order, f], y[order, f]] for f in frames])
    vals = vals[np.isfinite(vals)]
    lo, hi = vals.min(), vals.max()
    pad = 0.1 * max(hi - lo, 1e-6)
    ylim = (lo - pad, hi + pad)
    dx = np.diff(xs)
    step = float(np.median(dx)) if dx.size else 0.5
    unit = "$v_{tor}$ (km/s)" if k else "$T_i$ (eV)"

    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            n = len(frames)
            size = (7.2, 2.1) if ctx == "paper" else (4 * n, 3.8)
            fig, axes = plt.subplots(1, n, figsize=size, sharey=True,
                                     squeeze=False)
            for j, (ax, f) in enumerate(zip(axes[0], frames)):
                yy = y[order, f]
                fitm = np.isfinite(yy)
                # grey where no conventional fit exists
                nf = ~fitm
                a = None
                for q, flag in enumerate(list(nf) + [False]):
                    if flag and a is None:
                        a = q
                    if not flag and a is not None:
                        lo_x = xs[a] - 0.5 * ((xs[a] - xs[a - 1]) if a > 0 else step)
                        b = q - 1
                        hi_x = xs[b] + 0.5 * ((xs[b + 1] - xs[b]) if b + 1 < len(xs) else step)
                        ax.axvspan(lo_x, hi_x, color="0.9", zorder=0, lw=0)
                        a = None
                m_ = mu[order, f]
                if ps is not None:
                    s_ = ps[order, f]
                    nof_ext = nf | np.r_[nf[1:], False] | np.r_[False, nf[:-1]]
                    ax.fill_between(xs, m_ - s_, m_ + s_, where=~nf,
                                    color="C0", alpha=0.22, lw=0, zorder=1)
                    ax.fill_between(xs, m_ - s_, m_ + s_, where=nof_ext,
                                    color="C0", alpha=0.07, lw=0, zorder=1)
                if j > 0 and not args.no_reference:
                    ax.plot(xs, mu[order, frames[0]], "--", color="0.55",
                            lw=1.0, zorder=1.5)
                ax.plot(xs, m_, "-", color="C0", lw=1.5, zorder=2)
                ax.errorbar(xs[fitm], yy[fitm], yerr=sl[order, f][fitm],
                            fmt="o", ms=2.6, lw=0.6, color="k", ecolor="0.45",
                            zorder=3)
                ax.set_ylim(*ylim)
                ax.set_xlim(xs[0] - 0.5 * step, xs[-1] + 0.5 * step)
                ax.set_title(f"t = {f / FS + args.t_offset:.3f} s",
                             fontsize="small")
                ax.set_xlabel(xlab, fontsize="small")
                if j == 0:
                    ax.set_ylabel(unit)
            hh = [Line2D([], [], color="C0", lw=1.5), Patch(color="C0", alpha=0.3),
                  Line2D([], [], color="k", marker="o", ls="", ms=3),
                  Patch(color="0.9")]
            ll = ["model", "model uncertainty", "conventional fit",
                  "no conventional fit"]
            if not args.no_reference:
                hh.insert(1, Line2D([], [], color="0.55", ls="--", lw=1.0))
                ll.insert(1, f"model, t = {frames[0] / FS + args.t_offset:.3f} s")
            fig.tight_layout(rect=(0, 0, 1, 0.86))
            fig.legend(hh, ll, loc="upper center", ncol=len(hh), fontsize="x-small",
                       frameon=False, bbox_to_anchor=(0.5, 1.0))
            fig.savefig(args.out / f"{args.out_name}.{ext}", dpi=300,
                        bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {args.out}/{args.out_name}.png/.pdf | times " + ", ".join(
        f"{f / FS + args.t_offset:.3f}" for f in frames) + " s")


if __name__ == "__main__":
    main()
