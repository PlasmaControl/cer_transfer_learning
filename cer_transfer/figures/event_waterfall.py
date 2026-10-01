"""Waterfall event figure: one profile per time slice, stacked upward with
a constant offset (earliest at the bottom). Model: solid where a
conventional fit exists, dotted where none exists; conventional fits as
black points on their slice; time written at the right end of each line;
a scale bar instead of y-tick values.

    pixi run python -u -m cer_transfer.figures.event_waterfall --preds gallery/e116939.npz \
        --t0 0.340 --t1 0.435 --t-offset -0.235 --n-times 8 --target vtor \
        --profile-x figs/chords_Rtan_116939.csv --profile-xlabel '$R_\\mathrm{tan}$ (m)' \
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

sns.set_style("white")
from cer_transfer.figures.common import (FRAME_HZ, chord_coordinate, load_predictions,
                                         sorted_chords)

FS = FRAME_HZ  # overridden by --fs


def nice(v):
    """Round a scale-bar length to 1, 2 or 5 times a power of ten."""
    e = 10 ** np.floor(np.log10(v))
    for m in (1, 2, 5, 10):
        if m * e >= v:
            return m * e
    return 10 * e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", type=float, default=FRAME_HZ, help="frame rate (Hz)")
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--t0", type=float, required=True)
    p.add_argument("--t1", type=float, required=True)
    p.add_argument("--t-offset", type=float, default=0.0)
    p.add_argument("--n-times", type=int, default=8)
    p.add_argument("--times", type=float, nargs="*", default=None)
    p.add_argument("--target", choices=("ti", "vtor"), default="vtor")
    p.add_argument("--profile-x", type=Path, default=None)
    p.add_argument("--profile-xlabel", type=str, default=None)
    p.add_argument("--k", type=float, nargs=2, default=None)
    p.add_argument("--offset", type=float, default=None,
                   help="vertical offset between slices (data units); "
                        "default: 0.3 x the value range")
    p.add_argument("--no-band", action="store_true")
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", type=str, default="event_waterfall")
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
    ps = (dmp.pred_sigma[..., k] * kk
          if (dmp.pred_sigma is not None and not args.no_band) else None)

    xm, xlab = chord_coordinate(C, args.profile_x, args.profile_xlabel)
    order, xs = sorted_chords(xm)

    f0 = int(round((args.t0 - args.t_offset) * FS))
    f1 = int(round((args.t1 - args.t_offset) * FS))
    fitted = [f for f in range(max(f0, 0), min(f1, T - 1) + 1)
              if np.isfinite(y[:, f]).sum() >= 5]
    if not fitted:
        raise SystemExit("no fitted frames in the window")
    if args.times:
        frames = sorted({min(fitted, key=lambda f: abs(f / FS + args.t_offset - tt))
                         for tt in args.times})
    else:
        idx = np.unique(np.linspace(0, len(fitted) - 1, args.n_times).round().astype(int))
        frames = [fitted[i] for i in idx]

    vals = np.concatenate([np.r_[mu[order, f], y[order, f]] for f in frames])
    vals = vals[np.isfinite(vals)]
    rng = max(vals.max() - vals.min(), 1e-6)
    off = args.offset if args.offset is not None else 0.3 * rng
    unit = "km/s" if k else "eV"
    qty = "$v_{tor}$" if k else "$T_i$"
    bar = nice(0.25 * rng)

    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            size = (4.6, 5.2) if ctx == "paper" else (8, 9)
            fig, ax = plt.subplots(figsize=size)
            for j, f in enumerate(frames):
                o = j * off
                m_ = mu[order, f]
                yy = y[order, f]
                fitm = np.isfinite(yy)
                # extend the solid segment one point into gaps so it connects
                solid = fitm | np.r_[fitm[1:], False] | np.r_[False, fitm[:-1]]
                if ps is not None:
                    s_ = ps[order, f]
                    ax.fill_between(xs, m_ - s_ + o, m_ + s_ + o, where=fitm,
                                    color="C0", alpha=0.18, lw=0, zorder=1)
                ax.plot(xs, np.where(solid, m_, np.nan) + o, "-", color="C0",
                        lw=1.4, zorder=2)
                ax.plot(xs, np.where(~fitm | ~solid, m_, np.nan) + o, ":",
                        color="C0", lw=1.2, zorder=2)
                ax.plot(xs[fitm], yy[fitm] + o, "o", color="k", ms=2.4,
                        zorder=3)
                ax.text(xs[-1] + 0.02 * (xs[-1] - xs[0]), m_[-1] + o,
                        f"{f / FS + args.t_offset:.3f} s", va="center",
                        ha="left", fontsize="x-small")
            # scale bar in its own strip in the left margin (never on data)
            span = xs[-1] - xs[0]
            ylo, yhi = ax.get_ylim()
            xb = xs[0] - 0.05 * span
            yb = ylo + 0.03 * (yhi - ylo)
            ax.plot([xb, xb], [yb, yb + bar], "-", color="k", lw=1.6,
                    solid_capstyle="butt", clip_on=False)
            ax.text(xb - 0.015 * span, yb + 0.5 * bar, f"{bar:g} {unit}",
                    va="center", ha="right", rotation=90, fontsize="x-small")
            ax.set_yticks([])
            ax.set_ylabel(f"{qty}, offset by time $\\rightarrow$")
            ax.set_xlabel(xlab)
            ax.set_xlim(xs[0] - 0.09 * span, xs[-1] + 0.16 * span)
            ax.spines["bottom"].set_bounds(xs[0], xs[-1])
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            hh = [Line2D([], [], color="C0", lw=1.4),
                  Line2D([], [], color="C0", ls=":", lw=1.2),
                  Line2D([], [], color="k", marker="o", ls="", ms=3)]
            ll = ["model", "model, no conventional fit", "conventional fit"]
            if ps is not None:
                hh.insert(1, Patch(color="C0", alpha=0.25))
                ll.insert(1, "model uncertainty")
            ax.legend(hh, ll, loc="upper center", bbox_to_anchor=(0.5, 1.12),
                      ncol=2, fontsize="x-small", frameon=False)
            fig.tight_layout()
            fig.savefig(args.out / f"{args.out_name}.{ext}", dpi=300,
                        bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {args.out}/{args.out_name}.png/.pdf | {len(frames)} slices: "
          + ", ".join(f"{f / FS + args.t_offset:.3f}" for f in frames) + " s")


if __name__ == "__main__":
    main()
