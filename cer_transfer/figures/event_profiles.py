"""Event figure: profile evolution through a known physics event.

One panel: profiles of one target at successive times inside a window,
model dense (lines colored by time), conventional fits at their sparse
cadence (points, same color scale). Example events (e.g.
Zhu PRL 2006 discharges 116939 / 115600).

    pixi run python -u -m cer_transfer.figures.event_profiles --preds gallery/116939/legacy.npz \
        --t0 0.355 --t1 0.435 --n-profiles 8 --target vtor \
        --out figs --out-name event_ntv_116939
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm, colors

sns.set_style("whitegrid")
from cer_transfer.figures.common import FRAME_HZ

FS = FRAME_HZ  # overridden by --fs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", type=float, default=FRAME_HZ, help="frame rate (Hz)")
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--t0", type=float, required=True)
    p.add_argument("--t1", type=float, required=True)
    p.add_argument("--n-profiles", type=int, default=8)
    p.add_argument("--frames-at", choices=("uniform", "fits", "both"),
                   default="both",
                   help="model profiles at uniform times, at the fitted "
                        "frames, or both (default: every fit gets its "
                        "exact model counterpart)")
    p.add_argument("--target", choices=("ti", "vtor", "both"),
                   default="vtor")
    p.add_argument("--t-offset", type=float, default=0.0,
                   help="absolute time of frame 0 (s): displayed time = "
                        "frame/200 + offset. Locate the event in the "
                        "fitted history to calibrate against literature "
                        "times; window args --t0/--t1 are in displayed "
                        "(absolute) time.")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", type=str, default="event")
    args = p.parse_args()
    global FS
    FS = args.fs
    args.out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.preds)
    ch = d["chord"]
    C = int(ch.max()) + 1
    T = len(ch) // C
    y = d["y"].reshape(C, T, -1)
    pr = d["pred"].reshape(C, T, -1)
    sg = d["sigma"].reshape(C, T, -1)
    targets = (0, 1) if args.target == "both" else \
        ((1,) if args.target == "vtor" else (0,))
    units = ("$T_i$ (eV)", "$v_{tor}$ (km/s)")
    t = targets[0]

    f0 = int((args.t0 - args.t_offset) * FS)
    f1 = int((args.t1 - args.t_offset) * FS)
    f0 = max(f0, 0)
    f1 = min(f1, T - 1)
    frame_set = set(np.linspace(f0, f1, args.n_profiles)
                    .round().astype(int).tolist())
    if args.frames_at in ("fits", "both"):
        # a model line at EVERY fitted frame in the window, so each set
        # of fit points has its exact model counterpart
        fitted = [f for f in range(f0, f1 + 1)
                  if np.isfinite(y[:, f, t]).any()]
        frame_set = set(fitted) if args.frames_at == "fits" \
            else frame_set | set(fitted)
    frames = np.array(sorted(frame_set))
    norm = colors.Normalize(vmin=f0 / FS + args.t_offset,
                            vmax=f1 / FS + args.t_offset)
    cmap = cm.plasma
    cx = np.arange(C)

    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            n_p = len(targets)
            size = ((4.6 * n_p, 3.4) if ctx == "paper"
                    else (8 * n_p, 5.5))
            fig, axes = plt.subplots(1, n_p, figsize=size, squeeze=False)
            for ax, t in zip(axes[0], targets):
                for f in frames:
                    ax.plot(cx, pr[:, f, t], "-", lw=1.5,
                            color=cmap(norm(f / FS + args.t_offset)))
                for f in range(f0, f1 + 1):
                    m = np.isfinite(y[:, f, t])
                    if m.any():
                        ax.errorbar(cx[m], y[m, f, t], yerr=sg[m, f, t],
                                    fmt="o", ms=3, lw=0.7, zorder=5,
                                    color=cmap(norm(f / FS
                                                    + args.t_offset)),
                                    mec="k", mew=0.4)
                ax.set_xlabel("chord (core -> edge)")
                ax.set_ylabel(units[t])
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            cb = fig.colorbar(sm, ax=list(axes[0]), fraction=0.03,
                              pad=0.02)
            cb.set_label("time (s)")
            if args.title:
                fig.suptitle(args.title, fontsize="medium")
            fig.savefig(args.out / f"{args.out_name}.{ext}", dpi=300,
                        bbox_inches="tight")
            plt.close(fig)
    nfit = sum(np.isfinite(y[:, f, targets[0]]).any()
               for f in range(f0, f1 + 1))
    for t in targets:
        yw = y[:, f0:f1 + 1, t]
        pw = pr[:, f0:f1 + 1, t]
        m = np.isfinite(yw)
        if m.any():
            ae = np.abs(yw[m] - pw[m])
            unit = "km/s" if t == 1 else "eV"
            print(f"event-window medae vs fits ({unit}): "
                  f"{np.median(ae):.3g} | p95 "
                  f"{np.percentile(ae, 95):.3g} | n {m.sum()}")
    print(f"wrote {args.out}/{args.out_name}.png/.pdf | "
          f"{len(frames)} model profiles, {nfit} fitted frames in window")


if __name__ == "__main__":
    main()
