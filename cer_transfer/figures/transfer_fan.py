"""Profile fan across the scaling-subset models.

One axes per target: the discharge's profile at the selected time,
reconstructed by every fine-tuned subset model (color-graded by number
of training discharges), conventional fits as points.

    pixi run python -u -m cer_transfer.figures.transfer_fan \
        --preds 325:g/r325.npz 650:g/r650.npz ... 7307:g/legacy.npz \
        --out figs [--out-name transfer_fan] [--frame auto]
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", nargs="+", required=True,
                   help="N:path.npz per subset model, ascending N")
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", type=str, default="transfer_fan")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    for s in args.preds:
        n, path = s.split(":", 1)
        d = np.load(path)
        ch = d["chord"]
        C = int(ch.max()) + 1
        T = len(ch) // C
        runs.append((int(n), d["y"].reshape(C, T, -1),
                     d["pred"].reshape(C, T, -1),
                     d["sigma"].reshape(C, T, -1)))
    runs.sort(key=lambda r: r[0])
    y = runs[-1][1]
    C, T = y.shape[:2]

    # display frame: same rule as recon_composite
    lab_frac = np.isfinite(y[..., 0]).mean(axis=0)
    cover = lab_frac > 0.5 * lab_frac.max()
    edge = max(T // 10, 1)
    cover[:edge] = False
    cover[-edge:] = False
    with np.errstate(all="ignore"):
        med = np.where(cover, np.nanmedian(
            np.where(np.isfinite(y[..., 0]), y[..., 0], np.nan), axis=0),
            -np.inf)
    f = int(np.argmax(med))

    norm = colors.LogNorm(vmin=runs[0][0], vmax=runs[-1][0])
    cmap = cm.viridis
    cx = np.arange(C)
    names = ("$T_i$ (eV)", "$v_{tor}$ (km/s)")

    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            size = (7.2, 3.0) if ctx == "paper" else (12, 5)
            fig, axes = plt.subplots(1, 2, figsize=size, sharex=True)
            for t, ax in enumerate(axes):
                for n, _, pr, _ in runs:
                    ax.plot(cx, pr[:, f, t], "-", lw=1.4,
                            color=cmap(norm(n)))
                m = np.isfinite(y[:, f, t])
                ax.errorbar(cx[m], y[m, f, t],
                            yerr=runs[-1][3][m, f, t], fmt="o", ms=3.5,
                            lw=0.9, color="k", zorder=5,
                            label="conventional fit")
                ax.set_xlabel("chord (core -> edge)")
                ax.set_ylabel(names[t])
                if t == 0:
                    ax.legend(loc="upper right", fontsize="small")
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
            cb.set_label("NSTX training discharges (fine-tuned)")
            cb.set_ticks([r[0] for r in runs])
            cb.set_ticklabels([str(r[0]) for r in runs])
            t_s = f / 200.0
            fig.suptitle(f"profiles at t = {t_s:.2f} s", y=1.02,
                         fontsize="medium")
            fig.savefig(args.out / f"{args.out_name}.{ext}", dpi=300,
                        bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {args.out}/{args.out_name}.png/.pdf | frame {f} "
          f"(t={f/200:.3f}s)")


if __name__ == "__main__":
    main()
