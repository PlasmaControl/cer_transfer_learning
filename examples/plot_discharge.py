r"""Run a trained model on one discharge and save a figure of the result.

The figure shows, for the chosen chords, the time traces of the ion temperature
and the toroidal rotation reconstructed by the model (line with its uncertainty
band) together with the conventional fits (points with error bars), and the
radial profile of both quantities at one time.

    pixi run --frozen python examples/plot_discharge.py \
        --checkpoint ../models/nstx_fixed_ft_rfull.pt \
        --shot nstx/chers_137711.joblib --chords 13 20 --out discharge_137711.png

``--shot`` is a discharge file, absolute or relative to ``CER_DATA_ROOT``.
``--machine nstx`` applies a DIII-D model to NSTX data (zero-shot).
``--profile-x`` takes a ``chord,x`` CSV (e.g. the tangency radius of each chord);
without it the profile is drawn over the chord index.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cer_transfer.configs import data_path  # noqa: E402
from cer_transfer.figures.timeaxis import file_times  # noqa: E402
from cer_transfer.inference import load_model, predict_file  # noqa: E402


def read_coordinate(path, n_chords):
    """Read a ``chord,x`` CSV into an array of shape (n_chords,)."""
    x = np.full(n_chords, np.nan)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            c = int(row["chord"])
            if 0 <= c < n_chords:
                x[c] = float(row["x"])
    return x


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True, help="trained model (.pt)")
    ap.add_argument("--shot", required=True, help="discharge file (.joblib)")
    ap.add_argument("--machine", default=None, help="e.g. nstx for a DIII-D model")
    ap.add_argument("--chords", type=int, nargs="+", default=[13, 20])
    ap.add_argument("--time", type=float, default=None, help="profile time (s)")
    ap.add_argument("--profile-x", default=None, help="chord,x CSV")
    ap.add_argument("--fs", type=float, default=200.0, help="frames per second")
    ap.add_argument("--t-offset", type=float, default=0.0, help="time of frame 0")
    ap.add_argument("--out", default="discharge.png", help="output figure")
    args = ap.parse_args()

    model = load_model(args.checkpoint, machine=args.machine)
    r = predict_file(model, args.shot)
    C, T, _ = r.pred.shape
    t = file_times(data_path(args.shot), T, args.fs, args.t_offset)
    scale = (1e-3, 1.0)  # T_i in keV, v_tor in km/s
    labels = ("$T_i$ (keV)", "$v_{tor}$ (km/s)")
    fitted = np.isfinite(r.y[..., 1]).sum(0)  # chords with a fit per frame
    if args.time is None:
        f = int(np.argmax(fitted))  # the best-covered fitted frame
    else:
        f = int(np.argmin(np.abs(t - args.time)))
    if args.profile_x:
        x, xlab = read_coordinate(args.profile_x, C), "$R_\\mathrm{tan}$ (m)"
    else:
        x, xlab = np.arange(C, dtype=float), "chord"
    order = np.argsort(np.where(np.isfinite(x), x, np.inf))
    order = order[np.isfinite(x[order])]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), width_ratios=(2, 1))
    for k in range(2):
        mu, sd = r.pred[..., k] * scale[k], r.pred_sigma[..., k] * scale[k]
        y, sy = r.y[..., k] * scale[k], r.sigma[..., k] * scale[k]
        ax = axes[k, 0]
        for i, c in enumerate(args.chords):
            col = f"C{i}"
            ax.fill_between(t, mu[c] - sd[c], mu[c] + sd[c], color=col, alpha=0.2, lw=0)
            ax.plot(t, mu[c], color=col, lw=1.3, label=f"chord {c}: model")
            ok = np.isfinite(y[c])
            ax.errorbar(
                t[ok],
                y[c, ok],
                yerr=sy[c, ok],
                fmt="o",
                ms=2.5,
                lw=0.6,
                color="k",
                label="conventional fit" if i == 0 else None,
            )
        ax.axvline(t[f], color="0.6", ls=":", lw=0.8)
        ax.set_ylabel(labels[k])
        ax = axes[k, 1]
        ax.fill_between(
            x[order],
            (mu - sd)[order, f],
            (mu + sd)[order, f],
            color="C0",
            alpha=0.2,
            lw=0,
        )
        ax.plot(x[order], mu[order, f], color="C0", lw=1.5, label="model")
        ok = np.isfinite(y[order, f])
        ax.errorbar(
            x[order][ok],
            y[order, f][ok],
            yerr=sy[order, f][ok],
            fmt="o",
            ms=2.5,
            lw=0.6,
            color="k",
            label="conventional fit",
        )
    axes[0, 0].legend(fontsize="small")
    axes[0, 1].legend(fontsize="small")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 1].set_xlabel(xlab)
    axes[0, 1].set_title(f"profiles at t = {t[f]:.3f} s", fontsize="medium")
    fig.suptitle(f"{Path(args.shot).stem} | {Path(args.checkpoint).stem}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
