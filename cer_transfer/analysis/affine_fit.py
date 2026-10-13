r"""Affine diagnostic of a prediction dump.

How much of the error is a scale and offset per target (expected for zero-shot
transfer between spectrometers), and how much remains after that correction.

For each target, fit ``y = a * pred + b`` to the finite labels (least
squares, with a 5-95 % quantile clip of the prediction range to limit the
influence of failed fits) and report R2 and RMSE before and after the
correction, plus the fitted ``a`` and ``b``.

    pixi run python -u -m cer_transfer.analysis.affine_fit \
        --preds gallery/nstx_val_zero_shot.npz [--plot figs/affine_zero_shot]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cer_transfer.figures.common import load_predictions


def affine(pred, y):
    """Least-squares a, b with y ~ a*pred + b on the central prediction range.

    Computed in float64 in closed form: the dumps are float32, and
    ``np.polyfit`` on float32 input solves in float32, which breaks down
    for a few million points.
    """
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lo, hi = np.percentile(pred, [5, 95])
    m = (pred >= lo) & (pred <= hi)
    xm, ym = pred[m].mean(), y[m].mean()
    dx = pred[m] - xm
    a = float((dx * (y[m] - ym)).sum() / (dx * dx).sum())
    return a, float(ym - a * xm)


def scores(pred, y):
    """(R2, RMSE) of pred against y, in float64."""
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r2 = 1.0 - np.mean((pred - y) ** 2) / np.var(y)
    return float(r2), float(np.sqrt(np.mean((pred - y) ** 2)))


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True)
    p.add_argument("--plot", type=Path, default=None, help="output stem for a figure")
    p.add_argument(
        "--y-max",
        type=float,
        nargs="+",
        default=None,
        help="per-target upper bound of the conventional fit (e.g. 1500 1e9): score "
        "only that label range",
    )
    p.add_argument(
        "--csv", type=Path, default=None, help="write one row per target to this CSV"
    )
    p.add_argument(
        "--meta",
        nargs="*",
        default=[],
        help="extra key=value columns for the CSV (e.g. tag=..., n_train=...)",
    )
    args = p.parse_args()
    d = load_predictions(args.preds)
    names = d.targets or ["ti", "vtor"]
    units = {"ti": "eV", "vtor": "km/s"}
    rows = []
    for k, name in enumerate(names):
        y, mu = d.y[..., k].ravel(), d.pred[..., k].ravel()
        ok = np.isfinite(y) & np.isfinite(mu)
        if args.y_max is not None and k < len(args.y_max):
            ok &= y <= args.y_max[k]
        y, mu = y[ok], mu[ok]
        a, b = affine(mu, y)
        r2_raw, rmse_raw = scores(mu, y)
        r2_fit, rmse_fit = scores(a * mu + b, y)
        rows.append((name, a, b, r2_raw, rmse_raw, r2_fit, rmse_fit, mu, y))
        print(
            f"{name}: y = {a:.3f} * pred + {b:.3g} {units.get(name, '')} | "
            f"R2 {r2_raw:.3f} -> {r2_fit:.3f} after affine correction | "
            f"RMSE {rmse_raw:.4g} -> {rmse_fit:.4g} | n = {y.size}"
        )
    if args.csv is not None:
        import csv

        meta = dict(kv.split("=", 1) for kv in args.meta)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([*meta, "target", "slope", "offset", "r2_raw", "r2_affine", "n"])
            for name, a, b, r2_raw, _, r2_fit, _, _, y in rows:
                w.writerow([*meta.values(), name, a, b, r2_raw, r2_fit, y.size])
        print(f"wrote {args.csv}")
    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        labels = {"ti": "$T_i$", "vtor": "$v_{tor}$"}
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(
            1, len(rows), figsize=(4.0 * len(rows), 3.6), layout="constrained"
        )
        for ax, (name, a, b, _, _, r2_fit, _, mu, y) in zip(np.atleast_1d(axes), rows):
            cal = a * mu.astype(np.float64) + b
            lo, hi = np.percentile(np.r_[y, cal], [1, 99])
            hb = ax.hexbin(
                y,
                cal,
                gridsize=60,
                extent=(lo, hi, lo, hi),
                norm=LogNorm(),
                cmap="Blues",
                mincnt=1,
                linewidths=0,
            )
            ax.plot([lo, hi], [lo, hi], "-", color="0.25", lw=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            u = units.get(name, "")
            ax.set_xlabel(f"conventional fit ({u})")
            ax.set_ylabel(f"model, calibrated ({u})")
            ax.set_title(
                f"{labels.get(name, name)}: $R^2$ = {r2_fit:.2f}", fontsize="medium"
            )
            ax.text(
                0.03,
                0.97,
                f"calibration: {a:.2f} x + {b:.3g} {u}",
                transform=ax.transAxes,
                va="top",
                fontsize="x-small",
                color="0.35",
            )
        fig.colorbar(hb, ax=np.atleast_1d(axes).tolist(), shrink=0.8, label="points")
        for ext in ("png", "pdf"):
            fig.savefig(f"{args.plot}.{ext}", dpi=200, bbox_inches="tight")
        print(f"wrote {args.plot}.png/.pdf")


if __name__ == "__main__":
    main()
