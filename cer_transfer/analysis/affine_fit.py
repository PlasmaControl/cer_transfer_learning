"""Affine diagnostic of a prediction dump: how much of the error is a scale
and offset per target (expected for zero-shot transfer between
spectrometers), and how much remains after that correction.

For each target, fit ``y = a * pred + b`` to the finite labels (least
squares, with a 5-95 % quantile clip of the prediction range to limit the
influence of failed fits) and report R2 and RMSE before and after the
correction, plus the fitted ``a`` and ``b``.

    pixi run python -u -m cer_transfer.analysis.affine_fit \\
        --preds gallery/nstx_val_zero_shot.npz [--plot figs/affine_zero_shot]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cer_transfer.figures.common import load_predictions


def affine(pred, y):
    """Least-squares a, b with y ~ a*pred + b on the central prediction range."""
    lo, hi = np.percentile(pred, [5, 95])
    m = (pred >= lo) & (pred <= hi)
    a, b = np.polyfit(pred[m], y[m], 1)
    return float(a), float(b)


def scores(pred, y):
    """(R2, RMSE) of pred against y."""
    r2 = 1.0 - np.mean((pred - y) ** 2) / np.var(y)
    return float(r2), float(np.sqrt(np.mean((pred - y) ** 2)))


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True)
    p.add_argument("--plot", type=Path, default=None, help="output stem for a figure")
    args = p.parse_args()
    d = load_predictions(args.preds)
    names = d.targets or ["ti", "vtor"]
    units = {"ti": "eV", "vtor": "km/s"}
    rows = []
    for k, name in enumerate(names):
        y, mu = d.y[..., k].ravel(), d.pred[..., k].ravel()
        ok = np.isfinite(y) & np.isfinite(mu)
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
    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 4))
        for ax, (name, a, b, r2_raw, _, r2_fit, _, mu, y) in zip(
            np.atleast_1d(axes), rows
        ):
            idx = np.random.default_rng(0).choice(
                y.size, min(y.size, 20000), replace=False
            )
            ax.plot(mu[idx], y[idx], ".", ms=2, alpha=0.3, color="C0")
            lo, hi = np.percentile(mu, [1, 99])
            xx = np.linspace(lo, hi, 2)
            ax.plot(xx, xx, "--", color="0.4", label="identity")
            ax.plot(
                xx, a * xx + b, "-", color="C3", label=f"y = {a:.2f} pred + {b:.3g}"
            )
            ax.set_xlabel(f"prediction ({units.get(name, '')})")
            ax.set_ylabel(f"conventional fit ({units.get(name, '')})")
            ax.set_title(f"{name}: R2 {r2_raw:.2f} -> {r2_fit:.2f}", fontsize="small")
            ax.legend(fontsize="x-small")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"{args.plot}.{ext}", dpi=200)
        print(f"wrote {args.plot}.png/.pdf")


if __name__ == "__main__":
    main()
