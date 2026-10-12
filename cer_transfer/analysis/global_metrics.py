"""Rows of the manuscript's global-metrics table from a prediction dump.

All labels: R2, RMSE, median |error|; consistent labels (label-only criteria
of cer_transfer.analysis.coverage.consistent_mask, ~83 % of the labels): R2,
RMSE. Canonical units (ti eV, vtor km/s), pooled over the dump.

    pixi run --frozen python -u -m cer_transfer.analysis.global_metrics \\
        --preds gallery/agnostic_scaling/nstx_agn_ft_rfull.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from cer_transfer.analysis.coverage import consistent_mask
from cer_transfer.figures.common import load_predictions


def scores(p, y):
    """R2, RMSE, median |error|."""
    e = p - y
    r2 = 1.0 - (e**2).sum() / ((y - y.mean()) ** 2).sum()
    return float(r2), float(np.sqrt((e**2).mean())), float(np.median(np.abs(e)))


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preds", required=True, help="eval_checkpoint.py --dump-preds file")
    args = ap.parse_args()
    d = load_predictions(args.preds)
    names = d.targets or ["ti", "vtor"]
    res = {}
    for k, name in enumerate(names):
        y, p = d.y[..., k].astype(np.float64), d.pred[..., k].astype(np.float64)
        lab = np.isfinite(y)
        cons = consistent_mask(y) & lab
        res[name] = (scores(p[lab], y[lab]), scores(p[cons], y[cons]), int(lab.sum()), int(cons.sum()))
        a, c, n, m = res[name]
        print(f"{name}: all labels (n={n}): R2 {a[0]:.3f} RMSE {a[1]:.4g} median|err| {a[2]:.3g} | "
              f"consistent (n={m}, {100 * m / n:.1f}%): R2 {c[0]:.3f} RMSE {c[1]:.4g} median|err| {c[2]:.3g}")
    ti, vt = res["ti"], res["vtor"]
    u = ("eV", "km/s")
    print("\nLaTeX rows (tab:global_metrics):")
    print(f"        $R^2$ & \\num{{{ti[0][0]:.3f}}} & \\num{{{vt[0][0]:.3f}}} \\\\")
    print(f"        \\gls{{rmse}} & \\SI{{{ti[0][1]:.3g}}}{{\\eV}} & \\SI{{{vt[0][1]:.3g}}}{{\\km\\per\\s}} \\\\")
    print(f"        Median $|$error$|$ & \\SI{{{ti[0][2]:.3g}}}{{\\eV}} & \\SI{{{vt[0][2]:.3g}}}{{\\km\\per\\s}} \\\\")
    print("        % consistent labels")
    print(f"        $R^2$ & \\num{{{ti[1][0]:.3f}}} & \\num{{{vt[1][0]:.3f}}} \\\\")
    print(f"        \\gls{{rmse}} & \\SI{{{ti[1][1]:.3g}}}{{\\eV}} & \\SI{{{vt[1][1]:.3g}}}{{\\km\\per\\s}} \\\\")


if __name__ == "__main__":
    main()
