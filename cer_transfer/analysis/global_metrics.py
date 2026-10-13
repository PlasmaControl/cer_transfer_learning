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
    ap.add_argument("--chords", default=None,
                    help="chord,x CSV (tangency radius per chord; same order as the dump) for core/edge medians")
    ap.add_argument("--core-max", type=float, default=1.2, help="core: R_tan below (m)")
    ap.add_argument("--edge-min", type=float, default=1.3, help="edge: R_tan above (m)")
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
        sg = d.sigma[..., k].astype(np.float64)
        def best(sel):  # best possible R2: a perfect model still differs from the labels by their noise
            return 1.0 - np.nanmean(sg[sel] ** 2) / np.var(y[sel])
        share = 1.0 - (m * c[1] ** 2) / (n * a[1] ** 2)
        print(f"{name}: best possible R2 all {best(lab):.3f} | consistent {best(cons):.3f} | "
              f"flagged {100 * (1 - m / n):.1f}% carry {100 * share:.0f}% of the squared error")
        print(f"{name}: all labels (n={n}): R2 {a[0]:.3f} RMSE {a[1]:.4g} median|err| {a[2]:.3g} | "
              f"consistent (n={m}, {100 * m / n:.1f}%): R2 {c[0]:.3f} RMSE {c[1]:.4g} median|err| {c[2]:.3g}")
    if args.chords:
        import csv
        x = np.full(d.C, np.nan)
        for r in csv.DictReader(open(args.chords)):
            i = int(r["chord"])
            if 0 <= i < d.C:
                x[i] = float(r["x"])
        print("\ncore/edge (all labels; median |error| and median target):")
        for name, sel in (("core", x < args.core_max), ("edge", x > args.edge_min)):
            parts = []
            for k, tn in enumerate(names):
                y, p = d.y[sel, :, k].astype(np.float64), d.pred[sel, :, k].astype(np.float64)
                ok = np.isfinite(y) & np.isfinite(p)
                u = "eV" if tn == "ti" else "km/s"
                parts.append(f"{tn}: median|err| {np.median(np.abs(p[ok] - y[ok])):.3g} {u}, "
                             f"median target {np.median(np.abs(y[ok])):.3g} {u} (n={ok.sum()})")
            print(f"  {name} ({int(sel.sum())} chords): " + " | ".join(parts))
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
