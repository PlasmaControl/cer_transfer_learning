"""Per-discharge summary of the edge-frame comparison: beam-on (x) vs
beam-off (y) difference from the nearest conventional measurement, one
point per discharge, with the diagonal. Reads cer_transfer.beamoff.edge_frame_test CSVs.

    pixi run python -u -m cer_transfer.figures.edge_scatter gallery/edge_pool_val.csv \
        gallery/edge_pool_test.csv --select 2005-06,2010 --target vtor \
        --out figs --out-name edge_scatter
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cer_transfer.beamoff.pool_edge import campaign

sns.set_style("whitegrid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+")
    p.add_argument("--select", type=str, default=None)
    p.add_argument("--target", choices=("ti", "vtor"), default="vtor")
    p.add_argument("--log", action="store_true", help="log-log axes")
    p.add_argument("--label-top", type=int, default=0,
                   help="annotate the N largest beam-off values with shot numbers")
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", type=str, default="edge_scatter")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sel = ({s.strip() for s in args.select.split(",")}
           if args.select else None)
    rows, seen = [], set()
    for path in args.csv:
        for r in csv.DictReader(open(path)):
            if r["shot"] in seen:
                continue
            seen.add(r["shot"])
            if sel is not None and campaign(r["shot"]) not in sel:
                continue
            rows.append(r)
    k = args.target
    good = [r for r in rows if r.get(f"{k}_on") and r.get(f"{k}_off")]
    x = np.array([float(r[f"{k}_on"]) for r in good])
    y = np.array([float(r[f"{k}_off"]) for r in good])
    shots = [r["shot"] for r in good]
    n = x.size
    order = np.argsort(y)[::-1]
    print("largest beam-off values: " + ", ".join(
        f"{shots[i]} ({y[i]:.3g} vs {x[i]:.3g})" for i in order[:5]))
    unit = "km/s" if k == "vtor" else "eV"
    q = "$v_{tor}$" if k == "vtor" else "$T_i$"
    print(f"{n} discharges | median beam on {np.median(x):.3g} {unit} | "
          f"median beam off {np.median(y):.3g} {unit} | median ratio "
          f"{np.median(y / x):.2f}")

    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            size = (3.4, 3.4) if ctx == "paper" else (6, 6)
            fig, ax = plt.subplots(figsize=size)
            hi = 1.08 * max(x.max(), y.max())
            lo = 0.8 * min(x.min(), y.min()) if args.log else 0
            ax.plot([max(lo, 1e-3), hi], [max(lo, 1e-3), hi], "--",
                    color="0.4", lw=1.0, label="beam off = beam on")
            ax.plot(x, y, "o", color="C0", ms=5, alpha=0.8,
                    label=f"discharge (n = {n})")
            ax.plot([np.median(x)], [np.median(y)], "D", color="C3", ms=8,
                    label="median")
            if args.log:
                from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
                ax.set_xscale("log")
                ax.set_yscale("log")
                ticks = [v for v in (2, 5, 10, 20, 50, 100, 200, 500)
                         if max(lo, 1e-3) <= v <= hi]
                for axis in (ax.xaxis, ax.yaxis):
                    axis.set_major_locator(FixedLocator(ticks))
                    axis.set_minor_locator(NullLocator())
                    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_xlim(lo if args.log else 0, hi)
            ax.set_ylim(lo if args.log else 0, hi)
            for i in order[:args.label_top]:
                ax.annotate(shots[i], (x[i], y[i]), xytext=(4, 2),
                            textcoords="offset points", fontsize="xx-small")
            ax.set_aspect("equal")
            ax.set_xlabel(f"beam on ({unit})")
            ax.set_ylabel(f"beam off ({unit})")
            ax.set_title(f"{q}: difference to the nearest\n"
                         "conventional measurement", fontsize="small")
            ax.annotate("median", (np.median(x), np.median(y)), xytext=(6, -10),
                        textcoords="offset points", fontsize="x-small", color="C3")
            ax.text(0.97, 0.90, "beam off\n= beam on", transform=ax.transAxes,
                    ha="right", va="top", fontsize="x-small", color="0.4")
            ax.text(0.03, 0.03, f"n = {n} discharges", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize="x-small")
            fig.tight_layout()
            fig.savefig(args.out / f"{args.out_name}.{ext}", dpi=300,
                        bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {args.out}/{args.out_name}.png/.pdf")


if __name__ == "__main__":
    main()
