"""Collect the agnostic scaling ladders (slurm/submit_agnostic_ladder.sh).

Reads ``gallery/ladder/*.csv`` written by each rung (own-machine and
zero-shot affine diagnostics) and prints, per direction and training size,
the own-machine R2 and the zero-shot R2 after the affine calibration for
T_i and v_tor; optionally plots both against the number of training
discharges.

    pixi run python -u -m cer_transfer.analysis.agnostic_ladder \\
        [--dir gallery/ladder] [--plot figs/agnostic_ladder]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(directory: Path):
    """{(src, tgt_of_zero_shot): {n_train: {(eval, target): row}}}."""
    runs = defaultdict(dict)
    zs_target = {}
    rows = []
    for f in sorted(directory.glob("*.csv")):
        with open(f) as fh:
            rows.extend(csv.DictReader(fh))
    for r in rows:
        if r["eval"] == "zero_shot":
            zs_target[r["tag"]] = r["tgt"]
    for r in rows:
        key = (r["src"], zs_target.get(r["tag"], "?"))
        cell = runs[key].setdefault(int(r["n_train"]), {"tag": r["tag"]})
        cell[(r["eval"], r["target"])] = r
    return runs


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=Path("gallery/ladder"))
    p.add_argument("--plot", type=Path, default=None, help="output stem for a figure")
    args = p.parse_args()
    runs = load(args.dir)
    if not runs:
        raise SystemExit(f"no ladder results in {args.dir}")

    def val(cell, ev, tgt, col):
        r = cell.get((ev, tgt))
        return float(r[col]) if r else float("nan")

    for (src, tgt), ladder in sorted(runs.items()):
        print(f"\n{src} -> {tgt}")
        print(
            f"{'n_train':>8} | {'own Ti':>7} {'own vtor':>8} | "
            f"{'zs Ti raw':>9} {'zs vtor raw':>11} | "
            f"{'zs Ti cal':>9} {'zs vtor cal':>11} | tag"
        )
        for n in sorted(ladder):
            c = ladder[n]
            print(
                f"{n:>8} | {val(c, 'own', 'ti', 'r2_raw'):7.3f} "
                f"{val(c, 'own', 'vtor', 'r2_raw'):8.3f} | "
                f"{val(c, 'zero_shot', 'ti', 'r2_raw'):9.3f} "
                f"{val(c, 'zero_shot', 'vtor', 'r2_raw'):11.3f} | "
                f"{val(c, 'zero_shot', 'ti', 'r2_affine'):9.3f} "
                f"{val(c, 'zero_shot', 'vtor', 'r2_affine'):11.3f} | {c['tag']}"
            )
        print(
            "  raw: model output as it is, no fitting; cal: after a linear map "
            "fitted on the target's fits (diagnostic only)"
        )
    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        args.plot.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(runs)
        pretty = {"nstx": "NSTX", "d3d": "DIII-D", "d3d_tangential": "DIII-D"}
        ymin = -0.1
        # one row per training machine: own machine | zero-shot on the other
        fig, axes = plt.subplots(
            len(keys),
            2,
            figsize=(8, 3.3 * len(keys) + 0.3),
            squeeze=False,
            sharey=True,
            layout="constrained",
        )
        for row, (src, tgt) in zip(axes, keys):
            ladder = runs[(src, tgt)]
            ns = np.array(sorted(ladder))
            S, Tg = pretty.get(src, src), pretty.get(tgt, tgt)
            panels = (
                (row[0], "own", "r2_raw", f"trained and evaluated on {S}"),
                (
                    row[1],
                    "zero_shot",
                    "r2_affine",
                    f"trained on {S}, applied to {Tg}\n"
                    f"(no {Tg} training; 2 calibration constants per quantity)",
                ),
            )
            for ax, ev, col, title in panels:
                for target, color, lab in (
                    ("ti", "C3", "$T_i$"),
                    ("vtor", "C0", "$v_{tor}$"),
                ):
                    v = np.array([val(ladder[n], ev, target, col) for n in ns])
                    ax.plot(ns, np.clip(v, ymin, 1), "-o", color=color, ms=4, label=lab)
                    low = v < ymin  # off-scale values: marked at the lower edge
                    if low.any():  # (an empty unclipped line breaks the layout)
                        ax.plot(
                            ns[low],
                            np.full(low.sum(), ymin),
                            "v",
                            color=color,
                            ms=8,
                            clip_on=False,
                        )
                ax.set_xscale("log")
                ax.set_ylim(ymin, 1)
                ax.axhline(0, color="0.8", lw=0.8)
                ax.set_title(title, fontsize="small")
                ax.set_xlabel(f"training discharges ({S})")
            row[0].set_ylabel("$R^2$")
        axes[0][0].legend(fontsize="small", loc="lower right")
        fig.text(
            0.01,
            -0.01,
            "triangles on the lower edge: $R^2$ below the axis range",
            fontsize="x-small",
            color="0.4",
        )
        for ext in ("png", "pdf"):
            fig.savefig(f"{args.plot}.{ext}", dpi=200, bbox_inches="tight")
        print(f"\nwrote {args.plot}.png/.pdf")


if __name__ == "__main__":
    main()
