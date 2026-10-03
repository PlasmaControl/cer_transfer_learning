"""Per-chord median |error| in physical units of
models trained under different regimes, against the per-chord
label-noise floor (median label sigma) -- the analysis-chain limit.

Inputs: per-chord CSVs from eval_checkpoint.py (--out), one per run,
and the validation list for the floor (computed CPU-only from the
label uncertainties in the data files).

    pixi run python -u -m cer_transfer.figures.transfer_floor \
        --runs "from scratch (7307):eval_scratch.csv" \
               "fine-tuned (325):eval_r325.csv" \
               "fine-tuned (7307):eval_ft.csv" \
        --floor-list splits/nstx_val.txt \
        --out figs [--out-name transfer_floor]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cer_transfer.configs import data_path

sns.set_style("whitegrid")


def label_floor(list_path: Path, scale, limit: int = 0):
    """Per-chord median label sigma (physical units) over a file list.
    scale: MachineConfig.target_scale -- files may store other units
    (NSTX: Ti in keV); the same transform the training data applies."""
    from joblib import load

    files = [
        ln.strip()
        for ln in list_path.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if limit:
        files = files[:limit]
    acc = None
    for fp in files:
        d = load(data_path(fp), mmap_mode="r")
        end = int(d["end_index"])
        err = np.asarray(d["target_error"][:, :end, :], dtype=np.float32)
        if acc is None:
            acc = [[] for _ in range(err.shape[0])]
        for c in range(err.shape[0]):
            v = err[c][np.isfinite(err[c][..., 0])]
            if v.size:
                acc[c].append(v.reshape(-1, err.shape[-1]))
    C = len(acc)
    fl = np.full((C, 2), np.nan)
    for c in range(C):
        if acc[c]:
            v = np.concatenate(acc[c])
            fl[c] = np.nanmedian(np.abs(v), axis=0)
    return fl * np.asarray(scale, dtype=np.float32)[None, :]


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="'label:per_chord.csv' per run, plot order",
    )
    p.add_argument("--floor-list", type=Path, required=True)
    p.add_argument(
        "--machine",
        type=str,
        default="nstx",
        help="registry machine for target_scale (file units "
        "-> physical units), matching the training data",
    )
    p.add_argument(
        "--floor-limit",
        type=int,
        default=0,
        help="cap shots scanned for the floor (0 = all)",
    )
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", type=str, default="transfer_floor")
    p.add_argument(
        "--log-y",
        action="store_true",
        help="log error axis: keeps mid-array differences "
        "visible against the large edge errors",
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    for s in args.runs:
        parts = s.split(":")
        if len(parts) == 3:
            label, path, ls = parts
        else:
            label, path = parts[0], ":".join(parts[1:])
            ls = "-"
        runs.append((label, pd.read_csv(path), ls))
    from cer_transfer.configs import get_machine

    scale = get_machine(args.machine).target_scale
    fl = label_floor(args.floor_list, scale, args.floor_limit)
    cx = np.arange(len(fl))

    cols = ("medae_ti", "medae_vtor")
    names = ("median $|T_i$ error$|$ (eV)", "median $|v_{tor}$ error$|$ (km/s)")
    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            size = (7.2, 3.0) if ctx == "paper" else (12, 5)
            fig, axes = plt.subplots(1, 2, figsize=size)
            for t, ax in enumerate(axes):
                pos = np.concatenate(
                    [fl[:, t][np.isfinite(fl[:, t]) & (fl[:, t] > 0)]]
                    + [df[cols[t]].to_numpy() for _, df, _ in runs]
                )
                lo = 0.6 * np.nanmin(pos[pos > 0])
                hi = 1.5 * np.nanmax(pos)
                ax.fill_between(
                    cx,
                    lo if args.log_y else 0,
                    fl[:, t],
                    color="0.82",
                    label="label-noise floor" if t == 0 else None,
                    zorder=0,
                )
                for i, (label, df, ls) in enumerate(runs):
                    ax.plot(
                        df["chord"],
                        df[cols[t]],
                        ls,
                        lw=1.6,
                        color=f"C{i}",
                        label=label if t == 0 else None,
                    )
                ax.set_xlabel("chord (core -> edge)")
                ax.set_ylabel(names[t])
                if args.log_y:
                    ax.set_yscale("log")
                    ax.set_ylim(lo, hi)
                else:
                    ax.set_ylim(bottom=0)
            axes[0].legend(fontsize="small", loc="upper right")
            fig.tight_layout()
            fig.savefig(
                args.out / f"{args.out_name}.{ext}", dpi=300, bbox_inches="tight"
            )
            plt.close(fig)
    print(
        f"wrote {args.out}/{args.out_name}.png/.pdf | "
        f"{len(runs)} runs, floor from {args.floor_list.name}"
    )


if __name__ == "__main__":
    main()
