"""Compare spectrogram statistics for ONE shot across datasets (e.g. NSTX
legacy vs active vs passive arrays).

    pixi run python -u -m cer_transfer.analysis.inspect_shot \
        --file <legacy>/chers_115500.joblib --label nstx \
        --file <active>/chers_115500.joblib --label active \
        --file <passive>/chers_115500.joblib --label passive \
        --out figs [--chord 25]

Prints per-dataset count statistics and writes ONE simple figure,
<shot>_spectra.(png|pdf), with 3 rows:
  1) count histogram (log-x, all chords/frames pooled), one line per dataset
  2) mean spectrum of the chosen chord (counts vs wavelength bin, log-y)
  3) line amplitude (baseline-subtracted sum) vs time for that chord
Differences between active and passive show up as: histogram shift (gain /
count regime), line presence/shape in the mean spectrum, and temporal
structure (beam modulation visible in active amplitude, absent in passive).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import load

from cer_transfer.configs import data_path

sns.set_style("whitegrid")


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--file", action="append", required=True, type=Path)
    p.add_argument("--label", action="append", required=True)
    p.add_argument("--chord", type=int, default=25)
    p.add_argument("--out", type=Path, default=Path("figs"))
    args = p.parse_args()
    assert len(args.file) == len(args.label), "--file/--label must pair up"

    data = []
    print(
        f"{'dataset':>10} {'C':>4} {'T_end':>6} {'W':>4} "
        f"{'median':>8} {'p95':>8} {'max':>10} {'line_med':>9}"
    )
    for fp, lab in zip(args.file, args.label):
        d = load(data_path(fp), mmap_mode="r")
        end = int(d["end_index"])
        spec = np.asarray(d["input"][:, :end, :], dtype=np.float64)
        med = np.median(spec, axis=-1, keepdims=True)
        line = np.clip(spec - med, 0, None).sum(-1)  # (C, T) amplitude
        print(
            f"{lab:>10} {spec.shape[0]:>4} {end:>6} {spec.shape[2]:>4} "
            f"{np.median(spec):>8.4g} {np.percentile(spec, 95):>8.4g} "
            f"{spec.max():>10.4g} {np.median(line):>9.4g}"
        )
        data.append((lab, spec, line))

    c = args.chord

    def build(ctx):
        """Draw the inspection figure for the current plotting context."""
        size = (7.0, 6.0) if ctx == "paper" else (10, 8.5)
        fig, axes = plt.subplots(3, 1, figsize=size)
        for i, (lab, spec, line) in enumerate(data):
            v = spec.ravel()
            v = v[v > 0]
            bins = np.geomspace(v.min(), v.max(), 80)
            axes[0].hist(
                v,
                bins=bins,
                histtype="step",
                lw=1.5,
                density=True,
                label=lab,
                color=f"C{i}",
            )
            axes[1].plot(spec[c].mean(axis=0), lw=1.5, label=lab, color=f"C{i}")
            axes[2].plot(line[c], lw=1.0, label=lab, color=f"C{i}")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("counts")
        axes[0].set_ylabel("density")
        axes[0].legend()
        axes[1].set_yscale("log")
        axes[1].set_xlabel("wavelength bin")
        axes[1].set_ylabel(f"mean counts (chord {c})")
        axes[2].set_xlabel("frame")
        axes[2].set_ylabel(f"line amplitude (chord {c})")
        fig.tight_layout()
        return fig

    args.out.mkdir(parents=True, exist_ok=True)
    shot = args.file[0].stem
    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            fig = build(ctx)
            fig.savefig(
                args.out / f"{shot}_spectra.{ext}", dpi=300, bbox_inches="tight"
            )
            plt.close(fig)
    print(f"wrote {args.out / shot}_spectra.png/.pdf")


if __name__ == "__main__":
    main()
