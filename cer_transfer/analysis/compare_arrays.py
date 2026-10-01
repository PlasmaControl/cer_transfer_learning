"""Dataset-wide statistical comparison of ACTIVE vs PASSIVE spectrogram
arrays. Pairs shots by filename stem across two lists, aggregates simple
per-(shot, chord) statistics over the whole dataset, writes ONE figure +
one CSV.

    pixi run python -u -m cer_transfer.analysis.compare_arrays \
        --active-list splits/nstx_active_train.txt \
        --passive-list splits/nstx_passive_train.txt \
        --out figs/compare_nstx [--limit 500] [--rail 4095]

Per (shot, chord): continuum level (per-spectrum median counts, time-median)
and line amplitude (continuum-subtracted clipped sum, time-median). Per
chord, aggregated over shots: median continuum, median line amplitude,
median active/passive amplitude ratio (paired per shot), and passive
saturation fraction (counts >= --rail). Duplicate passive rows (48 slots
from ~31 fibers) are detected per shot via identical-row checks and the
distinct-fiber count is reported.

Figure (4 stacked panels, both arrays overlaid where applicable):
  1) pooled count histogram (log-x)
  2) per-chord median line amplitude
  3) per-chord median A/P amplitude ratio (log-y; note: ~x3 of it is
     instrumental — array A sums ~3 CCD bins, array B selects one)
  4) per-chord passive saturation fraction
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import load

from cer_transfer.configs import data_path

sns.set_style("whitegrid")


def shot_stats(fp: Path, rail: float):
    d = load(data_path(fp), mmap_mode="r")
    end = int(d["end_index"])
    if end <= 0:
        return None
    spec = np.asarray(d["input"][:, :end, :], dtype=np.float64)
    if np.isnan(spec).any():
        return None
    cont = np.median(spec, axis=-1)                    # (C, T)
    line = np.clip(spec - cont[..., None], 0, None).sum(-1)
    # distinct rows: chords whose spectrogram equals a previous chord's
    C = spec.shape[0]
    distinct = np.ones(C, dtype=bool)
    first_frame = spec[:, 0, :]
    for c in range(1, C):
        if distinct[c] and np.array_equal(first_frame[c], first_frame[c - 1]):
            if np.array_equal(spec[c], spec[c - 1]):
                distinct[c] = False
    return {
        "cont": np.median(cont, axis=1),               # (C,)
        "line": np.median(line, axis=1),               # (C,)
        "sat": (spec >= rail).mean(axis=(1, 2)),       # (C,)
        # mean RAW spectrum PER CHORD (C, W): the figure's core quantity.
        # Frame filtering (beam-on/off) plugs in here once the extraction
        # stores a per-frame beam_on key.
        "mean_cw": spec.mean(axis=1).astype(np.float32).copy(),
        "sub1": float((spec < 1.0).mean()),            # sub-count fraction
        "n_distinct": int(distinct.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--active-list", type=Path, required=True)
    p.add_argument("--passive-list", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--rail", type=float, default=4095.0)
    p.add_argument("--out", type=Path, required=True,
                   help="output prefix, e.g. figs/compare_nstx")
    p.add_argument("--chords", type=str, default="",
                   help="comma-separated chord indices for the line-plot "
                        "row (default: 6 evenly spaced)")
    args = p.parse_args()

    def read_list(f):
        return {Path(ln.strip()).stem: Path(ln.strip())
                for ln in f.read_text().splitlines()
                if ln.strip() and not ln.strip().startswith("#")}

    act, pas = read_list(args.active_list), read_list(args.passive_list)
    shots = sorted(set(act) & set(pas))
    if args.limit:
        shots = shots[: args.limit]
    print(f"paired shots: {len(shots)} "
          f"(active-only {len(set(act) - set(pas))}, "
          f"passive-only {len(set(pas) - set(act))})")

    keys = ("cont", "line", "sat", "mean_cw", "sub1")
    A = {k: [] for k in keys}
    P = {k: [] for k in keys}
    ratio = []
    n_distinct = []
    skipped = 0
    for i, s in enumerate(shots):
        sa = shot_stats(act[s], args.rail)
        sp = shot_stats(pas[s], args.rail)
        if sa is None or sp is None or len(sa["cont"]) != len(sp["cont"]):
            skipped += 1
            continue
        for k in keys:
            A[k].append(sa[k])
            P[k].append(sp[k])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio.append(sa["line"] / np.where(sp["line"] > 0, sp["line"],
                                               np.nan))
        n_distinct.append(sp["n_distinct"])
        if (i + 1) % 200 == 0:
            print(f"... {i + 1}/{len(shots)}")
    print(f"skipped (bad/NaN/shape): {skipped}")
    print(f"passive distinct fibers per shot: median "
          f"{int(np.median(n_distinct))} of {len(A['cont'][0])} slots")

    for lab, D in (("active", A), ("passive", P)):
        f = float(np.mean(D["sub1"]))
        if f > 1e-6:
            print(f"WARNING: {lab} spectra contain values < 1 count "
                  f"(fraction {f:.2e}) — check extraction units/offsets")
    a_cont = np.vstack(A["cont"]); p_cont = np.vstack(P["cont"])
    a_line = np.vstack(A["line"]); p_line = np.vstack(P["line"])
    p_sat = np.vstack(P["sat"]); a_sat = np.vstack(A["sat"])
    rat = np.vstack(ratio)
    C = a_cont.shape[1]
    x = np.arange(C)

    print(f"\n{'':>10} {'continuum':>10} {'line amp':>10} {'sat frac':>9}")
    for lab, cont, line, sat in (("active", a_cont, a_line, a_sat),
                                 ("passive", p_cont, p_line, p_sat)):
        print(f"{lab:>10} {np.median(cont):>10.4g} {np.median(line):>10.4g} "
              f"{sat.mean():>9.2e}")
    print(f"median A/P line-amplitude ratio: {np.nanmedian(rat):.3g} "
          f"(~3x of this is instrumental: A sums ~3 CCD bins, B selects 1)")

    def build(ctx):
        # native W varies per shot (79-87 bins): crop everything to the
        # common minimum so shots and arrays stack
        Wmin = min(m.shape[1] for m in A["mean_cw"] + P["mean_cw"])
        ma = np.mean(np.stack([m[:, :Wmin] for m in A["mean_cw"]]), axis=0)
        mp = np.mean(np.stack([m[:, :Wmin] for m in P["mean_cw"]]), axis=0)
        C_, W_ = ma.shape
        if args.chords:
            sel = [int(s) for s in args.chords.split(",")]
        else:
            sel = list(np.linspace(0, C_ - 1, 6).astype(int))

        size = (7.0, 6.6) if ctx == "paper" else (13, 10.5)
        fig = plt.figure(figsize=size)
        gs = fig.add_gridspec(3, 3, height_ratios=[1.4, 1, 1],
                              hspace=0.62, wspace=0.3)

        # Row 1: heatmaps. Per-chord baseline (median over wavelength)
        # removed HERE ONLY: raw pedestal offsets between chords otherwise
        # dominate the color scale and hide the line structure. The line
        # plots below stay raw.
        ha = ma - np.median(ma, axis=1, keepdims=True)
        hp = mp - np.median(mp, axis=1, keepdims=True)
        vmax = np.percentile(np.concatenate([ha, hp]), 99.5)
        vmin = 0.0
        for j, (m, title) in enumerate(((ha, "active"), (hp, "passive"))):
            ax = fig.add_subplot(gs[0, j])
            im = ax.imshow(m, aspect="auto", origin="lower", cmap="Blues",
                           vmin=vmin, vmax=vmax)
            ax.set_title(title)
            if j == 0:
                ax.set_ylabel("chord (core -> edge)")
            else:
                ax.set_yticklabels([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                         label="above baseline" if j == 1 else None)
        axd = fig.add_subplot(gs[0, 2])
        diff = ha - hp
        dmax = np.percentile(np.abs(diff), 99.5)
        imd = axd.imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r",
                         vmin=-dmax, vmax=dmax)
        axd.set_title("active - passive")
        axd.set_yticklabels([])
        fig.colorbar(imd, ax=axd, fraction=0.046, pad=0.03)

        # Rows 2-3: per-chord overlays (linear y, per-panel scale)
        for k, cc in enumerate(sel[:6]):
            ax = fig.add_subplot(gs[1 + k // 3, k % 3])
            ax.plot(ma[cc], color="C0", lw=1.4, label="active")
            ax.plot(mp[cc], color="C1", lw=1.4, label="passive")
            ax.set_title(f"chord {cc}", fontsize="medium")
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            if k % 3 == 0:
                ax.set_ylabel("counts")
            if k // 3 == 1:
                ax.set_xlabel("wavelength bin")
            if k == 0:
                ax.legend(fontsize="small")
        fig.suptitle("mean raw spectrum per chord "
                     f"({len(A['mean_cw'])} shots averaged)")
        return fig

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            fig = build(ctx)
            fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)
    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chord", "act_cont", "pas_cont", "act_line", "pas_line",
                    "ap_ratio", "pas_sat_frac"])
        for c in range(C):
            w.writerow([c, f"{np.median(a_cont[:, c]):.5g}",
                        f"{np.median(p_cont[:, c]):.5g}",
                        f"{np.median(a_line[:, c]):.5g}",
                        f"{np.median(p_line[:, c]):.5g}",
                        f"{np.nanmedian(rat[:, c]):.5g}",
                        f"{p_sat[:, c].mean():.3e}"])
    print(f"wrote {args.out}.png/.pdf/.csv")


if __name__ == "__main__":
    main()
