r"""Chronological NSTX study: transfer to later campaigns.

Figure per model (``chrono_campaigns`` = DIII-D+NSTX, fine-tuned; ``chrono_campaigns_scratch``
= NSTX, from scratch), panels (a) T_i and (b) v_tor. Rows: every training window with a
finished result for either model (same rows in the same order in both figures: by window
start, then length; missing results are empty cells), below a top row "random split (ceiling)" (the random-split model on the test discharges of each
campaign). Columns: campaigns FY2005-FY2010. In each row a dark grey bar spans the training
campaigns; every later campaign is a cell coloured by R2 (all labels) with the value
inside (shared colour scale 0.60-0.90; no colourbar, the values are printed). The grey
bars carry the text "training campaigns", or "training" in a single campaign (no legend).

Above every campaign column an inset shows the distribution of that campaign's labels
(KDE of all labels of its discharges, T_i in panel a, v_tor in panel b; shared x range per
panel; T_i on a log axis), so a hard campaign can be compared with what it contains. The label samples are
read from the discharge files of splits/chrono/nstx_<campaign>.txt and cached in
gallery/chrono/campaign_labels.npz.

Inputs: gallery/chrono/nstx_chrono_<finetune|scratch>_<window>_on_<campaign>.metrics.txt
(global_metrics output) and random_split_reference.json (per-campaign test R2). The
plotted matrices are printed as tables.

    pixi run python -m cer_transfer.figures.chrono_campaigns --out figs
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize

from cer_transfer.figures import style
from cer_transfer.figures.paperfig import PAPER_W, bbox

CAMPAIGNS = ["FY2005", "FY2006", "FY2007", "FY2008", "FY2009", "FY2010"]
TARGETS = ("ti", "vtor")
MODES = ("finetune", "scratch")
MODEL_NAME = {"finetune": "DIII-D+NSTX model (fine-tuned)", "scratch": "NSTX model (from scratch)"}
WINDOWS = [(i, j) for i in range(5) for j in range(i, 5)]  # 15 windows of consecutive campaigns
VMIN, VMAX = 0.60, 0.90
# white -> model colour -> dark model colour (fine-tuned blue, from scratch vermillion, as in the other figures)
CMAP = {"finetune": LinearSegmentedColormap.from_list("r2_ft", ["#ffffff", style.COLORS["finetuned"], "#00304d"]),
        "scratch": LinearSegmentedColormap.from_list("r2_sc", ["#ffffff", style.COLORS["scratch"], "#5c2800"])}
TRAIN_GREY = "0.35"


def short(c):
    """'FY2007' -> '07'."""
    return c[-2:]


def win_name(i, j):
    """Window of campaigns i..j (file naming)."""
    return CAMPAIGNS[i] if i == j else f"{CAMPAIGNS[i]}-{CAMPAIGNS[j]}"


N_SHOTS = {}  # discharges per campaign (main: from the campaign lists)


def win_label(i, j):
    """Row label: 'FY2005' or 'FY2005–FY2007'."""
    return CAMPAIGNS[i] if i == j else f"{CAMPAIGNS[i]}–{CAMPAIGNS[j]}"


def read_r2(path):
    """R2 on all labels per target from a global_metrics output, or None."""
    if not path.exists():
        return None
    out = {}
    for ln in open(path):
        m = re.match(r"(ti|vtor): all labels .*?R2 (-?[0-9.]+)", ln)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out if set(out) == set(TARGETS) else None


def load_results(directory):
    """{(mode, i, j, k): {target: R2}} for every finished window-campaign pair."""
    res = {}
    for mode in MODES:
        for i, j in WINDOWS:
            for k in range(j + 1, 6):
                v = read_r2(directory / f"nstx_chrono_{mode}_{win_name(i, j)}_on_{CAMPAIGNS[k]}.metrics.txt")
                if v is not None:
                    res[(mode, i, j, k)] = v
    return res


def campaign_labels(split_dir, cache, n_max=200_000, seed=0):
    """{campaign: {target: label sample}} of all labels (T_i keV, v_tor km/s) of every
    discharge of each campaign, at most n_max per campaign; cached in ``cache``."""
    if cache.exists():
        z = np.load(cache)
        return {c: {t: z[f"{c}_{t}"] for t in TARGETS} for c in CAMPAIGNS}
    from joblib import load
    from cer_transfer.configs import data_path
    rng = np.random.default_rng(seed)
    out = {}
    for c in CAMPAIGNS:
        files = [ln.strip() for ln in open(split_dir / f"nstx_{c}.txt") if ln.strip() and not ln.startswith("#")]
        ys = []
        for f in files:
            s = load(data_path(f), mmap_mode="r")
            end = int(s["end_index"])
            y = np.asarray(s["target"][:, :end] if end > 0 else s["target"], np.float32)  # Ti in keV
            ys.append(y.reshape(-1, 2))
        y = np.concatenate(ys)
        out[c] = {}
        for t, name in enumerate(TARGETS):
            v = y[np.isfinite(y[:, t]), t]
            out[c][name] = rng.choice(v, n_max, replace=False) if v.size > n_max else v
        print(f"{c}: {len(files)} discharges, {np.isfinite(y[:, 0]).sum()} labels")
    np.savez_compressed(cache, **{f"{c}_{t}": out[c][t] for c in CAMPAIGNS for t in TARGETS})
    return out


LOG_INSET = {"ti": True, "vtor": False}  # T_i spans more than a decade: log x axis


def draw_label_insets(ax, labels, t, fs):
    """One inset per campaign column above the panel: KDE of the campaign's labels of
    target t, on an x range shared by the six insets (0.5-99.5 percentile of all labels).
    T_i: KDE of log10 T_i (positive labels) on a log x axis."""
    from scipy.stats import gaussian_kde
    log = LOG_INSET[t]
    tf = (lambda v: np.log10(v[v > 0])) if log else (lambda v: v)
    pooled = np.concatenate([tf(labels[c][t]) for c in CAMPAIGNS])
    lo, hi = np.percentile(pooled, [0.5, 99.5])
    grid = np.linspace(lo, hi, 200)
    dens = []
    for c in CAMPAIGNS:
        v = tf(labels[c][t])
        v = v[(v >= lo) & (v <= hi)]
        sub = v[:: max(1, v.size // 20_000)]  # KDE on <= 20k points
        dens.append(gaussian_kde(sub)(grid))
    top = 1.15 * max(d.max() for d in dens)
    n_rows = ax.get_ylim()[0] + 0.5
    hfrac = 1.6 / n_rows  # inset height: 1.6 rows
    insets = []
    for k, (c, d) in enumerate(zip(CAMPAIGNS, dens)):
        ia = ax.inset_axes([k - 0.45, 1.0 + 0.25 / n_rows, 0.9, hfrac], transform=ax.get_xaxis_transform())
        x = 10 ** grid if log else grid
        ia.fill_between(x, d, color="0.8", lw=0)
        ia.plot(x, d, color="0.3", lw=0.8)
        if log:
            ia.set_xscale("log")
            ia.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1, 1.0]))
            ia.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ia.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        else:
            ia.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2, prune=None))
        ia.set_xlim(x[0], x[-1])
        ia.set_ylim(0, top)
        ia.set_yticks([])
        # tick labels only under FY2005 (its column has no ceiling cell below), bare ticks elsewhere
        ia.tick_params(axis="x", labelsize=fs - 1, length=1.5, pad=1, labelbottom=k == 0)
        ia.grid(False)
        ia.set_facecolor("none")
        for sp in ("left", "right", "top"):
            ia.spines[sp].set_visible(False)
        ia.spines["bottom"].set_linewidth(0.5)
        insets.append(ia)
    return insets


def finished_windows(res):
    """Windows with at least one finished result for either model (same rows in both
    figures), ordered by start, then length; missing results stay empty cells."""
    done = [(i, j) for i, j in WINDOWS
            if any((m, i, j, k) in res for m in MODES for k in range(j + 1, 6))]
    return sorted(done, key=lambda w: (w[0], w[1]))  # by start, then length: a staircase of training bars


def text_colour(v, cmap):
    """White on dark cells, black on light ones."""
    r, g, b, _ = cmap(Normalize(VMIN, VMAX)(v))
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"


def draw_window_panel(ax, res, ref, mode, t, windows, fs):
    """Rows = ceiling + windows, columns = campaigns 05-10: training bar and R2 cells."""
    norm = Normalize(VMIN, VMAX)
    rows = [None] + windows
    for r, w in enumerate(rows):
        if w is None:  # ceiling row: random split on every campaign's test discharges
            cells = {k: ref[mode][CAMPAIGNS[k]][t] for k in range(1, 6)}  # FY2005: nothing to compare with
        else:
            i, j = w
            ax.add_patch(plt.Rectangle((i - 0.45, r - 0.3), j - i + 0.9, 0.6, color=TRAIN_GREY, lw=0))
            # the bar labels itself (no legend); "training" where "training campaigns" is too long
            ax.text((i + j) / 2, r, "training campaigns" if j > i else "training",
                    ha="center", va="center", color="white", fontsize=fs - 1)
            cells = {k: res[(mode, i, j, k)][t] for k in range(j + 1, 6) if (mode, i, j, k) in res}
        for k, v in cells.items():
            ax.add_patch(plt.Rectangle((k - 0.5, r - 0.5), 1, 1, color=CMAP[mode](norm(v)), lw=0))
            ax.text(k, r, f"{v:.2f}", ha="center", va="center", fontsize=fs, color=text_colour(v, CMAP[mode]))
    ax.axhline(0.5, color="k", lw=0.6)  # ceiling row separated by a thin rule
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(6))
    # column labels with the discharges per campaign: 'FY2005 (1033)'
    ax.set_xticklabels([f"{c} ({N_SHOTS[c]})" if c in N_SHOTS else c for c in CAMPAIGNS],
                       rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["random split (ceiling)"] + [win_label(*w) for w in windows])
    ax.tick_params(length=0, labelsize=fs + 1)
    ax.set_xlabel("campaign (discharges)" if N_SHOTS else "campaign", fontsize=fs + 2)
    ax.set_ylabel("training window", fontsize=fs + 2)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def panel_a(ax, res, ref, mode, windows, labels, fs):
    """(a) T_i, with the label distribution of every campaign on top."""
    draw_window_panel(ax, res, ref, mode, "ti", windows, fs)
    ins = draw_label_insets(ax, labels, "ti", fs)
    ins[0].set_title(style.TARGET_LABEL["ti"], loc="left", fontsize=fs + 1, pad=2)


def panel_b(ax, res, ref, mode, windows, labels, fs):
    """(b) v_tor, with the label distribution of every campaign on top."""
    draw_window_panel(ax, res, ref, mode, "vtor", windows, fs)
    ins = draw_label_insets(ax, labels, "vtor", fs)
    ins[0].set_title(style.TARGET_LABEL["vtor"], loc="left", fontsize=fs + 1, pad=2)


def print_tables(res, ref, mode, windows):
    """The plotted matrices, one per target (#### = training campaigns)."""
    for t in TARGETS:
        print(f"\n{MODEL_NAME[mode]}, {t}: R2 (all labels); columns = campaign")
        print(f"{'window':>24s} " + " ".join(f"{c:>6s}" for c in CAMPAIGNS))
        print(f"{'random split (ceiling)':>24s}       " + " ".join(f"{ref[mode][c][t]:6.3f}" for c in CAMPAIGNS[1:]))
        for i, j in windows:
            row = [(f"{res[(mode, i, j, k)][t]:6.3f}" if (mode, i, j, k) in res else "  ----") if k > j
                   else ("  ####" if k >= i else "      ") for k in range(6)]
            print(f"{win_label(i, j):>24s} " + " ".join(row))


def window_figure(res, ref, mode, labels, out, name):
    """Two panels (T_i, v_tor) for one model; talk png and paper pdf."""
    windows = finished_windows(res)
    print_tables(res, ref, mode, windows)
    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx), sns.axes_style("whitegrid"):
            paper = ctx == "paper"
            fs = 6 if paper else 11
            h = 1.25 + 0.2 * (len(windows) + 1)  # + the label insets on top
            fig, axes = plt.subplots(1, 2, figsize=(PAPER_W, h) if paper else (15, 2.3 * h), layout="constrained")
            panel_a(axes[0], res, ref, mode, windows, labels, fs)
            panel_b(axes[1], res, ref, mode, windows, labels, fs)
            axes[1].set_ylabel("")
            axes[1].set_yticklabels([])
            style.panel_letters(axes, x=-0.02, y=1.0 + 2.2 / (len(windows) + 1), fontsize=fs + 3)
            fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches=bbox(ext))
            plt.close(fig)
    print(f"wrote {out}/{name}.png/.pdf ({len(windows)} finished windows)")


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=Path("gallery/chrono"))
    p.add_argument("--reference", type=Path, default=Path("gallery/chrono/random_split_reference.json"))
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", default="chrono_campaigns")
    p.add_argument("--splits", type=Path, default=Path("splits/chrono"), help="campaign discharge lists")
    p.add_argument("--labels-cache", type=Path, default=Path("gallery/chrono/campaign_labels.npz"))
    args = p.parse_args()
    ref = json.load(open(args.reference))
    res = load_results(args.dir)
    print(f"{len(res)} finished window-campaign results")
    labels = campaign_labels(args.splits, args.labels_cache)
    for c in CAMPAIGNS:  # discharges per campaign, shown in the column labels
        N_SHOTS[c] = sum(1 for ln in open(args.splits / f"nstx_{c}.txt") if ln.strip() and not ln.startswith("#"))
    window_figure(res, ref, "finetune", labels, args.out, args.out_name)
    window_figure(res, ref, "scratch", labels, args.out, f"{args.out_name}_scratch")


if __name__ == "__main__":
    main()
