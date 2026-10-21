"""Paper/analysis plots. Three subcommands (see bottom for CLI).

Figures are rendered twice: PNG in seaborn context "talk" (slides) and PDF
in context "paper" (manuscript), default font sizes in each.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.patches
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from matplotlib.lines import Line2D

from cer_transfer.configs import data_path, get_machine
from cer_transfer.figures import style
from cer_transfer.figures.paperfig import PAPER_W, bbox

sns.set_style("whitegrid")
UNITS = {"ti": "keV", "vtor": "km/s"}
DISPLAY = {"ti": "Ti", "vtor": "vtor"}
SCALE = {"ti": 1e-3, "vtor": 1.0}  # canonical eV -> display keV
AXLBL = {"ti": "T$_i$ (keV)", "vtor": "v$_{tor}$ (km/s)"}


def _render(build_fig, out: Path, name: str):
    """Render twice: talk context to png, paper context to pdf.

    ``build_fig`` receives the context name and must size the figure
    accordingly.
    """
    out.mkdir(parents=True, exist_ok=True)
    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            fig = build_fig(ctx)
            if fig.get_layout_engine() is None:  # figures with their own engine keep it
                fig.tight_layout()
            fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches=bbox(ext))
            plt.close(fig)
    print(f"wrote {out / name}.png/.pdf")


def _read_files(args, machine):
    files = [
        data_path(ln.strip())
        for ln in args.list.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if args.limit:
        files = files[: args.limit]
    scale = np.asarray(machine.target_scale, dtype=np.float64)
    for fp in files:
        d = load(data_path(fp), mmap_mode="r")
        end = int(d["end_index"])
        if end <= 0:
            continue
        y = np.asarray(d["target"][:, :end, :], dtype=np.float64) * scale
        if args.first_chords:
            y = y[: args.first_chords]
        yield fp, y


def cmd_hist(args):
    """Subcommand: label histograms per chord and pooled."""
    machine = get_machine(args.machine)
    per_chord = None
    pooled = [[], []]
    for _, y in _read_files(args, machine):
        C = y.shape[0]
        if per_chord is None:
            per_chord = [[[] for _ in range(C)], [[] for _ in range(C)]]
        for t in range(2):
            v = y[..., t] * SCALE[machine.targets[t]]
            pooled[t].append(v[np.isfinite(v)])
            for c in range(C):
                vc = v[c][np.isfinite(v[c])]
                if len(vc):
                    per_chord[t][c].append(vc)
    pooled = [np.concatenate(p) for p in pooled]

    def build_summary(ctx):
        """Pooled label histograms, one panel per target."""
        size = (7.0, 2.6) if ctx == "paper" else (11, 4)
        fig, axes = plt.subplots(1, 2, figsize=size)
        for t, (ax, name) in enumerate(zip(axes, machine.targets)):
            v = pooled[t]
            if name == "ti":
                vpos = v[v > 0]
                bins = np.geomspace(
                    vpos.min(), vpos.max(), int(np.sqrt(len(vpos))) // 4 or 50
                )
                sns.histplot(
                    vpos,
                    bins=bins,
                    ax=ax,
                    stat="percent",
                    color=f"C{t}",
                    edgecolor=None,
                )
                ax.set_xscale("log")
            else:
                sns.histplot(
                    v, bins="sqrt", ax=ax, stat="percent", color=f"C{t}", edgecolor=None
                )
            ax.set_xlabel(AXLBL[name])
            ax.set_title(f"{machine.name}: {DISPLAY[name]} targets " f"(n={len(v):,})")
        fig.tight_layout()
        return fig

    _render(build_summary, args.out, f"{machine.name}_hist_summary")

    C = len(per_chord[0])
    chord_vals = [
        [
            np.concatenate(per_chord[t][c]) if per_chord[t][c] else np.array([])
            for c in range(C)
        ]
        for t in range(2)
    ]

    for t, name in enumerate(machine.targets):

        def build_chords(ctx, t=t, name=name):
            """Per-chord label histograms."""
            ncols = 8
            nrows = int(np.ceil(C / ncols))
            cw, ch = (7.0 / ncols, 0.75) if ctx == "paper" else (2.4, 1.9)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(cw * ncols, ch * nrows), sharex=True
            )
            axes = np.atleast_2d(axes)
            for c in range(nrows * ncols):
                ax = axes[c // ncols, c % ncols]
                if c >= C or len(chord_vals[t][c]) == 0:
                    ax.set_axis_off()
                    continue
                sns.histplot(
                    chord_vals[t][c],
                    bins="sqrt",
                    ax=ax,
                    stat="percent",
                    color=f"C{t}",
                    edgecolor=None,
                )
                ax.set_title(f"ch {c}")
                ax.set_xlabel("")
                ax.set_ylabel("")
            fig.suptitle(
                f"{machine.name}: {DISPLAY[name]} targets per " f"chord ({AXLBL[name]})"
            )
            fig.tight_layout()
            return fig

        _render(build_chords, args.out, f"{machine.name}_hist_chords_{DISPLAY[name]}")


def cmd_recon(args):
    """Subcommand: reconstruction vs fits for one discharge."""
    import torch

    from cer_transfer.configs import ModelConfig
    from cer_transfer.data import ShotDataset
    from cer_transfer.models import build_model
    from cer_transfer.training import denormalize_mu_sigma, load_checkpoint

    ckpt = load_checkpoint(args.checkpoint)
    mk = ckpt["machine"]
    machine = get_machine(mk["name"] if isinstance(mk, dict) else mk)
    model = build_model(machine, ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    stats = {k: v.to(device) for k, v in ckpt["norm_stats"].items()}

    files = [
        data_path(ln.strip())
        for ln in args.list.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    ds = ShotDataset([files[args.shot_index]], machine, subseq_len=-1, mmap=False)
    spec, tgt_t, err_t, mom_t = ds[0]
    with torch.no_grad():
        mu_n, sigma_n = model(
            spec.unsqueeze(0).to(device), mom_t.unsqueeze(0).to(device)
        )
        mu_d, sigma_d = denormalize_mu_sigma(mu_n, sigma_n, stats, machine.targets)
    mu = mu_d[0].cpu().numpy()
    sigma = sigma_d[0].cpu().numpy()
    tgt = tgt_t.numpy()
    err = err_t.numpy()
    T = mu.shape[1]
    tms = np.arange(T) * 1000.0 / args.rate_hz
    c = args.chord
    shot_name = Path(str(files[args.shot_index])).stem

    def build(ctx):
        """Draw the figure for the current plotting context."""
        size = (7.0, 4.0) if ctx == "paper" else (12, 6.5)
        fig = plt.figure(figsize=size)
        gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], hspace=0.35, wspace=0.3)
        for t, name in enumerate(machine.targets):
            s = SCALE[name]
            ax = fig.add_subplot(gs[t, 0])
            ax.plot(tms, mu[c, :, t] * s, color="C0", lw=1.4, label="Reconstruction")
            ax.fill_between(
                tms,
                (mu[c, :, t] - sigma[c, :, t]) * s,
                (mu[c, :, t] + sigma[c, :, t]) * s,
                color="C0",
                alpha=0.25,
            )
            m = np.isfinite(tgt[c, :, t])
            ax.errorbar(
                tms[m],
                tgt[c, m, t] * s,
                yerr=err[c, m, t] * s,
                fmt=".",
                ms=4,
                color="C1",
                ecolor="C1",
                elinewidth=0.8,
                capsize=0,
                label="Ground truth",
            )
            ax.set_ylabel(AXLBL[name])
            if t == 0:
                ax.legend(loc="upper right")
                ax.set_title(f"{shot_name}, chord {c}")
            if t == len(machine.targets) - 1:
                ax.set_xlabel("Time (ms)")

            axs = fig.add_subplot(gs[t, 1])
            yy = tgt[..., t].ravel() * s
            pp = mu[..., t].ravel() * s
            mm = np.isfinite(yy)
            yy, pp = yy[mm], pp[mm]
            axs.scatter(yy, pp, s=2, color="k", alpha=0.25)
            lim = np.percentile(np.concatenate([yy, pp]), [1, 99])
            axs.plot(lim, lim, color="red", alpha=0.5, lw=1)
            axs.set_xlim(lim)
            axs.set_ylim(lim)
            axs.set_xlabel(f"Ground truth {DISPLAY[name]}")
            axs.set_ylabel(f"Reconstructed {DISPLAY[name]}")
        return fig

    _render(
        build, args.out, f"{Path(args.checkpoint).stem}_shot{args.shot_index}_ch{c}"
    )


def _clean_curve(label, v, args):
    """Optionally interpolate over collapsed epochs (see --drop-outliers)."""
    if not args.drop_outliers:
        return v
    # epochs whose score falls more than --outlier-drop below the running best of the
    # preceding epochs AND below the median of a wide window are collapsed epochs
    # (resume after a SLURM requeue, bad batch): replaced by linear interpolation
    # between the neighbouring valid epochs, and reported
    vs = pd.Series(v)
    med = vs.rolling(9, center=True, min_periods=3).median().to_numpy()
    run_best = vs.cummax().shift(1).fillna(vs.iloc[0]).to_numpy()
    late = np.arange(v.size) >= 10  # early epochs climb steeply: neither rule applies
    bad = (
        (v < med - args.outlier_drop) | (v < run_best - 1.5 * args.outlier_drop)
    ) & late
    if bad.any():
        print(
            f"{label}: interpolated over collapsed epochs "
            + ", ".join(f"{e} ({v[e]:.3f})" for e in np.where(bad)[0])
        )
    return pd.Series(np.where(bad, np.nan, v)).interpolate(limit_area="inside").to_numpy()


def cmd_curves(args):
    """Subcommand: training curves from metrics CSVs.

    The --csv runs are the from-scratch baselines (colour by role from the label:
    DIII-D model green, NSTX model red); --ft-csv runs (fine-tuned DIII-D+NSTX
    models) are drawn as dashed blue lines with the early-stopping epoch (best
    val_score) marked by a star.
    """

    def read(specs, labels, default_role):
        out = []
        for i, sp in enumerate(specs or []):
            lab, path = sp.split(":", 1) if ":" in sp and not Path(sp).exists() else (None, sp)
            df = pd.read_csv(path)
            if args.metric not in df.columns:
                print(f"skip {path}: no column {args.metric}")
                continue
            if lab is None:
                lab = labels[i] if i < len(labels) else Path(path).stem.replace(".metrics", "")
            role = style.role_of(lab) or default_role
            v = _clean_curve(lab, df[args.metric].astype(float).to_numpy(), args)
            if args.x == "hours" and "epoch_seconds" in df.columns:
                x = (df["epoch_seconds"].cumsum() / 3600.0).to_numpy()
            else:
                x = np.arange(len(df))
            j = int(np.nanargmax(v))
            print(f"{lab}: best {args.metric} {v[j]:.4f} at epoch {j} of {len(v)}")
            out.append((lab, role, x, v, j))
        return out

    runs = read(args.csv, args.labels or [], None)
    ft = read(args.ft_csv, args.ft_labels or [], "finetuned")
    xlabel = (
        "Training wall-clock (h)"
        if args.x == "hours"
        else "Epoch"
    )

    def build(ctx):
        """Draw the figure for the current plotting context."""
        # with fine-tuning curves the legend goes below the axes (no free corner)
        if ft:
            size = (PAPER_W, 3.9) if ctx == "paper" else (9, 7.2)
        else:
            size = (PAPER_W, 3.2) if ctx == "paper" else (9, 5.5)
        fig, ax = plt.subplots(figsize=size)
        for i, (label, role, x, v, j) in enumerate(runs):
            c = style.COLORS[role] if role else f"C{i}"
            ax.plot(x, v, lw=1.4, color=c, label=label)
            if args.mark_best:
                ax.plot([x[j]], [v[j]], "o", color=c, ms=5)
                ax.annotate(
                    f"{v[j]:.3f}", (x[j], v[j]), xytext=(4, 5),
                    textcoords="offset points", fontsize="x-small", color=c,
                )
        dashes = [(5, 2), (2, 1.5), (8, 2, 2, 2), (1, 1)]
        for i, (label, role, x, v, j) in enumerate(ft):
            c = style.COLORS[role]
            ax.plot(x, v, lw=1.4, color=c, dashes=dashes[i % len(dashes)], label=label)
            # early-stopping epoch (best val_score): the checkpoint that is used
            ax.plot([x[j]], [v[j]], marker="*", ms=11, color=c, mec="k", mew=0.6,
                    ls="none", zorder=6)
            if args.mark_best:
                ax.annotate(
                    f"{v[j]:.3f}", (x[j], v[j]), xytext=(-6, 5), ha="right",
                    textcoords="offset points", fontsize="x-small", color=c,
                )
        if ft:
            ax.plot([], [], marker="*", ms=11, color="w", mec="k", mew=0.6, ls="none",
                    label="early-stopping epoch")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Validation score" if args.metric == "val_score" else args.metric)
        if args.ylim:
            ax.set_ylim(*args.ylim)
        if ft:
            style.ordered_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                                 ncol=2, fontsize="small" if ctx == "paper" else "x-small")
        else:
            style.ordered_legend(ax, loc="lower right", fontsize="small")
        return fig

    _render(build, args.out, args.out_name or f"learning_curves_{args.metric}")


def cmd_scaling(args):
    """Scaling figure: best val_score vs number of training shots."""

    def best(path):
        """Best (max) value of a metric column."""
        d = pd.read_csv(path)
        return float(d["val_score"].max())

    def aggregate(specs):
        """Group 'n:csv' specs by n: list of (n, mean, std, count), std=0 for one."""
        by_n = {}
        for sp in specs or []:
            n, path = sp.split(":", 1)
            by_n.setdefault(int(n), []).append(best(path))
        out = []
        for n in sorted(by_n):
            v = np.asarray(by_n[n])
            out.append(
                (
                    n,
                    float(v.mean()),
                    float(v.std(ddof=1)) if v.size > 1 else 0.0,
                    v.size,
                )
            )
        return out

    runs_agg = aggregate(args.runs)
    scr_agg = aggregate(args.scratch)
    for n, m, sd, c in runs_agg + scr_agg:
        if c > 1:
            print(f"n={n}: {c} runs, mean {m:.3f} +- {sd:.3f}")
    runs = [(n, m) for n, m, _, _ in runs_agg]
    scr = [(n, m) for n, m, _, _ in scr_agg]
    agn = sorted(
        (int(s.split(":")[0]), best(s.split(":", 1)[1]))
        for s in (getattr(args, "agnostic_runs", None) or [])
    )
    agn_scr = [
        (int(s.split(":")[0]), best(s.split(":", 1)[1]))
        for s in (getattr(args, "agnostic_scratch", None) or [])
    ]
    zero = None
    if getattr(args, "zero_shot_csv", None):
        z = pd.read_csv(args.zero_shot_csv)
        # same definition as the training's val_score: mean of the per-target R2;
        # the column selects raw R2 or the label-free protocol R2 (physics_correction
        # --csv)
        col = getattr(args, "zero_shot_column", "r2_raw")
        zero = float(z[col].mean())

    trace = None
    if getattr(args, "trace_preds", None):
        from cer_transfer.figures.timeaxis import file_times

        def chord_series(path):
            z = np.load(path)
            ch = z["chord"]
            C = int(ch.max()) + 1
            T = len(ch) // C
            if not np.array_equal(ch, np.repeat(np.arange(C), T)):
                raise SystemExit(f"{path}: points are not chord-major")
            f = lambda k: z[k].reshape(C, T, -1)
            return f("pred"), f("pred_sigma"), f("y"), f("sigma"), T

        xs_ch = pd.read_csv(args.trace_chords)
        c = int(xs_ch.iloc[int(np.argmin(np.abs(xs_ch["x"].to_numpy() - args.trace_x)))]["chord"])
        rtan = float(xs_ch.set_index("chord").loc[c, "x"])
        k = 1  # v_tor
        models = []
        for path, lab in ((args.trace_preds, args.trace_label), (args.trace_preds2, args.trace_label2)):
            if path is None:
                continue
            mu, ps, y, sl, T = chord_series(path)
            lab_ok = np.isfinite(y[c, :, k])
            mae = float(np.median(np.abs(mu[c, lab_ok, k] - y[c, lab_ok, k])))
            models.append((lab, mu[c, :, k], ps[c, :, k], mae, int(lab_ok.sum())))
            print(f"trace: {lab}: chord {c} (R_tan {rtan:.2f} m), median |error| over the "
                  f"{int(lab_ok.sum())} labelled frames of the discharge {mae:.3g} km/s")
        t = file_times(args.trace_shot, T, 200.0, args.trace_t_offset) if args.trace_shot \
            else np.arange(T) / 200.0 + args.trace_t_offset
        fr = np.where((t >= args.trace_t0) & (t <= args.trace_t1))[0]
        trace = dict(t=t[fr], y=y[c, fr, k], sl=sl[c, fr, k], models=[
            (lab, m_[fr], p_[fr], mae, n) for lab, m_, p_, mae, n in models], rtan=rtan,
            shot=args.trace_shot)

    def draw_trace(ax, ctx):
        """Panel b: core-chord v_tor trace, fit black, models with bands and MAE."""
        lw, ms = (1.0, 2.0) if ctx == "paper" else (1.5, 3.0)
        hs = []
        for lab, m_, p_, mae, n in trace["models"]:
            role = style.role_of(lab) or "finetuned"
            hs.append(style.model_line(ax, trace["t"], m_, p_, role=role,
                                       label=f"{lab}: MAE {mae:.3g} km/s", marker="o", ms=ms, lw=lw))
        fm = np.isfinite(trace["y"])
        hs.append(style.fit_points(ax, trace["t"][fm], trace["y"][fm], trace["sl"][fm], ms=ms,
                                   elinewidth=0.6 * lw))
        if args.trace_ylim:
            ax.set_ylim(*args.trace_ylim)
        ax.set_xlim(trace["t"][0], trace["t"][-1])
        ax.set_xlabel(style.TIME_LABEL)
        ax.set_ylabel(style.VTOR_LABEL)
        style.ordered_legend(ax, handles=hs, loc="upper right",
                             fontsize="xx-small" if ctx == "paper" else "x-small")

    def build(ctx):
        """Draw the figure for the current plotting context."""
        size = (PAPER_W, 3.2) if ctx == "paper" else (9, 5.5)
        if trace is not None:
            size = (PAPER_W, 3.4) if ctx == "paper" else (16, 6.0)
        xs = sorted(
            {n for n, _ in runs}
            | {n for n, _ in scr}
            | {n for n, _ in agn}
            | {n for n, _ in agn_scr}
        )
        # broken y-axis when the zero-shot line lies far below the curves: the curves
        # keep a readable scale (error bars visible), the line gets a narrow panel
        curves = runs_agg + scr_agg
        lo = min(m - sd for _, m, sd, _ in curves) if curves else None
        hi = max(m + sd for _, m, sd, _ in curves) if curves else None
        broken = zero is not None and curves and zero < lo - 0.5 * (hi - lo)
        axt = None
        if trace is not None:
            fig = plt.figure(figsize=size)
            gs = fig.add_gridspec(2, 2, height_ratios=[5, 1], width_ratios=[1.25, 1])
            if broken:
                ax = fig.add_subplot(gs[0, 0])
                axz = fig.add_subplot(gs[1, 0], sharex=ax)
            else:
                ax = axz = fig.add_subplot(gs[:, 0])
            axt = fig.add_subplot(gs[:, 1])
        elif broken:
            fig, (ax, axz) = plt.subplots(
                2, 1, sharex=True, figsize=size,
                gridspec_kw={"height_ratios": [5, 1]},  # no hspace: breaks tight_layout
            )
        else:
            fig, ax = plt.subplots(figsize=size)
            axz = ax
        eb = dict(capsize=4, capthick=1.4, elinewidth=1.4, lw=1.8)
        handles = []
        if zero is not None:
            handles.append(
                axz.axhline(
                    zero,
                    color=style.COLORS["zero_shot"],
                    ls="-.",
                    lw=1.4,
                    label=getattr(args, "zero_shot_label", None) or "DIII-D model",
                )
            )
            if runs:
                ref = runs[-1][1]
                print(
                    f"zero-shot score {zero:.3f} = {100 * zero / ref:.0f} % of the "
                    f"full-data fine-tuned score {ref:.3f}"
                )
        if scr_agg:
            handles.append(
                ax.errorbar(
                    [n for n, _, _, _ in scr_agg],
                    [m for _, m, _, _ in scr_agg],
                    yerr=[sd for _, _, sd, _ in scr_agg],
                    fmt="s-",
                    mfc="none",
                    color=style.COLORS["scratch"],
                    ms=6,
                    label=getattr(args, "scratch_label", None) or "NSTX model",
                    **eb,
                )
            )
        handles.append(
            ax.errorbar(
                [n for n, _, _, _ in runs_agg],
                [m for _, m, _, _ in runs_agg],
                yerr=[sd for _, _, sd, _ in runs_agg],
                fmt="o-",
                color=style.COLORS["finetuned"],
                ms=5,
                label=getattr(args, "runs_label", None) or "DIII-D+NSTX model",
                **eb,
            )
        )
        if agn:
            handles.append(
                ax.plot(
                    [n for n, _ in agn],
                    [v for _, v in agn],
                    "o--",
                    color="C2",
                    lw=1.6,
                    label=args.agnostic_label,
                )[0]
            )
        for i, (n, v) in enumerate(agn_scr):
            h = ax.plot(
                [n],
                [v],
                "D",
                mfc="none",
                color="C2",
                ms=7,
                label="machine-agnostic, trained from scratch" if i == 0 else None,
            )[0]
            if i == 0:
                handles.append(h)
        if args.annotate and runs and scr:
            x0, y0 = runs[0]
            xF, yF = runs[-1]
            s_full = max(v for _, v in scr)
            # annotation 1: smallest subset relative to the full-data score
            ax.annotate(
                f"{x0} discharges:\n{100*y0/yF:.0f}% of full-data " "score",
                xy=(x0, y0),
                xytext=(x0 * 2.1, y0 + 0.002),
                fontsize=8,
                color="C0",
                arrowprops=dict(arrowstyle="->", color="C0", lw=1.0),
            )
            # annotation 2 (optional): equivalence point vs the full from-scratch model
            if getattr(args, "annotate_match", False):
                ax.axhline(s_full, color="C3", ls=":", lw=1.0, alpha=0.8)
            eq = next((n for n, v in runs if v >= s_full), None)
            if eq is not None and getattr(args, "annotate_match", False):
                ax.annotate(
                    f"{eq} discharges match\nfrom-scratch on all "
                    f"{scr[0][0] if len(scr)==1 else max(n for n,_ in scr)}",
                    xy=(eq, s_full),
                    xytext=(eq * 0.28, s_full + 0.004),
                    fontsize=8,
                    color="C3",
                    arrowprops=dict(arrowstyle="->", color="C3", lw=1.0),
                )
        if getattr(args, "scratch_full_line", False) and scr:
            n_full, s_full = max(scr)
            handles.append(ax.axhline(s_full, color=style.COLORS["scratch"], ls=":", lw=1.2,
                                      label=f"{getattr(args, 'scratch_label', None) or 'NSTX model'}, "
                                            f"all {n_full} discharges"))
        if args.ceiling:
            handles.append(
                ax.axhline(
                    args.ceiling[0],
                    color="gray",
                    ls="--",
                    lw=1.2,
                    label="label-noise ceiling",
                )
            )
            if len(args.ceiling) > 1:
                ax.axhline(args.ceiling[1], color="gray", ls=":", lw=1.0)
        if broken:
            pad = 0.12 * (hi - lo)
            # extra headroom: the legend sits top left above the curves
            ax.set_ylim(lo - pad, hi + pad + 0.55 * (hi - lo))
            half = 0.5 * (hi - lo + 2 * pad + 0.55 * (hi - lo)) / 5  # same y-scale
            axz.set_ylim(zero - half, zero + half)
            axz.set_yticks([round(zero, 3)])
            ax.spines["bottom"].set_visible(False)
            axz.spines["top"].set_visible(False)
            ax.tick_params(axis="x", which="both", bottom=False)
            d = 0.012  # diagonal break marks
            kw = dict(transform=ax.transAxes, color="k", clip_on=False, lw=1)
            ax.plot((-d, +d), (-d * 1.0, +d * 1.0), **kw)
            ax.plot((1 - d, 1 + d), (-d * 1.0, +d * 1.0), **kw)
            kw["transform"] = axz.transAxes
            ax.plot((-d, +d), (1 - 5 * d, 1 + 5 * d), **kw)
            ax.plot((1 - d, 1 + d), (1 - 5 * d, 1 + 5 * d), **kw)
        axz.set_xscale("log")
        axz.set_xticks(xs)
        axz.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axz.minorticks_off()
        axz.tick_params(axis="x", rotation=30)
        axz.set_xlabel("Number of NSTX discharges")
        ylabel = getattr(args, "ylabel", None) or "NSTX validation score"
        if broken and axt is None:
            fig.supylabel(ylabel, fontsize=ax.yaxis.label.get_size())
        else:
            ax.set_ylabel(ylabel)
        # top axis: the same discharge positions in days of operation
        # (style.DISCHARGES_PER_DAY discharges per day); same log scale
        # (a twin axis, not secondary_xaxis: tight_layout handles twins only)
        per_day = float(style.DISCHARGES_PER_DAY)
        days = ax.twiny()
        days.set_xscale("log")
        days.set_xlim(axz.get_xlim())
        days.set_xticks(xs)
        days.set_xticklabels([f"{n / per_day:.0f}" for n in xs])
        days.minorticks_off()
        days.set_xlabel("Days of operation")
        days.grid(False)
        days.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        if broken:  # twiny() switched the bottom ticks of ax back on
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        if args.sizes_csv is not None:
            szd = pd.read_csv(args.sizes_csv)
            m = szd["list"].str.match(r"nstx_train(_r\d+)?\.txt$")
            if m.any():
                szd = szd[m]
            sz = szd.drop_duplicates("shots").set_index("shots")["frames"]
            fr = [float(sz.get(n, np.nan)) for n in xs]
            top = ax.twiny()
            top.set_xscale("log")
            top.set_xlim(axz.get_xlim())
            top.spines["top"].set_position(("outward", 42))
            top.grid(False)
            top.set_xticks(xs)
            top.set_xticklabels(
                [f"{f/1e3:.0f}k" if f < 1e6 else f"{f/1e6:.2f}M" for f in fr],
                rotation=30,
            )
            top.minorticks_off()
            top.set_xlabel("Number of frames")
        if axt is not None:  # merged figure: compact two-column legend in the headroom
            style.ordered_legend(ax, handles=handles, loc="upper left", ncol=2,
                                 fontsize="xx-small" if ctx == "paper" else "x-small")
        else:
            style.ordered_legend(ax, handles=handles, loc="upper left")
        if axt is not None:
            draw_trace(axt, ctx)
            style.panel_letters([ax, axt], x=-0.12, y=1.18 if ctx == "talk" else 1.22)
            axt.text(0.0, 1.02, f"discharge {trace['shot'].split('_')[-1].split('.')[0]}, "
                     f"$R_\\mathrm{{tan}}$ = {trace['rtan']:.2f} m" if trace["shot"] else "",
                     transform=axt.transAxes, va="bottom", fontsize="small")
        return fig

    _render(build, args.out, getattr(args, "out_name", None) or "scaling_val_score")


def cmd_scaling_targets(args):
    """Label efficiency per target: (a) R2 T_i, (b) R2 v_tor vs number of NSTX discharges
    (fine-tuned blue, scratch red, zero-shot green line), (c) core-chord T_i (top) and
    v_tor (bottom) of one discharge for two models trained on the same small subset."""
    from cer_transfer.figures.timeaxis import file_times

    keys = ("ti", "vtor")

    def best_row(path):
        d = pd.read_csv(path)
        r = d.loc[d["val_score"].idxmax()]  # the epoch that selects the checkpoint
        return {k: float(r[f"r2_{k}"]) for k in keys}

    def aggregate(specs):
        by_n = {}
        for sp in specs or []:
            n, path = sp.split(":", 1)
            by_n.setdefault(int(n), []).append(best_row(path))
        out = {k: [] for k in keys}
        for n in sorted(by_n):
            for k in keys:
                v = np.asarray([r[k] for r in by_n[n]])
                out[k].append((n, float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, v.size))
        return out

    runs, scr = aggregate(args.runs), aggregate(args.scratch)
    for name, agg in (("fine-tuned", runs), ("scratch", scr)):
        for k in keys:
            print(f"{name} {k}: " + ", ".join(f"{n}: {m:.3f}+-{sd:.3f} ({c})" for n, m, sd, c in agg[k]))
    zero = {}
    if args.zero_shot_csv:
        z = pd.read_csv(args.zero_shot_csv).set_index("target")
        zero = {k: float(z.loc[k, args.zero_shot_column]) for k in keys if k in z.index}
    xs = sorted({n for k in keys for n, *_ in runs[k] + scr[k]})

    # panel c data
    def chord_series(path):
        z = np.load(path)
        ch = z["chord"]
        C = int(ch.max()) + 1
        T = len(ch) // C
        if not np.array_equal(ch, np.repeat(np.arange(C), T)):
            raise SystemExit(f"{path}: points are not chord-major")
        f = lambda q: z[q].reshape(C, T, -1)
        return f("pred"), f("pred_sigma"), f("y"), f("sigma"), T

    chords = pd.read_csv(args.trace_chords)
    c = int(chords.iloc[int(np.argmin(np.abs(chords["x"].to_numpy() - args.trace_x)))]["chord"])
    rtan = float(chords.set_index("chord").loc[c, "x"])
    models = []
    for path, lab in ((args.trace_preds, args.trace_label), (args.trace_preds2, args.trace_label2)):
        mu, ps, y, sl, T = chord_series(path)
        mae = {}
        for j, k in enumerate(keys):
            ok = np.isfinite(y[c, :, j])
            mae[k] = float(np.median(np.abs(mu[c, ok, j] - y[c, ok, j]))) * style.TARGET_SCALE[k]
        models.append((lab, mu, ps, mae))
        print(f"trace {lab}: chord {c} (R_tan {rtan:.2f} m), MAE over the labelled frames of the "
              f"discharge: T_i {mae['ti']:.3g} keV, v_tor {mae['vtor']:.3g} km/s")
    t = file_times(args.trace_shot, T, 200.0, args.trace_t_offset) if args.trace_shot \
        else np.arange(T) / 200.0 + args.trace_t_offset
    fr = np.where((t >= args.trace_t0) & (t <= args.trace_t1))[0]
    shot = (str(args.trace_shot).split("_")[-1].split(".")[0]) if args.trace_shot else ""

    def build(ctx):
        size = (PAPER_W, 4.2) if ctx == "paper" else (17, 7.0)
        small = "xx-small" if ctx == "paper" else "x-small"
        fig = plt.figure(figsize=size, layout="constrained")
        gs = fig.add_gridspec(2, 3, height_ratios=[5, 1], width_ratios=[1, 1, 1.3])
        eb = dict(capsize=3, capthick=1.2, elinewidth=1.2, lw=1.5)
        top_axes = []
        for i, k in enumerate(keys):
            ax = fig.add_subplot(gs[0, i])
            axz = fig.add_subplot(gs[1, i], sharex=ax)
            hs = []
            if k in zero:
                hs.append(axz.axhline(zero[k], color=style.COLORS["zero_shot"], ls="-.", lw=1.3,
                                      label=args.zero_shot_label))
            for agg, role, fmt, lab in ((scr, "scratch", "s-", args.scratch_label),
                                        (runs, "finetuned", "o-", args.runs_label)):
                a = agg[k]
                hs.append(ax.errorbar([n for n, *_ in a], [m for _, m, _, _ in a],
                                      yerr=[sd for _, _, sd, _ in a], fmt=fmt, color=style.COLORS[role],
                                      mfc="none" if role == "scratch" else None,
                                      ms=4 if ctx == "paper" else 6, label=lab, **eb))
            vals = [m - sd for _, m, sd, _ in runs[k] + scr[k]] + [m + sd for _, m, sd, _ in runs[k] + scr[k]]
            lo, hi = min(vals), max(vals)
            pad = 0.12 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad + 0.5 * (hi - lo))
            half = 0.5 * (hi - lo + 2 * pad + 0.5 * (hi - lo)) / 5
            if k in zero:
                axz.set_ylim(zero[k] - half, zero[k] + half)
                axz.set_yticks([round(zero[k], 3)])
            ax.spines["bottom"].set_visible(False)
            axz.spines["top"].set_visible(False)
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            d = 0.02
            kw = dict(transform=ax.transAxes, color="k", clip_on=False, lw=1)
            ax.plot((-d, +d), (-d, +d), **kw)
            ax.plot((1 - d, 1 + d), (-d, +d), **kw)
            kw["transform"] = axz.transAxes
            ax.plot((-d, +d), (1 - 5 * d, 1 + 5 * d), **kw)
            ax.plot((1 - d, 1 + d), (1 - 5 * d, 1 + 5 * d), **kw)
            axz.set_xscale("log")
            axz.set_xticks(xs)
            axz.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            axz.minorticks_off()
            axz.tick_params(axis="x", rotation=40, labelsize=small if ctx == "paper" else None)
            axz.set_xlabel("NSTX discharges")
            ax.set_ylabel(f"$R^2$, {style.TARGET_SYMBOL[k]}")
            days = ax.twiny()
            days.set_xscale("log")
            days.set_xlim(axz.get_xlim())
            days.set_xticks(xs)
            days.set_xticklabels([f"{n / style.DISCHARGES_PER_DAY:.0f}" for n in xs],
                                 fontsize=small if ctx == "paper" else None, rotation=40)
            days.minorticks_off()
            days.set_xlabel("days of operation")
            days.grid(False)
            days.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            if i == 0:
                style.ordered_legend(ax, handles=hs, loc="upper left", fontsize=small)
            top_axes.append(ax)
        # panel c: T_i (top) and v_tor (bottom)
        sub = gs[:, 2].subgridspec(2, 1)
        axc = [fig.add_subplot(sub[0]), None]
        axc[1] = fig.add_subplot(sub[1], sharex=axc[0])
        lw, ms = (1.0, 1.8) if ctx == "paper" else (1.5, 3.0)
        for j, k in enumerate(keys):
            ax = axc[j]
            sc = style.TARGET_SCALE[k]
            hs = []
            for lab, mu, ps, mae in models:
                role = style.role_of(lab) or "finetuned"
                unit = "keV" if k == "ti" else "km/s"
                hs.append(style.model_line(ax, t[fr], mu[c, fr, j] * sc, ps[c, fr, j] * sc, role=role,
                                           label=f"{lab}: MAE {mae['ti']:.2g} keV, {mae['vtor']:.2g} km/s",
                                           marker="o", ms=ms, lw=lw))
            _, _, y, sl, _ = chord_series(args.trace_preds)
            yy, ss = y[c, fr, j] * sc, sl[c, fr, j] * sc
            ok = np.isfinite(yy)
            hs.append(style.fit_points(ax, t[fr][ok], yy[ok], ss[ok], ms=ms, elinewidth=0.6 * lw))
            ax.set_ylabel(style.TARGET_LABEL[k])
            ax.set_xlim(t[fr][0], t[fr][-1])
            lo_, hi_ = np.nanmin(np.r_[yy[ok], *(m_[c, fr, j] * sc for _, m_, _, _ in models)]), \
                np.nanmax(np.r_[yy[ok], *(m_[c, fr, j] * sc for _, m_, _, _ in models)])
            ax.set_ylim(lo_ - 0.1 * (hi_ - lo_), hi_ + 0.1 * (hi_ - lo_))
            if j == 1:  # one legend for panel c: figure legend at the bottom right
                rank = {r: i for i, r in enumerate(style.LEGEND_ORDER)}
                order = sorted(range(len(hs)), key=lambda i: rank.get(style.role_of(hs[i].get_label()), 9))
                fig.legend([hs[i] for i in order], [hs[i].get_label() for i in order],
                           loc="outside lower right", fontsize=small, frameon=False,
                           title=f"c: discharge {shot},  $R_\\mathrm{{tan}}$ = {rtan:.2f} m",
                           title_fontsize=small)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        axc[0].tick_params(axis="x", labelbottom=False)
        axc[1].set_xlabel(style.TIME_LABEL)
        style.panel_letters([top_axes[0], top_axes[1], axc[0]], x=-0.18, y=1.16)
        return fig

    _render(build, args.out, args.out_name)


def _metrics_r2(path):
    """R2 on all labels per target from a global_metrics output file."""
    r2 = {}
    for line in open(path):
        m = re.match(r"(ti|vtor): all labels .*?R2 (-?[0-9.]+)", line)
        if m:
            r2[m.group(1)] = float(m.group(2))
    if set(r2) != set(TARGETS):
        raise SystemExit(f"{path}: no 'all labels' R2 for both targets")
    return r2


def cmd_nstxu(args):
    """Transfer to NSTX-U: R2 of T_i (a) and v_tor (b) on the NSTX-U test set.

    Point plot with a broken y-axis as in the scaling panels: one column per model
    (DIII-D model green, DIII-D+NSTX model blue, NSTX-U model red = the same
    architecture trained from scratch on NSTX-U). Filled: trained on the NSTX-U
    training discharges (fine-tuned, or from scratch for the NSTX-U model), mean +-
    std over the seeds (upper axis). Open: without NSTX-U data (zero-shot, one model,
    lower axis). Inputs are global_metrics outputs of the test-set dumps.
    """
    import glob

    def seeds(tag):
        files = sorted(glob.glob(str(args.dir / f"{tag}_s*_test.metrics.txt")))
        if not files:
            raise SystemExit(f"no {args.dir}/{tag}_s*_test.metrics.txt")
        return [_metrics_r2(f) for f in files]

    cols = [  # label, role (colour), marker, zero-shot runs (or None), trained runs
        ("DIII-D\nmodel", "zero_shot", "D", [_metrics_r2(args.zs_d3d)], seeds(args.ft_d3d_tag)),
        ("NSTX-U\nmodel", "scratch", "s", None, seeds(args.scratch_tag)),
        ("DIII-D+NSTX\nmodel", "finetuned", "o", [_metrics_r2(args.zs_nstx)], seeds(args.ft_nstx_tag)),
    ]
    stat = lambda runs, t: (np.mean([r[t] for r in runs]),
                            np.std([r[t] for r in runs], ddof=1) if len(runs) > 1 else 0.0)
    for lab, _, _, zs, tr in cols:
        for what, runs in (("zero-shot", zs), (f"{args.n} NSTX-U", tr)):
            if runs:
                print(f"{lab.replace(chr(10), ' '):18s} {what:11s} n={len(runs)}: " + " | ".join(
                    f"{t} {stat(runs, t)[0]:.3f} +- {stat(runs, t)[1]:.3f}" for t in TARGETS))

    # original layout: zero-shot group | trained group, one column per model
    entries = [  # tick label, role, marker, runs
        ("DIII-D model", "zero_shot", "D", cols[0][3]),
        ("DIII-D+NSTX model", "finetuned", "o", cols[2][3]),
        ("DIII-D model\nfine-tuned", "zero_shot", "D", cols[0][4]),
        ("NSTX-U model", "scratch", "s", cols[1][4]),
        ("DIII-D+NSTX model\nfine-tuned", "finetuned", "o", cols[2][4]),
    ]

    def build(ctx):
        """Draw the figure for the current plotting context."""
        paper = ctx == "paper"
        small = "x-small" if paper else "small"
        size = (PAPER_W, 2.6) if paper else (14, 5.8)
        fig, axes = plt.subplots(1, 2, figsize=size, layout="constrained", sharey=True)
        xs = np.array([0, 1, 2.6, 3.6, 4.6])  # gap between zero-shot and trained
        eb = dict(capsize=3, capthick=1.2, elinewidth=1.2, ls="none")  # as the scaling panels
        for ax, t in zip(axes, TARGETS):
            for x, (lab, role, mk, runs) in zip(xs, entries):
                m, sd = stat(runs, t)
                ax.errorbar([x], [m], yerr=[sd] if len(runs) > 1 else None, marker=mk,
                            ms=4 if paper else 7, color=style.COLORS[role],
                            mfc="none" if role == "scratch" else None, **eb)
            ax.axvline(1.8, color="0.6", lw=0.8, ls=":")
            ax.set_xticks(xs)
            ax.set_xticklabels([e[0] for e in entries], rotation=45, ha="right",
                               rotation_mode="anchor", fontsize=small)
            ax.set_xlim(-0.5, 5.1)
            ax.set_ylabel(f"$R^2$, {style.TARGET_SYMBOL[t]} (NSTX-U test)")
            ax.grid(False, axis="x")
            tf = ax.get_xaxis_transform()
            ax.text(0.5, 1.01, "zero-shot\n(no NSTX-U data)", transform=tf, ha="center",
                    va="bottom", fontsize=small, color="0.35")
            ax.text(3.6, 1.01, f"trained on\n{args.n} NSTX-U discharges", transform=tf,
                    ha="center", va="bottom", fontsize=small, color="0.35")
        axes[1].tick_params(labelleft=True)
        style.panel_letters(axes, x=-0.12, y=1.12)
        return fig

    _render(build, args.out, args.out_name)


def cmd_sizes(args):
    """Frames vs shots from a cer_transfer.analysis.count_samples CSV."""
    d = pd.read_csv(args.csv)
    if args.shots:
        d = d[d["shots"].isin(args.shots)]
        d = d[d["list"].str.contains(args.match)]
    else:
        # default: exactly the scaling-curve points — r-subsets + full train
        d = d[d["list"].str.match(r"nstx_train(_r\d+)?\.txt$")]
    d = d.sort_values("shots").drop_duplicates("shots")

    def build(ctx):
        """Draw the figure for the current plotting context."""
        size = (PAPER_W, 3.2) if ctx == "paper" else (9, 5.5)
        fig, ax = plt.subplots(figsize=size)
        ax.plot(d["shots"], d["frames"], "o-", color="C0", lw=1.8)
        ax.set_xscale("log")
        ax.set_xticks(list(d["shots"]))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.minorticks_off()
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel("Number of NSTX discharges")
        ax.set_ylabel("Number of frames")
        return fig

    _render(build, args.out, "frames_vs_shots")


def cmd_chords(args):
    """Plot per-chord R2 curves from eval_checkpoint CSVs.

    One line per model, one panel per target. An optional per-chord ceiling
    CSV (ceiling_check --per-chord --out) is drawn as a gray band/line.
    """
    frames = [
        (lab, pd.read_csv(path)) for lab, path in (s.split(":", 1) for s in args.evals)
    ]
    ceil = pd.read_csv(args.ceiling_csv) if args.ceiling_csv else None
    if args.normalize and ceil is None:
        raise SystemExit("--normalize requires --ceiling-csv")

    def build(ctx):
        """Draw the figure for the current plotting context."""
        size = (7.0, 2.8) if ctx == "paper" else (12, 4.5)
        fig, axes = plt.subplots(1, 2, figsize=size, sharex=True)
        cols = (
            (("medae_ti", "$T_i$ (eV)"), ("medae_vtor", "$v_{tor}$ (km/s)"))
            if args.mae
            else (
                (("rmse_ti", "$T_i$ (eV)"), ("rmse_vtor", "$v_{tor}$ (km/s)"))
                if (args.rmse_ratio or args.rmse)
                else (("r2_ti", "$T_i$"), ("r2_vtor", "$v_{tor}$"))
            )
        )
        for t, (col, name) in enumerate(cols):
            ax = axes[t]
            if args.mae:
                fv = ceil.set_index("chord")[col.replace("medae_", "medfloor_")]
                cv = None
            elif args.rmse_ratio or args.rmse:
                fcol = col.replace("rmse_", "floor_")
                fv = ceil.set_index("chord")[fcol]
                cv = None
            else:
                cv = (
                    ceil.set_index("chord")[col]
                    if ceil is not None and col in ceil.columns
                    else None
                )
            series = []
            for i, (lab, d) in enumerate(frames):
                y = d[col].to_numpy(dtype=float)
                x_ = d["chord"].to_numpy()
                if args.rmse_ratio:
                    y = y / fv.reindex(x_).to_numpy(dtype=float)
                elif args.rmse or args.mae:
                    pass  # physical units, no transform
                elif cv is not None:
                    cc = cv.reindex(x_).to_numpy(dtype=float)
                    mask = ~(cc >= args.min_ceiling)
                    if args.normalize:
                        y = np.clip(y / cc, None, 1.2)
                    y = np.where(mask, np.nan, y)
                series.append((lab, x_, y))
            if args.diff:
                ref_lab, ref_x, ref_y = series[0]
                series = [
                    (f"{ref_lab} - {lab}", x_, ref_y - y) for lab, x_, y in series[1:]
                ]
            if args.smooth and args.smooth > 1:
                series = [
                    (
                        lab,
                        x_,
                        pd.Series(y)
                        .rolling(args.smooth, center=True, min_periods=1)
                        .median()
                        .to_numpy(),
                    )
                    for lab, x_, y in series
                ]
            for i, (lab, x_, y) in enumerate(series):
                ax.plot(
                    x_, y, "-", color=f"C{i}", lw=1.6, label=lab if t == 0 else None
                )
            if args.diff:
                ax.axhline(0.0, color="gray", lw=1.0)
            if cv is not None and not args.normalize:
                yc = np.where(
                    cv.to_numpy(dtype=float) >= args.min_ceiling,
                    cv.to_numpy(dtype=float),
                    np.nan,
                )
                ax.plot(
                    cv.index,
                    yc,
                    color="gray",
                    ls="--",
                    lw=1.2,
                    label="ceiling" if t == 0 else None,
                )
            if (args.rmse or args.mae) and not args.diff:
                ax.plot(
                    fv.index,
                    fv.to_numpy(dtype=float),
                    color="gray",
                    ls="--",
                    lw=1.4,
                    label=(
                        ("median label sigma" if args.mae else "label noise floor")
                        if t == 0
                        else None
                    ),
                )
                ax.set_ylim(bottom=0)
            if args.rmse_ratio and not args.diff:
                ax.axhline(1.0, color="gray", ls="--", lw=1.2)
                ax.set_ylim(bottom=0)
            if args.diff:
                pass  # autoscale around zero
            elif args.normalize:
                ax.axhline(1.0, color="gray", ls="--", lw=1.0)
                ax.set_ylim(0, 1.15)
            else:
                ax.set_ylim(bottom=min(-0.05, *(d[col].min() for _, d in frames)))
            ax.set_title(name)
            ax.set_xlabel("Chord index")
            if t == 0:
                ax.set_ylabel(
                    "Median |error|"
                    if args.mae
                    else (
                        "RMSE"
                        if args.rmse
                        else (
                            "RMSE / noise floor"
                            if args.rmse_ratio
                            else (
                                "$\\Delta R^2$"
                                if args.diff
                                else (
                                    "Fraction of achievable $R^2$"
                                    if args.normalize
                                    else "$R^2$"
                                )
                            )
                        )
                    )
                )
        fig.legend(
            loc="upper center",
            ncol=len(frames) + (ceil is not None),
            bbox_to_anchor=(0.5, 1.12),
        )
        return fig

    _render(build, args.out, "per_chord_r2")


# ---------------------------------------------------------------------------
# Figures from prediction dumps (eval_checkpoint.py --dump-preds): flat arrays,
# chord-major over all discharges concatenated in time, reshaped to (C, T, 2);
# canonical units T_i eV, v_tor km/s (displayed: keV, km/s via style.TARGET_SCALE)
# ---------------------------------------------------------------------------
TARGETS = ("ti", "vtor")
CLASS_NAMES = {0: "consistent", 1: "run edge", 2: "spike"}
CLASS_COLORS = {0: "0.6", 1: "#ff7f0e", 2: "#9467bd"}  # not model colours
CLASS_MARKERS = {1: "s", 2: "D"}


def _chord_x(path, C):
    """R_tan (m) per chord from a chord,x CSV; NaN for chords not listed."""
    x = np.full(C, np.nan)
    d = pd.read_csv(path)
    for c, v in zip(d["chord"].astype(int), d["x"].astype(float)):
        if 0 <= c < C:
            x[c] = v
    return x


def _r2(p, y):
    """Coefficient of determination over the finite pairs."""
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 3:
        return np.nan
    y, p = y[ok].astype(np.float64), p[ok].astype(np.float64)
    return float(1.0 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _r2_per_chord(d, t, consistent=False):
    """R2 per chord of target t of a (C, T, 2) prediction dump; with consistent=True
    only the consistent labels (cer_transfer.analysis.coverage.label_classes)."""
    y = d.y[..., t]
    if consistent:
        from cer_transfer.analysis.coverage import label_classes

        y = np.where(label_classes(y) == 0, y, np.nan)
    return np.array([_r2(d.pred[c, :, t], y[c]) for c in range(d.C)])


def cmd_zeroshot(args):
    """Zero-shot transfer: predicted vs target (2D histograms) and R2 per chord.

    Consistent labels only (cer_transfer.analysis.coverage.label_classes). a, b: DIII-D
    model on the NSTX test set (--zs), T_i and v_tor, 2D histogram (log colour), R2 and
    median error inset; c, d: R2 per chord vs R_tan, DIII-D model (zero-shot) and
    DIII-D+NSTX model (fine-tuned).
    """
    from matplotlib.colors import LogNorm

    from cer_transfer.figures.common import load_predictions

    zs = load_predictions(args.zs)
    ft = load_predictions(args.ft)
    x = _chord_x(args.chords, zs.C)
    from cer_transfer.analysis.coverage import label_classes

    classes, stats, r2c = [], [], []
    for t, name in enumerate(TARGETS):
        s = style.TARGET_SCALE[name]
        cls = label_classes(zs.y[..., t])
        classes.append(cls)
        y, p = zs.y[..., t] * s, zs.pred[..., t] * s
        ok = np.isfinite(y) & np.isfinite(p)
        cons = ok & (cls == 0)
        r2, r2a = _r2(p[cons], y[cons]), _r2(p[ok], y[ok])
        medae = float(np.median(np.abs(p - y)[cons]))
        stats.append((r2, r2a, medae))
        unit = "keV" if name == "ti" else "km/s"
        print(f"zero-shot {name}: R2 consistent labels {r2:.3f}, all labels {r2a:.3f}; median "
              f"|error| consistent {medae:.3g} {unit}, all {np.median(np.abs(p - y)[ok]):.3g} "
              f"(n={ok.sum()}, consistent {100 * cons.sum() / ok.sum():.1f}%)")
        rz, rf = _r2_per_chord(zs, t, True), _r2_per_chord(ft, t, True)
        rza, rfa = _r2_per_chord(zs, t), _r2_per_chord(ft, t)
        r2c.append((rz, rf, rza, rfa))
        print(f"  R2 per chord (zero-shot / fine-tuned) median over chords: consistent labels "
              f"{np.nanmedian(rz):.3f} / {np.nanmedian(rf):.3f}, all labels "
              f"{np.nanmedian(rza):.3f} / {np.nanmedian(rfa):.3f}; fine-tuned R2 "
              f"{_r2(ft.pred[..., t].ravel(), ft.y[..., t].ravel()):.3f}")
    order = np.argsort(x)
    order = order[np.isfinite(x[order])]
    # one R2 range for both targets (comparable panels c, d)
    vals = np.concatenate([np.r_[rr[0][order], rr[1][order]] for rr in r2c])
    bottom = max(np.nanmin(vals) - 0.05, args.r2_min)

    def build(ctx):
        """Draw the figure for the current plotting context."""
        size = (PAPER_W, 5.4) if ctx == "paper" else (12, 10)
        fig, axes = plt.subplots(2, 2, figsize=size, layout="constrained",
                                 gridspec_kw={"height_ratios": [1.25, 1]})
        for t, name in enumerate(TARGETS):
            ax = axes[0, t]
            small = "x-small" if ctx == "paper" else "small"
            *_, m = _draw_flaghist(ax, zs.y[..., t], zs.pred[..., t], classes[t],
                                   style.TARGET_SCALE[name], ctx == "paper", small,
                                   range_from_consistent=True, show_flagged=False,
                                   cmap="viridis")
            ax.grid(False)
            lbl = style.TARGET_LABEL[name]
            ax.set_xlabel("conventional fit " + lbl)
            ax.set_ylabel("predicted " + lbl)
            unit = "keV" if name == "ti" else "km/s"
            r2, r2a, medae = stats[t]
            ax.text(0.04, 0.96, f"$R^2$ = {r2:.2f}\n"
                    f"median |error| = {medae:.2g} {unit}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=small, zorder=6,
                    bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.9))
            fig.colorbar(m, ax=ax, label="labels per bin", fraction=0.046, pad=0.03)
            ax = axes[1, t]
            rz, rf, rza, rfa = r2c[t]
            for r, role in ((rz, "zero_shot"), (rf, "finetuned")):
                # chords below the lower limit: drawn on it as down-triangles
                rr = r[order]
                cl = rr < bottom
                ax.plot(x[order], np.where(cl, bottom, rr), "-", lw=1.4,
                        color=style.COLORS[role],
                        label=args.ft_label if role == "finetuned" else style.NAMES[role],
                        marker="o", ms=2 if ctx == "paper" else 3.5, clip_on=False)
                if cl.any():
                    ax.plot(x[order][cl], np.full(cl.sum(), bottom), "v", ms=5,
                            mfc="w", mec=style.COLORS[role], clip_on=False, zorder=5)
            ax.axhline(0.0, color="0.5", lw=0.8)
            ax.set_ylim(bottom, 1.0)
            ax.set_xlabel(style.RTAN_LABEL)
            ax.set_ylabel(f"$R^2$ per chord, {style.TARGET_SYMBOL[name]}")
        h, l_ = axes[1, 0].get_legend_handles_labels()
        fig.legend(h, l_, loc="outside lower center", ncol=2, frameon=False,
                   columnspacing=1.0,
                   fontsize="small" if ctx == "paper" else "x-small")
        style.panel_letters(axes, x=-0.12, y=1.05)
        return fig

    _render(build, args.out, args.out_name)


def _draw_flaghist(ax, y, p, cls, scale, paper, small, range_from_consistent=False,
                   show_flagged=True, cmap=None):
    """Prediction vs conventional fit: consistent labels as a grey 2D histogram (log
    colour), flagged labels (run edge, spike) as orange points below it.

    The axis range covers 0.05-99.95 % of all values, or with
    range_from_consistent of the consistent fits only (the share outside is printed)."""
    from matplotlib.colors import ListedColormap, LogNorm

    ok = np.isfinite(y) & np.isfinite(p) & (cls >= 0)
    yy, pp, fl = y[ok] * scale, p[ok] * scale, cls[ok] >= 1
    lo, hi = np.percentile(yy[~fl] if range_from_consistent else np.r_[yy, pp], [0.05, 99.95])
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    if lo > -pad and lo < 0.1 * hi:  # non-negative quantity (T_i): start at 0
        lo = min(lo, 0.0)
    out = (yy < lo) | (yy > hi) | (pp < lo) | (pp > hi)
    print(f"plot range {lo:.3g}..{hi:.3g}: outside {100 * out[~fl].mean():.2f}% of the consistent "
          f"and {100 * out[fl].mean():.1f}% of the flagged labels")
    bins = np.linspace(lo, hi, 120)
    # flagged points below, the consistent ridge on top (empty bins transparent)
    if show_flagged:
        ax.scatter(yy[fl], pp[fl], s=0.5 if paper else 2, color=CLASS_COLORS[1], alpha=0.35,
                   lw=0, rasterized=True, zorder=2)
    if cmap is None:  # grey, so that the orange flagged points stand out
        cmap = ListedColormap(plt.get_cmap("Greys")(np.linspace(0.3, 1.0, 256)))
    *_, mappable = ax.hist2d(yy[~fl], pp[~fl], bins=[bins, bins], cmap=cmap, norm=LogNorm(),
                             rasterized=True, zorder=3, cmin=1)
    ax.plot([lo, hi], [lo, hi], "--", color="k", lw=0.8, alpha=0.8, zorder=4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    e2 = (pp - yy) ** 2
    print(f"flagged {100 * fl.mean():.1f}% of labels, {100 * e2[fl].sum() / e2.sum():.1f}% "
          "of the squared error")
    return fl.mean(), e2[fl].sum() / e2.sum(), mappable


def cmd_flaghist(args):
    """Predicted vs fitted values of one target, consistent labels as a grey 2D
    histogram and flagged labels as orange points (SM version of panel a of
    error_decomposition)."""
    from cer_transfer.analysis.coverage import label_classes
    from cer_transfer.figures.common import load_predictions

    d = load_predictions(args.preds)
    t = TARGETS.index(args.target)
    cls = label_classes(d.y[..., t])
    sym = style.TARGET_SYMBOL[args.target]
    unit = style.TARGET_LABEL[args.target].split("(")[-1].rstrip(")")

    def build(ctx):
        """Draw the figure for the current plotting context."""
        paper = ctx == "paper"
        small = "x-small" if paper else "small"
        fig, ax = plt.subplots(figsize=(0.5 * PAPER_W, 0.5 * PAPER_W) if paper else (7, 7),
                               layout="constrained")
        _draw_flaghist(ax, d.y[..., t], d.pred[..., t], cls, style.TARGET_SCALE[args.target],
                       paper, small)
        ax.set_xlabel(f"fitted {sym} ({unit})")
        ax.set_ylabel(f"predicted {sym} ({unit})")
        ax.legend(handles=[matplotlib.patches.Patch(color="0.45", label="consistent"),
                           Line2D([], [], ls="", marker="o", ms=4, color=CLASS_COLORS[1],
                                  label="flagged (run edge, spike)")],
                  fontsize=small, loc="upper left", frameon=True, framealpha=0.9,
                  handlelength=1.0)
        return fig

    _render(build, args.out, args.out_name or f"pred_vs_fit_flagged_{args.target}")


def _abserr(d, x, core_max, edge_min):
    """Median |pred - y| and median sigma_fit per chord and target (canonical units).

    Prints core/edge summaries; returns {target: (median |error|, median sigma_fit)}.
    """
    out = {}
    for t, name in enumerate(TARGETS):
        y, p, sg = (a[..., t].astype(np.float64) for a in (d.y, d.pred, d.sigma))
        ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(sg)
        e = np.where(ok, np.abs(p - y), np.nan)
        s_ = np.where(ok, sg, np.nan)
        me = np.array([np.nanmedian(e[c]) if ok[c].any() else np.nan for c in range(d.C)])
        ms = np.array([np.nanmedian(s_[c]) if ok[c].any() else np.nan for c in range(d.C)])
        out[name] = (me, ms)
        unit = "eV" if name == "ti" else "km/s"
        for reg, sel in ((f"core (R_tan < {core_max:g} m)", x < core_max),
                         (f"edge (R_tan > {edge_min:g} m)", x > edge_min)):
            print(f"{name} {reg}: median |error| {np.median(e[sel][ok[sel]]):.3g} {unit}, "
                  f"median sigma_fit {np.median(s_[sel][ok[sel]]):.3g} {unit}")
    return out


def cmd_errclass(args):
    """Where the error lives (fine-tuned model on the validation set).

    a: predicted vs fitted v_tor, consistent labels as a grey 2D histogram, flagged
    labels (run edge + spike, cer_transfer.analysis.coverage.label_classes, pooled
    per chord as in the Methods) as orange points. b: a v_tor trace with flagged
    labels marked. c, d: median |error| per chord vs R_tan with the median
    sigma_fit, T_i (eV) and v_tor (km/s).
    """
    from types import SimpleNamespace

    from cer_transfer.analysis.coverage import label_classes
    from cer_transfer.figures.common import frame_times, load_predictions
    from cer_transfer.figures.timeaxis import file_times

    d = load_predictions(args.preds)
    t = 1  # v_tor
    # label classes pooled over the dump, per chord (the Methods definition)
    cls_all = label_classes(d.y[..., t])
    if args.trace:  # a separate single-discharge dump
        tr = load_predictions(args.trace)
        ctr = label_classes(tr.y[..., t])
    else:  # cut the trace discharge out of --preds, keeping the pooled classes
        L = np.load(args.file_lengths)
        files, Ts = list(L["files"]), L["T"]
        if Ts.sum() != d.T:
            raise SystemExit(f"--file-lengths cover {Ts.sum()} frames, the dump has {d.T}")
        fi = next((i for i, f in enumerate(files) if Path(f).name == Path(args.trace_shot).name),
                  None)
        if fi is None:
            raise SystemExit(f"{args.trace_shot} not in --file-lengths")
        s0, T = int(Ts[:fi].sum()), int(Ts[fi])
        tr = SimpleNamespace(C=d.C, T=T, **{k: getattr(d, k)[:, s0:s0 + T]
                                            for k in ("y", "pred", "sigma", "pred_sigma")})
        ctr = cls_all[:, s0:s0 + T]
    ytr = tr.y[..., t]
    tt = (file_times(args.trace_shot, tr.T, 200.0, args.t_offset) if args.trace_shot
          else frame_times(tr.T, t_offset=args.t_offset))
    nW = int(round(args.window * 200.0))
    pick = None
    chords = [args.chord] if args.chord is not None else range(tr.C)
    for c in chords:
        starts = ([int(np.argmin(np.abs(tt - args.t0)))] if args.t0 is not None
                  else np.where(ctr[c] >= 0)[0])
        for i0 in starts:
            w = ctr[c, i0:i0 + nW]
            if (w == 1).any() and (w == 2).any() and (w == 0).sum() >= 4:
                pick = (c, i0)
                break
        if pick:
            break
    if pick is None:
        raise SystemExit("no window with a spike and a run-edge label; set --chord/--t0")
    c, i0 = pick
    sl = slice(i0, min(i0 + nW, tr.T))
    x = _chord_x(args.chords, d.C)
    ae = _abserr(d, x, args.core_max, args.edge_min)
    xtr = _chord_x(args.trace_chords, tr.C) if args.trace_chords else None
    w = ctr[c, sl]
    zc = (np.abs(tr.pred[c, sl, t] - ytr[c, sl]) / tr.sigma[c, sl, t])[w == 0]
    print(f"trace: {args.trace or args.trace_shot}, chord {c}, window {tt[sl][0]:.3f}-{tt[sl][-1]:.3f} s "
          f"(spikes {int((w == 2).sum())}, run edges {int((w == 1).sum())}; unflagged "
          f"labels: max |error|/sigma_fit {np.nanmax(zc):.2f}; model band max "
          f"{np.nanmax(tr.pred_sigma[c, sl, t]) * args.k:.1f} km/s)")

    def build(ctx):
        """Draw the figure for the current plotting context."""
        paper = ctx == "paper"
        small = "x-small" if paper else "small"
        size = (PAPER_W, 3.0) if paper else (20, 7.0)
        fig = plt.figure(figsize=size, layout="constrained")
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1])
        a = fig.add_subplot(gs[0])
        b = fig.add_subplot(gs[1])
        gc = gs[2].subgridspec(2, 1)
        cx = fig.add_subplot(gc[0])
        dx = fig.add_subplot(gc[1], sharex=cx)
        # a: predicted vs fitted v_tor, flagged labels on top
        sc = style.TARGET_SCALE["vtor"]
        _draw_flaghist(a, d.y[..., t], d.pred[..., t], cls_all, sc, paper, small)
        a.set_xlabel(r"fitted $v_\mathrm{tor}$ (km/s)")
        a.set_ylabel(r"predicted $v_\mathrm{tor}$ (km/s)")
        a.legend(handles=[matplotlib.patches.Patch(color="0.45", label="consistent"),
                          Line2D([], [], ls="", marker="o", ms=4, color=CLASS_COLORS[1],
                                 label="flagged (run edge, spike)")],
                 fontsize=small, loc="upper left", frameon=True, framealpha=0.9,
                 handlelength=1.0)
        # b: trace with flagged labels (fit markers on top of the flag symbols)
        s = style.TARGET_SCALE["vtor"]
        tw = tt[sl]
        mu, sg = tr.pred[c, sl, t] * s, tr.pred_sigma[c, sl, t] * s * args.k
        ok = np.isfinite(mu)
        style.model_line(b, tw[ok], mu[ok], sg[ok], role="finetuned", marker="", lw=1.4)
        yy, ee, cc = ytr[c, sl] * s, tr.sigma[c, sl, t] * s, ctr[c, sl]
        m = cc >= 0
        style.fit_points(b, tw[m], yy[m], ee[m])
        for k in (1, 2):
            mk = cc == k
            b.plot(tw[mk], yy[mk], CLASS_MARKERS[k], ms=13 if not paper else 7,
                   mfc="none", mec=CLASS_COLORS[k], mew=1.5, zorder=4,
                   label=f"flagged: {CLASS_NAMES[k]}")
        b.set_xlabel(style.TIME_LABEL)
        b.set_ylabel(style.VTOR_LABEL)
        b.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        rt = (f"$R_\\mathrm{{tan}}$ = {xtr[c]:.2f} m" if xtr is not None and np.isfinite(xtr[c])
              else f"chord {c}")
        shot = re.search(r"(\d{6})", str(args.trace_shot or args.trace))
        b.text(0.97, 0.03, (f"discharge {shot.group(1)}\n" if shot else "") + rt,
               transform=b.transAxes, ha="right", va="bottom", fontsize=small)
        style.ordered_legend(b, fontsize=small, loc="upper center",
                             bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False,
                             columnspacing=0.8, handlelength=1.5)
        # c, d: median |error| and median sigma_fit vs R_tan
        order = np.argsort(x)
        order = order[np.isfinite(x[order])]
        col = style.COLORS["finetuned"]
        for ax, name, unit in ((cx, "ti", "eV"), (dx, "vtor", "km/s")):
            me, ms = ae[name]
            ax.plot(x[order], me[order], "o-", color=col, ms=2.5 if paper else 4, lw=1.2,
                    label="median $|$error$|$")
            ax.plot(x[order], ms[order], "-", color="0.55", lw=1.4, label=r"median $\sigma_\mathrm{fit}$")
            ax.axvspan(args.core_max, args.edge_min, color="0.92", lw=0, zorder=0)
            ax.set_yscale("log")  # the outermost chords reach 10x the core values
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_ylabel(f"{style.TARGET_SYMBOL[name]} ({unit})")
        tf = cx.get_xaxis_transform()
        kw = dict(transform=tf, va="top", fontsize=small, color="0.35")
        cx.text(args.core_max - 0.01, 0.97, "core", ha="right", **kw)
        cx.text(args.edge_min + 0.01, 0.97, "edge", ha="left", **kw)
        cx.annotate("not assigned", (0.5 * (args.core_max + args.edge_min), 1.0), xycoords=tf,
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize="xx-small" if paper else "x-small", color="0.45")
        plt.setp(cx.get_xticklabels(), visible=False)
        dx.set_xlabel(style.RTAN_LABEL)
        dx.legend(fontsize=small, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=1,
                  frameon=False, handlelength=1.5)
        style.panel_letters([a, b, cx, dx], x=-0.14)
        return fig

    _render(build, args.out, args.out_name)


def _relerr(d, x, floor, core_max):
    """Median relative error |pred - y| / |y| per chord, per target (prints a summary).

    Labels with |y| below floor[t] (canonical units: T_i eV, v_tor km/s) are
    excluded: v_tor crosses zero, where the relative error diverges.
    """
    rel = {}
    for t, name in enumerate(TARGETS):
        y, p = d.y[..., t].astype(np.float64), d.pred[..., t].astype(np.float64)
        ok = np.isfinite(y) & np.isfinite(p)
        keep = ok & (np.abs(y) >= floor[t])
        r = np.where(keep, np.abs(p - y) / np.where(keep, np.abs(y), 1.0), np.nan)
        per = np.array([np.nanmedian(r[c]) if keep[c].any() else np.nan
                        for c in range(d.C)])
        rel[name] = per
        unit = "eV" if name == "ti" else "km/s"
        print(f"{name}: floor |y| >= {floor[t]:g} {unit} excludes "
              f"{100 * (1 - keep.sum() / ok.sum()):.1f}% of the labels")
        for reg, sel in (("core", x < core_max), ("edge", x >= core_max),
                         ("all", np.isfinite(x))):
            rr = r[sel][keep[sel]]
            ratio = np.median(np.abs(p[sel] - y[sel])[ok[sel]]) / np.median(
                np.abs(y[sel])[ok[sel]])
            print(f"  {reg:4s} ({int(sel.sum())} chords): median relative error "
                  f"{100 * np.median(rr):.1f}% (pooled labels); median over chords "
                  f"{100 * np.nanmedian(per[sel]):.1f}%; median|err|/median|y| "
                  f"{100 * ratio:.1f}% (global_metrics style)")
    return rel


def _draw_relerr(ax, x, rel, core_max, ms=4, lw=1.4):
    """Median relative error per chord vs R_tan, both targets, core/edge marked."""
    order = np.argsort(x)
    order = order[np.isfinite(x[order])]
    c = style.COLORS["finetuned"]
    for name, fmt in (("ti", "o-"), ("vtor", "s--")):
        ax.plot(x[order], 100 * rel[name][order], fmt, color=c, ms=ms, lw=lw,
                mfc=c if name == "ti" else "none", label=style.TARGET_SYMBOL[name])
    ax.axvline(core_max, color="0.5", lw=1.0, ls=":")
    ax.text(core_max, 0.97, " edge", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize="small", color="0.4")
    ax.text(core_max, 0.97, "core ", transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize="small", color="0.4")
    ax.set_yscale("log")  # a few % in the core, more at the outermost chords
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(
        [1, 2, 5, 10, 20, 50, 100, 200, 500]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(style.RTAN_LABEL)
    ax.set_ylabel("median relative error (%)")


def cmd_relerr(args):
    """Median relative error |pred - y| / |y| per chord vs R_tan (fine-tuned model)."""
    from cer_transfer.figures.common import load_predictions

    d = load_predictions(args.preds)
    x = _chord_x(args.chords, d.C)
    rel = _relerr(d, x, args.floor, args.core_max)

    def build(ctx):
        """Draw the figure for the current plotting context."""
        size = (PAPER_W, 2.8) if ctx == "paper" else (9, 5.5)
        fig, ax = plt.subplots(figsize=size)
        _draw_relerr(ax, x, rel, args.core_max)
        ax.legend(title=style.NAMES["finetuned"], fontsize="small",
                  title_fontsize="small", loc="upper left")
        return fig

    _render(build, args.out, args.out_name)


def cmd_reliability(args):
    """Reliability diagram: empirical vs nominal coverage of +-z sigma_hat.

    sigma_hat = sqrt((k * sigma_model)^2 + sigma_label^2) as in
    cer_transfer.analysis.coverage; consistent labels only.
    """
    from scipy.special import erf

    from cer_transfer.analysis.coverage import consistent_mask
    from cer_transfer.figures.common import load_predictions

    d = load_predictions(args.preds)
    x = _chord_x(args.chords, d.C)
    zg = np.linspace(0, 3, 121)
    nominal = erf(zg / np.sqrt(2))
    regions = (("all", np.ones(d.C, bool)), ("core", x < args.core_max),
               ("edge", x > args.edge_min))
    curves = {}
    for t, name in enumerate(TARGETS):
        y, mu = d.y[..., t].astype(np.float64), d.pred[..., t].astype(np.float64)
        s = np.sqrt((args.k[t] * d.pred_sigma[..., t]) ** 2
                    + np.nan_to_num(d.sigma[..., t]) ** 2)
        cons = consistent_mask(y) & np.isfinite(mu)
        z = np.abs(y - mu) / s
        for reg, sel in regions:
            zz = np.sort(z[sel][cons[sel]])
            emp = np.searchsorted(zz, zg, side="left") / zz.size  # fraction |z| < zg
            c1, c2 = 100 * np.mean(zz < 1), 100 * np.mean(zz < 2)
            curves[name, reg] = (emp, c1, c2)
            print(f"{name} {reg:4s} (k={args.k[t]:g}, n={zz.size}): coverage "
                  f"1sigma {c1:.1f}%  2sigma {c2:.1f}%  (Gaussian 68.3 / 95.4)")

    def build(ctx):
        """Draw the figure for the current plotting context."""
        size = (PAPER_W, 3.3) if ctx == "paper" else (12, 6)
        fig, axes = plt.subplots(1, 2, figsize=size, sharey=True)
        c = style.COLORS["finetuned"]
        ls = {"all": "-", "core": "--", "edge": ":"}
        for t, (ax, name) in enumerate(zip(axes, TARGETS)):
            ax.plot([0, 1], [0, 1], color="0.5", lw=1.0)
            lines = []
            for reg, _ in regions:
                emp, c1, c2 = curves[name, reg]
                ax.plot(nominal, emp, ls[reg], color=c, lw=1.8 if reg == "all" else 1.4,
                        label=reg)
                lines.append(f"{reg}: {c1:.0f} % / {c2:.0f} %")
            for z in (1, 2):
                ax.axvline(erf(z / np.sqrt(2)), color="0.7", lw=0.8, ls="-.")
            ax.text(0.97, 0.04,
                    rf"{style.TARGET_SYMBOL[name]}, $\pm1\sigma$ / $\pm2\sigma$" + "\n"
                    + "\n".join(lines),
                    transform=ax.transAxes, ha="right", va="bottom", fontsize="x-small",
                    bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.9))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xlabel("nominal coverage")
            if t == 0:
                ax.set_ylabel("empirical coverage")
                ax.legend(title=style.NAMES["finetuned"], loc="upper left",
                          fontsize="x-small", title_fontsize="x-small")
        style.panel_letters(axes)
        return fig

    _render(build, args.out, args.out_name)


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hist")
    h.add_argument("--machine", required=True)
    h.add_argument("--list", type=Path, required=True)
    h.add_argument("--limit", type=int, default=0)
    h.add_argument("--first-chords", type=int, default=0)
    h.add_argument("--out", type=Path, default=Path("figs"))
    h.set_defaults(fn=cmd_hist)

    r = sub.add_parser("recon")
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--list", type=Path, required=True)
    r.add_argument("--shot-index", type=int, default=0)
    r.add_argument("--chord", type=int, default=25)
    r.add_argument("--rate-hz", type=float, default=200.0)
    r.add_argument("--out", type=Path, default=Path("figs"))
    r.set_defaults(fn=cmd_recon)

    c = sub.add_parser("curves")
    c.add_argument("--csv", nargs="+", required=True)
    c.add_argument(
        "--ft-csv",
        nargs="*",
        default=[],
        help="metrics CSVs of fine-tuned (DIII-D+NSTX) runs, each 'label:path' or "
        "path; dashed blue, early-stopping epoch marked by a star",
    )
    c.add_argument(
        "--ft-labels", nargs="*", default=None, help="legend text per --ft-csv path"
    )
    c.add_argument("--metric", default="val_score")
    c.add_argument("--x", choices=["epoch", "hours"], default="epoch")
    c.add_argument(
        "--labels", nargs="*", default=None, help="legend text per CSV (same order)"
    )
    c.add_argument(
        "--mark-best", action="store_true", help="mark the best epoch of each curve"
    )
    c.add_argument(
        "--drop-outliers",
        action="store_true",
        help="interpolate linearly over epochs whose score collapses (resume dips)",
    )
    c.add_argument(
        "--outlier-drop",
        type=float,
        default=0.03,
        help="drop below the 9-epoch rolling median that marks a collapsed epoch "
        "(default 0.03)",
    )
    c.add_argument("--ylim", type=float, nargs=2, default=None)
    c.add_argument(
        "--out-name", default=None, help="file stem (default learning_curves_<metric>)"
    )
    c.add_argument("--out", type=Path, default=Path("figs"))
    c.set_defaults(fn=cmd_curves)

    s = sub.add_parser("scaling", help="best val_score vs training shots")
    s.add_argument(
        "--agnostic-runs",
        nargs="*",
        default=None,
        help="N:metrics.csv of the machine-agnostic model fine-tuned on N discharges",
    )
    s.add_argument(
        "--annotate-match",
        action="store_true",
        help="with --annotate: also mark the from-scratch full-data score (dotted "
        "line) "
        "and the smallest fine-tuned subset that reaches it",
    )
    s.add_argument(
        "--agnostic-scratch",
        nargs="*",
        default=None,
        help="N:metrics.csv of the machine-agnostic model trained from scratch on N "
        "discharges",
    )
    s.add_argument(
        "--agnostic-label",
        default="machine-agnostic, trained from scratch",
        help="legend label of the --agnostic-runs curve",
    )
    s.add_argument(
        "--zero-shot-csv",
        type=Path,
        default=None,
        help="affine_fit CSV of the machine-agnostic model on the NSTX validation "
        "set without NSTX training (score = mean of the raw per-target R2)",
    )
    s.add_argument(
        "--zero-shot-column",
        default="r2_raw",
        help="column of --zero-shot-csv to average: r2_raw, or r2_protocol from "
        "cer_transfer.analysis.physics_correction --csv (label-free metadata "
        "correction)",
    )
    s.add_argument(
        "--zero-shot-label",
        default=None,
        help="legend text for the zero-shot line (default: 'DIII-D only')",
    )
    s.add_argument(
        "--runs-label",
        default=None,
        help="legend text of --runs (default 'DIII-D+NSTX (fine-tuned)')",
    )
    s.add_argument(
        "--scratch-label",
        default=None,
        help="legend text of --scratch (default 'NSTX only')",
    )
    s.add_argument(
        "--ylabel", default=None, help="y-axis label (default 'NSTX validation score')"
    )
    s.add_argument(
        "--annotate",
        action="store_true",
        help="annotate the two punchlines: %%-of-full at the "
        "smallest subset, and the from-scratch "
        "equivalence point",
    )
    s.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="n:metrics.csv pairs for the transfer arms",
    )
    s.add_argument(
        "--scratch",
        nargs="*",
        default=[],
        help="n:metrics.csv reference points (open markers)",
    )
    s.add_argument(
        "--ceiling",
        type=float,
        nargs="*",
        default=[],
        help="honest [nominal] ceiling value(s)",
    )
    s.add_argument(
        "--sizes-csv",
        type=Path,
        default=None,
        help="count_samples CSV; adds a top axis with the "
        "number of frames per point",
    )
    s.add_argument("--out", type=Path, default=Path("figs"))
    s.add_argument("--scratch-full-line", action="store_true",
                   help="dotted horizontal line at the all-data from-scratch score")
    s.add_argument("--out-name", default=None, help="default scaling_val_score")
    s.add_argument("--trace-preds", type=Path, default=None,
                   help="panel b: single-discharge dump of the fine-tuned model (e.g. 325 discharges)")
    s.add_argument("--trace-preds2", type=Path, default=None, help="panel b: second model (scratch)")
    s.add_argument("--trace-label", default="DIII-D+NSTX (325 discharges) model")
    s.add_argument("--trace-label2", default="NSTX (325 discharges) model")
    s.add_argument("--trace-shot", default=None, help="discharge file (measured time axis)")
    s.add_argument("--trace-chords", type=Path, default=None, help="chord,x CSV (R_tan per chord)")
    s.add_argument("--trace-x", type=float, default=1.10, help="R_tan of the trace chord (m)")
    s.add_argument("--trace-t0", type=float, default=0.370)
    s.add_argument("--trace-t1", type=float, default=0.445)
    s.add_argument("--trace-t-offset", type=float, default=-0.235)
    s.add_argument("--trace-ylim", type=float, nargs=2, default=None)
    s.set_defaults(fn=cmd_scaling)
    s = sub.add_parser("scaling_targets", help="R2 of T_i and v_tor vs NSTX discharges + core-chord traces")
    s.add_argument("--runs", nargs="+", required=True, help="n:metrics.csv of the fine-tuned runs")
    s.add_argument("--scratch", nargs="+", required=True, help="n:metrics.csv of the from-scratch runs")
    s.add_argument("--runs-label", default="DIII-D+NSTX model")
    s.add_argument("--scratch-label", default="NSTX model")
    s.add_argument("--zero-shot-csv", default=None)
    s.add_argument("--zero-shot-column", default="r2_protocol")
    s.add_argument("--zero-shot-label", default="DIII-D model")
    s.add_argument("--trace-preds", type=Path, required=True)
    s.add_argument("--trace-preds2", type=Path, required=True)
    s.add_argument("--trace-label", default="DIII-D+NSTX (325 discharges) model")
    s.add_argument("--trace-label2", default="NSTX (325 discharges) model")
    s.add_argument("--trace-shot", default=None)
    s.add_argument("--trace-chords", type=Path, required=True)
    s.add_argument("--trace-x", type=float, default=1.10)
    s.add_argument("--trace-t0", type=float, default=0.370)
    s.add_argument("--trace-t1", type=float, default=0.445)
    s.add_argument("--trace-t-offset", type=float, default=-0.235)
    s.add_argument("--out", type=Path, default=Path("figs"))
    s.add_argument("--out-name", default="scaling_and_transfer_event")
    s.set_defaults(fn=cmd_scaling_targets)

    z = sub.add_parser("sizes", help="frames vs shots from count_samples CSV")
    z.add_argument("--csv", type=Path, required=True)
    z.add_argument(
        "--match", default="nstx_train", help="substring filter, used with --shots"
    )
    z.add_argument(
        "--shots",
        type=int,
        nargs="*",
        default=[],
        help="explicit shot counts to plot (overrides the "
        "default r-subset selection)",
    )
    z.add_argument("--out", type=Path, default=Path("figs"))
    z.set_defaults(fn=cmd_sizes)

    ch = sub.add_parser("chords", help="per-chord R2 from eval CSVs")
    ch.add_argument("--evals", nargs="+", required=True, help="label:eval_csv pairs")
    ch.add_argument(
        "--ceiling-csv",
        type=Path,
        default=None,
        help="per-chord ceiling CSV (ceiling_check --per-chord " "--out ...)",
    )
    ch.add_argument(
        "--normalize",
        action="store_true",
        help="plot R2 / per-chord ceiling (fraction of "
        "achievable); requires --ceiling-csv",
    )
    ch.add_argument(
        "--mae",
        action="store_true",
        help="per-chord MEDIAN absolute error in physical "
        "units, with the median label sigma as the floor "
        "curve (typical-frame precision; robust to tails)",
    )
    ch.add_argument(
        "--rmse",
        action="store_true",
        help="plot per-chord RMSE in PHYSICAL units (eV, km/s) "
        "with the label noise floor as a gray curve; the "
        "most direct reviewer-facing view",
    )
    ch.add_argument(
        "--rmse-ratio",
        action="store_true",
        help="plot per-chord RMSE / label noise floor "
        "(1.0 = at the measurement limit); needs rmse_* "
        "columns in the eval CSVs and floor_* in "
        "--ceiling-csv. Interpretable on every chord, "
        "unlike R2.",
    )
    ch.add_argument(
        "--diff",
        action="store_true",
        help="plot the FIRST eval minus each other eval "
        "(deficit curves); shared radial structure "
        "cancels",
    )
    ch.add_argument(
        "--smooth",
        type=int,
        default=0,
        metavar="W",
        help="rolling-median window over chords (odd; 0=off)",
    )
    ch.add_argument(
        "--min-ceiling",
        type=float,
        default=0.2,
        help="mask chords whose ceiling is below this "
        "(unlearnable labels; default 0.2)",
    )
    ch.add_argument("--out", type=Path, default=Path("figs"))
    ch.set_defaults(fn=cmd_chords)

    zs = sub.add_parser("zeroshot", help="zero-shot transfer: pred vs target, R2/chord")
    zs.add_argument("--zs", required=True, help="zero-shot dump, NSTX validation set")
    zs.add_argument("--ft", required=True, help="fine-tuned dump, same validation set")
    zs.add_argument("--chords", required=True, help="chord,x CSV (R_tan in m)")
    zs.add_argument("--bins", type=int, default=150)
    zs.add_argument("--r2-min", type=float, default=-1.0,
                    help="lower y-limit of the R2-per-chord panels (clipped chords "
                    "marked on the edge)")
    zs.add_argument("--out", type=Path, default=Path("figs"))
    zs.add_argument("--out-name", default="zero_shot_decomposition")
    zs.add_argument("--ft-label", default="DIII-D+NSTX (7307 discharges) model")
    zs.set_defaults(fn=cmd_zeroshot)

    nu = sub.add_parser("nstxu", help="transfer to NSTX-U: R2 on the NSTX-U test set")
    nu.add_argument("--dir", type=Path, default=Path("gallery/nstxu"))
    nu.add_argument("--zs-d3d", default="gallery/paper/zero_shot_nstxu_test.txt",
                    help="global_metrics output: DIII-D model zero-shot (NSTX readout)")
    nu.add_argument("--zs-nstx", default="gallery/paper/nstxu_test_nstx_rt_init_rfull.txt",
                    help="global_metrics output: DIII-D+NSTX model without NSTX-U data")
    nu.add_argument("--ft-d3d-tag", default="nstxu_ft_d3d_r325")
    nu.add_argument("--ft-nstx-tag", default="nstxu_ft_nstx_r325")
    nu.add_argument("--scratch-tag", default="nstxu_scratch64_r325")
    nu.add_argument("--n", type=int, default=325, help="NSTX-U training discharges")
    nu.add_argument("--out", type=Path, default=Path("figs"))
    nu.add_argument("--out-name", default="nstxu_transfer")
    nu.set_defaults(fn=cmd_nstxu)
    fh = sub.add_parser("flaghist", help="predicted vs fit with flagged labels (one target)")
    fh.add_argument("--preds", required=True, help="fine-tuned validation dump")
    fh.add_argument("--target", choices=TARGETS, default="ti")
    fh.add_argument("--out", type=Path, default=Path("figs"))
    fh.add_argument("--out-name", default=None)
    fh.set_defaults(fn=cmd_flaghist)
    ec = sub.add_parser("errclass", help="v_tor error by label class + flagged trace")
    ec.add_argument("--preds", required=True, help="fine-tuned validation dump")
    ec.add_argument("--trace", default=None,
                    help="single-discharge dump (default: cut --trace-shot out of --preds)")
    ec.add_argument("--t-offset", type=float, default=-0.235,
                    help="time of frame 0 (s); frames at 200 Hz")
    ec.add_argument("--window", type=float, default=0.15, help="trace window (s)")
    ec.add_argument("--chord", type=int, default=None, help="trace chord (default: auto)")
    ec.add_argument("--t0", type=float, default=None, help="window start (s; default auto)")
    ec.add_argument("--k", type=float, default=1.0,
                    help="calibration factor of the v_tor model sigma (band)")
    ec.add_argument("--chords", required=True, help="chord,x CSV (R_tan in m) of --preds")
    ec.add_argument("--trace-chords", default=None, help="chord,x CSV of the trace discharge")
    ec.add_argument("--trace-shot", default=None, help="discharge file of the trace (time axis)")
    ec.add_argument("--file-lengths", default=None,
                    help="npz with T = frames per file of --preds (label classes per file)")
    ec.add_argument("--core-max", type=float, default=1.2, help="core: R_tan below (m)")
    ec.add_argument("--edge-min", type=float, default=1.3, help="edge: R_tan above (m)")
    ec.add_argument("--out", type=Path, default=Path("figs"))
    ec.add_argument("--out-name", default="error_decomposition")
    ec.set_defaults(fn=cmd_errclass)

    re_ = sub.add_parser("relerr", help="median relative error per chord vs R_tan")
    re_.add_argument("--preds", required=True, help="fine-tuned validation dump")
    re_.add_argument("--chords", required=True, help="chord,x CSV (R_tan in m)")
    re_.add_argument("--core-max", type=float, default=1.2, help="core: R_tan below (m)")
    re_.add_argument("--floor", type=float, nargs=2, default=[0.0, 5.0],
                     help="exclude |y| below: T_i (eV), v_tor (km/s); default 0 5")
    re_.add_argument("--out", type=Path, default=Path("figs"))
    re_.add_argument("--out-name", default="relative_error_rtan")
    re_.set_defaults(fn=cmd_relerr)

    rl = sub.add_parser("reliability", help="reliability diagram (coverage)")
    rl.add_argument("--preds", required=True, help="fine-tuned TEST-set dump")
    rl.add_argument("--k", type=float, nargs=2, default=[1.0, 1.0],
                    help="calibration factors for T_i and v_tor")
    rl.add_argument("--chords", required=True, help="chord,x CSV (R_tan in m)")
    rl.add_argument("--core-max", type=float, default=1.2, help="core: R_tan below (m)")
    rl.add_argument("--edge-min", type=float, default=1.3, help="edge: R_tan above (m)")
    rl.add_argument("--out", type=Path, default=Path("figs"))
    rl.add_argument("--out-name", default="calibration_reliability")
    rl.set_defaults(fn=cmd_reliability)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
