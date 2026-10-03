"""Paper/analysis plots. Three subcommands (see bottom for CLI).

Figures are rendered twice: PNG in seaborn context "talk" (slides) and PDF
in context "paper" (manuscript), default font sizes in each.
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
from joblib import load

from cer_transfer.configs import data_path, get_machine

sns.set_style("whitegrid")
UNITS = {"ti": "keV", "vtor": "km/s"}
DISPLAY = {"ti": "Ti", "vtor": "vtor"}
SCALE = {"ti": 1e-3, "vtor": 1.0}  # canonical eV -> display keV
AXLBL = {"ti": "T$_i$ (keV)", "vtor": "v$_{tor}$ (km/s)"}


def _render(build_fig, out: Path, name: str):
    """Render twice: talk context -> png (slide-sized), paper context ->
    pdf (column-sized). build_fig receives the context name and must size
    the figure accordingly."""
    out.mkdir(parents=True, exist_ok=True)
    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            fig = build_fig(ctx)
            fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {out / name}.png/.pdf")


def _read_files(args, machine):
    files = [
        Path(ln.strip())
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
        Path(ln.strip())
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


def cmd_curves(args):
    """Subcommand: training curves from metrics CSVs."""
    runs = []
    for path in args.csv:
        df = pd.read_csv(path)
        if args.metric not in df.columns:
            print(f"skip {path}: no column {args.metric}")
            continue
        runs.append((Path(path).stem.replace(".metrics", ""), df))

    def build(ctx):
        """Draw the figure for the current plotting context."""
        size = (5.5, 3.4) if ctx == "paper" else (9, 5.5)
        fig, ax = plt.subplots(figsize=size)
        for i, (label, df) in enumerate(runs):
            if args.x == "hours" and "epoch_seconds" in df.columns:
                x = df["epoch_seconds"].cumsum() / 3600.0
                ax.set_xlabel("training wall-clock (h)")
            else:
                x = np.arange(len(df))
                ax.set_xlabel("Epoch")
            ax.plot(
                x, df[args.metric].astype(float), lw=1.4, color=f"C{i}", label=label
            )
        ax.set_ylabel("Validation score" if args.metric == "val_score" else args.metric)
        ax.legend()
        fig.tight_layout()
        return fig

    _render(build, args.out, f"learning_curves_{args.metric}")


def cmd_scaling(args):
    """Scaling figure: best val_score vs number of training shots."""

    def best(path):
        """Best (max) value of a metric column."""
        d = pd.read_csv(path)
        return float(d["val_score"].max())

    runs = sorted((int(s.split(":")[0]), best(s.split(":", 1)[1])) for s in args.runs)
    scr = [
        (int(s.split(":")[0]), best(s.split(":", 1)[1])) for s in (args.scratch or [])
    ]

    def build(ctx):
        """Draw the figure for the current plotting context."""
        size = (5.5, 3.4) if ctx == "paper" else (9, 5.5)
        fig, ax = plt.subplots(figsize=size)
        xs = [n for n, _ in runs]
        ys = [v for _, v in runs]
        ax.plot(xs, ys, "o-", color="C0", lw=1.8, label="fine-tuned from DIII-D model")
        for i, (n, v) in enumerate(scr):
            ax.plot(
                [n],
                [v],
                "s",
                mfc="none",
                color="C3",
                ms=8,
                label="trained from scratch" if i == 0 else None,
            )
        if args.annotate and runs and scr:
            x0, y0 = runs[0]
            xF, yF = runs[-1]
            s_full = max(v for _, v in scr)
            # annotation 1: smallest subset relative to the full-data score
            ax.annotate(
                f"{x0} shots:\n{100*y0/yF:.0f}% of full-data " "score",
                xy=(x0, y0),
                xytext=(x0 * 2.1, y0 + 0.002),
                fontsize=8,
                color="C0",
                arrowprops=dict(arrowstyle="->", color="C0", lw=1.0),
            )
            # annotation 2: equivalence point vs the full from-scratch model
            ax.axhline(s_full, color="C3", ls=":", lw=1.0, alpha=0.8)
            eq = next((n for n, v in runs if v >= s_full), None)
            if eq is not None:
                ax.annotate(
                    f"{eq} shots match\nfrom-scratch on all "
                    f"{scr[0][0] if len(scr)==1 else max(n for n,_ in scr)}",
                    xy=(eq, s_full),
                    xytext=(eq * 0.28, s_full + 0.004),
                    fontsize=8,
                    color="C3",
                    arrowprops=dict(arrowstyle="->", color="C3", lw=1.0),
                )
        if args.ceiling:
            ax.axhline(
                args.ceiling[0],
                color="gray",
                ls="--",
                lw=1.2,
                label="label-noise ceiling",
            )
            if len(args.ceiling) > 1:
                ax.axhline(args.ceiling[1], color="gray", ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.minorticks_off()
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel("Number of NSTX shots")
        ax.set_ylabel("Validation score")
        if args.sizes_csv is not None:
            szd = pd.read_csv(args.sizes_csv)
            m = szd["list"].str.match(r"nstx_train(_r\d+)?\.txt$")
            if m.any():
                szd = szd[m]
            sz = szd.drop_duplicates("shots").set_index("shots")["frames"]
            fr = [float(sz.get(n, np.nan)) for n in xs]
            top = ax.twiny()
            top.set_xscale("log")
            top.set_xlim(ax.get_xlim())
            top.set_xticks(xs)
            top.set_xticklabels(
                [f"{f/1e3:.0f}k" if f < 1e6 else f"{f/1e6:.2f}M" for f in fr],
                rotation=30,
            )
            top.minorticks_off()
            top.set_xlabel("Number of frames")
            top.grid(False)
        ax.legend()
        return fig

    _render(build, args.out, "scaling_val_score")


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
        size = (5.5, 3.4) if ctx == "paper" else (9, 5.5)
        fig, ax = plt.subplots(figsize=size)
        ax.plot(d["shots"], d["frames"], "o-", color="C0", lw=1.8)
        ax.set_xscale("log")
        ax.set_xticks(list(d["shots"]))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.minorticks_off()
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel("Number of NSTX shots")
        ax.set_ylabel("Number of frames")
        return fig

    _render(build, args.out, "frames_vs_shots")


def cmd_chords(args):
    """Per-chord R2 curves from eval_checkpoint CSVs, one line per model,
    one panel per target. Optional per-chord ceiling CSV (ceiling_check
    --per-chord --out) drawn as a gray band/line."""
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
    c.add_argument("--metric", default="val_score")
    c.add_argument("--x", choices=["epoch", "hours"], default="epoch")
    c.add_argument("--out", type=Path, default=Path("figs"))
    c.set_defaults(fn=cmd_curves)

    s = sub.add_parser("scaling", help="best val_score vs training shots")
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
    s.set_defaults(fn=cmd_scaling)

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

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
