r"""Beam-off examples: reconstruction while the neutral beam is off.

The model runs on the regular CHERS spectra (only passive emission left during
the phase) and is drawn next to the conventional fits before and after.

Everything comes from one model checkpoint and the discharge files of the
regular array; no background array, no stored prediction dumps:

* beam-off phases are found in the same file: runs of native (100 Hz)
  frames inside the plasma without any conventional fit and with clearly
  reduced line brightness, bounded by fitted frames on both sides
* the model is run on the discharge directly (``cer_transfer.inference``)

    pixi run python -u -m cer_transfer.figures.beamoff_examples \
        --checkpoint cer_ckpts/nstx_agn_ft_rfull.pt --list splits/nstx_val.txt \
        --n 2 --chord 2 --target vtor --min-off-ms 50 --margin-before 0.20 0.06 \
        [--k 0.99 1.17] [--machine nstx] [--checkpoint2 <fine-tuned> --label2 ...] \
        [--checks gallery/paper/beamoff_checks.npz] --out figs --out-name beamoff_examples

Top row: the examples (core chord, both models with +-1 sigma, fits, NBI power
below). With ``--checks`` (``analysis.beamoff_checks --dump``) a bottom row is
added: (c) median |full-context - gap-only prediction| vs. time since the beam
turned off over all beam-off phases (interquartile band), (d) |prediction - first
fit after the beam returns| vs. the same statistic in beam-on data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from joblib import load  # noqa: E402

from cer_transfer.configs import data_path  # noqa: E402
from cer_transfer.figures.common import FRAME_HZ  # noqa: E402
from cer_transfer.figures.paperfig import PAPER_W, bbox  # noqa: E402
from cer_transfer.figures.style import (  # noqa: E402
    BAND_ALPHA,
    COLORS,
    NAMES,
    TARGET_LABEL,
    TARGET_SCALE,
    TARGET_SYMBOL,
    TIME_LABEL,
    beamoff_band,
    fit_points,
    model_line,
    ordered_legend,
    role_of,
)
from cer_transfer.figures.timeaxis import file_times  # noqa: E402

TARGETS = ("ti", "vtor")


def panel_label(ax, letter, ident=None):
    """Bold panel letter at the top left, followed by an identifier (e.g. discharge)."""
    ax.text(-0.02, 1.03, letter, transform=ax.transAxes, fontweight="bold", va="bottom", ha="right")
    if ident:
        ax.text(0.0, 1.03, ident, transform=ax.transAxes, va="bottom", ha="left", fontsize="small")


def load_checks(path, tgt, fs):
    """Per-model context differences and rejoin / control errors from beamoff_checks --dump."""
    z = np.load(path, allow_pickle=False)
    out = {"n_gaps": int(z["n_gaps"]), "fs": float(z["fs"]) if "fs" in z else fs,
           "margin": int(z["margin"]), "n_after": int(z["n_after"]),
           "splits": dict(zip(z["splits"].tolist(), z["split_n_gaps"].tolist())), "models": []}
    for i, name in enumerate(z["models"].tolist()):
        out["models"].append({
            "name": name, "role": role_of(name) or ("zero_shot" if i == 0 else "finetuned"),
            "ms": z[f"m{i}_ctx_ms"], "gap": z[f"m{i}_ctx_gap"], "d": z[f"m{i}_{tgt}_ctx_d"],
            "rej": z[f"m{i}_{tgt}_rej"], "ctl": z[f"m{i}_{tgt}_ctl"],
        })
    out["models"].sort(key=lambda mm: ("fit", "zero_shot", "scratch", "finetuned").index(mm["role"]))
    return out


def context_profile(mm, min_gaps):
    """Median and interquartile range of |full - gap only| per position in the gap."""
    pos = np.unique(mm["ms"])
    keep, med, q1, q3 = [], [], [], []
    for p_ in pos:
        sel = mm["ms"] == p_
        if np.unique(mm["gap"][sel]).size < min_gaps:
            continue
        d = mm["d"][sel]
        d = d[np.isfinite(d)]
        keep.append(p_); med.append(np.median(d))
        q1.append(np.percentile(d, 25)); q3.append(np.percentile(d, 75))
    return np.array(keep), np.array(med), np.array(q1), np.array(q3)


def draw_context(ax, checks, tsc, tgt):
    """Panel: median |full context - gap only| vs. time since the beam turned off."""
    min_gaps = max(3, int(round(0.2 * checks["n_gaps"])))  # positions reached by >= 20 % of the gaps
    for mm in checks["models"]:
        p_, med, q1, q3 = context_profile(mm, min_gaps)
        c = COLORS[mm["role"]]
        ax.fill_between(p_, q1 * tsc, q3 * tsc, color=c, alpha=BAND_ALPHA, lw=0)
        ax.plot(p_, med * tsc, "-", color=c, lw=1.4, marker=".", ms=3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time since beam off (ms)")
    sym = TARGET_SYMBOL[tgt].strip("$")
    unit = TARGET_LABEL[tgt].split("(")[-1].rstrip(")")
    ax.set_ylabel(f"median $|\\Delta {sym}|$ ({unit})")


def draw_rejoin(ax, checks, tsc, tgt):
    """Panel: |prediction - first fit after the beam returns| vs. the same in beam-on data."""
    pos, data, cols, fills = [], [], [], []
    for i, mm in enumerate(checks["models"]):
        for j, key in enumerate(("rej", "ctl")):
            v = mm[key][np.isfinite(mm[key])] * tsc
            pos.append(3 * i + j); data.append(v); cols.append(COLORS[mm["role"]]); fills.append(j == 0)
    bp = ax.boxplot(data, positions=pos, widths=0.7, whis=(5, 95), showfliers=False, patch_artist=True,
                    medianprops={"color": "k", "lw": 1.2})
    for b_, c, f in zip(bp["boxes"], cols, fills):
        b_.set_facecolor(c if f else "white"); b_.set_edgecolor(c); b_.set_alpha(1.0)
        if f:
            b_.set_facecolor((*matplotlib.colors.to_rgb(c), 0.45))
    for key in ("whiskers", "caps"):
        for w_, c in zip(bp[key], np.repeat(cols, 2)):
            w_.set_color(c)
    ax.set_xticks(pos)
    ax.set_xticklabels(["after\ngap", "beam\non"] * len(checks["models"]), fontsize="small")
    ax.set_xlim(-0.7, pos[-1] + 0.7)
    ax.set_ylim(bottom=0)
    sym = TARGET_SYMBOL[tgt].strip("$")
    unit = TARGET_LABEL[tgt].split("(")[-1].rstrip(")")
    ax.set_ylabel(f"$|{sym} - {sym}^\\mathrm{{fit}}|$ ({unit})")


def load_bridge(path, tmin, splits=("val", "test")):
    """Rows of analysis.beamoff_bridge (per gap, chord, target) on held-out discharges with
    the gap starting at or after ``tmin`` s (flat-top: no ramp-up gaps). The gap start
    times come from ``<path stem>_tstart.npy`` next to the npz."""
    z = dict(np.load(path, allow_pickle=False))
    ts = z["t_start"] if "t_start" in z else np.load(Path(path).with_name(Path(path).stem + "_tstart.npy"))
    keep = np.isin(z["split"], splits) & (ts >= tmin)
    out = {q: v[keep] for q, v in z.items() if np.ndim(v) and len(v) == len(keep)}
    out["models"] = list(z["models"])
    return out


def draw_bridge(ax, br, tgt, mi, color):
    """Predicted vs actual change across the gap: x = first fit after - last fit before,
    y = model at the last beam-off frame - last fit before. Diagonal: the model knew
    where the plasma went; y = 0: holding the last fit."""
    j = TARGETS.index(tgt)
    sel = br["target"] == j
    tsc = TARGET_SCALE[tgt]
    act = (br["after"] - br["before"])[sel] * tsc
    pre = (br[f"model_{mi}"] - br["before"])[sel] * tsc
    lim = np.percentile(np.abs(np.r_[act, pre]), 99) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "-", color="0.25", lw=0.9, zorder=1)
    ax.axhline(0, color="0.25", lw=0.9, ls="--", zorder=1)
    ax.scatter(act, pre, s=4, color=color, alpha=0.35, lw=0, zorder=2, rasterized=True)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    # the two reference lines labelled where they are drawn
    ax.text(0.45 * lim, 0.45 * lim, "model knew the change", rotation=45, rotation_mode="anchor",
            ha="center", va="bottom", fontsize="x-small", color="0.25", transform_rotates_text=True)
    ax.text(-0.95 * lim, 0.02 * lim, "hold last fit", ha="left", va="bottom", fontsize="x-small", color="0.25")
    unit = TARGET_LABEL[tgt].split("(")[-1].rstrip(")")
    sym = TARGET_SYMBOL[tgt].strip("$")
    ax.set_xlabel(f"actual change $\\Delta {sym}$ ({unit})")
    ax.set_ylabel(f"predicted change ({unit})")
    err = np.median(np.abs(pre - act))
    hold = np.median(np.abs(act))
    ax.text(0.97, 0.03, f"median |error|\nmodel {err:.2g} {unit}\nhold {hold:.2g} {unit}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize="x-small")
    gaps = len({(e, a) for e, a in zip(br["entry"][sel], br["gap_a"][sel])})
    print(f"bridge {tgt} ({br['models'][mi]}): {sel.sum()} cases in {gaps} gaps; median |error| "
          f"{err:.3g}, hold {hold:.3g} {unit}; slope {np.polyfit(act, pre, 1)[0]:.2f}")
    return gaps, int(sel.sum())


def rotating_context(y, a, b, fs, context_ms, min_rot, min_fits, chord=None):
    """Return True if the plasma rotates on both sides of the phase (a, b).

    Within ``context_ms`` before a and after b there must be at least
    ``min_fits`` fitted frames whose median |v_tor| (over the chords with a
    fit, or the given chord) exceeds ``min_rot``. Uses labels only.
    """
    n = int(round(context_ms / 1000 * fs))
    for lo, hi in ((max(a - n, 0), a), (b + 1, min(b + 1 + n, y.shape[1]))):
        seg = y[:, lo:hi] if chord is None else y[chord : chord + 1, lo:hi]
        per_frame = np.nanmedian(np.abs(seg), axis=0) if seg.size else np.array([])
        ok = per_frame[np.isfinite(per_frame)]
        if ok.size < min_fits or np.median(ok) < min_rot:
            return False
    return True


def off_phases(path, min_frames, k_target=1, rotating=None):
    """Beam-off phases of one discharge file from its fits and brightness.

    Returns a list of (first, last) frames of runs of native frames without
    any fit, inside the plasma, with brightness below half of the median
    brightness of the fitted frames, bounded by fitted frames on both sides.
    ``rotating`` = (fs, context_ms, min_rot, min_fits, chord) keeps only phases
    with a rotating plasma on both sides (labels only; see rotating_context).
    """
    d = load(data_path(path), mmap_mode="r")
    x = np.asarray(d["input"], np.float32)
    end = int(d["end_index"]) if "end_index" in d else -1
    T = x.shape[1] if end <= 0 else min(end, x.shape[1])
    y = np.asarray(d["target"])[:, :T, k_target]
    fitted = np.isfinite(y).any(axis=0)
    if fitted.sum() < 10:
        return [], T
    bright = (x[:, :T].max(-1) - np.median(x[:, :T], -1)).mean(0)
    fr = np.where(fitted)[0]
    par = np.bincount(fr % 2, minlength=2).argmax()  # native 100 Hz frames
    on_level = np.median(bright[fitted])
    phases, run = [], []
    for f in range(fr[0], fr[-1] + 1):
        if f % 2 != par:
            continue
        off = (not fitted[f]) and bright[f] < 0.5 * on_level
        if off:
            run.append(f)
        else:
            if run and len(run) * 2 >= min_frames:
                phases.append((run[0], run[-1]))
            run = []
    if rotating is not None:
        fs, cms, mrot, mfit, ch = rotating
        phases = [
            ab
            for ab in phases
            if rotating_context(y, ab[0], ab[1], fs, cms, mrot, mfit, ch)
        ]
    return phases, T


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--readout",
        type=Path,
        default=None,
        help="label-free readout applied to --checkpoint's outputs "
        "(cer_transfer.analysis.label_free_calibration --save-readout)",
    )
    p.add_argument(
        "--machine", default=None, help="apply an agnostic model to this machine"
    )
    p.add_argument(
        "--list",
        type=Path,
        nargs="+",
        required=True,
        help="discharge list(s) to scan; several lists are concatenated",
    )
    p.add_argument(
        "--shots", nargs="*", default=None, help="use these discharges instead"
    )
    p.add_argument("--scan-limit", type=int, default=0)
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--chord", type=int, default=2)
    p.add_argument("--chords", type=int, nargs="+", default=None,
                   help="chord per example (order of --shots); default --chord for all")
    p.add_argument("--target", choices=("ti", "vtor"), default="vtor")
    p.add_argument("--min-off-ms", type=float, default=50.0)
    p.add_argument("--fs", type=float, default=FRAME_HZ)
    p.add_argument("--t-offset", type=float, default=-0.235)
    p.add_argument(
        "--k", type=float, nargs=2, default=(1.0, 1.0), help="calibration (Ti, vtor)"
    )
    p.add_argument("--margin", type=float, default=0.06)
    p.add_argument("--margin-before", type=float, nargs="+", default=None)
    p.add_argument("--margin-after", type=float, nargs="+", default=None)
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", default="beamoff_examples")
    p.add_argument("--label", default="model", help="legend text of the model line")
    p.add_argument(
        "--min-rotation",
        type=float,
        default=0.0,
        help="keep only phases with median |v_tor| above this (km/s) on both sides "
        "(labels only); 0 = no requirement",
    )
    p.add_argument(
        "--context-ms",
        type=float,
        default=50.0,
        help="window on each side for --min-rotation",
    )
    p.add_argument(
        "--min-fits-side",
        type=int,
        default=3,
        help="fitted frames required on each side",
    )
    p.add_argument(
        "--rotation-on-chord",
        action="store_true",
        help="evaluate --min-rotation on the plotted chord only (default: median "
        "over chords)",
    )
    p.add_argument(
        "--checkpoint2",
        type=Path,
        default=None,
        help="second model drawn in the same panels, e.g. the fine-tuned "
        "NSTX model",
    )
    p.add_argument(
        "--label2", default="second model", help="legend text of --checkpoint2"
    )
    p.add_argument(
        "--k2",
        type=float,
        nargs=2,
        default=(1.0, 1.0),
        help="calibration factors (Ti, vtor) of the second model's uncertainty",
    )
    p.add_argument(
        "--machine2",
        default=None,
        help="machine config for --checkpoint2 (default: its own)",
    )
    p.add_argument(
        "--checks",
        type=Path,
        default=None,
        help="npz of cer_transfer.analysis.beamoff_checks --dump: adds the bottom row "
        "(context ablation and rejoin error over all beam-off phases)",
    )
    p.add_argument("--bridge", type=Path, default=None,
                   help="npz of cer_transfer.analysis.beamoff_bridge: bottom row = predicted vs "
                   "actual change across the gap, (c) v_tor, (d) T_i, second model (--checkpoint2)")
    p.add_argument("--bridge-tmin", type=float, default=0.25,
                   help="flat-top: only gaps starting at or after this time (s)")
    args = p.parse_args()
    from cer_transfer.inference import load_model, predict_file

    k = 1 if args.target == "vtor" else 0
    min_frames = int(round(args.min_off_ms / 1000 * args.fs))
    entries = [
        ln.strip()
        for lst in args.list
        for ln in lst.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if args.shots:
        entries = [e for e in entries if any(s in e for s in args.shots)]
    elif args.scan_limit:
        entries = entries[: args.scan_limit]
    rot = None
    if args.min_rotation > 0:
        rot = (
            args.fs,
            args.context_ms,
            args.min_rotation,
            args.min_fits_side,
            args.chord if args.rotation_on_chord else None,
        )
    cands = []
    for e in entries:
        ph, _ = off_phases(e, min_frames, k, rotating=rot)
        if ph:
            longest = max(ph, key=lambda ab: ab[1] - ab[0])
            cands.append(((longest[1] - longest[0]) / args.fs, e, longest))
    if not cands:
        raise SystemExit("no beam-off phase found")
    cands.sort()
    if args.shots:  # in the order of --shots
        order = {s_: q for q, s_ in enumerate(args.shots)}
        cands.sort(key=lambda c_: next(q for s_, q in order.items() if s_ in c_[1]))
        chosen = cands[: args.n]
    else:  # representative: around the median phase length
        mid = len(cands) // 2
        lo = max(0, mid - args.n // 2)
        chosen = cands[lo : lo + args.n]
    print(
        f"{len(cands)} discharges with beam-off phases >= {args.min_off_ms:.0f} ms; "
        "using:"
    )
    for dur, e, (a, b) in chosen:
        te = file_times(e, b + 1, args.fs, args.t_offset)
        print(f"  {e}: {te[a]:.3f}-{te[b]:.3f} s ({1000 * dur:.0f} ms)")

    m = load_model(args.checkpoint, machine=args.machine)
    if args.readout is not None:
        from cer_transfer.analysis.label_free_calibration import (
            load_readout,
            predict_file_label_free,
        )

        ro = load_readout(args.readout)
        res = [predict_file_label_free(m, e, ro) for _, e, _ in chosen]
    else:
        res = [predict_file(m, e) for _, e, _ in chosen]
    res2 = None
    if args.checkpoint2 is not None:
        m2 = load_model(args.checkpoint2, machine=args.machine2)
        res2 = [predict_file(m2, e) for _, e, _ in chosen]
    tgt = TARGETS[k]
    tsc = TARGET_SCALE[tgt]  # T_i displayed in keV
    role1 = role_of(args.label) or "zero_shot"
    role2 = role_of(args.label2) or "finetuned"
    checks = None
    if args.checks is not None:
        checks = load_checks(args.checks, tgt, args.fs)
    bridge = load_bridge(args.bridge, args.bridge_tmin) if args.bridge is not None else None
    if bridge is not None:
        checks = True  # bottom row present (layout)
    chords = args.chords or [args.chord] * len(chosen)
    n = len(chosen)
    args.out.mkdir(parents=True, exist_ok=True)
    # NBI power from the discharge files (key 'nbi_power_MW' (T,)), time-aligned
    # with the frames: thin panel below each example
    nbi = []
    for _, e, _ in chosen:
        dd = load(data_path(e), mmap_mode="r")
        nbi.append(np.asarray(dd["nbi_power_MW"], np.float64) if "nbi_power_MW" in dd else None)
    has_nbi = any(t_ is not None for t_ in nbi)
    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx), sns.axes_style("ticks"):
            if ctx == "paper":
                size = (PAPER_W, (2.9 if has_nbi else 2.3) + (2.3 if checks else 0))
            else:
                size = (13, (6.2 if has_nbi else 4.8) + (5.0 if checks else 0))
            fig = plt.figure(figsize=size)
            hr_top = [1, 0.28] if has_nbi else [1]
            # explicit margins: the paper PDF is saved at exactly PAPER_W without a tight box
            mar = dict(left=0.095, right=0.985, bottom=0.115 if checks else 0.15, top=0.91 if checks else 0.85)
            if checks:
                outer = fig.add_gridspec(2, 1, height_ratios=[sum(hr_top) * 1.05, 1.0], hspace=0.42, **mar)
                gtop = outer[0].subgridspec(len(hr_top), n, height_ratios=hr_top, hspace=0.08, wspace=0.25)
                gbot = outer[1].subgridspec(1, 2, wspace=0.3, width_ratios=[1, 1] if bridge is not None else [1.25, 1])
            else:
                gtop = fig.add_gridspec(len(hr_top), n, height_ratios=hr_top, hspace=0.08, wspace=0.25, **mar)
            axes = [[fig.add_subplot(gtop[0, j]) for j in range(n)]]
            if has_nbi:
                axes.append([fig.add_subplot(gtop[1, j], sharex=axes[0][j]) for j in range(n)])
            for i, (ax, (dur, e, (a, b)), r) in enumerate(zip(axes[0], chosen, res)):
                ch = chords[i]
                pr = r.pred[ch, :, k] * tsc
                ps = r.pred_sigma[ch, :, k] * args.k[k] * tsc
                y = r.y[ch, :, k] * tsc
                T = pr.shape[0]
                t = file_times(e, T, args.fs, args.t_offset)  # time_s from the file when present
                mb = args.margin_before or [args.margin]
                ma = args.margin_after or [args.margin]
                f0 = max(a - int(round(mb[min(i, len(mb) - 1)] * args.fs)), 0)
                f1 = min(b + int(round(ma[min(i, len(ma) - 1)] * args.fs)), T - 1)
                w = np.arange(f0, f1 + 1)
                beamoff_band(ax, t[a] - 0.5 / args.fs, t[b] + 0.5 / args.fs, label="beam off")
                model_line(ax, t[w], pr[w], ps[w], role=role1, label=args.label, ms=2.5, lw=1.2)
                if res2 is not None:
                    pr2 = res2[i].pred[ch, :, k] * tsc
                    ps2 = res2[i].pred_sigma[ch, :, k] * args.k2[k] * tsc
                    model_line(ax, t[w], pr2[w], ps2[w], role=role2, label=args.label2, ms=2.5, lw=1.2)
                mm = np.isfinite(y[w])
                ysig = r.sigma[ch, :, k] * tsc
                fit_points(ax, t[w][mm], y[w][mm], ysig[w][mm], label=NAMES["fit"])
                ax.set_xlim(t[f0], t[f1])
                shot = "".join(ch for ch in Path(e).stem if ch.isdigit())
                panel_label(ax, "ab"[i] if n <= 2 else chr(97 + i), f"discharge {shot}")
                if i == 0:
                    ax.set_ylabel(TARGET_LABEL[tgt])
                if has_nbi:
                    plt.setp(ax.get_xticklabels(), visible=False)
                    axn = axes[1][i]
                    beamoff_band(axn, t[a] - 0.5 / args.fs, t[b] + 0.5 / args.fs)
                    if nbi[i] is not None:
                        axn.plot(t[w], nbi[i][w], "-", color="0.3", lw=1.2)
                    axn.set_ylim(bottom=0)
                    axn.set_xlabel(TIME_LABEL)
                    axn.locator_params(axis="y", nbins=2)
                    if i == 0:
                        axn.set_ylabel(r"$P_\mathrm{NBI}$ (MW)", fontsize="small")
                else:
                    ax.set_xlabel(TIME_LABEL)
            h, l_ = axes[0][0].get_legend_handles_labels()
            if bridge is not None:
                axc, axd = fig.add_subplot(gbot[0, 0]), fig.add_subplot(gbot[0, 1])
                mi = len(bridge["models"]) - 1  # the fine-tuned model (last in the npz)
                g, nc = draw_bridge(axc, bridge, "vtor", mi, COLORS[role2])
                draw_bridge(axd, bridge, "ti", mi, COLORS[role2])
                panel_label(axc, "c", f"{g} beam-off phases, held-out discharges")
                panel_label(axd, "d")
            elif checks:
                axc, axd = fig.add_subplot(gbot[0, 0]), fig.add_subplot(gbot[0, 1])
                draw_context(axc, checks, tsc, tgt)
                draw_rejoin(axd, checks, tsc, tgt)
                panel_label(axc, "c", f"{checks['n_gaps']} beam-off phases")
                panel_label(axd, "d", f"first {1000 * checks['n_after'] / checks['fs']:.0f} ms after the beam returns")
            fig.align_ylabels([row[0] for row in axes])
            fig.canvas.draw()
            top = max(a_.get_position().y1 for a_ in axes[0])
            leg = ordered_legend(axes[0][0], h, l_, loc="lower center", ncol=len(h),
                                 bbox_to_anchor=(0.5, top + (0.035 if checks else 0.07)),
                                 bbox_transform=fig.transFigure, frameon=False,
                                 fontsize="small", handletextpad=0.4, columnspacing=1.2)
            leg.set_in_layout(False)
            fig.savefig(args.out / f"{args.out_name}.{ext}", dpi=300, bbox_inches="tight"
                        if ext == "png" else bbox(ext), bbox_extra_artists=[leg])
            plt.close(fig)
    print(f"wrote {args.out}/{args.out_name}.png/.pdf")


if __name__ == "__main__":
    main()
