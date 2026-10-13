"""Beam-off examples: reconstruction from the regular CHERS spectra while the
neutral beam is off (only passive emission left), next to the conventional
fits before and after.

Everything comes from one model checkpoint and the discharge files of the
regular array; no background array, no stored prediction dumps:

* beam-off phases are found in the same file: runs of native (100 Hz)
  frames inside the plasma without any conventional fit and with clearly
  reduced line brightness, bounded by fitted frames on both sides
* the model is run on the discharge directly (``cer_transfer.inference``)

    pixi run python -u -m cer_transfer.figures.beamoff_examples \\
        --checkpoint cer_ckpts/nstx_agn_ft_rfull.pt --list splits/nstx_val.txt \\
        --n 2 --chord 2 --target vtor --min-off-ms 50 --margin-before 0.20 0.06 \\
        [--k 0.99 1.17] [--machine nstx] --out figs --out-name beamoff_examples
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
from cer_transfer.figures.timeaxis import file_times  # noqa: E402
from cer_transfer.figures.paperfig import PAPER_W, bbox  # noqa: E402


def rotating_context(y, a, b, fs, context_ms, min_rot, min_fits, chord=None):
    """True if the plasma rotates on both sides of the phase (a, b).

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
        phases = [ab for ab in phases if rotating_context(y, ab[0], ab[1], fs, cms, mrot, mfit, ch)]
    return phases, T


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
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
    p.add_argument("--min-rotation", type=float, default=0.0,
                   help="keep only phases with median |v_tor| above this (km/s) on both sides "
                   "(labels only); 0 = no requirement")
    p.add_argument("--context-ms", type=float, default=50.0, help="window on each side for --min-rotation")
    p.add_argument("--min-fits-side", type=int, default=3, help="fitted frames required on each side")
    p.add_argument("--rotation-on-chord", action="store_true",
                   help="evaluate --min-rotation on the plotted chord only (default: median over chords)")
    p.add_argument("--checkpoint2", type=Path, default=None,
                   help="second model drawn in the same panels (orange), e.g. the fine-tuned NSTX model")
    p.add_argument("--label2", default="second model", help="legend text of --checkpoint2")
    p.add_argument("--k2", type=float, nargs=2, default=(1.0, 1.0),
                   help="calibration factors (Ti, vtor) of the second model's uncertainty")
    p.add_argument("--machine2", default=None, help="machine config for --checkpoint2 (default: its own)")
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
        rot = (args.fs, args.context_ms, args.min_rotation, args.min_fits_side,
               args.chord if args.rotation_on_chord else None)
    cands = []
    for e in entries:
        ph, _ = off_phases(e, min_frames, k, rotating=rot)
        if ph:
            longest = max(ph, key=lambda ab: ab[1] - ab[0])
            cands.append(((longest[1] - longest[0]) / args.fs, e, longest))
    if not cands:
        raise SystemExit("no beam-off phase found")
    cands.sort()
    if args.shots:
        chosen = cands[: args.n]
    else:  # representative: around the median phase length
        mid = len(cands) // 2
        lo = max(0, mid - args.n // 2)
        chosen = cands[lo : lo + args.n]
    print(
        f"{len(cands)} discharges with beam-off phases >= {args.min_off_ms:.0f} ms; using:"
    )
    for dur, e, (a, b) in chosen:
        print(
            f"  {e}: {file_times(e, b + 1, args.fs, args.t_offset)[a]:.3f}-{file_times(e, b + 1, args.fs, args.t_offset)[b]:.3f} s "
            f"({1000 * dur:.0f} ms)"
        )

    m = load_model(args.checkpoint, machine=args.machine)
    res = [predict_file(m, e) for _, e, _ in chosen]
    res2 = None
    if args.checkpoint2 is not None:
        m2 = load_model(args.checkpoint2, machine=args.machine2)
        res2 = [predict_file(m2, e) for _, e, _ in chosen]
    unit = "$v_{tor}$ (km/s)" if k else "$T_i$ (keV)"
    tsc = 1.0 if k else 1e-3  # T_i displayed in keV
    n = len(chosen)
    args.out.mkdir(parents=True, exist_ok=True)
    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            # NBI power from the discharge files (keys 'nbi_power_MW' (T,), optionally
            # 'nbi_power_src_MW' per source), time-aligned with the frames: second row
            nbi = []
            for _, e, _ in chosen:
                dd = load(data_path(e), mmap_mode="r")
                tot = np.asarray(dd["nbi_power_MW"], np.float64) if "nbi_power_MW" in dd else None
                src = np.asarray(dd["nbi_power_src_MW"], np.float64) if "nbi_power_src_MW" in dd else None
                nbi.append((tot, src))
            has_nbi = any(t_ is not None for t_, _ in nbi)
            size = ((PAPER_W, 3.4) if has_nbi else (PAPER_W, 2.5)) if ctx == "paper" else ((13, 6.2) if has_nbi else (13, 4.5))
            fig = plt.figure(figsize=size)
            gs = fig.add_gridspec(2 if has_nbi else 1, n, height_ratios=[1, 0.42] if has_nbi else [1])
            axes = [[fig.add_subplot(gs[0, j]) for j in range(n)]]
            if has_nbi:
                axes.append([fig.add_subplot(gs[1, j], sharex=axes[0][j]) for j in range(n)])
            for i, (ax, (dur, e, (a, b)), r) in enumerate(zip(axes[0], chosen, res)):
                pr = r.pred[args.chord, :, k] * tsc
                ps = r.pred_sigma[args.chord, :, k] * args.k[k] * tsc
                y = r.y[args.chord, :, k] * tsc
                T = pr.shape[0]
                t = file_times(e, T, args.fs, args.t_offset)  # time_s from the file when present
                mb = args.margin_before or [args.margin]
                ma = args.margin_after or [args.margin]
                f0 = max(a - int(round(mb[min(i, len(mb) - 1)] * args.fs)), 0)
                f1 = min(b + int(round(ma[min(i, len(ma) - 1)] * args.fs)), T - 1)
                w = np.arange(f0, f1 + 1)
                ax.axvspan(
                    t[a] - 0.5 / args.fs,
                    t[b] + 0.5 / args.fs,
                    color="0.88",
                    zorder=0,
                    lw=0,
                )
                ax.fill_between(
                    t[w], pr[w] - ps[w], pr[w] + ps[w], color="C0", alpha=0.22, lw=0
                )
                ax.plot(t[w], pr[w], "-", color="C0", lw=1.5, label=args.label, marker="o", ms=2.2)  # one marker per output
                if res2 is not None:
                    pr2 = res2[i].pred[args.chord, :, k] * tsc
                    ps2 = res2[i].pred_sigma[args.chord, :, k] * args.k2[k] * tsc
                    ax.fill_between(t[w], pr2[w] - ps2[w], pr2[w] + ps2[w], color="C1", alpha=0.18, lw=0)
                    ax.plot(t[w], pr2[w], "-", color="C1", lw=1.3, label=args.label2, marker="o", ms=2.0)
                mm = np.isfinite(y[w])
                ysig = r.sigma[args.chord, :, k] * tsc
                ax.errorbar(
                    t[w][mm],
                    y[w][mm],
                    yerr=ysig[w][mm],
                    fmt="o",
                    color="k",
                    ecolor="0.45",
                    ms=3,
                    lw=0.7,
                    zorder=5,
                    label="conventional fit and uncertainty",
                )
                ax.set_xlim(t[f0], t[f1])
                shot = "".join(ch for ch in Path(e).stem if ch.isdigit())
                ax.set_title(f"discharge {shot}", fontsize="small")
                if i == 0:
                    ax.set_ylabel(unit)
                    ax.legend(fontsize="x-small", loc="best")
                if has_nbi:
                    plt.setp(ax.get_xticklabels(), visible=False)
                    axn = axes[1][i]
                    tot, src = nbi[i]
                    axn.axvspan(t[a] - 0.5 / args.fs, t[b] + 0.5 / args.fs, color="0.88", zorder=0, lw=0)
                    if tot is not None:
                        axn.plot(t[w], tot[w], "-", color="C2", lw=1.5)
                    axn.set_ylim(bottom=0)
                    axn.set_xlabel("time (s)")
                    if i == 0:
                        axn.set_ylabel(r"$P_\mathrm{NBI}$ (MW)")
                else:
                    ax.set_xlabel("time (s)")
            fig.tight_layout()
            fig.savefig(
                args.out / f"{args.out_name}.{ext}", dpi=300, bbox_inches=bbox(ext)
            )
            plt.close(fig)
    print(f"wrote {args.out}/{args.out_name}.png/.pdf")


if __name__ == "__main__":
    main()
