"""Time traces through a beam-off window from BOTH
array models, with the beam state determined from the spectra.

Top strip: active/background line-brightness ratio. The viewed beam's
charge-exchange emission appears only on the active array, so the ratio
collapses when that beam is off -- a data-driven beam state, independent
of whether conventional fits exist. Panels: selected chords, Ti and
vtor; background model (solid, 1-sigma band), active-array model
(dashed, band), conventional fits (points); spectrally beam-off frames
shaded.

    pixi run python -u -m cer_transfer.figures.beamoff_traces \
        --bg gallery/bo_141710_bg.npz --fg gallery/bo_141710_fg.npz \
        --bg-file "$(cat gallery/bo_141710_bg.txt)" \
        --fg-file "$(cat gallery/bo_141710_fg.txt)" \
        --t0 0.15 --t1 0.42 --t-offset -0.235 --chords 2,15,30 \
        --out figs --out-name beamoff_traces_141710
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

from cer_transfer.beamstate import valid_end
from cer_transfer.configs import data_path

sns.set_style("whitegrid")
from cer_transfer.figures.common import FRAME_HZ, load_predictions

FS = FRAME_HZ  # overridden by --fs
PRE = 47  # recording pre-trigger frames: pre-plasma by construction


def load_pred(npz):
    """Prediction dump as (pred, pred_sigma, y, T) with (C, T, K) arrays."""
    dmp = load_predictions(npz)
    return dmp.pred, dmp.pred_sigma, dmp.y, dmp.T


def brightness(fp):
    """Dark-subtracted total line brightness per frame, its dark noise,
    the fitted-frame mask, and the valid length."""
    d = load(data_path(fp), mmap_mode="r")
    end = valid_end(d)
    spec = np.asarray(d["input"][:, :end, :], dtype=np.float32)
    tot = spec.sum(axis=(0, 2))
    pre = min(PRE, end - 1)
    dw = tot[: max(8, pre - 2)]
    dk = np.median(dw)
    nz = 1.4826 * np.median(np.abs(dw - dk))
    lab = np.isfinite(np.asarray(d["target"][:, :end, 0])).any(axis=0)
    return tot - dk, max(nz, 1e-6 * abs(dk) + 1e-6), lab, end


def runs(mask):
    """Contiguous True runs as (start, stop_inclusive)."""
    out, f = [], 0
    while f < len(mask):
        if mask[f]:
            g = f
            while f < len(mask) and mask[f]:
                f += 1
            out.append((g, f - 1))
        else:
            f += 1
    return out


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--fs", type=float, default=FRAME_HZ, help="frame rate (Hz)")
    p.add_argument("--bg", type=Path, required=True)
    p.add_argument("--fg", type=Path, required=True)
    p.add_argument("--bg-file", type=Path, required=True)
    p.add_argument("--fg-file", type=Path, required=True)
    p.add_argument("--t0", type=float, required=True)
    p.add_argument("--t1", type=float, required=True)
    p.add_argument("--t-offset", type=float, default=0.0)
    p.add_argument("--chords", type=str, default="2,15,30")
    p.add_argument(
        "--targets",
        type=str,
        default="ti,vtor",
        help="columns to draw: ti, vtor or ti,vtor",
    )
    p.add_argument(
        "--mark",
        nargs="*",
        default=[],
        help="event markers 'time:label' (experimental time), "
        "e.g. '0.155:mode locking'",
    )
    p.add_argument(
        "--no-fg", action="store_true", help="omit the foreground-array model curve"
    )
    p.add_argument(
        "--spans",
        action="store_true",
        help="shade the full periods without conventional "
        "measurement (last fit before to first fit after "
        "each verified beam-off phase); implied by --hero",
    )
    p.add_argument(
        "--no-strip",
        action="store_true",
        help="omit the brightness-ratio strip (single-panel figure)",
    )
    p.add_argument(
        "--hero",
        action="store_true",
        help="compact single-chord layout: phase labels on the "
        "beam-state strip, descriptive legend",
    )
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", type=str, default="beamoff_traces")
    args = p.parse_args()
    global FS
    FS = args.fs
    args.out.mkdir(parents=True, exist_ok=True)

    pb, sb, yb, Tb = load_pred(args.bg)
    if args.no_fg and not Path(args.fg).exists():
        pf, sf, yf, Tf = pb, sb, yb, Tb  # not drawn; placeholder
    else:
        pf, sf, yf, Tf = load_pred(args.fg)
    bb, nzb, labb, eb = brightness(args.bg_file)
    bf, _, _, ef = brightness(args.fg_file)
    T = min(Tb, Tf, eb, ef)
    bb, bf, labb = bb[:T], bf[:T], labb[:T]

    active = bb > 8 * nzb
    active[:PRE] = False
    with np.errstate(all="ignore"):
        R = np.where(active, bf / np.maximum(bb, 1e-12), np.nan)

    on_ref = active & labb
    R_on = np.nanmedian(R[on_ref]) if on_ref.any() else np.nan
    R_low = np.nanpercentile(R[active], 5) if active.any() else np.nan
    if not np.isfinite(R_on) or not np.isfinite(R_low) or R_low > 0.7 * R_on:
        thr = np.nan
        beam_off = np.zeros(T, bool)
        print(
            "spectral beam state: NO beam-off detected in this shot "
            f"(R_on {R_on:.3g}, R_low {R_low:.3g})"
        )
    else:
        thr = np.sqrt(R_on * R_low)
        beam_off = active & (R < thr)
    # frames without fits, ignoring the 1-frame interleave gaps of the
    # native 100 Hz fit grid resampled to 200 Hz: runs of >= 3 frames
    nofit = np.zeros(T, bool)
    for a, b in runs(active & ~labb):
        if b - a + 1 >= 3:
            nofit[a : b + 1] = True
    n_nf = int(nofit.sum())
    n_both = int((nofit & beam_off).sum())
    n_fit_off = int((labb & beam_off).sum())
    print(
        f"spectral beam state: R_on {R_on:.3g} | R_low {R_low:.3g} | "
        f"threshold {thr:.3g}"
    )
    print(f"  spectrally beam-off frames: {int(beam_off.sum())}")
    print(
        f"  plasma frames without fits: {n_nf}, of which {n_both} "
        f"({100 * n_both / max(n_nf, 1):.0f}%) spectrally beam-off"
    )
    print(f"  fitted frames flagged beam-off: {n_fit_off} (expect ~0)")

    f0 = max(int(round((args.t0 - args.t_offset) * FS)), 0)
    f1 = min(int(round((args.t1 - args.t_offset) * FS)), T - 1)
    w = np.arange(f0, f1 + 1)
    t = np.arange(T) / FS + args.t_offset
    chords = [int(c) for c in args.chords.split(",")]
    C = pb.shape[0]
    for c in chords:
        if not 0 <= c < C:
            raise SystemExit(f"chord {c} out of range 0..{C - 1}")

    bo_w = beam_off[w]
    on_w = labb[w]
    for k, n in enumerate(("Ti (eV)", "vtor (km/s)")):
        d_all = np.abs(pf[:, w, k] - pb[:, w, k])
        if bo_w.any() and on_w.any():
            print(
                f"  window inter-array median|diff| {n}: beam-off "
                f"{np.median(d_all[:, bo_w]):.3g} | beam-on "
                f"{np.median(d_all[:, on_w]):.3g}"
            )

    marks = []
    for m_ in args.mark:
        tm, _, lb = m_.partition(":")
        marks.append((float(tm), lb))

    use_spans = args.spans or args.hero
    spans = []
    if use_spans:
        from cer_transfer.beamstate import beam_state

        st = beam_state(args.bg_file, args.fg_file)
        fr = np.where(labb)[0]
        for a, b, cls, ok, _ in st["segments"]:
            if not ok or a >= T:
                continue
            left, right = fr[fr < a], fr[fr > b]
            s0 = int(left[-1]) + 1 if left.size else a
            s1 = int(right[0]) - 1 if right.size else min(b, T - 1)
            if spans and s0 <= spans[-1][1] + 1:
                spans[-1] = (spans[-1][0], max(spans[-1][1], s1))
            else:
                spans.append((s0, s1))
        print(
            "periods without conventional measurement: "
            + (
                ", ".join(
                    f"{a / FS + args.t_offset:.3f}-{b / FS + args.t_offset:.3f} s"
                    for a, b in spans
                )
                or "none"
            )
        )

    def shade(ax):
        """Shade beam-off spans (or verified frames) and draw markers."""
        for tm, _ in marks:
            ax.axvline(tm, ls="--", color="k", lw=0.9, zorder=4)
        if use_spans:
            for a, b in spans:
                if b < f0 or a > f1:
                    continue
                ax.axvspan(
                    t[max(a, f0)] - 0.5 / FS,
                    t[min(b, f1)] + 0.5 / FS,
                    color="0.88",
                    zorder=0,
                    lw=0,
                )
            return
        for a, b in runs(beam_off[f0 : f1 + 1]):
            ax.axvspan(
                t[f0 + a] - 0.5 / FS, t[f0 + b] + 0.5 / FS, color="0.88", zorder=0, lw=0
            )

    names = ("$T_i$ (eV)", "$v_{tor}$ (km/s)")
    ks = [{"ti": 0, "vtor": 1}[s.strip()] for s in args.targets.split(",")]
    n = len(chords)
    ncol = len(ks)
    lab_bg, lab_fg = (
        (
            "passive reconstruction (background array)",
            "reconstruction from foreground array",
        )
        if args.hero
        else ("background model", "active-array model")
    )
    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            wd = 3.6 * ncol if ncol > 1 else 5.0
            size = (
                (wd, 1.2 + 1.6 * n) if ctx == "paper" else (6.5 * ncol, 2.2 + 2.8 * n)
            )
            fig = plt.figure(figsize=size)
            ns = 0 if args.no_strip else 1
            gs = fig.add_gridspec(
                n + ns,
                ncol,
                height_ratios=([0.6] if ns else []) + [1.0] * n,
                hspace=0.25,
                wspace=0.25,
            )

            def span_labels(axx):
                """Label wide spans "no conventional measurement"."""
                for a, b in spans:
                    vis = min(b, f1) - max(a, f0)
                    if vis >= 0.15 * (f1 - f0):
                        axx.text(
                            0.5 * (t[max(a, f0)] + t[min(b, f1)]),
                            0.95,
                            "no conventional measurement",
                            ha="center",
                            va="top",
                            fontsize="x-small",
                            transform=axx.get_xaxis_transform(),
                        )

            # one beam-state strip per column, so its time axis aligns
            # with the traces below it
            tops = [None] * ncol
            for j, k in enumerate(ks):
                if not ns:
                    break
                a0 = fig.add_subplot(gs[0, j])
                shade(a0)
                for tm, lb in marks:
                    if lb:
                        a0.text(
                            tm,
                            0.92,
                            " " + lb,
                            ha="left",
                            va="top",
                            fontsize="x-small",
                            transform=a0.get_xaxis_transform(),
                        )
                a0.plot(t[w], R[w], "-", color="k", lw=1.2)
                if np.isfinite(thr):
                    a0.axhline(thr, ls=":", color="0.4", lw=1.0)
                if j == 0:
                    a0.set_ylabel("fg / bg\nbrightness", fontsize="small")
                if args.hero:
                    span_labels(a0)
                a0.tick_params(labelbottom=False)
                a0.set_xlim(t[f0], t[f1])
                tops[j] = a0
            first = True
            for r, c in enumerate(chords):
                for j, k in enumerate(ks):
                    ax = fig.add_subplot(gs[r + ns, j], sharex=tops[j])
                    if tops[j] is None:
                        tops[j] = ax
                        ax.set_xlim(t[f0], t[f1])
                        if args.hero:
                            span_labels(ax)
                    shade(ax)
                    curves = [(pb, sb, "C0", "-", lab_bg)]
                    if not args.no_fg:
                        curves.append((pf, sf, "C1", "--", lab_fg))
                    for pr, sg, col, ls, lab in curves:
                        m_ = pr[c, w, k]
                        s_ = sg[c, w, k]
                        ax.fill_between(
                            t[w], m_ - s_, m_ + s_, color=col, alpha=0.22, lw=0
                        )
                        ax.plot(
                            t[w],
                            m_,
                            ls,
                            color=col,
                            lw=1.4,
                            label=lab if first else None,
                        )
                    yv = yb[c, w, k]
                    mm = np.isfinite(yv)
                    ax.plot(
                        t[w][mm],
                        yv[mm],
                        "o",
                        color="k",
                        ms=3,
                        zorder=5,
                        label="conventional fit" if first else None,
                    )
                    vals = np.concatenate([pb[c, w, k], pf[c, w, k]])
                    lo, hi = np.nanpercentile(vals, [1, 99])
                    if mm.any():
                        flo, fhi = np.nanpercentile(yv[mm], [5, 95])
                        lo, hi = min(lo, flo), max(hi, fhi)
                    pad = 0.15 * max(hi - lo, 1e-6)
                    ax.set_ylim(lo - pad, hi + pad)
                    ax.set_ylabel(f"chord {c}\n{names[k]}", fontsize="small")
                    if r < n - 1:
                        ax.tick_params(labelbottom=False)
                    else:
                        ax.set_xlabel("time (s)")
                    if first:
                        if args.hero and args.no_strip:
                            ax.legend(fontsize="x-small", loc="lower right")
                        elif args.hero:
                            ax.legend(
                                fontsize="x-small",
                                loc="upper left",
                                bbox_to_anchor=(0.0, -0.32),
                                ncol=1,
                                frameon=False,
                            )
                        else:
                            ax.legend(fontsize="x-small", loc="best")
                        first = False
            if args.title:
                fig.suptitle(args.title, fontsize="medium")
            fig.savefig(
                args.out / f"{args.out_name}.{ext}", dpi=300, bbox_inches="tight"
            )
            plt.close(fig)
    print(f"wrote {args.out}/{args.out_name}.png/.pdf")


if __name__ == "__main__":
    main()
