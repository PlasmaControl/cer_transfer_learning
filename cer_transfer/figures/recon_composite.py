"""Reconstruction composite from REAL data (one discharge): input
spectrograms, time traces, and profiles; optional second-model overlay.
Generic tool -- the output name states the figure role:

  paper Fig 2 (legacy reconstruction):
    cer_transfer.figures.recon_composite --preds legacy.npz --shot <legacy joblib> \
        --label1 "model (51-chord)" --t-offset -0.235 --unl-gap 2 --out-name fig2_reconstruction
  paper Fig 3 (background vs active):
    cer_transfer.figures.recon_composite --preds passive.npz --preds2 active.npz \
        --label1 "background model" --label2 "active model" \
        --shot <background joblib> --t-offset -0.235 --unl-gap 2 --out-name fig3_background

    pixi run python -u fig1_composite.py --preds fig1_shot.npz \
        [--shot chers_115500.joblib] [--out figs]

Panels (data-driven; no labeled boxes):
  b1  input spectrogram of one chord (from --shot; skipped if absent)
  b2  labels: sparse fitted T_i/v_tor points (chord x time)
  b3  model reconstruction: dense T_i and v_tor maps + sigma
Geometry panel (a) is built separately once real tangency radii exist.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cer_transfer.configs import data_path
from cer_transfer.figures.common import chord_coordinate, sorted_chords
import seaborn as sns

sns.set_style("white")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--preds2", type=Path, default=None,
                   help="second predictions npz (e.g. the PASSIVE model on "
                        "the same discharge) overlaid as dash-dot lines; "
                        "labels/points still come from --preds")
    p.add_argument("--label1", type=str, default="model",
                   help="name of the --preds model in legends")
    p.add_argument("--label2", type=str, default="background model")
    p.add_argument("--shot", type=Path, default=None)
    p.add_argument("--trace-chords", type=str, default="13,20",
                   help="chords for the time-trace panels")
    p.add_argument("--gate", choices=("info", "amplitude"), default="info",
                   help="validity gate: 'info' = predicted sigma beats the "
                        "prior spread (model-internal, parameter-free); "
                        "'amplitude' = line emission above baseline")
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", type=str, default="recon_composite",
                   help="output stem; names the figure role")
    p.add_argument("--k", type=float, nargs=2, default=None,
                   help="calibration factors (Ti, vtor) applied to the model "
                        "sigma, e.g. 0.992 1.171; bands show the calibrated 1-sigma")
    p.add_argument("--no-band", action="store_true",
                   help="omit the model uncertainty bands")
    p.add_argument("--no-unl", action="store_true",
                   help="omit the profile at a time without conventional fit")
    p.add_argument("--profile-x", type=Path, default=None,
                   help="CSV with columns chord,x: physical coordinate per "
                        "chord for the profile panels (e.g. psi_N or R)")
    p.add_argument("--profile-xlabel", type=str, default=None,
                   help="axis label for --profile-x, e.g. '$\\psi_N$'")
    p.add_argument("--t-offset", type=float, default=0.0,
                   help="absolute time of frame 0 (s); all displayed "
                        "times = frame/200 + offset")
    p.add_argument("--unl-gap", type=int, default=1,
                   help="frames between the fitted and the no-fit "
                        "displayed profile (2 = 10 ms = NSTX native "
                        "cadence, avoids showing an interpolated frame)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.preds)
    ch = d["chord"]
    C = int(ch.max()) + 1
    T = len(ch) // C
    y = d["y"].reshape(C, T, -1)
    sl = d["sigma"].reshape(C, T, -1)
    pr = d["pred"].reshape(C, T, -1)
    kk = np.asarray(args.k if args.k is not None else (1.0, 1.0), dtype=float)
    ps = (d["pred_sigma"].reshape(C, T, -1) * kk[None, None, :]
          if ("pred_sigma" in d and not args.no_band) else None)
    band_lab = "model uncertainty"
    pr2 = None
    if args.preds2 is not None:
        d2 = np.load(args.preds2)
        assert len(d2["chord"]) == len(ch), "preds2 shape mismatch"
        pr2 = d2["pred"].reshape(C, T, -1)
    t_s = np.arange(T) / 200.0 + args.t_offset            # 200 Hz common time base

    lab_frac = np.isfinite(y[..., 0]).mean(axis=0)
    # well-labeled frame at PEAK plasma (max mean labeled Ti among the
    # well-covered frames), so the profile is representative
    cover = lab_frac > 0.5 * lab_frac.max()
    # outlier-immune: MEDIAN labeled Ti over chords (broken fits with
    # multi-keV outliers must not win the argmax), and exclude the first/
    # last 10% of frames (shot-edge frames are not representative)
    edge = max(T // 10, 1)
    cover[:edge] = False
    cover[-edge:] = False
    with np.errstate(all="ignore"):
        med_ti = np.where(cover,
                          np.nanmedian(np.where(np.isfinite(y[..., 0]),
                                                y[..., 0], np.nan), axis=0),
                          -np.inf)
    f_lab = int(np.argmax(med_ti))
    unl = np.where(lab_frac == 0)[0]
    f_unl = None
    if unl.size:
        # beam-off but plasma-active frame: just outside the labeled
        # window, ranked by the model's own mean Ti (plasma proxy)
        lab_t = np.where(lab_frac > 0)[0]
        lo, hi = lab_t.min(), lab_t.max()
        # between-fits frame: inside the beam window (spectra carry
        # signal) but at a time the fitting cadence left unlabeled
        cand = unl[(unl > lo) & (unl < hi)]
        if cand.size == 0:
            cand = unl
        # at least unl-gap frames from the fitted display frame (gap 2
        # = 10 ms = NSTX native cadence -> not an interpolated frame)
        far = cand[np.abs(cand - f_lab) >= args.unl_gap]
        if far.size:
            cand = far
        f_unl = int(cand[np.argmin(np.abs(cand - f_lab))])

    tr_chords = [int(x) for x in args.trace_chords.split(",")]

    gate = None
    if args.gate == "info" and "pred_sigma" in d:
        # information gate: the model's prediction counts as a measurement
        # only where its own predicted sigma beats the prior spread of the
        # target. Frame-level: median over chords, both targets must pass.
        psig = d["pred_sigma"].reshape(C, T, -1)
        ok = np.ones(T, bool)
        for t in range(y.shape[-1]):
            prior = np.nanstd(y[..., t])
            ok &= np.median(psig[..., t], axis=0) < prior
        k = 5
        gate = np.convolve(ok.astype(int), np.ones(k, int), "same") >= k

    spec = None
    if args.shot is not None:
        from joblib import load
        s = load(data_path(args.shot), mmap_mode="r")
        end = int(s["end_index"])
        spec = {cc: np.asarray(s["input"][cc, :end, :], dtype=np.float32)
                for cc in tr_chords}
        if args.gate == "amplitude" or gate is None:
            # emission gate: mean line amplitude above per-chord baseline
            allspec = np.asarray(s["input"][:, :end, :], dtype=np.float32)
            base = np.median(allspec, axis=2, keepdims=True)
            amp = np.clip(allspec - base, 0, None).mean(axis=(0, 2))
            thr = 0.05 * np.percentile(amp, 95)
            g = amp > thr
            k = 5
            g = np.convolve(g.astype(int), np.ones(k, int), "same") >= k
            gate = np.zeros(T, bool)
            gate[: min(end, T)] = g[: min(end, T)]

    if args.no_unl:
        f_unl = None

    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            ncols = 3 if spec is not None else 2
            size = (7.2, 4.6) if ctx == "paper" else (13.5, 8)
            fig, axes = plt.subplots(2, ncols, figsize=size)
            col = 0
            if spec is not None:
                for r, cc in enumerate(tr_chords[:2]):
                    ax = axes[r, 0]
                    sp = spec[cc]
                    z = np.log10(np.clip(sp - sp.min() + 1, 1, None)).T
                    ax.imshow(z, aspect="auto", origin="lower",
                              cmap="magma",
                              extent=[t_s[0], t_s[min(len(sp)-1, T-1)],
                                      0, z.shape[0]])
                    ax.set_title(f"input spectrogram, chord {cc}",
                                 fontsize="medium")
                    ax.set_ylabel("wavelength bin")
                    if r == 1:
                        ax.set_xlabel("time (s)")
                col = 1

            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            gm = (gate if gate is not None else np.ones(T, bool))[:T]

            def trace_panel(ax, k, ylab):
                vals = []
                for i, cc in enumerate(tr_chords):
                    if ps is not None:
                        ax.fill_between(t_s, pr[cc, :, k] - ps[cc, :, k],
                                        pr[cc, :, k] + ps[cc, :, k],
                                        color=f"C{i}", alpha=0.2, lw=0,
                                        zorder=1)
                    ax.plot(t_s, pr[cc, :, k], "-", color=f"C{i}", lw=1.6,
                            zorder=2)
                    if pr2 is not None:
                        ax.plot(t_s, pr2[cc, :, k], "-.", color=f"C{i}",
                                lw=1.2, alpha=0.9, zorder=2)
                    m = np.isfinite(y[cc, :, k])
                    ax.errorbar(t_s[m], y[cc, m, k], yerr=sl[cc, m, k],
                                fmt="o", ms=2.2, lw=0.6, color="k",
                                ecolor="0.45", zorder=3)
                    vals += [pr[cc, gm, k], y[cc, m & gm, k]]
                v = np.concatenate([x_[np.isfinite(x_)] for x_ in vals])
                if v.size:
                    lo, hi = np.percentile(v, [1, 99])
                    pad = 0.12 * max(hi - lo, 1e-6)
                    ax.set_ylim(lo - pad, hi + pad)
                ax.set_xlabel("time (s)")
                ax.set_ylabel(ylab)

            # time traces at fixed chords
            ax = axes[0, col]
            trace_panel(ax, 1, "$v_{tor}$ (km/s)")
            ax.set_title("time traces", fontsize="medium")
            if pr2 is None:
                hnd = [Line2D([], [], color=f"C{i}", lw=1.6)
                       for i in range(len(tr_chords))]
                lbl = [f"chord {cc} ({args.label1})" for cc in tr_chords]
                if ps is not None:
                    hnd.append(Patch(color="0.6", alpha=0.35))
                    lbl.append(band_lab)
                hnd.append(Line2D([], [], color="k", marker="o", ls="", ms=3))
                lbl.append("conventional fit")
                ax.legend(hnd, lbl, fontsize="x-small", loc="best",
                          framealpha=0.9)
            else:
                # two models: chords labelled at their traces, legend for
                # line styles only (keeps it short)
                tg = t_s[gm]
                for i, cc in enumerate(tr_chords):
                    tail = pr[cc, gm, 1][int(0.6 * len(tg)):]
                    yl = float(np.nanpercentile(tail, 90)) if tail.size else 0.0
                    ax.annotate(f"chord {cc}", (tg[int(0.78 * len(tg))], yl),
                                xytext=(0, 4), textcoords="offset points",
                                ha="center", va="bottom", fontsize="xx-small",
                                color=f"C{i}", fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                          ec="none", alpha=0.85), zorder=6)
                hnd = [Line2D([], [], color="0.3", lw=1.6),
                       Line2D([], [], color="0.3", ls="-.", lw=1.2)]
                lbl = [args.label1, args.label2]
                if ps is not None:
                    hnd.append(Patch(color="0.6", alpha=0.35))
                    lbl.append(band_lab)
                hnd.append(Line2D([], [], color="k", marker="o", ls="", ms=3))
                lbl.append("conventional fit")
                ax.legend(hnd, lbl, fontsize="xx-small", loc="lower right",
                          framealpha=0.9, handlelength=1.8)
            trace_panel(axes[1, col], 0, "$T_i$ (eV)")

            # profiles at fixed times, optionally in a physical coordinate
            xm, xlab = chord_coordinate(C, args.profile_x, args.profile_xlabel)
            order, xs = sorted_chords(xm)

            def nofit_spans(k):
                # runs (in x order) of chords with a model value but no fit
                nf = [not np.isfinite(y[i_, f_lab, k]) for i_ in order]
                spans, a = [], None
                for n_, flag in enumerate(nf + [False]):
                    if flag and a is None:
                        a = n_
                    if not flag and a is not None:
                        b = n_ - 1
                        dx = np.diff(xs)
                        step = float(np.median(dx)) if dx.size else 0.5
                        lo_ = xs[a] - 0.5 * ((xs[a] - xs[a - 1]) if a > 0 else step)
                        hi_ = xs[b] + 0.5 * ((xs[b + 1] - xs[b])
                                             if b + 1 < len(xs) else step)
                        spans.append((lo_, hi_))
                        a = None
                return spans

            def profile_panel(ax, k, ylab, legend, label_span):
                spans = nofit_spans(k)
                for lo_, hi_ in spans:
                    ax.axvspan(lo_, hi_, color="0.9", zorder=0, lw=0)
                if ps is not None:
                    # band only for the fitted-time profile; lighter where no
                    # fit exists (large uncertainty, may leave the axes)
                    lo_b = pr[order, f_lab, k] - ps[order, f_lab, k]
                    hi_b = pr[order, f_lab, k] + ps[order, f_lab, k]
                    fitm = np.isfinite(y[order, f_lab, k])
                    nof = ~fitm
                    nof_ext = nof | np.r_[nof[1:], False] | np.r_[False, nof[:-1]]
                    ax.fill_between(xs, lo_b, hi_b, where=fitm | ~nof_ext | (fitm & nof_ext),
                                    color="C0", alpha=0.2, lw=0, zorder=1)
                    ax.fill_between(xs, lo_b, hi_b, where=nof_ext,
                                    color="C0", alpha=0.07, lw=0, zorder=1)
                ax.plot(xs, pr[order, f_lab, k], "-", color="C0", lw=1.6,
                        zorder=2)
                curves = [pr[order, f_lab, k]]
                if pr2 is not None:
                    ax.plot(xs, pr2[order, f_lab, k], "-.", color="C2",
                            lw=1.5, zorder=2)
                    curves.append(pr2[order, f_lab, k])
                if f_unl is not None:
                    ax.plot(xs, pr[order, f_unl, k], "--", color="C3",
                            lw=1.5, zorder=2)
                    curves.append(pr[order, f_unl, k])
                m = np.isfinite(y[order, f_lab, k])
                ax.errorbar(xs[m], y[order, f_lab, k][m],
                            yerr=sl[order, f_lab, k][m], fmt="o", ms=2.8,
                            lw=0.6, color="k", ecolor="0.45", zorder=3)
                # y-limits from the curves and fits, not from the band
                vals = np.concatenate([np.ravel(v_) for v_ in curves]
                                      + [y[order, f_lab, k][m]])
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    lo, hi = vals.min(), vals.max()
                    padv = 0.12 * max(hi - lo, 1e-6)
                    ax.set_ylim(lo - padv, hi + padv)
                ax.set_ylabel(ylab)
                dx = np.diff(xs)
                pad = 0.5 * (float(np.median(dx)) if dx.size else 0.5)
                ax.set_xlim(xs[0] - pad, xs[-1] + pad)
                if label_span and spans:
                    lo_, hi_ = max(spans, key=lambda s_: s_[1] - s_[0])
                    ax.text(0.5 * (lo_ + hi_), 0.96, "no\nconventional\nfit",
                            ha="center", va="top", fontsize="xx-small",
                            color="0.35", transform=ax.get_xaxis_transform())
                if legend:
                    hh = [Line2D([], [], color="C0", lw=1.6)]
                    ll = [f"{args.label1}, t = {t_s[f_lab]:.3f} s"]
                    if pr2 is not None:
                        hh.append(Line2D([], [], color="C2", ls="-.", lw=1.5))
                        ll.append(f"{args.label2}, t = {t_s[f_lab]:.3f} s")
                    if f_unl is not None:
                        hh.append(Line2D([], [], color="C3", ls="--", lw=1.5))
                        ll.append(f"{args.label1}, t = {t_s[f_unl]:.3f} s (no fit)")
                    hh.append(Line2D([], [], color="k", marker="o", ls="",
                                     ms=3))
                    ll.append(f"conventional fit, t = {t_s[f_lab]:.3f} s")
                    # core rotation is high, so the lower left stays empty
                    ax.legend(hh, ll, fontsize="xx-small", loc="lower left",
                              framealpha=0.9, handlelength=1.6)

            prof_legend = []
            ax = axes[0, col + 1]
            profile_panel(ax, 1, "$v_{tor}$ (km/s)", legend=True,
                          label_span=True)
            ax.set_title("profiles: model vs. fit", fontsize="medium")
            ax = axes[1, col + 1]
            profile_panel(ax, 0, "$T_i$ (eV)", legend=False, label_span=False)
            ax.set_xlabel(xlab)

            t0 = t_s[int(np.argmax(gate))] if (gate is not None
                                               and gate.any()) else t_s[0]
            t1 = (t_s[len(gate) - 1 - int(np.argmax(gate[::-1]))]
                  if (gate is not None and gate.any()) else t_s[-1])
            for r in range(2):
                for cix in range(col + 1):
                    axes[r, cix].set_xlim(t0, t1)

            fig.tight_layout()
            fig.savefig(args.out / f"{args.out_name}.{ext}", dpi=300,
                        bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {args.out}/{args.out_name}.png/.pdf | "
          f"labeled frame t={t_s[f_lab]:.3f}s"
          + (f", unlabeled frame t={t_s[f_unl]:.3f}s" if f_unl is not None
             else ""))


if __name__ == "__main__":
    main()
