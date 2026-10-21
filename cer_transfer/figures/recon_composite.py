r"""Reconstruction composite of one discharge.

Input spectrograms, time traces, and profiles; optional second-model overlay.
Generic tool -- the output name states the figure role:

  paper Fig 2 (reconstruction, fine-tuned model):
    cer_transfer.figures.recon_composite --preds gallery/137711/agnostic_ft.npz \
        --shot <137711 joblib> --label1 "DIII-D+NSTX (7307 discharges) model" \
        --t-offset -0.235 --unl-gap 2 --k 0.734 0.817 \
        --profile-x figs/chords_Rtan_137711.csv --profile-xlabel '$R_\mathrm{tan}$ (m)' \
        --out figs --out-name recon_legacy_137711
  zero-shot vs fine-tuned (two models in the same panels):
    ... --preds <DIII-D model dump> --label1 "DIII-D model" \
        --preds2 <fine-tuned dump> --label2 "DIII-D+NSTX (7307 discharges) model" ...

Layout (one row per trace chord, rows share the time axis):
  [input spectrogram | v_tor trace | T_i trace]  of that chord, panel label
  "<letter>  R_tan = ... m"; last column: the v_tor (top) and T_i (bottom)
  profile at one fitted frame (label "<letter>  t = ... s"), optionally with
  the model profile at a native frame without conventional fit next to it.
  Vertical lines in the trace panels mark the profile time(s): dashed = the
  fitted frame, dotted = the frame without fit.

Conventions (cer_transfer.figures.style): conventional fit = black points
with error bars; each model in the colour of its role, taken from its legend
label (DIII-D model = green, NSTX model = red, DIII-D+NSTX model = blue), with
a +-1 sigma band at 20 % alpha; legend order fit, zero-shot, scratch,
fine-tuned; T_i in keV; no panel titles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cer_transfer.configs import data_path
from cer_transfer.figures import style
from cer_transfer.figures.common import chord_coordinate, sorted_chords
from cer_transfer.figures.paperfig import PAPER_W, bbox  # noqa: E402
from cer_transfer.figures.timeaxis import file_times  # noqa: E402

sns.set_style("white")

T_LINE = dict(color="0.35", lw=0.9, zorder=0.5)  # profile-time markers in the traces
LS_LAB, LS_UNL = "--", ":"  # fitted / no-fit profile time


def panel_label(ax, letter, text=None, **kw):
    """Bold panel letter, then the panel's identifier in normal weight."""
    s = rf"$\mathbf{{{letter}}}$" + (f"  {text}" if text else "")
    ax.text(0.0, 1.02, s, transform=ax.transAxes, ha="left", va="bottom", **kw)


def model_roles(label1, label2):
    """Role (colour) of each model from its legend label; distinct fallbacks."""
    r1 = style.role_of(label1)
    r1 = r1 if r1 in style.COLORS and r1 != "fit" else "finetuned"
    r2 = style.role_of(label2)
    if r2 not in style.COLORS or r2 == "fit" or r2 == r1:
        r2 = next(r for r in ("zero_shot", "scratch", "finetuned") if r != r1)
    return r1, r2


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument(
        "--preds2",
        type=Path,
        default=None,
        help="second predictions npz of the same discharge, drawn in the same "
        "panels in the colour of its role (--label2); fits come from --preds",
    )
    p.add_argument(
        "--label1",
        type=str,
        default="model",
        help="name of the --preds model in legends (its role sets the colour)",
    )
    p.add_argument(
        "--t-start",
        type=float,
        default=0.0,
        help="start of the time axis in s (default 0)",
    )
    p.add_argument(
        "--vtor-only",
        action="store_true",
        help="draw only rotation: one row [spectrogram | trace chord 1 | trace "
        "chord 2 | profile]",
    )
    p.add_argument("--label2", type=str, default="background model")
    p.add_argument("--shot", type=Path, default=None)
    p.add_argument(
        "--trace-chords",
        type=str,
        default="13,20",
        help="two chords for the time-trace panels (one row each)",
    )
    p.add_argument(
        "--gate",
        choices=("info", "amplitude"),
        default="info",
        help="validity gate: 'info' = predicted sigma beats the "
        "prior spread (model-internal, parameter-free); "
        "'amplitude' = line emission above baseline",
    )
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument(
        "--out-name",
        type=str,
        default="recon_composite",
        help="output stem; names the figure role",
    )
    p.add_argument(
        "--k",
        type=float,
        nargs=2,
        default=None,
        help="calibration factors (Ti, vtor) applied to the --preds model "
        "sigma, e.g. 0.992 1.171; bands show the calibrated 1-sigma",
    )
    p.add_argument(
        "--k2",
        type=float,
        nargs=2,
        default=None,
        help="calibration factors (Ti, vtor) of the --preds2 model sigma "
        "(default: uncalibrated)",
    )
    p.add_argument(
        "--no-band", action="store_true", help="omit the model uncertainty bands"
    )
    p.add_argument(
        "--no-unl",
        action="store_true",
        help="omit the profile at a time without conventional fit",
    )
    p.add_argument(
        "--profile-time",
        type=float,
        default=None,
        help="time (s) of the profile panels: the nearest well-fitted frame "
        "(default: the frame of peak median T_i that has a native no-fit "
        "neighbour)",
    )
    p.add_argument(
        "--profile-x",
        type=Path,
        default=None,
        help="CSV with columns chord,x: physical coordinate per "
        "chord for the profile panels (e.g. psi_N or R)",
    )
    p.add_argument(
        "--profile-xlabel",
        type=str,
        default=None,
        help="axis label for --profile-x, e.g. '$R_\\mathrm{tan}$ (m)'",
    )
    p.add_argument(
        "--t-offset",
        type=float,
        default=0.0,
        help="absolute time of frame 0 (s); all displayed "
        "times = frame/200 + offset",
    )
    p.add_argument(
        "--unl-gap",
        type=int,
        default=1,
        help="frames between the fitted and the no-fit "
        "displayed profile (2 = 10 ms = NSTX native "
        "cadence, avoids showing an interpolated frame)",
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.preds)
    ch = d["chord"]
    C = int(ch.max()) + 1
    T = len(ch) // C
    sc = np.array([style.TARGET_SCALE["ti"], style.TARGET_SCALE["vtor"]])  # T_i eV -> keV
    y = d["y"].reshape(C, T, -1).astype(np.float64) * sc
    sl = d["sigma"].reshape(C, T, -1).astype(np.float64) * sc
    pr = d["pred"].reshape(C, T, -1).astype(np.float64) * sc
    kk = np.asarray(args.k if args.k is not None else (1.0, 1.0), dtype=float)
    ps = (
        d["pred_sigma"].reshape(C, T, -1).astype(np.float64) * kk * sc
        if ("pred_sigma" in d and not args.no_band)
        else None
    )
    pr2 = ps2 = None
    if args.preds2 is not None:
        d2 = np.load(args.preds2)
        assert len(d2["chord"]) == len(ch), "preds2 shape mismatch"
        pr2 = d2["pred"].reshape(C, T, -1).astype(np.float64) * sc
        kk2 = np.asarray(args.k2 if args.k2 is not None else (1.0, 1.0), dtype=float)
        if "pred_sigma" in d2 and not args.no_band:
            ps2 = d2["pred_sigma"].reshape(C, T, -1).astype(np.float64) * kk2 * sc
    role1, role2 = model_roles(args.label1, args.label2 if pr2 is not None else None)
    col1, col2 = style.COLORS[role1], style.COLORS[role2]
    t_s = (
        file_times(args.shot, T, 200.0, args.t_offset)
        if args.shot
        else np.arange(T) / 200.0 + args.t_offset
    )

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
        med_ti = np.where(
            cover,
            np.nanmedian(np.where(np.isfinite(y[..., 0]), y[..., 0], np.nan), axis=0),
            -np.inf,
        )
    f_lab = int(np.argmax(med_ti))
    f_unl = None
    lab_t = np.where(lab_frac > 0)[0]
    if lab_t.size:
        # NSTX records at 100 Hz: on the 200 Hz grid only every second frame
        # is a measurement (same parity as the labeled frames); the others
        # are interpolated and are never used as the no-fit frame.
        par = np.bincount(lab_t % 2, minlength=2).argmax()
        step = args.unl_gap if args.unl_gap % 2 == 0 else args.unl_gap + 1
        native = np.arange(T) % 2 == par
        if args.profile_time is not None:
            # the requested time: nearest well-covered native frame, and its
            # no-fit neighbour if one exists
            cand = np.where(cover & native)[0]
            if not cand.size:
                cand = np.where(lab_frac > 0)[0]
            f_lab = int(cand[np.argmin(np.abs(t_s[cand] - args.profile_time))])
            nb = [g for g in (f_lab + step, f_lab - step) if 0 <= g < T and lab_frac[g] == 0]
            f_unl = int(nb[0]) if nb else None
        else:
            # continuity pair: a well-fitted native frame and the native frame
            # `step` frames (10 ms for step 2) before or after it that has no
            # fit at all (beam switched off or fit failed) -- the plasma barely
            # changes in between, so the two profiles must agree
            pairs = []
            for f in np.where(cover & native)[0]:
                for g in (f + step, f - step):
                    if 0 <= g < T and lab_frac[g] == 0:
                        pairs.append((med_ti[f], f, g))
            if pairs:
                _, f_lab, f_unl = max(pairs)
                f_lab, f_unl = int(f_lab), int(f_unl)
        if f_unl is None and not args.no_unl:
            print(
                "warning: no fitted frame with a native no-fit neighbour; "
                "showing the fitted profile only"
            )

    tr_chords = [int(x) for x in args.trace_chords.split(",")][:2]

    gate = None
    if args.gate == "info" and "pred_sigma" in d:
        # information gate: the model's prediction counts as a measurement
        # only where its own predicted sigma beats the prior spread of the
        # target. Frame-level: median over chords, both targets must pass.
        psig = d["pred_sigma"].reshape(C, T, -1) * sc
        ok = np.ones(T, bool)
        gate_targets = [1] if args.vtor_only else range(y.shape[-1])
        for t in gate_targets:
            prior = np.nanstd(y[..., t])
            ok &= np.median(psig[..., t], axis=0) < prior
        k = 5
        gate = np.convolve(ok.astype(int), np.ones(k, int), "same") >= k
        if gate.sum() < k:
            # the model's own sigma never beats the prior (e.g. a target it
            # cannot read on this machine): fall back to the emission gate
            print("info gate empty -> using the emission (amplitude) gate")
            gate = None

    spec = None
    if args.shot is not None:
        from joblib import load

        s = load(data_path(args.shot), mmap_mode="r")
        end = int(s["end_index"])
        if end <= 0:  # <= 0 means the whole recording
            end = s["input"].shape[1]
        spec = {
            cc: np.asarray(s["input"][cc, :end, :], dtype=np.float32)
            for cc in tr_chords
        }
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

    gm = (gate if gate is not None else np.ones(T, bool))[:T]
    xm, xlab = chord_coordinate(C, args.profile_x, args.profile_xlabel)
    order, xs = sorted_chords(xm)
    models = [(pr, ps, col1)] + ([(pr2, ps2, col2)] if pr2 is not None else [])

    def chord_name(cc):
        """Identifier of a chord in the panel label."""
        if args.profile_x is not None and np.isfinite(xm[cc]):
            sym = (args.profile_xlabel or style.RTAN_LABEL).split("(")[0].strip()
            unit = "m" if "(m)" in (args.profile_xlabel or "(m)") else ""
            return f"{sym} = {xm[cc]:.2f} {unit}".rstrip()
        return f"chord {cc}"

    # time-axis range from the validity gate
    t0 = t_s[int(np.argmax(gate))] if (gate is not None and gate.any()) else t_s[0]
    t1 = (
        t_s[len(gate) - 1 - int(np.argmax(gate[::-1]))]
        if (gate is not None and gate.any())
        else t_s[-1]
    )
    if args.t_start is not None:
        t0 = args.t_start  # fixed start of the time axis (default 0 s)
    # the profile time(s) must lie inside the shown time range
    t_prof = [t_s[f_lab]] + ([t_s[f_unl]] if f_unl is not None else [])
    t1 = max(t1, max(t_prof) + 0.01)

    # trace y-limits per target, shared by the chords: from the model curves and
    # fits inside the gate (not from the bands)
    tr_ylim = {}
    for k in (1, 0):
        vals = []
        for cc in tr_chords:
            for mu_, _, _ in models:
                vals.append(mu_[cc, gm, k])
            vals.append(y[cc, gm, k])
        v = np.concatenate([x_[np.isfinite(x_)] for x_ in vals])
        if v.size:
            lo, hi = np.percentile(v, [1, 99])
            pad = 0.12 * max(hi - lo, 1e-6)
            tr_ylim[k] = (lo - pad, hi + pad)

    # layout: per row the cells [(kind, chord, target)]
    if args.vtor_only:
        rows = [
            ([("spec", tr_chords[0], None)] if spec is not None else [])
            + [("trace", cc, 1) for cc in tr_chords]
            + [("prof", None, 1)]
        ]
    else:
        rows = [
            ([("spec", cc, None)] if spec is not None else [])
            + [("trace", cc, 1), ("trace", cc, 0), ("prof", None, k_)]
            for cc, k_ in zip(tr_chords, (1, 0))
        ]
    ncols = len(rows[0])
    wr = ([0.85] if spec is not None else []) + [1.0] * (ncols - int(spec is not None))
    wr[-1] = 1.1  # profile column

    # line and marker sizes per output (the paper PDF is 6.5 in wide, so
    # the dense time traces need thinner lines and smaller points)
    sz = {}

    def trace_panel(ax, cc, k):
        """Time trace of one chord and target: models with bands, fits."""
        for mu_, ps_, c_ in models:
            if ps_ is not None:
                ax.fill_between(
                    t_s,
                    mu_[cc, :, k] - ps_[cc, :, k],
                    mu_[cc, :, k] + ps_[cc, :, k],
                    color=c_,
                    alpha=style.BAND_ALPHA,
                    lw=0,
                    zorder=1,
                )
            ax.plot(t_s, mu_[cc, :, k], "-", color=c_, lw=sz["lw_t"], zorder=2)
        m = np.isfinite(y[cc, :, k])
        style.fit_points(ax, t_s[m], y[cc, m, k], sl[cc, m, k], ms=sz["ms_t"],
                         elinewidth=sz["elw"])
        ax.axvline(t_s[f_lab], ls=LS_LAB, **T_LINE)
        if f_unl is not None:
            ax.axvline(t_s[f_unl], ls=LS_UNL, **T_LINE)
        if k in tr_ylim:
            ax.set_ylim(*tr_ylim[k])
        ax.set_xlim(t0, t1)
        ax.set_ylabel(style.TARGET_LABEL["ti" if k == 0 else "vtor"])

    def profile_panel(ax, k):
        """Profile at the fitted (and fit-free) frame for one target."""
        fitm = np.isfinite(y[order, f_lab, k])
        nof = ~fitm
        nof_ext = nof | np.r_[nof[1:], False] | np.r_[False, nof[:-1]]
        curves = []
        for mu_, ps_, c_ in models:
            if ps_ is not None:
                # band; lighter where no fit exists (large uncertainty)
                lo_b = mu_[order, f_lab, k] - ps_[order, f_lab, k]
                hi_b = mu_[order, f_lab, k] + ps_[order, f_lab, k]
                ax.fill_between(xs, lo_b, hi_b, where=fitm, color=c_,
                                alpha=style.BAND_ALPHA, lw=0, zorder=1)
                ax.fill_between(xs, lo_b, hi_b, where=nof_ext, color=c_,
                                alpha=0.07, lw=0, zorder=1)
            ax.plot(xs, mu_[order, f_lab, k], "-", color=c_, lw=sz["lw_p"], zorder=2)
            curves.append(mu_[order, f_lab, k])
        if f_unl is not None:
            ax.plot(xs, pr[order, f_unl, k], "--", color=col1, lw=sz["lw_p"], zorder=2.2)
            curves.append(pr[order, f_unl, k])
        style.fit_points(ax, xs[fitm], y[order, f_lab, k][fitm],
                         sl[order, f_lab, k][fitm], ms=sz["ms_p"], elinewidth=sz["elw"])
        dx = np.diff(xs)
        pad = 0.5 * (float(np.median(dx)) if dx.size else 0.5)
        fitx = xs[fitm]
        xl = (fitx.min() - pad, fitx.max() + pad) if fitx.size >= 2 else (xs[0] - pad, xs[-1] + pad)
        ax.set_xlim(*xl)
        # y-limits from the curves and fits inside the shown x-range
        inx = (xs >= xl[0]) & (xs <= xl[1])
        # (a single fit far outside the model curves, e.g. a failed edge fit,
        # is left outside the axes instead of compressing the profile)
        cv = np.concatenate([c_[inx] for c_ in curves])
        cv = cv[np.isfinite(cv)]
        fv = y[order, f_lab, k][fitm & inx]
        if cv.size:
            span = max(cv.max() - cv.min(), 1e-6)
            fv = fv[(fv > cv.min() - 0.25 * span) & (fv < cv.max() + 0.25 * span)]
        vals = np.concatenate([cv, fv])
        if vals.size:
            lo, hi = vals.min(), vals.max()
            padv = 0.12 * max(hi - lo, 1e-6)
            ax.set_ylim(lo - padv, hi + padv)
        ax.set_ylabel(style.TARGET_LABEL["ti" if k == 0 else "vtor"])

    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            if ctx == "paper":
                sz.update(lw_t=0.7, ms_t=0.9, lw_p=1.0, ms_p=1.6, elw=0.4)
            else:
                sz.update(lw_t=1.1, ms_t=1.6, lw_p=1.3, ms_p=2.0, elw=0.5)
            nr = len(rows)
            if ctx == "paper":
                size = (PAPER_W, 2.5) if args.vtor_only else (PAPER_W, 4.0)
            else:
                size = (18, 5.2) if args.vtor_only else (18, 9)
            fig = plt.figure(figsize=size, layout="constrained")
            fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.04, hspace=0.06, rect=(0, 0, 0.985, 1))
            gs = fig.add_gridspec(nr, ncols, width_ratios=wr)
            axes = np.empty((nr, ncols), dtype=object)
            letters = iter("abcdefghijklmnop")
            first_col_ax = {}
            for r, cells in enumerate(rows):
                for cix, (kind, cc, k) in enumerate(cells):
                    share = first_col_ax.get(cix)
                    ax = fig.add_subplot(gs[r, cix], sharex=share)
                    first_col_ax.setdefault(cix, ax)
                    axes[r, cix] = ax
                    if kind == "spec":
                        sp = spec[cc]
                        z = np.log10(np.clip(sp - sp.min() + 1, 1, None)).T
                        ax.imshow(
                            z,
                            aspect="auto",
                            origin="lower",
                            cmap="magma",
                            extent=[t_s[0], t_s[min(len(sp) - 1, T - 1)], 0, z.shape[0]],
                        )
                        ax.set_ylabel("wavelength bin")
                        ax.set_xlim(t0, t1)
                        panel_label(ax, next(letters), chord_name(cc))
                    elif kind == "trace":
                        trace_panel(ax, cc, k)
                        panel_label(ax, next(letters), chord_name(cc))
                    else:
                        profile_panel(ax, k)
                        panel_label(ax, next(letters), f"t = {t_s[f_lab]:.3f} s")
                    last = r == nr - 1
                    if last:
                        ax.set_xlabel(xlab if kind == "prof" else style.TIME_LABEL)
                    else:
                        plt.setp(ax.get_xticklabels(), visible=False)

            # one legend for the figure, fixed order: fit, models by role, then
            # the no-fit profile and the profile-time markers
            fit_h = Line2D([], [], color=style.COLORS["fit"], marker="o", ls="", ms=3)
            rank = {r: i for i, r in enumerate(style.LEGEND_ORDER)}
            mods = [(rank.get(style.role_of(args.label1), 9), col1, args.label1)]
            if pr2 is not None:
                mods.append((rank.get(style.role_of(args.label2), 10), col2, args.label2))
            hh, ll = [fit_h], [style.NAMES["fit"]]
            for _, c_, lab in sorted(mods, key=lambda m_: m_[0]):
                ln = Line2D([], [], color=c_, lw=1.4)
                hh.append((Patch(color=c_, alpha=style.BAND_ALPHA, lw=0), ln) if ps is not None else ln)
                ll.append(lab)
            hh.append(Line2D([], [], ls=LS_LAB, **T_LINE))
            ll.append(f"profile time, t = {t_s[f_lab]:.3f} s")
            if f_unl is not None:
                hh.append(Line2D([], [], color=col1, ls="--", lw=1.2))
                ll.append(f"{args.label1}, t = {t_s[f_unl]:.3f} s (no conventional fit)")
                hh.append(Line2D([], [], ls=LS_UNL, **T_LINE))
                ll.append(f"profile time without fit, t = {t_s[f_unl]:.3f} s")
            # rows read left to right (matplotlib fills the columns first)
            ncol = 3 if len(hh) > 4 else len(hh)
            nrow = -(-len(hh) // ncol)
            idx = [r_ * ncol + c_ for c_ in range(ncol) for r_ in range(nrow)
                   if r_ * ncol + c_ < len(hh)]
            if nrow * ncol != len(hh):  # pad incomplete last row with empty entries
                hh += [Line2D([], [], ls="")] * (nrow * ncol - len(hh))
                ll += [""] * (nrow * ncol - len(ll))
                idx = [r_ * ncol + c_ for c_ in range(ncol) for r_ in range(nrow)]
            fig.legend(
                [hh[i] for i in idx],
                [ll[i] for i in idx],
                loc="outside upper center",
                ncol=ncol,
                fontsize="x-small" if ctx == "paper" else "small",
                frameon=False,
                handlelength=2.0,
            )
            fig.savefig(
                args.out / f"{args.out_name}.{ext}", dpi=300, bbox_inches=bbox(ext)
            )
            plt.close(fig)
    print(
        f"wrote {args.out}/{args.out_name}.png/.pdf | "
        f"labeled frame t={t_s[f_lab]:.3f}s"
        + (f", unlabeled frame t={t_s[f_unl]:.3f}s" if f_unl is not None else "")
    )


if __name__ == "__main__":
    main()
