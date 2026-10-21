r"""Momentum-transport events in one figure.

One row per event and target (--targets, e.g. "vtor ti" "vtor"); columns:
equilibrium with the chord tangency radii (spanning the rows of its event) |
profile at t0 (before) | during | after | time trace of the core chord
with the three profile times marked. Later profile panels repeat the model
profile at t0 as a grey dashed reference. Conventions of
cer_transfer.figures.style (fit black, model in the colour of its role with a
+-1 sigma band, legend once for the figure, panel letters a-e / f-j with the
identifier after the letter, no titles).

Every per-event option takes one value per event (row order), e.g.

    pixi run python -u -m cer_transfer.figures.events_merged \
        --preds gallery/e116939.npz gallery/e115600.npz \
        --t0 0.370 0.315 --t1 0.445 0.400 --trace-x 1.10 1.00 \
        --times "0.376 0.406 0.426" "0.315 0.365 0.395" --min-fit-frac 0.5 0.7 \
        --targets "vtor" "vtor ti" --shot chers_set5/chers_116939.joblib chers_set5/chers_115600.joblib \
        --efit $CER_DATA_ROOT/chers_set5/efit_116939.joblib \
               $CER_DATA_ROOT/chers_set5/efit_115600.joblib \
        --profile-x figs/chords_Rtan_116939.csv figs/chords_Rtan_115600.csv \
        --profile-xlabel '$R_\mathrm{tan}$ (m)' --t-offset -0.235 --k 0.734 0.817 \
        --label1 "DIII-D+NSTX (7307 discharges) model" --out figs --out-name events_merged
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import load
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from cer_transfer.figures import style
from cer_transfer.figures.common import (
    FRAME_HZ,
    chord_coordinate,
    load_predictions,
    sorted_chords,
)
from cer_transfer.figures.event_panels import model_handle, model_roles, panel_label
from cer_transfer.figures.paperfig import PAPER_W, bbox
from cer_transfer.figures.timeaxis import file_times

sns.set_style("whitegrid")

K_IDX = {"ti": 0, "vtor": 1}  # target index in the dumps
T_LINE = dict(color="0.3", lw=1.0, ls=":", zorder=0.5)  # profile times in the trace
REF = dict(color="0.55", ls="--", lw=1.0, zorder=1.5)  # model at t0
EFIT_KEYS = ("time_s", "psirz", "rgrid", "zgrid", "psi_axis", "psi_bdry", "r_axis",
             "z_axis", "rbbbs", "zbbbs", "nbbbs", "rlim", "zlim")


def per_event(vals, n, name, cast=float):
    """One value per event; a single value is used for every event."""
    if vals is None:
        return [None] * n
    if len(vals) == 1:
        vals = list(vals) * n
    if len(vals) != n:
        raise SystemExit(f"--{name}: give one value per event ({n})")
    return [None if v in ("", "none", "None") else cast(v) for v in vals]


def prepare(ev, args, tgt="vtor", ref=None):
    """Load one event and target: arrays, time axis, shown frames and chords.

    With ref (the first target's result) the chords, profile frames and trace
    chord are taken from it, so all rows of an event show the same chords."""
    dmp = load_predictions(ev["preds"])
    C, T = dmp.C, dmp.T
    ki = K_IDX[tgt]
    kk = 1.0 if args.k is None else args.k[ki]
    sc = style.TARGET_SCALE[tgt]
    y = dmp.y[..., ki] * sc
    sl = dmp.sigma[..., ki] * sc
    mu = dmp.pred[..., ki] * sc
    ps = dmp.pred_sigma[..., ki] * kk * sc if dmp.pred_sigma is not None else None
    tax = (file_times(ev["shot"], T, args.fs, args.t_offset) if ev["shot"]
           else np.arange(T) / args.fs + args.t_offset)
    xm, xlab = chord_coordinate(C, ev["profile_x"], args.profile_xlabel)
    order, xs = sorted_chords(xm)
    f0 = int(np.argmin(np.abs(tax - ev["t0"])))
    f1 = int(np.argmin(np.abs(tax - ev["t1"])))
    fitted = [f for f in range(max(f0, 0), min(f1, T - 1) + 1)
              if np.isfinite(y[:, f]).sum() >= 5]
    if not fitted:
        raise SystemExit(f"{ev['preds']}: no fitted frames in the window")
    # chords with a conventional fit in at least min-fit-frac of the fitted frames
    frac = np.isfinite(y[:, fitted]).mean(1)
    keep = frac >= ev["min_fit_frac"]
    if keep.sum() >= 5:
        order = [c for c in order if keep[c]]
        xs = np.asarray([xm[c] for c in order])
    else:
        print("warning: --min-fit-frac keeps fewer than 5 chords; drawing all")
    if ev["times"]:
        frames = [min(fitted, key=lambda f: abs(tax[f] - tt)) for tt in ev["times"]]
    else:
        idx = np.linspace(0, len(fitted) - 1, 3).round().astype(int)
        frames = [fitted[i] for i in idx]
    c_tr = order[int(np.argmin(np.abs(xs - ev["trace_x"])))]
    if ref is not None:
        order, xs, frames, c_tr = ref["order"], ref["xs"], ref["frames"], ref["c_tr"]
    fr = np.arange(max(f0, 0), min(f1, T - 1) + 1)
    # profile y-limits from the fits and the model at fitted chords (not the
    # bands; the model may leave the axes where no conventional fit exists),
    # shared in the row
    vals = np.concatenate([
        np.r_[mu[order, f][np.isfinite(y[order, f])], y[order, f]] for f in frames])
    vals = vals[np.isfinite(vals)]
    lo, hi = vals.min(), vals.max()
    pad = 0.1 * max(hi - lo, 1e-6)
    ylim = (lo - pad, hi + pad) if ev["ylim"] is None or tgt != "vtor" else tuple(ev["ylim"])
    tv = np.r_[mu[c_tr, fr], y[c_tr, fr]]
    tv = tv[np.isfinite(tv)]
    tpad = 0.1 * max(tv.max() - tv.min(), 1e-6)
    efit = None
    if ev["efit"] is not None and Path(ev["efit"]).exists():
        ef = load(ev["efit"])
        efit = {kq: np.asarray(ef[kq]) for kq in EFIT_KEYS}
    m_ = re.search(r"(\d{6})", str(ev["shot"] or ev["preds"]))
    return dict(tgt=tgt, y=y, sl=sl, mu=mu, ps=ps, tax=tax, xm=xm, xlab=xlab, order=order,
                xs=xs, frames=frames, c_tr=c_tr, fr=fr, ylim=ylim,
                tylim=(tv.min() - tpad, tv.max() + tpad), efit=efit,
                shot=m_.group(1) if m_ else None)


def draw_equilibrium(ax, e, fs):
    """Flux surfaces, boundary, limiter, chord tangency radii at t0."""
    efit, f = e["efit"], e["frames"][0]
    ie = int(np.argmin(np.abs(efit["time_s"] - e["tax"][f])))
    psin = (efit["psirz"][ie] - efit["psi_axis"][ie]) / (
        efit["psi_bdry"][ie] - efit["psi_axis"][ie])
    R, Z = np.meshgrid(efit["rgrid"], efit["zgrid"])
    ax.contour(R, Z, psin, levels=[0.1, 0.25, 0.4, 0.55, 0.7, 0.85], cmap="viridis",
               linewidths=0.5, vmin=0.0, vmax=1.2)
    nb = int(efit["nbbbs"][ie])
    ax.plot(efit["rbbbs"][ie, :nb], efit["zbbbs"][ie, :nb], "-", color="k", lw=0.8)
    ax.plot(efit["rlim"], efit["zlim"], "-", color="0.5", lw=0.8)
    ax.plot(efit["r_axis"][ie], efit["z_axis"][ie], "+", color="k", ms=5, mew=1.0)
    ax.plot(e["xs"], np.zeros_like(e["xs"]), "|", color="0.25", ms=4, mew=0.6, zorder=4)
    ax.plot([e["xm"][e["c_tr"]]], [0.0], "o", mfc="white", mec="k", ms=3.5, mew=0.9,
            zorder=5)
    ax.set_xlim(float(efit["rlim"].min()) - 0.03, float(efit["rlim"].max()) + 0.03)
    ax.set_ylim(float(efit["zlim"].min()) - 0.03, float(efit["zlim"].max()) + 0.03)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0.5, 1.0, 1.5])
    ax.set_xticklabels(["0.5", "1", "1.5"])
    ax.tick_params(labelsize=fs)
    ax.set_xlabel("$R$ (m)", fontsize=fs)
    ax.set_ylabel("$Z$ (m)", fontsize=fs, labelpad=1)
    return efit["time_s"][ie]


def draw_profile(ax, e, j, col, band, lw, ms):
    """Profile at the j-th time: no-fit spans, model band, reference, fits."""
    xs, order, f = e["xs"], e["order"], e["frames"][j]
    dx = np.diff(xs)
    step = float(np.median(dx)) if dx.size else 0.5
    yy = e["y"][order, f]
    fitm = np.isfinite(yy)
    nf = ~fitm
    a = None
    for q, flag in enumerate(list(nf) + [False]):  # grey: no conventional fit
        if flag and a is None:
            a = q
        if not flag and a is not None:
            b_ = q - 1
            lo_x = xs[a] - 0.5 * ((xs[a] - xs[a - 1]) if a > 0 else step)
            hi_x = xs[b_] + 0.5 * ((xs[b_ + 1] - xs[b_]) if b_ + 1 < len(xs) else step)
            ax.axvspan(lo_x, hi_x, color="0.9", zorder=0, lw=0)
            a = None
    m_ = e["mu"][order, f]
    if band and e["ps"] is not None:
        s_ = e["ps"][order, f]
        nof_ext = nf | np.r_[nf[1:], False] | np.r_[False, nf[:-1]]
        ax.fill_between(xs, m_ - s_, m_ + s_, where=~nf, color=col,
                        alpha=style.BAND_ALPHA, lw=0, zorder=1)
        ax.fill_between(xs, m_ - s_, m_ + s_, where=nof_ext, color=col, alpha=0.07,
                        lw=0, zorder=1)
    if j > 0:
        ax.plot(xs, e["mu"][order, e["frames"][0]], **REF)
    ax.plot(xs, m_, "-", color=col, lw=lw, zorder=2)
    style.fit_points(ax, xs[fitm], yy[fitm], e["sl"][order, f][fitm], ms=ms,
                     elinewidth=0.5)
    ax.set_xlim(xs[0] - 0.5 * step, xs[-1] + 0.5 * step)
    ax.set_ylim(*e["ylim"])


def draw_trace(ax, e, col, band, lw, ms):
    """Core-chord trace over the window, profile times marked."""
    c, fr, tax = e["c_tr"], e["fr"], e["tax"]
    tt = tax[fr]
    for f in e["frames"]:
        ax.axvline(tax[f], **T_LINE)
    m_t = e["mu"][c, fr]
    if band and e["ps"] is not None:
        ax.fill_between(tt, m_t - e["ps"][c, fr], m_t + e["ps"][c, fr], color=col,
                        alpha=style.BAND_ALPHA, lw=0, zorder=1)
    ax.plot(tt, m_t, "-", color=col, lw=lw, marker="o", ms=0.7 * ms, zorder=2)
    yt = e["y"][c, fr]
    fm = np.isfinite(yt)
    style.fit_points(ax, tt[fm], yt[fm], e["sl"][c, fr][fm], ms=ms, elinewidth=0.5)
    ax.set_xlim(tt[0], tt[-1])
    ax.set_ylim(*e["tylim"])
    ax.xaxis.set_major_locator(MaxNLocator(3))


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds", type=Path, nargs="+", required=True,
                   help="prediction dump per event (one row each, in this order)")
    p.add_argument("--t0", nargs="+", required=True, help="window start per event (s)")
    p.add_argument("--t1", nargs="+", required=True, help="window end per event (s)")
    p.add_argument("--times", nargs="+", default=None,
                   help='three profile times per event as one quoted string, e.g. '
                   '"0.376 0.406 0.426"; nearest fitted frames are used')
    p.add_argument("--trace-x", nargs="+", required=True,
                   help="chord coordinate of the time trace per event")
    p.add_argument("--min-fit-frac", nargs="+", default=["0.5"],
                   help="per event: draw chords with a fit in at least this fraction "
                   "of the window's fitted frames")
    p.add_argument("--ylim", nargs="+", default=None,
                   help='per event: fixed v_tor range of the profiles as "lo hi", or ""')
    p.add_argument("--targets", nargs="+", default=["vtor"],
                   help='per event: targets shown as rows, e.g. "vtor ti" (ti, vtor)')
    p.add_argument("--shot", nargs="+", default=None,
                   help="discharge file per event (time axis from its time_s)")
    p.add_argument("--efit", nargs="+", default=None, help="EFIT joblib per event")
    p.add_argument("--profile-x", nargs="+", default=None,
                   help="chord,x CSV per event (e.g. chords_Rtan_<shot>.csv)")
    p.add_argument("--profile-xlabel", default=style.RTAN_LABEL)
    p.add_argument("--t-offset", type=float, default=0.0)
    p.add_argument("--fs", type=float, default=FRAME_HZ, help="frame rate (Hz)")
    p.add_argument("--k", type=float, nargs=2, default=None,
                   help="calibration factors (T_i, v_tor) of the model sigma")
    p.add_argument("--label1", default=style.NAMES["finetuned"],
                   help="legend text of the model (its role sets the colour)")
    p.add_argument("--no-band", action="store_true", help="omit the model bands")
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", default="events_merged")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    n = len(args.preds)
    times = per_event(args.times, n, "times",
                      cast=lambda s: [float(x) for x in s.replace(",", " ").split()])
    ylims = per_event(args.ylim, n, "ylim",
                      cast=lambda s: [float(x) for x in s.replace(",", " ").split()])
    events = [
        dict(preds=args.preds[i], t0=t0, t1=t1, times=times[i], trace_x=tx,
             min_fit_frac=mf, ylim=ylims[i], shot=sh, efit=ef, profile_x=px)
        for i, (t0, t1, tx, mf, sh, ef, px) in enumerate(zip(
            per_event(args.t0, n, "t0"), per_event(args.t1, n, "t1"),
            per_event(args.trace_x, n, "trace-x"),
            per_event(args.min_fit_frac, n, "min-fit-frac"),
            per_event(args.shot, n, "shot", cast=str),
            per_event(args.efit, n, "efit", cast=str),
            per_event(args.profile_x, n, "profile-x", cast=Path)))
    ]
    tgts = per_event(args.targets, n, "targets", cast=lambda s: s.replace(",", " ").split())
    rows = []  # (event index, prepared target), in row order
    for i, ev in enumerate(events):
        first = prepare(ev, args, tgts[i][0])
        rows.append((i, first))
        rows += [(i, prepare(ev, args, t, ref=first)) for t in tgts[i][1:]]
    E = [e for _, e in rows]
    nr = len(rows)
    role, _ = model_roles(args.label1, None)
    col = style.COLORS[role]
    band = not args.no_band
    eq = all(e["efit"] is not None for e in E)
    sym = args.profile_xlabel.split("(")[0].strip()

    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            paper = ctx == "paper"
            fs = 7 if paper else 13  # small text: ticks of the equilibrium, labels
            lw, ms = (0.9, 2.0) if paper else (1.5, 3.0)
            size = (PAPER_W, 1.25 + 1.75 * nr) if paper else (19, 2.2 + 4.2 * nr)
            fig = plt.figure(figsize=size, layout="constrained")
            fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.02,
                                        hspace=0.08, rect=(0, 0, 0.99, 1))
            wr = ([0.62] if eq else []) + [1, 1, 1, 1.25]
            gs = fig.add_gridspec(nr, len(wr), width_ratios=wr)
            c0 = 1 if eq else 0
            letters = iter("abcdefghijklmnopqrstuvwxyz")
            lab_kw = dict(fontsize=fs + 1 if paper else "small")
            for r, (i, e) in enumerate(rows):
                first_row = r == 0 or rows[r - 1][0] != i
                if eq and first_row:
                    nri = sum(1 for q, _ in rows if q == i)
                    axe = fig.add_subplot(gs[r:r + nri, 0])
                    draw_equilibrium(axe, e, fs)
                    panel_label(axe, next(letters), e["shot"], **lab_kw)
                prof = []
                for j in range(3):
                    ax = fig.add_subplot(gs[r, c0 + j], sharey=prof[0] if prof else None)
                    prof.append(ax)
                    draw_profile(ax, e, j, col, band, lw, ms)
                    tl = "$t_0$" if j == 0 else "t"
                    panel_label(ax, next(letters), f"{tl} = {e['tax'][e['frames'][j]]:.3f} s",
                                **lab_kw)
                    ax.xaxis.set_major_locator(MaxNLocator(3))
                    ax.set_xlabel(e["xlab"])
                    if j:
                        plt.setp(ax.get_yticklabels(), visible=False)
                    else:
                        ax.set_ylabel(style.TARGET_LABEL[e["tgt"]])
                axt = fig.add_subplot(gs[r, c0 + 3])
                draw_trace(axt, e, col, band, lw, ms)
                ident = (f"{sym} = {e['xm'][e['c_tr']]:.2f} m"
                         if events[i]["profile_x"] is not None else f"chord {e['c_tr']}")
                panel_label(axt, next(letters), ident, **lab_kw)
                axt.yaxis.tick_right()
                axt.yaxis.set_label_position("right")
                axt.set_ylabel(style.TARGET_LABEL[e["tgt"]])
                axt.set_xlabel(style.TIME_LABEL)
            hh = [Line2D([], [], color=style.COLORS["fit"], marker="o", ls="", ms=3),
                  model_handle(col, band and E[0]["ps"] is not None),
                  Line2D([], [], **REF), Line2D([], [], **T_LINE)]
            ll = [style.NAMES["fit"], args.label1, "model at $t_0$", "profile times"]
            fig.legend(hh, ll, loc="outside upper center", ncol=len(hh), frameon=False,
                       fontsize=fs + 0.5 if paper else "small", handlelength=2.2,
                       columnspacing=1.2)
            fig.savefig(args.out / f"{args.out_name}.{ext}", dpi=300, bbox_inches=bbox(ext))
            plt.close(fig)
    for e in E:
        print(f"  {e['shot']} {e['tgt']}: profile times "
              + ", ".join(f"{e['tax'][f]:.3f}" for f in e["frames"])
              + f" s, trace chord {e['c_tr']} ({sym} = {e['xm'][e['c_tr']]:.2f} m)")
    print(f"wrote {args.out}/{args.out_name}.png/.pdf")


if __name__ == "__main__":
    main()
