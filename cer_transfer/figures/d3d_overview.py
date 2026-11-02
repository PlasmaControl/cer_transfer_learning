r"""DIII-D overview: the DIII-D model on the DIII-D validation set and three discharges.

Top row: (a, b) reconstruction vs. target T_i and v_tor on the consistent labels of the
validation set (2D histogram in the model colour, opacity = log count, bins with fewer
than ``--cmin`` labels not drawn; R2 and median error inset); (c) R2 per chord on the
consistent labels, both targets, vs. the major radius where the chord crosses the beam
(``--chords``; chords without a radius are left out).
Bottom row, one column per validation discharge: v_tor trace over T_i trace of one
chord with clean targets (``clean_chord``: >= 95 % consistent labels, physically
possible values, signal well above the fit uncertainty), over the whole span of its
fits (conventional fits black with error bars, model with +-1 sigma band):

(d) the median-R2 discharge among the high-rotation validation discharges (typical,
    not the best one);
(e) 190570, the low-rotation validation discharge: v_tor only (its T_i fits are
    corrupt and not scored);
(f) a validation discharge with fits without gaps (``--gapless``).

    pixi run python -m cer_transfer.figures.d3d_overview \
        --preds gallery/d3d/d3d_heldout.npz --lengths gallery/d3d/heldout_file_lengths.npz \
        --chords figs/chords_R_d3d_tangential.csv --low-val 190570 --gapless 174074 --out figs
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cer_transfer.analysis.coverage import label_classes
from cer_transfer.figures import style
from cer_transfer.figures.common import load_predictions
from cer_transfer.figures.paperfig import PAPER_W, bbox
from cer_transfer.figures.plots import _draw_flaghist, _r2

sns.set_style("whitegrid")
FS = 200.0  # frames per second (frame i = 5 ms * i)
TARGETS = ("ti", "vtor")


def split_dump(d, lengths):
    """Per-file views (C, T_f, 2) of a concatenated dump: list of dicts."""
    out, s0 = [], 0
    for T in lengths:
        out.append({k: getattr(d, k)[:, s0:s0 + T] for k in ("y", "pred", "sigma", "pred_sigma")})
        s0 += T
    if s0 != d.T:
        raise SystemExit(f"lengths cover {s0} frames, the dump has {d.T}")
    return out


def per_file_stats(f):
    """R2 per target on all labels and the rotation level of one discharge."""
    st = {}
    for t, name in enumerate(TARGETS):
        y, p = f["y"][..., t], f["pred"][..., t]
        ok = np.isfinite(y) & np.isfinite(p)
        st[name] = _r2(p[ok], y[ok]) if ok.sum() > 200 else np.nan
    v = f["y"][..., 1]
    st["vt95"] = float(np.nanpercentile(np.abs(v), 95)) if np.isfinite(v).any() else 0.0
    st["dur"] = float(np.isfinite(v).any(0).sum() / FS)
    return st


def chord_quality(f, c, targets=(0, 1)):
    """(clean, n_labels): clean = >= 95 % consistent labels for every target, physically
    possible values (0 < T_i < 20 keV, |v_tor| < 500 km/s), a signal well above the
    fit uncertainty (std of the fits > 3x their median sigma) and smooth fits (lag-1
    autocorrelation of successive fits >= 0.8; fits alternating between two levels fail)."""
    y, sg = f["y"][c], f["sigma"][c]
    lab = np.ones(y.shape[0], bool)
    for t in targets:
        lab &= np.isfinite(y[:, t])
    if lab.sum() < 50:
        return False, 0
    for t in targets:
        v, e = y[lab, t], sg[lab, t]
        if (label_classes(y[None, :, t])[0][lab] == 0).mean() < 0.95:
            return False, 0
        if t == 0 and not (v.min() > 0 and v.max() < 20000):
            return False, 0
        if t == 1 and np.abs(v).max() >= 500:
            return False, 0
        if np.std(v) < 3 * np.median(e):
            return False, 0
        if np.corrcoef(v[:-1], v[1:])[0, 1] < 0.8:
            return False, 0
    return True, int(lab.sum())


def clean_chord(f, targets=(0, 1), prefer=None):
    """Chord with clean targets (``chord_quality``) and the most labels; ``prefer`` lists
    chords in order of preference (e.g. core to edge). No clean chord: error."""
    order = list(prefer) if prefer is not None else list(range(f["y"].shape[0]))
    q = [(c, *chord_quality(f, c, targets)) for c in order]
    ok = [(c, n) for c, good, n in q if good]
    if not ok:
        raise SystemExit("no chord with clean targets")
    nmax = max(n for _, n in ok)
    return next(c for c, n in ok if n >= 0.8 * nmax)  # most central among the well-labelled ones


def window(f, c):
    """Frames of the longest uninterrupted run of fits of chord c (gaps of at most twice
    the nominal spacing), with a short margin."""
    fr = np.where(np.isfinite(f["y"][c, :, 0]) & np.isfinite(f["y"][c, :, 1]))[0]
    g = np.diff(fr)
    brk = np.where(g > 2 * np.median(g))[0]
    bounds = np.r_[-1, brk, fr.size - 1]
    j = int(np.argmax(np.diff(bounds)))
    a, b = fr[bounds[j] + 1], fr[bounds[j + 1]]
    return max(a - 10, 0), min(b + 10, f["y"].shape[1] - 1)


def chord_score(f, c, a, b):
    """Lower of the two R2 (T_i, v_tor) of the model on the fits of chord c in frames a..b."""
    out = []
    for t in (0, 1):
        y, p = f["y"][c, a:b + 1, t], f["pred"][c, a:b + 1, t]
        m = np.isfinite(y) & np.isfinite(p)
        out.append(_r2(p[m], y[m]) if m.sum() > 50 else -np.inf)
    return min(out)


def beam_off_spans(f, a, b, min_frames=3):
    """Runs of frames in [a, b] without a fit on any chord (beam off), as (t0, t1) pairs."""
    off = ~np.isfinite(f["y"][:, a:b + 1, 1]).any(0)
    spans, start = [], None
    for k, o in enumerate(list(off) + [False]):
        if o and start is None:
            start = k
        if not o and start is not None:
            if k - start >= min_frames:
                spans.append(((a + start - 0.5) / FS, (a + k - 0.5) / FS))
            start = None
    return spans


def beam_on(spec, f, c):
    """Frames at which chord c sees the active line (beam on): line brightness above half
    of its median at the frames with a fit."""
    b = spec[c].max(-1) - np.median(spec[c], -1)
    lab = np.isfinite(f["y"][c, :, 1])
    return b > 0.5 * np.median(b[lab])


def draw_trace(ax, f, c, t, sc, a, b, ms, lw, on_all=None, tmark=None):
    """Fits (black) and model (green, +-1 sigma) of chord c, target t, frames a..b; the
    model only at the beam-on frames ``on_all`` of the chord."""
    tt = np.arange(a, b + 1) / FS
    sl = slice(a, b + 1)
    # the model at the beam-on frames of the chord: the beam is modulated (on/off every
    # few frames), and between the pulses there is no active line to reconstruct
    on = on_all[sl] if on_all is not None else np.ones(b - a + 1, bool)
    y = f["y"][c, sl, t]
    m = np.isfinite(y)
    style.fit_points(ax, tt[m], y[m] * sc, f["sigma"][c, sl, t][m] * sc, ms=0.6 * ms, elinewidth=0.4)
    style.model_line(ax, tt[on], f["pred"][c, sl, t][on] * sc, f["pred_sigma"][c, sl, t][on] * sc,
                     role="zero_shot", marker="", lw=lw, zorder=6)
    if tmark is not None:
        ax.axvline(tmark, color="0.35", lw=0.9, ls="--", zorder=7)
    ax.set_xlim(tt[0], tt[-1])
    yv = np.r_[y[m], f["pred"][c, sl, t][on]] * sc
    yv = yv[np.isfinite(yv)]
    if yv.size:
        lo, hi = np.percentile(yv, [1, 99])
        pad = 0.15 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)


# chords that view the 330L beam (all others: 30L); the two groups report v_tor with
# opposite sign, so the profile lines are drawn per group
BEAM_330 = {7, 8, 9, 10, 11, 12, 13, 14, 15, 22, 23, 40, 41, 42, 43, 44, 45, 46, 47}  # t08-t16, t23, t24, t41-t48


def evaluated_chords(f, xr):
    """Chords with a verified radius and clean targets in this discharge, core to edge."""
    return [c for c in np.argsort(xr) if np.isfinite(xr[c]) and chord_quality(f, c)[0]]


def profile_frame(f, chords, a, b, xr=None, margin=0.2):
    """Frame in the interior of a..b (``margin`` of the span left out at both ends, so the
    marker in the trace is visible) with the most core chords (R < 2.15 m, 30L beam) that
    have a fit of both targets; ties go to the frame with more chords overall. (The 30L and
    330L beams are not always on together, so the frame with the most chords overall can
    show only the edge.)"""
    a, b = a + int(margin * (b - a)), b - int(margin * (b - a))
    lab = np.isfinite(f["y"][chords, a:b + 1, 0]) & np.isfinite(f["y"][chords, a:b + 1, 1])
    n = 1e-3 * lab.sum(0).astype(float)
    if xr is not None:
        core = np.array([np.isfinite(xr[c]) and xr[c] < 2.15 for c in chords])
        n += lab[core].sum(0)
    return a + int(np.argmax(n))


def draw_profile(ax, f, chords, xr, i, t, sc, ms, lw):
    """Fits and model over R at frame i, evaluated chords with a fit only; one line per beam."""
    for grp in ([c for c in chords if c not in BEAM_330], [c for c in chords if c in BEAM_330]):
        g = [c for c in grp if np.isfinite(f["y"][c, i, t])]
        if not g:
            continue
        xs = xr[g]
        style.fit_points(ax, xs, f["y"][g, i, t] * sc, f["sigma"][g, i, t] * sc, ms=ms,
                         elinewidth=0.5)
        style.model_line(ax, xs, f["pred"][g, i, t] * sc, f["pred_sigma"][g, i, t] * sc,
                         role="zero_shot", marker="", lw=lw, zorder=6)


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds", required=True, help="DIII-D validation-set dump (concatenated)")
    p.add_argument("--lengths", required=True, help="npz: files, T (frames per file of --preds)")
    p.add_argument("--min-span", type=float, default=2.0, help="bottom row: minimum span of fits (s)")
    p.add_argument("--min-chords", type=int, default=10, help="bottom row: minimum evaluated chords (profile)")
    p.add_argument("--chords", required=True, help="chord,x CSV: major radius (m) where the chord crosses the beam")
    p.add_argument("--cmin", type=int, default=5, help="histogram: hide bins with fewer labels")
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", default="d3d_overview")
    args = p.parse_args()

    d = load_predictions(args.preds)
    L = np.load(args.lengths)
    files, lengths = [str(f) for f in L["files"]], L["T"]
    per = split_dump(d, lengths)
    shots = [re.search(r"(\d{6})", Path(f).name).group(1) for f in files]
    xr = np.full(d.C, np.nan)
    for ln in open(args.chords).read().split()[1:]:
        k, v = ln.split(",")
        xr[int(k)] = float(v)
    # traces: core chords of the 30L beam (R < 2.15 m; radii verified on the fitted
    # T_i profiles), ordered core to edge
    prefer = [c for c in np.argsort(xr) if np.isfinite(xr[c]) and 1.5 < xr[c] < 2.15]

    # panels a-c: consistent labels, pooled per chord over the dump (Methods definition)
    cls = [label_classes(d.y[..., t]) for t in range(2)]
    stats, r2c = {}, {}
    for t, name in enumerate(TARGETS):
        y, pr = d.y[..., t], d.pred[..., t]
        ok = np.isfinite(y) & np.isfinite(pr)
        cons = ok & (cls[t] == 0)
        sc = style.TARGET_SCALE[name]
        stats[name] = (_r2(pr[cons], y[cons]), float(np.median(np.abs(pr - y)[cons])) * sc)
        r2c[name] = np.full(d.C, np.nan)
        for c in range(d.C):
            m = cons[c]
            if m.sum() > 100:
                r2c[name][c] = _r2(pr[c][m], y[c][m])
        print(f"{name}: R2 consistent {stats[name][0]:.3f}, median |error| {stats[name][1]:.3g}; "
              f"per-chord R2 median {np.nanmedian(r2c[name]):.2f}, {np.isfinite(r2c[name]).sum()} chords")

    # bottom row: the three validation discharges whose clean core chord the model follows
    # best (lower of the two R2 on that chord), over at least --min-span s of fits
    ranked = []
    for k, f in enumerate(per):
        best = None
        for c in prefer:
            if not chord_quality(f, c)[0]:
                continue
            a, b = window(f, c)
            if (b - a) / FS < args.min_span:
                continue
            sc = chord_score(f, c, a, b)
            if best is None or sc > best[0]:
                best = (sc, c, a, b)
        if best is not None:
            ranked.append((best[0], k, *best[1:]))
    ranked.sort(reverse=True)
    print(f"{len(ranked)} validation discharges with a clean core chord over >= {args.min_span} s; "
          f"top scores " + ", ".join(f"{shots[k]} {sc:.3f}" for sc, k, *_ in ranked[:6]))
    cols, picks = [], []
    from joblib import load as jload
    from cer_transfer.configs import data_path
    # three typical examples: the discharges in the middle of the ranking (median and its
    # two neighbours) among those with >= --min-chords evaluated chords for the profile
    elig = [rw for rw in ranked if len(evaluated_chords(per[rw[1]], xr)) >= args.min_chords]
    if len(elig) < 3:
        raise SystemExit(f"only {len(elig)} discharges with >= {args.min_chords} evaluated chords")
    m = len(elig) // 2
    rows = sorted(elig[max(m - 1, 0):m + 2], key=lambda rw: shots[rw[1]])
    print("eligible (score order): " + ", ".join(
        f"{shots[rw[1]]} {rw[0]:.3f} ({len(evaluated_chords(per[rw[1]], xr))} ch)" for rw in elig)
          + f"; median {elig[m][0]:.3f}")
    for (sc, k, c, a, b), lab in zip(rows, "def"):
        spec = np.asarray(jload(data_path(files[k]), mmap_mode="r")["input"], np.float32)
        # model shown at the frames where the chord has a conventional fit (beam on and
        # accepted by the fitting routine); partly lit frames at the beam switching are left out
        on = np.isfinite(per[k]["y"][c, :, 0]) & np.isfinite(per[k]["y"][c, :, 1])
        ev = evaluated_chords(per[k], xr)
        ip = profile_frame(per[k], ev, a, b, xr)
        cols.append((lab, shots[k], per[k], True))
        picks.append((c, a, b, on, ev, ip))
        nfit = int((np.isfinite(per[k]["y"][ev, ip, 0]) & np.isfinite(per[k]["y"][ev, ip, 1])).sum())
        print(f"({lab}) {shots[k]}: chord t{c + 1:02d} (R = {xr[c]:.2f} m), {a / FS:.2f}-{b / FS:.2f} s, "
              f"min R2 on the chord {sc:.3f}; {len(ev)} evaluated chords, profile at {ip / FS:.3f} s "
              f"({nfit} with a fit)")

    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            paper = ctx == "paper"
            small = "x-small" if paper else "small"
            ms, lw = (2.0, 1.0) if paper else (3.5, 1.6)
            # columns: (a, b, c) summary | T_i traces (d, e, f) | T_i profiles; one row per discharge
            fig = plt.figure(figsize=(PAPER_W, 5.4) if paper else (19, 14), layout="constrained")
            fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0.0, wspace=0.0)
            gs = fig.add_gridspec(3, 3, width_ratios=[1.2, 1.45, 1])
            # (a, b) and their shared colour bar, which spans both rows; empty middle column: gap
            ab = gs[0:2, 0].subgridspec(2, 3, width_ratios=[1, 0.16, 0.05])
            axa = [fig.add_subplot(ab[j, 0]) for j in range(2)]
            cax = fig.add_subplot(ab[:, 2])
            axc = fig.add_subplot(gs[2, 0])
            maps = []
            for ax, name, t in zip(axa, TARGETS, (0, 1)):
                *_, m = _draw_flaghist(ax, d.y[..., t], d.pred[..., t], cls[t], style.TARGET_SCALE[name],
                                       paper, small, range_from_consistent=True, show_flagged=False,
                                       color=style.COLORS["zero_shot"], cmin=args.cmin, pct=(0.5, 99.5))
                ax.grid(False)
                xl, yl = style.target_rec_labels(name)  # short: superscript target / rec
                ax.set_xlabel(xl)
                ax.set_ylabel(yl)
                r2, med = stats[name]
                ax.text(0.03, 0.97, f"$R^2$ = {r2:.2f}\nmedian |error| = {med:.2g} "
                        f"{'keV' if name == 'ti' else 'km/s'}", transform=ax.transAxes, va="top",
                        ha="left", fontsize=5 if paper else "x-small", zorder=6,
                        bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.9))
                maps.append(m)
            vmax = max(mp.get_array().max() for mp in maps)
            for mp in maps:
                mp.set_clim(args.cmin, vmax)
            for ax in axa:
                ax.set_anchor("E")  # square panels next to the colour bar
            col = style.COLORS["zero_shot"]
            ch = xr
            bottom = -0.5
            for name, mk, mfc in (("ti", "o", col), ("vtor", "s", "white")):
                r = r2c[name]
                cl = r < bottom
                keep = np.isfinite(ch)
                r = np.where(keep, r, np.nan)
                axc.plot(ch, np.where(cl, bottom, r), mk, color=col, mfc=mfc, ms=3 if paper else 6,
                         ls="none", label=style.TARGET_SYMBOL[name], clip_on=False)
                if cl.any():
                    axc.plot(ch[cl], np.full(cl.sum(), bottom), "v", color=col, ms=4 if paper else 7,
                             clip_on=False)
            axc.axhline(0, color="0.5", lw=0.8)
            axc.set_ylim(min(0.0, np.nanmin([r2c["ti"][np.isfinite(xr)], r2c["vtor"][np.isfinite(xr)]]) - 0.05), 1.0)
            axc.set_xlabel("$R$ (m)")
            axc.set_ylabel("$R^2$ per chord")
            axc.legend(fontsize=small, loc="lower left")

            # traces share the time axis, profiles the R axis; tick labels on the bottom row only
            letters_axes = axa + [axc]
            trs, prs, tspan = [], [], []
            ti_sc = style.TARGET_SCALE["ti"]
            for j, ((lab, shot, f, has_ti), (c, a, b, on, ev, ip)) in enumerate(zip(cols, picks)):
                tm = ip / FS
                tr = fig.add_subplot(gs[j, 1], sharex=trs[0] if trs else None)
                pr = fig.add_subplot(gs[j, 2], sharex=prs[0] if prs else None)
                draw_trace(tr, f, c, 0, ti_sc, a, b, ms, lw, on, tm)
                draw_profile(pr, f, ev, xr, ip, 0, ti_sc, ms, lw)
                tspan += [a / FS, b / FS]
                for axx in (tr, pr):
                    axx.set_ylabel(style.TARGET_LABEL["ti"])
                tr.set_title(f"discharge {shot}, $R$ = {xr[c]:.2f} m", fontsize=small, loc="left", pad=3)
                pr.set_title(f"$t$ = {tm:.2f} s", fontsize=small, loc="left", pad=3)
                if j < len(cols) - 1:
                    plt.setp(tr.get_xticklabels() + pr.get_xticklabels(), visible=False)
                trs.append(tr)
                prs.append(pr)
            trs[0].set_xlim(min(tspan), max(tspan))
            trs[-1].set_xlabel(style.TIME_LABEL)
            prs[-1].set_xlabel("$R$ (m)")
            letters_axes += trs
            style.panel_letters(letters_axes, x=-0.14, y=1.06)
            hh = [Line2D([], [], color=style.COLORS["fit"], marker="o", ls="", ms=4),
                  Line2D([], [], color=style.COLORS["zero_shot"], lw=1.5),
                  Line2D([], [], color="0.35", lw=0.9, ls="--")]
            fig.legend(hh, [style.NAMES["fit"], "DIII-D model", "profile time"],
                       loc="outside upper center", ncol=3, frameon=False, fontsize=small)
            cb = fig.colorbar(maps[0], cax=cax)
            cb.ax.tick_params(labelsize="xx-small" if paper else "x-small", length=2, pad=1)
            fig.savefig(args.out / f"{args.out_name}.{ext}", dpi=300, bbox_inches=bbox(ext))
            plt.close(fig)
    print(f"wrote {args.out}/{args.out_name}.png/.pdf")


if __name__ == "__main__":
    main()
