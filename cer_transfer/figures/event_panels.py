r"""Event figure: equilibrium, profiles at fitted times, and time traces.

One row of panels per target, each the profile at one fitted time inside
the event window. Same visual language as the
reconstruction figure: model line with its uncertainty band, conventional
fits as black points, grey where no conventional fit exists; shared y-axis
across the row so the evolution reads left to right.

    pixi run python -u -m cer_transfer.figures.event_panels \
        --preds gallery/e116939.npz \
        --t0 0.340 --t1 0.435 --t-offset -0.235 --n-times 4 --target vtor \
        --profile-x figs/chords_Rtan.csv --profile-xlabel '$R_\mathrm{tan}$ (m)' \
        --k 0.992 1.171 --out figs --out-name event_ntv_116939
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import load
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cer_transfer.figures.common import (  # noqa: E402
    FRAME_HZ,
    chord_coordinate,
    load_predictions,
    sorted_chords,
)
from cer_transfer.figures.paperfig import PAPER_W, bbox  # noqa: E402
from cer_transfer.figures.timeaxis import file_times  # noqa: E402

sns.set_style("whitegrid")

FS = FRAME_HZ  # overridden by --fs


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--fs", type=float, default=FRAME_HZ, help="frame rate (Hz)")
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--t0", type=float, required=True)
    p.add_argument("--t1", type=float, required=True)
    p.add_argument("--t-offset", type=float, default=0.0)
    p.add_argument("--n-times", type=int, default=4)
    p.add_argument(
        "--times",
        type=float,
        nargs="*",
        default=None,
        help="explicit times (s); the nearest fitted frames are used",
    )
    p.add_argument(
        "--target",
        choices=("ti", "vtor", "both"),
        default="vtor",
        help="both: a v_tor and a T_i profile row and trace, same times and chords",
    )
    p.add_argument("--profile-x", type=Path, default=None)
    p.add_argument("--profile-xlabel", type=str, default=None)
    p.add_argument("--k", type=float, nargs=2, default=None)
    p.add_argument(
        "--preds2",
        type=Path,
        default=None,
        help="second model drawn in the same panels (orange), e.g. from scratch vs "
        "fine-tuned",
    )
    p.add_argument(
        "--efit",
        type=Path,
        default=None,
        help="EFIT joblib (time_s, psirz, rgrid, zgrid, psi_axis, psi_bdry, r_axis, "
        "z_axis, "
        "rbbbs, zbbbs, nbbbs, rlim, zlim): adds a top row with the equilibrium at "
        "each panel time, "
        "chords marked on the midplane",
    )
    p.add_argument(
        "--efit-zrange",
        type=float,
        nargs=2,
        default=None,
        help="Z range of the equilibrium panel (default: the limiter extent, whole "
        "cross-section)",
    )
    p.add_argument(
        "--efit-all-times",
        action="store_true",
        help="one equilibrium per panel time (default: one, at t0, above the first "
        "profile panel)",
    )
    p.add_argument(
        "--shot",
        default=None,
        help="discharge file (list entry); its time_s gives the time axis, else "
        "frames/FS + t-offset",
    )
    p.add_argument(
        "--legend-no-fit",
        action="store_true",
        help="add the grey 'no conventional fit' band to the legend (default: "
        "caption only)",
    )
    p.add_argument("--label1", default="model", help="legend text of --preds")
    p.add_argument("--label2", default="second model", help="legend text of --preds2")
    p.add_argument(
        "--trace-x",
        type=float,
        default=None,
        help="chord coordinate of a time trace (second row): model every frame with "
        "its "
        "band, fits where they exist, over the whole window",
    )
    p.add_argument(
        "--min-fit-frac",
        type=float,
        default=0.5,
        help="draw only chords with a conventional fit in at least this fraction "
        "of the window's fitted frames (0 = all chords)",
    )
    p.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        help="fixed y-range, e.g. to compare two models on identical axes",
    )
    p.add_argument(
        "--no-reference",
        action="store_true",
        help="omit the first profile as a grey reference in later panels",
    )
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", type=str, default="event_panels")
    args = p.parse_args()
    global FS
    FS = args.fs
    args.out.mkdir(parents=True, exist_ok=True)

    dmp = load_predictions(args.preds)
    C, T = dmp.C, dmp.T
    tnames = {"vtor": [1], "ti": [0], "both": [1, 0]}[args.target]  # v_tor first
    D = {}
    d2 = None
    if args.preds2 is not None:
        d2 = load_predictions(args.preds2)
        if (d2.C, d2.T) != (C, T):
            raise SystemExit("--preds2 must be a dump of the same discharge")
    for k in tnames:
        kk = 1.0 if args.k is None else args.k[k]
        sc = 1.0 if k == 1 else 1e-3  # T_i displayed in keV
        D[k] = dict(
            y=dmp.y[..., k] * sc,
            sl=dmp.sigma[..., k] * sc,
            mu=dmp.pred[..., k] * sc,
            ps=dmp.pred_sigma[..., k] * kk * sc if dmp.pred_sigma is not None else None,
            mu2=d2.pred[..., k] * sc if d2 is not None else None,
            ps2=(
                d2.pred_sigma[..., k] * kk * sc
                if d2 is not None and d2.pred_sigma is not None
                else None
            ),
            unit="$v_{tor}$ (km/s)" if k == 1 else "$T_i$ (keV)",
        )
    k0 = tnames[0]
    y = D[k0]["y"]  # frame/chord selection from the first target (v_tor if drawn)
    mu2 = D[k0]["mu2"]

    efit = None
    if args.efit is not None:
        ef = load(args.efit)
        efit = {
            kq: np.asarray(ef[kq])
            for kq in (
                "time_s",
                "psirz",
                "rgrid",
                "zgrid",
                "psi_axis",
                "psi_bdry",
                "r_axis",
                "z_axis",
                "rbbbs",
                "zbbbs",
                "nbbbs",
                "rlim",
                "zlim",
            )
        }

    xm, xlab = chord_coordinate(C, args.profile_x, args.profile_xlabel)
    order, xs = sorted_chords(xm)

    tax = (
        file_times(args.shot, T, FS, args.t_offset)
        if args.shot
        else np.arange(T) / FS + args.t_offset
    )
    f0 = int(np.argmin(np.abs(tax - args.t0)))
    f1 = int(np.argmin(np.abs(tax - args.t1)))
    fitted = [
        f
        for f in range(max(f0, 0), min(f1, T - 1) + 1)
        if np.isfinite(y[:, f]).sum() >= 5
    ]
    if not fitted:
        raise SystemExit("no fitted frames in the window")
    # chords that carry a conventional fit in at least --min-fit-frac of the
    # window's fitted frames; the outermost chords never do (they view outside
    # the labeled plasma) and the model has no label there
    frac = np.isfinite(y[:, fitted]).mean(1)
    keep = frac >= args.min_fit_frac
    if keep.sum() >= 5:
        order = [c for c in order if keep[c]]
        xs = np.asarray([xm[c] for c in order])
    else:
        print("warning: --min-fit-frac keeps fewer than 5 chords; drawing all")
    if args.times:
        frames = [min(fitted, key=lambda f: abs(tax[f] - tt)) for tt in args.times]
    else:
        idx = np.linspace(0, len(fitted) - 1, args.n_times).round().astype(int)
        frames = [fitted[i] for i in idx]

    # y-limits per target from curves and fits (not from the bands)
    for k in tnames:
        d = D[k]
        vals = np.concatenate(
            [np.r_[d["mu"][order, f], d["y"][order, f]] for f in frames]
            + ([d["mu2"][order, f] for f in frames] if d["mu2"] is not None else [])
        )
        vals = vals[np.isfinite(vals)]
        lo, hi = vals.min(), vals.max()
        pad = 0.1 * max(hi - lo, 1e-6)
        d["ylim"] = (
            (lo - pad, hi + pad) if (args.ylim is None or k != 1) else tuple(args.ylim)
        )
    dx = np.diff(xs)
    step = float(np.median(dx)) if dx.size else 0.5

    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            n = len(frames)
            trace = args.trace_x is not None
            eq = efit is not None
            zr = args.efit_zrange
            xpad = 0.02 * (xs.max() - xs.min())
            xlim_prof = (xs.min() - xpad, xs.max() + xpad)
            # two rows: [equilibrium | n profile panels] and the time trace; the
            # equilibrium shows the
            # whole cross-section (limiter extent in R and Z, equal aspect) so its width
            # follows its height
            if eq:
                r_lo, r_hi = (
                    float(efit["rlim"].min()) - 0.03,
                    float(efit["rlim"].max()) + 0.03,
                )
                z_lo, z_hi = (
                    (float(efit["zlim"].min()) - 0.03, float(efit["zlim"].max()) + 0.03)
                    if zr is None
                    else zr
                )
                eq_w = (r_hi - r_lo) / (
                    z_hi - z_lo
                )  # width/height of the cross-section
            nt = len(tnames)
            hr = [1] * nt + (([0.85] if nt == 1 else [0.7] * nt) if trace else [])
            w_fig = PAPER_W if ctx == "paper" else 4 * n
            base_h = (
                (2.3 + (1.9 if trace else 0))
                if nt == 1
                else (3.6 + (2.6 if trace else 0))
            )
            size = (w_fig, base_h) if ctx == "paper" else (w_fig, base_h * 1.75)
            fig = plt.figure(figsize=size)
            prof_h_in = (
                (size[1]) * 0.86 * nt / sum(hr)
            )  # equilibrium spans all profile rows
            eq_col = (
                (eq_w * prof_h_in / (w_fig * 0.9 / (n + 1)))
                * (1.4 if nt == 1 else 0.85)
                if eq
                else 0.0
            )
            # separate grids: [equilibrium | spacer | profiles] on top, the trace(s)
            # below
            wr = ([eq_col, 0.42] if eq else []) + [
                1
            ] * n  # spacer column: room for the profile y-label
            outer = fig.add_gridspec(
                2 if trace else 1,
                1,
                height_ratios=[nt, sum(hr[nt:])] if trace else [1],
                hspace=0.45 if nt == 1 else 0.30,
            )
            gs_top = outer[0].subgridspec(
                nt, len(wr), width_ratios=wr, hspace=0.12, wspace=0.12
            )
            gs_bot = outer[1].subgridspec(nt, 1, hspace=0.12) if trace else None
            c0 = 2 if eq else 0
            prow = {}
            for r_, k in enumerate(tnames):
                row = []
                for j in range(n):
                    row.append(
                        fig.add_subplot(
                            gs_top[r_, c0 + j],
                            sharey=row[0] if row else None,
                            sharex=prow[tnames[0]][j] if r_ else None,
                        )
                    )
                    if j:
                        plt.setp(row[j].get_yticklabels(), visible=False)
                    row[j].set_xlim(*xlim_prof)
                prow[k] = row
            if eq:
                axe = fig.add_subplot(gs_top[0:nt, 0])
                f = frames[0]
                c_tr = (
                    order[int(np.argmin(np.abs(xs - args.trace_x)))] if trace else None
                )
                ie = int(np.argmin(np.abs(efit["time_s"] - tax[f])))
                psi = efit["psirz"][ie]
                psin = (psi - efit["psi_axis"][ie]) / (
                    efit["psi_bdry"][ie] - efit["psi_axis"][ie]
                )
                R, Z = np.meshgrid(efit["rgrid"], efit["zgrid"])
                axe.contour(
                    R,
                    Z,
                    psin,
                    levels=[0.1, 0.25, 0.4, 0.55, 0.7, 0.85],
                    cmap="viridis",
                    linewidths=0.6,
                    vmin=0.0,
                    vmax=1.2,
                )
                nb = int(efit["nbbbs"][ie])
                axe.plot(
                    efit["rbbbs"][ie, :nb],
                    efit["zbbbs"][ie, :nb],
                    "-",
                    color="k",
                    lw=0.9,
                )
                axe.plot(efit["rlim"], efit["zlim"], "-", color="0.5", lw=0.9)
                axe.plot(
                    efit["r_axis"][ie],
                    efit["z_axis"][ie],
                    "+",
                    color="C3",
                    ms=6,
                    mew=1.2,
                )
                axe.plot(
                    xs, np.zeros_like(xs), "|", color="C0", ms=5, mew=0.8, zorder=4
                )
                if c_tr is not None:
                    axe.plot([xm[c_tr]], [0.0], "o", color="C0", ms=4, zorder=5)
                axe.set_xlim(r_lo, r_hi)
                axe.set_ylim(z_lo, z_hi)
                axe.set_aspect("equal", adjustable="box")
                axe.set_anchor("W")
                axe.set_zorder(prow[tnames[0]][0].get_zorder() - 1)
                axe.set_title(
                    (
                        "equilibrium at $t_0$"
                        if not args.no_reference
                        else f"equilibrium, t = {efit['time_s'][ie]:.3f} s"
                    ),
                    fontsize="small",
                )
                axe.tick_params(labelsize="x-small")
                axe.set_xlabel("$R$ (m)", fontsize="small")
                axe.set_ylabel("$Z$ (m)", fontsize="small")
                axe.set_xticks([0.5, 1.0, 1.5])
            for r_, k in enumerate(tnames):
                d = D[k]
                y_, sl_, mu_, ps_, mu2_, ps2_ = (
                    d["y"],
                    d["sl"],
                    d["mu"],
                    d["ps"],
                    d["mu2"],
                    d["ps2"],
                )
                for j, (ax, f) in enumerate(zip(prow[k], frames)):
                    yy = y_[order, f]
                    fitm = np.isfinite(yy)
                    nf = ~fitm
                    a = None
                    for q, flag in enumerate(list(nf) + [False]):
                        if flag and a is None:
                            a = q
                        if not flag and a is not None:
                            lo_x = xs[a] - 0.5 * (
                                (xs[a] - xs[a - 1]) if a > 0 else step
                            )
                            b_ = q - 1
                            hi_x = xs[b_] + 0.5 * (
                                (xs[b_ + 1] - xs[b_]) if b_ + 1 < len(xs) else step
                            )
                            ax.axvspan(lo_x, hi_x, color="0.9", zorder=0, lw=0)
                            a = None
                    m_ = mu_[order, f]
                    if ps_ is not None:
                        s_ = ps_[order, f]
                        nof_ext = nf | np.r_[nf[1:], False] | np.r_[False, nf[:-1]]
                        ax.fill_between(
                            xs,
                            m_ - s_,
                            m_ + s_,
                            where=~nf,
                            color="C0",
                            alpha=0.22,
                            lw=0,
                            zorder=1,
                        )
                        ax.fill_between(
                            xs,
                            m_ - s_,
                            m_ + s_,
                            where=nof_ext,
                            color="C0",
                            alpha=0.07,
                            lw=0,
                            zorder=1,
                        )
                    if j > 0 and not args.no_reference:
                        ax.plot(
                            xs,
                            mu_[order, frames[0]],
                            "--",
                            color="0.55",
                            lw=1.0,
                            zorder=1.5,
                        )
                    ax.plot(xs, m_, "-", color="C0", lw=1.5, zorder=2)
                    if mu2_ is not None:
                        m2 = mu2_[order, f]
                        if ps2_ is not None:
                            s2 = ps2_[order, f]
                            ax.fill_between(
                                xs,
                                m2 - s2,
                                m2 + s2,
                                where=~nf,
                                color="C1",
                                alpha=0.15,
                                lw=0,
                                zorder=1,
                            )
                        ax.plot(xs, m2, "-", color="C1", lw=1.3, zorder=2)
                    ax.errorbar(
                        xs[fitm],
                        yy[fitm],
                        yerr=sl_[order, f][fitm],
                        fmt="o",
                        ms=2.6,
                        lw=0.6,
                        color="k",
                        ecolor="0.45",
                        zorder=3,
                    )
                    ax.set_ylim(*d["ylim"])
                    ax.set_xlim(xs[0] - 0.5 * step, xs[-1] + 0.5 * step)
                    if r_ == 0:
                        tlabel = "$t_0$" if (j == 0 and not args.no_reference) else "t"
                        ax.set_title(f"{tlabel} = {tax[f]:.3f} s", fontsize="small")
                    if r_ == nt - 1:
                        ax.set_xlabel(xlab, fontsize="small")
                    else:
                        plt.setp(ax.get_xticklabels(), visible=False)
                    if j == 0:
                        ax.set_ylabel(d["unit"])
            if trace:
                c = order[int(np.argmin(np.abs(xs - args.trace_x)))]
                fr = np.arange(max(f0, 0), min(f1, T - 1) + 1)
                tt = tax[fr]
                ax_prev = None
                for r_, k in enumerate(tnames):
                    d = D[k]
                    axt = fig.add_subplot(gs_bot[r_, 0], sharex=ax_prev)
                    ax_prev = axt
                    for src_mu, src_ps, col, lw in (
                        (d["mu"], d["ps"], "C0", 1.5),
                        (d["mu2"], d["ps2"], "C1", 1.3),
                    ):
                        if src_mu is None:
                            continue
                        m_t = src_mu[c, fr]
                        if src_ps is not None:
                            axt.fill_between(
                                tt,
                                m_t - src_ps[c, fr],
                                m_t + src_ps[c, fr],
                                color=col,
                                alpha=0.2,
                                lw=0,
                            )
                        axt.plot(tt, m_t, "-", color=col, lw=lw, zorder=2)
                        axt.plot(tt, m_t, "o", color=col, ms=2.2, zorder=2.5)
                    yt = d["y"][c, fr]
                    fm = np.isfinite(yt)
                    axt.errorbar(
                        tt[fm],
                        yt[fm],
                        yerr=d["sl"][c, fr][fm],
                        fmt="o",
                        ms=2.6,
                        lw=0.6,
                        color="k",
                        ecolor="0.45",
                        zorder=3,
                    )
                    axt.set_xlim(tt[0], tt[-1])
                    axt.set_ylim(*d["ylim"])
                    axt.set_ylabel(d["unit"])
                    if r_ == 0:
                        axt.set_title(
                            f"{xlab.split('(')[0].strip()} = {xm[c]:.2f} m, model "
                            f"every {1000 / FS:.0f} ms",
                            fontsize="small",
                        )
                    if r_ == nt - 1:
                        axt.set_xlabel("time (s)", fontsize="small")
                    else:
                        plt.setp(axt.get_xticklabels(), visible=False)
            hh = [
                Line2D([], [], color="C0", lw=1.5, marker="o", ms=3),
                Patch(color="C0", alpha=0.3),
                Line2D([], [], color="k", marker="o", ls="", ms=3),
            ]
            ll = [
                args.label1,
                "model uncertainty",
                "conventional fit and uncertainty",
            ]
            if (
                args.legend_no_fit
            ):  # the grey band is explained in the caption by default
                hh.append(Patch(color="0.9"))
                ll.append("no conventional fit")
            if mu2 is not None:
                hh.insert(1, Line2D([], [], color="C1", lw=1.3, marker="o", ms=3))
                ll.insert(1, args.label2)
            if not args.no_reference:
                hh.insert(1, Line2D([], [], color="0.55", ls="--", lw=1.0))
                ll.insert(1, "model at $t_0$")
            ncol = (
                len(hh) if sum(len(x) for x in ll) < 95 else (len(hh) + 1) // 2
            )  # wrap when wider than the page
            top = (0.93 if trace else 0.86) - (0.025 if ncol < len(hh) else 0.0)
            fig.subplots_adjust(
                left=0.105, right=0.97, bottom=0.07 if nt > 1 else 0.10, top=top - 0.03
            )
            fig.legend(
                hh,
                ll,
                loc="upper center",
                ncol=ncol,
                fontsize="x-small",
                frameon=False,
                bbox_to_anchor=(0.5, 1.0),
            )
            fig.savefig(
                args.out / f"{args.out_name}.{ext}", dpi=300, bbox_inches=bbox(ext)
            )
            plt.close(fig)
    print(
        f"wrote {args.out}/{args.out_name}.png/.pdf | times "
        + ", ".join(f"{tax[f]:.3f}" for f in frames)
        + " s"
    )


if __name__ == "__main__":
    main()
