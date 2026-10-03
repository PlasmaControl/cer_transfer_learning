"""3D waterfall of an event: the model profile for every frame in the
window (x: coordinate across the lines of sight, y: time, z: quantity),
with the conventional fits as black dots at their own times.

    pixi run python -u -m cer_transfer.figures.event_3d --preds gallery/e116939.npz \
        --t0 0.340 --t1 0.435 --t-offset -0.235 --target vtor \
        --profile-x figs/chords_Rtan_116939.csv \\
        --profile-xlabel '$R_\\mathrm{tan}$ (m)' \
        --out figs --out-name event_ntv_116939
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from cer_transfer.figures.common import (
    FRAME_HZ,
    chord_coordinate,
    fitted_frames,
    frame_of,
    load_predictions,
    sorted_chords,
)


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--t0", type=float, required=True)
    p.add_argument("--t1", type=float, required=True)
    p.add_argument("--t-offset", type=float, default=0.0)
    p.add_argument("--fs", type=float, default=FRAME_HZ, help="frame rate (Hz)")
    p.add_argument("--target", choices=("ti", "vtor"), default="vtor")
    p.add_argument("--profile-x", type=Path, default=None)
    p.add_argument("--profile-xlabel", type=str, default=None)
    p.add_argument("--elev", type=float, default=25.0)
    p.add_argument(
        "--azim",
        type=float,
        default=20.0,
        help="default 20: time runs left to right across the screen, "
        "the radius goes into the depth, so slices stand side by side",
    )
    p.add_argument("--no-fits", action="store_true")
    p.add_argument(
        "--min-fit-frac",
        type=float,
        default=0.5,
        help="plot only chords with a conventional fit in at least "
        "this fraction of the fitted frames in the window",
    )
    p.add_argument(
        "--zlim",
        type=float,
        nargs=2,
        default=None,
        help="fixed vertical axis range, e.g. to compare two models "
        "on identical axes",
    )
    p.add_argument(
        "--no-band", action="store_true", help="omit the model uncertainty bands"
    )
    p.add_argument(
        "--no-errorbars",
        action="store_true",
        help="omit the error bars of the conventional fits",
    )
    p.add_argument(
        "--k",
        type=float,
        nargs=2,
        default=None,
        help="scale factors (Ti, vtor) for the model sigma, as in "
        "cer_transfer.figures.recon_composite",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="draw every n-th frame (default: every frame)",
    )
    p.add_argument(
        "--time-toward",
        action="store_true",
        help="reverse the time axis (time increasing toward the viewer)",
    )
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", type=str, default="event_3d")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    FS = args.fs
    dmp = load_predictions(args.preds)
    C, T = dmp.C, dmp.T
    k = 1 if args.target == "vtor" else 0
    y, mu, sl = dmp.y[..., k], dmp.pred[..., k], dmp.sigma[..., k]
    kk = 1.0 if args.k is None else args.k[k]
    ps = (
        dmp.pred_sigma[..., k] * kk
        if (dmp.pred_sigma is not None and not args.no_band)
        else None
    )

    xm, xlab = chord_coordinate(C, args.profile_x, args.profile_xlabel)
    order, xs = sorted_chords(xm)

    f0 = max(frame_of(args.t0, FS, args.t_offset), 0)
    f1 = min(frame_of(args.t1, FS, args.t_offset), T - 1)
    frames = list(range(f0, f1 + 1, max(args.stride, 1)))
    ts = np.array([f / FS + args.t_offset for f in frames])
    # restrict to chords that have targets in the window
    fit_frames = fitted_frames(y, f0, f1)
    if fit_frames:
        frac = np.isfinite(y[:, fit_frames]).mean(axis=1)
        keep = [i for i in order if frac[i] >= args.min_fit_frac]
        if len(keep) >= 3:
            order = keep
            xs = xm[order]
    print(
        f"chords plotted: {len(order)} (with fits in >= "
        f"{100 * args.min_fit_frac:.0f}% of the fitted frames), "
        f"x {xs[0]:.3f}-{xs[-1]:.3f}"
    )
    unit = "$v_{tor}$ (km/s)" if k else "$T_i$ (eV)"

    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        fs = 8 if ext == "pdf" else 11
        plt.rcParams.update({"font.size": fs})
        fig = plt.figure(figsize=(5.2, 4.2) if ext == "pdf" else (8.5, 7))
        ax = fig.add_subplot(111, projection="3d")
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # explicit drawing order, far to near: each slice is filled white
        # below its line, so nearer slices hide the ones behind (ridge-line
        # occlusion); fits of a slice are drawn with that slice
        ax.computed_zorder = False
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        zall = mu[order][:, frames]
        zmax = float(np.nanmax(zall)) * 1.05
        zb = float(np.nanmin(zall)) - 0.02 * float(np.nanmax(zall) - np.nanmin(zall))
        if args.zlim is not None:
            zb, zmax = args.zlim
        # chords measured anywhere in the window: the band is drawn in full
        # there; on chords never measured it is light, clipped and does not
        # occlude (the model's uncertainty is large there)
        meas = np.isfinite(y[order][:, frames]).any(axis=1)
        # drawing order from the actual camera: depth of a slice along the
        # viewing direction (normalized time coordinate x y-component of the
        # eye vector); farthest first. Works for any azimuth/elevation and
        # for either direction of the time axis.
        t_lo, t_hi = (ts[-1], ts[0]) if args.time_toward else (ts[0], ts[-1])
        vy = np.cos(np.radians(args.elev)) * np.sin(np.radians(args.azim))
        depth = lambda tt: ((tt - t_lo) / (t_hi - t_lo)) * vy
        far_to_near = sorted(zip(frames, ts), key=lambda ft: depth(ft[1]))
        for rank, (f, tt) in enumerate(far_to_near):
            z = mu[order, f]
            zo = 5 * rank
            top = z.copy()
            if ps is not None:
                s_ = ps[order, f]
                top = np.where(meas, np.minimum(z + s_, zmax), z)
            # white occluder up to the top of the band (measured chords)
            verts = [
                list(zip(xs, np.full_like(xs, tt), top))
                + [(xs[-1], tt, zb), (xs[0], tt, zb)]
            ]
            ax.add_collection3d(
                Poly3DCollection(verts, facecolor="white", edgecolor="none", zorder=zo)
            )
            if ps is not None:
                lo_b = np.clip(z - s_, zb, zmax)
                hi_b = np.clip(z + s_, zb, zmax)
                for mask, alpha in ((meas, 0.30), (~meas, 0.06)):
                    # contiguous runs, extended by one point so runs connect
                    run_m = mask | np.r_[mask[1:], False] | np.r_[False, mask[:-1]]
                    idx = np.where(run_m)[0]
                    if idx.size < 2:
                        continue
                    for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                        if run.size < 2:
                            continue
                        band = list(
                            zip(xs[run], np.full(run.size, tt), hi_b[run])
                        ) + list(
                            zip(xs[run][::-1], np.full(run.size, tt), lo_b[run][::-1])
                        )
                        ax.add_collection3d(
                            Poly3DCollection(
                                [band],
                                facecolor="C0",
                                alpha=alpha,
                                edgecolor="none",
                                zorder=zo + 1,
                            )
                        )
            ax.plot(xs, np.full_like(xs, tt), z, "-", color="C0", lw=0.9, zorder=zo + 2)
            if not args.no_fits:
                yy = y[order, f]
                m = np.isfinite(yy)
                if m.any():
                    if not args.no_errorbars:
                        ee = np.nan_to_num(sl[order, f][m])
                        segs = [
                            [(xx, tt, max(v - e, zb)), (xx, tt, min(v + e, zmax))]
                            for xx, v, e in zip(xs[m], yy[m], ee)
                        ]
                        ax.add_collection3d(
                            Line3DCollection(
                                segs, colors="0.3", linewidths=0.5, zorder=zo + 3
                            )
                        )
                    ax.scatter(
                        xs[m],
                        np.full(int(m.sum()), tt),
                        yy[m],
                        s=2.2,
                        c="k",
                        depthshade=False,
                        zorder=zo + 4,
                    )
        ax.set_zlim(zb, zmax)
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.zaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel(xlab, labelpad=9)
        ax.set_ylabel("time (s)", labelpad=6)
        ax.set_zlabel(unit, labelpad=4)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_box_aspect(None, zoom=0.86)  # keep the z label inside
        if args.time_toward:
            ax.set_ylim(ts[-1], ts[0])  # time increases toward the viewer
        else:
            ax.set_ylim(ts[0], ts[-1])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor("0.85")
        ax.grid(True)
        from matplotlib.patches import Patch

        hh = [Line2D([], [], color="C0", lw=1.2)]
        ll = [f"model, every {1000 * max(args.stride, 1) / FS:.0f} ms"]
        if ps is not None:
            hh.append(Patch(color="C0", alpha=0.35))
            ll.append("model uncertainty")
        if not args.no_fits:
            hh.append(Line2D([], [], color="k", marker="o", ls="", ms=3))
            ll.append(
                "conventional fit"
                if args.no_errorbars
                else "conventional fit and uncertainty"
            )
        ax.legend(hh, ll, loc="upper left", fontsize="small", frameon=False)
        fig.savefig(args.out / f"{args.out_name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(
        f"wrote {args.out}/{args.out_name}.png/.pdf | {len(frames)} frames "
        f"{ts[0]:.3f}-{ts[-1]:.3f} s"
    )


if __name__ == "__main__":
    main()
