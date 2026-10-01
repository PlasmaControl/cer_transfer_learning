"""Spectra-first beam-off figure for one chord: foreground-array
spectrogram (active line vanishes when the beam is off), background-
array spectrogram (passive line persists, Doppler-shifted), and the
reconstruction from the passive spectra against the conventional fits
(which exist only while the beam is on). Verified beam-off phases shaded.

    pixi run python -u -m cer_transfer.figures.passive_spectra \
        --bg gallery/bo_141710_bg.npz \
        --bg-file "$(cat gallery/bo_141710_bg.txt)" \
        --fg-file "$(cat gallery/bo_141710_fg.txt)" \
        --chord 2 --target vtor --t0 0.12 --t1 0.45 --t-offset -0.235 \
        --out figs --out-name passive_spectra_141710
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

from cer_transfer.beamstate import beam_state, valid_end

from cer_transfer.figures.common import FRAME_HZ

FS = FRAME_HZ  # overridden by --fs


def spec_of(fp, chord):
    d = load(fp, mmap_mode="r")
    end = valid_end(d)
    return np.asarray(d["input"][chord, :end, :], dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", type=float, default=FRAME_HZ, help="frame rate (Hz)")
    p.add_argument("--bg", type=Path, required=True, help="passive-model npz")
    p.add_argument("--bg-file", type=Path, required=True)
    p.add_argument("--fg-file", type=Path, required=True)
    p.add_argument("--chord", type=int, default=2)
    p.add_argument("--target", choices=("ti", "vtor"), default="vtor")
    p.add_argument("--t0", type=float, required=True)
    p.add_argument("--t1", type=float, required=True)
    p.add_argument("--t-offset", type=float, default=0.0)
    p.add_argument("--px-halfwin", type=int, default=20,
                   help="wavelength crop: +- pixels around the line peak")
    p.add_argument("--no-centroid", action="store_true",
                   help="omit the line-centre trace on the spectrograms")
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", type=str, default="passive_spectra")
    args = p.parse_args()
    global FS
    FS = args.fs
    args.out.mkdir(parents=True, exist_ok=True)

    c = args.chord
    sf = spec_of(args.fg_file, c)
    sb = spec_of(args.bg_file, c)
    d = np.load(args.bg)
    ch = d["chord"]
    C = int(ch.max()) + 1
    Tn = len(ch) // C
    pred = d["pred"].reshape(C, Tn, -1)
    psig = d["pred_sigma"].reshape(C, Tn, -1)
    y = d["y"].reshape(C, Tn, -1)
    st = beam_state(args.bg_file, args.fg_file)
    T = min(len(sf), len(sb), Tn, st["T"])
    off = np.zeros(T, bool)
    for a, b, cls, ok, _ in st["segments"]:
        if ok:
            off[a:min(b, T - 1) + 1] = True

    # period without conventional measurement: from the last fit before
    # each verified beam-off phase to the first fit after it; merged
    kk = 1 if args.target == "vtor" else 0
    fitted = np.isfinite(y[:, :T, kk]).any(axis=0)
    spans = []
    fr = np.where(fitted)[0]
    for a, b, cls, ok, _ in st["segments"]:
        if not ok or a >= T:
            continue
        left = fr[fr < a]
        right = fr[fr > b]
        s0 = int(left[-1]) + 1 if left.size else a
        s1 = int(right[0]) - 1 if right.size else min(b, T - 1)
        if spans and s0 <= spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], max(spans[-1][1], s1))
        else:
            spans.append((s0, s1))

    t = np.arange(T) / FS + args.t_offset
    f0 = max(int(round((args.t0 - args.t_offset) * FS)), 0)
    f1 = min(int(round((args.t1 - args.t_offset) * FS)), T - 1)
    w = np.arange(f0, f1 + 1)
    k = 1 if args.target == "vtor" else 0
    unit = "$v_{tor}$ (km/s)" if k else "$T_i$ (eV)"

    def runs(mask):
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

    def shade(ax, img=False):
        for a, b in spans:
            if b < f0 or a > f1:
                continue
            x0 = t[max(a, f0)] - 0.5 / FS
            x1 = t[min(b, f1)] + 0.5 / FS
            if img:
                ax.axvline(x0, color="w", lw=1.0, ls="--")
                ax.axvline(x1, color="w", lw=1.0, ls="--")
            else:
                ax.axvspan(x0, x1, color="0.88", zorder=0, lw=0)
                ax.text(0.5 * (x0 + x1), 0.97, "no conventional\nmeasurement",
                        ha="center", va="top", fontsize="x-small",
                        transform=ax.get_xaxis_transform())

    # common colour scale per array: dark-subtracted, window percentiles
    Wn = sf.shape[1]
    pk = int(np.argmax((sf[w] + sb[w]).mean(axis=0)))
    p0, p1 = max(0, pk - args.px_halfwin), min(Wn, pk + args.px_halfwin + 1)

    def prep(s):
        z = s[w] - np.median(s[:40], axis=0, keepdims=True)
        z = z[:, p0:p1]
        lo, hi = np.percentile(z, [1, 99.5])
        return z.T, lo, hi

    def centroid(s):
        z = s[w] - np.median(s[:40], axis=0, keepdims=True)
        q0, q1 = max(0, pk - 8), min(Wn, pk + 9)
        v = np.clip(z[:, q0:q1], 0, None)
        tot = v.sum(axis=1)
        cen = np.full(len(w), np.nan)
        good = tot > np.percentile(tot, 20)
        cen[good] = (v[good] * np.arange(q0, q1)).sum(axis=1) / tot[good]
        # light 3-frame median smoothing
        cs = cen.copy()
        for i in range(1, len(cen) - 1):
            cs[i] = np.nanmedian(cen[i - 1:i + 2])
        return cs

    zf, lof, hif = prep(sf)
    zb, lob, hib = prep(sb)
    cf, cb = centroid(sf), centroid(sb)
    ext = [t[f0] - 0.5 / FS, t[f1] + 0.5 / FS, p0, p1]

    for ctx, extn in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            size = (5.2, 6.2) if ctx == "paper" else (9, 10)
            fig, axes = plt.subplots(3, 1, figsize=size, sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1, 1.2]})
            for ax, z, lo, hi, lab, cc in (
                    (axes[0], zf, lof, hif, "foreground array\n(views the beam)", cf),
                    (axes[1], zb, lob, hib, "background array\n(passive light)", cb)):
                ax.imshow(z, aspect="auto", origin="lower", extent=ext,
                          vmin=lo, vmax=hi, cmap="magma",
                          interpolation="nearest")
                if not args.no_centroid:
                    ax.plot(t[w], cc, "-", color="c", lw=1.1,
                            label="line centre")
                ax.set_ylim(p0, p1)
                shade(ax, img=True)
                ax.set_ylabel(f"{lab}\nwavelength (px)", fontsize="small")
                ax.grid(False)
            ax = axes[2]
            shade(ax)
            m_, s_ = pred[c, w, k], psig[c, w, k]
            ax.fill_between(t[w], m_ - s_, m_ + s_, color="C0", alpha=0.22,
                            lw=0)
            ax.plot(t[w], m_, "-", color="C0", lw=1.6,
                    label="reconstruction from passive spectra")
            yv = y[c, w, k]
            mm = np.isfinite(yv)
            ax.plot(t[w][mm], yv[mm], "o", color="k", ms=3.5, zorder=5,
                    label="conventional fit")
            ax.set_ylabel(unit)
            ax.set_xlabel("time (s)")
            ax.legend(fontsize="x-small", loc="best")
            axes[0].set_title(f"chord {c}", fontsize="medium")
            fig.tight_layout()
            fig.savefig(args.out / f"{args.out_name}.{extn}", dpi=300,
                        bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {args.out}/{args.out_name}.png/.pdf | chord {c} | "
          f"periods without conventional measurement: "
          + ", ".join(f"{t[a]:.3f}-{t[b]:.3f} s" for a, b in spans))


if __name__ == "__main__":
    main()
