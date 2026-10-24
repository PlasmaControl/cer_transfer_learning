"""Gallery of all real beam-off phases: targets and both models, for picking examples.

Real beam-off phase: a gap of ``analysis.beamoff_bridge`` with NBI power at the last
fits before the gap above 0.5 MW and below 20 % of that inside the gap. One block per
gap: v_tor and T_i of one core chord (the chord <= 12 with the best match to the fits
before / after the gap and the smoothest bridge; see beamoff_bridge) with the
targets (black), the DIII-D model (green) and the DIII-D+NSTX model (red), the gap
shaded, and the NBI power below. Order: gaps starting on the flat-top (>= --tmin)
first, then by gap length. Multi-page PDF, six gaps per page.

    pixi run --frozen python -u -m cer_transfer.figures.beamoff_gallery \
        --bridge gallery/paper/beamoff_bridge.npz --out gallery/beamoff/beamoff_gallery.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from joblib import load  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from cer_transfer.analysis.beamoff_checks import load_readout, predict  # noqa: E402
from cer_transfer.configs import data_path  # noqa: E402
from cer_transfer.figures import style  # noqa: E402
from cer_transfer.figures.timeaxis import file_times  # noqa: E402
from cer_transfer.inference import load_model  # noqa: E402


def real_gaps(z, tmin):
    """Real beam-off gaps (entry, gap_a, gap_b, split, chord) ordered flat-top first, then by length."""
    worst = np.maximum.reduce([z["fit_err_before_1"], z["fit_err_after_1"], z["gap_dev_1"], z["gap_rough_1"]])
    worst = np.where(np.isfinite(worst), worst, np.inf)
    best = {}
    for i in np.where((z["target"] == 1) & (z["chord"] <= 12))[0]:
        key = (z["entry"][i], int(z["gap_a"][i]), int(z["gap_b"][i]), z["split"][i])
        if key not in best or worst[i] < worst[best[key]]:
            best[key] = i
    out = []
    for (e, a, b, sp), i in best.items():
        d = load(data_path(e), mmap_mode="r")
        p = np.asarray(d["nbi_power_MW"], float)
        fitted = np.isfinite(np.asarray(d["target"][:, :a, 1])).any(0)
        lf = np.where(fitted)[0][-1]
        pre, ins = p[max(lf - 4, 0):lf + 1].mean(), p[a:b + 1].mean()
        if pre > 0.5 and ins < 0.2 * pre:
            t0 = file_times(e, d["input"].shape[1], 200.0, -0.235)[a]
            out.append((e, a, b, sp, int(z["chord"][i]), t0, worst[i]))
    out.sort(key=lambda g: (g[5] < tmin, -(g[2] - g[1])))
    return out


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bridge", type=Path, default=Path("gallery/paper/beamoff_bridge.npz"))
    ap.add_argument("--zs", default="cer_ckpts/d3d_fixed_dg_rt.pt")
    ap.add_argument("--zs-readout", default="cer_ckpts/d3d_fixed_dg_rt_labelfree_nstx.npz")
    ap.add_argument("--ft", default="cer_ckpts/nstx_rt_init_rfull.pt")
    ap.add_argument("--tmin", type=float, default=0.25, help="flat-top: gaps starting after this time first")
    ap.add_argument("--window", type=float, default=0.12, help="s shown before and after the gap")
    ap.add_argument("--per-page", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("gallery/beamoff/beamoff_gallery.pdf"))
    a = ap.parse_args()

    z = np.load(a.bridge)
    gaps = real_gaps(z, a.tmin)
    print(f"{len(gaps)} real beam-off gaps ({sum(g[5] >= a.tmin for g in gaps)} starting >= {a.tmin} s)", flush=True)
    models = [("DIII-D model", "zero_shot", load_model(a.zs, device=a.device, machine="nstx"), load_readout(a.zs_readout)),
              ("DIII-D+NSTX model", "finetuned", load_model(a.ft, device=a.device, machine="nstx"), None)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    nw = int(round(a.window * 200))
    with PdfPages(a.out) as pdf:
        for p0 in range(0, len(gaps), a.per_page):
            page = gaps[p0:p0 + a.per_page]
            fig = plt.figure(figsize=(18, 11))
            outer = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.22, left=0.04, right=0.99, top=0.95, bottom=0.04)
            for q, (e, ga, gb, sp, c, t0, w) in enumerate(page):
                if e not in cache:
                    d = load(data_path(e))
                    end = int(d["end_index"]); T = d["input"].shape[1] if end <= 0 else end
                    x = np.asarray(d["input"][:, :T], np.float32)
                    disp = float(np.abs(np.median(np.diff(np.asarray(d["wavelength_nm"], np.float64)))))
                    preds = [predict(m, x, disp, ro) for _, _, m, ro in models]
                    cache = {e: (d, T, preds)}  # one discharge at a time (memory)
                d, T, preds = cache[e]
                t = file_times(e, d["input"].shape[1], 200.0, -0.235)[:T]
                lo, hi = max(ga - nw, 0), min(gb + nw, T - 1)
                sl = slice(lo, hi + 1)
                y = np.asarray(d["target"][:, :T], np.float32) * np.array([1.0, 1.0], np.float32)  # T_i keV
                ye = np.asarray(d["target_error"][:, :T], np.float32)
                g = outer[q // 3, q % 3].subgridspec(3, 1, height_ratios=[1, 1, 0.35], hspace=0.08)
                axs = [fig.add_subplot(g[k]) for k in range(3)]
                for k, (j, lab, sc) in enumerate(((1, style.VTOR_LABEL, 1.0), (0, style.TI_LABEL, 1e-3))):
                    ax = axs[k]
                    ax.axvspan(t[ga] - 0.0025, t[gb] + 0.0025, color="0.88", zorder=0)
                    for (name, role, _, _), (mu, sg) in zip(models, preds):
                        style.model_line(ax, t[sl], mu[c, sl, j] * sc, sg[c, sl, j] * sc, role=role, label=name,
                                         marker="", lw=1.1)
                    ok = np.isfinite(y[c, sl, j])
                    ax.errorbar(t[sl][ok], y[c, sl, j][ok], yerr=ye[c, sl, j][ok], fmt="o", color="k", ms=2.5,
                                lw=0.7, label="target")
                    vals = np.r_[y[c, sl, j][ok], *(mu[c, sl, j] * sc for mu, _ in preds)]
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        q1, q9 = np.percentile(vals, [2, 98]); pad = 0.15 * (q9 - q1 + 1e-6)
                        ax.set_ylim(q1 - pad, q9 + pad)
                    ax.set_ylabel(lab, fontsize=8)
                    ax.tick_params(labelsize=7, labelbottom=False)
                    ax.set_xlim(t[lo], t[hi])
                axn = axs[2]
                axn.axvspan(t[ga] - 0.0025, t[gb] + 0.0025, color="0.88", zorder=0)
                axn.plot(t[sl], np.asarray(d["nbi_power_MW"], float)[sl], color="0.3", lw=1)
                axn.set_ylim(bottom=0); axn.set_xlim(t[lo], t[hi])
                axn.set_ylabel("NBI (MW)", fontsize=8); axn.tick_params(labelsize=7)
                axn.set_xlabel(style.TIME_LABEL, fontsize=8)
                shot = re.search(r"(\d{6})", e).group(1)
                axs[0].set_title(f"#{p0 + q + 1}  {shot} ({sp}), chord {c}, gap {t[ga]:.3f}-{t[gb]:.3f} s "
                                 f"({(gb + 1 - ga) * 5} ms){'' if t0 >= a.tmin else ', ramp-up'}",
                                 fontsize=9, loc="left")
                if q == 0:
                    axs[0].legend(fontsize=7, loc="best")
            pdf.savefig(fig)
            if p0 == 0:
                fig.savefig(a.out.with_suffix(".page1.png"), dpi=90)
            plt.close(fig)
            print(f"  page {p0 // a.per_page + 1}: gaps {p0 + 1}-{p0 + len(page)}", flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
