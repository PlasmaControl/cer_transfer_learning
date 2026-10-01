"""Per-chord median |error| of several models against the per-chord
label-noise floor for the MEDIAN ABSOLUTE ERROR of an ideal model:

    floor_c = 0.674 * median_c( k_d * sigma )

0.674 = median |N(0,1)| (an ideal model's median |error| under Gaussian
label noise of scale s is 0.674 s, not s); k_d = per-line-amplitude-decile
self-consistency factor of the quoted sigma (as cer_transfer.analysis.ceiling_check --honest:
median adjacent-frame z per decile / lowest decile, clipped at 1).
Supersedes cer_transfer.figures.transfer_floor (kept unchanged; it draws median sigma).

Inputs: per-point dumps from eval_checkpoint.py --dump-preds (patched, with
shot/frame/amp provenance), one per model, all on the SAME list. The floor
comes from the labels of the first dump.

    pixi run --frozen python -u -m cer_transfer.figures.transfer_floor_honest \
        --runs "from scratch (7307):audit/nstx_scratch.npz" \
               "fine-tuned (325):audit/nstx_ft_r325.npz:--" \
               "fine-tuned (7307):audit/ft_full.npz" \
        --out figs --csv figs/transfer_floor.csv [--consistent] [--nominal]

--consistent: models AND floor on consistent labels only (label-only
run-edge + spike flags, same rules as cer_transfer.analysis.label_audit with the bound flag off).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
MED_ABS_N = 0.6745
MIN_PTS = 500


def groups(shot, chord, frame, lab):
    """Index arrays of labeled points per (shot, chord), frame-sorted."""
    order = np.lexsort((frame, chord, shot))
    so, co = shot[order], chord[order]
    brk = np.r_[0, np.flatnonzero((so[1:] != so[:-1]) | (co[1:] != co[:-1]))
                + 1, len(order)]
    lo = lab[order]
    for b0, b1 in zip(brk[:-1], brk[1:]):
        ii = order[b0:b1][lo[b0:b1]]
        if len(ii):
            yield ii


def consistent_mask(y, chord, shot, frame, lab, spike_k=8.0, gap_factor=1.5):
    """Label-only QC, same rules as cer_transfer.analysis.label_audit (bound flag off)."""
    flag = np.zeros(len(chord), bool)
    trip = [[] for _ in range(y.shape[1])]
    for ii in groups(shot, chord, frame, lab):
        g = np.diff(frame[ii])
        cut = g > gap_factor * (np.median(g) if len(g) else 1)
        flag[ii[np.r_[True, cut] | np.r_[cut, True]]] = True
        if len(ii) >= 3:
            ok = (~cut[:-1]) & (~cut[1:]) & (g[:-1] == g[1:])
            i0, i1, i2 = ii[:-2][ok], ii[1:-1][ok], ii[2:][ok]
            for t in range(y.shape[1]):
                e = y[i1, t] - 0.5 * (y[i0, t] + y[i2, t])
                m = np.isfinite(e)
                trip[t].append((i1[m], e[m]))
    for t in range(y.shape[1]):
        if not trip[t]:
            continue
        idx = np.concatenate([u for u, _ in trip[t]])
        ev = np.concatenate([v for _, v in trip[t]])
        for c in np.unique(chord[idx]):
            m = chord[idx] == c
            mad = 1.4826 * np.median(np.abs(ev[m] - np.median(ev[m])))
            if mad > 0:
                flag[idx[m][np.abs(ev[m]) > spike_k * mad]] = True
    return lab & ~flag


def k_decile(y, s, amp, chord, shot, frame, lab, t, max_gap=10):
    """Self-consistency inflation per amplitude decile; (edges, k>=1)."""
    zs, az = [], []
    ok_s = lab & np.isfinite(s[:, t]) & (s[:, t] > 0)
    for ii in groups(shot, chord, frame, ok_s):
        if len(ii) < 2:
            continue
        ok = np.diff(frame[ii]) <= max_gap
        i0, i1 = ii[:-1][ok], ii[1:][ok]
        z = np.abs(y[i1, t] - y[i0, t]) / np.sqrt(s[i1, t] ** 2
                                                   + s[i0, t] ** 2)
        av = 0.5 * (amp[i1] + amp[i0])
        m = np.isfinite(z) & np.isfinite(av)
        zs.append(z[m]); az.append(av[m])
    z, av = np.concatenate(zs), np.concatenate(az)
    edges = np.quantile(av, np.linspace(0, 1, 11))
    edges[-1] += 1e-9
    zmed = np.array([np.median(z[(av >= edges[b]) & (av < edges[b + 1])])
                     for b in range(10)])
    return edges, np.maximum(zmed / max(zmed[0], 1e-9), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="'label:dump.npz[:linestyle]' per model, plot order")
    ap.add_argument("--consistent", action="store_true")
    ap.add_argument("--nominal", action="store_true",
                    help="also draw 0.674*sigma (no k_d) as a dotted line")
    ap.add_argument("--log-y", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("figs"))
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--csv", type=Path, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    name = a.out_name or ("transfer_floor_consistent" if a.consistent
                          else "transfer_floor")

    runs = []
    for spec in a.runs:
        parts = spec.split(":")
        runs.append((parts[0], parts[1], parts[2] if len(parts) > 2 else "-"))

    d0 = np.load(runs[0][1])
    chord = d0["chord"].astype(np.int64)
    shot, frame = d0["shot"].astype(np.int64), d0["frame"].astype(np.int64)
    y, s = d0["y"].astype(np.float64), d0["sigma"].astype(np.float64)
    amp = d0["amp"].astype(np.float64)   # log10(m0)/4: monotone in m0
    tg = [str(v) for v in d0["targets"]]
    C, T = int(chord.max()) + 1, y.shape[1]
    lab = np.isfinite(y).all(1)
    sel = consistent_mask(y, chord, shot, frame, lab) if a.consistent \
        else lab
    print(f"labeled {lab.sum():,}; evaluated {sel.sum():,} "
          f"({sel.sum() / lab.sum():.1%})")

    floor = np.full((C, T), np.nan)
    nom = np.full((C, T), np.nan)
    for t in range(T):
        edges, k = k_decile(y, s, amp, chord, shot, frame, lab, t)
        db = np.clip(np.searchsorted(edges, amp, side="right") - 1, 0, 9)
        sh = s[:, t] * k[db]
        print(f"{tg[t]}: k_d by decile " + " ".join(f"{v:.2f}" for v in k))
        for c in range(C):
            m = sel & (chord == c) & np.isfinite(sh) & (sh > 0)
            if m.sum() > MIN_PTS:
                floor[c, t] = MED_ABS_N * np.median(sh[m])
                nom[c, t] = MED_ABS_N * np.median(s[m, t])

    medae = []
    for label, path, _ in runs:
        d = np.load(path)
        assert np.array_equal(d["shot"], d0["shot"]) and \
            np.array_equal(d["frame"], d0["frame"]), f"{path}: other list"
        p = d["pred"].astype(np.float64)
        me = np.full((C, T), np.nan)
        for c in range(C):
            m = sel & (chord == c)
            if m.sum() > MIN_PTS:
                me[c] = np.median(np.abs(p[m] - y[m]), axis=0)
        medae.append(me)
        r = me / floor
        print(f"{label}: medae/floor per chord, median [min, max]: " +
              " | ".join(f"{tg[t]} {np.nanmedian(r[:, t]):.2f} "
                         f"[{np.nanmin(r[:, t]):.2f}, {np.nanmax(r[:, t]):.2f}]"
                         for t in range(T)))

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["chord"] + [f"floor_{x}" for x in tg]
                       + [f"floor_nominal_{x}" for x in tg]
                       + [f"medae_{x}_{r[0]}" for r in runs for x in tg])
            for c in range(C):
                w.writerow([c] + [f"{v:.5g}" for v in floor[c]]
                           + [f"{v:.5g}" for v in nom[c]]
                           + [f"{me[c, t]:.5g}" for me in medae
                              for t in range(T)])
        print(f"wrote {a.csv}")

    cx = np.arange(C)
    ylab = ("median $|T_i$ error$|$ (eV)", "median $|v_{tor}$ error$|$ (km/s)")
    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            fig, axes = plt.subplots(
                1, 2, figsize=(7.2, 3.0) if ctx == "paper" else (12, 5))
            for t, ax in enumerate(axes):
                vals = np.concatenate([floor[:, t]] + [m[:, t] for m in medae])
                lo = 0.6 * np.nanmin(vals[vals > 0])
                ok = np.isfinite(floor[:, t])
                ax.fill_between(cx[ok], lo if a.log_y else 0, floor[ok, t],
                                color="0.82", zorder=0,
                                label="label-noise floor" if t == 0 else None)
                if a.nominal:
                    ax.plot(cx, nom[:, t], ":", color="0.4", lw=1,
                            label="nominal (quoted $\\sigma$)"
                            if t == 0 else None)
                for i, ((label, _, ls), me) in enumerate(zip(runs, medae)):
                    ax.plot(cx, me[:, t], ls, lw=1.6, color=f"C{i}",
                            label=label if t == 0 else None)
                ax.set_xlabel("chord (core $\\rightarrow$ edge)")
                ax.set_ylabel(ylab[t])
                if a.log_y:
                    ax.set_yscale("log")
                else:
                    ax.set_ylim(bottom=0)
            axes[0].legend(fontsize="small", loc="upper right")
            fig.tight_layout()
            fig.savefig(a.out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {a.out}/{name}.png/.pdf")


if __name__ == "__main__":
    main()
