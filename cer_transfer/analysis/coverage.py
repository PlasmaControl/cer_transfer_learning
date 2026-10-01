"""Empirical coverage of the 1-sigma and 2-sigma intervals (Table S6):
raw (k = 1) and calibrated (fitted k), on all labels and on consistent
labels.

z = (y - mu) / sqrt((k * sigma_model)^2 + sigma_label^2), as in the
calibration. Consistent labels follow the label-only criteria of the
Methods: (i) run edges -- per chord, labels form runs broken wherever the
gap to the next labeled frame exceeds 1.5x the median gap; the first and
last label of every run are flagged; (ii) spikes -- for interior labels
with equally spaced neighbours, e = y_t - (y_{t-1} + y_{t+1})/2, flagged
if |e| > 8 x 1.4826 x MAD(e) of that chord. Works on a pooled
eval_checkpoint.py --dump-preds file: every discharge starts with
unlabeled pre-trigger frames, so runs break at discharge boundaries.

    pixi run python -u -m cer_transfer.analysis.coverage --preds gallery/ft_TEST.npz \
        --k 0.992 1.171
"""
import argparse

import numpy as np


def consistent_mask(y):
    """y: (C, T) labels of one target; returns (C, T) bool, True = kept."""
    C, T = y.shape
    keep = np.zeros((C, T), bool)
    for c in range(C):
        idx = np.where(np.isfinite(y[c]))[0]
        if idx.size < 3:
            continue
        gaps = np.diff(idx)
        med = np.median(gaps)
        brk = gaps > 1.5 * med
        edge = np.zeros(idx.size, bool)
        edge[0] = edge[-1] = True
        edge[1:][brk] = True      # first label after a break
        edge[:-1][brk] = True     # last label before a break
        # spikes: interior labels with equally spaced neighbours
        spike = np.zeros(idx.size, bool)
        eq = np.zeros(idx.size, bool)
        eq[1:-1] = (gaps[:-1] == gaps[1:]) & ~brk[:-1] & ~brk[1:]
        if eq.any():
            v = y[c, idx]
            e = np.full(idx.size, np.nan)
            e[1:-1] = v[1:-1] - 0.5 * (v[:-2] + v[2:])
            ee = e[eq]
            mad = 1.4826 * np.median(np.abs(ee - np.median(ee)))
            if mad > 0:
                spike[eq] = np.abs(e[eq]) > 8 * mad
        keep[c, idx[~edge & ~spike]] = True
    return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True)
    p.add_argument("--k", type=float, nargs=2, default=[1.0, 1.0],
                   help="calibration factors k for Ti and vtor")
    args = p.parse_args()
    d = np.load(args.preds)
    ch = d["chord"]
    C = int(ch.max()) + 1
    T = len(ch) // C
    y = d["y"].reshape(C, T, -1)
    mu = d["pred"].reshape(C, T, -1)
    sl = d["sigma"].reshape(C, T, -1)
    sm = d["pred_sigma"].reshape(C, T, -1)
    names = ("Ti", "vtor")
    rows = []
    for t, name in enumerate(names):
        lab = np.isfinite(y[..., t]) & np.isfinite(mu[..., t])
        cons = consistent_mask(y[..., t]) & lab
        frac = cons.sum() / max(lab.sum(), 1)
        print(f"{name}: {lab.sum()} labeled points, consistent "
              f"{100 * frac:.1f}% (manuscript: 83.1%)")
        for kname, k in (("raw", 1.0), ("calibrated", args.k[t])):
            s = np.sqrt((k * sm[..., t]) ** 2 + np.nan_to_num(sl[..., t]) ** 2)
            z = np.abs(y[..., t] - mu[..., t]) / s
            for sub, m in (("all", lab), ("consistent", cons)):
                zz = z[m]
                c1 = 100 * np.mean(zz < 1)
                c2 = 100 * np.mean(zz < 2)
                rows.append((name, kname, sub, c1, c2, np.median(zz)))
    print(f"\n{'target':6s} {'':11s} {'labels':11s} {'1sigma':>7s} "
          f"{'2sigma':>7s} {'median|z|':>9s}   (Gaussian: 68.3 / 95.4 / 0.674)")
    for r in rows:
        print(f"{r[0]:6s} {r[1]:11s} {r[2]:11s} {r[3]:6.1f}% {r[4]:6.1f}% "
              f"{r[5]:9.3f}")
    print("\nLaTeX rows (target & calibration & labels & 1sigma & 2sigma):")
    for r in rows:
        tt = "$T_\\mathrm{i}$" if r[0] == "Ti" else "$v_\\mathrm{tor}$"
        print(f"{tt} & {r[1]} & {r[2]} & \\SI{{{r[3]:.1f}}}{{\\percent}} & "
              f"\\SI{{{r[4]:.1f}}}{{\\percent}} \\\\")


if __name__ == "__main__":
    main()
