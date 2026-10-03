"""Label audit: where does the squared error live, and what is the ceiling
once label error is described by something other than the quoted sigma?

    pixi run python -u -m cer_transfer.analysis.label_audit --dump zdf_eval/nstx_ft_full.npz \
        [--vbound 380] [--spike-k 8] [--gap-factor 1.5] [--out audit.csv]

Input: a per-point dump from eval_checkpoint.py --dump-preds (the patched
version stores shot/frame provenance; a single-shot dump without it is
reshaped from the chord axis).

All flags are LABEL-ONLY (y, sigma, frame structure) -- no model output
enters any flag, so the QC is identical for every model and for the
ceilings, and is independent of any model output:

  edge   first/last labeled frame of each contiguous labeled run per
         (shot, chord). Beam-transition frames: partial beam-on exposure,
         mismatched background. No interior neighbours => invisible to any
         frame-difference consistency test.
  bound  |vtor| label within 2% of the fitter bound (fit pinned; the quoted
         sigma of such fits is meaningless).
  spike  interior label whose second difference
         e_t = y_t - (y_{t-1}+y_{t+1})/2 exceeds spike_k robust scatters
         (1.4826*MAD of e over that chord and target).

Ceilings reported on the kept set:
  nominal   1 - E[sigma^2]/Var(y)                (upper bound; sigma honest)
  diff      1 - (E[e^2]/1.5)/Var(y) on interior triples: MSE-based label
            reproducibility incl. tails; e^2 also contains true curvature,
            so this is a LOWER bound on the ceiling. The model should land
            between diff and nominal if it is label-limited.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np


def r2_kept(y, p, mask, t):
    """R2 of target column t on the masked points, NaN below 10 points."""
    return r2(y[mask, t], p[mask, t]) if mask.sum() > 10 else np.nan


def r2(y, p):
    """Coefficient of determination of p against y."""
    return 1.0 - np.mean((p - y) ** 2) / np.var(y)


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument(
        "--vbound",
        type=float,
        default=380.0,
        help="flag |vtor| >= 0.98*vbound (fitter bound, km/s); "
        "check the pile-up print below and set exactly",
    )
    ap.add_argument("--spike-k", type=float, default=8.0)
    ap.add_argument(
        "--gap-factor",
        type=float,
        default=1.5,
        help="run break if gap > gap_factor * median label gap",
    )
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = np.load(a.dump, allow_pickle=False)
    chord = d["chord"].astype(np.int64)
    y, p, s = (
        d["y"].astype(np.float64),
        d["pred"].astype(np.float64),
        d["sigma"].astype(np.float64),
    )
    names = [str(t) for t in d["targets"]]
    C = int(chord.max()) + 1
    if "shot" in d.files:
        shot, frame = d["shot"].astype(np.int64), d["frame"].astype(np.int64)
    else:  # single-shot legacy dump: chord-major (C, T) ravel
        T = len(chord) // C
        assert (chord.reshape(C, T) == np.arange(C)[:, None]).all()
        shot = np.zeros_like(chord)
        frame = np.tile(np.arange(T), C)
        print(
            "no provenance in dump -> treated as ONE shot " f"({C} chords x {T} frames)"
        )

    iv = names.index("vtor")
    lab = np.isfinite(y[:, iv]) & np.isfinite(p).all(1)
    # --- bound pile-up report --------------------------------------------
    av = np.abs(y[lab, iv])
    top = np.sort(av)[-2000:] if av.size > 2000 else np.sort(av)
    h, e_ = np.histogram(top, bins=20)
    print(
        f"|vtor| label max {av.max():.1f}; top-{len(top)} histogram "
        "(pile-up = fitter bound):"
    )
    print("  " + " ".join(f"{lo:.0f}:{n}" for lo, n in zip(e_[:-1], h)))

    edge = np.zeros(len(y), bool)
    spike = np.zeros(len(y), bool)
    e2 = [[] for _ in names]  # interior second differences^2
    em = [[] for _ in names]  # same stencil on the MODEL (attribution)
    nb_pairs = []  # (edge point, same-run inner neighbour)
    order = np.lexsort((frame, chord, shot))
    so, co = shot[order], chord[order]
    brk = np.r_[
        0, np.flatnonzero((so[1:] != so[:-1]) | (co[1:] != co[:-1])) + 1, len(order)
    ]
    lab_o = lab[order]
    for b0, b1 in zip(brk[:-1], brk[1:]):
        ii = order[b0:b1][lab_o[b0:b1]]
        if len(ii) == 0:
            continue
        fr = frame[ii]
        g = np.diff(fr)
        gmed = np.median(g) if len(g) else 1
        cut = g > a.gap_factor * gmed
        first = np.r_[True, cut]
        last = np.r_[cut, True]
        edge[ii[first | last]] = True
        # inner neighbour of each run edge (only runs of length >= 2)
        run_id = np.cumsum(np.r_[True, cut])
        for j in np.flatnonzero(first | last):
            if first[j] and j + 1 < len(ii) and run_id[j + 1] == run_id[j]:
                nb_pairs.append((ii[j], ii[j + 1]))
            elif last[j] and j >= 1 and run_id[j - 1] == run_id[j]:
                nb_pairs.append((ii[j], ii[j - 1]))
        # interior equal-gap triples
        if len(ii) >= 3:
            ok = (~cut[:-1]) & (~cut[1:]) & (g[:-1] == g[1:])
            i0, i1, i2 = ii[:-2][ok], ii[1:-1][ok], ii[2:][ok]
            for t in range(len(names)):
                et = y[i1, t] - 0.5 * (y[i0, t] + y[i2, t])
                m = np.isfinite(et)
                if not m.any():
                    continue
                e2[t].append((i1[m], et[m]))
                em[t].append(p[i1[m], t] - 0.5 * (p[i0[m], t] + p[i2[m], t]))
    for t in range(len(names)):
        if not e2[t]:
            continue
        idx = np.concatenate([u for u, _ in e2[t]])
        ev = np.concatenate([v for _, v in e2[t]])
        for c in np.unique(chord[idx]):
            m = chord[idx] == c
            mad = 1.4826 * np.median(np.abs(ev[m] - np.median(ev[m])))
            if mad > 0:
                spike[idx[m][np.abs(ev[m]) > a.spike_k * mad]] = True
        e2[t] = (idx, ev)
        em[t] = np.concatenate(em[t])
    bound = lab & (np.abs(y[:, iv]) >= 0.98 * a.vbound)
    flags = {"edge": edge & lab, "bound": bound, "spike": spike & lab}
    flags["ANY"] = flags["edge"] | flags["bound"] | flags["spike"]
    keep = lab & ~flags["ANY"]

    rows = []
    print(f"\nlabeled points {lab.sum():,}")
    for t, n in enumerate(names):
        err2 = (p[:, t] - y[:, t]) ** 2
        m = lab & np.isfinite(y[:, t])
        tot = err2[m].sum()
        V = np.var(y[m, t])
        print(f"\n=== {n} ===")
        print(
            f"  all:  R2 {r2(y[m, t], p[m, t]):.4f}  RMSE "
            f"{np.sqrt(err2[m].mean()):.4g}  nominal ceiling "
            f"{1 - np.nanmean(s[m, t] ** 2) / V:.4f}"
        )
        for k, f in flags.items():
            fm = f & m
            print(
                f"  flag {k:5s}: {fm.sum() / m.sum():6.2%} of labels carry "
                f"{err2[fm].sum() / tot:6.2%} of squared error"
            )
            rows.append([n, k, fm.sum() / m.sum(), err2[fm].sum() / tot])
        k_ = keep & np.isfinite(y[:, t])
        Vk = np.var(y[k_, t])
        nom = 1 - np.nanmean(s[k_, t] ** 2) / Vk
        idx, ev = (
            e2[t] if isinstance(e2[t], tuple) else (np.array([], int), np.array([]))
        )
        ki = keep[idx]
        diff = 1 - (np.mean(ev[ki] ** 2) / 1.5) / Vk if ki.any() else np.nan
        diff_all = 1 - (np.mean(ev**2) / 1.5) / V if len(ev) else np.nan
        rk = r2(y[k_, t], p[k_, t])
        print(
            f"  kept: R2 {rk:.4f}  RMSE {np.sqrt(err2[k_].mean()):.4g}  "
            f"| ceiling bracket [diff {diff:.4f}, nominal {nom:.4f}]"
        )
        print(
            f"  (diff ceiling on ALL interior triples, unfiltered: " f"{diff_all:.4f})"
        )
        rows.append([n, "kept_R2", rk, np.nan])
        rows.append([n, "kept_ceiling_diff", diff, np.nan])
        rows.append([n, "kept_ceiling_nominal", nom, np.nan])
        # --- attribution: is the flagged LABEL off, or the model? --------
        # edge: label jump to its inner run neighbour vs model jump to the
        # same neighbour label. spike: label 2nd difference vs model's.
        if nb_pairs:
            pe = np.array(nb_pairs)
            pe = pe[np.isfinite(y[pe[:, 0], t]) & np.isfinite(y[pe[:, 1], t])]
            j0, j1 = pe[:, 0], pe[:, 1]
            dl = np.abs(y[j0, t] - y[j1, t])  # label vs its neighbour
            dm = np.abs(p[j0, t] - y[j1, t])  # model vs that neighbour
            w = err2[j0]
            lab_off = dl > 2 * dm
            mod_off = dm > 2 * dl
            print(
                f"  edge attribution ({len(j0):,} pts): SSE share where "
                f"LABEL departs from its neighbour (dl>2dm) "
                f"{w[lab_off].sum() / w.sum():6.2%} | MODEL departs "
                f"(dm>2dl) {w[mod_off].sum() / w.sum():6.2%} | mixed "
                f"{w[~lab_off & ~mod_off].sum() / w.sum():6.2%}"
            )
            rows.append([n, "edge_sse_label_off", w[lab_off].sum() / w.sum(), np.nan])
            rows.append([n, "edge_sse_model_off", w[mod_off].sum() / w.sum(), np.nan])
        if len(ev):
            sp = spike[idx] & lab[idx]
            if sp.any():
                el, emv = np.abs(ev[sp]), np.abs(em[t][sp])
                w = err2[idx[sp]]
                print(
                    f"  spike attribution ({sp.sum():,} pts): SSE share "
                    f"where model stays smooth (|e_model|<0.25|e_label|) "
                    f"{w[emv < 0.25 * el].sum() / w.sum():6.2%} | model "
                    f"follows (|e_model|>0.5|e_label|) "
                    f"{w[emv > 0.5 * el].sum() / w.sum():6.2%}"
                )
                rows.append(
                    [
                        n,
                        "spike_sse_model_smooth",
                        w[emv < 0.25 * el].sum() / w.sum(),
                        np.nan,
                    ]
                )
        vm = np.abs(y[:, iv])
        q = np.quantile(vm[m], np.linspace(0, 1, a.n_bins + 1))
        q[-1] += 1e-9
        print("  |vtor| bins: SSE share all -> SSE share of flagged pts in bin")
        for b in range(a.n_bins):
            bm = m & (vm >= q[b]) & (vm < q[b + 1])
            print(
                f"    [{q[b]:6.1f},{q[b+1]:6.1f}) share "
                f"{err2[bm].sum() / tot:6.2%}  flagged-in-bin "
                f"{err2[bm & flags['ANY']].sum() / max(err2[bm].sum(), 1e-12):6.2%}"
                "  R2 kept "
                f"{r2_kept(y, p, bm & keep, t):.3f}"
            )
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["target", "item", "value", "sse_share"])
            w.writerows(rows)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
