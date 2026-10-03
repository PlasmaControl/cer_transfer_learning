"""Estimate label-noise ceilings on R2 for a split: pooled, per-chord, and
|vtor|-binned.

    pixi run python -u -m cer_transfer.analysis.ceiling_check --machine nstx \
        --list splits/nstx_val.txt [--per-chord] [--vtor-bins 5] [--limit N]

If labels y = truth + noise with per-point 1-sigma uncertainty sigma
(target_error), a perfect model scores, against the labels,

    R2_ceiling = 1 - E[sigma^2] / Var[y]

computed over whichever subset the mode selects: everything (pooled), one
chord (bounds per-chord-median R2), or one |vtor| quantile bin (same binning
as eval_checkpoint.py; aligns row-for-row with its tables). Canonical units
(ti eV, vtor km/s). Caveats: assumes target_error is honest 1-sigma random
noise, uncorrelated with truth; systematic fit biases shared by model and
labels are invisible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from joblib import load

from cer_transfer.configs import data_path, get_machine


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--machine", required=True)
    p.add_argument("--list", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = all files")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV of per-chord ceilings (with --per-chord)",
    )
    p.add_argument(
        "--per-chord",
        action="store_true",
        help="also report the PER-CHORD ceiling distribution",
    )
    p.add_argument(
        "--first-chords",
        type=int,
        default=0,
        metavar="N",
        help="restrict to chords [0, N) — match "
        "eval_checkpoint.py --first-chords for aligned "
        "denominators",
    )
    p.add_argument(
        "--honest",
        action="store_true",
        help="HONEST ceilings: inflate each point's sigma by its "
        "amplitude-decile's self-consistency factor "
        "k_d = median_z(d)/median_z(lowest decile) — "
        "corrects photon-statistics-only errors at high SNR "
        "(dynamics common to deciles cancels to 1st order; "
        "bright chords' faster dynamics make these ceilings "
        "slightly conservative). Implies --label-consistency.",
    )
    p.add_argument(
        "--label-consistency",
        action="store_true",
        help="label self-consistency: median |y(t+1)-y(t)| / "
        "sqrt(s_t^2+s_{t+1}^2) per amplitude decile. ~0.67 = "
        "honest sigmas + static plasma (median |N(0,1)|); "
        "real dynamics raise it; >>1 at high amplitude = "
        "errors overstate high-SNR precision (true ceiling "
        "lower than nominal)",
    )
    p.add_argument(
        "--vtor-bins",
        type=int,
        default=0,
        metavar="N",
        help="also report ceilings within N |vtor|-quantile bins",
    )
    args = p.parse_args()
    if args.honest:
        args.label_consistency = True

    machine = get_machine(args.machine)
    scale = np.asarray(machine.target_scale, dtype=np.float64)
    files = [
        Path(ln.strip())
        for ln in args.list.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if args.limit:
        files = files[: args.limit]

    n_t = len(machine.targets)
    C = machine.n_chords
    if args.first_chords:
        C = min(args.first_chords, C)
        print(f"restricted to chords [0, {C})")
    n = np.zeros(n_t)
    s1 = np.zeros(n_t)
    s2 = np.zeros(n_t)
    se2 = np.zeros(n_t)
    n_bad = np.zeros(n_t)
    cn = np.zeros((C, n_t))
    cs1 = np.zeros((C, n_t))
    cs2 = np.zeros((C, n_t))
    cse2 = np.zeros((C, n_t))
    sig_samp = [[[] for _ in range(n_t)] for _ in range(C)]  # strided sigmas
    keep = [[] for _ in range(n_t)]  # (y, sigma, |vtor|) for binned mode
    cons = [[] for _ in range(n_t)]  # (z_step, amp_proxy) for consistency
    hon = [[] for _ in range(n_t)]  # (y, sigma, amp) for honest ceilings

    for fp in files:
        d = load(data_path(fp), mmap_mode="r")
        end = int(d["end_index"])
        if end <= 0:
            continue
        y = np.asarray(d["target"][:, :end, :], dtype=np.float64) * scale
        s = np.asarray(d["target_error"][:, :end, :], dtype=np.float64) * scale
        if args.first_chords:
            y = y[: args.first_chords]
            s = s[: args.first_chords]

        m = np.isfinite(y)
        g = m & np.isfinite(s) & (s > 0)
        # pooled + per-chord accumulators
        n += m.sum(axis=(0, 1))
        n_bad += (m & ~g).sum(axis=(0, 1))
        s1 += np.where(m, y, 0.0).sum(axis=(0, 1))
        s2 += np.where(m, y**2, 0.0).sum(axis=(0, 1))
        se2 += np.where(g, s**2, 0.0).sum(axis=(0, 1))
        cn += m.sum(axis=1)
        cs1 += np.where(m, y, 0.0).sum(axis=1)
        cs2 += np.where(m, y**2, 0.0).sum(axis=1)
        if args.per_chord:
            for cc_ in range(C):
                for ti_ in range(n_t):
                    sv = s[cc_, :, ti_][m[cc_, :, ti_]]
                    if sv.size:
                        sig_samp[cc_][ti_].append(
                            sv[:: max(1, sv.size // 50)].astype(np.float32)
                        )
        cse2 += np.where(g, s**2, 0.0).sum(axis=1)
        if args.honest:
            spec_h = np.asarray(d["input"][:, :end, :], dtype=np.float64)
            if args.first_chords:
                spec_h = spec_h[: args.first_chords]
            med_h = np.median(spec_h, axis=-1, keepdims=True)
            amp_h = np.clip(spec_h - med_h, 0, None).sum(-1)  # (C, T)
            for i in range(n_t):
                yi, si = y[..., i], s[..., i]
                mm = np.isfinite(yi) & np.isfinite(si) & (si > 0) & np.isfinite(amp_h)
                if mm.any():
                    hon[i].append(
                        np.stack([yi[mm], si[mm], amp_h[mm]], axis=1).astype(np.float32)
                    )
        if args.label_consistency:
            # adjacent-frame label steps in units of combined sigma; the
            # amplitude proxy is the per-(c,t) line sum from raw counts
            spec = np.asarray(d["input"][:, :end, :], dtype=np.float64)
            if args.first_chords:
                spec = spec[: args.first_chords]
            med = np.median(spec, axis=-1, keepdims=True)
            amp = np.clip(spec - med, 0, None).sum(-1)  # (C, T)
            for i in range(n_t):
                yi, si = y[..., i], s[..., i]
                for cc in range(yi.shape[0]):
                    idx = np.where(
                        np.isfinite(yi[cc]) & np.isfinite(si[cc]) & (si[cc] > 0)
                    )[0]
                    if len(idx) < 2:
                        continue
                    gap = np.diff(idx)
                    ok = gap <= 10  # consecutive labeled frames only
                    if not ok.any():
                        continue
                    i0, i1 = idx[:-1][ok], idx[1:][ok]
                    z = np.abs(yi[cc, i1] - yi[cc, i0]) / np.sqrt(
                        si[cc, i1] ** 2 + si[cc, i0] ** 2
                    )
                    a2 = 0.5 * (amp[cc, i1] + amp[cc, i0])
                    gm = np.isfinite(z) & np.isfinite(a2)
                    if gm.any():
                        cons[i].append(
                            np.stack(
                                [z[gm], a2[gm], gap[ok][gm].astype(np.float32)], axis=1
                            ).astype(np.float32)
                        )
        if args.vtor_bins:
            vmag = np.abs(y[..., 1])
            for i in range(n_t):
                mi = m[..., i] & np.isfinite(vmag)
                sig_i = np.where(g[..., i], s[..., i], np.nan)
                keep[i].append(
                    np.stack([y[..., i][mi], sig_i[mi], vmag[mi]], axis=1).astype(
                        np.float32
                    )
                )

    print(
        f"machine={machine.name}, {len(files)} files, canonical units "
        f"(ti eV, vtor km/s)\n"
    )
    for i, name in enumerate(machine.targets):
        if n[i] == 0:
            print(f"{name}: no labeled points")
            continue
        var_y = s2[i] / n[i] - (s1[i] / n[i]) ** 2
        mean_s2 = se2[i] / max(n[i] - n_bad[i], 1)
        print(f"{name}:")
        print(
            f"  labeled points        {int(n[i]):,} "
            f"(bad/missing sigma on {int(n_bad[i]):,})"
        )
        print(f"  label std             {np.sqrt(var_y):10.4g}")
        print(f"  noise floor RMSE      {np.sqrt(mean_s2):10.4g}")
        print(f"  R2 ceiling            {1 - mean_s2 / var_y:10.4f}")

    if args.per_chord:
        print("\nper-chord ceilings (1 - E[sigma^2]/Var_chord):")
        ceil_cols = {}
        for i, name in enumerate(machine.targets):
            var_c = (
                cs2[:, i] / np.maximum(cn[:, i], 1)
                - (cs1[:, i] / np.maximum(cn[:, i], 1)) ** 2
            )
            ms2 = cse2[:, i] / np.maximum(cn[:, i], 1)
            ceil = 1.0 - ms2 / np.maximum(var_c, 1e-12)
            ok = cn[:, i] > 500
            v = ceil[ok]
            print(
                f"  {name}: min {v.min():.3f} | p25 "
                f"{np.percentile(v, 25):.3f} | median {np.median(v):.3f} "
                f"| p75 {np.percentile(v, 75):.3f} | max {v.max():.3f}   "
                f"({int(ok.sum())}/{C} chords with >500 pts)"
            )
            ceil_cols[f"r2_{name}"] = np.where(ok, ceil, np.nan)
            ceil_cols[f"floor_{name}"] = np.where(ok, np.sqrt(ms2), np.nan)
            med_s = np.array(
                [
                    (
                        np.median(np.concatenate(sig_samp[cc_][i]))
                        if sig_samp[cc_][i]
                        else np.nan
                    )
                    for cc_ in range(C)
                ]
            )
            ceil_cols[f"medfloor_{name}"] = np.where(ok, med_s, np.nan)
        if args.out:
            import csv

            with open(args.out, "w", newline="") as f:
                w = csv.writer(f)
                names = list(ceil_cols)
                w.writerow(["chord"] + names)
                for cch in range(C):
                    w.writerow([cch] + [f"{ceil_cols[k][cch]:.5f}" for k in names])
            print(f"wrote {args.out}")

    if args.vtor_bins:
        print(f"\n|vtor|-binned ceilings ({args.vtor_bins} quantile bins):")
        v_all = np.concatenate([k[:, 2] for k in keep[1]])
        edges = np.quantile(v_all, np.linspace(0, 1, args.vtor_bins + 1))
        edges[-1] += 1e-9
        for i, name in enumerate(machine.targets):
            arr = np.concatenate(keep[i], axis=0)
            print(f"  {name}:")
            for b in range(args.vtor_bins):
                bm = (arr[:, 2] >= edges[b]) & (arr[:, 2] < edges[b + 1])
                yb = arr[bm, 0]
                sb = arr[bm, 1]
                var_b = yb.var()
                ms2 = np.nanmean(sb.astype(np.float64) ** 2)
                print(
                    f"    [{edges[b]:7.1f},{edges[b+1]:7.1f}) "
                    f"n={int(bm.sum()):>10,}  label std "
                    f"{np.sqrt(var_b):9.4g}  noise floor "
                    f"{np.sqrt(ms2):9.4g}  ceiling "
                    f"{1 - ms2 / max(var_b, 1e-12):7.4f}"
                )

    if args.label_consistency:
        print(
            "\nlabel self-consistency (adjacent-frame steps / combined "
            "sigma),\nmedian per line-amplitude decile "
            "(0.67 ~= honest sigmas + static plasma; dynamics raise it;\n"
            ">>1 at high amplitude => overconfident high-SNR errors):"
        )
        for i, name in enumerate(machine.targets):
            if not cons[i]:
                print(f"  {name}: no consecutive labeled pairs (gap <= 10)")
                continue
            arr = np.concatenate(cons[i], axis=0)
            z, a = arr[:, 0].astype(np.float64), arr[:, 1]
            print(
                f"  {name}: median label-frame gap "
                f"{np.median(arr[:, 2]):.0f} frames"
            )
            edges = np.quantile(a, np.linspace(0, 1, 11))
            edges[-1] += 1e-9
            row = []
            for b in range(10):
                bm = (a >= edges[b]) & (a < edges[b + 1])
                row.append(np.median(z[bm]) if bm.any() else np.nan)
            print(
                f"  {name}: "
                + " ".join(f"{v:5.2f}" for v in row)
                + f"   (overall median {np.median(z):.2f}, n={len(z):,})"
            )
    if args.honest:
        print(
            "\nHONEST ceilings (sigma inflated by per-amplitude-decile "
            "self-consistency factor,\nnormalized to lowest decile; "
            "slightly conservative where bright chords evolve faster):"
        )
        for i, name in enumerate(machine.targets):
            if not cons[i] or not hon[i]:
                print(f"  {name}: insufficient data")
                continue
            carr = np.concatenate(cons[i], axis=0)
            zc, ac = carr[:, 0].astype(np.float64), carr[:, 1]
            edges = np.quantile(ac, np.linspace(0, 1, 11))
            edges[-1] += 1e-9
            zmed = np.array(
                [
                    np.median(zc[(ac >= edges[b]) & (ac < edges[b + 1])])
                    for b in range(10)
                ]
            )
            k = np.maximum(zmed / max(zmed[0], 1e-9), 1.0)
            harr = np.concatenate(hon[i], axis=0)
            yv = harr[:, 0].astype(np.float64)
            sv = harr[:, 1].astype(np.float64)
            av = harr[:, 2]
            db = np.clip(np.searchsorted(edges, av, side="right") - 1, 0, 9)
            s_hon = sv * k[db]
            var_y = yv.var()
            ceil_nom = 1 - (sv**2).mean() / var_y
            ceil_hon = 1 - (s_hon**2).mean() / var_y
            print(
                f"  {name}: nominal {ceil_nom:7.4f} -> honest "
                f"{ceil_hon:7.4f}   (inflation k by decile: "
                + " ".join(f"{v:.1f}" for v in k)
                + f"; honest floor RMSE {np.sqrt((s_hon**2).mean()):.4g})"
            )
    print(
        "\nassumes target_error = 1-sigma random noise, uncorrelated with "
        "truth;\nshared systematic fit biases are invisible to this "
        "estimate."
    )


if __name__ == "__main__":
    main()
