"""Beam-off (passive CER) verdict: dual-array agreement, label-free
calibration check, re-entry error, and sigma inflation.

Beam-off frames are plasma-active frames without fits (as in
cer_transfer.beamoff.scan). The two arrays view the same flux surfaces through
independent optics, so their agreement during beam-off is a label-free
validation, and the ratio of their disagreement to the predicted
combined sigma tests whether the uncertainties remain literal without
labels. Beam-ON frames provide the self-baseline for both statistics,
so no absolute calibration is assumed.

    pixi run python -u -m cer_transfer.beamoff.analyze --shot 116614 \
        --bg gallery/bo_116614_bg.npz --fg gallery/bo_116614_fg.npz \
        --bg-file "$(grep 116614 splits/nstx_passive_val.txt)" \
        [--trim-head-end 2] [--tau]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cer_transfer.figures.common import load_predictions
from joblib import load

from cer_transfer.beamstate import valid_end
from cer_transfer.configs import data_path


def arr(npz):
    """Prediction dump as (pred, pred_sigma, y, T) with (C, T, K) arrays."""
    dmp = load_predictions(npz)
    return dmp.pred, dmp.pred_sigma, dmp.y, dmp.T


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=str, required=True)
    p.add_argument("--bg", type=Path, required=True)
    p.add_argument("--fg", type=Path, required=True)
    p.add_argument(
        "--bg-file",
        type=Path,
        required=True,
        help="background joblib (masks: labels + activity)",
    )
    p.add_argument(
        "--trim-head-end",
        type=int,
        default=0,
        help="drop the last N frames of the pre-beam head "
        "(beam already on, fits not yet started)",
    )
    p.add_argument(
        "--fg-file",
        type=Path,
        default=None,
        help="active-array joblib: if given, beam-off is "
        "determined from the spectra (active/background "
        "brightness ratio collapses when the viewed beam "
        "is off) instead of from missing fits",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="keep unverified beam-off segments (default: only "
        "phases bounded by clean beam switches)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="append one summary row per shot (pooling)",
    )
    p.add_argument(
        "--tau",
        action="store_true",
        help="fit exponential decay time to core rotation " "over the post-beam tail",
    )
    args = p.parse_args()

    pb, sb, yb, Tb = arr(args.bg)
    pf, sf, yf, Tf = arr(args.fg)
    T = min(Tb, Tf)

    d = load(data_path(args.bg_file), mmap_mode="r")
    end = valid_end(d)
    spec = np.asarray(d["input"][:, :end, :], dtype=np.float32)
    T = min(T, end)
    lab = np.isfinite(np.asarray(d["target"][:, :T, 0])).any(axis=0)
    tot = spec.sum(axis=(0, 2))[:T]
    PRE = min(47, T - 1)  # pre-trigger frames: pre-plasma by construction
    dw = tot[: max(8, PRE - 2)]
    dk = np.median(dw)
    nz = 1.4826 * np.median(np.abs(dw - dk))
    active = tot > dk + 8 * max(nz, 1e-6 * abs(dk) + 1e-6)
    active[:PRE] = False

    li = np.where(lab)[0]
    lo, hi = li.min(), li.max()
    off = active & ~lab
    if args.fg_file is not None:
        from cer_transfer.beamstate import beam_state

        st = beam_state(args.bg_file, args.fg_file)
        om = np.zeros(T, bool)
        n_ = min(T, st["T"])
        om[:n_] = st["off_raw" if args.no_verify else "off"][:n_]
        off = om & active
        rej = [s_ for s_ in st["segments"] if not s_[3]]
        print(
            f"beam state from spectra: R_on {st['R_on']:.3g} | R_low "
            f"{st['R_low']:.3g} | threshold {st['thr']:.3g} | "
            f"{len(st['segments'])} beam-off segments, "
            f"{len(st['segments']) - len(rej)} verified"
            + (
                ""
                if not rej
                else " | rejected: "
                + ", ".join(f"[{a_}..{b_}] {w_}" for a_, b_, _, _, w_ in rej)
            )
        )
    # trim contaminated head end
    if args.trim_head_end:
        h = lo - 1
        k = args.trim_head_end
        while h >= 0 and off[h] and k > 0:
            off[h] = False
            h -= 1
            k -= 1
    on = lab.copy()

    names = ("Ti (eV)", "vtor (km/s)")
    head_m = off.copy()
    head_m[lo:] = False
    gap_m = off.copy()
    gap_m[:lo] = False
    gap_m[hi + 1 :] = False
    tail_m = off.copy()
    tail_m[: hi + 1] = False
    classes = [
        ("ALL beam-off", off),
        ("notches (in-distribution)", gap_m),
        ("post-beam tail", tail_m),
        ("startup head (out-of-distribution)", head_m),
    ]
    print(
        f"shot {args.shot}: {int(off.sum())} beam-off frames "
        f"({int(head_m.sum())} head / {int(gap_m.sum())} notches / "
        f"{int(tail_m.sum())} tail), {int(on.sum())} fitted frames"
    )
    row = {
        "shot": args.shot,
        "n_off": int(off.sum()),
        "n_head": int(head_m.sum()),
        "n_notch": int(gap_m.sum()),
        "n_tail": int(tail_m.sum()),
    }
    for t, n in enumerate(names):
        tn = ("ti", "vtor")[t]
        dfb = pf[:, :T, t] - pb[:, :T, t]
        comb = np.sqrt(sf[:, :T, t] ** 2 + sb[:, :T, t] ** 2)
        sb_t = sb[:, :T, t]
        agree_on = np.median(np.abs(dfb[:, on]))
        z_on = np.median(np.abs(dfb[:, on] / comb[:, on]))
        s_on = np.median(sb_t[:, on])
        print(
            f"  {n}:  [beam-on baseline: |diff| {agree_on:.3g}, "
            f"|z| {z_on:.3f}, sigma {s_on:.3g}]"
        )
        for cname, cm_ in classes:
            if cm_.sum() < 4:
                continue
            a = np.median(np.abs(dfb[:, cm_]))
            z = np.median(np.abs(dfb[:, cm_] / comb[:, cm_]))
            s = np.median(sb_t[:, cm_])
            print(
                f"    {cname:36s} |diff| {a:8.3g} (x{a/agree_on:4.2f}) "
                f"| |z| {z:.3f} | sigma x{s/s_on:.2f}"
            )
            if cname == "ALL beam-off":
                row.update(
                    {
                        f"{tn}_diff": a,
                        f"{tn}_diff_ratio": a / agree_on,
                        f"{tn}_z": z,
                        f"{tn}_sig_infl": s / s_on,
                    }
                )
        # label-free continuity: step size across beam on/off boundaries
        # relative to the typical frame-to-frame step while the beam is
        # on. A model that does not need the beam should not notice the
        # transition (ratio ~1); a model relying on the beam's signal
        # jumps (ratio >> 1).
        onb = active & ~off
        bnd = np.where(off[:-1] != off[1:])[0]
        if bnd.size:
            jumps = []
            for pr in (pb, pf):
                st = np.abs(np.diff(pr[:, :T, t], axis=1))
                base = np.median(st[:, onb[:-1] & onb[1:]])
                jumps.append(np.median(st[:, bnd]) / max(base, 1e-9))
            row.update({f"{tn}_step_bg": jumps[0], f"{tn}_step_fg": jumps[1]})
            sf_t = sf[:, :T, t]
            s_on_f = np.median(sf_t[:, on])
            s_off_f = np.median(sf_t[:, off]) if off.any() else np.nan
            print(
                f"    step at beam transitions / typical step: "
                f"background model x{jumps[0]:.2f} | active-array "
                f"model x{jumps[1]:.2f}   ({bnd.size} transitions; "
                f"active-array sigma beam-off x{s_off_f/s_on_f:.2f})"
            )
        # re-entry: first 3 fitted frames after each off-segment
        segs = []
        f = 0
        while f < T:
            if off[f]:
                g = f
                while f < T and off[f]:
                    f += 1
                segs.append((g, f - 1))
            else:
                f += 1
        errs, glob = [], []
        yb_t = yb[:, :T, t]
        for _, e in segs:
            nxt = [
                f for f in range(e + 1, min(e + 40, T)) if np.isfinite(yb_t[:, f]).any()
            ][:3]
            for f in nxt:
                m = np.isfinite(yb_t[:, f])
                errs.append(np.abs(yb_t[m, f] - pb[m, f, t]))
        m = np.isfinite(yb_t)
        glob = np.abs(yb_t[m] - pb[:, :T, t][m])
        if errs:
            errs = np.concatenate(errs)
            print(
                f"    re-entry medae (first fits after beam-off): "
                f"{np.median(errs):.3g} | global {np.median(glob):.3g}"
            )
            row[f"{tn}_reentry_ratio"] = np.median(errs) / np.median(glob)
        if args.tau and t == 1 and off[hi + 1 :].sum() >= 8:
            fr = np.arange(hi + 1, T)[off[hi + 1 :]]
            v = np.nanmean(pb[:6, fr, 1], axis=0)
            good = v > 5.0
            if good.sum() >= 6:
                A = np.vstack([fr[good] / 200.0, np.ones(good.sum())]).T
                sl, _ = np.linalg.lstsq(A, np.log(v[good]), rcond=None)[0:2][0]
                if sl < 0:
                    print(
                        f"    post-beam core rotation decay: tau_phi "
                        f"= {-1.0/sl*1000:.0f} ms over "
                        f"{good.sum()} frames"
                    )

    if args.csv is not None:
        import csv

        fields = ["shot", "n_off", "n_head", "n_notch", "n_tail"] + [
            f"{tn}_{k}"
            for tn in ("ti", "vtor")
            for k in (
                "diff",
                "diff_ratio",
                "z",
                "sig_infl",
                "step_bg",
                "step_fg",
                "reentry_ratio",
            )
        ]
        new = not args.csv.exists()
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval="")
            if new:
                w.writeheader()
            w.writerow(
                {
                    k: (f"{float(v):.4g}" if isinstance(v, (float, np.floating)) else v)
                    for k, v in row.items()
                }
            )


if __name__ == "__main__":
    main()
