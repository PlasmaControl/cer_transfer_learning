"""Edge-frame test for beam-off reconstructions, with ground truth.

At every verified beam switch, compare the passive reconstruction in a
BEAM-OFF frame with the conventional CHERS fit a few frames away on the
beam-on side of the switch (the plasma evolves slowly compared with
5-15 ms). The identical procedure applied to BEAM-ON test frames gives
the baseline, so the headline number is off/on under the same
conditions (same distance to the reference fit, same shots).

  entry edge: test frame a+guard (first beam-off frames), reference =
              last fit before the switch, at most --max-gap frames away
  exit edge:  test frame b-guard (last beam-off frames), reference =
              first fit after the switch
  reference change: |fit(f+d) - fit(f)| between beam-on fits at the same
              distance d -- plasma evolution + label noise, the floor any
              reconstruction can reach in this test

    pixi run python -u -m cer_transfer.beamoff.edge_frame_test --shot 115520 \\
        --bg gallery/bo_115520_bg.npz \\
        --bg-file "$(cat gallery/bo_115520_bg.txt)" \\
        --fg-file "$(cat gallery/bo_115520_fg.txt)" \\
        [--csv gallery/edge_pool.csv]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cer_transfer.beamstate import beam_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=str, required=True)
    p.add_argument("--bg", type=Path, required=True)
    p.add_argument("--bg-file", type=Path, required=True)
    p.add_argument("--fg-file", type=Path, required=True)
    p.add_argument("--guard", type=int, default=1,
                   help="skip this many frames next to the switch (frames "
                        "in which the beam changes mid-exposure)")
    p.add_argument("--max-gap", type=int, default=3,
                   help="max frames between the switch and the reference fit")
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--t-offset", type=float, default=-0.235)
    p.add_argument("--exclude-head", action="store_true",
                   help="drop pairs at the exit of the startup phase (before "
                        "the first beam injection)")
    p.add_argument("--min-after-first-fit", type=float, default=0.0,
                   help="ignore test frames earlier than the first fit + this "
                        "(s), for beam-off AND beam-on frames (startup transient, "
                        "unreliable first fits)")
    p.add_argument("--verbose", action="store_true",
                   help="print every edge pair: phase, side, times, and the "
                        "median difference over chords")
    args = p.parse_args()

    d = np.load(args.bg)
    ch = d["chord"]
    C = int(ch.max()) + 1
    Tn = len(ch) // C
    pred = d["pred"].reshape(C, Tn, -1)
    y = d["y"].reshape(C, Tn, -1)
    st = beam_state(args.bg_file, args.fg_file)
    T = min(Tn, st["T"])
    pred, y = pred[:, :T], y[:, :T]
    active = st["active"][:T]
    fitted = np.isfinite(y[..., 0]).any(axis=0)
    off = np.zeros(T, bool)
    segs = [(a, min(b, T - 1), cls) for a, b, cls, ok, _ in st["segments"]
            if ok and a < T]
    for a, b, _ in segs:
        off[a:b + 1] = True
    on = active & ~off
    g, mg = args.guard, args.max_gap

    fit_idx = np.where(fitted)[0]
    t_min = (int(fit_idx.min()) + int(round(args.min_after_first_fit * 200))
             if fit_idx.size else 0)
    on = on & (np.arange(T) >= t_min)
    pairs = []  # (test frame, reference fit frame, class, side)
    for a, b, cls in segs:
        if args.exclude_head and cls == "head":
            continue
        ft = a + g
        if ft <= b:
            refs = [f for f in range(a - 1, max(a - 1 - mg, -1), -1)
                    if fitted[f] and on[f]]
            if refs:
                pairs.append((ft, refs[0], cls, "entry"))
        ft = b - g
        if ft >= a:
            refs = [f for f in range(b + 1, min(b + 1 + mg, T))
                    if fitted[f] and on[f]]
            if refs:
                pairs.append((ft, refs[0], cls, "exit"))

    pairs = [pp for pp in pairs if pp[0] >= t_min and pp[1] >= t_min]
    names = (("ti", "eV"), ("vtor", "km/s"))
    if args.verbose:
        to = args.t_offset
        for ft, fr, cls, side in pairs:
            m = np.isfinite(y[:, fr, 1]) & np.isfinite(pred[:, ft, 1])
            dv = np.median(np.abs(pred[m, ft, 1] - y[m, fr, 1])) if m.any() else np.nan
            fit_v = np.median(y[m, fr, 1]) if m.any() else np.nan
            rec_v = np.median(pred[m, ft, 1]) if m.any() else np.nan
            print(f"  pair {cls:5s} {side:5s}: test frame t={ft / 200 + to:.3f} s, "
                  f"nearest fit t={fr / 200 + to:.3f} s | vtor median: "
                  f"reconstruction {rec_v:.3g}, fit {fit_v:.3g}, |diff| {dv:.3g} km/s")
    print(f"shot {args.shot}: {len(segs)} verified beam-off phases, "
          f"{len(pairs)} edge pairs (guard {g}, max gap {mg} frames)")
    row = {"shot": args.shot, "n_pairs": len(pairs)}
    if not pairs:
        print("  no usable edge pairs")
    dists = sorted({abs(ft - fr) for ft, fr, _, _ in pairs})
    for t, (tn, u) in enumerate(names):
        if not pairs:
            break
        e_off, r_ref, e_on = [], [], []
        for ft, fr, _, _ in pairs:
            m = np.isfinite(y[:, fr, t]) & np.isfinite(pred[:, ft, t])
            e_off.append(np.abs(pred[m, ft, t] - y[m, fr, t]))
        # same procedure on beam-on test frames, same distances
        for dd in dists:
            # test frame: ANY beam-on frame (the prediction exists at every
            # frame; fits sit on every other frame, so requiring a fit at
            # the test frame loses all odd distances)
            for f in np.where(on)[0]:
                for fr in (f - dd, f + dd):
                    if 0 <= fr < T and on[fr] and fitted[fr] and dd > 0:
                        m = (np.isfinite(y[:, fr, t])
                             & np.isfinite(pred[:, f, t]))
                        e_on.append(np.abs(pred[m, f, t] - y[m, fr, t]))
            # change of the fits themselves: fitted pairs at the nearest
            # realizable distance (fit cadence may forbid odd distances)
            for dd in dists:
                for d2 in (dd, dd + 1):
                    got = False
                    for f in np.where(on & fitted)[0]:
                        fr = f + d2
                        if fr < T and on[fr] and fitted[fr]:
                            m2 = (np.isfinite(y[:, fr, t])
                                  & np.isfinite(y[:, f, t]))
                            r_ref.append(np.abs(y[m2, fr, t] - y[m2, f, t]))
                            got = True
                    if got:
                        break
        e_off = np.concatenate(e_off) if e_off else np.array([])
        e_on = np.concatenate(e_on) if e_on else np.array([])
        r_ref = np.concatenate(r_ref) if r_ref else np.array([])
        if not (e_off.size and e_on.size):
            print(f"  {tn}: insufficient data")
            continue
        mo, mn = np.median(e_off), np.median(e_on)
        mr = np.median(r_ref) if r_ref.size else np.nan
        print(f"  {tn} ({u}): beam-off test frames vs adjacent fit "
              f"{mo:.3g} | same test with beam-on frames {mn:.3g} "
              f"(ratio off/on {mo / mn:.2f}) | change of the fits "
              f"themselves over the same distance {mr:.3g}")
        row.update({f"{tn}_off": mo, f"{tn}_on": mn, f"{tn}_ratio": mo / mn,
                    f"{tn}_fitchange": mr})
    by_cls = {}
    for _, _, cls, _ in pairs:
        by_cls[cls] = by_cls.get(cls, 0) + 1
    if by_cls:
        print("  pairs by phase: " + ", ".join(
            f"{k} {v}" for k, v in sorted(by_cls.items())))
    if args.csv is not None:
        fields = ["shot", "n_pairs"] + [f"{tn}_{k}" for tn, _ in names
                                        for k in ("off", "on", "ratio",
                                                  "fitchange")]
        new = not args.csv.exists()
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval="")
            if new:
                w.writeheader()
            w.writerow({k: (f"{float(v):.4g}" if isinstance(
                v, (float, np.floating)) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
