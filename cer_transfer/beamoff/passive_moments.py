"""Model-free reference for beam-off phases: Doppler shift and width of
the passive line, computed directly from the background-array spectra
(no ML), mapped to vtor and Ti per chord by a linear calibration against
the conventional fits while the beam is on.

Protocol per chord:
  1. line window: +-halfwin pixels around the peak of the mean spectrum
     over fitted frames; baseline = low percentile of the window
  2. moments per frame: centroid (-> vtor), variance (-> Ti)
  3. calibrate vtor ~ a + b*centroid and Ti ~ c + d*variance on the EVEN
     fitted frames; validate on the ODD fitted frames (reference error
     with the beam on, against fits)
  4. only chords whose reference is valid on held-out beam-on frames
     (correlation >= --min-r) are used
  5. verified beam-off frames (cer_transfer.beamstate; startup phase
     excluded by default): compare network vs moment reference, relative
     to the reference's own held-out beam-on error

    pixi run python -u -m cer_transfer.beamoff.passive_moments --shot 141710 \
        --bg gallery/bo_141710_bg.npz \
        --bg-file "$(cat gallery/bo_141710_bg.txt)" \
        --fg-file "$(cat gallery/bo_141710_fg.txt)" \
        [--csv gallery/moments_pool.csv]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from joblib import load
from cer_transfer.beamstate import valid_end

from cer_transfer.beamstate import beam_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=str, required=True)
    p.add_argument("--bg", type=Path, required=True,
                   help="passive-model npz (eval_checkpoint --dump-preds)")
    p.add_argument("--bg-file", type=Path, required=True)
    p.add_argument("--fg-file", type=Path, required=True)
    p.add_argument("--halfwin", type=int, default=8)
    p.add_argument("--min-r", type=float, default=0.7)
    p.add_argument("--min-fits", type=int, default=12)
    p.add_argument("--include-head", action="store_true",
                   help="also use startup-phase beam-off frames")
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()

    d = np.load(args.bg)
    ch = d["chord"]
    C = int(ch.max()) + 1
    Tn = len(ch) // C
    pred = d["pred"].reshape(C, Tn, -1)
    y = d["y"].reshape(C, Tn, -1)

    js = load(args.bg_file, mmap_mode="r")
    end = valid_end(js)
    spec = np.asarray(js["input"][:, :end, :], dtype=np.float32)
    st = beam_state(args.bg_file, args.fg_file)
    T = min(Tn, end, st["T"])
    pred, y, spec = pred[:, :T], y[:, :T], spec[:, :T]
    off = np.zeros(T, bool)
    for a, b, cls, ok, _ in st["segments"]:
        if ok and (args.include_head or cls != "head"):
            off[a:min(b, T - 1) + 1] = True
    W = spec.shape[-1]
    pix = np.arange(W, dtype=np.float64)

    names = (("ti", "eV"), ("vtor", "km/s"))
    res = {k: [] for k, _ in names}
    used = 0
    for c in range(C):
        fit = np.isfinite(y[c, :, 0]) & np.isfinite(y[c, :, 1])
        fr = np.where(fit)[0]
        if fr.size < args.min_fits:
            continue
        mean_sp = spec[c, fr].mean(axis=0)
        pk = int(np.argmax(mean_sp))
        lo, hi = max(0, pk - args.halfwin), min(W, pk + args.halfwin + 1)
        win = spec[c, :, lo:hi].astype(np.float64)
        base = np.percentile(win, 10, axis=1, keepdims=True)
        w = np.clip(win - base, 0, None)
        s = w.sum(axis=1)
        good = s > 0
        cen = np.full(T, np.nan)
        var = np.full(T, np.nan)
        cen[good] = (w[good] * pix[lo:hi]).sum(axis=1) / s[good]
        var[good] = (w[good] * (pix[lo:hi] - cen[good, None]) ** 2
                     ).sum(axis=1) / s[good]
        feat = (var, cen)  # Ti ~ variance, vtor ~ centroid
        fr = fr[np.isfinite(cen[fr])]
        cal, val = fr[0::2], fr[1::2]
        if cal.size < 6 or val.size < 6:
            continue
        ok_c = True
        rows = []
        for t, (tn, _) in enumerate(names):
            x = feat[t]
            A = np.vstack([x[cal], np.ones(cal.size)]).T
            coef = np.linalg.lstsq(A, y[c, cal, t], rcond=None)[0]
            ref = coef[0] * x + coef[1]
            r = np.corrcoef(ref[val], y[c, val, t])[0, 1]
            if not np.isfinite(r) or r < args.min_r:
                ok_c = False
                break
            ref_on = np.median(np.abs(ref[val] - y[c, val, t]))
            nn_on = np.median(np.abs(pred[c, val, t] - y[c, val, t]))
            m = off & np.isfinite(ref)
            if m.sum() < 2:
                ok_c = False
                break
            d_off = np.median(np.abs(pred[c, m, t] - ref[m]))
            rows.append((tn, r, ref_on, nn_on, d_off, int(m.sum())))
        if ok_c and rows:
            used += 1
            for tn, r, ref_on, nn_on, d_off, n in rows:
                res[tn].append((c, r, ref_on, nn_on, d_off, n))

    n_off = int(off.sum())
    print(f"shot {args.shot}: {n_off} verified beam-off frames "
          f"({'incl.' if args.include_head else 'excl.'} startup); "
          f"{used} chords with a valid moment reference "
          f"(held-out beam-on correlation >= {args.min_r})")
    row = {"shot": args.shot, "n_off": n_off, "n_chords": used}
    for tn, u in names:
        if not res[tn]:
            print(f"  {tn}: no chord with a valid reference")
            continue
        a = np.array([r_[1:5] for r_ in res[tn]], dtype=float)
        chs = [r_[0] for r_ in res[tn]]
        r_m, ref_on, nn_on, d_off = np.median(a, axis=0)
        print(f"  {tn} ({u}), chords {chs}:")
        print(f"    beam on, held-out fits: moment reference error "
              f"{ref_on:.3g} | network error {nn_on:.3g} "
              f"(median corr of reference {r_m:.2f})")
        print(f"    beam off: |network - moment reference| {d_off:.3g}  "
              f"(x{d_off / ref_on:.2f} the reference's own beam-on error)")
        row.update({f"{tn}_ref_on": ref_on, f"{tn}_nn_on": nn_on,
                    f"{tn}_d_off": d_off, f"{tn}_ratio": d_off / ref_on})
    if args.csv is not None:
        fields = ["shot", "n_off", "n_chords"] + [
            f"{tn}_{k}" for tn, _ in names
            for k in ("ref_on", "nn_on", "d_off", "ratio")]
        new = not args.csv.exists()
        with open(args.csv, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields, restval="")
            if new:
                wr.writeheader()
            wr.writerow({k: (f"{float(v):.4g}" if isinstance(
                v, (float, np.floating)) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
