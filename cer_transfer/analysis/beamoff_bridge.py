"""Beam-off bridge test: does the in-gap estimate know where the plasma went?

For every beam-off phase (gap; ``figures.beamoff_examples.off_phases``) and every
chord with a fit shortly before and shortly after the gap:

* ``before``: the last fit before the gap, ``after``: the first fit after it;
* ``model``: the prediction of the model at the last frame of the gap, while the
  beam is still off (no active line in that frame or before it inside the gap);
* errors against ``after``: the model, and "hold the last fit" (|before - after|).

A model that only remembers the pre-gap state does no better than holding the
last fit; a model that reads the evolution from the passive light does better,
the more so the more the plasma changed across the gap. The script also prints
the gaps ranked for the paper examples: long gaps where the plasma changed and
the model ends close to the first fit after.

For the examples, "good reconstruction" means the model matches the fits right
before and right after the gap: per row, ``fit_err_before_<i>`` / ``fit_err_after_<i>``
= median |model - fit| over the fits within ``--context-ms`` before the gap start /
after the gap end (same chord and target), with their counts ``n_fit_before`` /
``n_fit_after``. Inside the gap the output must connect the two fits smoothly:
``gap_dev_<i>`` = RMS deviation of the in-gap prediction from the straight line
between the last fit before and the first fit after, ``gap_rough_<i>`` = RMS of its
frame-to-frame change.

    pixi run --frozen python -u -m cer_transfer.analysis.beamoff_bridge \
        --model "DIII-D=cer_ckpts/d3d_fixed_dg_rt.pt:cer_ckpts/d3d_fixed_dg_rt_labelfree_nstx.npz" \
        --model "DIII-D+NSTX=cer_ckpts/nstx_rt_init_rfull.pt" \
        --list splits/nstx_val.txt splits/nstx_test.txt splits/nstx_train.txt \
        --out gallery/paper/beamoff_bridge.npz

Output npz, one row per (gap, chord, target): ``split``, ``entry``, ``gap_a``,
``gap_b``, ``gap_ms``, ``chord``, ``target`` (0 T_i eV, 1 v_tor km/s), ``before``,
``after``, ``t_start`` (s, time of the gap start), ``dt_before`` / ``dt_after`` (frames from the gap edge to the fit), and
``model_<i>`` (prediction at the last gap frame) for the models in ``models``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from joblib import load

from cer_transfer.analysis.beamoff_checks import load_readout, predict
from cer_transfer.configs import data_path
from cer_transfer.figures.beamoff_examples import off_phases
from cer_transfer.figures.timeaxis import file_times
from cer_transfer.inference import load_model

TARGETS = ("ti", "vtor")


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", action="append", required=True, help="name=checkpoint[:readout]")
    ap.add_argument("--list", type=Path, nargs="+", required=True)
    ap.add_argument("--min-off-ms", type=float, default=50.0)
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--t-offset", type=float, default=-0.235, help="time of frame 0 (s), as in the figures")
    ap.add_argument("--max-delay", type=int, default=10, help="frames from the gap edge to the fit, each side")
    ap.add_argument("--context-ms", type=float, default=50.0, help="window of fits before / after a gap")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    entries = [(lst.stem.replace("nstx_", ""), ln.strip()) for lst in a.list
               for ln in lst.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    min_frames = int(round(a.min_off_ms / 1000 * a.fs))
    gaps = []
    for split, e in entries:
        ph, T = off_phases(e, min_frames, 1)
        if ph:
            gaps.append((split, e, ph, T))
    print(f"{sum(len(g[2]) for g in gaps)} gaps in {len(gaps)} discharges", flush=True)

    # rows (gap, chord, target) with a fit on both sides, from the labels only
    rows = []
    nctx = int(round(a.context_ms / 1000 * a.fs))
    for split, e, ph, T in gaps:
        dd = load(data_path(e), mmap_mode="r")
        t_s = file_times(e, dd["input"].shape[1], a.fs, a.t_offset)  # gap start time (flat-top selection)
        y = np.asarray(dd["target"][:, :T], np.float32)
        y = y * np.array([1000.0, 1.0], np.float32)  # files store T_i in keV
        for ga, gb in ph:
            for c in range(y.shape[0]):
                for j in range(2):
                    pre = np.where(np.isfinite(y[c, max(ga - a.max_delay, 0):ga, j]))[0]
                    post = np.where(np.isfinite(y[c, gb + 1:gb + 1 + a.max_delay, j]))[0]
                    if not pre.size or not post.size:
                        continue
                    lb = max(ga - a.max_delay, 0) + pre[-1]
                    la = gb + 1 + post[0]
                    wb = (max(ga - nctx, 0), ga)  # fits before / after the gap, for the match score
                    wa = (gb + 1, min(gb + 1 + nctx, T))
                    rows.append(dict(split=split, entry=e, T=T, t_start=t_s[ga], gap_a=ga, gap_b=gb,
                                     wb0=wb[0], wb1=wb[1],
                                     wa0=wa[0], wa1=wa[1],
                                     n_fit_before=int(np.isfinite(y[c, wb[0]:wb[1], j]).sum()),
                                     n_fit_after=int(np.isfinite(y[c, wa[0]:wa[1], j]).sum()),
                                     gap_ms=(gb + 1 - ga) * 1000 / a.fs, chord=c, target=j,
                                     before=y[c, lb, j], after=y[c, la, j],
                                     dt_before=ga - lb, dt_after=la - gb))
    print(f"{len(rows)} (gap, chord, target) rows with a fit on both sides", flush=True)
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    names = []
    for mi, spec in enumerate(a.model):
        name, rest = spec.split("=", 1)
        names.append(name)
        ck, _, ro = rest.partition(":")
        m = load_model(ck, device=a.device, machine="nstx")
        if not m.config.causal:
            raise SystemExit(f"{ck} is not causal")
        readouts = load_readout(ro) if ro else None
        pred = np.full(len(rows), np.nan, np.float32)
        eb = np.full(len(rows), np.nan, np.float32)
        ea = np.full(len(rows), np.nan, np.float32)
        dev = np.full(len(rows), np.nan, np.float32)
        rough = np.full(len(rows), np.nan, np.float32)
        idx = {}
        for i, r in enumerate(rows):
            idx.setdefault(r["entry"], []).append(i)
        for n, (e, ii) in enumerate(idx.items()):
            d = load(data_path(e))
            T = max(rows[i]["wa1"] for i in ii)  # causal: frames after the last window are not needed
            x = np.asarray(d["input"][:, :T], np.float32)
            y = np.asarray(d["target"][:, :T], np.float32) * np.array([1000.0, 1.0], np.float32)
            disp = float(np.abs(np.median(np.diff(np.asarray(d["wavelength_nm"], np.float64)))))
            mu, _ = predict(m, x, disp, readouts)
            for i in ii:
                r = rows[i]
                c, j = r["chord"], r["target"]
                pred[i] = mu[c, r["gap_b"], j]
                # in-gap output vs the straight line from the fit before to the fit after
                lb, la = r["gap_a"] - r["dt_before"], r["gap_b"] + r["dt_after"]
                f = np.arange(r["gap_a"], r["gap_b"] + 1)
                line = r["before"] + (r["after"] - r["before"]) * (f - lb) / (la - lb)
                g = mu[c, r["gap_a"]:r["gap_b"] + 1, j]
                dev[i] = np.sqrt(np.mean((g - line) ** 2))
                rough[i] = np.sqrt(np.mean(np.diff(g) ** 2))
                for arr, lo, hi in ((eb, r["wb0"], r["wb1"]), (ea, r["wa0"], r["wa1"])):
                    ok = np.isfinite(y[c, lo:hi, j])
                    if ok.any():
                        arr[i] = np.median(np.abs(mu[c, lo:hi, j][ok] - y[c, lo:hi, j][ok]))
            if n % 50 == 0:
                print(f"  {name}: {n}/{len(idx)} discharges", flush=True)
        out[f"model_{mi}"] = pred
        out[f"fit_err_before_{mi}"] = eb
        out[f"fit_err_after_{mi}"] = ea
        out[f"gap_dev_{mi}"] = dev
        out[f"gap_rough_{mi}"] = rough
    out["models"] = np.array(names)
    np.savez_compressed(a.out, **out)

    # summary: v_tor and T_i, per model, against holding the last fit
    held = np.abs(out["before"] - out["after"])
    for j, k in enumerate(TARGETS):
        for split_sel, lab in ((("val", "test"), "val+test"), (("train",), "train")):
            s = (out["target"] == j) & np.isin(out["split"], split_sel)
            if not s.any():
                continue
            msg = [f"{k} {lab}: {s.sum()} rows, hold last fit median |err| {np.median(held[s]):.3g}"]
            for mi, name in enumerate(names):
                err = np.abs(out[f"model_{mi}"] - out["after"])
                msg.append(f"{name} {np.median(err[s]):.3g} (better than holding in "
                           f"{np.mean(err[s] < held[s]) * 100:.0f}%)")
            print("; ".join(msg))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
