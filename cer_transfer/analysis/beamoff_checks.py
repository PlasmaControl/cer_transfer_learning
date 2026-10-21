"""Beam-off checks for causal models: context ablation and rejoin error.

Beam-off phases (gaps) are found as for the beam-off figure
(``figures.beamoff_examples.off_phases``). For every gap (a, b):

* context ablation: prediction with the full causal context (all frames up to
  t) vs. with an input that starts only ``--margin`` frames before the gap
  ("only the gap"). Reported in units of the full-context predicted sigma:
  median |delta|/sigma and the fraction of gap frames with |delta| < sigma.
  Agreement means the estimate comes from the passive spectrum in the gap, not
  from the beam-on frames before it.
* rejoin error: per chord, the first label after the gap (frame L); error of the
  first ``--n-after`` predictions after the beam returns against that label,
  compared with the same frame offset in beam-on data (labelled frames L' with
  at least ``--clean`` beam-on frames before them, no gap nearby).

Models: ``name=checkpoint`` or ``name=checkpoint:readout.npz`` (label-free
readout of ``label_free_calibration``).

    pixi run --frozen python -u -m cer_transfer.analysis.beamoff_checks \
        --model "DIII-D=cer_ckpts/d3d_fixed_dg_rt.pt:cer_ckpts/d3d_fixed_dg_rt_labelfree_nstx.npz" \
        --model "DIII-D+NSTX=cer_ckpts/nstx_rt_init_rfull.pt" \
        --list splits/nstx_val.txt splits/nstx_test.txt --csv gallery/paper/beamoff_checks.csv \
        --dump gallery/paper/beamoff_checks.npz

``--dump`` writes the per-frame / per-(gap, chord) values behind the CSV (read
by ``figures.beamoff_examples --checks``). Keys: ``models`` (names), ``n_gaps``,
``splits`` / ``split_n_gaps`` (gaps per split), ``gap_split`` / ``gap_entry`` /
``gap_a`` / ``gap_b`` (per gap), ``fs``, ``margin``, ``n_after``; per model i
(order of ``models``):

* ``m{i}_ctx_gap``, ``m{i}_ctx_chord``, ``m{i}_ctx_pos`` (frames since the gap
  start), ``m{i}_ctx_ms`` (ms since the gap start) and, per target k,
  ``m{i}_{k}_ctx_d`` (|full context - gap only|) and ``m{i}_{k}_ctx_dz`` (the
  same in units of the full-context sigma), one entry per (gap, chord, frame);
* ``m{i}_{k}_rej`` with ``m{i}_{k}_rej_gap`` / ``m{i}_{k}_rej_chord``: rejoin
  error per (gap, chord); ``m{i}_{k}_ctl`` with ``m{i}_{k}_ctl_gap`` /
  ``m{i}_{k}_ctl_chord``: beam-on control errors (several per (gap, chord)).

Units: T_i eV, v_tor km/s.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from joblib import load

from cer_transfer.analysis.label_free_calibration import (
    ZERO_SHOT_PASS,
    apply_readout,
    load_readout,
    make_preprocessors,
    run_passes,
)
from cer_transfer.configs import data_path
from cer_transfer.data import Preprocessor
from cer_transfer.figures.beamoff_examples import off_phases
from cer_transfer.inference import load_model
from cer_transfer.training import denormalize_mu_sigma

TARGETS = ("ti", "vtor")


@torch.no_grad()
def predict(m, x, disp, readouts=None):
    """mu, sigma (C, T, 2) for raw spectra x (C, T, W); with readouts: model + readout."""
    if readouts is None:
        pre = Preprocessor(m.machine, m.config.resample_w, causal=m.config.causal)
        mu, sg = m.model(pre(x).unsqueeze(0).to(m.device), None)
        mu, sg = denormalize_mu_sigma(mu, sg, m.stats, m.machine.targets)
        return mu[0].cpu().numpy(), sg[0].cpu().numpy()
    mus, sgs = run_passes(m, make_preprocessors(m), x, disp)
    C, T = mus.shape[:2]
    chord = np.broadcast_to(np.arange(C)[:, None], (C, T)).reshape(-1)
    flat = mus.reshape(C * T, *mus.shape[2:])
    mu = np.stack([apply_readout(readouts[k], flat, chord, C) for k in TARGETS], -1).reshape(C, T, 2)
    sg = np.stack([sgs[:, :, j, ZERO_SHOT_PASS[k]] for j, k in enumerate(TARGETS)], -1)
    return mu, sg


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", action="append", required=True, help="name=checkpoint[:readout]")
    ap.add_argument("--list", type=Path, nargs="+", required=True)
    ap.add_argument("--min-off-ms", type=float, default=50.0)
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--margin", type=int, default=5, help="frames before the gap kept in 'only the gap'")
    ap.add_argument("--n-after", type=int, default=5)
    ap.add_argument("--max-label-delay", type=int, default=10, help="frames to the first label after a gap")
    ap.add_argument("--clean", type=int, default=40, help="beam-on frames required before a control label")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--dump", type=Path, default=None, help="npz with the per-frame / per-gap values")
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

    rows = []
    gap_list = [(split, e, ga, gb) for split, e, ph, _ in gaps for ga, gb in ph]
    splits = sorted({g[0] for g in gap_list})
    dump = {
        "models": np.array([s_.split("=", 1)[0] for s_ in a.model]),
        "n_gaps": np.int64(len(gap_list)),
        "splits": np.array(splits),
        "split_n_gaps": np.array([sum(g[0] == sp for g in gap_list) for sp in splits], np.int64),
        "gap_split": np.array([g[0] for g in gap_list]),
        "gap_entry": np.array([g[1] for g in gap_list]),
        "gap_a": np.array([g[2] for g in gap_list], np.int64),
        "gap_b": np.array([g[3] for g in gap_list], np.int64),
        "fs": np.float64(a.fs), "margin": np.int64(a.margin), "n_after": np.int64(a.n_after),
    }
    for mi, spec in enumerate(a.model):
        dm = {q: [] for q in ("ctx_gap", "ctx_chord", "ctx_pos")}
        dm |= {f"{k}_{q}": [] for k in TARGETS
               for q in ("ctx_d", "ctx_dz", "rej", "rej_gap", "rej_chord", "ctl", "ctl_gap", "ctl_chord")}
        gi = -1
        name, rest = spec.split("=", 1)
        ck, _, ro = rest.partition(":")
        m = load_model(ck, device=a.device, machine="nstx")
        if not m.config.causal:
            raise SystemExit(f"{ck} is not causal")
        readouts = load_readout(ro) if ro else None
        acc = {}  # split -> dict of lists
        for split, e, ph, T in gaps:
            d = load(data_path(e))
            x = np.asarray(d["input"][:, :T], np.float32)
            y = np.asarray(d["target"][:, :T], np.float32) * np.asarray(m.machine.target_scale, np.float32)
            disp = float(np.abs(np.median(np.diff(np.asarray(d["wavelength_nm"], np.float64)))))
            mu, sg = predict(m, x, disp, readouts)
            off = np.zeros(T, bool)
            for ga, gb in ph:
                off[ga:gb + 1] = True
            near = np.convolve(off.astype(float), np.ones(a.clean + 2 * a.n_after + a.max_label_delay), "same") > 0
            for ga, gb in ph:
                gi += 1
                s = acc.setdefault(split, {f"{k}_{q}": [] for k in TARGETS for q in ("z", "d", "rej", "ctl")} | {"n": 0})
                s["n"] += 1
                lo = max(ga - a.margin, 0)
                mg, _ = predict(m, x[:, lo:gb + 1], disp, readouts)
                C_, n_ = mu.shape[0], gb + 1 - ga
                dm["ctx_gap"].append(np.full(C_ * n_, gi))
                dm["ctx_chord"].append(np.repeat(np.arange(C_), n_))
                dm["ctx_pos"].append(np.tile(np.arange(n_), C_))
                for j, k in enumerate(TARGETS):
                    dlt = np.abs(mg[:, ga - lo:, j] - mu[:, ga:gb + 1, j])
                    s[f"{k}_d"].append(dlt.ravel())
                    s[f"{k}_z"].append((dlt / sg[:, ga:gb + 1, j]).ravel())
                    dm[f"{k}_ctx_d"].append(dlt.ravel())
                    dm[f"{k}_ctx_dz"].append((dlt / sg[:, ga:gb + 1, j]).ravel())
                    for c in range(y.shape[0]):
                        lab = np.where(np.isfinite(y[c, gb + 1:, j]))[0]
                        if not lab.size or lab[0] > a.max_label_delay or gb + 1 + a.n_after > T:
                            continue
                        L, o = gb + 1 + lab[0], lab[0]
                        s[f"{k}_rej"].append(np.mean(np.abs(mu[c, gb + 1:gb + 1 + a.n_after, j] - y[c, L, j])))
                        dm[f"{k}_rej"].append(s[f"{k}_rej"][-1]); dm[f"{k}_rej_gap"].append(gi)
                        dm[f"{k}_rej_chord"].append(c)
                        # control: same offset in beam-on data
                        cand = np.where(np.isfinite(y[c, :, j]) & ~near)[0]
                        cand = cand[(cand - o - a.clean >= 0) & (cand - o + a.n_after <= T)]
                        for Lp in cand[:: max(1, len(cand) // 20)]:
                            st = Lp - o
                            s[f"{k}_ctl"].append(np.mean(np.abs(mu[c, st:st + a.n_after, j] - y[c, Lp, j])))
                            dm[f"{k}_ctl"].append(s[f"{k}_ctl"][-1]); dm[f"{k}_ctl_gap"].append(gi)
                            dm[f"{k}_ctl_chord"].append(c)
        for q, v in dm.items():
            if q.endswith(("ctx_gap", "ctx_chord", "ctx_pos", "ctx_d", "ctx_dz")):
                arr = np.concatenate(v) if v else np.zeros(0)
            else:
                arr = np.asarray(v)
            if q.endswith(("_gap", "_chord", "_pos")):
                arr = arr.astype(np.int64)
            else:
                arr = arr.astype(np.float32)
            dump[f"m{mi}_{q}"] = arr
        dump[f"m{mi}_ctx_ms"] = (dump[f"m{mi}_ctx_pos"] * 1000.0 / a.fs).astype(np.float32)
        for split, s in acc.items():
            r = {"model": name, "split": split, "gaps": s["n"]}
            for k in TARGETS:
                z = np.concatenate(s[f"{k}_z"]); dd = np.concatenate(s[f"{k}_d"])
                r[f"{k}_ctx_median_dz"] = float(np.median(z))
                r[f"{k}_ctx_frac_within_sigma"] = float(np.mean(z < 1))
                r[f"{k}_ctx_median_d"] = float(np.median(dd))
                r[f"{k}_rejoin_err"] = float(np.median(s[f"{k}_rej"])) if s[f"{k}_rej"] else np.nan
                r[f"{k}_control_err"] = float(np.median(s[f"{k}_ctl"])) if s[f"{k}_ctl"] else np.nan
                r[f"{k}_n_rejoin"] = len(s[f"{k}_rej"])
            rows.append(r)
            print(f"{name:14s} {split:5s} gaps {s['n']:3d} | context: T_i |d|/sigma median {r['ti_ctx_median_dz']:.2f} "
                  f"({100 * r['ti_ctx_frac_within_sigma']:.0f}% < sigma), v_tor {r['vtor_ctx_median_dz']:.2f} "
                  f"({100 * r['vtor_ctx_frac_within_sigma']:.0f}% < sigma) | rejoin: T_i {r['ti_rejoin_err']:.0f} eV "
                  f"(beam-on {r['ti_control_err']:.0f}), v_tor {r['vtor_rejoin_err']:.1f} km/s "
                  f"(beam-on {r['vtor_control_err']:.1f})", flush=True)
    if a.csv is not None and rows:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {a.csv}")
    if a.dump is not None:
        a.dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.dump, **dump)
        print(f"wrote {a.dump}")


if __name__ == "__main__":
    main()
