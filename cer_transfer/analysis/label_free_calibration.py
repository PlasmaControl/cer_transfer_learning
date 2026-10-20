"""Label-free calibration of a frozen DIII-D model on NSTX (no NSTX labels).

The DIII-D model is applied to NSTX spectra under several input calibrations
("passes": wavelength shift in bins of the 256-bin grid, dispersion factor,
Gaussian broadening in nm). A linear readout of the pass outputs (plus one offset
per chord) is fitted to physics pseudo-labels obtained from unlabelled NSTX
training spectra and the NSTX hardware calibration only:

* T_i: Gaussian line fit with the lamp-calibrated instrument width per chord
  (``instrument_function_nstx_chers.csv``), T_i free;
* v_tor: line fit with an additional cold passive component at rest; the active
  line centre gives the line-of-sight velocity, divided by the sightline
  geometry factor g = R_tan / R_int (``chord_geometric_factor_nstx.csv``).

Training frames are selected by line brightness only (peak S/N > 30, half
maximum inside the window); training labels are never read. Validation labels
are used only for scoring. With ``--zero-shot-csv`` the result is written in the
format of ``physics_correction --csv`` (column ``r2_protocol``), which draws the
zero-shot line of the scaling figure; with ``--save-readout`` the fitted readout
is stored, and ``predict_file_label_free`` applies model + readout to a discharge
(used by eval_checkpoint.py and the beam-off figure via ``--readout``).

    pixi run --frozen python -u -m cer_transfer.analysis.label_free_calibration \
        --checkpoint cer_ckpts/d3d_fixed_dg_rt.pt --zero-shot-csv gallery/paper/zero_shot_val.csv \
        --save-readout cer_ckpts/d3d_fixed_dg_rt_labelfree_nstx.npz
"""

from __future__ import annotations

import argparse
import csv
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from types import SimpleNamespace

from joblib import load
from scipy.optimize import least_squares

from cer_transfer.configs import data_path
from cer_transfer.data import Preprocessor
from cer_transfer.inference import load_model
from cer_transfer.training import denormalize_mu_sigma

K_EV_PER_NM2 = 12.0 * 931.494102e6 / 529.05**2  # C VI: T_i = K * sigma_doppler^2
LAMBDA0_NM, C_KMS = 529.05, 299792.458
# (shift [bins], dispersion factor, broadening [nm]) found by the input-calibration search
PASSES = [
    (0, 1.0, 0.0),
    (15, 1.05, 0.0),
    (28, 0.94, 0.04),
    (13, 1.7, 0.0),
    (6, 1.3, 0.0),
    (21, 0.99, 0.0),
]
ZERO_SHOT_PASS = {"ti": 2, "vtor": 3}  # best single pass per target without readout


def read_list(path, n):
    """``n`` evenly spaced entries of a split list (all if ``n`` <= 0)."""
    e = [ln.strip() for ln in open(path) if ln.strip() and not ln.startswith("#")]
    return e if n <= 0 else [e[i] for i in np.linspace(0, len(e) - 1, n).astype(int)]


def broaden(x, sigma_nm, disp):
    """Gaussian broadening of raw spectra (C, T, W) along W, edges replicated."""
    if sigma_nm <= 0:
        return x
    s = sigma_nm / disp
    h = int(np.ceil(4 * s))
    k = np.exp(-0.5 * (np.arange(-h, h + 1) / s) ** 2)
    k /= k.sum()
    p = np.pad(x, ((0, 0), (0, 0), (h, h)), mode="edge")
    return sum(kk * p[..., i : i + x.shape[-1]] for i, kk in enumerate(k)).astype(
        np.float32
    )


def make_preprocessors(m):
    """One causal Preprocessor (+ broadening) per pass for a loaded model."""
    return [
        (Preprocessor(m.machine, m.config.resample_w, w_scale=sc, w_shift=s, causal=True), b)
        for s, sc, b in PASSES
    ]


@torch.no_grad()
def run_passes(m, pres, x, disp):
    """Model outputs of every pass for raw spectra (C, T, W).

    Returns mu, sigma of shape (C, T, n_targets, n_passes).
    """
    mus, sgs = [], []
    for pre, b in pres:
        mu_n, sg_n = m.model(pre(broaden(x, b, disp)).unsqueeze(0).to(m.device), None)
        mu, sg = denormalize_mu_sigma(mu_n, sg_n, m.stats, m.machine.targets)
        mus.append(mu[0].cpu().numpy())
        sgs.append(sg[0].cpu().numpy())
    return np.stack(mus, -1), np.stack(sgs, -1)


def fit_readout(P, C, y, n_chords, ridge=10.0):
    """Ridge readout: all pass outputs + one offset per chord -> y. Returns its parameters."""
    F = np.c_[P.reshape(len(P), -1), np.eye(n_chords)[C]]
    mu, sd = F.mean(0), F.std(0) + 1e-6
    Fn = (F - mu) / sd
    w = np.linalg.solve(Fn.T @ Fn + ridge * np.eye(Fn.shape[1]), Fn.T @ (y - y.mean()))
    return {"mu": mu, "sd": sd, "w": w, "ymean": float(y.mean())}


def apply_readout(r, P, C, n_chords):
    """Readout r applied to pass outputs P (N, n_targets, n_passes) of chords C (N,)."""
    F = np.c_[P.reshape(len(P), -1), np.eye(n_chords)[C]]
    return ((F - r["mu"]) / r["sd"]) @ r["w"] + r["ymean"]


def save_readout(path, readouts, checkpoint):
    """Store the per-target readouts (keys 'ti', 'vtor') with their passes."""
    np.savez(
        path,
        passes=np.array(PASSES, dtype=float),
        checkpoint=str(checkpoint),
        **{f"{k}_{q}": v for k, r in readouts.items() for q, v in r.items()},
    )


def load_readout(path):
    """Readouts saved by ``save_readout`` -> {'ti': params, 'vtor': params}."""
    z = np.load(path, allow_pickle=False)
    if [tuple(p) for p in z["passes"]] != [tuple(map(float, p)) for p in PASSES]:
        raise ValueError(f"{path}: passes differ from this module's PASSES")
    return {k: {q: z[f"{k}_{q}"] for q in ("mu", "sd", "w")} | {"ymean": float(z[f"{k}_ymean"])}
            for k in ("ti", "vtor")}


def predict_file_label_free(m, path, readouts):
    """``inference.predict_file`` with the label-free readout applied.

    pred: readout of the pass outputs; pred_sigma: the network's sigma of the best
    single pass per target (the readout has no uncertainty of its own); y, sigma:
    conventional fits (NaN where none).
    """
    d = load(data_path(path))
    mc = m.machine
    x = np.asarray(d["input"], np.float32)
    if mc.input_channel_indices is not None:
        x = x[list(mc.input_channel_indices)]
    disp = float(np.abs(np.median(np.diff(np.asarray(d["wavelength_nm"], np.float64)))))
    mus, sgs = run_passes(m, make_preprocessors(m), x, disp)
    C, T = mus.shape[:2]
    chord = np.broadcast_to(np.arange(C)[:, None], (C, T)).reshape(-1)
    flat = mus.reshape(C * T, *mus.shape[2:])
    pred = np.stack([apply_readout(readouts[k], flat, chord, mc.n_chords) for k in ("ti", "vtor")], -1)
    psig = np.stack([sgs[:, :, j, ZERO_SHOT_PASS[k]] for j, k in enumerate(("ti", "vtor"))], -1)
    scale = np.asarray(mc.target_scale, np.float32)
    y = np.asarray(d["target"], np.float32) * scale
    sig = np.asarray(d["target_error"], np.float32) * scale
    if mc.chord_indices is not None:
        y, sig = y[list(mc.chord_indices)], sig[list(mc.chord_indices)]
    return SimpleNamespace(
        pred=pred.reshape(C, T, 2), pred_sigma=psig, y=y, sigma=sig, targets=list(mc.targets)
    )


def bright_frames(x, snr_min=30.0):
    """Label-free frame selection on raw spectra (C, T, W) -> (C, T) bool."""
    bg = np.percentile(x, 10, -1, keepdims=True)
    s = np.clip(x - bg, 0, None)
    pk = s.max(-1)
    above = s >= pk[..., None] / 2
    W = x.shape[-1]
    first = np.argmax(above, -1)
    last = W - 1 - np.argmax(above[..., ::-1], -1)
    noise = np.median(np.abs(np.diff(s, axis=-1)), -1) + 1e-9
    return (pk / noise > snr_min) & (first > 0) & (last < W - 1)


def fit_block(job):
    """Pseudo-labels (T_i, v_tor) for one discharge's frames.

    job = (spectra (N, W), wavelength axis (W,), instrument sigma (N,), geometry g (N,)).
    T_i from an active-line fit; v_tor from an active + cold passive fit.
    """
    X, lam, sinst, g = job
    out = np.full((len(X), 2), np.nan)
    l0 = lam.mean()
    for i in range(len(X)):
        y = X[i]
        bg = np.percentile(y, 10)
        ip = int(np.argmax(y))
        si = sinst[i]
        w = 1 / np.sqrt(np.maximum(y - y.min() + 10, 10))

        def active(p):
            b0, b1, a, mu, t = p
            return b0 + b1 * (lam - l0) + a * np.exp(
                -0.5 * (lam - mu) ** 2 / (si**2 + max(t, 0) / K_EV_PER_NM2)
            )

        def with_passive(p):
            return active(p[:5]) + p[5] * np.exp(
                -0.5 * (lam - p[6]) ** 2 / (si**2 + max(p[7], 0) / K_EV_PER_NM2)
            )

        try:
            r1 = least_squares(
                lambda p: (active(p) - y) * w,
                [bg, 0, y.max() - bg, lam[ip], 300.0],
                bounds=([-np.inf, -np.inf, 0, lam[0], 0], [np.inf, np.inf, np.inf, lam[-1], 2e4]),
                max_nfev=200,
            )
            r2 = least_squares(
                lambda p: (with_passive(p) - y) * w,
                [bg, 0, y.max() - bg, lam[ip], 300.0, 0.1 * (y.max() - bg), LAMBDA0_NM, 30.0],
                bounds=(
                    [-np.inf, -np.inf, 0, lam[0], 0, 0, LAMBDA0_NM - 0.03, 0],
                    [np.inf, np.inf, np.inf, lam[-1], 2e4, np.inf, LAMBDA0_NM + 0.03, 150.0],
                ),
                max_nfev=300,
            )
            out[i] = (
                r1.x[4] if r1.success else np.nan,
                C_KMS * (r2.x[3] - LAMBDA0_NM) / LAMBDA0_NM / g[i] if r2.success else np.nan,
            )
        except Exception:  # a failed fit leaves NaN
            pass
    return out


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", default="cer_ckpts/d3d_fixed_dg_rt.pt")
    ap.add_argument("--train-list", default="splits/nstx_train.txt")
    ap.add_argument("--val-list", default="splits/nstx_val.txt")
    ap.add_argument("--ntrain", type=int, default=60, help="unlabelled training discharges")
    ap.add_argument("--nval", type=int, default=0, help="validation discharges (0: all)")
    ap.add_argument("--max-frames", type=int, default=120000, help="frames line-fitted")
    ap.add_argument("--instrument-csv", default="instrument_function_nstx_chers.csv")
    ap.add_argument("--geometry-csv", default="chord_geometric_factor_nstx.csv")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--zero-shot-csv", type=Path, default=None)
    ap.add_argument("--save-readout", type=Path, default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    torch.set_num_threads(a.threads)
    m = load_model(a.checkpoint, device="cpu", machine="nstx")
    mc = m.machine
    if not m.config.causal:
        raise SystemExit(f"{a.checkpoint} is not a causal model")
    pres = make_preprocessors(m)
    cal = list(csv.DictReader(open(a.instrument_csv)))
    epochs = sorted(
        {(int(r["valid_from"].split()[1]), int(r["valid_to"].split()[1]), r["epoch"]) for r in cal}
    )
    sig_inst = {(r["epoch"], int(r["chord_id"][1:])): float(r["sigma_inst_nm"]) for r in cal}
    geom = {
        (r["epoch"], int(r["chord"])): float(r["g"])
        for r in csv.DictReader(open(a.geometry_csv))
        if r["is_analysis_radius"] == "1"
    }

    def epoch_of(shot):
        return next(e for lo, hi, e in epochs if lo <= shot <= hi)

    def read(entry):
        d = load(data_path(entry))
        T = int(d["end_index"])
        T = d["input"].shape[1] if T <= 0 else T
        lam = np.asarray(d["wavelength_nm"], np.float64)
        return d, np.asarray(d["input"][:, :T], np.float32), lam, float(np.abs(np.median(np.diff(lam))))

    # training side: no labels
    tp, tc, jobs = [], [], []
    for e in read_list(a.train_list, a.ntrain):
        _, x, lam, disp = read(e)
        ep = epoch_of(int(re.search(r"(\d{6})", e).group(1)))
        c, f = np.nonzero(bright_frames(x))
        if len(c) == 0:
            continue
        tp.append(run_passes(m, pres, x, disp)[0][c, f])
        tc.append(c)
        jobs.append(
            (
                x[c, f].astype(np.float64),
                lam,
                np.array([sig_inst[(ep, ci)] for ci in c]),
                np.array([geom[(ep, ci)] for ci in c]),
            )
        )
    tp, tc = np.concatenate(tp), np.concatenate(tc)
    keep = np.zeros(len(tc), bool)
    keep[rng.choice(len(tc), min(a.max_frames, len(tc)), replace=False)] = True
    off, sub = 0, []
    for X, lam, si, g in jobs:
        k = keep[off : off + len(X)]
        off += len(X)
        sub.append((X[k], lam, si[k], g[k]))
    tp, tc = tp[keep], tc[keep]
    with Pool(a.threads) as pool:
        pseudo = np.concatenate(pool.map(fit_block, sub, chunksize=1))
    good = {
        "ti": np.isfinite(pseudo[:, 0]) & (pseudo[:, 0] > 0) & (pseudo[:, 0] < 5000) & (tc >= 15),
        "vtor": np.isfinite(pseudo[:, 1]) & (np.abs(pseudo[:, 1]) < 400),
    }
    print(
        f"{len(keep)} bright training frames, {keep.sum()} fitted; pseudo-labels: "
        f"T_i {good['ti'].sum()}, v_tor {good['vtor'].sum()}",
        flush=True,
    )

    readouts = {
        k: fit_readout(tp[good[k]], tc[good[k]], pseudo[good[k], j], mc.n_chords)
        for j, k in enumerate(("ti", "vtor"))
    }
    if a.save_readout is not None:
        save_readout(a.save_readout, readouts, a.checkpoint)
        print(f"wrote {a.save_readout}")

    # validation: labels only for scoring
    vp, vy, vc = [], [], []
    for e in read_list(a.val_list, a.nval):
        d, x, _, disp = read(e)
        y = np.asarray(d["target"][:, : x.shape[1]], np.float32) * np.asarray(mc.target_scale, np.float32)
        ok = np.isfinite(y).all(-1)
        if not ok.any():
            continue
        c, f = np.nonzero(ok)
        vp.append(run_passes(m, pres, x, disp)[0][c, f])
        vy.append(y[c, f])
        vc.append(c)
    vp, vy, vc = np.concatenate(vp), np.concatenate(vy), np.concatenate(vc)

    def r2(p, y):
        return float(1 - np.mean((p - y) ** 2) / np.var(y))

    rows = []
    for j, k in enumerate(("ti", "vtor")):
        zs = r2(vp[:, j, ZERO_SHOT_PASS[k]], vy[:, j])
        lf = r2(apply_readout(readouts[k], vp, vc, mc.n_chords), vy[:, j])
        rows.append((k, zs, lf, len(vy)))
        print(f"{k}: zero-shot (best pass) R2 {zs:.4f} | label-free calibration R2 {lf:.4f} | n = {len(vy)}")
    print(f"mean: zero-shot {np.mean([r[1] for r in rows]):.4f} | label-free {np.mean([r[2] for r in rows]):.4f}")
    if a.zero_shot_csv is not None:
        a.zero_shot_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(a.zero_shot_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["target", "r2_raw", "r2_protocol", "r2_affine", "slope_ratio", "n"])
            for k, zs, lf, n in rows:
                w.writerow([k, zs, lf, "", "", n])
        print(f"wrote {a.zero_shot_csv}")


if __name__ == "__main__":
    main()
