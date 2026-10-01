"""Acceptance check for --zero-dark-frames retrains.

Pre-registered criteria (per checkpoint, vs its official counterpart):
  1. labeled-point val score matches within noise (compare the eval
     console/CSV of the zdf run against the official run)
  2. per-chord medae unchanged within noise (same CSVs)
  3. NEW BEHAVIOR: on dark frames (criterion cer_transfer.dark, the one
     trained on) the model outputs ~0 with small predicted sigma.

This tool checks (3) offline from a --dump-preds npz + the shot list it
was produced from:

    pixi run python -u -m cer_transfer.analysis.check_zdf --list /tmp/darkcheck.txt \
        --preds zdf_eval/nstx_ft_zdf.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from joblib import load

from cer_transfer.configs import data_path
from cer_transfer.beamstate import valid_end

from cer_transfer.dark import dark_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", type=Path, required=True)
    p.add_argument("--preds", type=Path, required=True)
    args = p.parse_args()

    d = np.load(args.preds)
    ch = d["chord"]
    C = int(ch.max()) + 1
    Ttot = len(ch) // C
    pr = d["pred"].reshape(C, Ttot, -1)
    ps = d["pred_sigma"].reshape(C, Ttot, -1)

    files = [ln.strip() for ln in args.list.read_text().splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    off = 0
    dark_pred, dark_sig, plasma_sig = [], [], []
    for fp in files:
        s = load(data_path(fp), mmap_mode="r")
        end = valid_end(s)
        spec = np.asarray(s["input"][:, :end, :], dtype=np.float32)
        tgt = np.asarray(s["target"][:, :end, :], dtype=np.float32)
        dm = dark_mask(spec, tgt)
        lab = np.isfinite(tgt[..., 0]).any(axis=0)
        seg_p = pr[:, off:off + end]
        seg_s = ps[:, off:off + end]
        if dm.any():
            dark_pred.append(seg_p[:, dm].reshape(-1, seg_p.shape[-1]))
            dark_sig.append(seg_s[:, dm].reshape(-1, seg_s.shape[-1]))
        if lab.any():
            plasma_sig.append(seg_s[:, lab].reshape(-1, seg_s.shape[-1]))
        off += end
    assert off == Ttot, f"frame count mismatch: list {off} vs npz {Ttot}"

    if not dark_pred:
        raise SystemExit("no dark frames in this list -- use a list with "
                         "pre-discharge recording (any standard val list)")
    dp = np.concatenate(dark_pred)
    ds = np.concatenate(dark_sig)
    pssig = np.concatenate(plasma_sig)
    names = ("Ti (eV)", "vtor (km/s)")
    print(f"dark frames pooled over {len(files)} shots: {len(dp)} "
          f"chord-points")
    for t, name in enumerate(names):
        print(f"{name}: median |pred| on dark = "
              f"{np.median(np.abs(dp[:, t])):.3g} | p95 "
              f"{np.percentile(np.abs(dp[:, t]), 95):.3g} | "
              f"median pred_sigma dark = {np.median(ds[:, t]):.3g} "
              f"(labeled frames: {np.median(pssig[:, t]):.3g})")
    print("ACCEPT (3) iff median |pred| << dark_sigma (100 eV / 10 km/s) "
          "and dark pred_sigma is small; criteria (1)/(2) from the eval "
          "CSVs vs the officials.")


if __name__ == "__main__":
    main()
