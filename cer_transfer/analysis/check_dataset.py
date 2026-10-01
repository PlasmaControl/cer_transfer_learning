"""Validate converted joblib shot files against the pipeline's assumptions.

    pixi run python -u -m cer_transfer.analysis.check_dataset --machine nstxu --dir /path/to/files
    pixi run python -u -m cer_transfer.analysis.check_dataset --machine nstx  --dir ... --limit 500

Checks per file (cheap: mmap, no bulk reads except sampled stats):
  keys, array shapes/dtypes, chord count vs machine config, wavelength bins
  (target stats are reported in CANONICAL units: ti eV, vtor km/s —
  machine.target_scale is applied)
  vs pooling arithmetic, end_index sanity, target/target_error shape match,
  NaN fraction of targets, negative-count check on sampled spectra, and
  basic intensity/target magnitude stats for cross-machine comparison.

Exit code 0 = all files pass; 1 = any hard failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import load

from cer_transfer.configs import ModelConfig, get_machine

REQUIRED_KEYS = ("input", "target", "target_error", "end_index")


def check_file(fp: Path, machine, min_w: int, rng) -> tuple[list, dict]:
    errors, stats = [], {}
    d = load(fp, mmap_mode="r")
    for k in REQUIRED_KEYS:
        if k not in d:
            return [f"missing key '{k}'"], stats
    inp, tgt, err = d["input"], d["target"], d["target_error"]
    end = int(d["end_index"])

    C, T, W = inp.shape if inp.ndim == 3 else (None, None, None)
    if inp.ndim != 3:
        errors.append(f"input ndim={inp.ndim}, expected 3 (C, T, W)")
        return errors, stats
    if C != machine.n_raw_channels:
        errors.append(f"input C={C}, config expects {machine.n_raw_channels}")
    if W < min_w:
        errors.append(f"W={W} < minimum {min_w} "
                      f"(hidden_dim * 2^pools for this machine)")
    if tgt.shape != (C, T, machine.n_targets):
        errors.append(f"target shape {tgt.shape}, expected "
                      f"{(C, T, machine.n_targets)}")
    if err.shape != tgt.shape:
        errors.append(f"target_error shape {err.shape} != target {tgt.shape}")
    if end > 0:
        inp_v = np.asarray(inp[:, :end, :], dtype=np.float32)
        n_nan = int(np.isnan(inp_v).sum())
        if n_nan:
            errors.append(f"NaN in input spectra ({n_nan} values)")
    if not (0 < end <= T):
        errors.append(f"end_index={end} outside (0, T={T}]")
    for name, arr in (("input", inp), ("target", tgt)):
        if arr.dtype != np.float32:
            errors.append(f"{name} dtype={arr.dtype}, expected float32")

    # sampled bulk checks (a few timesteps, all chords)
    ts = rng.choice(max(end, 1), size=min(4, max(end, 1)), replace=False)
    spec = np.asarray(inp[:, ts, :], dtype=np.float64)
    if not np.isfinite(spec).all():
        errors.append("non-finite values in sampled spectra")
    if (spec < 0).any():
        errors.append("negative photon counts in sampled spectra")

    tgt_v = np.asarray(tgt[:, :end, :], dtype=np.float64) \
        * np.asarray(machine.target_scale, dtype=np.float64)  # canonical
    nan_frac = float(np.isnan(tgt_v).mean())
    stats.update(
        W=W, T=T, end_index=end, nan_frac=nan_frac,
        spec_median=float(np.median(spec)),
        spec_max=float(spec.max()),
        ti_median=float(np.nanmedian(tgt_v[..., 0]))
        if nan_frac < 1.0 else float("nan"),
        vtor_median=float(np.nanmedian(tgt_v[..., 1]))
        if nan_frac < 1.0 else float("nan"),
    )
    if nan_frac == 1.0:
        stats["unlabeled"] = True
    return errors, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--machine", required=True)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = all files")
    p.add_argument("--hidden-dim", type=int, default=8)
    p.add_argument("--trunk-w-pools", type=int, default=3)
    args = p.parse_args()

    machine = get_machine(args.machine)
    cfg = ModelConfig(hidden_dim=args.hidden_dim,
                      trunk_w_pools=args.trunk_w_pools, feature_width=128)
    n_pools = machine.stem_w_pools + cfg.trunk_w_pools
    min_w = args.hidden_dim << n_pools
    print(f"machine={machine.name}: C={machine.n_raw_channels}, "
          f"pools={n_pools} -> min W={min_w}\n")

    files = sorted(args.dir.glob("*.joblib"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("no .joblib files found"); sys.exit(1)

    rng = np.random.default_rng(0)
    n_bad, agg = 0, []
    for i, fp in enumerate(files):
        try:
            errors, stats = check_file(fp, machine, min_w, rng)
        except Exception as e:  # unreadable / corrupt
            errors, stats = [f"load failed: {type(e).__name__}: {e}"], {}
        if errors:
            n_bad += 1
            print(f"FAIL {fp.name}: " + "; ".join(errors))
        elif stats.get("unlabeled"):
            print(f"UNLABELED {fp.name}: all targets NaN "
                  "(format-valid; belongs in the SSL pool, not a labeled dir)")
        if stats:
            agg.append(stats)
        if (i + 1) % 500 == 0:
            print(f"... {i + 1}/{len(files)} checked ({n_bad} failures)")

    if agg:
        def q(key):
            v = np.array([a[key] for a in agg if key in a and
                          np.isfinite(a.get(key, np.nan))])
            return (f"median {np.median(v):.4g}, "
                    f"range [{v.min():.4g}, {v.max():.4g}]") if len(v) else "-"
        n_unlab = sum(1 for a in agg if a.get("unlabeled"))
        print(f"\nsummary over {len(agg)} readable files "
              f"({n_unlab} fully unlabeled):")
        for key in ("W", "T", "end_index", "nan_frac", "spec_median",
                    "spec_max", "ti_median", "vtor_median"):
            print(f"  {key:12s} {q(key)}")

    print(f"\n{len(files) - n_bad}/{len(files)} files OK")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
