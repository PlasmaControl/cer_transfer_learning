"""Create stratified, reproducible shot-level train/val/test splits plus
nested scaling-curve subsets, written as explicit file lists.

    pixi run python -u -m cer_transfer.datasets.make_splits --machine nstxu \
        --dir /path/to/nstxu_labeled --out splits \
        --val-frac 0.15 --test-frac 0.15 --subsets 25 50 100 200 400

Outputs (plain text, one absolute file path per line):
    splits/<machine>_train.txt
    splits/<machine>_val.txt
    splits/<machine>_test.txt
    splits/<machine>_train_n<K>.txt   (nested: n25 subset of n50 subset of ...)
    splits/<machine>_split_report.txt (stratification sanity report)

Stratification: shots are binned by per-shot median |vtor| (canonical km/s,
NaN-labeled frames ignored); each bin is split with the same fractions, so
train/val/test all span the rotation range (caveats 1-2). Shots that fail
basic usability checks (end_index <= 0, all-NaN targets) are EXCLUDED and
listed in the report.

Deterministic under --seed. Test set discipline: generate ONCE, commit the
lists, never regenerate after experiments begin.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from joblib import load

from cer_transfer.configs import get_machine


def shot_summary(fp: Path, machine):
    d = load(fp, mmap_mode="r")
    end = int(d["end_index"])
    if end <= 0:
        return None, "end_index<=0"
    # any NaN in the input spectra makes the shot untrainable (the model
    # consumes spectra densely; NaN targets are fine — masked loss)
    inp = np.asarray(d["input"][:, :end, :], dtype=np.float32)
    if np.isnan(inp).any():
        n_bad = int(np.isnan(inp).sum())
        return None, f"NaN in spectra ({n_bad} values)"
    tgt = np.asarray(d["target"][:, :end, :], dtype=np.float64)
    tgt = tgt * np.asarray(machine.target_scale, dtype=np.float64)
    v = tgt[..., 1]
    if np.isnan(tgt).all():
        return None, "all targets NaN"
    vmed = float(np.nanmedian(np.abs(v))) if not np.isnan(v).all() else 0.0
    return vmed, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--machine", required=True)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("splits"))
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--n-strata", type=int, default=5)
    p.add_argument("--subsets", type=int, nargs="*", default=[],
                   help="nested scaling-curve sizes, e.g. 25 50 100 200 400")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    machine = get_machine(args.machine)
    files = sorted(args.dir.glob("*.joblib"))
    if not files:
        raise SystemExit(f"no .joblib files in {args.dir}")

    rng = np.random.default_rng(args.seed)
    usable, excluded, vmeds = [], [], []
    for fp in files:
        try:
            vmed, reason = shot_summary(fp, machine)
        except Exception as e:
            vmed, reason = None, f"load failed: {e}"
        if reason:
            excluded.append((fp, reason))
        else:
            usable.append(fp)
            vmeds.append(vmed)
    vmeds = np.asarray(vmeds)
    print(f"{len(usable)} usable shots, {len(excluded)} excluded")

    # stratify by |vtor| median quantile bins
    edges = np.quantile(vmeds, np.linspace(0, 1, args.n_strata + 1))
    edges[-1] += 1e-9
    strata = np.digitize(vmeds, edges[1:-1])

    train, val, test = [], [], []
    for s in range(args.n_strata):
        idx = np.where(strata == s)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * args.test_frac))
        n_val = int(round(n * args.val_frac))
        test += [usable[i] for i in idx[:n_test]]
        val += [usable[i] for i in idx[n_test:n_test + n_val]]
        train += [usable[i] for i in idx[n_test + n_val:]]

    # nested scaling subsets from the train pool, rotation-stratified order
    train_v = {fp: vmeds[usable.index(fp)] for fp in train}
    order = list(train)
    rng.shuffle(order)
    # greedy round-robin over strata for balanced small subsets
    by_stratum = {}
    for fp in order:
        s = int(np.digitize(train_v[fp], edges[1:-1]))
        by_stratum.setdefault(s, []).append(fp)
    interleaved = []
    while any(by_stratum.values()):
        for s in sorted(by_stratum):
            if by_stratum[s]:
                interleaved.append(by_stratum[s].pop())

    args.out.mkdir(parents=True, exist_ok=True)

    def write(name, paths):
        f = args.out / f"{args.machine}_{name}.txt"
        f.write_text("\n".join(str(p.resolve()) for p in sorted(paths)) + "\n")
        print(f"  {f}  ({len(paths)} shots)")

    write("train", train)
    write("val", val)
    write("test", test)
    for k in sorted(set(args.subsets)):
        if k > len(interleaved):
            print(f"  skip subset n{k}: only {len(interleaved)} train shots")
            continue
        write(f"train_n{k}", interleaved[:k])

    # report
    rep = [f"machine={args.machine} seed={args.seed} "
           f"total={len(files)} usable={len(usable)} excluded={len(excluded)}",
           f"split: train={len(train)} val={len(val)} test={len(test)}",
           "", "|vtor| median (km/s) per split:"]
    for name, paths in (("train", train), ("val", val), ("test", test)):
        v = np.array([train_v.get(fp, vmeds[usable.index(fp)])
                      for fp in paths])
        rep.append(f"  {name:5s} median {np.median(v):7.2f}  "
                   f"p10 {np.percentile(v, 10):7.2f}  "
                   f"p90 {np.percentile(v, 90):7.2f}")
    if excluded:
        rep += ["", "excluded:"] + [f"  {fp.name}: {r}" for fp, r in excluded]
    (args.out / f"{args.machine}_split_report.txt").write_text(
        "\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
