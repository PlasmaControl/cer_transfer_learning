"""Nested random D3D training subsets for the NSTX -> D3D scaling study.

    pixi run --frozen python -u -m cer_transfer.datasets.make_d3d_subsets \
        --sizes 500 1000 2000 4000 8000 [--seed 42] [--out splits]

Writes splits/d3d_train_r<K>.txt (one absolute path per line), nested:
r500 is a subset of r1000 ... of r8000. Pool = the D3D config train_dir
(training_set_30), i.e. the same files the full-data runs glob. Val stays
the config val_dir for every size. Generate ONCE; never regenerate after
runs start (the lists define the study).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cer_transfer.configs import get_machine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("splits"))
    a = p.parse_args()

    m = get_machine("d3d")
    pool = sorted(Path(m.train_dir).glob("*.joblib"))
    print(f"pool: {len(pool)} files in {m.train_dir}")
    order = np.random.default_rng(a.seed).permutation(len(pool))
    a.out.mkdir(parents=True, exist_ok=True)
    for k in sorted(a.sizes):
        if k > len(pool):
            raise SystemExit(f"size {k} > pool {len(pool)}")
        f = a.out / f"d3d_train_r{k}.txt"
        if f.exists():
            raise SystemExit(f"{f} exists -- lists are write-once; "
                             "delete by hand only if no run used it")
        sel = sorted(str(pool[i].resolve()) for i in order[:k])
        f.write_text(f"# seed={a.seed} nested prefix of permutation, "
                     f"pool={len(pool)}\n" + "\n".join(sel) + "\n")
        print(f"  {f}  ({k} shots)")


if __name__ == "__main__":
    main()
