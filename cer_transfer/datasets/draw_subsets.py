r"""Draw random subsets of a training list for the error bars of the scaling study.

The validation and test lists are untouched: subsets are drawn from the
existing training list only, so every run with a drawn subset is scored on the
same validation discharges as the ladder. The draw is uniform without
replacement and deterministic per seed. Subset 0 of each size is the ladder's
own subset (``<train>_r<K>.txt``) and is not rewritten.

    pixi run --frozen python -u -m cer_transfer.datasets.draw_subsets \
        --train-list splits/nstx_train.txt --sizes 325 650 --seeds 1 2
    -> splits/nstx_train_r325_s1.txt, ..._s2.txt, nstx_train_r650_s1.txt, ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train-list", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()
    entries = [
        ln.strip()
        for ln in args.train_list.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    stem = args.train_list.with_suffix("")
    for seed in args.seeds:
        rng = np.random.default_rng(1000 + seed)
        order = rng.permutation(len(entries))
        for k in args.sizes:
            if k > len(entries):
                raise SystemExit(
                    f"size {k} exceeds the {len(entries)} training entries"
                )
            out = Path(f"{stem}_r{k}_s{seed}.txt")
            out.write_text("\n".join(entries[i] for i in sorted(order[:k])) + "\n")
            print(f"wrote {out} ({k} of {len(entries)}, seed {seed})")


if __name__ == "__main__":
    main()
