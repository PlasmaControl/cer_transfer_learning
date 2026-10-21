"""Cut single discharges out of a concatenated prediction dump.

eval_checkpoint.py --dump-preds over a list concatenates the files along time
(chord-major). This writes the frames per file of the list (--lengths) and,
for each --shot, a single-discharge dump in the same format.

    pixi run python -m cer_transfer.analysis.slice_dump \
        --dump gallery/agnostic_scaling/nstx_rt_init_rfull.npz --list splits/nstx_val.txt \
        --lengths gallery/val_file_lengths.npz --shot 138321 --out gallery/val_{shot}.npz
"""

from __future__ import annotations

import argparse

import numpy as np
from joblib import load

from cer_transfer.configs import data_path


def file_lengths(list_path):
    """Files of a list and their number of frames (len of time_s)."""
    files = [ln.strip() for ln in open(list_path) if ln.strip()]
    T = np.array([len(load(data_path(f), mmap_mode="r")["time_s"]) for f in files])
    return files, T


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True, help="concatenated prediction dump")
    p.add_argument("--list", required=True, help="file list the dump was made from")
    p.add_argument("--lengths", required=True, help="output npz: files, T (frames per file)")
    p.add_argument("--shot", nargs="*", default=[], help="discharges to cut out")
    p.add_argument("--out", default="gallery/val_{shot}.npz", help="output pattern")
    args = p.parse_args()

    files, Ts = file_lengths(args.list)
    np.savez(args.lengths, files=np.array(files), T=Ts)
    z = np.load(args.dump)
    C = int(z["chord"].max()) + 1
    Ttot = len(z["chord"]) // C
    if Ts.sum() != Ttot:
        raise SystemExit(f"{args.list}: {Ts.sum()} frames, {args.dump}: {Ttot}")
    print(f"{len(files)} files, {Ttot} frames -> {args.lengths}")
    for shot in args.shot:
        fi = next((i for i, f in enumerate(files) if shot in f), None)
        if fi is None:
            raise SystemExit(f"{shot} not in {args.list}")
        s0, T = int(Ts[:fi].sum()), int(Ts[fi])
        out = {k: z[k].reshape(C, Ttot, -1)[:, s0:s0 + T].reshape(C * T, -1)
               for k in ("y", "pred", "sigma", "pred_sigma")}
        out["chord"] = np.repeat(np.arange(C), T).astype(np.int16)
        out["targets"] = z["targets"]
        path = args.out.format(shot=shot)
        np.savez_compressed(path, **out)
        print(f"{files[fi]}: frames {s0}..{s0 + T} -> {path}")


if __name__ == "__main__":
    main()
