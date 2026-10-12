"""Add the wavelength axis (nm per bin) to preprocessed discharge files.

DIII-D: the axis is rebuilt from the raw ``<shot>_cer_spectra.h5`` files the
way the preprocessing notebook built its master grid (union of the chord
ranges at the finest chord step, descending), see
:mod:`cer_transfer.datasets.wavelength_axes`. NSTX (``--constant-range``): one
axis for all files, from the window limits.

Every file gets two extra keys, ``wavelength_nm`` (float64, one value per
bin) and ``wavelength_source``; the keys the loader reads (input, target,
target_error, end_index) are copied unchanged. Safety:

* files that already have ``wavelength_nm`` are skipped (re-runs are safe)
* a file whose rebuilt axis does not match its number of bins is left
  untouched and reported
* the new file is written next to the old one under a temporary name,
  re-opened and checked (same keys, shapes and contents), and only then
  renamed over the original (atomic on the same filesystem)

    pixi run python -m cer_transfer.datasets.add_wavelength_axis \\
        --dirs training_set_30 test_set_30 --raw-dir /scratch/gpfs/NSTX-U/passive_micer/raw \\
        [--shard 0 --n-shards 20] [--dry-run]
    pixi run python -m cer_transfer.datasets.add_wavelength_axis \\
        --dirs chers_nstx_labeled --constant-range 528.20 530.28
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
from joblib import dump, load

from cer_transfer.configs import data_path
from cer_transfer.datasets.wavelength_axes import (
    chord_axes,
    estimate_axis,
    notebook_chord_filter,
)

LOADER_KEYS = ("input", "target", "target_error", "end_index")


def union_range(raw: Path, chords: str):
    """(max, min) wavelength over the chords of a discharge (nm)."""
    flt = notebook_chord_filter if chords == "notebook" else (lambda n: n[0] in "tv")
    axes = chord_axes(raw, flt)
    if not axes:
        return None
    return max(max(a[0], a[-1]) for a in axes), min(min(a[0], a[-1]) for a in axes)


def rebuilt_axis(path: Path, raw_dir: Path, constant, chords: str = "notebook"):
    """Wavelength axis for one file, or (None, reason).

    As in the preprocessing (paper, Sec. data): a uniformly spaced master
    axis from the maximum to the minimum wavelength over the chords of the
    discharge, descending, with the file's number of bins.
    """
    n = load(path, mmap_mode="r")["input"].shape[-1]
    if constant is not None:
        lo, hi = constant
        return np.linspace(lo, hi, n), f"constant {lo}-{hi} nm (ascending)"
    shot = re.findall(r"\d{6}", path.stem)
    raw = raw_dir / f"{shot[-1]}_cer_spectra.h5" if shot else None
    if raw is None or not raw.exists():
        return None, "no raw file"
    r = union_range(raw, chords)
    if r is None:
        return None, "no chord axes in raw file"
    hi, lo = r
    return (
        np.linspace(hi, lo, n),
        f"raw {raw.name}: master axis {hi:.4f} -> {lo:.4f} nm over {n} bins "
        f"(min/max over {chords} chords, descending)",
    )


def add_axis(path: Path, axis, source: str):
    """Write path with the axis added; atomic and verified."""
    d = dict(load(path))
    d["wavelength_nm"] = np.asarray(axis, dtype=np.float64)
    d["wavelength_source"] = source
    tmp = path.with_name(path.name + ".tmp_axis")
    dump(d, tmp)
    chk = load(tmp, mmap_mode="r")
    for k in LOADER_KEYS:
        if k not in d:
            continue
        a, b = np.asarray(d[k]), np.asarray(chk[k])
        same = a.shape == b.shape and (
            np.array_equal(a, b, equal_nan=True)
            if a.dtype.kind == "f"
            else np.array_equal(a, b)
        )
        if not same:
            os.remove(tmp)
            raise RuntimeError(f"verification failed for {path} (key {k})")
    if len(chk["wavelength_nm"]) != np.asarray(d["input"]).shape[-1]:
        os.remove(tmp)
        raise RuntimeError(f"axis length check failed for {path}")
    os.replace(tmp, path)


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dirs", nargs="+", required=True, help="folders under CER_DATA_ROOT"
    )
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument(
        "--constant-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("FIRST_NM", "LAST_NM"),
        help="one axis for all files: wavelength of the first and last bin",
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="check only, write nothing")
    ap.add_argument(
        "--min-bins",
        type=int,
        default=0,
        help="only files with at least this many bins",
    )
    ap.add_argument(
        "--max-bins",
        type=int,
        default=10**9,
        help="only files with at most this many bins",
    )
    ap.add_argument(
        "--chords",
        choices=("notebook", "all"),
        default="notebook",
        help="chords whose windows define min/max: t01-t48 + v01-v32 (notebook) or all t/v",
    )
    ap.add_argument(
        "--check",
        type=int,
        default=0,
        help="compare the master-axis rule with an independent fit to the raw "
        "spectra on this many files, write nothing, and exit",
    )
    args = ap.parse_args()
    if args.constant_range is None and args.raw_dir is None:
        raise SystemExit("give --raw-dir (DIII-D) or --constant-range (NSTX)")
    files = []
    for d in args.dirs:
        p = data_path(d)
        files += sorted(p / f for f in os.listdir(p) if f.endswith(".joblib"))
    files = files[args.shard :: args.n_shards]
    if args.check:
        k = 0
        for f in files:
            if k >= args.check:
                break
            shot = re.findall(r"\d{6}", f.stem)
            raw = args.raw_dir / f"{shot[-1]}_cer_spectra.h5" if shot else None
            if raw is None or not raw.exists():
                continue
            n = load(f, mmap_mode="r")["input"].shape[-1]
            if not args.min_bins <= n <= args.max_bins:
                continue
            est = estimate_axis(f, raw)
            if est is None:
                continue
            first, last, corr = est
            line = f"{f.name} ({n} bins): fitted {first:.3f} -> {last:.3f} (corr {corr:.5f})"
            for ch in ("notebook", "all"):
                hi, lo = union_range(raw, ch)
                line += f" | rule[{ch}] {hi:.3f} -> {lo:.3f} (dev {first - hi:+.3f}/{last - lo:+.3f})"
            print(line, flush=True)
            k += 1
        print(f"checked {k} files")
        return
    done = skipped = failed = 0
    for f in files:
        z = load(f, mmap_mode="r")
        if "wavelength_nm" in z:
            skipped += 1
            continue
        axis, why = rebuilt_axis(f, args.raw_dir, args.constant_range, args.chords)
        if axis is None:
            failed += 1
            print(f"SKIP {f.name}: {why}", flush=True)
            continue
        if not args.dry_run:
            add_axis(f, axis, why)
        done += 1
    verb = "would add" if args.dry_run else "added"
    print(
        f"shard {args.shard}/{args.n_shards}: {verb} axis to {done} files, "
        f"{skipped} already had one, {failed} left untouched"
    )


if __name__ == "__main__":
    main()
