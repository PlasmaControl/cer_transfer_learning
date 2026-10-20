r"""Find discharges similar to a reference set, from what the files contain.

Each discharge is described by three numbers computed from its labels and
spectra (no external database):

* ``vtor_core``: median |v_tor| of the conventional fits on the core chords
  (low for low-torque and ohmic discharges),
* ``ti_core``: median T_i of the fits on the core chords,
* ``cvi_contrast``: median over beam-on frames and chords of the C VI peak
  height over the spectrum's 10th percentile, in the file's native window
  (low where C VI is weak and neighbouring lines matter).

The reference discharges span a box in these three numbers (``--quantiles``
of each, default 10th-90th percentile); candidates inside the box are printed,
nearest to the reference median first, and optionally written as a list.

    pixi run --frozen python -u -m cer_transfer.analysis.similar_shots \
        --ref-shots 194367 195604 190308 ... --ref-dir training_set_30 \
        --cand-dir test_set_30 --out splits/d3d_similar_test.txt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from joblib import load

from cer_transfer.configs import data_path

CORE = range(0, 16)  # tangential chords t01-t16: core views


def features(entry):
    """Return (vtor_core, ti_core, cvi_contrast) of one discharge, or None."""
    try:
        d = load(data_path(entry), mmap_mode="r")
    except Exception:
        return None
    end = int(d["end_index"])
    y = np.asarray(d["target"][:48, :end, :], np.float64)
    core = list(CORE)
    lab = np.isfinite(y[core, :, 1])
    if lab.sum() < 20:
        return None
    vt = float(np.median(np.abs(y[core, :, 1][lab])))
    ti = float(np.nanmedian(y[core, :, 0]))
    on = np.isfinite(y[:, :, 1]).any(0)  # frames with any fit: beam on
    x = np.asarray(d["input"][:48, :end:4, :], np.float64)
    x = x[:, on[::4]]
    if x.size == 0:
        return None
    base = np.percentile(x, 10, axis=-1)
    contrast = (x.max(-1) - base) / np.maximum(np.abs(base), 1.0)
    return vt, ti, float(np.median(contrast))


def entries_in(folder, shots=None):
    """List entries ``folder/<shot>.joblib``, optionally only the given shots."""
    root = data_path(folder)
    names = sorted(f for f in os.listdir(root) if f.endswith(".joblib"))
    if shots is not None:
        names = [f for f in names if f[: -len(".joblib")] in shots]
    return [f"{folder}/{f}" for f in names]


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref-shots", nargs="+", required=True)
    ap.add_argument("--ref-dir", default="training_set_30")
    ap.add_argument("--cand-dir", default="test_set_30")
    ap.add_argument("--quantiles", type=float, nargs=2, default=(10.0, 90.0))
    ap.add_argument("--max", type=int, default=30, help="candidates written")
    ap.add_argument("--out", type=Path, default=None, help="list of the selected")
    args = ap.parse_args()

    ref = [
        f for e in entries_in(args.ref_dir, set(args.ref_shots)) if (f := features(e))
    ]
    if not ref:
        raise SystemExit("no reference discharge found")
    ref = np.array(ref)
    lo = np.percentile(ref, args.quantiles[0], axis=0)
    hi = np.percentile(ref, args.quantiles[1], axis=0)
    med, scale = np.median(ref, axis=0), np.maximum(hi - lo, 1e-9)
    names = ("vtor_core km/s", "ti_core eV", "cvi_contrast")
    print(
        f"{len(ref)} reference discharges; box ({args.quantiles[0]:.0f}-"
        f"{args.quantiles[1]:.0f} %):"
    )
    for n, a, b, m in zip(names, lo, hi, med):
        print(f"  {n:15s} {a:9.3g} .. {b:9.3g}   (median {m:.3g})")

    cands = entries_in(args.cand_dir)
    sel = []
    for i, e in enumerate(cands):
        f = features(e)
        if f is None:
            continue
        f = np.array(f)
        if np.all((f >= lo) & (f <= hi)):
            sel.append((float(np.sqrt((((f - med) / scale) ** 2).sum())), e, f))
        if (i + 1) % 200 == 0:
            print(f"  scanned {i + 1}/{len(cands)}, {len(sel)} inside the box")
    sel.sort()
    print(
        f"\n{len(sel)} of {len(cands)} candidates in {args.cand_dir} inside the box; "
        f"nearest {min(args.max, len(sel))}:"
    )
    for dist, e, f in sel[: args.max]:
        print(
            f"  {e:32s} dist {dist:5.2f} | vtor {f[0]:6.1f} | ti {f[1]:6.0f} | "
            f"contrast {f[2]:6.2f}"
        )
    if args.out is not None and sel:
        args.out.write_text("\n".join(e for _, e, _ in sel[: args.max]) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
