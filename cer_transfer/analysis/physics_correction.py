"""Zero-shot evaluation protocol: one label-free metadata correction, applied to both targets.

A model trained on the source machine maps the observed line width w (in bins
of its input grid) to T_i through the source's calibration,

    T_pred = K [ (w p_src)^2 - sigma_src^2 ],     K = m c^2 / lambda_0^2,

where p_src is the nm per input bin the source spectra had during training
(window / (256 * stretch)) and sigma_src the source instrument width. On a
target machine the same bins mean p_tgt nm each and the instrument width is
sigma_tgt, so with s = p_tgt / p_src the physically consistent outputs are

    v_corr = s v_pred                      (Doppler shift: linear in the scale)
    T_corr = s^2 (T_pred + K sigma_src^2) - K sigma_tgt^2 .

The same s is applied to both targets; this is the evaluation step of the
paper, identical for T_i and v_tor.

Every constant comes from metadata (windows, stretch, lambda_0, ion mass) or
from ``cer_transfer.analysis.instrument_width`` (spectra only). No label is
used, so the corrected score is still a zero-shot result -- provided the model
really encodes T_i as a width, which this script tests: it prints the slope
and offset the physics predicts next to the ones a label-based affine fit
finds (diagnostic). Agreement within a few tens of percent means the model is
physical and the correction is admissible; a large mismatch means it reads
T_i from other cues and the correction is not.

    pixi run --frozen python -u -m cer_transfer.analysis.physics_correction \\
        --preds gallery/zs_d3d_full_dg.npz \\
        --src-window-nm 6.0 --src-stretch 1.4 --tgt-window-nm 2.08 \\
        --sigma-src gallery/instrument_width_d3d.csv \\
        --sigma-tgt gallery/instrument_width_nstx.csv

``--sigma-src/--sigma-tgt`` take a number (nm) or an instrument_width CSV
(source: median over chords of the label-anchored column; target: per-chord
lower envelope). Note that the envelope is an upper bound on sigma_tgt: if
the target has no cold frames, the corrected T_i is biased low by
K (envelope^2 - sigma_tgt^2) -- a few hundred eV at most; see instrument_width.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cer_transfer.analysis.affine_fit import affine, scores
from cer_transfer.figures.common import load_predictions

REF_BINS = 256
AMU_EV = 931.494_102e6  # eV / c^2


def read_sigma(arg, n_chords=None, column="sigma_inst_nm"):
    """Instrument width in nm: a number, or per-chord values from a CSV column.

    Returns (per_chord_array_or_None, median).
    """
    try:
        v = float(arg)
        return None, v
    except ValueError:
        pass
    rows = list(csv.DictReader(open(arg)))
    vals = {int(r["chord"]): float(r[column]) for r in rows if r.get(column, "") != ""}
    med = float(np.nanmedian(list(vals.values())))
    if n_chords is None:
        return None, med
    per = np.full(n_chords, med)
    for c, s in vals.items():
        if 0 <= c < n_chords and np.isfinite(s):
            per[c] = s
    return per, med


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preds", required=True, help="eval_checkpoint.py --dump-preds file")
    ap.add_argument("--src-window-nm", type=float, required=True, help="source window span")
    ap.add_argument("--src-stretch", type=float, required=True, help="source wavelength_scale used in training")
    ap.add_argument("--tgt-window-nm", type=float, required=True, help="target window span")
    ap.add_argument("--tgt-stretch", type=float, default=1.0, help="target wavelength_scale (1 for the reference)")
    ap.add_argument("--sigma-src", required=True, help="source sigma_inst: nm or instrument_width CSV")
    ap.add_argument("--sigma-tgt", required=True, help="target sigma_inst: nm or instrument_width CSV")
    ap.add_argument(
        "--sigma-column",
        default="sigma_anchored_nm",
        help="CSV column for the SOURCE width (default: label-anchored estimate; "
        "'sigma_inst_nm' = lower envelope). The target always uses the envelope.",
    )
    ap.add_argument("--lambda0-nm", type=float, default=529.05)
    ap.add_argument("--mass-amu", type=float, default=12.0)
    ap.add_argument("--y-max", type=float, default=None, help="score only labels <= y_max eV")
    ap.add_argument("--out", type=Path, default=None, help="write corrected dump (npz)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="write one row per target: target, r2_raw, r2_protocol, r2_affine, slope_ratio, n")
    args = ap.parse_args()

    d = load_predictions(args.preds)
    names = d.targets or ["ti", "vtor"]
    k_ti = names.index("ti")
    sigma_src_pc, sigma_src = read_sigma(args.sigma_src, column=args.sigma_column)
    sigma_tgt_pc, sigma_tgt = read_sigma(args.sigma_tgt, d.C, column="sigma_inst_nm")

    K = args.mass_amu * AMU_EV / args.lambda0_nm**2  # eV per nm^2
    p_src = args.src_window_nm / (REF_BINS * args.src_stretch)
    p_tgt = args.tgt_window_nm / (REF_BINS * args.tgt_stretch)
    s2 = (p_tgt / p_src) ** 2
    sig_t = sigma_tgt_pc if sigma_tgt_pc is not None else np.full(d.C, sigma_tgt)
    offset_pc = s2 * K * sigma_src**2 - K * sig_t**2  # per target chord

    print(
        f"K = {K:.4g} eV/nm^2 | p_src = {p_src:.5f} nm/bin, p_tgt = {p_tgt:.5f} nm/bin, "
        f"s = {np.sqrt(s2):.4f} | sigma_src = {sigma_src:.5f} nm ({K * sigma_src**2:.0f} eV), "
        f"sigma_tgt = {sigma_tgt:.5f} nm ({K * sigma_tgt**2:.0f} eV)"
    )
    print(
        f"protocol (label-free, both targets): v_corr = {np.sqrt(s2):.4f} * v_pred ; "
        f"T_corr = {s2:.4f} * T_pred + {np.median(offset_pc):.4g} eV"
        + (f" (per-chord offsets {offset_pc.min():.4g} .. {offset_pc.max():.4g})"
           if sigma_tgt_pc is not None else "")
    )
    corrected = {}
    rows = []
    for k, name in enumerate(names):
        pred = d.pred[..., k].astype(np.float64)
        y = d.y[..., k].astype(np.float64)
        if name == "ti":
            corr = s2 * pred + offset_pc[:, None]
            phys_slope = s2
        elif name == "vtor":
            corr = np.sqrt(s2) * pred
            phys_slope = np.sqrt(s2)
        else:
            continue
        corrected[name] = corr
        ok = np.isfinite(y) & np.isfinite(pred)
        if args.y_max is not None and name == "ti":
            ok &= y <= args.y_max
        a, b = affine(pred[ok], y[ok])
        r2_raw, rmse_raw = scores(pred[ok], y[ok])
        r2_cor, rmse_cor = scores(corr[ok], y[ok])
        r2_aff, rmse_aff = scores(a * pred[ok] + b, y[ok])
        ratio = phys_slope / a if a else float("nan")
        unit = "eV" if name == "ti" else "km/s"
        print(
            f"{name}: R2 raw {r2_raw:.3f} | protocol {r2_cor:.3f} | affine (diagnostic) {r2_aff:.3f}"
            f"   RMSE {rmse_raw:.4g} -> {rmse_cor:.4g} ({rmse_aff:.4g}) {unit} | n = {int(ok.sum())}"
        )
        print(
            f"      affine fit y = {a:.4f} * pred + {b:.4g} {unit}; "
            f"slope physics/affine = {ratio:.2f}"
        )
        rows.append((name, r2_raw, r2_cor, r2_aff, ratio, int(ok.sum())))
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["target", "r2_raw", "r2_protocol", "r2_affine", "slope_ratio", "n"])
            w.writerows(rows)
        print(f"wrote {args.csv}")
    pred_ti = d.pred[..., k_ti]  # for --out below
    if args.out is not None:
        raw = np.load(args.preds)
        out = {k: raw[k] for k in raw.files}
        p = out["pred"].reshape(d.C, d.T, -1).astype(np.float32)
        for name, corr in corrected.items():
            p[..., names.index(name)] = corr.astype(np.float32)
        out["pred"] = p.reshape(-1, p.shape[-1])
        np.savez_compressed(args.out, **out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
