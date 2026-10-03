"""Find the wavelength-axis convention of a target machine for zero-shot
transfer, from the data: evaluate an agnostic checkpoint on target
discharges with the wavelength axis optionally reversed and stretched by a
factor, and report, per combination, the affine R2 of each target (slope,
offset and R2 after fitting ``y = a * pred + b``). The maximizing (flip,
scale) is the axis orientation and dispersion ratio between the two
spectrometers; the R2 left at the maximum is the transfer error that is not
an axis convention.

Target labels are used only to score two physical constants, not to train.

    pixi run python -u -m cer_transfer.analysis.axis_sweep \\
        --checkpoint cer_ckpts/nstx_agnostic_quick.pt --machine d3d_tangential \\
        --list splits/d3d_val.txt --limit 100 \\
        --scales 0.33 0.5 0.7 1 1.4 2 3 --out gallery/axis_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from cer_transfer.analysis.affine_fit import affine, scores
from cer_transfer.configs import data_path
from cer_transfer.data import ShotDataset
from cer_transfer.inference import load_model
from cer_transfer.training import denormalize_mu_sigma


@torch.no_grad()
def predict(m, files, flip, scale):
    """Pooled (pred, y) per target over the files for one axis setting."""
    ds = ShotDataset(
        files,
        m.machine,
        subseq_len=-1,
        mmap=False,
        resample_w=m.config.resample_w,
        w_scale=scale,
        flip_w=flip,
    )
    preds, ys = [], []
    for i in range(len(ds)):
        spec, tgt, _, moments = ds[i]
        mu_n, sig_n = m.model(
            spec.unsqueeze(0).to(m.device), moments.unsqueeze(0).to(m.device)
        )
        mu, _ = denormalize_mu_sigma(mu_n, sig_n, m.stats, m.machine.targets)
        preds.append(mu[0].cpu().numpy().reshape(-1, len(m.machine.targets)))
        ys.append(tgt.numpy().reshape(-1, len(m.machine.targets)))
    return np.concatenate(preds), np.concatenate(ys)


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--machine", required=True, help="target machine of the data")
    p.add_argument("--list", type=Path, required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument(
        "--scales", type=float, nargs="+", default=[0.33, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
    )
    p.add_argument("--no-flip", action="store_true", help="only test flip = False")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None, help="CSV of all results")
    args = p.parse_args()

    m = load_model(args.checkpoint, device=args.device, machine=args.machine)
    files = [
        data_path(ln.strip())
        for ln in args.list.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ][: args.limit]
    names = list(m.machine.targets)
    rows = []
    print(f"{len(files)} {args.machine} discharges; checkpoint {args.checkpoint.name}")
    print(
        f"{'flip':>5} {'scale':>6} | "
        + " | ".join(f"{n}: slope  offset  R2raw  R2aff" for n in names)
    )
    for flip in ((False,) if args.no_flip else (False, True)):
        for scale in args.scales:
            pred, y = predict(m, files, flip, scale)
            cells, rec = [], {"flip": int(flip), "scale": scale}
            for k, n in enumerate(names):
                ok = np.isfinite(y[:, k]) & np.isfinite(pred[:, k])
                a, b = affine(pred[ok, k], y[ok, k])
                r2_raw, _ = scores(pred[ok, k], y[ok, k])
                r2_aff, _ = scores(a * pred[ok, k] + b, y[ok, k])
                cells.append(f"{a:7.3f} {b:8.3g} {r2_raw:6.3f} {r2_aff:6.3f}")
                rec.update(
                    {
                        f"{n}_slope": a,
                        f"{n}_offset": b,
                        f"{n}_r2_raw": r2_raw,
                        f"{n}_r2_affine": r2_aff,
                    }
                )
            rows.append(rec)
            print(f"{str(flip):>5} {scale:6.2f} | " + " | ".join(cells), flush=True)
    for n in names:
        best = max(rows, key=lambda r: r[f"{n}_r2_affine"])
        print(
            f"best for {n}: flip={bool(best['flip'])} scale={best['scale']:.2f} -> "
            f"affine R2 {best[f'{n}_r2_affine']:.3f} (slope {best[f'{n}_slope']:.3f})"
        )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
