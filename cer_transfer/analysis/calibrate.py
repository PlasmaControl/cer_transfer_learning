"""Calibrate a checkpoint's predicted uncertainties by temperature scaling.

    pixi run python -u -m cer_transfer.analysis.calibrate --checkpoint cer_ckpts/nstx_passive_ft.pt \
        --list splits/nstx_passive_val.txt [--limit 400] [--per-chord]

For each target, finds the scalar temperature k that makes the model's
predicted sigma statistically honest against the labels:
    z = (y - mu) / sqrt((k * sigma_pred)^2 + sigma_label^2)
k is chosen so median |z| = 0.674 (median of |N(0,1)|) — robust to the
heavy tails that break variance-matching. Reported per target, optionally
per chord (--per-chord, written to --out CSV), and stored in the
checkpoint under "calibration" with --write (a separate, explicit step;
nothing is modified by default).

Use: predicted sigma for downstream plots/claims = k * sigma_pred.
The label-sigma term uses the labels' quoted errors at face value; where
those are themselves overconfident (high amplitude, see ceiling_check
--label-consistency) k absorbs the difference conservatively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cer_transfer.configs import ModelConfig, get_machine
from cer_transfer.data import ShotDataset
from cer_transfer.models import build_model
from cer_transfer.training import denormalize_mu_sigma, load_checkpoint


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--list", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--per-chord", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV of per-chord temperatures (with --per-chord)",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="store the temperatures in the checkpoint under "
        "'calibration' (explicit opt-in)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    mk = ckpt["machine"]
    machine = get_machine(mk["name"] if isinstance(mk, dict) else mk)
    cfg = ModelConfig(**ckpt["model_config"])
    model = build_model(machine, cfg).to(args.device).eval()
    model.load_state_dict(ckpt["model_state"])
    stats = {k: v.to(args.device) for k, v in ckpt["norm_stats"].items()}
    files = [
        Path(l)
        for l in args.list.read_text().splitlines()
        if l.strip().endswith(".joblib")
    ]
    if args.limit:
        files = files[: args.limit]
    ds = ShotDataset(
        files, machine, subseq_len=-1, mmap=False, resample_w=cfg.resample_w
    )
    print(
        f"checkpoint: {args.checkpoint.name} (machine={machine.name}); "
        f"indexed {len(ds)} files"
    )

    n_t = len(machine.targets)
    res, sig_p, sig_l = (
        [[] for _ in range(n_t)],
        [[] for _ in range(n_t)],
        [[] for _ in range(n_t)],
    )
    chords = [[] for _ in range(n_t)]
    with torch.no_grad():
        for i in range(len(ds)):
            spec, tgt, err, moments = ds[i]
            x = spec.unsqueeze(0).to(args.device)
            mu_n, sigma_n = model(x, moments.unsqueeze(0).to(args.device))
            mu, sg = denormalize_mu_sigma(mu_n, sigma_n, stats, machine.targets)
            mu = mu[0].cpu().numpy()
            sg = sg[0].cpu().numpy()
            y = tgt.numpy()
            e = err.numpy()
            C = y.shape[0]
            cc = np.broadcast_to(np.arange(C)[:, None], y.shape[:2])
            for t in range(n_t):
                m = (
                    np.isfinite(y[..., t])
                    & np.isfinite(e[..., t])
                    & (e[..., t] > 0)
                    & np.isfinite(sg[..., t])
                )
                if not m.any():
                    continue
                res[t].append((y[..., t][m] - mu[..., t][m]))
                sig_p[t].append(sg[..., t][m])
                sig_l[t].append(e[..., t][m])
                chords[t].append(cc[m])
            if (i + 1) % 200 == 0:
                print(f"... {i + 1}/{len(ds)}")

    MED = 0.6745  # median of |N(0,1)|

    def temperature(r, sp, sl):
        """k s.t. median |r| / sqrt((k sp)^2 + sl^2) = MED, by bisection."""
        lo, hi = 1e-3, 1e3
        for _ in range(60):
            k = np.sqrt(lo * hi)
            z = np.median(np.abs(r) / np.sqrt((k * sp) ** 2 + sl**2))
            if z > MED:
                lo = k
            else:
                hi = k
        return float(np.sqrt(lo * hi))

    calib = {}
    rows = []
    for t, name in enumerate(machine.targets):
        r = np.concatenate(res[t])
        sp = np.concatenate(sig_p[t])
        sl = np.concatenate(sig_l[t])
        ch = np.concatenate(chords[t])
        z_raw = np.median(np.abs(r) / np.sqrt(sp**2 + sl**2))
        k = temperature(r, sp, sl)
        z_cal = np.median(np.abs(r) / np.sqrt((k * sp) ** 2 + sl**2))
        print(
            f"{name}: raw median|z| {z_raw:.2f} -> k = {k:.3f} "
            f"(calibrated median|z| {z_cal:.3f}, n={len(r):,})"
        )
        calib[name] = k
        if args.per_chord:
            for c in np.unique(ch):
                m = ch == c
                kc = temperature(r[m], sp[m], sl[m])
                rows.append((name, int(c), kc, int(m.sum())))
    if args.per_chord:
        print("\nper-chord temperatures (target, chord, k, n):")
        for name, c, kc, n in rows:
            print(f"  {name:>6} {c:>3} {kc:7.3f} {n:>9,}")
        if args.out:
            import csv

            with open(args.out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["target", "chord", "k", "n"])
                w.writerows(rows)
            print(f"wrote {args.out}")
    if args.write:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ck["calibration"] = calib
        torch.save(ck, args.checkpoint)
        print(f"stored calibration {calib} in {args.checkpoint}")


if __name__ == "__main__":
    main()
