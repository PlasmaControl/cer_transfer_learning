"""Label-free choice of the target machine's wavelength convention.

Following domain adaptation under location-scale conditional shift (Zhang,
Schölkopf, Muandet & Wang, ICML 2013): the spectrum given (T_i, v_tor)
differs between spectrometers by a location-scale transform of the
wavelength axis. That transform is chosen here **from spectra only**: for
each candidate (flip, stretch, shift) of the target axis, the trained model
encodes target spectra, and the candidate whose feature distribution is
closest to that of the source spectra (maximum mean discrepancy, RBF kernel)
wins. No label of either machine enters the choice.

Labels are used only afterwards, to score: the raw R2 of the unmodified
model output (no fitting) at each candidate, printed for transparency and
marked as evaluation.

    pixi run python -u -m cer_transfer.analysis.align_unsupervised \\
        --checkpoint cer_ckpts/ladder_nstx_r650.pt \\
        --source nstx:splits/nstx_val.txt --target d3d_tangential:splits/d3d_val.txt \\
        --n-shots 20 --out gallery/align_nstx_r650_to_d3d.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

import numpy as np
import torch
from joblib import load

from cer_transfer.configs import data_path, get_machine
from cer_transfer.data import Preprocessor
from cer_transfer.inference import load_model
from cer_transfer.training import denormalize_mu_sigma

WINDOW = 128  # frames per discharge (contiguous, around the strongest signal)
EDGE = 18  # frames dropped at each window edge (temporal receptive field)


def read_list(path, n):
    """First n entries of a file list, resolved against CER_DATA_ROOT."""
    lines = [
        ln.strip()
        for ln in Path(path).read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return [data_path(ln) for ln in lines[:n]]


def cache_windows(files, machine):
    """Raw spectra and labels of one contiguous high-signal window per shot.

    The window is chosen by line brightness (max minus median over the
    wavelength axis, summed over chords): spectra only, no labels.
    """
    out = []
    ci = (
        None
        if machine.input_channel_indices is None
        else np.asarray(machine.input_channel_indices)
    )
    ki = None if machine.chord_indices is None else np.asarray(machine.chord_indices)
    scale = np.asarray(machine.target_scale, dtype=np.float32)
    for f in files:
        d = load(f, mmap_mode="r")
        x = np.asarray(d["input"], dtype=np.float32)
        end = int(d["end_index"])
        T = x.shape[1] if end <= 0 else min(end, x.shape[1])
        if T < WINDOW:
            continue
        bright = (x[:, :T].max(-1) - np.median(x[:, :T], axis=-1)).sum(0)
        smooth = np.convolve(bright, np.ones(WINDOW) / WINDOW, mode="valid")
        a = int(np.argmax(smooth))
        spec = x[:, a : a + WINDOW]
        y = np.asarray(d["target"][:, a : a + WINDOW], dtype=np.float32) * scale
        if ki is not None:
            y = y[ki]
        out.append((spec, ci, y))
    return out


@torch.no_grad()
def encode(m, windows, machine, resample_w, flip, scale, shift, device):
    """Per-frame trunk features and raw predictions for one axis setting."""
    pre = Preprocessor(machine, resample_w, scale, flip, shift)
    feats, preds, ys, sigs = [], [], [], []
    for spec, ci, y in windows:
        x = pre(spec)
        if ci is not None:
            x = x[ci]
        x = x.unsqueeze(0).to(device)
        shared, chord_feats = m.model.backbone(x)
        f = shared[0].permute(1, 0, 2).reshape(shared.shape[2], -1)  # (T, F*H)
        mu_n, sig_n = m.model.head((shared, chord_feats))
        mu, _ = denormalize_mu_sigma(mu_n, sig_n, m.stats, m.machine.targets)
        sl = slice(EDGE, -EDGE)
        sigs.append(sig_n[0][:, sl].cpu().numpy())  # normalized units
        feats.append(f[sl].cpu().numpy())
        preds.append(mu[0][:, sl].cpu().numpy())
        ys.append(y[:, sl])
    return np.concatenate(feats), preds, ys, np.concatenate([q.ravel() for q in sigs])


def mmd(a, b, gamma):
    """Unbiased-free (biased) squared MMD with an RBF kernel."""

    def k(u, v):
        d = (u * u).sum(1)[:, None] + (v * v).sum(1)[None, :] - 2 * u @ v.T
        return np.exp(-gamma * np.clip(d, 0, None))

    return float(k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean())


def raw_r2(preds, ys, k):
    """R2 of the raw model output against the labels (evaluation only)."""
    p = np.concatenate([q[..., k].ravel() for q in preds]).astype(np.float64)
    y = np.concatenate([q[..., k].ravel() for q in ys]).astype(np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    return float(1 - np.mean((p - y) ** 2) / np.var(y)) if y.size > 10 else np.nan


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--source", required=True, help="machine:list (training machine)")
    ap.add_argument("--target", required=True, help="machine:list (new machine)")
    ap.add_argument("--n-shots", type=int, default=20)
    ap.add_argument(
        "--n-frames", type=int, default=1500, help="frames per side for MMD"
    )
    ap.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0],
    )
    ap.add_argument(
        "--shifts",
        type=float,
        nargs="+",
        default=[-40, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 40],
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--criterion",
        choices=("sigma", "mmd"),
        default="sigma",
        help="label-free selection: the model's median predicted uncertainty "
        "(sigma) or the feature-distribution distance to the source (mmd)",
    )
    ap.add_argument(
        "--keep-stored",
        action="store_true",
        help="search relative to the stored target convention (default: from the "
        "raw target axis, strictly label-free)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    s_name, s_list = args.source.split(":", 1)
    t_name, t_list = args.target.split(":", 1)
    m = load_model(args.checkpoint, device=args.device, machine=t_name)
    rw = m.config.resample_w
    src_m, tgt_m = get_machine(s_name), get_machine(t_name)
    if not args.keep_stored:
        # strictly label-free: start from the raw target axis, not from a
        # convention that was derived with labels
        tgt_m = dataclasses.replace(
            tgt_m, wavelength_flip=False, wavelength_scale=1.0, wavelength_shift=0.0
        )
    src_w = cache_windows(read_list(s_list, args.n_shots), src_m)
    tgt_w = cache_windows(read_list(t_list, args.n_shots), tgt_m)
    print(
        f"{len(src_w)} {s_name} and {len(tgt_w)} {t_name} discharges, "
        f"{WINDOW - 2 * EDGE} frames each; stored {t_name} convention "
        f"flip={tgt_m.wavelength_flip} scale={tgt_m.wavelength_scale} "
        f"shift={tgt_m.wavelength_shift}"
    )

    # source features, standardized; kernel width from the source (median rule)
    fs, _, _, _ = encode(m, src_w, src_m, rw, False, 1.0, 0.0, args.device)
    mean, std = fs.mean(0), fs.std(0) + 1e-6
    fs = (fs - mean) / std
    fs = fs[rng.choice(len(fs), min(args.n_frames, len(fs)), replace=False)]
    sub = fs[rng.choice(len(fs), min(500, len(fs)), replace=False)]
    d2 = ((sub[:, None] - sub[None]) ** 2).sum(-1)
    gamma = 1.0 / np.median(d2[d2 > 0])

    rows = []

    def score(flip, scale, shift):
        ft, preds, ys, sg = encode(m, tgt_w, tgt_m, rw, flip, scale, shift, args.device)
        ft = (ft - mean) / std
        ft = ft[rng.choice(len(ft), min(args.n_frames, len(ft)), replace=False)]
        r = dict(
            flip=int(flip),
            scale=scale,
            shift=shift,
            mmd=mmd(fs, ft, gamma),
            sigma=float(np.median(sg)),
            r2_ti_raw=raw_r2(preds, ys, 0),
            r2_vtor_raw=raw_r2(preds, ys, 1),
        )
        rows.append(r)
        print(
            f"  flip={flip!s:5} scale={scale:5.2f} shift={shift:6.1f} | "
            f"MMD {r['mmd']:.4f} sigma {r['sigma']:.4f} | eval: raw R2 Ti {r['r2_ti_raw']:7.3f} "
            f"vtor {r['r2_vtor_raw']:7.3f}",
            flush=True,
        )
        return r[args.criterion]

    print(f"stage 1: orientation x stretch (selection by {args.criterion} only)")
    best = min(
        ((f, s) for f in (False, True) for s in args.scales),
        key=lambda fs_: score(fs_[0], fs_[1], 0.0),
    )
    print("stage 2: shift at the best orientation/stretch")
    best_shift = min(args.shifts, key=lambda sh: score(best[0], best[1], sh))
    print("stage 3: stretch refinement at the best shift")
    fine = sorted({round(best[1] * f, 3) for f in (0.9, 0.95, 1.05, 1.1)})
    best_scale = min([best[1], *fine], key=lambda s: score(best[0], s, best_shift))
    chosen = next(
        r
        for r in reversed(rows)
        if r["flip"] == int(best[0])
        and r["scale"] == best_scale
        and r["shift"] == best_shift
    )

    print(
        f"\nchosen without labels: flip={best[0]} scale={best_scale} shift={best_shift} "
        f"(relative to the stored {t_name} convention)"
    )
    print(
        f"evaluation of the unmodified model output there: raw R2 "
        f"Ti {chosen['r2_ti_raw']:.3f}, vtor {chosen['r2_vtor_raw']:.3f}"
    )
    base = next(
        (r for r in rows if r["flip"] == 0 and r["scale"] == 1.0 and r["shift"] == 0.0),
        None,
    )
    if base:
        print(
            f"stored convention (no probe): raw R2 Ti {base['r2_ti_raw']:.3f}, "
            f"vtor {base['r2_vtor_raw']:.3f}"
        )
    print(
        "full evaluation with the chosen setting:\n"
        f"  eval_checkpoint.py --checkpoint {args.checkpoint} --machine {t_name} "
        f"--list {t_list} --limit 200"
        + (" --flip-w" if best[0] else "")
        + f" --w-scale {best_scale} --w-shift {best_shift}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
