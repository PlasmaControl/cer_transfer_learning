"""Evaluate a checkpoint on a file list: overall, per-chord, and
per-|vtor|-bin metrics. The diagnostic for WHERE performance is lost.

    pixi run python -u eval_checkpoint.py \
        --checkpoint cer_ckpts/nstx_scratch.pt \
        --list splits/nstx_val.txt [--device cuda] [--limit 200] \
        [--out eval_nstx_scratch.csv]

Reports per target (canonical units):
  overall   R2, RMSE, chi2 (mean ((pred-y)/sigma_label)^2; ~1 at noise floor)
  per-chord R2 table (min / p25 / median / max) + worst chords named
  per-bin   R2/RMSE/chi2 binned by |vtor| label quintiles (mechanism
            localization: gap at high |vtor| -> window-edge/blending physics;
            uniform -> optimization; chord-structured -> extraction bug)

Signature guide (from the project notes):
  - a few chords with strongly negative R2 while others are fine
    -> suspect chord-index misalignment in extraction
  - vtor R2 collapsing in the top |vtor| bin -> high-Mach physics
  - uniform shortfall vs cer_transfer.analysis.ceiling_check -> tuning/architecture
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from cer_transfer.configs import ModelConfig, data_path, get_machine
from cer_transfer.data import ShotDataset
from cer_transfer.models import build_model
from cer_transfer.training import denormalize_mu_sigma, load_checkpoint


@torch.no_grad()
def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--flip-w",
        action="store_true",
        help="reverse the wavelength axis (zero-shot probe)",
    )
    p.add_argument(
        "--w-scale",
        type=float,
        default=1.0,
        help="stretch the wavelength axis by this factor before cropping/padding "
        "to --resample-w (zero-shot probe)",
    )
    p.add_argument(
        "--machine",
        type=str,
        default=None,
        help="evaluate on another machine's data (agnostic checkpoints only, "
        "e.g. a DIII-D model on NSTX lists)",
    )
    p.add_argument("--list", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--n-bins", type=int, default=5)
    p.add_argument(
        "--first-chords",
        type=int,
        default=0,
        metavar="N",
        help="restrict METRICS to chords [0, N) — e.g. 48 on d3d "
        "= tangential views only (vertical-chord vtor is a "
        "different velocity projection). Model still runs on "
        "all chords; only the evaluation is sliced.",
    )
    p.add_argument(
        "--out", type=Path, default=None, help="optional CSV of per-chord metrics"
    )
    p.add_argument(
        "--dump-preds",
        type=Path,
        default=None,
        help="write per-point predictions to a compressed .npz "
        "(chord, y, pred, sigma_label per target) so any "
        "future metric is an offline recompute, no GPU",
    )
    p.add_argument(
        "--predict-only",
        action="store_true",
        help="skip all metrics and only write --dump-preds "
        "(for shots without conventional fits, e.g. "
        "ohmic discharges)",
    )
    p.add_argument(
        "--diagnose",
        action="store_true",
        help="residual anatomy: R2 vs label time-shift (timing "
        "misalignment test), residual vs line amplitude "
        "(low-signal test), frame-mean residual histogram "
        "(bimodality test)",
    )
    args = p.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    cfg = ModelConfig(**ckpt["model_config"])
    if args.machine is not None:
        if not cfg.agnostic:
            raise SystemExit(
                "--machine is only valid for an agnostic checkpoint; this one has "
                "machine-specific parameters"
            )
        ckpt["machine"] = {"name": args.machine}
    mk = ckpt["machine"]
    machine = get_machine(mk["name"] if isinstance(mk, dict) else mk)
    model = build_model(machine, cfg).to(args.device).eval()
    model.load_state_dict(ckpt["model_state"])
    stats = {k: v.to(args.device) for k, v in ckpt["norm_stats"].items()}
    print(
        f"checkpoint: {args.checkpoint.name} (machine={machine.name}, "
        f"best={ckpt.get('best_score')}, epoch={ckpt.get('epoch')})"
    )

    files = [
        data_path(ln.strip())
        for ln in args.list.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    missing = [f for f in files if not f.exists()]
    if missing:
        import os

        root = os.environ.get("CER_DATA_ROOT")
        raise FileNotFoundError(
            f"{len(missing)} of {len(files)} files in {args.list} missing, e.g. "
            f"{missing[0]}"
            + (
                " (CER_DATA_ROOT is not set)"
                if root is None
                else f" (CER_DATA_ROOT={root})"
            )
        )
    if args.limit:
        files = files[: args.limit]
    ds = ShotDataset(
        files,
        machine,
        subseq_len=-1,
        mmap=False,
        resample_w=cfg.resample_w,
        w_scale=args.w_scale,
        flip_w=args.flip_w,
    )

    n_t = len(machine.targets)
    C = machine.n_chords
    preds, labels, sigmas_l = [], [], []  # per-shot flattened collections
    psigs = []  # model-predicted sigmas
    amps = []  # per-(chord, frame) line amp m0
    for i in range(len(ds)):
        spec, tgt, err, moments = ds[i]
        x = spec.unsqueeze(0).to(args.device)
        mu_n, sigma_n = model(x, moments.unsqueeze(0).to(args.device))
        mu, psig = denormalize_mu_sigma(mu_n, sigma_n, stats, machine.targets)
        preds.append(mu[0].cpu().numpy())  # (C, T, n_t)
        psigs.append(psig[0].cpu().numpy())
        labels.append(tgt.numpy())
        sigmas_l.append(err.numpy())
        amps.append(moments[..., 0].numpy())  # scaled log10 amplitude
        if (i + 1) % 200 == 0:
            print(f"... {i + 1}/{len(ds)} shots")

    pred = np.concatenate([p.reshape(C, -1, n_t) for p in preds], axis=1)
    lab = np.concatenate([l.reshape(C, -1, n_t) for l in labels], axis=1)
    sig = np.concatenate([s.reshape(C, -1, n_t) for s in sigmas_l], axis=1)
    if args.first_chords:
        C = min(args.first_chords, C)
        pred, lab, sig = pred[:C], lab[:C], sig[:C]
        print(f"metrics restricted to chords [0, {C})")

    if args.predict_only:
        if args.dump_preds is None:
            raise SystemExit("--predict-only requires --dump-preds")
        cc_all = np.broadcast_to(
            np.arange(lab.shape[0])[:, None], lab.shape[:2]
        ).ravel()
        np.savez_compressed(
            args.dump_preds,
            chord=cc_all.astype(np.int16),
            y=lab.reshape(-1, lab.shape[-1]).astype(np.float32),
            pred=pred.reshape(-1, pred.shape[-1]).astype(np.float32),
            sigma=sig.reshape(-1, sig.shape[-1]).astype(np.float32),
            pred_sigma=np.concatenate([q[:C] for q in psigs], axis=1)
            .reshape(-1, n_t)
            .astype(np.float32),
            targets=np.array(machine.targets),
        )
        n_lab = int(np.isfinite(lab[..., 0]).sum())
        print(
            f"predict-only: dumped {pred.shape[1]} frames x {C} chords "
            f"({n_lab} labeled points) -> {args.dump_preds}"
        )
        return

    def r2(y, yh):
        """R2 over finite labels (NaN below 10 points)."""
        m = np.isfinite(y)
        if m.sum() < 10:
            return np.nan
        y, yh = y[m], yh[m]
        ss = ((y - yh) ** 2).sum()
        tv = ((y - y.mean()) ** 2).sum()
        return 1 - ss / tv if tv > 0 else np.nan

    def rmse(y, yh):
        """RMSE over finite labels."""
        m = np.isfinite(y)
        return float(np.sqrt(((y[m] - yh[m]) ** 2).mean())) if m.any() else np.nan

    def medae(y, yh):
        """Median absolute error over finite labels."""
        m = np.isfinite(y) & np.isfinite(yh)
        return float(np.median(np.abs(y[m] - yh[m]))) if m.any() else np.nan

    def chi2(y, yh, s):
        """Mean squared error in units of the label sigma."""
        m = np.isfinite(y) & np.isfinite(s) & (s > 0)
        return float((((y[m] - yh[m]) / s[m]) ** 2).mean()) if m.any() else np.nan

    per_chord = np.full((C, n_t), np.nan)
    per_chord_rmse = np.full((C, n_t), np.nan)
    per_chord_medae = np.full((C, n_t), np.nan)
    for t, name in enumerate(machine.targets):
        y, yh, s = lab[..., t], pred[..., t], sig[..., t]
        print(f"\n=== {name} ===")
        m_ = np.isfinite(y) & np.isfinite(yh)
        ae_ = np.abs(y[m_] - yh[m_])
        print(
            f"overall: R2 {r2(y, yh):.4f} | RMSE {rmse(y, yh):.4g} | "
            f"chi2 {chi2(y, yh, s):.2f} | pooled medae "
            f"{np.median(ae_):.4g} | pooled p95 "
            f"{np.percentile(ae_, 95):.4g}"
        )
        for c in range(C):
            per_chord[c, t] = r2(y[c], yh[c])
            per_chord_rmse[c, t] = rmse(y[c], yh[c])
            per_chord_medae[c, t] = medae(y[c], yh[c])
        pc = per_chord[:, t]
        finite = pc[np.isfinite(pc)]
        if finite.size == 0:
            print("per-chord R2: no chord with enough labeled points")
            continue
        print(
            f"per-chord R2: min {np.min(finite):.3f} | "
            f"p25 {np.percentile(finite, 25):.3f} | "
            f"median {np.median(finite):.3f} | max {np.max(finite):.3f}"
        )
        worst = np.argsort(np.where(np.isfinite(pc), pc, np.inf))[:5]
        print("worst chords: " + ", ".join(f"#{c} ({pc[c]:.3f})" for c in worst))
        # |vtor|-binned (bin by the LABEL vtor magnitude, all targets)
        vmag = np.abs(lab[..., 1])
        mask = np.isfinite(y) & np.isfinite(vmag)
        edges = np.nanquantile(vmag[mask], np.linspace(0, 1, args.n_bins + 1))
        edges[-1] += 1e-9
        print("|vtor|-binned (label km/s):")
        for b in range(args.n_bins):
            bm = mask & (vmag >= edges[b]) & (vmag < edges[b + 1])
            print(
                f"  [{edges[b]:7.1f},{edges[b+1]:7.1f}) "
                f"n={int(bm.sum()):>9,}  R2 {r2(y[bm], yh[bm]):7.4f}  "
                f"RMSE {rmse(y[bm], yh[bm]):8.4g}  "
                f"chi2 {chi2(y[bm], yh[bm], s[bm]):6.2f}"
            )

    if args.dump_preds:
        cc_all = np.broadcast_to(
            np.arange(lab.shape[0])[:, None], lab.shape[:2]
        ).ravel()
        np.savez_compressed(
            args.dump_preds,
            chord=cc_all.astype(np.int16),
            y=lab.reshape(-1, lab.shape[-1]).astype(np.float32),
            pred=pred.reshape(-1, pred.shape[-1]).astype(np.float32),
            sigma=sig.reshape(-1, sig.shape[-1]).astype(np.float32),
            pred_sigma=np.concatenate([p.reshape(C, -1, n_t) for p in psigs], axis=1)
            .reshape(-1, n_t)
            .astype(np.float32),
            targets=np.array(machine.targets),
        )
        print(f"dumped per-point predictions -> {args.dump_preds}")
    if args.diagnose:
        print("\n=== diagnostics ===")
        amp = np.concatenate([a.reshape(C, -1) for a in amps], axis=1)
        if args.first_chords:
            amp = amp[: min(args.first_chords, amp.shape[0])]
        # (1) label time-shift scan, per shot then pooled (vtor, the
        # position-sensitive target). Shift labels by k frames vs preds.
        print("R2(vtor) vs label time-shift (frames): timing test")
        for k in range(-3, 4):
            num = den = 0.0
            cnt = 0
            for p, l in zip(preds, labels):
                pv, lv = p[..., 1], l[..., 1]
                if k > 0:
                    pv, lv = pv[:, k:], lv[:, : lv.shape[1] - k]
                elif k < 0:
                    pv, lv = pv[:, :k], lv[:, -k:]
                m = np.isfinite(lv)
                if m.sum() < 10:
                    continue
                num += ((lv[m] - pv[m]) ** 2).sum()
                den += ((lv[m] - lv[m].mean()) ** 2).sum()
                cnt += m.sum()
            print(f"  shift {k:+d}: R2 {1 - num / den:7.4f}  (n={cnt:,})")
        # (2) residual (in label sigmas) vs line-amplitude decile
        print("\n|resid|/sigma vs line-amplitude decile: low-signal test")
        for t, name in enumerate(machine.targets):
            y, yh, s = lab[..., t], pred[..., t], sig[..., t]
            m = np.isfinite(y) & np.isfinite(s) & (s > 0) & np.isfinite(amp)
            z = np.abs(y[m] - yh[m]) / s[m]
            a = amp[m]
            edges = np.quantile(a, np.linspace(0, 1, 11))
            edges[-1] += 1e-9
            row = []
            for b in range(10):
                bm = (a >= edges[b]) & (a < edges[b + 1])
                row.append(np.median(z[bm]) if bm.any() else np.nan)
            print(
                f"  {name}: median|z| per amp decile: "
                + " ".join(f"{v:5.2f}" for v in row)
            )
        # (3) frame-mean residual histogram (bimodality test)
        print("\nframe-mean |z| distribution (bimodality test):")
        for t, name in enumerate(machine.targets):
            fm = []
            for p, l, s in zip(preds, labels, sigmas_l):
                y, yh, sg = l[..., t], p[..., t], s[..., t]
                m = np.isfinite(y) & np.isfinite(sg) & (sg > 0)
                with np.errstate(invalid="ignore"):
                    z = np.where(m, np.abs(y - yh) / np.where(m, sg, 1), np.nan)
                fmean = np.nanmean(z, axis=0)  # (T,)
                fm.append(fmean[np.isfinite(fmean)])
            fm = np.concatenate(fm)
            qs = np.percentile(fm, [5, 25, 50, 75, 95, 99])
            print(
                f"  {name}: frame|z| p5 {qs[0]:.2f} p25 {qs[1]:.2f} "
                f"median {qs[2]:.2f} p75 {qs[3]:.2f} p95 {qs[4]:.2f} "
                f"p99 {qs[5]:.2f}"
            )
    if args.diagnose:
        # (4) offset-correction test: subtract each (shot, chord)'s mean
        # SIGNED residual, recompute R2 — separates constant per-shot/chord
        # offsets (calibration-era zero points) from irreducible error.
        print(
            "\noffset-corrected R2 (per-(shot,chord) mean signed residual " "removed):"
        )
        for t, name in enumerate(machine.targets):
            num = den = 0.0
            pc_plain = [[] for _ in range(C)]
            pc_corr = [[] for _ in range(C)]
            for p, l in zip(preds, labels):
                y, yh = l[..., t], p[..., t]
                for cc in range(C):
                    m = np.isfinite(y[cc])
                    if m.sum() < 10:
                        continue
                    r = y[cc][m] - yh[cc][m]
                    pc_plain[cc].append((r, y[cc][m]))
                    r_c = r - r.mean()
                    pc_corr[cc].append((r_c, y[cc][m]))

            def pooled_r2(store):
                """R2 of pooled per-chord residuals (plain and corrected)."""
                num = den = 0.0
                allv = []
                for cc in range(C):
                    for r, yv in store[cc]:
                        num += (r**2).sum()
                        allv.append(yv)
                y_all = np.concatenate(allv)
                den = ((y_all - y_all.mean()) ** 2).sum()
                return 1 - num / den

            def chord_medians(store):
                """Median absolute error per chord."""
                meds = []
                for cc in range(C):
                    if not store[cc]:
                        continue
                    num = sum((r**2).sum() for r, _ in store[cc])
                    yv = np.concatenate([y for _, y in store[cc]])
                    den = ((yv - yv.mean()) ** 2).sum()
                    if den > 0:
                        meds.append(1 - num / den)
                return float(np.median(meds))

            print(
                f"  {name}: pooled {pooled_r2(pc_plain):.4f} -> "
                f"{pooled_r2(pc_corr):.4f}   per-chord median "
                f"{chord_medians(pc_plain):.4f} -> "
                f"{chord_medians(pc_corr):.4f}"
            )
        print(
            "  (large jump => constant per-shot offsets dominate: "
            "calibration-era zero points; no jump => sigma honesty or "
            "genuine model floor)"
        )
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["chord"]
                + [f"r2_{t}" for t in machine.targets]
                + [f"rmse_{t}" for t in machine.targets]
                + [f"medae_{t}" for t in machine.targets]
            )
            for c in range(C):
                w.writerow(
                    [c]
                    + [f"{per_chord[c, t]:.4f}" for t in range(n_t)]
                    + [f"{per_chord_rmse[c, t]:.5g}" for t in range(n_t)]
                    + [f"{per_chord_medae[c, t]:.5g}" for t in range(n_t)]
                )
        print(f"\nper-chord table -> {args.out}")


if __name__ == "__main__":
    main()
