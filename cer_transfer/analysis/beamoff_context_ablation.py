"""Temporal-context ablation for beam-off predictions of a causal model.

For each discharge, the longest beam-off phase (a, b) is found exactly as for the
beam-off figure (``figures.beamoff_examples.off_phases``). The beam-on frames
directly before the gap, [a - M, a), are replaced by beam-off spectra (the
per-chord median spectrum of the gap), and the predictions inside the gap are
compared with those on the unchanged discharge. A causal model sees M <= its
receptive field directly; effects beyond that can only enter through the
running offset of the preprocessing. Frames after the gap are irrelevant for a
causal model and are not masked.

Also reported: the unchanged prediction in the gap versus the conventional fits
just before and after it (median over ``--context-ms`` on each side).

    pixi run --frozen python -u -m cer_transfer.analysis.beamoff_context_ablation \
        --checkpoint cer_ckpts/nstx_rt_init_r5200_s1.pt --shots 142239 135660 \
        --list splits/nstx_set5.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from joblib import load

from cer_transfer.configs import data_path
from cer_transfer.data import Preprocessor
from cer_transfer.figures.beamoff_examples import off_phases
from cer_transfer.inference import load_model
from cer_transfer.training import denormalize_mu_sigma

RF_FRAMES = 38  # temporal receptive field of the causal model (frames)


@torch.no_grad()
def predict(m, x):
    """Model predictions (C, T, 2) in eV, km/s for raw spectra x (C, T, W)."""
    pre = Preprocessor(m.machine, m.config.resample_w, causal=m.config.causal)
    mu, sg = m.model(pre(x).unsqueeze(0).to(m.device), None)
    mu, _ = denormalize_mu_sigma(mu, sg, m.stats, m.machine.targets)
    return mu[0].cpu().numpy()


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    ap.add_argument("--shots", nargs="+", required=True)
    ap.add_argument("--list", type=Path, nargs="+", default=[Path("splits/nstx_set5.txt")])
    ap.add_argument("--masks", type=int, nargs="+", default=[0, 10, 20, 38, 60, 100],
                    help="numbers of beam-on frames before the gap to replace")
    ap.add_argument("--min-off-ms", type=float, default=50.0)
    ap.add_argument("--fs", type=float, default=200.0, help="frame rate of the files (Hz)")
    ap.add_argument("--context-ms", type=float, default=50.0)
    a = ap.parse_args()

    entries = [ln.strip() for lst in a.list for ln in lst.read_text().splitlines()
               if ln.strip() and not ln.strip().startswith("#")]
    shots = {s: next(e for e in entries if s in e) for s in a.shots}
    min_frames = int(round(a.min_off_ms / 1000 * a.fs))
    n_ctx = int(round(a.context_ms / 1000 * a.fs))
    for s, e in shots.items():
        ph, T = off_phases(e, min_frames, 1)
        if not ph:
            print(f"{s}: no beam-off phase >= {a.min_off_ms} ms"); continue
        ga, gb = max(ph, key=lambda ab: ab[1] - ab[0])
        d = load(data_path(e))
        x = np.asarray(d["input"][:, :T], np.float32)
        y = np.asarray(d["target"][:, :T], np.float32) * np.array([1000.0, 1.0], np.float32)
        print(f"\n=== {s} ({e}): beam-off frames {ga}-{gb} ({(gb - ga + 1) / a.fs * 1000:.0f} ms); "
              f"receptive field {RF_FRAMES} frames = {RF_FRAMES / a.fs * 1000:.0f} ms")
        sub = np.median(x[:, ga:gb + 1], axis=1, keepdims=True)    # per-chord beam-off spectrum
        before = y[:, max(ga - n_ctx, 0):ga]; after = y[:, gb + 1:gb + 1 + n_ctx]
        fit_b, fit_a = np.nanmedian(before, 1), np.nanmedian(after, 1)   # (C, 2)
        early = slice(ga, min(ga + RF_FRAMES, gb + 1)); late = slice(min(ga + RF_FRAMES, gb + 1), gb + 1)
        for ck in a.checkpoint:
            m = load_model(ck, machine="nstx")
            p0 = predict(m, x)
            gap = np.nanmean(p0[:, ga:gb + 1], 1)                       # (C, 2)
            ok = np.isfinite(fit_b).all(1) & np.isfinite(fit_a).all(1)
            mid = 0.5 * (fit_b + fit_a)
            print(f"  {ck.name}: gap mean vs mean of fits before/after ({ok.sum()} chords with both): "
                  f"|dT_i| {np.mean(np.abs(gap[ok, 0] - mid[ok, 0])):.0f} eV, "
                  f"|dv_tor| {np.mean(np.abs(gap[ok, 1] - mid[ok, 1])):.1f} km/s "
                  f"(fits before/after differ by {np.mean(np.abs(fit_b[ok, 0] - fit_a[ok, 0])):.0f} eV, "
                  f"{np.mean(np.abs(fit_b[ok, 1] - fit_a[ok, 1])):.1f} km/s)")
            print(f"    {'masked frames':>14} {'ms':>5} | mean |change| in gap: first {RF_FRAMES * 1000 / a.fs:.0f} ms"
                  f"  T_i / v_tor   | rest of gap  T_i / v_tor")
            for M in a.masks:
                xm = x.copy()
                lo = max(ga - M, 0)
                xm[:, lo:ga] = sub
                pm = predict(m, xm)
                dE = np.abs(pm[:, early] - p0[:, early]).mean((0, 1))
                dL = np.abs(pm[:, late] - p0[:, late]).mean((0, 1)) if late.stop > late.start else (np.nan, np.nan)
                print(f"    {M:>14d} {M * 1000 / a.fs:5.0f} | {dE[0]:8.1f} eV / {dE[1]:5.2f} km/s"
                      f"   | {dL[0]:8.1f} eV / {dL[1]:5.2f} km/s")


if __name__ == "__main__":
    main()
