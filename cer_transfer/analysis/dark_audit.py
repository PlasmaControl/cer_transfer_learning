"""Audit the dark-frame criterion on real files BEFORE training with
--zero-dark-frames: amplitude separation between label-free candidate-dark
frames and labeled (plasma) frames, plus boundary-risk count.

    pixi run python -u -m cer_transfer.analysis.dark_audit --list splits/nstx_passive_train.txt \
        --sample 200
"""

import argparse
from pathlib import Path

import numpy as np
from joblib import load

from cer_transfer.configs import data_path

p = argparse.ArgumentParser()
p.add_argument("--list", type=Path, required=True)
p.add_argument("--sample", type=int, default=200)
p.add_argument("--thresh", type=float, default=0.05)
a = p.parse_args()

files = [Path(l) for l in a.list.read_text().splitlines() if l.strip()]
rng = np.random.default_rng(0)
files = [files[i] for i in rng.permutation(len(files))[: a.sample]]

r_dark, r_plasma, boundary = [], [], 0
for f in files:
    d = load(data_path(f), mmap_mode="r")
    end = int(d["end_index"])
    spec = np.asarray(d["input"][:, :end, :], dtype=np.float32)
    tgt = np.asarray(d["target"][:, :end, :], dtype=np.float32)
    base = np.median(spec, axis=2, keepdims=True)
    amp = np.clip(spec - base, 0, None).mean(axis=(0, 2))
    r = amp / max(float(amp.max()), 1e-6)
    lab = np.isfinite(tgt).any(axis=(0, 2))
    r_plasma.append(r[lab])
    r_dark.append(r[~lab])
    idx = np.flatnonzero(~lab & (r < a.thresh))
    if idx.size and lab.any():
        libx = np.flatnonzero(lab)
        boundary += int((np.abs(idx[:, None] - libx[None, :]).min(axis=1) <= 3).sum())

r_dark = np.concatenate(r_dark)
r_plasma = np.concatenate(r_plasma)
sel = r_dark < a.thresh
print(
    f"files: {len(files)} | label-free frames: {r_dark.size} "
    f"({sel.mean()*100:.1f}% below thresh={a.thresh}) | "
    f"labeled frames: {r_plasma.size}"
)
print(
    f"label-free frames: p50 {np.percentile(r_dark,50)*100:.2f}%  "
    f"p95 {np.percentile(r_dark,95)*100:.2f}% of shot max"
)
print(
    f"labeled (plasma) frames: p5 {np.percentile(r_plasma,5)*100:.1f}%  "
    f"p50 {np.percentile(r_plasma,50)*100:.1f}%"
)
print(
    f"would-be-dark frames within 3 frames of a label: {boundary} "
    f"(boundary risk; should be ~0)"
)
print(
    "VERDICT: safe if labeled-p5 >> thresh and boundary ~ 0; if "
    "label-free p95 crowds the threshold, dim beam-off phases exist "
    "and the threshold must drop (or the guard band widen)."
)
