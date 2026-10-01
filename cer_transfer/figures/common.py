"""Shared helpers for the figure and analysis modules.

Prediction dumps are written by ``eval_checkpoint.py --dump-preds``: flat
per-point arrays ``chord, y, pred, sigma, pred_sigma`` (and ``targets``),
one discharge per file, frames in recording order. ``load_predictions``
reshapes them to ``(chords, frames, targets)``.

Frame timing: NSTX CHERS records at 200 Hz; recordings start 235 ms before
the experimental clock zero, so experimental time is
``frame / fs + t_offset`` with ``t_offset = -0.235``. Both are plot options
(``--fs``, ``--t-offset``) rather than constants, so other machines can use
their own values.
"""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FRAME_HZ = 200.0          # NSTX CHERS frame rate (default for --fs)
NSTX_T_OFFSET = -0.235    # experimental time of frame 0 on NSTX (s)


def load_predictions(path) -> SimpleNamespace:
    """Load a single-discharge prediction dump as (C, T, K) arrays."""
    d = np.load(path)
    chord = d["chord"]
    C = int(chord.max()) + 1
    T = len(chord) // C
    out = SimpleNamespace(C=C, T=T, targets=list(d["targets"]) if "targets" in d else None)
    for key in ("y", "pred", "sigma", "pred_sigma"):
        setattr(out, key, d[key].reshape(C, T, -1) if key in d else None)
    return out


def frame_times(T: int, fs: float = FRAME_HZ, t_offset: float = 0.0) -> np.ndarray:
    return np.arange(T) / fs + t_offset


def frame_of(t: float, fs: float = FRAME_HZ, t_offset: float = 0.0) -> int:
    return int(round((t - t_offset) * fs))


def chord_coordinate(C: int, csv_path=None, label: str | None = None):
    """Per-chord plotting coordinate.

    With a ``chord,x`` CSV (e.g. tangency radius or psi_N), returns its
    values (NaN for chords not listed) and the given label; without one,
    the chord index and the label 'chord (core -> edge)'.
    """
    if csv_path is None:
        return np.arange(C, dtype=float), "chord (core -> edge)"
    x = np.full(C, np.nan)
    for r in csv.DictReader(open(csv_path)):
        i = int(r["chord"])
        if 0 <= i < C:
            x[i] = float(r["x"])
    return x, (label or "x")


def sorted_chords(x: np.ndarray):
    """Chord indices in increasing coordinate order (NaN dropped), and xs."""
    order = [i for i in np.argsort(x) if np.isfinite(x[i])]
    return order, x[order]


def runs(mask: np.ndarray):
    """Contiguous True runs of a boolean array as (first, last) indices."""
    out, a = [], None
    for i, flag in enumerate(list(mask) + [False]):
        if flag and a is None:
            a = i
        elif not flag and a is not None:
            out.append((a, i - 1))
            a = None
    return out


def coordinate_spans(xs: np.ndarray, mask: np.ndarray):
    """Extent in coordinate units of each True run of ``mask`` over the
    sorted coordinates ``xs``: half a step beyond the run's end points."""
    dx = np.diff(xs)
    step = float(np.median(dx)) if dx.size else 0.5
    spans = []
    for a, b in runs(mask):
        lo = xs[a] - 0.5 * ((xs[a] - xs[a - 1]) if a > 0 else step)
        hi = xs[b] + 0.5 * ((xs[b + 1] - xs[b]) if b + 1 < len(xs) else step)
        spans.append((lo, hi))
    return spans


def fitted_frames(y: np.ndarray, f0: int, f1: int, min_fits: int = 5):
    """Frames in [f0, f1] with at least ``min_fits`` finite labels (y: C x T)."""
    return [f for f in range(max(f0, 0), min(f1, y.shape[1] - 1) + 1)
            if np.isfinite(y[:, f]).sum() >= min_fits]


def write_coordinate_csv(path, x: np.ndarray):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["chord", "x"])
        for i, v in enumerate(x):
            if np.isfinite(v):
                w.writerow([i, f"{v:.5f}"])
    return Path(path)
