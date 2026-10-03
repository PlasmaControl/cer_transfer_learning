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

FRAME_HZ = 200.0  # NSTX CHERS frame rate (default for --fs)
NSTX_T_OFFSET = -0.235  # experimental time of frame 0 on NSTX (s)


def load_predictions(path) -> SimpleNamespace:
    """Load a single-discharge prediction dump.

    Parameters
    ----------
    path : str or Path
        ``.npz`` written by ``eval_checkpoint.py --dump-preds``.

    Returns
    -------
    SimpleNamespace
        ``C``, ``T``, and the arrays ``y``, ``pred``, ``sigma``,
        ``pred_sigma`` reshaped to (C, T, targets); ``targets`` if stored.
    """
    d = np.load(path)
    chord = d["chord"]
    C = int(chord.max()) + 1
    T = len(chord) // C
    out = SimpleNamespace(
        C=C, T=T, targets=list(d["targets"]) if "targets" in d else None
    )
    for key in ("y", "pred", "sigma", "pred_sigma"):
        setattr(out, key, d[key].reshape(C, T, -1) if key in d else None)
    return out


def frame_times(T: int, fs: float = FRAME_HZ, t_offset: float = 0.0) -> np.ndarray:
    """Experimental time of each frame.

    Parameters
    ----------
    T : int
        Number of frames.
    fs : float, default=200
        Frame rate (Hz).
    t_offset : float, default=0.0
        Experimental time of frame 0 (s).

    Returns
    -------
    ndarray of shape (T,)
    """
    return np.arange(T) / fs + t_offset


def frame_of(t: float, fs: float = FRAME_HZ, t_offset: float = 0.0) -> int:
    """Frame index closest to an experimental time.

    Parameters
    ----------
    t : float
        Time (s).
    fs : float, default=200
    t_offset : float, default=0.0

    Returns
    -------
    int
    """
    return int(round((t - t_offset) * fs))


def chord_coordinate(C: int, csv_path=None, label: str | None = None):
    """Per-chord plotting coordinate.

    Parameters
    ----------
    C : int
        Number of chords.
    csv_path : str or Path, optional
        ``chord,x`` CSV (e.g. tangency radius or normalized flux).
    label : str, optional
        Axis label for the CSV coordinate.

    Returns
    -------
    x : ndarray of shape (C,)
        Coordinate per chord (NaN for chords not listed), or the chord index
        without a CSV.
    label : str
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
    """Chord indices in increasing coordinate order.

    Parameters
    ----------
    x : ndarray of shape (C,)

    Returns
    -------
    order : list of int
        Chord indices with finite coordinate, sorted by it.
    xs : ndarray
        The corresponding coordinates.
    """
    order = [i for i in np.argsort(x) if np.isfinite(x[i])]
    return order, x[order]


def runs(mask: np.ndarray):
    """Contiguous runs of True in a boolean array.

    Parameters
    ----------
    mask : ndarray of bool

    Returns
    -------
    list of (first, last)
        Inclusive index ranges.
    """
    out, a = [], None
    for i, flag in enumerate(list(mask) + [False]):
        if flag and a is None:
            a = i
        elif not flag and a is not None:
            out.append((a, i - 1))
            a = None
    return out


def coordinate_spans(xs: np.ndarray, mask: np.ndarray):
    """Extent of each True run in coordinate units.

    Parameters
    ----------
    xs : ndarray
        Sorted coordinates.
    mask : ndarray of bool
        Same length as ``xs``.

    Returns
    -------
    list of (lo, hi)
        Half a step beyond the run's end points.
    """
    dx = np.diff(xs)
    step = float(np.median(dx)) if dx.size else 0.5
    spans = []
    for a, b in runs(mask):
        lo = xs[a] - 0.5 * ((xs[a] - xs[a - 1]) if a > 0 else step)
        hi = xs[b] + 0.5 * ((xs[b + 1] - xs[b]) if b + 1 < len(xs) else step)
        spans.append((lo, hi))
    return spans


def fitted_frames(y: np.ndarray, f0: int, f1: int, min_fits: int = 5):
    """Frames with enough conventional fits.

    Parameters
    ----------
    y : ndarray of shape (C, T)
        Labels (NaN where none exists).
    f0, f1 : int
        Inclusive frame range.
    min_fits : int, default=5

    Returns
    -------
    list of int
    """
    return [
        f
        for f in range(max(f0, 0), min(f1, y.shape[1] - 1) + 1)
        if np.isfinite(y[:, f]).sum() >= min_fits
    ]


def write_coordinate_csv(path, x: np.ndarray):
    """Write a ``chord,x`` coordinate file.

    Parameters
    ----------
    path : str or Path
    x : ndarray of shape (C,)
        NaN entries are skipped.

    Returns
    -------
    Path
    """
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["chord", "x"])
        for i, v in enumerate(x):
            if np.isfinite(v):
                w.writerow([i, f"{v:.5f}"])
    return Path(path)
