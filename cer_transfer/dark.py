"""Dark-frame criterion shared by training (zero_dark_frames) and the
confidence check. A frame is DARK only if:
  1. it lies BEFORE the labeled span (t < first_label - guard) by
     default; post-span frames are excluded because decaying ramp-down
     plasma is too dim to separate safely (pre_only=False restores
     both sides), and
  2. its mean line amplitude is below k x the shot's noise floor,
     where the floor is the median amplitude of the lowest-decile
     frames (photon-noise anchor, robust to bright transients).
Shots without any label yield NO dark frames (no anchor -> no claim).
"""

from __future__ import annotations

import numpy as np


def dark_mask(
    spec: np.ndarray,
    target: np.ndarray,
    guard: int = 60,
    k: float = 3.0,
    pre_only: bool = True,
) -> np.ndarray:
    """spec (C, T, W) raw counts; target (C, T, n_t) with NaN where
    unlabeled. Returns (T,) bool.

    pre_only (default): dark frames are claimed ONLY before the first
    label. Rationale: after the last beam blip the plasma decays with
    real, finite T_i at low emission (ramp-down) -- measured to sit
    within ~1.4x of the dark amplitude ceiling, too close to separate
    safely. Before the first blip, sub-threshold frames are
    breakdown-or-vacuum, where T_i = 0 holds within the pseudo-label
    sigma.

    guard (pre-side) defaults to 60 frames (300 ms at 200 Hz): the
    active array is nearly blind to the pre-beam OHMIC phase, so the
    amplitude test cannot exclude it -- the guard must span the typical
    breakdown-to-first-beam delay. Measured on the validation set, the
    quiet test alone runs up to the guard edge in >= half the shots,
    i.e. the guard, not the threshold, is the operative protection."""
    T = spec.shape[1]
    labeled_t = np.isfinite(target[..., 0]).any(axis=0)
    if not labeled_t.any():
        return np.zeros(T, bool)
    lab = np.flatnonzero(labeled_t)
    lo = max(int(lab.min()) - guard, 0)
    hi = min(int(lab.max()) + guard, T - 1)
    outside = np.zeros(T, bool)
    outside[:lo] = True
    if not pre_only:
        outside[hi + 1 :] = True

    base = np.median(spec, axis=2, keepdims=True)
    amp_t = np.clip(spec - base, 0, None).mean(axis=(0, 2))
    n_low = max(T // 10, 3)
    floor = float(np.median(np.sort(amp_t)[:n_low]))
    quiet = amp_t < k * max(floor, 1e-9)
    return outside & quiet
