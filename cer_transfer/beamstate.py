"""Beam state from the spectra of the two NSTX CHERS arrays.

The viewed beam's charge-exchange emission appears only on the
foreground (active) array. A frame is beam-off when the ratio of
foreground to background line brightness collapses. A beam-off phase
is VERIFIED only if every transition bounding it is a clean beam
switch: the foreground brightness steps (> step_fg x its typical
frame-to-frame step, and > dominance x the background step) while the
background brightness stays within noise (< step_bg x its typical
step). Transitions where both arrays jump together are
plasma events; transitions where only the background array jumps are
not beam-related. Both are rejected.
"""
from __future__ import annotations

import numpy as np
from joblib import load

PRE = 47  # recording pre-trigger frames (235 ms): pre-plasma


def valid_end(d):
    """Number of valid frames. end_index <= 0 (e.g. the -1 sentinel of
    requested shots) or beyond the array means: the whole recording."""
    n = int(d["end_index"])
    T = int(d["input"].shape[1])
    return T if (n <= 0 or n > T) else n


def _brightness(fp):
    d = load(fp, mmap_mode="r")
    end = valid_end(d)
    tot = np.asarray(d["input"][:, :end, :], np.float32).sum(axis=(0, 2))
    pre = min(PRE, end - 1)
    dw = tot[:max(8, pre - 2)]
    dk = np.median(dw)
    nz = 1.4826 * np.median(np.abs(dw - dk))
    lab = np.isfinite(np.asarray(d["target"][:, :end, 0])).any(axis=0)
    return tot - dk, max(nz, 1e-6 * abs(dk) + 1e-6), lab


def _runs(mask):
    out, f = [], 0
    while f < len(mask):
        if mask[f]:
            g = f
            while f < len(mask) and mask[f]:
                f += 1
            out.append((g, f - 1))
        else:
            f += 1
    return out


def beam_state(bg_file, fg_file, step_fg=2.0, step_bg=4.0, dominance=3.0,
               min_len=2):
    """Returns dict with T, active, lab, off_raw, off (verified), thr,
    R_on, R_low, and segments [(a, b, cls, verified, why)]."""
    bb, nzb, lab = _brightness(bg_file)
    bf, _, _ = _brightness(fg_file)
    T = min(len(bb), len(bf))
    bb, bf, lab = bb[:T], bf[:T], lab[:T]
    active = bb > 8 * nzb
    active[:min(PRE, T)] = False
    out = dict(T=T, active=active, lab=lab, thr=np.nan, R_on=np.nan,
               R_low=np.nan, off_raw=np.zeros(T, bool),
               off=np.zeros(T, bool), segments=[])
    if not (active & lab).any():
        return out
    with np.errstate(all="ignore"):
        R = np.where(active, bf / np.maximum(bb, 1e-12), np.nan)
        R_on = np.nanmedian(R[active & lab])
        R_low = np.nanpercentile(R[active], 5)
    out.update(R_on=R_on, R_low=R_low)
    if not (R_low < 0.7 * R_on) or not (R_low > 0) or not (R_on > 0):
        return out
    thr = np.sqrt(R_on * R_low)
    with np.errstate(all="ignore"):
        off_raw = active & (R < thr) & ~lab
    out.update(thr=thr, off_raw=off_raw)

    with np.errstate(all="ignore"):
        lb = np.log(np.maximum(bb, 1e-12))
        lf = np.log(np.maximum(bf, 1e-12))
    pair = active[:-1] & active[1:]
    same = pair & (off_raw[:-1] == off_raw[1:])
    tb = np.median(np.abs(np.diff(lb))[same]) if same.any() else np.nan
    tf = np.median(np.abs(np.diff(lf))[same]) if same.any() else np.nan

    def clean(i):  # transition between frames i and i+1
        if not (0 <= i < T - 1 and active[i] and active[i + 1]):
            return None
        sf = abs(lf[i + 1] - lf[i])
        sb = abs(lb[i + 1] - lb[i])
        # background: below step_bg x its median step (4x: ~0.7% false
        # rejection from noise alone; 2x rejected ~18% per boundary);
        # foreground: a real step that dominates the background change
        return bool(sf > step_fg * tf and sb < step_bg * tb
                    and sf > dominance * sb)

    # non-circular test of "the background array does not see the beam":
    # over ALL transitions at which the foreground brightness steps
    # (beam switches or plasma events), how often does the background
    # brightness stay flat?
    sf_all = np.abs(np.diff(lf))
    sb_all = np.abs(np.diff(lb))
    fgstep = pair & (sf_all > step_fg * tf)
    out["n_fg_steps"] = int(fgstep.sum())
    out["n_fg_steps_bg_flat"] = int((fgstep & (sb_all < step_bg * tb)).sum())
    lo = int(np.where(lab)[0].min())
    hi = int(np.where(lab)[0].max())
    off = np.zeros(T, bool)
    for a, b in _runs(off_raw):
        if b - a + 1 < min_len:
            continue
        cls = "head" if b < lo else ("tail" if a > hi else "notch")
        edges = [clean(a - 1), clean(b)]
        known = [e for e in edges if e is not None]
        ok = bool(known) and all(known)
        why = ("clean beam switch" if ok else
               "no bounding transition" if not known else
               "not a beam switch")
        out["segments"].append((a, b, cls, ok, why))
        if ok:
            off[a:b + 1] = True
    out["off"] = off
    return out
