"""Measured time axis of a discharge file.

Regenerated NSTX files carry ``time_s`` (one value per frame, shared by
spectra, targets and NBI power); older files do not, and frames are then
placed at frame / fs + t_offset.
"""

from __future__ import annotations

import numpy as np


def file_times(source, T, fs=200.0, t_offset=0.0):
    """Time (s) of frames 0..T-1 of ``source`` (file path, list entry or loaded dict)."""
    dd = source
    if dd is not None and not hasattr(dd, "keys"):
        from joblib import load

        from cer_transfer.configs import data_path

        dd = load(data_path(source), mmap_mode="r")
    if dd is not None and "time_s" in dd:
        ts = np.asarray(dd["time_s"], np.float64)
        if ts.size >= T:
            return ts[:T]
    return np.arange(T) / fs + t_offset
