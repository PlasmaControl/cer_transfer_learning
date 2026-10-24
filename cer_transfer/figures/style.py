r"""Conventions shared by every paper figure.

Colours (fixed everywhere): target (conventional fit) = black points with error bars;
DIII-D model (zero-shot) = green; NSTX model (from scratch) = blue;
DIII-D+NSTX model (fine-tuned) = red; NSTX-U models: NSTX-U model (from scratch) =
brown, DIII-D model fine-tuned on NSTX-U = pink, DIII-D+NSTX model fine-tuned on NSTX-U
= dark gold; beam-off phase = grey band. Model
uncertainty = band of the model's colour, 20 % alpha, +-1 sigma.
Legend order always: fit, zero-shot, scratch, fine-tuned.
Units in the axis labels; no titles inside panels; panel letters a, b, c, ...
"""

from __future__ import annotations

import string

import numpy as np

COLORS = {
    "fit": "black",
    "zero_shot": "#2ca02c",   # green
    "scratch": "#1f77b4",     # blue
    "finetuned": "#d62728",   # red (distinct from the green zero-shot model)
    "nstxu_scratch": "#8c564b",       # brown: NSTX-U model (from scratch on NSTX-U)
    "nstxu_ft_d3d": "#e377c2",        # pink: DIII-D model fine-tuned on NSTX-U
    "nstxu_ft_d3d_nstx": "#b8860b",   # dark gold: DIII-D+NSTX model fine-tuned on NSTX-U
}
BAND_ALPHA = 0.20
BEAMOFF_COLOR = "0.85"       # grey band of a beam-off phase
LEGEND_ORDER = ("fit", "zero_shot", "scratch", "finetuned")
NAMES = {
    "fit": "target",  # the conventional fit, the training target
    "zero_shot": "DIII-D model",
    "scratch": "NSTX model",
    "finetuned": "DIII-D+NSTX model",
}
DISCHARGES_PER_DAY = 25      # second x-axis of the scaling figure

TI_LABEL = r"$T_\mathrm{i}$ (keV)"
VTOR_LABEL = r"$v_\mathrm{tor}$ (km/s)"
TIME_LABEL = "time (s)"
RTAN_LABEL = r"$R_\mathrm{tan}$ (m)"
TARGET_LABEL = {"ti": TI_LABEL, "vtor": VTOR_LABEL}
TARGET_SYMBOL = {"ti": r"$T_\mathrm{i}$", "vtor": r"$v_\mathrm{tor}$"}
TARGET_SCALE = {"ti": 1e-3, "vtor": 1.0}  # canonical (eV, km/s) -> displayed (keV, km/s)
TARGET_UNIT = {"ti": "keV", "vtor": "km/s"}


def target_rec_labels(name):
    """Axis labels of a reconstruction-vs-target plot: ('$T_\\mathrm{i}^\\mathrm{target}$ (keV)',
    '$T_\\mathrm{i}^\\mathrm{rec}$ (keV)'); rec = reconstruction."""
    sym = TARGET_SYMBOL[name][:-1]  # without the closing $
    unit = TARGET_UNIT[name]
    return sym + r"^\mathrm{target}$ (" + unit + ")", sym + r"^\mathrm{rec}$ (" + unit + ")"


def role_of(label: str | None) -> str | None:
    """Map a legend text to its role (fit, zero_shot, scratch, finetuned), else None."""
    if not label:
        return None
    s = label.lower()
    if "fit" in s:
        return "fit"
    if "diii-d+nstx" in s or "fine-tun" in s:
        return "finetuned"
    if s.startswith("nstx") or "scratch" in s:
        return "scratch"
    if "diii-d" in s or "zero-shot" in s:
        return "zero_shot"
    return None


def model_line(ax, t, mu, sigma=None, role="finetuned", label=None, marker=".", **kw):
    """Model trace in its role colour, with a +-1 sigma band (20 % alpha) if sigma is given."""
    c = COLORS[role]
    (h,) = ax.plot(t, mu, marker=marker, ms=kw.pop("ms", 3), lw=kw.pop("lw", 1.2), color=c,
                   label=label if label is not None else NAMES[role], **kw)
    if sigma is not None:
        ax.fill_between(t, mu - sigma, mu + sigma, color=c, alpha=BAND_ALPHA, lw=0)
    return h


def fit_points(ax, t, y, err=None, label=None, **kw):
    """Conventional fit: black points with error bars."""
    return ax.errorbar(t, y, yerr=err, fmt="o", ms=kw.pop("ms", 3), color=COLORS["fit"],
                       elinewidth=kw.pop("elinewidth", 0.8), capsize=0, zorder=5,
                       label=label if label is not None else NAMES["fit"], **kw)


def beamoff_band(ax, t0, t1, **kw):
    """Grey band of a beam-off phase."""
    return ax.axvspan(t0, t1, color=BEAMOFF_COLOR, lw=0, zorder=0, **kw)


def ordered_legend(ax, handles=None, labels=None, **kw):
    """Legend with entries in the fixed order fit, zero-shot, scratch, fine-tuned (others last)."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    elif labels is None:
        labels = [h.get_label() for h in handles]
    rank = {r: i for i, r in enumerate(LEGEND_ORDER)}
    order = sorted(range(len(handles)), key=lambda i: rank.get(role_of(labels[i]), len(rank)))
    return ax.legend([handles[i] for i in order], [labels[i] for i in order], **kw)


def panel_letters(axes, start="a", x=-0.08, y=1.02, **kw):
    """Bold panel letters a, b, c, ... at the top left of each axis (row-major order)."""
    axes = list(np.ravel(axes))
    i0 = string.ascii_lowercase.index(start)
    for i, ax in enumerate(axes):
        ax.text(x, y, string.ascii_lowercase[i0 + i], transform=ax.transAxes, fontweight="bold",
                va="bottom", ha="right", **kw)
