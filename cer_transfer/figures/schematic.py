"""Fig. 1 draft: (a) viewing geometry — active vs passive arrays;
(b) transferable architecture. Panel (c), the instrument comparison,
comes from compare_arrays output at assembly time.

    pixi run python -u -m cer_transfer.figures.schematic --out figs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, FancyArrow, FancyBboxPatch

sns.set_style("white")

C_ACT, C_PAS, C_BEAM = "C0", "C1", "0.45"


def panel_geometry(ax):
    """Draw the tokamak cross-section with beam and lines of sight."""
    R0, a = 1.0, 0.62
    for r, lw, col in ((R0 + a, 1.5, "0.2"), (R0 - a, 1.5, "0.2")):
        ax.add_patch(Circle((0, 0), r, fill=False, lw=lw, color=col))
    ax.add_patch(Circle((0, 0), R0 + a, fill=True, color="0.96", zorder=0))
    ax.add_patch(Circle((0, 0), R0 - a, fill=True, color="white", zorder=1))
    # neutral beam: tangential chord through the plasma
    b0 = np.array([2.05, -0.72])
    bdir = np.array([-0.92, 0.39])
    bdir /= np.linalg.norm(bdir)
    ax.add_patch(
        FancyArrow(
            *b0,
            *(2.9 * bdir),
            width=0.10,
            length_includes_head=True,
            head_width=0.19,
            head_length=0.16,
            color=C_BEAM,
            alpha=0.55,
            zorder=2,
        )
    )
    ax.text(1.78, -0.98, "neutral\nbeam", ha="center", fontsize=7, color="0.3")
    # active fan: from a port, chords crossing the beam path
    pa = np.array([1.62, 1.30])
    for rt in np.linspace(0.18, 0.95, 6):
        d = np.array([-pa[0], rt - pa[1]])
        d /= np.linalg.norm(d)
        ax.plot(
            [pa[0], pa[0] + 3.1 * d[0]],
            [pa[1], pa[1] + 3.1 * d[1]],
            color=C_ACT,
            lw=1.0,
            alpha=0.85,
            zorder=3,
        )
    ax.plot(*pa, "s", color=C_ACT, ms=6, zorder=4)
    ax.text(
        pa[0] + 0.06,
        pa[1] + 0.10,
        "active array\n(crosses beam)",
        fontsize=7,
        color=C_ACT,
    )
    # passive fan: same tangency radii, geometry missing the beam
    pp = np.array([-1.62, 1.30])
    for rt in np.linspace(0.18, 0.95, 6):
        d = np.array([-pp[0], rt - pp[1]])
        d /= np.linalg.norm(d)
        ax.plot(
            [pp[0], pp[0] + 3.1 * d[0]],
            [pp[1], pp[1] + 3.1 * d[1]],
            color=C_PAS,
            lw=1.0,
            alpha=0.85,
            zorder=3,
        )
    ax.plot(*pp, "s", color=C_PAS, ms=6, zorder=4)
    ax.text(
        pp[0] - 0.06,
        pp[1] + 0.10,
        "passive array\n(misses beam)",
        fontsize=7,
        color=C_PAS,
        ha="right",
    )
    # one shared tangency circle to make "same flux surfaces" visible
    ax.add_patch(
        Circle((0, 0), 0.55, fill=False, ls="--", lw=0.9, color="0.5", zorder=2)
    )
    ax.text(0, -0.42, "same flux\nsurfaces", ha="center", fontsize=7, color="0.4")
    ax.set_xlim(-2.35, 2.35)
    ax.set_ylim(-1.95, 2.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("a  Viewing geometry (top view)", loc="left", fontsize=9)


def _box(ax, x, y, w, h, text, fc, fontsize=7):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec="0.3", lw=0.8
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def panel_architecture(ax):
    """Full pipeline: D3D pretrain -> transfer -> NSTX active
    (validated) -> readout transfer -> NSTX passive (beam-free)."""
    Y1, Y2, Y3 = 0.74, 0.42, 0.10  # rows: D3D, NSTX active, NSTX passive
    BH = 0.20

    # --- stage boxes ---
    _box(
        ax,
        0.00,
        Y1,
        0.20,
        BH,
        "DIII-D\nactive CER\n(13k discharges)",
        (0.85, 0.90, 1.0),
    )
    _box(ax, 0.30, Y1, 0.22, BH, "model\n(pretrained)", (1.0, 0.94, 0.80))
    _box(ax, 0.00, Y2, 0.20, BH, "NSTX CHERS\nactive array", (0.85, 1.0, 0.87))
    _box(
        ax,
        0.30,
        Y2,
        0.22,
        BH,
        "model\n(fine-tuned,\n25 - 7k discharges)",
        (1.0, 0.94, 0.80),
    )
    _box(
        ax,
        0.62,
        Y2,
        0.36,
        BH,
        "$T_i$, $v_{tor}$ profiles\n"
        "validated against fits\n(median 37 eV, 5.6 km/s)",
        (0.92, 0.92, 0.92),
    )
    _box(
        ax,
        0.00,
        Y3,
        0.20,
        BH,
        "NSTX CHERS\npassive array\n(no beam" " crossing)",
        (1.0, 0.90, 0.80),
    )
    _box(ax, 0.30, Y3, 0.22, BH, "model\n(array-adapted)", (1.0, 0.94, 0.80))
    _box(
        ax,
        0.62,
        Y3,
        0.36,
        BH,
        "$T_i$, $v_{tor}$ profiles\n"
        "without neutral beam\n(validated frame-synchronously)",
        (0.92, 0.92, 0.92),
    )

    # --- horizontal arrows: spectra -> model -> profiles ---
    for y in (Y1, Y2, Y3):
        ax.annotate(
            "",
            xy=(0.30, y + BH / 2),
            xytext=(0.20, y + BH / 2),
            arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2),
        )
    for y in (Y2, Y3):
        ax.annotate(
            "",
            xy=(0.62, y + BH / 2),
            xytext=(0.52, y + BH / 2),
            arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2),
        )

    # --- transfer arrows (red): D3D->NSTX, active->passive ---
    ax.annotate(
        "",
        xy=(0.41, Y2 + BH),
        xytext=(0.41, Y1),
        arrowprops=dict(arrowstyle="->", color="C3", lw=2.0),
    )
    ax.text(
        0.435,
        (Y1 + Y2 + BH) / 2,
        "transfer\n(cross-machine)",
        color="C3",
        fontsize=7,
        va="center",
    )
    ax.annotate(
        "",
        xy=(0.41, Y3 + BH),
        xytext=(0.41, Y2),
        arrowprops=dict(arrowstyle="->", color="C3", lw=2.0),
    )
    ax.text(
        0.435,
        (Y2 + Y3 + BH) / 2,
        "transfer\n(cross-array)",
        color="C3",
        fontsize=7,
        va="center",
    )

    # --- supervision annotations ---
    ax.text(
        0.10,
        Y1 + BH + 0.030,
        "labels: CERAUTO fits",
        fontsize=6.5,
        color="0.35",
        ha="center",
    )
    ax.text(
        0.10,
        Y2 + BH + 0.030,
        "labels: CHERS fits",
        fontsize=6.5,
        color="0.35",
        ha="center",
    )
    ax.text(
        0.10,
        Y3 - 0.045,
        "same labels (synchronous frames);\n" "deployment: no beam required",
        fontsize=6.5,
        color="0.35",
        ha="center",
    )

    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.axis("off")
    ax.set_title(
        "b  Transfer pipeline: cross-machine, cross-array," " beam-free",
        loc="left",
        fontsize=9,
    )


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("figs"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for ctx, ext, size in (("talk", "png", (11, 4.6)), ("paper", "pdf", (7.2, 3.0))):
        with sns.plotting_context(ctx):
            fig, axes = plt.subplots(
                1, 2, figsize=size, gridspec_kw={"width_ratios": [1, 1.5]}
            )
            panel_geometry(axes[0])
            panel_architecture(axes[1])
            fig.tight_layout()
            fig.savefig(
                args.out / f"fig1_schematic.{ext}", dpi=300, bbox_inches="tight"
            )
            plt.close(fig)
    print(f"wrote {args.out}/fig1_schematic.png/.pdf")


if __name__ == "__main__":
    main()
