"""Temporal receptive field of a checkpoint: analytic (from its model
config) and empirical (gradient of one output frame w.r.t. all input
frames).

Analytic: every temporal operation is a centred convolution (kernel k_t,
stride 1, no dilation) or a 3-wide max-pool (stride 1), so the local
receptive field is symmetric: (RF - 1) / 2 frames past and future.

Empirical: GroupNorm statistics span the whole time axis, so every output
depends weakly on ALL input frames. The script therefore reports the
gradient magnitude by distance from the output frame: exactly zero
outside the local field only if the model has no GroupNorm.

    pixi run python -u -m cer_transfer.analysis.receptive_field --checkpoint cer_ckpts/nstx_ft.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from cer_transfer.configs import ModelConfig, get_machine
from cer_transfer.models import build_model
from cer_transfer.training import load_checkpoint


def temporal_k(ks):
    """Temporal kernel size of an int or (time, wavelength) kernel spec."""
    return ks[0] if isinstance(ks, (tuple, list)) else int(ks)


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--frames", type=int, default=301, help="length of the test window (odd)"
    )
    p.add_argument(
        "--frame-ms",
        type=float,
        default=5.0,
        help="frame interval for the ms conversion (NSTX: 5)",
    )
    args = p.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    mk = ckpt["machine"]
    machine = get_machine(mk["name"] if isinstance(mk, dict) else mk)
    cfg = ModelConfig(**ckpt["model_config"])

    # --- analytic ---------------------------------------------------------
    k = temporal_k(cfg.kernel_size)
    n_blocks = len(cfg.encoder_widths) + 1
    per_block = 2 * (k - 1) + 2  # two convs + 3-wide max-pool
    head_convs = 2  # head residual block (all head types)
    rf = 1 + n_blocks * per_block + (k - 1) + head_convs * (k - 1)
    half = (rf - 1) // 2
    print(
        f"checkpoint {args.checkpoint.name}: machine {machine.name}, "
        f"kernel {cfg.kernel_size}, encoder_widths {cfg.encoder_widths}, "
        f"head {cfg.head_type}, norm {cfg.norm}"
    )
    print(
        f"analytic local receptive field: {rf} frames = {half} past + "
        f"{half} future (+/-{half * args.frame_ms:g} ms at "
        f"{args.frame_ms:g} ms/frame)"
    )

    # --- empirical ----------------------------------------------------------
    model = build_model(machine, cfg).eval()
    model.load_state_dict(ckpt["model_state"])
    pools = model.backbone.n_pools
    W = cfg.hidden_dim << pools
    T = args.frames | 1
    x = torch.randn(1, machine.n_input_channels, T, W, requires_grad=True)
    mu, _ = model(x)
    c = T // 2
    mu[:, :, c, :].sum().backward()
    g = x.grad.abs().sum(dim=(0, 1, 3)).numpy()
    g = g / g.max()
    d = np.arange(T) - c
    for thr in (1e-3, 1e-6):
        nz = d[g > thr]
        print(
            f"empirical, |grad| > {thr:g} of max: frames {nz.min()} .. "
            f"{nz.max()} relative to the output frame"
        )
    exact = d[g > 0]
    print(
        f"empirical, any nonzero gradient: {exact.min()} .. {exact.max()}"
        + (
            "  (whole window: GroupNorm statistics span the time axis)"
            if exact.min() == d[0] and exact.max() == d[-1]
            else ""
        )
    )
    inside = g[np.abs(d) <= half].sum() / g.sum()
    print(
        f"share of the total gradient inside the local field: " f"{100 * inside:.2f}%"
    )


if __name__ == "__main__":
    main()
