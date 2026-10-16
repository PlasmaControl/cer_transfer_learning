"""Check that causal models and preprocessing use no future frames.

For every architecture variant, the predictions for frames < t0 must not
change when the input from frame t0 on is replaced by noise, and a
non-causal model must fail the same test (needs PyTorch; CPU, seconds).

    python -m tests.causality_check
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from cer_transfer.configs import ModelConfig, get_machine
from cer_transfer.data import Preprocessor
from cer_transfer.models import build_model

T, T0, W = 48, 30, 256

VARIANTS = {
    "legacy": dict(feature_width=None, encoder_widths=(16, 16)),
    "stem/trunk": dict(feature_width=32, encoder_widths=(32, 32, 32)),
    "stem/trunk, full head": dict(
        feature_width=32, encoder_widths=(32, 32, 32), head_type="full"
    ),
    "stem/trunk, batch norm": dict(
        feature_width=32, encoder_widths=(32, 32, 32), norm="batch"
    ),
    "agnostic, attention": dict(
        feature_width=32,
        encoder_widths=(32, 32, 32),
        agnostic=True,
        agnostic_stem_width=8,
        chord_attention=True,
        chord_attention_heads=2,
    ),
    "agnostic, transformer": dict(
        feature_width=32,
        encoder_widths=(32, 32, 32),
        agnostic=True,
        agnostic_stem_width=8,
        chord_attention=True,
        chord_attention_heads=2,
        chord_attention_depth=1,
    ),
}


def _leak(model, x, moments) -> float:
    """Largest change of a past prediction when the future input changes."""
    y = x.clone()
    y[:, :, T0:] = torch.randn_like(y[:, :, T0:]) * 3
    m2 = moments.clone()
    m2[:, :, T0:] = torch.randn_like(m2[:, :, T0:])
    with torch.no_grad():
        a = model(x, moments)
        b = model(y, m2)
    return max(float((p[:, :, :T0] - q[:, :, :T0]).abs().max()) for p, q in zip(a, b))


def check_models() -> None:
    """Every variant: causal leaks nothing, non-causal does."""
    torch.manual_seed(0)
    machine = get_machine("nstx")
    for name, kw in VARIANTS.items():
        base = ModelConfig(hidden_dim=8, dropout=0.0, moment_features=True, **kw)
        for causal in (True, False):
            cfg = dataclasses.replace(base, causal=causal)
            model = build_model(machine, cfg).eval()
            x = torch.randn(1, machine.n_input_channels, T, W)
            moments = torch.randn(1, machine.n_chords, T, 3)
            leak = _leak(model, x, moments)
            ok = leak == 0.0 if causal else leak > 0.0
            print(
                f"{'ok ' if ok else 'FAIL'} {name:24s} causal={causal!s:5s} "
                f"max past change {leak:.2e}"
            )
            assert ok, f"{name}, causal={causal}: past change {leak}"


def check_preprocessing() -> None:
    """Causal offset: frames < t0 independent of later frames."""
    rng = np.random.default_rng(0)
    machine = get_machine("nstx")
    spec = rng.uniform(50, 200, (machine.n_raw_channels, T, 83)).astype(np.float32)
    later = spec.copy()
    later[:, T0:] = rng.uniform(0, 1000, later[:, T0:].shape)
    for causal in (True, False):
        pre = Preprocessor(machine, resample_w=256, causal=causal)
        leak = float((pre(spec)[:, :T0] - pre(later)[:, :T0]).abs().max())
        ok = leak == 0.0 if causal else leak > 0.0
        print(f"{'ok ' if ok else 'FAIL'} preprocessing causal={causal!s:5s} {leak:.2e}")
        assert ok


if __name__ == "__main__":
    check_preprocessing()
    check_models()
    print("causality check passed")
