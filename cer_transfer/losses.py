"""Masked losses and metrics.

MaskedR2Score updates on valid entries only — no fill values (the old
fill-with-nanmean approach injected zero-error points and inflated R2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torcheval.metrics import MeanSquaredError, R2Score


class MaskedGaussianNLL(nn.Module):
    """Gaussian NLL ignoring NaN targets."""

    def __init__(self, min_sigma: float = 1e-3):
        super().__init__()
        self.min_sigma = min_sigma

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
        label_sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """label_sigma (same normalized units as target), if given, enters as
        total variance = sigma^2 + label_sigma^2: the model's sigma then
        estimates the RESIDUAL uncertainty, and fitting label noise is no
        longer rewarded. Invalid label sigmas (NaN/<=0) contribute 0."""
        mask = ~torch.isnan(target)
        if not mask.any():
            return mu.new_tensor(0.0, requires_grad=True)
        mu, sigma, target = mu[mask], sigma[mask], target[mask]
        sigma = torch.clamp(sigma, min=self.min_sigma)
        var = sigma**2
        if label_sigma is not None:
            ls = label_sigma[mask]
            ls = torch.where(torch.isfinite(ls) & (ls > 0), ls, torch.zeros_like(ls))
            var = var + ls**2
        return torch.mean(
            0.5 * torch.log(2 * torch.pi * var) + 0.5 * (target - mu) ** 2 / var
        )


class MaskedR2Score(R2Score):
    """R2 score that ignores NaN targets (torcheval ``R2Score`` with masking)."""

    def update(self, input: torch.Tensor, target: torch.Tensor):
        """Accumulate a batch, dropping entries with NaN targets.

        Parameters
        ----------
        input, target : Tensor
            Predictions and labels of the same shape.
        """
        mask = ~torch.isnan(target)
        if mask.any():
            super().update(input[mask].flatten(), target[mask].flatten())
        return self


class MaskedRMSE:
    """Root-mean-square error accumulated over batches, ignoring NaN targets."""

    def __init__(self):
        self._mse = MeanSquaredError()

    def reset(self):
        """Clear the accumulated sums."""
        self._mse.reset()
        return self

    def update(self, input: torch.Tensor, target: torch.Tensor):
        """Accumulate a batch, dropping entries with NaN targets.

        Parameters
        ----------
        input, target : Tensor
            Predictions and labels of the same shape.
        """
        mask = ~torch.isnan(target)
        if mask.any():
            self._mse.update(input[mask], target[mask])
        return self

    def compute(self) -> torch.Tensor:
        """Return the RMSE over everything accumulated since the last reset."""
        return torch.sqrt(self._mse.compute())

    def to(self, device):
        """Move the accumulators to a device and return ``self``."""
        self._mse = self._mse.to(device)
        return self


def combined_score(per_target_r2: dict[str, float]) -> float:
    """Model-selection scalar from per-target R2 values.

    Mean, not sqrt(product): the geometric mean is NaN whenever exactly one
    R2 is negative (guaranteed early in training) and would checkpoint/stop
    on garbage. Use min(...) instead if you want to force both targets good.
    """
    vals = list(per_target_r2.values())
    return float(sum(vals) / len(vals))
