"""Random convolutions along the wavelength axis (domain generalization).

Reimplementation for 1-D spectra of RandConv: Z. Xu, D. Liu, J. Yang,
C. Raffel, M. Niethammer, "Robust and Generalizable Visual Representation
Learning via Random Convolutions", ICLR 2021 (github.com/wildphoton/RandConv;
that repository has no license, so no code is copied, only the method).

As in RandConv:

* a convolution with random weights is applied to the training input and
  re-drawn every step (``randomize``); one kernel per step for the whole batch
* multi-scale: the kernel size is drawn from ``kernel_sizes`` (default 1,3,5,7)
* weights from a Kaiming-normal distribution
* either ``mixing`` (output = alpha * randconv(x) + (1 - alpha) * x, alpha ~ U(0,1)
  per step) or ``identity_prob`` (the input is passed unchanged with that
  probability)
* a consistency loss between the predictions for the clean input and two
  further random views (``consistency_loss``), which RandConv computes as the
  KL divergence of each class distribution to their mixture

Adaptation to spectra: the convolution runs along the wavelength axis only,
with one kernel shared by all chords and frames (the instrument function of a
spectrometer is common to its chords), so chords are never mixed. The
``instrument`` mode draws positive, normalized Gaussian kernels instead, i.e.
random physical instrument broadening. Because RandConv's KL is defined for
class probabilities, the consistency loss here uses the regression analogue:
the KL divergence of each predicted Gaussian to the moment-matched Gaussian of
the three views.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandConv1d(nn.Module):
    """Random convolution along the last (wavelength) axis of (B, C, T, W).

    Parameters
    ----------
    kernel_sizes : sequence of int, default=(1, 3, 5, 7)
        Kernel sizes drawn from at every ``randomize`` (multi-scale).
    mode : {'randconv', 'instrument'}, default='randconv'
        ``randconv``: Kaiming-normal weights as in RandConv. ``instrument``:
        positive normalized Gaussian kernel with random width.
    mixing : bool, default=True
        Blend the output with the input with a random weight alpha ~ U(0, 1).
    identity_prob : float, default=0.0
        Probability of passing the input unchanged.
    max_shift : int, default=0
        If > 0, also shift the spectrum by a random number of bins in
        [-max_shift, max_shift] (edge values replicated), all chords together.
    """

    def __init__(
        self,
        kernel_sizes=(1, 3, 5, 7),
        mode: str = "randconv",
        mixing: bool = True,
        identity_prob: float = 0.0,
        max_shift: int = 0,
    ):
        super().__init__()
        if mode not in ("randconv", "instrument"):
            raise ValueError(f"unknown mode {mode!r}")
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.mode = mode
        self.mixing = mixing
        self.identity_prob = float(identity_prob)
        self.max_shift = int(max_shift)
        self.register_buffer("weight", torch.ones(1, 1, 1))
        self.alpha = 1.0
        self.shift = 0
        self.identity = False
        self.randomize()

    @torch.no_grad()
    def randomize(self):
        """Draw a new kernel (size, weights), mixing weight and shift."""
        k = random.choice(self.kernel_sizes)
        if self.mode == "randconv":
            w = torch.empty(1, 1, k)
            # Kaiming normal for a 1-in-channel conv: std = gain / sqrt(fan_in)
            nn.init.normal_(w, 0.0, math.sqrt(2.0) / math.sqrt(k))
        else:
            sig = random.uniform(0.3, max(k / 2.0, 0.6))
            x = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
            w = torch.exp(-0.5 * (x / sig) ** 2)
            w = (w / w.sum()).reshape(1, 1, k)
        self.weight = w.to(self.weight.device)
        self.alpha = random.random() if self.mixing else 1.0
        self.shift = (
            random.randint(-self.max_shift, self.max_shift) if self.max_shift else 0
        )
        self.identity = self.identity_prob > 0 and random.random() < self.identity_prob

    def forward(self, x):
        """Apply the current random transform to (B, C, T, W) input."""
        if self.identity:
            return x
        b, c, t, w = x.shape
        k = self.weight.shape[-1]
        flat = x.reshape(b * c * t, 1, w)
        y = F.conv1d(F.pad(flat, (k // 2, k // 2), mode="replicate"), self.weight)
        y = y.reshape(b, c, t, w)
        if self.mixing:
            y = self.alpha * y + (1.0 - self.alpha) * x
        if self.shift:
            s = abs(self.shift)
            if self.shift > 0:
                y = F.pad(y[..., :-s].reshape(-1, 1, w - s), (s, 0), mode="replicate")
            else:
                y = F.pad(y[..., s:].reshape(-1, 1, w - s), (0, s), mode="replicate")
            y = y.reshape(b, c, t, w)
        return y


def gaussian_consistency_loss(mus, sigmas, mask=None):
    """Regression analogue of RandConv's consistency loss.

    Each view's predictive Gaussian is compared with the moment-matched
    Gaussian of all views by KL divergence, averaged over views (RandConv:
    KL of each class distribution to the mixture).

    Parameters
    ----------
    mus, sigmas : list of Tensor, each (B, C, T, K)
        Predicted means and standard deviations of the views.
    mask : Tensor of bool, optional
        Points to include (e.g. frames with plasma); default all.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    mu = torch.stack(mus)
    var = torch.stack(sigmas) ** 2
    mu_m = mu.mean(0)
    var_m = (var + mu**2).mean(0) - mu_m**2
    var_m = var_m.clamp_min(1e-6)
    kl = 0.5 * (torch.log(var_m / var) + (var + (mu - mu_m) ** 2) / var_m - 1.0)
    kl = kl.mean(0)
    if mask is not None:
        kl = kl[mask]
    return kl.mean()
