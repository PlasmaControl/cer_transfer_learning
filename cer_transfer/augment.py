"""SpecAugment-style masking for spectrogram batches (B, C, T, W).

torchaudio's masking transforms treat the last two dims as (freq, time) of a
spectrogram; our layout is (T, W) = (time, wavelength). FrequencyMasking
therefore masks along W only if applied to the transposed view — to keep
semantics explicit we name parameters by OUR axes:

    wavelength_mask_param -> contiguous span masked along W
    time_mask_param       -> contiguous span masked along T

iid_masks=True draws an independent mask per batch element and channel.

NOTE: the old code applied FrequencyMasking(120)/TimeMasking(200) directly
to (B, C, T, W) — torchaudio then masked T with 120 and W with 200, i.e.
the opposite axes of what the parameter names suggested. Defaults below
reproduce that EFFECTIVE behavior (T<=120, W<=200); rename-adjust if the
original intent was the other way around.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchaudio.transforms import FrequencyMasking, TimeMasking


class AugmentationPipeline(nn.Module):
    def __init__(self, wavelength_mask_param: int = 200,
                 time_mask_param: int = 120):
        super().__init__()
        # On (B, C, T, W): FrequencyMasking masks dim -2 (our T),
        # TimeMasking masks dim -1 (our W). Map accordingly.
        self.spec_aug = nn.Sequential(
            TimeMasking(time_mask_param=wavelength_mask_param,
                        iid_masks=True),          # masks W
            FrequencyMasking(freq_mask_param=time_mask_param,
                             iid_masks=True),     # masks T
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spec
        return self.spec_aug(spec)
