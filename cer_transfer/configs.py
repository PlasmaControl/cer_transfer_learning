"""Machine configurations.

All machine-specific facts (channel maps, chord counts, spectrogram geometry,
preprocessing choices, data locations) live here. Nothing machine-specific is
allowed in data.py / models.py / training.py.

To add a new machine: define a MachineConfig and register it. No other file
should need to change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Root of the discharge files. Split lists and the default data directories
# hold paths relative to it; absolute paths are used as they are.
DATA_ROOT = Path(os.environ.get("CER_DATA_ROOT", "."))


def data_path(p) -> Path:
    """Resolve a discharge-file path against ``CER_DATA_ROOT``.

    Parameters
    ----------
    p : str or Path
        Entry of a split list or a command-line path.

    Returns
    -------
    Path
        ``p`` unchanged if absolute, else ``CER_DATA_ROOT / p``.
    """
    p = Path(p)
    return p if p.is_absolute() else DATA_ROOT / p


def list_entry(p) -> str:
    """Path as written into a split list.

    Parameters
    ----------
    p : str or Path
        Discharge file.

    Returns
    -------
    str
        Relative to ``CER_DATA_ROOT`` when the file lies inside it, else absolute.
    """
    p = Path(p).resolve()
    try:
        return str(p.relative_to(DATA_ROOT.resolve()))
    except ValueError:
        return str(p)


from typing import Optional, Sequence


@dataclass(frozen=True)
class MachineConfig:
    """Everything the pipeline needs to know about one machine's diagnostic.

    Parameters
    ----------
    name : str
        Registry key, e.g. "d3d".
    n_raw_channels : int
        Number of spectrometer channels in the raw data files (dim 0 of
        'input' array).
    input_channel_indices : sequence of int, optional
        Which raw channels feed the model. None = all.
    chord_indices : sequence of int, optional
        Which target chords to predict/evaluate. None = all raw channels.
    n_wavelength_bins : int, optional
        Informational only — the model is agnostic to the wavelength bin
        count (variable W between shots is supported; W must be constant
        within a shot since batches are per-shot). All shots are assumed to
        cover the same physical wavelength range; the encoder's adaptive
        pooling maps that range onto hidden_dim bins regardless of W.
    targets : sequence of str
        Names of the regression targets, in target-array order.
    baseline_region : (int, int), optional
        Wavelength-bin slice used to estimate the additive background offset.
        If None, the per-timestep minimum over all wavelength bins is used.
        (Per-timestep, never per-subsequence: preprocessing must not depend
        on how a shot was chunked.)
    train_dir, val_dir : Path, optional
        Default data locations on the cluster.
    """

    name: str
    n_raw_channels: int
    n_wavelength_bins: Optional[int] = None
    input_channel_indices: Optional[Sequence[int]] = None
    chord_indices: Optional[Sequence[int]] = None
    targets: Sequence[str] = ("ti", "vtor")
    # sigma (physical units per target) for the zero pseudo-labels on
    # dark frames (--zero-dark-frames); conservative ~ median label sigma
    dark_sigma: Sequence[float] = (100.0, 10.0)
    # Multiplicative factor per target converting the FILES' stored units
    # into the pipeline's canonical units, applied at load time (targets and
    # target_error alike; NaNs pass through). Canonical units:
    #   ti   -> eV
    #   vtor -> km/s
    # All reported metrics (rmse_*), norm stats, and denormalized model
    # outputs are therefore in canonical units on every machine.
    target_scale: Sequence[float] = (1.0, 1.0)
    baseline_region: Optional[tuple[int, int]] = None
    stem_w_pools: int = 0  # stem/trunk mode only: wavelength halvings done
    # in the machine-specific stem. Purpose: machines
    # with different dispersion (bins over the same
    # physical range) are brought to a comparable
    # wavelength scale BEFORE the shared trunk, so
    # trunk filters see machine-independent pixel
    # physics. Total W halvings = stem_w_pools +
    # ModelConfig.trunk_w_pools; min usable W =
    # hidden_dim * 2**(total).
    # Wavelength-axis convention relative to the NSTX CHERS reference, used
    # only when the axis is resampled (agnostic pipeline): reverse the pixel
    # direction, and stretch by the dispersion ratio. Determined from the
    # data with cer_transfer.analysis.axis_sweep; verify against instrument
    # calibration when available.
    wavelength_flip: bool = False
    wavelength_scale: float = 1.0
    train_dir: Optional[Path] = None
    val_dir: Optional[Path] = None

    @property
    def n_input_channels(self) -> int:
        """Number of spectrogram channels fed to the model."""
        if self.input_channel_indices is None:
            return self.n_raw_channels
        return len(self.input_channel_indices)

    @property
    def n_chords(self) -> int:
        """Number of output chords (labeled lines of sight)."""
        if self.chord_indices is None:
            return self.n_raw_channels
        return len(self.chord_indices)

    @property
    def n_targets(self) -> int:
        """Number of regression targets."""
        return len(self.targets)


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters (machine-independent)."""

    hidden_dim: int = 16  # freq dim after adaptive pooling
    kernel_size: int = 3
    dropout: float = 0.3
    bias: bool = True
    norm: str = "group"  # "group" | "batch"; group recommended with
    # single-shot batches (per-shot BN stats are
    # unreliable and poison fine-tuning)
    head_hidden: int = 128  # shared head trunk MLP width
    # Per-chord capacity ladder (empirically critical — shared-linear readout
    # cost ~0.28 R2 vs full per-chord heads on DIII-D):
    #   "linear": per-chord Linear readout of shared trunk (cheapest)
    #   "mlp":    per-chord 2-layer MLP readout of shared trunk (default)
    #   "full":   per-chord ResidualBlock + MLP on backbone features
    #             (old CERProcessor behavior, ~80x params)
    head_type: str = "mlp"
    readout_hidden: int = 64  # per-chord MLP width for head_type="mlp"
    # Transfer-ready backbone: when set, the encoder ends at this fixed
    # feature width instead of n_input_channels, and the first block is a
    # machine-specific STEM (C_machine -> first encoder width). Transfer then
    # loads the trunk (+ head trunk) across machines with different channel
    # counts; stem and per-chord readouts are re-initialized per machine.
    # None = legacy architecture (channel-count-bound at both ends).
    feature_width: Optional[int] = None
    moment_features: bool = False  # classical spectral moments (amplitude,
    # centroid, width) per chord/frame at
    # native W, fed to that chord's readout
    trunk_w_pools: int = 3  # stem/trunk mode: how many of the trunk blocks
    # halve W (the rest keep W). Machine-independent.
    encoder_widths: tuple = (128, 256, 256, 128)
    # Machine-agnostic variant (zero-shot transfer): the stem processes each
    # chord with shared weights, the trunk sees the chord-averaged map, and one
    # shared readout combines trunk and per-chord features. No parameter then
    # depends on the number of chords, so a checkpoint applies to any machine.
    agnostic: bool = False
    # width of the shared per-chord stem (kept small: its activations scale
    # with the number of chords); it is projected to encoder_widths[0] after
    # averaging over chords
    agnostic_stem_width: int = 32
    # agnostic mode: let each chord attend over all chords (per frame) before
    # the readout, instead of relying on its own stem features alone; set
    # based, permutation-invariant, no positional information
    chord_attention: bool = False
    chord_attention_heads: int = 4
    # Resample the wavelength axis to this many bins in preprocessing (plain
    # interpolation), so spectrometers with different resolution share a
    # pixel scale. None keeps the native axis.
    resample_w: Optional[int] = None


@dataclass(frozen=True)
class TrainConfig:
    """Optimization hyperparameters shared by pre-training and fine-tuning.

    Attributes
    ----------
    lr, weight_decay : float
        AdamW learning rate and weight decay.
    batch_size, subseq_len : int
        Subsequences per batch and frames per subsequence (``-1``: whole shot).
    num_epochs, patience, scheduler_patience : int
        Epoch limit, early-stopping patience and ReduceLROnPlateau patience.
    accumulation_steps, grad_clip : int, float
        Gradient accumulation and clipping.
    max_label_relerr : float
        Drop labels whose relative uncertainty exceeds this (``0``: keep all).
    per_chord_norm, noise_aware_nll : bool
        Per-chord target normalization; label sigma in the NLL variance.
    scheduler_monitor : str
        Quantity watched by the LR scheduler (``'val_loss'`` or ``'val_score'``).
    seed, num_workers, profile : int, int, bool
        Reproducibility seed, loader workers, per-epoch profiling.
    augment, wavelength_mask_param, time_mask_param : bool, int, int
        SpecAugment-style masking of the input spectrograms.
    """

    lr: float = 1e-3
    weight_decay: float = 0.05
    batch_size: int = 256
    subseq_len: int = 256
    num_epochs: int = 500
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    patience: int = 50
    scheduler_patience: int = 10
    max_label_relerr: float = 0.0  # training label QC (train.py --help)
    per_chord_norm: bool = False  # normalize targets per chord (removes
    # cross-chord profile variance from the
    # loss -> model must learn dynamics, not
    # climatology)
    noise_aware_nll: bool = False  # NLL with total var = sigma_model^2 +
    # sigma_label^2 (uses target_error;
    # stops rewarding label-noise fitting)
    scheduler_monitor: str = "val_loss"  # "val_loss" (legacy, min) or
    # "val_score" (max; recommended for
    # new runs — val_loss plateaus are
    # meaningless under sigma overfit)
    seed: int = 42
    num_workers: int = 8
    profile: bool = False  # per-batch phase timing to stdout
    augment: bool = False  # SpecAugment on training batches
    wavelength_mask_param: int = 200  # max masked span along W
    time_mask_param: int = 120  # max masked span along T


@dataclass(frozen=True)
class FinetuneConfig(TrainConfig):
    """LP-FT schedule: linear-probe the new head first, then unfreeze."""

    probe_epochs: int = 30  # phase 1: backbone frozen, head only
    probe_lr: float = 1e-3
    backbone_lr: float = 1e-5  # phase 2 discriminative LRs
    late_backbone_lr: float = 1e-4  # last encoder block
    head_lr: float = 1e-3
    reset_norm_running_stats: bool = True  # AdaBN-style; only affects BatchNorm


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, MachineConfig] = {}


def register(cfg: MachineConfig) -> MachineConfig:
    """Register a machine configuration under its name.

    Parameters
    ----------
    cfg : MachineConfig
        Configuration to add to the registry.

    Returns
    -------
    MachineConfig
        The same object, for use as a decorator-like helper.
    """
    if cfg.name in _REGISTRY:
        raise ValueError(f"machine '{cfg.name}' already registered")
    _REGISTRY[cfg.name] = cfg
    return cfg


def get_machine(name: str) -> MachineConfig:
    """Look up a registered machine configuration.

    Parameters
    ----------
    name : str
        Registry key, e.g. ``'d3d'``, ``'nstx'``, ``'nstx_passive'``.

    Returns
    -------
    MachineConfig

    Raises
    ------
    KeyError
        If no machine of that name is registered.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown machine '{name}'; known: {sorted(_REGISTRY)}"
        ) from None


D3D_FULL = register(
    MachineConfig(
        name="d3d",
        n_raw_channels=80,
        input_channel_indices=None,
        chord_indices=None,
        stem_w_pools=2,  # ~400-600 bins -> ~100-150 at trunk entry
        train_dir=Path("training_set_30"),  # relative to CER_DATA_ROOT
        # axis sweep of the NSTX model on DIII-D (tangential, 100 shots): the
        # vtor slope is positive only with the axis reversed and the affine
        # R2 peaks at a stretch of 1.4 -> DIII-D convention relative to NSTX
        wavelength_flip=True,
        wavelength_scale=1.4,
        val_dir=Path("test_set_30"),
    )
)

D3D_SUBSET = register(
    MachineConfig(
        name="d3d_subset",
        n_raw_channels=80,
        input_channel_indices=tuple(range(0, 80, 5)),
        chord_indices=(0, 1, 2, 3, 4, 5, 6, 16, 17, 18, 19, 20, 21, 32, 33, 34, 35),
        train_dir=Path("training_set_30"),  # relative to CER_DATA_ROOT
        # axis sweep of the NSTX model on DIII-D (tangential, 100 shots): the
        # vtor slope is positive only with the axis reversed and the affine
        # R2 peaks at a stretch of 1.4 -> DIII-D convention relative to NSTX
        wavelength_flip=True,
        wavelength_scale=1.4,
        val_dir=Path("test_set_30"),
    )
)

# Tangential CER system only (channels 0-47 of the 80-channel files); the
# vertical chords measure a different velocity component and have no
# counterpart on NSTX. Source machine for the zero-shot transfer study.
D3D_TANGENTIAL = register(
    MachineConfig(
        name="d3d_tangential",
        n_raw_channels=80,
        input_channel_indices=tuple(range(48)),
        chord_indices=tuple(range(48)),
        stem_w_pools=2,
        train_dir=Path("training_set_30"),  # relative to CER_DATA_ROOT
        # axis sweep of the NSTX model on DIII-D (tangential, 100 shots): the
        # vtor slope is positive only with the axis reversed and the affine
        # R2 peaks at a stretch of 1.4 -> DIII-D convention relative to NSTX
        wavelength_flip=True,
        wavelength_scale=1.4,
        val_dir=Path("test_set_30"),
    )
)

# Fill in once NSTX CHERS data files exist. Variable W is handled
# natively (per-shot batches + adaptive pooling over the wavelength axis).
NSTX = register(
    MachineConfig(
        name="nstx",
        n_raw_channels=51,  # 51 tangential CHERS channels
        target_scale=(1000.0, 1.0),  # files store Ti in keV -> canonical eV
        stem_w_pools=0,  # 80-100 bins: no stem pooling; trunk entry
        # scale then matches d3d within ~1.5x
        # train_dir/val_dir stay None DELIBERATELY: NSTX-family runs must use
        # cer_transfer.datasets.make_splits file lists
        # (--train-list/--val-list), which exclude
        # curated-out shots (end_index<=0, unlabeled). A list-less run fails
        # with a clear message instead of globbing uncurated directories.
        train_dir=None,
        val_dir=None,
    )
)

# NSTX-U: same CHERS chord count as NSTX (51/51), so stem and readouts are
# shape-compatible and can be warm-started from an NSTX checkpoint
# (finetune.py --transfer-readouts). Geometry may differ (center-stack
# rebuild -> shifted tangency radii); fine-tuning absorbs that.
NSTXU = register(
    MachineConfig(
        name="nstxu",
        n_raw_channels=51,
        target_scale=(1000.0, 1.0),  # files store Ti in keV -> canonical eV
        stem_w_pools=0,
        train_dir=None,  # TODO once extracted
        val_dir=None,
    )
)

# --- Dual-array NSTX(-U) datasets (second extraction generation) ---------
# The CHERS diagnostic has two spectrometer arrays viewing the SAME flux
# surfaces: one beam-intersecting (active) and one passive. New extraction:
# 48 chords per array, one spectrogram each; labels are always the ACTIVE
# analysis fits. Active configs are training targets; passive configs exist
# for EVALUATION of active-trained models on passive inputs (information-
# content study). 48 != 51, so these start a NEW stem/readout lineage —
# never mix with the legacy 51-chord checkpoints. Photon-count scale is
# much lower on passive; current working assumption is that log preprocess
# + GroupNorm absorbs it (test: zero-shot per-chord eval, edge vs core).
NSTX_ACTIVE = register(
    MachineConfig(
        name="nstx_active",
        n_raw_channels=48,
        target_scale=(1000.0, 1.0),
        stem_w_pools=0,
        train_dir=None,  # list-enforced, as all NSTX-family configs
        val_dir=None,
    )
)

NSTX_PASSIVE = register(
    MachineConfig(
        name="nstx_passive",  # evaluation-only by design (labels = active
        n_raw_channels=48,  # fits on the same frames; sightlines index-
        target_scale=(1000.0, 1.0),  # aligned with nstx_active)
        stem_w_pools=0,
        train_dir=None,
        val_dir=None,
    )
)

NSTXU_ACTIVE = register(
    MachineConfig(
        name="nstxu_active",
        n_raw_channels=48,
        target_scale=(1000.0, 1.0),
        stem_w_pools=0,
        train_dir=None,
        val_dir=None,
    )
)

NSTXU_PASSIVE = register(
    MachineConfig(
        name="nstxu_passive",
        n_raw_channels=48,
        target_scale=(1000.0, 1.0),
        stem_w_pools=0,
        train_dir=None,
        val_dir=None,
    )
)
