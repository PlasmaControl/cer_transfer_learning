"""Model: machine-agnostic Backbone + machine-specific Head.

Transfer design:
  - Backbone (encoder + pool + refine conv) is shared/transferred.
  - Head is swapped per machine (chord count differs), re-initialized for
    the target machine, and fine-tuned.

Head design: ONE shared trunk (ResidualBlock + MLP) + per-chord linear
output layers. This replaces the old ModuleList of full per-chord heads
(~80x duplicated feature refinement) with a shared representation and a
cheap per-chord readout.

Extension point (geometry conditioning): to share a single head across
machines, replace the per-chord Linear list with one Linear conditioned on
a chord-geometry embedding (R, tangency angle, ...) concatenated to the
trunk features. Then only the embedding table is machine-specific.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import MachineConfig, ModelConfig


def _norm2d(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        groups = 8 if channels % 8 == 0 else 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"unknown norm '{kind}'")


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with normalization and GELU, plus a skip path.

    Parameters
    ----------
    in_channels, out_channels : int
        Channel counts; a 1x1 projection is added to the skip path when they
        differ.
    kernel_size : int, default=3
        Convolution kernel (time x wavelength).
    bias : bool, default=True
        Convolution bias.
    norm : {'group', 'batch'}, default='group'
        Normalization layer after each convolution.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, bias=True, norm="group"
    ):
        super().__init__()
        padding = (
            tuple(k // 2 for k in kernel_size)
            if isinstance(kernel_size, tuple)
            else kernel_size // 2
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=bias
        )
        self.norm1 = _norm2d(norm, out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, padding=padding, bias=bias
        )
        self.norm2 = _norm2d(norm, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=bias)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        """Apply the block.

        Parameters
        ----------
        x : Tensor of shape (B, C_in, T, W)

        Returns
        -------
        Tensor of shape (B, C_out, T, W)
        """
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + self.skip(x))


class Backbone(nn.Module):
    """Encoder over (B, C_in, T, W) -> features (B, F, T, hidden_dim).

    Time resolution is preserved; wavelength dim is pooled down.

    Legacy mode (cfg.feature_width is None): F = C_in, whole encoder is one
    machine-bound block ('encoder'); transfer requires equal channel counts.

    Stem/trunk mode (cfg.feature_width set): F = cfg.feature_width. The
    first block ('stem', C_in -> encoder_widths[0]) is machine-specific and
    re-initialized per machine; the rest ('trunk') is machine-agnostic and
    transfers across machines with different channel counts.
    """

    def __init__(self, n_input_channels: int, cfg: ModelConfig, stem_w_pools: int = 0):
        super().__init__()
        self.feature_width = cfg.feature_width
        out_width = cfg.feature_width or n_input_channels
        self.out_width = out_width
        widths = [n_input_channels, *cfg.encoder_widths, out_width]

        def block(w_in, w_out, w_pool=1):
            """w_pool: number of W-halvings in this block (0 = keep W)."""
            layers = [
                ResidualBlock(w_in, w_out, cfg.kernel_size, cfg.bias, cfg.norm),
                nn.Dropout(cfg.dropout),
            ]
            if w_pool > 0:
                f = 2**w_pool
                layers.append(
                    nn.MaxPool2d(kernel_size=(3, f), stride=(1, f), padding=(1, 0))
                )
            else:
                layers.append(
                    nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
                )
            return layers

        self.agnostic = cfg.agnostic
        if cfg.agnostic:
            if cfg.feature_width is None:
                raise ValueError("agnostic mode requires feature_width")
            # One light stem shared by all chords (chords as batch): a strided
            # convolution and a wavelength pool bring W down by 4 before the
            # residual block, keeping activations small (they scale with the
            # number of chords). The chord-averaged map is projected to the
            # trunk width. The machine's stem_w_pools is not used: with
            # --resample-w all machines share a pixel scale (256 bins ->
            # hidden_dim 8 after the trunk).
            stem_w_pools = 2
        if cfg.feature_width is None:
            # legacy: every block halves W once; machine-bound at both ends
            blocks = []
            for w_in, w_out in zip(widths[:-1], widths[1:]):
                blocks += block(w_in, w_out, w_pool=1)
            self.encoder = nn.Sequential(*blocks)
            self.early = self.encoder[: len(self.encoder) // 2]
            self.n_pools = len(widths) - 1
        else:
            # stem absorbs machine-specific wavelength scale (stem_w_pools
            # halvings); the shared trunk halves W in its first
            # cfg.trunk_w_pools blocks and keeps W in the rest, so trunk
            # filters see a comparable wavelength scale on every machine.
            if cfg.agnostic:
                sw = cfg.agnostic_stem_width
                k = cfg.kernel_size
                self.stem = nn.Sequential(
                    nn.Conv2d(1, sw, k, stride=(1, 2), padding=k // 2, bias=cfg.bias),
                    _norm2d(cfg.norm, sw),
                    nn.GELU(),
                    nn.MaxPool2d(kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)),
                    ResidualBlock(sw, sw, k, cfg.bias, cfg.norm),
                    nn.Dropout(cfg.dropout),
                )
                self.chord_merge = nn.Conv2d(sw, widths[1], 1, bias=cfg.bias)
                self.chord_dim = sw * cfg.hidden_dim
                if cfg.chord_attention:
                    self.chord_attn = nn.MultiheadAttention(
                        self.chord_dim, cfg.chord_attention_heads, batch_first=True
                    )
                    self.chord_norm = nn.LayerNorm(self.chord_dim)
                else:
                    self.chord_attn = None
            else:
                self.stem = nn.Sequential(
                    *block(widths[0], widths[1], w_pool=stem_w_pools)
                )
            trunk_blocks = []
            n_trunk = len(widths) - 2
            for i, (w_in, w_out) in enumerate(zip(widths[1:-1], widths[2:])):
                trunk_blocks += block(
                    w_in, w_out, w_pool=1 if i < cfg.trunk_w_pools else 0
                )
            self.trunk = nn.Sequential(*trunk_blocks)
            self.encoder = nn.Sequential(self.stem, self.trunk)
            self.early = nn.Sequential(self.stem, self.trunk[: len(self.trunk) // 2])
            self.n_pools = stem_w_pools + min(cfg.trunk_w_pools, n_trunk)
        self.hidden_dim = cfg.hidden_dim
        self.pool = nn.AdaptiveAvgPool2d((None, cfg.hidden_dim))
        self.refine = nn.Conv2d(
            out_width,
            out_width,
            cfg.kernel_size,
            padding=cfg.kernel_size // 2,
            bias=cfg.bias,
        )

    def forward(self, x):
        # Variable wavelength dim between shots is supported (batches are
        # per-shot, so W is uniform within a batch), but W must survive the
        # pooling cascade with at least hidden_dim bins — otherwise the
        # adaptive pool would silently UPSAMPLE by duplication.
        """Encode spectrograms into time-resolved features.

        Parameters
        ----------
        x : Tensor of shape (B, C_in, T, W)
            Preprocessed (log10) spectrograms.

        Returns
        -------
        Tensor of shape (B, F, T, hidden_dim)
            ``F`` is ``C_in`` in legacy mode or ``cfg.feature_width`` in
            stem/trunk mode; the time axis is preserved.
        """
        w_after = x.shape[-1] >> self.n_pools
        if w_after < self.hidden_dim:
            raise ValueError(
                f"wavelength dim {x.shape[-1]} reduces to {w_after} bins "
                f"after {self.n_pools} pooling stages, below hidden_dim="
                f"{self.hidden_dim}; need W >= "
                f"{self.hidden_dim << self.n_pools} (or reduce hidden_dim /"
                f" encoder depth)"
            )
        if not self.agnostic:
            return self.refine(self.pool(self.encoder(x)))
        b, c, t, w = x.shape
        per_chord = self.stem(x.reshape(b * c, 1, t, w))  # (B*C, F0, T, W1)
        f0, w1 = per_chord.shape[1], per_chord.shape[-1]
        per_chord = per_chord.reshape(b, c, f0, t, w1)
        merged = self.chord_merge(per_chord.mean(dim=1))
        shared = self.refine(self.pool(self.trunk(merged)))
        # per-chord tokens: (B, C, T, F0 * hidden)
        tok = self.pool(per_chord.reshape(b * c, f0, t, w1))
        tok = tok.reshape(b, c, f0, t, self.hidden_dim).permute(0, 1, 3, 2, 4)
        tok = tok.reshape(b, c, t, f0 * self.hidden_dim)
        if self.chord_attn is not None:
            # every chord attends over all chords of the same frame
            q = tok.permute(0, 2, 1, 3).reshape(b * t, c, -1)  # (B*T, C, D)
            att, _ = self.chord_attn(q, q, q, need_weights=False)
            q = self.chord_norm(q + att)
            tok = q.reshape(b, t, c, -1).permute(0, 2, 1, 3)
        return shared, tok

    def param_groups(self):
        """(early, late) parameter lists for discriminative fine-tuning.
        In stem/trunk mode the stem is EXCLUDED from both (it is fresh per
        machine and should train at head-level LR; see stem_parameters)."""
        early_ids = {id(p) for p in self.early.parameters()}
        stem_ids = (
            {id(p) for p in self.stem.parameters()}
            if self.feature_width is not None and not self.agnostic
            else set()
        )
        early, late = [], []
        for p in self.parameters():
            if id(p) in stem_ids:
                continue
            (early if id(p) in early_ids else late).append(p)
        return early, late

    def stem_parameters(self):
        """Parameters of the machine-specific stem (empty list in legacy mode).

        Returns
        -------
        list of Parameter
        """
        if self.feature_width is None or self.agnostic:
            return []  # legacy: no stem; agnostic: the stem is shared, not fresh
        return list(self.stem.parameters())


class SharedTrunkHead(nn.Module):
    """Shared trunk + per-chord readout ('linear' or 'mlp').

    Output: mu, sigma each (B, n_chords, T, n_targets).
    """

    def __init__(
        self, machine: MachineConfig, n_backbone_channels: int, cfg: ModelConfig
    ):
        super().__init__()
        self.n_chords = machine.n_chords
        self.n_targets = machine.n_targets
        self.trunk_conv = ResidualBlock(
            n_backbone_channels,
            n_backbone_channels,
            cfg.kernel_size,
            cfg.bias,
            cfg.norm,
        )
        feat = n_backbone_channels * cfg.hidden_dim
        self.trunk_mlp = nn.Sequential(
            nn.Linear(feat, cfg.head_hidden, bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(cfg.head_hidden, cfg.head_hidden, bias=True),
            nn.LeakyReLU(0.1),
        )

        self.use_moments = bool(getattr(cfg, "moment_features", False))
        ro_in = cfg.head_hidden + (3 if self.use_moments else 0)

        def make_readout():
            """One per-chord readout: ``head_hidden`` (+ moments) -> 2 * n_targets."""
            if cfg.head_type == "linear":
                return nn.Linear(ro_in, 2 * self.n_targets, bias=True)
            if cfg.head_type == "mlp":
                return nn.Sequential(
                    nn.Linear(ro_in, cfg.readout_hidden, bias=True),
                    nn.LeakyReLU(0.1),
                    nn.Linear(cfg.readout_hidden, 2 * self.n_targets, bias=True),
                )
            raise ValueError(f"unknown head_type '{cfg.head_type}'")

        self.agnostic = cfg.agnostic
        if cfg.agnostic:
            # per-chord stem features (agnostic_stem_width x hidden_dim per
            # frame) projected to head_hidden and concatenated with the
            # shared trunk features; one readout for all chords
            self.chord_proj = nn.Sequential(
                nn.Linear(cfg.agnostic_stem_width * cfg.hidden_dim, cfg.head_hidden),
                nn.LeakyReLU(0.1),
            )
            ro_in += cfg.head_hidden
            self.readout = make_readout()
        else:
            self.readout = nn.ModuleList([make_readout() for _ in range(self.n_chords)])

    def forward(self, x, moments=None):
        # x: (B, C, T, F); moments: (B, n_chords, T, 3) or None
        """Predict per-chord means and sigmas from backbone features.

        Parameters
        ----------
        x : Tensor of shape (B, F, T, hidden_dim)
            Backbone features.
        moments : Tensor, optional
            Per-chord line moments, concatenated to the shared features when
            ``cfg.moment_features`` is set.

        Returns
        -------
        mu, sigma : Tensor of shape (B, n_chords, T, n_targets)
            Normalized-unit predictions; ``sigma`` is positive (softplus).
        """
        chord_feats = None
        if isinstance(x, tuple):
            x, chord_feats = x
        h = self.trunk_conv(x).permute(0, 2, 1, 3)  # (B, T, C, F)
        b, t = h.shape[:2]
        h = self.trunk_mlp(h.reshape(b, t, -1))  # (B, T, H)
        if self.agnostic:
            if chord_feats is None:
                raise RuntimeError("agnostic head needs per-chord stem features")
            c = chord_feats.shape[1]
            g = self.chord_proj(chord_feats)  # (B, C, T, H)
            z = torch.cat([h.unsqueeze(1).expand(-1, c, -1, -1), g], dim=-1)
            if self.use_moments:
                z = torch.cat([z, moments], dim=-1)
            outs = self.readout(z)
        elif self.use_moments:
            if moments is None:
                raise RuntimeError(
                    "model built with moment_features=True " "but no moments passed"
                )
            outs = torch.stack(
                [
                    ro(torch.cat([h, moments[:, c]], dim=-1))
                    for c, ro in enumerate(self.readout)
                ],
                dim=1,
            )
        else:
            outs = torch.stack([ro(h) for ro in self.readout], dim=1)
        # (B, n_chords, T, 2*n_targets)
        mu = outs[..., : self.n_targets]
        sigma = F.softplus(outs[..., self.n_targets :]) + 1e-4
        return mu, sigma


class FullChordHead(nn.Module):
    """Per-chord ResidualBlock + MLP on backbone features (old CERProcessor
    behavior). Maximum per-chord capacity; ~n_chords x the parameters of the
    shared-trunk variants. Empirically strongest on DIII-D (R2 0.7 -> 0.98
    vs a single shared head), but a fresh instance is expensive to fit on
    label-scarce target machines.
    """

    class _OneChord(nn.Module):
        def __init__(self, n_channels, cfg: ModelConfig, n_targets: int):
            super().__init__()
            self.conv = ResidualBlock(
                n_channels, n_channels, cfg.kernel_size, cfg.bias, cfg.norm
            )
            feat = n_channels * cfg.hidden_dim
            self.mlp = nn.Sequential(
                nn.Linear(feat, 128, bias=True),
                nn.LeakyReLU(0.1),
                nn.Linear(128, 64, bias=True),
                nn.LeakyReLU(0.1),
                nn.Linear(64, 2 * n_targets, bias=True),
            )

        def forward(self, x):
            """Per-chord features (B, F, T, hidden_dim) -> (B, T, 2 * n_targets)."""
            h = self.conv(x).permute(0, 2, 1, 3)
            b, t = h.shape[:2]
            return self.mlp(h.reshape(b, t, -1))  # (B, T, 2*n_targets)

    def __init__(
        self, machine: MachineConfig, n_backbone_channels: int, cfg: ModelConfig
    ):
        super().__init__()
        self.n_targets = machine.n_targets
        self.chord_heads = nn.ModuleList(
            [
                self._OneChord(n_backbone_channels, cfg, self.n_targets)
                for _ in range(machine.n_chords)
            ]
        )

    def forward(self, x):
        """Predict per-chord means and sigmas with one head per chord.

        Parameters
        ----------
        x : Tensor of shape (B, F, T, hidden_dim)

        Returns
        -------
        mu, sigma : Tensor of shape (B, n_chords, T, n_targets)
        """
        outs = torch.stack([h(x) for h in self.chord_heads], dim=1)
        mu = outs[..., : self.n_targets]
        sigma = F.softplus(outs[..., self.n_targets :]) + 1e-4
        return mu, sigma


def build_head(
    machine: MachineConfig, n_backbone_channels: int, cfg: ModelConfig
) -> nn.Module:
    """Instantiate the head selected by ``cfg.head_type``.

    Parameters
    ----------
    machine : MachineConfig
    n_backbone_channels : int
        Feature channels produced by the backbone.
    cfg : ModelConfig

    Returns
    -------
    nn.Module
        ``SharedTrunkHead`` for ``'linear'`` / ``'mlp'``, ``FullChordHead``
        for ``'full'``.
    """
    if cfg.head_type in ("linear", "mlp"):
        return SharedTrunkHead(machine, n_backbone_channels, cfg)
    if cfg.head_type == "full":
        return FullChordHead(machine, n_backbone_channels, cfg)
    raise ValueError(f"unknown head_type '{cfg.head_type}'")


class CERModel(nn.Module):
    """Backbone plus head: spectrograms in, per-chord (mu, sigma) out.

    Parameters
    ----------
    machine : MachineConfig
        Defines input channels, chords and targets.
    cfg : ModelConfig
        Architecture hyperparameters.
    """

    def __init__(self, machine: MachineConfig, cfg: ModelConfig):
        super().__init__()
        self.backbone = Backbone(
            machine.n_input_channels, cfg, stem_w_pools=machine.stem_w_pools
        )
        self.head = build_head(machine, self.backbone.out_width, cfg)

    def forward(self, x, moments=None):
        """Run backbone and head.

        Parameters
        ----------
        x : Tensor of shape (B, C_in, T, W)
        moments : Tensor, optional
            Per-chord line moments (see ``SharedTrunkHead``).

        Returns
        -------
        mu, sigma : Tensor of shape (B, n_chords, T, n_targets)
        """
        f = self.backbone(x)
        try:
            return self.head(f, moments)
        except TypeError:
            return self.head(f)


def build_model(machine: MachineConfig, cfg: ModelConfig) -> CERModel:
    """Construct a ``CERModel`` for a machine.

    Parameters
    ----------
    machine : MachineConfig
    cfg : ModelConfig

    Returns
    -------
    CERModel
    """
    if cfg.agnostic:
        if cfg.head_type == "full":
            raise ValueError("agnostic mode supports head_type 'linear' or 'mlp'")
        if machine.n_input_channels != machine.n_chords:
            raise ValueError(
                "agnostic mode needs one input channel per chord "
                f"({machine.name}: {machine.n_input_channels} channels, "
                f"{machine.n_chords} chords)"
            )
    return CERModel(machine, cfg)


def _strip_running_stats(state: dict) -> dict:
    return {
        k: v
        for k, v in state.items()
        if "running_mean" not in k
        and "running_var" not in k
        and "num_batches_tracked" not in k
    }


def transfer_backbone(
    target_model: CERModel,
    source_backbone_state: dict,
    reset_norm_running_stats: bool = True,
    trunk_only: bool = False,
) -> CERModel:
    """Load source backbone weights into the target model; head stays fresh.

    trunk_only=False (legacy): full backbone, requires equal channel counts.
    trunk_only=True (stem/trunk mode): loads 'trunk.*' and 'refine.*' only;
    the machine-specific stem keeps its fresh initialization. Use this when
    source and target machines differ in channel count.

    reset_norm_running_stats: drop BatchNorm running_mean/var so they are
    re-estimated on target-machine data (AdaBN). No-op for GroupNorm.
    """
    if reset_norm_running_stats:
        source_backbone_state = _strip_running_stats(source_backbone_state)
    if trunk_only:
        state = {
            k: v
            for k, v in source_backbone_state.items()
            if k.startswith("trunk.") or k.startswith("refine.")
        }
        if not state:
            raise RuntimeError(
                "trunk_only transfer requested but source checkpoint has no "
                "'trunk.*' keys — the source model was trained in legacy "
                "mode (feature_width=None); retrain the source with "
                "feature_width set, or match channel counts."
            )
        missing, unexpected = target_model.backbone.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"trunk transfer: unexpected={unexpected}")
        loaded = set(state)
        want = {
            k
            for k in target_model.backbone.state_dict()
            if k.startswith("trunk.") or k.startswith("refine.")
        }
        dropped = [
            k for k in want - loaded if "running" not in k and "num_batches" not in k
        ]
        if dropped:
            raise RuntimeError(f"trunk transfer incomplete: missing={dropped}")
        return target_model
    missing, unexpected = target_model.backbone.load_state_dict(
        source_backbone_state, strict=False
    )
    dropped = [m for m in missing if "running" not in m and "num_batches" not in m]
    if dropped or unexpected:
        raise RuntimeError(
            f"backbone transfer mismatch: missing={dropped}, "
            f"unexpected={unexpected}"
        )
    return target_model


def transfer_full_head(
    target_head: nn.Module,
    source_head_state: dict,
    reset_norm_running_stats: bool = True,
) -> None:
    """Warm-start the ENTIRE head (shared trunk + all per-chord readouts)
    from a source machine. Requires identical head architecture and chord
    count (e.g. NSTX -> NSTX-U, 51/51). Geometry shifts between machines are
    handled by fine-tuning, not by re-initialization."""
    state = (
        _strip_running_stats(source_head_state)
        if reset_norm_running_stats
        else dict(source_head_state)
    )
    missing, unexpected = target_head.load_state_dict(state, strict=False)
    dropped = [k for k in missing if "running" not in k and "num_batches" not in k]
    if dropped or unexpected:
        raise RuntimeError(
            f"full-head transfer mismatch (chord count or head type differ?):"
            f" missing={dropped}, unexpected={unexpected}"
        )


def transfer_head_trunk(
    target_head: nn.Module,
    source_head_state: dict,
    reset_norm_running_stats: bool = True,
) -> None:
    """Load the shared trunk of a SharedTrunkHead (trunk_conv + trunk_mlp)
    from a source machine; per-chord readouts stay fresh. Valid because the
    head trunk's input width is the backbone's fixed feature width."""
    state = {
        k: v
        for k, v in source_head_state.items()
        if k.startswith("trunk_conv.") or k.startswith("trunk_mlp.")
    }
    if reset_norm_running_stats:
        state = _strip_running_stats(state)
    if not state:
        raise RuntimeError(
            "source head has no shared trunk to transfer " "(head_type='full'?)"
        )
    missing, unexpected = target_head.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"head-trunk transfer: unexpected={unexpected}")
