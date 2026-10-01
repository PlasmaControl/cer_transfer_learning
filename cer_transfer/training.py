"""Trainer, EarlyStopping, checkpoint I/O.

Checkpoint is a single self-describing dict:
    {model_state, backbone_state, head_state, norm_stats, machine,
     model_config, best_score, epoch}
so weights can never be separated from the normalization stats and machine
they belong to. Loading DIII-D weights for NSTX fine-tuning therefore
carries the DIII-D stats explicitly — and target stats are recomputed from
target data, never reused.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import time as _time

from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader

from .augment import AugmentationPipeline
from .configs import MachineConfig, ModelConfig, TrainConfig
from .losses import MaskedGaussianNLL, MaskedR2Score, MaskedRMSE, combined_score
from .models import CERModel


# --------------------------------------------------------------------------
# Checkpoint I/O
# --------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: CERModel,
    norm_stats: dict[str, torch.Tensor],
    machine: MachineConfig,
    model_config: ModelConfig,
    best_score: float,
    epoch: int,
    optimizer_state: Optional[dict] = None,
    scheduler_state: Optional[dict] = None,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone_state": model.backbone.state_dict(),
            "head_state": model.head.state_dict(),
            "norm_stats": {k: v.cpu() for k, v in norm_stats.items()},
            "machine": dataclasses.asdict(machine),
            "model_config": dataclasses.asdict(model_config),
            "best_score": best_score,
            "epoch": epoch,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
        },
        path,
    )


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


# --------------------------------------------------------------------------
# Early stopping
# --------------------------------------------------------------------------

class EarlyStopping:
    """Maximizes `score`. NaN is checked FIRST and never becomes best_score.

    `best_score` can be seeded from a previous run so that resuming never
    overwrites a better checkpoint with a worse model.
    """

    def __init__(self, patience: int, delta: float = 0.0,
                 best_score: Optional[float] = None,
                 on_improve: Optional[Callable[[float], None]] = None,
                 verbose: bool = False):
        self.patience = patience
        self.delta = delta
        self.best_score = best_score
        self.on_improve = on_improve
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.improved = False

    def step(self, score: float) -> bool:
        """Returns True if this score is a new best."""
        if score is None or (isinstance(score, float) and math.isnan(score)):
            self.improved = False
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            self.improved = True
            if self.on_improve is not None:
                self.on_improve(score)
            return True

        self.improved = False
        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping: {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
        return False


# --------------------------------------------------------------------------
# Normalization stats
# --------------------------------------------------------------------------

def compute_norm_stats_per_chord(dataset, targets):
    """Per-chord mean/std of each target over the dataset (canonical units).
    Returns {"mean": (C, n_t), "std": (C, n_t)} float32 tensors. NaN labels
    ignored; chords with <100 labeled points fall back to global stats."""
    from joblib import load as _load
    chord_idx = (None if dataset.machine.chord_indices is None
                 else np.asarray(dataset.machine.chord_indices))
    scale = np.asarray(dataset.machine.target_scale, dtype=np.float64)
    n_t = len(targets)
    C = dataset.machine.n_chords
    n = np.zeros((C, n_t))
    s1 = np.zeros((C, n_t))
    s2 = np.zeros((C, n_t))
    for fp in tqdm(dataset.file_paths, desc="per-chord norm stats",
                   mininterval=30):
        d = _load(fp, mmap_mode="r" if dataset.mmap else None)
        end = int(d["end_index"])
        if end <= 0:
            continue
        tgt = np.asarray(d["target"][:, :end, :], dtype=np.float64)
        if chord_idx is not None:
            tgt = tgt[chord_idx]
        tgt = tgt * scale
        m = np.isfinite(tgt)
        n += m.sum(axis=1)
        s1 += np.where(m, tgt, 0.0).sum(axis=1)
        s2 += np.where(m, tgt ** 2, 0.0).sum(axis=1)
    mean = s1 / np.maximum(n, 1)
    var = np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 1e-12)
    std = np.sqrt(var)
    g_mean = s1.sum(0) / np.maximum(n.sum(0), 1)
    g_var = np.maximum(s2.sum(0) / np.maximum(n.sum(0), 1) - g_mean ** 2,
                       1e-12)
    g_std = np.sqrt(g_var)
    starved = n < 100
    mean = np.where(starved, g_mean, mean)
    std = np.where(starved, g_std, std)
    return {"mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32)}


def compute_norm_stats(
    dataset, targets: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Per-target mean/std over non-NaN TRAIN targets.

    Reads ONLY the target arrays directly from the joblib mmaps — never the
    spectrograms and never through the DataLoader. Targets are ~0.4% of the
    file bytes, so this is orders of magnitude faster than iterating the
    training pipeline (which would load and log-preprocess every input just
    to discard it).

    Note: statistics cover all end_index samples per shot (the training
    pipeline's subseq chunking drops each shot's tail remainder; including
    it here is harmless and marginally more correct).
    """
    from joblib import load as _load

    chord_idx = (None if dataset.machine.chord_indices is None
                 else np.asarray(dataset.machine.chord_indices))
    scale = np.asarray(dataset.machine.target_scale, dtype=np.float64)
    sums = np.zeros(len(targets), dtype=np.float64)
    sqs = np.zeros(len(targets), dtype=np.float64)
    cnts = np.zeros(len(targets), dtype=np.float64)
    for fp in tqdm(dataset.file_paths, desc="normalization stats",
                   mininterval=30, miniters=1):
        d = _load(fp, mmap_mode="r")
        tgt = np.array(d["target"][:, : d["end_index"], :],
                       dtype=np.float64)
        del d
        if chord_idx is not None:
            tgt = tgt[chord_idx]
        tgt = tgt * scale  # canonical units (ti: eV, vtor: km/s)
        for i in range(len(targets)):
            v = tgt[..., i]
            m = ~np.isnan(v)
            v = v[m]
            sums[i] += v.sum()
            sqs[i] += (v ** 2).sum()
            cnts[i] += m.sum()
    stats = {}
    for i, t in enumerate(targets):
        mean = sums[i] / cnts[i]
        stats[f"{t}_mean"] = torch.tensor(mean, dtype=torch.float32)
        stats[f"{t}_std"] = torch.tensor(
            float(np.sqrt(sqs[i] / cnts[i] - mean ** 2)),
            dtype=torch.float32)
    return stats


def normalize_targets(target, stats, targets):
    if "mean" in stats:
        # per-chord stats: mean/std are (C, n_t); target is (.., C, T, n_t)
        return (target - stats["mean"].unsqueeze(-2)) / \
               stats["std"].unsqueeze(-2)
    out = target.clone()
    for i, t in enumerate(targets):
        out[..., i] = (target[..., i] - stats[f"{t}_mean"]) / stats[f"{t}_std"]
    return out


def normalize_sigmas(err, stats, targets):
    """Scale label sigmas into normalized-target units (divide by std)."""
    if "mean" in stats:
        return err / stats["std"].unsqueeze(-2)
    out = err.clone()
    for i, t in enumerate(targets):
        out[..., i] = err[..., i] / stats[f"{t}_std"]
    return out


def denormalize_mu_sigma(mu, sigma, stats, targets):
    if "mean" in stats:
        std = stats["std"].unsqueeze(-2)
        return mu * std + stats["mean"].unsqueeze(-2), sigma * std
    mu_d, sigma_d = mu.clone(), sigma.clone()
    for i, t in enumerate(targets):
        mu_d[..., i] = mu[..., i] * stats[f"{t}_std"] + stats[f"{t}_mean"]
        sigma_d[..., i] = sigma[..., i] * stats[f"{t}_std"]
    return mu_d, sigma_d


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model: CERModel,
        machine: MachineConfig,
        model_config: ModelConfig,
        train_config: TrainConfig,
        optimizer: optim.Optimizer,
        device: str = "cuda",
        checkpoint_path: Optional[Path] = None,
        norm_stats: Optional[dict[str, torch.Tensor]] = None,
        initial_best_score: Optional[float] = None,
        start_epoch: int = 0,
        optimizer_state: Optional[dict] = None,
        scheduler_state: Optional[dict] = None,
        loggers: Optional[list] = None,
        log_extra: Optional[dict] = None,
    ):
        self.model = model.to(device)
        self.machine = machine
        self.model_config = model_config
        self.cfg = train_config
        self.optimizer = optimizer
        self.criterion = MaskedGaussianNLL()
        self.scheduler_monitor = getattr(train_config, "scheduler_monitor",
                                         "val_loss")
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if self.scheduler_monitor == "val_score" else "min",
            factor=0.5,
            patience=train_config.scheduler_patience,
        )
        self.device = device
        self.augment = (
            AugmentationPipeline(train_config.wavelength_mask_param,
                                 train_config.time_mask_param).to(device)
            if train_config.augment else None
        )
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.norm_stats = (
            {k: v.to(device) for k, v in norm_stats.items()}
            if norm_stats else None
        )
        self.early_stopping = EarlyStopping(
            patience=train_config.patience,
            best_score=initial_best_score,
            verbose=True,
        )
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            if scheduler_state.get("mode") == self.scheduler.mode:
                self.scheduler.load_state_dict(scheduler_state)
            else:
                print("warning: checkpoint scheduler mode "
                      f"'{scheduler_state.get('mode')}' != configured "
                      f"'{self.scheduler.mode}' — scheduler restarted fresh "
                      "(LR history not restored)")
        self.loggers = loggers or []
        self.log_extra = log_extra or {}
        tnames = machine.targets
        self.r2 = {t: MaskedR2Score().to(device) for t in tnames}
        self.rmse = {t: MaskedRMSE().to(device) for t in tnames}
        self.epoch = start_epoch

    # -- one epoch ---------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, steps = 0.0, 0
        accum = max(1, self.cfg.accumulation_steps)
        self.optimizer.zero_grad(set_to_none=True)
        bar = tqdm(loader, desc=f"epoch {self.epoch}", mininterval=30,
                   miniters=1)
        profile = getattr(self.cfg, "profile", False)
        _cuda = "cuda" in str(self.device)
        t_data = t_h2d = t_compute = 0.0
        _t = _time.time() if profile else 0.0
        for k, (inputs, targets, t_err, moments) in enumerate(bar):
            if profile:
                _now = _time.time(); t_data += _now - _t; _t = _now
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if profile:
                if _cuda:
                    torch.cuda.synchronize()
                _now = _time.time(); t_h2d += _now - _t; _t = _now
            if self.augment is not None:
                inputs = self.augment(inputs)
            mu, sigma = self.model(inputs,
                                   moments.to(self.device, non_blocking=True))
            targets_n = normalize_targets(targets, self.norm_stats,
                                          self.machine.targets)
            err_n = None
            if getattr(self.cfg, "noise_aware_nll", False):
                err_n = normalize_sigmas(
                    t_err.to(self.device, non_blocking=True),
                    self.norm_stats, self.machine.targets)
            loss = self.criterion(mu, sigma, targets_n, err_n) / accum
            loss.backward()
            if (k + 1) % accum == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.cfg.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            total += loss.item() * accum
            steps += 1
            if profile:
                if _cuda:
                    torch.cuda.synchronize()
                _now = _time.time(); t_compute += _now - _t; _t = _now
                if (k + 1) % 200 == 0:
                    tot = max(t_data + t_h2d + t_compute, 1e-9)
                    print(f"[profile] batches {k-198}-{k+1}: "
                          f"data_wait {t_data:.1f}s ({100*t_data/tot:.0f}%), "
                          f"h2d {t_h2d:.1f}s ({100*t_h2d/tot:.0f}%), "
                          f"compute {t_compute:.1f}s "
                          f"({100*t_compute/tot:.0f}%)", flush=True)
                    t_data = t_h2d = t_compute = 0.0
        if steps % accum != 0:  # flush remainder
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.cfg.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return total / max(steps, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader):
        self.model.eval()
        for m in (*self.r2.values(), *self.rmse.values()):
            m.reset()
        val_loss, steps = 0.0, 0
        for inputs, targets, t_err, moments in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            mu_n, sigma_n = self.model(inputs,
                                       moments.to(self.device,
                                                  non_blocking=True))
            mu, _ = denormalize_mu_sigma(mu_n, sigma_n, self.norm_stats,
                                         self.machine.targets)
            for i, t in enumerate(self.machine.targets):
                self.r2[t].update(mu[..., i], targets[..., i])
                self.rmse[t].update(mu[..., i], targets[..., i])
            targets_n = normalize_targets(targets, self.norm_stats,
                                          self.machine.targets)
            err_n = None
            if getattr(self.cfg, "noise_aware_nll", False):
                err_n = normalize_sigmas(
                    t_err.to(self.device, non_blocking=True),
                    self.norm_stats, self.machine.targets)
            val_loss += self.criterion(mu_n, sigma_n, targets_n,
                                       err_n).item()
            steps += 1
        per_r2 = {t: float(self.r2[t].compute().cpu())
                  for t in self.machine.targets}
        per_rmse = {t: float(self.rmse[t].compute().cpu())
                    for t in self.machine.targets}
        return {
            "score": combined_score(per_r2),
            "r2": per_r2,
            "rmse": per_rmse,
            "val_loss": val_loss / max(steps, 1),
        }

    # -- full run ----------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        report_fn: Optional[Callable[[dict], None]] = None,
    ) -> float:
        if self.norm_stats is None and getattr(self.cfg, "per_chord_norm",
                                               False):
            self.norm_stats = {
                k: v.to(self.device)
                for k, v in compute_norm_stats_per_chord(
                    train_loader.dataset, self.machine.targets).items()
            }
        if self.norm_stats is None:
            self.norm_stats = {
                k: v.to(self.device)
                for k, v in compute_norm_stats(
                    train_loader.dataset, tuple(self.machine.targets)
                ).items()
            }
            # Persist immediately: stats are expensive to recompute and
            # must never be lost to a crash before the first best epoch.
            # best_score=None marks "no validated weights yet"; any real
            # score later overwrites this checkpoint.
            if self.checkpoint_path is not None:
                save_checkpoint(
                    self.checkpoint_path, self.model, self.norm_stats,
                    self.machine, self.model_config,
                    best_score=None, epoch=-1,
                )

        for self.epoch in range(self.epoch, self.cfg.num_epochs):
            _t0 = _time.time()
            train_loss = self._train_epoch(train_loader)
            metrics = self.validate(val_loader)
            epoch_seconds = _time.time() - _t0
            self.scheduler.step(
                metrics["score"] if self.scheduler_monitor == "val_score"
                else metrics["val_loss"])

            is_best = self.early_stopping.step(metrics["score"])
            if is_best and self.checkpoint_path is not None:
                save_checkpoint(
                    self.checkpoint_path, self.model, self.norm_stats,
                    self.machine, self.model_config,
                    best_score=self.early_stopping.best_score,
                    epoch=self.epoch,
                    optimizer_state=self.optimizer.state_dict(),
                    scheduler_state=self.scheduler.state_dict(),
                )

            log = {
                **self.log_extra,
                "epoch": self.epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_seconds": round(epoch_seconds, 1),
                "train_loss": train_loss,
                # CURRENT epoch score — never best-so-far, or ASHA can't
                # observe degradation and the scheduler is defeated.
                "val_score": metrics["score"],
                "val_loss": metrics["val_loss"],
                **{f"r2_{t}": v for t, v in metrics["r2"].items()},
                **{f"rmse_{t}": v for t, v in metrics["rmse"].items()},
            }
            print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else
                             f"{k}={v}" for k, v in log.items()))
            for lg in self.loggers:
                lg.log(log)
            if report_fn is not None:
                report_fn(log)

            if self.early_stopping.early_stop:
                break

        for lg in self.loggers:
            lg.close()
        return self.early_stopping.best_score if \
            self.early_stopping.best_score is not None else float("nan")


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """Seed RNGs. deterministic=True additionally forces deterministic cuDNN
    kernels — bitwise reproducibility at a substantial per-batch GPU cost
    (deterministic conv algorithms are often 1.5-3x slower). The original
    training script never enabled this (it seeded nothing); default False
    matches that behavior. Note: cudnn.benchmark stays False because input
    shapes vary per shot (variable W), which would trigger continuous
    re-benchmarking."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
