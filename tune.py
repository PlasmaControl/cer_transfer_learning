"""Hyperparameter search with Ray Tune.

Fixes vs the old version:
  - seeding happens INSIDE the objective (trials are separate processes),
    and the train/val file split is deterministic — every trial sees the
    same data, so configs are actually comparable.
  - the CURRENT epoch's val score is reported, not best-so-far, so ASHA
    can observe degradation and prune.
  - norm stats are computed inside the trial in the trial's own directory,
    never read from a shared CWD file.
  - final analysis queries the metric name that is actually reported.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import ray
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch import optim
from torch.utils.data import DataLoader

from cer_transfer.configs import ModelConfig, TrainConfig, get_machine, data_path
from cer_transfer.logging_utils import CSVLogger
from cer_transfer.data import (FullShotSampler, PerShotBatchSampler,
                               ShotDataset, worker_init_fn)
from cer_transfer.models import build_model
from cer_transfer.training import Trainer, set_random_seed

METRIC = "val_score"
SEED = 42
N_TRAIN_FILES = 500
N_VAL_FILES = 50


def objective(config, machine_name: str, train_dir: str, val_dir: str):
    set_random_seed(SEED)  # per-trial process
    machine = get_machine(machine_name)

    train_files = sorted(data_path(train_dir).glob("*.joblib"))
    random.Random(SEED).shuffle(train_files)          # same split every trial
    val_files = sorted(data_path(val_dir).glob("*.joblib"))[:N_VAL_FILES]

    train_set = ShotDataset(train_files[:N_TRAIN_FILES], machine,
                            subseq_len=256)
    val_set = ShotDataset(val_files, machine, subseq_len=-1)

    gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_set,
        batch_sampler=PerShotBatchSampler(train_set, config["batch_size"], gen),
        pin_memory=True, num_workers=6, worker_init_fn=worker_init_fn,
        persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_set, batch_sampler=FullShotSampler(val_set),
        pin_memory=True, num_workers=6, worker_init_fn=worker_init_fn,
        persistent_workers=True, prefetch_factor=4,
    )

    model_cfg = ModelConfig(
        hidden_dim=config["hidden_dim"], kernel_size=config["kernel_size"],
        dropout=config["dropout"], bias=config["bias"],
    )
    train_cfg = TrainConfig(lr=config["lr"], batch_size=config["batch_size"],
                            num_epochs=500, patience=50, seed=SEED)

    model = build_model(machine, model_cfg)
    trainer = Trainer(
        model, machine, model_cfg, train_cfg,
        optimizer=optim.AdamW(model.parameters(), lr=config["lr"],
                              weight_decay=0.05),
        device="cuda",
        checkpoint_path=Path("best.pt"),  # trial's own working dir
        loggers=[CSVLogger(Path("metrics.csv"))],
    )
    trainer.fit(train_loader, val_loader,
                report_fn=lambda log: ray.tune.report(log))


class NanStopper(tune.Stopper):
    def __init__(self, metric: str):
        self.metric = metric

    def __call__(self, trial_id, result):
        import math
        v = result.get(self.metric)
        return v is not None and isinstance(v, float) and math.isnan(v)

    def stop_all(self):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--machine", default="d3d")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--name", default="cer_hpo")
    p.add_argument("--num-samples", type=int, default=200)
    args = p.parse_args()

    ray.init()
    search_space = {
        "kernel_size": tune.choice([1, 3]),
        "bias": tune.choice([False, True]),
        "batch_size": tune.choice([32, 64, 128, 256, 512]),
        "hidden_dim": tune.choice([2, 4, 8, 16, 32]),
        "dropout": tune.uniform(0.0, 0.7),
        "lr": tune.loguniform(1e-5, 1e-2),
    }
    scheduler = ASHAScheduler(metric=METRIC, mode="max", max_t=500,
                              grace_period=25, reduction_factor=2)

    result = tune.run(
        tune.with_parameters(objective, machine_name=args.machine,
                             train_dir=args.train_dir, val_dir=args.val_dir),
        resources_per_trial={"cpu": 6, "gpu": 1},
        name=args.name, config=search_space,
        num_samples=args.num_samples, scheduler=scheduler,
        resume="AUTO+ERRORED",
        stop=NanStopper(METRIC),
    )

    best_trial = result.get_best_trial(METRIC, "max", "last")
    best_ckpt = result.get_best_checkpoint(best_trial, metric=METRIC,
                                           mode="max")
    print(f"best config: {best_trial.config}")
    print(f"best {METRIC}: {best_trial.last_result[METRIC]}")
    print(f"best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
