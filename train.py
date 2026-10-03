"""Train from scratch on one machine.

    python train.py --machine d3d --checkpoint ckpts/d3d.pt \
        [--resume] [--lr 1e-3 --batch-size 256 ...]

Resuming (--resume) loads weights, norm stats, AND best_score from the
checkpoint, so a resumed run can never overwrite a better model with a
worse one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader

from cer_transfer.configs import ModelConfig, TrainConfig, data_path, get_machine
from cer_transfer.data import (
    FullShotSampler,
    PerShotBatchSampler,
    ShotDataset,
    compute_tail_weights,
    worker_init_fn,
)
from cer_transfer.logging_utils import CSVLogger, TensorBoardLogger
from cer_transfer.models import build_model
from cer_transfer.training import Trainer, load_checkpoint, set_random_seed


def resolve_files(dir_or_none, list_or_none, machine_dir, what):
    """A split is either a directory glob or an explicit file list (one path
    per line, relative to CER_DATA_ROOT or absolute; '#' comments allowed).
    Lists win over dirs."""
    if list_or_none is not None:
        lines = Path(list_or_none).read_text().splitlines()
        files = [
            data_path(ln.strip())
            for ln in lines
            if ln.strip() and not ln.strip().startswith("#")
        ]
        missing = [f for f in files if not f.exists()]
        if missing:
            import os

            root = os.environ.get("CER_DATA_ROOT")
            hint = (
                " (CER_DATA_ROOT is not set: relative list entries resolve "
                "against the current directory)"
                if root is None
                else f" (CER_DATA_ROOT={root})"
            )
            raise FileNotFoundError(
                f"{what} list {list_or_none}: {len(missing)} of {len(files)} files missing, "
                f"e.g. {missing[0]}{hint}"
            )
        return files
    if dir_or_none is None and machine_dir is None:
        raise SystemExit(
            f"no {what} data source: this machine has no default directory "
            f"in configs.py — pass --{what}-list (a cer_transfer.datasets."
            f"make_splits list; recommended, excludes curated-out shots) or "
            f"--{what}-dir."
        )
    d = data_path(dir_or_none or machine_dir)
    return sorted(d.glob("*.joblib"))


def make_loaders(
    machine,
    train_cfg,
    train_dir=None,
    val_dir=None,
    mmap=True,
    train_list=None,
    val_list=None,
    tail_oversample: float = 0.0,
    zero_dark_frames: bool = False,
    resample_w=None,
):
    """Build train/val DataLoaders from lists or directories for a machine."""
    train_files = resolve_files(train_dir, train_list, machine.train_dir, "train")
    val_files = resolve_files(val_dir, val_list, machine.val_dir, "val")
    train_set = ShotDataset(
        train_files,
        machine,
        subseq_len=train_cfg.subseq_len,
        mmap=mmap,
        max_label_relerr=getattr(train_cfg, "max_label_relerr", 0.0),
        zero_dark_frames=zero_dark_frames,
        resample_w=resample_w,
    )
    val_set = ShotDataset(
        val_files, machine, subseq_len=-1, mmap=mmap, resample_w=resample_w
    )
    gen = torch.Generator().manual_seed(train_cfg.seed)
    # persistent_workers: keep workers (and their cached file handles)
    # alive across epochs — without this, every epoch respawns workers.
    train_loader = DataLoader(
        train_set,
        batch_sampler=PerShotBatchSampler(
            train_set,
            train_cfg.batch_size,
            gen,
            shot_weights=(
                compute_tail_weights(train_set, tail_oversample)
                if tail_oversample > 0
                else None
            ),
        ),
        pin_memory=True,
        num_workers=train_cfg.num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=train_cfg.num_workers > 0,
        prefetch_factor=4 if train_cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=FullShotSampler(val_set),
        pin_memory=True,
        num_workers=train_cfg.num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=train_cfg.num_workers > 0,
        prefetch_factor=4 if train_cfg.num_workers > 0 else None,
    )
    return train_loader, val_loader


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--machine", default="d3d")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--force-resume",
        action="store_true",
        help="with --resume: continue even if the checkpoint is marked "
        "finished (early-stopped or epoch limit reached)",
    )
    p.add_argument("--train-dir", type=Path, default=None)
    p.add_argument("--val-dir", type=Path, default=None)
    p.add_argument(
        "--train-list",
        type=Path,
        default=None,
        help="explicit file list (from cer_transfer.datasets.make_splits); overrides "
        "--train-dir",
    )
    p.add_argument("--val-list", type=Path, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--norm", choices=["group", "batch"], default="group")
    p.add_argument("--head-type", choices=["linear", "mlp", "full"], default="mlp")
    p.add_argument(
        "--feature-width",
        type=int,
        default=None,
        help="enable transfer-ready stem/trunk backbone with this "
        "fixed feature width (e.g. 128); None = legacy",
    )
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument(
        "--augment",
        action="store_true",
        help="enable SpecAugment (was present but commented out "
        "in the original training loop; off reproduces old "
        "runs)",
    )
    p.add_argument("--wavelength-mask", type=int, default=200)
    p.add_argument("--time-mask", type=int, default=120)
    p.add_argument(
        "--tensorboard",
        action="store_true",
        help="also log to TensorBoard next to the checkpoint",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--subseq-len",
        type=int,
        default=256,
        help="training subsequence length; shots with fewer valid "
        "frames contribute nothing — set below the dataset's "
        "typical end_index (NSTX-U: median 144)",
    )
    p.add_argument(
        "--scheduler-monitor",
        choices=["val_loss", "val_score"],
        default="val_loss",
        help="metric ReduceLROnPlateau watches; val_loss is the "
        "legacy default, val_score recommended for new runs",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="force deterministic cuDNN kernels (bitwise "
        "reproducibility; 1.5-3x slower GPU step)",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="log data-wait / H2D / compute split every 200 batches",
    )
    p.add_argument(
        "--agnostic",
        action="store_true",
        help="machine-agnostic model: shared per-chord stem and shared readout, "
        "no parameter depends on the number of chords (zero-shot transfer); "
        "requires --feature-width",
    )
    p.add_argument(
        "--chord-attention",
        action="store_true",
        help="agnostic mode: attention over chords (per frame) before the readout",
    )
    p.add_argument(
        "--resample-w",
        type=int,
        default=None,
        help="resample the wavelength axis to this many bins in preprocessing",
    )
    p.add_argument(
        "--moment-features",
        action="store_true",
        help="feed classical spectral moments (amplitude, "
        "centroid, width; native-W, per chord/frame) into "
        "each chord's readout",
    )
    p.add_argument(
        "--max-label-relerr",
        type=float,
        default=0.0,
        metavar="R",
        help="TRAINING label QC: mask labels with "
        "target_error > R*|target| (0 = off; try 1.0)",
    )
    p.add_argument(
        "--tail-oversample",
        type=float,
        default=0.0,
        metavar="GAMMA",
        help="oversample high-|vtor| shots: weight = "
        "(p90|vtor|/median)^GAMMA, sampled with replacement. "
        "0 = off; 1-2 = moderate-strong. Training "
        "distribution only; val untouched.",
    )
    p.add_argument(
        "--per-chord-norm",
        action="store_true",
        help="normalize targets per chord: removes cross-chord "
        "profile variance from the loss (anti-climatology)",
    )
    p.add_argument(
        "--noise-aware-nll",
        action="store_true",
        help="NLL with total var = sigma_model^2 + sigma_label^2 "
        "(uses target_error; stops fitting label noise)",
    )
    p.add_argument(
        "--zero-dark-frames",
        action="store_true",
        help="add physics pseudo-labels (Ti=vtor=0) on dark "
        "frames (no emission on any chord)",
    )
    p.add_argument(
        "--no-mmap",
        action="store_true",
        help="read whole shots sequentially instead of mmap page "
        "faults (often faster on GPFS); ~300MB/shot per "
        "worker in RAM",
    )
    args = p.parse_args()

    set_random_seed(args.seed, deterministic=args.deterministic)
    machine = get_machine(args.machine)
    cli_cfg = ModelConfig(
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        norm=args.norm,
        head_type=args.head_type,
        feature_width=args.feature_width,
        moment_features=args.moment_features,
        agnostic=args.agnostic,
        chord_attention=args.chord_attention,
        resample_w=args.resample_w,
    )

    # On --resume the checkpoint's stored model_config is authoritative:
    # the architecture must match the saved weights, and requeues must not
    # depend on repeating every CLI flag correctly. CLI mismatches are
    # reported, not applied.
    model_cfg = cli_cfg
    if args.resume and args.checkpoint.exists():
        ckpt_cfg = ModelConfig(**load_checkpoint(args.checkpoint)["model_config"])
        diffs = {
            f: (getattr(cli_cfg, f), getattr(ckpt_cfg, f))
            for f in ckpt_cfg.__dataclass_fields__
            if getattr(cli_cfg, f) != getattr(ckpt_cfg, f)
        }
        if diffs:
            print(
                "resume: using the checkpoint's architecture; ignoring "
                "differing CLI flags: "
                + ", ".join(
                    f"{k} (cli={c!r}, ckpt={s!r})" for k, (c, s) in diffs.items()
                )
            )
        model_cfg = ckpt_cfg

    # On resume, the checkpoint's stored architecture is authoritative —
    # rebuilding from CLI flags invites silent/opaque key mismatches when a
    # flag is forgotten. CLI-vs-checkpoint differences are reported.
    if args.resume and args.checkpoint.exists():
        stored = ModelConfig(**load_checkpoint(args.checkpoint)["model_config"])
        if stored != model_cfg:
            diffs = {
                f: (getattr(model_cfg, f), getattr(stored, f))
                for f in stored.__dataclass_fields__
                if getattr(stored, f) != getattr(model_cfg, f)
            }
            print(
                f"resume: using checkpoint architecture; CLI differs on "
                f"{diffs} (cli, checkpoint)"
            )
            model_cfg = stored
    train_cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        accumulation_steps=args.accumulation_steps,
        seed=args.seed,
        scheduler_monitor=args.scheduler_monitor,
        per_chord_norm=args.per_chord_norm,
        max_label_relerr=args.max_label_relerr,
        noise_aware_nll=args.noise_aware_nll,
        num_workers=args.num_workers,
        subseq_len=args.subseq_len,
        profile=args.profile,
        augment=args.augment,
        wavelength_mask_param=args.wavelength_mask,
        time_mask_param=args.time_mask,
    )

    model = build_model(machine, model_cfg)

    norm_stats, best_score, start_epoch = None, None, 0
    optimizer_state, scheduler_state = None, None
    if args.resume and args.checkpoint.exists():
        ckpt = load_checkpoint(args.checkpoint)
        if ckpt.get("finished") and not args.force_resume:
            print(
                f"{args.checkpoint} finished ({ckpt['finished']}, best "
                f"{ckpt['best_score']} at epoch {ckpt['epoch']}); nothing to "
                "resume. Use --force-resume to continue anyway."
            )
            return
        model.load_state_dict(ckpt["model_state"])
        norm_stats = ckpt["norm_stats"]
        best_score = ckpt["best_score"]
        start_epoch = ckpt["epoch"] + 1  # stats-only ckpt has epoch=-1 -> 0
        optimizer_state = ckpt.get("optimizer_state")
        scheduler_state = ckpt.get("scheduler_state")
        best_str = (
            "none (stats-only checkpoint)"
            if best_score is None
            else f"{best_score:.4f}"
        )
        print(
            f"resumed from {args.checkpoint} "
            f"(best epoch {ckpt['epoch']}, best {best_str}, "
            f"continuing at epoch {start_epoch})"
        )

    train_loader, val_loader = make_loaders(
        machine,
        train_cfg,
        args.train_dir,
        args.val_dir,
        mmap=not args.no_mmap,
        train_list=args.train_list,
        val_list=args.val_list,
        tail_oversample=args.tail_oversample,
        zero_dark_frames=args.zero_dark_frames,
        resample_w=model_cfg.resample_w,
    )

    loggers = [CSVLogger(args.checkpoint.with_suffix(".metrics.csv"))]
    if args.tensorboard:
        loggers.append(
            TensorBoardLogger(args.checkpoint.parent / "tb" / args.checkpoint.stem)
        )
    trainer = Trainer(
        model,
        machine,
        model_cfg,
        train_cfg,
        optimizer=optim.AdamW(
            model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_path=args.checkpoint,
        norm_stats=norm_stats,
        initial_best_score=best_score,
        start_epoch=start_epoch,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        loggers=loggers,
    )
    best = trainer.fit(train_loader, val_loader)
    print(f"best val score: {best:.4f}")


if __name__ == "__main__":
    main()
