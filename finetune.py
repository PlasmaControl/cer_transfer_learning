"""Transfer a trained backbone to a new machine (LP-FT schedule).

    python finetune.py --source-checkpoint ckpts/d3d.pt --machine nstx \
        --checkpoint ckpts/nstx.pt --train-dir ... --val-dir ...

Phase 1 (linear probe): backbone frozen, only the fresh head trains.
Phase 2 (fine-tune):     everything unfrozen with discriminative LRs
                         (early backbone << late backbone << head).

Target-machine norm stats are ALWAYS recomputed from target train data —
source stats (e.g. DIII-D vtor scale, ~10x smaller than NSTX) are never
reused. If the source checkpoint used BatchNorm, running stats are dropped
and re-estimated on target data (AdaBN); GroupNorm is unaffected.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import optim

from cer_transfer.configs import FinetuneConfig, ModelConfig, get_machine
from cer_transfer.logging_utils import CSVLogger
from cer_transfer.models import (build_model, transfer_backbone,
                                 transfer_full_head, transfer_head_trunk)
from cer_transfer.training import (Trainer, compute_norm_stats,
                                   load_checkpoint, set_random_seed)
from train import make_loaders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-checkpoint", type=Path, required=True)
    p.add_argument("--machine", required=True, help="target machine, e.g. nstx")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="output checkpoint for the fine-tuned model")
    p.add_argument("--train-dir", type=Path, default=None)
    p.add_argument("--val-dir", type=Path, default=None)
    p.add_argument("--train-list", type=Path, default=None,
                   help="explicit file list (from cer_transfer.datasets.make_splits); overrides "
                        "--train-dir — use the train_n<K>.txt lists for the "
                        "scaling curve")
    p.add_argument("--val-list", type=Path, default=None)
    p.add_argument("--probe-epochs", type=int, default=30)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--late-backbone-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--head-type", choices=["linear", "mlp", "full"],
                   default=None,
                   help="override head capacity for the target machine "
                        "(default: same as source); backbone transfer is "
                        "unaffected since the head is fresh anyway")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--subseq-len", type=int, default=256,
                   help="training subsequence length; must be <= typical "
                        "end_index of the target dataset (NSTX-U median: "
                        "144 -> use 128 or 64)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--transfer-readouts", action="store_true",
                   help="warm-start the ENTIRE head incl. per-chord readouts "
                        "from the source (requires equal chord count and "
                        "head type, e.g. nstx -> nstxu)")
    p.add_argument("--moment-features", action="store_true",
                   help="enable moment-feature readouts on the TARGET model "
                        "(fresh readouts; incompatible with "
                        "--transfer-readouts from a source without them)")
    p.add_argument("--max-label-relerr", type=float, default=0.0,
                   metavar="R", help="training label QC (see train.py)")
    p.add_argument("--tail-oversample", type=float, default=0.0,
                   metavar="GAMMA",
                   help="oversample high-|vtor| shots (see train.py)")
    p.add_argument("--per-chord-norm", action="store_true",
                   help="normalize targets per chord: removes cross-chord "
                        "profile variance from the loss (anti-climatology)")
    p.add_argument("--noise-aware-nll", action="store_true",
                   help="NLL with total var = sigma_model^2 + sigma_label^2 "
                        "(uses target_error; stops fitting label noise)")
    p.add_argument("--zero-dark-frames", action="store_true",
                   help="add physics pseudo-labels (Ti=vtor=0) on dark "
                        "frames (no emission on any chord)")
    p.add_argument("--no-mmap", action="store_true",
                   help="sequential full-shot reads (2x faster on GPFS)")
    p.add_argument("--scheduler-monitor",
                   choices=["val_loss", "val_score"], default="val_score",
                   help="metric ReduceLROnPlateau watches (default "
                        "val_score for finetuning — new runs, no legacy "
                        "comparability constraint)")
    args = p.parse_args()

    set_random_seed(args.seed)  # deterministic kernels off (see training.py)
    target = get_machine(args.machine)

    ckpt = load_checkpoint(args.source_checkpoint)
    cfg_dict = dict(ckpt["model_config"])
    if args.head_type is not None:
        cfg_dict["head_type"] = args.head_type
    if args.moment_features:
        cfg_dict["moment_features"] = True
    model_cfg = ModelConfig(**cfg_dict)
    src_machine = ckpt["machine"]["name"]
    if ckpt["machine"]["n_raw_channels"] != target.n_raw_channels and \
            model_cfg.norm == "batch":
        print("note: BatchNorm running stats will be re-estimated on "
              f"{target.name} data")

    src_in = (ckpt["machine"]["n_raw_channels"]
              if ckpt["machine"]["input_channel_indices"] is None
              else len(ckpt["machine"]["input_channel_indices"]))
    channel_mismatch = src_in != target.n_input_channels
    stem_arch = model_cfg.feature_width is not None
    if channel_mismatch and not stem_arch:
        raise SystemExit(
            f"source backbone expects {src_in} input channels, "
            f"{target.name} provides {target.n_input_channels}, and the "
            "source was trained in legacy mode (feature_width=None), so the "
            "trunk cannot be separated from channel identity. Retrain the "
            "source with feature_width set (stem/trunk mode)."
        )

    model = build_model(target, model_cfg)
    transfer_backbone(model, ckpt["backbone_state"],
                      reset_norm_running_stats=True,
                      trunk_only=stem_arch)
    parts = ["trunk" if stem_arch else "backbone"]
    if args.transfer_readouts:
        transfer_full_head(model.head, ckpt["head_state"],
                           reset_norm_running_stats=True)
        parts.append("full head (readouts warm-started)")
        fresh_desc = "stem re-initialized" if stem_arch else "nothing fresh"
    elif stem_arch and model_cfg.head_type in ("linear", "mlp"):
        transfer_head_trunk(model.head, ckpt["head_state"],
                            reset_norm_running_stats=True)
        parts.append("head trunk")
        fresh_desc = (f"stem and {target.n_chords}-chord readouts "
                      "re-initialized")
    else:
        fresh_desc = f"{target.n_chords}-chord readouts re-initialized"
    print(f"transferred {' + '.join(parts)} {src_machine} -> {target.name}; "
          f"{fresh_desc}")

    ft_cfg = FinetuneConfig(
        batch_size=args.batch_size, num_epochs=args.epochs, seed=args.seed,
        subseq_len=args.subseq_len,
        probe_epochs=args.probe_epochs, probe_lr=args.probe_lr,
        backbone_lr=args.backbone_lr, late_backbone_lr=args.late_backbone_lr,
        head_lr=args.head_lr,
        scheduler_monitor=args.scheduler_monitor,
        per_chord_norm=args.per_chord_norm,
        max_label_relerr=args.max_label_relerr,
        noise_aware_nll=args.noise_aware_nll,
    )
    train_loader, val_loader = make_loaders(target, ft_cfg,
                                            args.train_dir, args.val_dir,
                                            mmap=not args.no_mmap,
                                            train_list=args.train_list,
                                            val_list=args.val_list,
                                            tail_oversample=args.tail_oversample,
                                            zero_dark_frames=args.zero_dark_frames)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Target stats from target data — never inherited from source.
    norm_stats = {k: v.to(device) for k, v in
                  compute_norm_stats(train_loader.dataset,
                                     tuple(target.targets)).items()}

    # ---- Phase 1: probe — freeze transferred parts; train head + fresh stem
    for prm in model.backbone.parameters():
        prm.requires_grad_(False)
    for prm in model.backbone.stem_parameters():
        prm.requires_grad_(True)
    probe_cfg = FinetuneConfig(**{**ft_cfg.__dict__,
                                  "num_epochs": ft_cfg.probe_epochs,
                                  "patience": ft_cfg.probe_epochs})
    csv_logger = CSVLogger(args.checkpoint.with_suffix(".metrics.csv"))
    probe = Trainer(
        model, target, model_cfg, probe_cfg,
        optimizer=optim.AdamW(
            list(model.head.parameters()) + model.backbone.stem_parameters(),
            lr=ft_cfg.probe_lr, weight_decay=ft_cfg.weight_decay),
        device=device, checkpoint_path=args.checkpoint, norm_stats=norm_stats,
        loggers=[csv_logger], log_extra={"phase": "probe"},
    )
    print(f"--- phase 1: linear probe ({ft_cfg.probe_epochs} epochs) ---")
    probe_best = probe.fit(train_loader, val_loader)

    # ---- Phase 2: full fine-tune with discriminative LRs ----------------
    for prm in model.backbone.parameters():
        prm.requires_grad_(True)
    early, late = model.backbone.param_groups()
    fresh = list(model.head.parameters()) + model.backbone.stem_parameters()
    param_groups = [
        {"params": early, "lr": ft_cfg.backbone_lr},
        {"params": late, "lr": ft_cfg.late_backbone_lr},
        {"params": fresh, "lr": ft_cfg.head_lr},
    ]
    ft = Trainer(
        model, target, model_cfg, ft_cfg,
        optimizer=optim.AdamW(param_groups,
                              weight_decay=ft_cfg.weight_decay),
        device=device, checkpoint_path=args.checkpoint, norm_stats=norm_stats,
        initial_best_score=probe_best,  # never regress below the probe
        loggers=[csv_logger], log_extra={"phase": "finetune"},
    )
    print("--- phase 2: full fine-tune ---")
    best = ft.fit(train_loader, val_loader)
    print(f"best val score: {best:.4f} (probe: {probe_best:.4f})")


if __name__ == "__main__":
    main()
