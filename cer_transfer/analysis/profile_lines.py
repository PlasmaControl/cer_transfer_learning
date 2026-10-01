"""Line-profile the data pipeline: mmap vs sequential reads.

    pixi run python -u -m cer_transfer.analysis.profile_lines --machine d3d --n-batches 100
    pixi run python -u -m cer_transfer.analysis.profile_lines --machine d3d --n-batches 100 --no-mmap

Uses line_profiler's API on the exact functions of interest. Constraint:
line_profiler cannot trace DataLoader worker processes, so this harness runs
with num_workers=0 (data path executes in the main process). Absolute
throughput is therefore NOT training throughput (no prefetch overlap, no
worker parallelism); per-line relative cost is the meaningful output.

The model forward/backward is intentionally excluded — this isolates the
loader. Use --with-model to include a forward pass per batch for context.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from line_profiler import LineProfiler
from torch.utils.data import DataLoader

from cer_transfer.configs import ModelConfig, get_machine
from cer_transfer.data import (PerShotBatchSampler, Preprocessor, ShotDataset)
from cer_transfer.models import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--machine", default="d3d")
    p.add_argument("--train-dir", type=Path, default=None)
    p.add_argument("--n-batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--subseq-len", type=int, default=256)
    p.add_argument("--no-mmap", action="store_true")
    p.add_argument("--file-offset", type=int, default=0,
                   help="skip this many files (sorted order) — pick an "
                        "untouched window for a cold-cache profile")
    p.add_argument("--file-limit", type=int, default=0,
                   help="use at most this many files (0 = all)")
    p.add_argument("--with-model", action="store_true",
                   help="also run a forward pass per batch (fp32, CUDA if "
                        "available) so loader cost can be seen in context")
    p.add_argument("--hidden-dim", type=int, default=8)
    args = p.parse_args()

    machine = get_machine(args.machine)
    train_dir = Path(args.train_dir or machine.train_dir)
    files = sorted(train_dir.glob("*.joblib"))
    if args.file_offset:
        files = files[args.file_offset:]
    if args.file_limit:
        files = files[: args.file_limit]
    dataset = ShotDataset(files, machine, subseq_len=args.subseq_len,
                          mmap=not args.no_mmap)

    gen = torch.Generator().manual_seed(0)
    loader = DataLoader(
        dataset,
        batch_sampler=PerShotBatchSampler(dataset, args.batch_size, gen),
        num_workers=0,  # REQUIRED: line_profiler cannot trace workers
    )

    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.with_model:
        model = build_model(machine, ModelConfig(hidden_dim=args.hidden_dim,
                                                 norm="batch")).to(device)
        model.train()

    lp = LineProfiler()
    lp.add_function(ShotDataset.__getitem__)
    lp.add_function(ShotDataset._get_file)
    lp.add_function(Preprocessor.__call__)

    def run():
        it = iter(loader)
        for i in range(args.n_batches):
            try:
                inputs, targets, errors = next(it)
            except StopIteration:
                break
            if model is not None:
                with torch.no_grad():
                    model(inputs.to(device))
                if device == "cuda":
                    torch.cuda.synchronize()

    mode = "no-mmap (sequential full reads)" if args.no_mmap else "mmap"
    print(f"profiling {args.n_batches} batches, mode: {mode}, "
          f"files[{args.file_offset}:{args.file_offset + len(files)}], "
          f"num_workers=0\n")
    lp_wrapper = lp(run)
    lp_wrapper()
    lp.print_stats()


if __name__ == "__main__":
    main()
