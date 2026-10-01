#!/bin/bash
# run from anywhere, after cer_transfer.datasets.make_d3d_subsets
set -e
cd "$(dirname "$0")/.."   # repository root
for K in 500 1000 2000 4000 8000; do
  sbatch slurm/finetune_d3d_r$K.sbatch
  sbatch slurm/scratch_d3d_r$K.sbatch
done
