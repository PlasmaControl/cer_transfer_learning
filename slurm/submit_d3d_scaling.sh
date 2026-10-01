#!/bin/bash
# run from the repo root after cer_transfer.datasets.make_d3d_subsets
set -e
for K in 500 1000 2000 4000 8000; do
  sbatch finetune_d3d_r$K.sbatch
  sbatch scratch_d3d_r$K.sbatch
done
