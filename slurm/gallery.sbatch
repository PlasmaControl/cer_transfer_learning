#!/bin/bash
#SBATCH --job-name=cer_gallery
#SBATCH --output=cer_gallery.out
#SBATCH --error=cer_gallery.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Hero-shot gallery: 20 random shots common to the legacy / active /
# passive validation lists (seeded, reproducible). Per shot, three
# one-shot evals (ft recipe on all arrays) and three directly named
# renders into gallery/<shot>/:
#   recon_legacy_<shot>, recon_active_<shot>,
#   recon_passive_vs_active_<shot>   (png + pdf each)
# Requires current cer_transfer.figures.recon_composite + cer_transfer.figures.gallery +
# eval_checkpoint.py in the repo root.
module purge
cd /scratch/gpfs/ps9551/passive_micer/cer_transfer
# clear leftovers from pre-rename runs so only current names remain
find gallery -name "fig[23]_*" -delete 2>/dev/null
find gallery -name "fig2_reconstruction.*" -delete 2>/dev/null
pixi run python -u -m cer_transfer.figures.gallery --n 20 --seed 42 --recipe ft --out gallery
