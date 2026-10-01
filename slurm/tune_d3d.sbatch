#!/bin/bash
#SBATCH --job-name=cer_hpo
#SBATCH --output=cer_hpo.out
#SBATCH --error=cer_hpo.err
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=90:00:00
#SBATCH --signal=B:USR1@300          # signal 5 min before timeout (for requeue)
#SBATCH --requeue
#SBATCH --mail-type=all
#SBATCH --mail-user=ps9551@princeton.edu

module purge
# Option A: pixi (recommended; hermetic, avoids the torch/torchaudio ABI
# mismatch in ml-env). Install once: curl -fsSL https://pixi.sh/install.sh | bash
# then `pixi install` in the repo root.
export PATH="$HOME/.pixi/bin:$PATH"
RUN="pixi run --frozen"
# Option B: conda (uncomment, and comment out the two lines above)
# module load anaconda3/2023.3
# conda activate ml-env
# RUN=""

cd /scratch/gpfs/ps9551/passive_micer/cer_transfer            # repo root (contains train.py)

# HPO (8-GPU variant needs --gres=gpu:8 --cpus-per-task=48 --mem-per-cpu=8G;
# adjust the header above accordingly before large sweeps).

srun $RUN python -u tune.py \
    --machine d3d \
    --train-dir /scratch/gpfs/EKOLEMEN/ps9551/passive_micer/training_set_30 \
    --val-dir   /scratch/gpfs/EKOLEMEN/ps9551/passive_micer/test_set_30 \
    --name cer_hpo_v2 \
    --num-samples 200
