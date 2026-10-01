#!/bin/bash
#SBATCH --job-name=cer_head_ablation
#SBATCH --output=cer_head_%a.out
#SBATCH --error=cer_head_%a.err
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
#SBATCH --array=0-2
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

# Head-capacity ablation on DIII-D: linear vs mlp vs full, same seed and
# hyperparameters. (Completed via individual runs; kept for reproducibility.)

HEAD_TYPES=(linear mlp full)
HEAD=${HEAD_TYPES[$SLURM_ARRAY_TASK_ID]}

CKPT=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/d3d_ablation_${HEAD}.pt
mkdir -p "$(dirname "$CKPT")"
RESUME=""
[ -f "$CKPT" ] && RESUME="--resume"

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

srun $RUN python -u train.py \
    --machine d3d \
    --checkpoint "$CKPT" \
    --head-type "$HEAD" \
    --norm batch \
    --hidden-dim 8 \
    --kernel-size 3 \
    --dropout 0.0488 \
    --batch-size 256 \
    --lr 0.00135 \
    --epochs 500 \
    --seed 42 \
    --no-mmap \
    $RESUME &
wait
