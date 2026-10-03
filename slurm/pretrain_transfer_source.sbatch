#!/bin/bash
#SBATCH --job-name=cer_pretrain_d3d_source
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
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

cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs
LOG="slurm/logs/${SLURM_JOB_NAME}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"

# Transfer-source pretrain: the D3D model whose trunk goes to NSTX/NSTX-U.
# Config: GroupNorm + mlp head (ablation winners) + stem/trunk backbone
# (feature_width) with machine-specific W-pooling; scheduler on val_score.

CKPT=cer_ckpts/d3d_transfer_source.pt
mkdir -p "$(dirname "$CKPT")"

# --resume is safe: checkpoint bundles best_score, so a requeued job can
# never overwrite a better model. Flag is a no-op if the file doesn't exist.
RESUME=""
[ -f "$CKPT" ] && RESUME="--resume"

# requeue on timeout signal
trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

srun $RUN python -u train.py \
    --machine d3d \
    --checkpoint "$CKPT" \
    --head-type mlp \
    --norm group \
    --feature-width 128 \
    --scheduler-monitor val_score \
    --hidden-dim 8 \
    --kernel-size 3 \
    --dropout 0.0488 \
    --batch-size 256 \
    --lr 0.00135 \
    --epochs 500 \
    --no-mmap \
    $RESUME &
wait
