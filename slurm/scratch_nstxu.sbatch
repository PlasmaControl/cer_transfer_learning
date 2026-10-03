#!/bin/bash
#SBATCH --job-name=cer_scratch_nstxu
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300          # signal 5 min before timeout (for requeue)
#SBATCH --requeue
#SBATCH --mail-type=all
#SBATCH --mail-user=ps9551@princeton.edu

module purge
export PATH="$HOME/.pixi/bin:$PATH"
RUN="pixi run --frozen"

cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs
LOG="slurm/logs/${SLURM_JOB_NAME}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# Control: NSTX-U trained from scratch (no transfer). The baseline every
# transfer arm must beat. Same architecture/config as the transfer arms.

CKPT=cer_ckpts/nstxu_scratch.pt
mkdir -p "$(dirname "$CKPT")"
RESUME=""
[ -f "$CKPT" ] && RESUME="--resume"

srun $RUN python -u train.py \
    --machine nstxu \
    --checkpoint "$CKPT" \
    --train-list splits/nstxu_train.txt \
    --val-list   splits/nstxu_val.txt \
    --head-type mlp \
    --norm group \
    --feature-width 128 \
    --scheduler-monitor val_score \
    --subseq-len 64 \
    --hidden-dim 8 \
    --kernel-size 3 \
    --dropout 0.0488 \
    --batch-size 256 \
    --lr 0.00135 \
    --epochs 500 \
    --no-mmap \
    $RESUME &
wait
