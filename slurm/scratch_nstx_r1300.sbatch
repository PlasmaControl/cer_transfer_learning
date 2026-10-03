#!/bin/bash
#SBATCH --job-name=cer_scratch_nstx_r1300
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

# From-scratch CONTROL at 1300 discharges: the equal-data
# counterpart to finetune_nstx_r1300 -- decides whether the
# label-efficiency claim belongs to transfer or to the
# architecture. Same recipe as scratch_nstx, subset list only.
module purge
export PATH="$HOME/.pixi/bin:$PATH"
RUN="pixi run --frozen"

cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs
LOG="slurm/logs/${SLURM_JOB_NAME}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"

# Control: NSTX trained from scratch (no D3D transfer). Comparisons:
#   vs hop 1 (nstx_ft.pt)      -> does D3D pretraining help at 10k shots?
#   as alt hop-2 source        -> scratch-NSTX -> NSTX-U isolates D3D's
#                                 contribution to the final NSTX-U model.
# Same architecture/config as the transfer arms.

CKPT=cer_ckpts/nstx_scratch_r1300.pt
mkdir -p "$(dirname "$CKPT")"
RESUME=""
[ -f "$CKPT" ] && RESUME="--resume"

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

srun $RUN python -u train.py \
    --machine nstx \
    --checkpoint "$CKPT" \
    --train-list splits/nstx_train_r1300.txt \
    --val-list   splits/nstx_val.txt \
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
