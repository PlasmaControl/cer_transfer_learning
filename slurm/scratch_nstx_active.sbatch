#!/bin/bash
#SBATCH --job-name=cer_scratch_nstx_active
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=90:00:00
#SBATCH --signal=B:USR1@300
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
trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# 48-chord ACTIVE lineage, scratch baseline (comparison anchor for the
# active-lineage hop 1). Canonical recipe, no experimental flags.
CKPT=cer_ckpts/nstx_active_scratch.pt
mkdir -p "$(dirname "$CKPT")"
RESUME=""
[ -f "$CKPT" ] && RESUME="--resume"

srun $RUN python -u train.py \
    --machine nstx_active \
    --checkpoint "$CKPT" \
    --train-list splits/nstx_active_train.txt \
    --val-list   splits/nstx_active_val.txt \
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
