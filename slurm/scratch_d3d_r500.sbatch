#!/bin/bash
#SBATCH --job-name=cer_scratch_d3d_r500
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
export PATH="$HOME/.pixi/bin:$PATH"
RUN="pixi run --frozen"

cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs
LOG="slurm/logs/${SLURM_JOB_NAME}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"

# From-scratch CONTROL at 500 D3D discharges: equal-data counterpart to
# finetune_d3d_r500. Same recipe as the D3D source pretrain
# (pretrain_transfer_source.sbatch), subset list only. Full-data
# counterpart = cer_ckpts/d3d_transfer_source.pt.

CKPT=cer_ckpts/d3d_scratch_r500.pt
LIST=splits/d3d_train_r500.txt
mkdir -p "$(dirname "$CKPT")"
[ -f "$LIST" ] || { echo "missing $LIST (run cer_transfer.datasets.make_d3d_subsets)"; exit 1; }
RESUME=""
[ -f "$CKPT" ] && RESUME="--resume"

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

srun $RUN python -u train.py \
    --machine d3d \
    --checkpoint "$CKPT" \
    --train-list "$LIST" \
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
