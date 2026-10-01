#!/bin/bash
#SBATCH --job-name=cer_scratch_nstx_active_zdf
#SBATCH --output=cer_scratch_nstx_active_zdf.out
#SBATCH --error=cer_scratch_nstx_active_zdf.err
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

# ZDF variant of the official recipe: only change is
# --zero-dark-frames (criterion: cer_transfer/dark.py).
# NSTX lists certified; NSTX-U: run cer_transfer.analysis.check_dark_frames once
# on that list before first use.
module purge
export PATH="$HOME/.pixi/bin:$PATH"
RUN="pixi run --frozen"

cd /scratch/gpfs/ps9551/passive_micer/cer_transfer

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# 48-chord ACTIVE lineage, scratch baseline (comparison anchor for the
# active-lineage hop 1). Canonical recipe, no experimental flags.
CKPT=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstx_active_scratch.pt
mkdir -p "$(dirname "$CKPT")"
RESUME=""
[ -f "$CKPT" ] && RESUME="--resume"

srun $RUN python -u train.py \
    --machine nstx_active \
    --checkpoint "$CKPT" \
    --train-list /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstx_active_train.txt \
    --val-list   /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstx_active_val.txt \
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
    --zero-dark-frames \
    --no-mmap \
    $RESUME &
wait
