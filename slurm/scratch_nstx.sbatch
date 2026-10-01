#!/bin/bash
#SBATCH --job-name=cer_scratch_nstx
#SBATCH --output=cer_scratch_nstx.out
#SBATCH --error=cer_scratch_nstx.err
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

cd /scratch/gpfs/ps9551/passive_micer/cer_transfer            # repo root

# Control: NSTX trained from scratch (no D3D transfer). Comparisons:
#   vs hop 1 (nstx_ft.pt)      -> does D3D pretraining help at 10k shots?
#   as alt hop-2 source        -> scratch-NSTX -> NSTX-U isolates D3D's
#                                 contribution to the final NSTX-U model.
# Same architecture/config as the transfer arms.

CKPT=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstx_scratch.pt
mkdir -p "$(dirname "$CKPT")"
RESUME=""
[ -f "$CKPT" ] && RESUME="--resume"

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

srun $RUN python -u train.py \
    --machine nstx \
    --checkpoint "$CKPT" \
    --train-list /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstx_train.txt \
    --val-list   /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstx_val.txt \
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
