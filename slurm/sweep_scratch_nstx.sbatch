#!/bin/bash
#SBATCH --job-name=cer_sweep_nstx
#SBATCH --output=cer_sweep_nstx_%a.out
#SBATCH --error=cer_sweep_nstx_%a.err
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --mail-type=all
#SBATCH --mail-user=ps9551@princeton.edu

module purge
export PATH="$HOME/.pixi/bin:$PATH"
RUN="pixi run --frozen"

cd /scratch/gpfs/ps9551/passive_micer/cer_transfer

# NSTX scratch-baseline sweep. Hypothesis: per-shot batches on NSTX = ~2
# chunks -> noisy gradients cap the model well below the label ceiling
# (0.765 vs 0.898/0.992). Arms probe effective batch (via gradient
# accumulation) and LR around the D3D-tuned reference.
#   ref (already run): accum 1, lr 1.35e-3 -> 0.7651

ACCUM=(8   32  8    1       8)
LR=(   1.35e-3 1.35e-3 3e-3 4.5e-4 1.35e-3)
DROP=( 0.0488 0.0488 0.0488 0.0488 0.15)
A=${ACCUM[$SLURM_ARRAY_TASK_ID]}
L=${LR[$SLURM_ARRAY_TASK_ID]}
D=${DROP[$SLURM_ARRAY_TASK_ID]}

CKPT=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstx_sweep_a${A}_lr${L}_d${D}.pt
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
    --dropout $D \
    --batch-size 256 \
    --accumulation-steps $A \
    --lr $L \
    --epochs 500 \
    --no-mmap \
    $RESUME &
wait
