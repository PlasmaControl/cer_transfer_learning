#!/bin/bash
#SBATCH --job-name=cer_scaling_nstxu
#SBATCH --output=cer_scaling_nstxu_%a.out
#SBATCH --error=cer_scaling_nstxu_%a.err
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300          # signal 5 min before timeout (for requeue)
#SBATCH --requeue
#SBATCH --mail-type=all
#SBATCH --mail-user=ps9551@princeton.edu
#SBATCH --array=0-4
module purge
export PATH="$HOME/.pixi/bin:$PATH"
RUN="pixi run --frozen"

cd /scratch/gpfs/ps9551/passive_micer/cer_transfer            # repo root

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# Scaling curve on hop 2: fine-tune from the NSTX checkpoint at nested
# training-set sizes. Shared val list for selection; test set stays
# untouched until final evaluation.

SIZES=(n25 n50 n100 n200 full)
SZ=${SIZES[$SLURM_ARRAY_TASK_ID]}

SRC=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstx_ft.pt
OUT=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstxu_scaling_${SZ}.pt
mkdir -p "$(dirname "$OUT")"

if [ "$SZ" = "full" ]; then
    TRAIN_LIST=/scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstxu_train.txt
else
    TRAIN_LIST=/scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstxu_train_${SZ}.txt
fi

srun $RUN python -u finetune.py \
    --source-checkpoint "$SRC" \
    --machine nstxu \
    --checkpoint "$OUT" \
    --train-list "$TRAIN_LIST" \
    --val-list   /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstxu_val.txt \
    --transfer-readouts \
    --subseq-len 64 \
    --head-type mlp \
    --probe-epochs 30 \
    --backbone-lr 1e-5 \
    --late-backbone-lr 1e-4 \
    --head-lr 1e-3 \
    --no-mmap &
wait
