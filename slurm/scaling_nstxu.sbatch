#!/bin/bash
#SBATCH --job-name=cer_scaling_nstxu
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
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

cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs
LOG="slurm/logs/${SLURM_JOB_NAME}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# Scaling curve on hop 2: fine-tune from the NSTX checkpoint at nested
# training-set sizes. Shared val list for selection; test set stays
# untouched until final evaluation.

SIZES=(n25 n50 n100 n200 full)
SZ=${SIZES[$SLURM_ARRAY_TASK_ID]}

SRC=cer_ckpts/nstx_ft.pt
OUT=cer_ckpts/nstxu_scaling_${SZ}.pt
RESUME=""; [ -f "$OUT" ] && RESUME="--resume"   # requeue: continue from the saved checkpoint
mkdir -p "$(dirname "$OUT")"

if [ "$SZ" = "full" ]; then
    TRAIN_LIST=splits/nstxu_train.txt
else
    TRAIN_LIST=splits/nstxu_train_${SZ}.txt
fi

srun $RUN python -u finetune.py $RESUME \
    --source-checkpoint "$SRC" \
    --machine nstxu \
    --checkpoint "$OUT" \
    --train-list "$TRAIN_LIST" \
    --val-list   splits/nstxu_val.txt \
    --transfer-readouts \
    --subseq-len 64 \
    --head-type mlp \
    --probe-epochs 30 \
    --backbone-lr 1e-5 \
    --late-backbone-lr 1e-4 \
    --head-lr 1e-3 \
    --no-mmap &
wait
