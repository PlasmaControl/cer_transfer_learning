#!/bin/bash
#SBATCH --job-name=cer_agnostic_ft_nstx_r325
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ps9551@princeton.edu
# Supervised comparison for the agnostic model (step 4): fine-tune the complete
# DIII-D agnostic model on 325 NSTX discharges. Compare with the default
# architecture's fine-tune on the same list (finetune_nstx_r325.sbatch) to see
# what the agnostic design costs with labels available.
module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs
LOG="slurm/logs/${SLURM_JOB_NAME}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"
RUN="pixi run --frozen"

SRC=cer_ckpts/d3d_tangential_agnostic.pt
OUT=cer_ckpts/nstx_agnostic_ft_r325.pt
[ -f "$SRC" ] || { echo "missing source checkpoint $SRC"; exit 1; }
RESUME=""; [ -f "$OUT" ] && RESUME="--resume"   # requeue: continue from the saved checkpoint
trap 'echo "USR1: requeue"; scontrol requeue $SLURM_JOB_ID' USR1

srun $RUN python -u finetune.py $RESUME \
    --source-checkpoint "$SRC" --machine nstx --checkpoint "$OUT" \
    --train-list splits/nstx_train_r325.txt --val-list splits/nstx_val.txt \
    --batch-size 8 --accumulation-steps 32 --subseq-len 256 --num-workers 8 --no-mmap &
wait
