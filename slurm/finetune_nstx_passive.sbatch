#!/bin/bash
#SBATCH --job-name=cer_ft_hop1_nstx_passive
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
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"
trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# 48-chord ACTIVE lineage hop 1: D3D -> nstx_passive. Reuses the EXISTING
# transfer source (trunk is machine-agnostic; stem + 48 readouts fresh).
# Data-rich recipe (matches legacy hop 1).
SRC=cer_ckpts/d3d_transfer_source.pt
OUT=cer_ckpts/nstx_passive_ft.pt
RESUME=""; [ -f "$OUT" ] && RESUME="--resume"   # requeue: continue from the saved checkpoint
mkdir -p "$(dirname "$OUT")"

srun $RUN python -u finetune.py $RESUME \
    --source-checkpoint "$SRC" \
    --machine nstx_passive \
    --checkpoint "$OUT" \
    --train-list splits/nstx_passive_train.txt \
    --val-list   splits/nstx_passive_val.txt \
    --head-type mlp \
    --subseq-len 64 \
    --probe-epochs 5 \
    --probe-lr 1e-3 \
    --backbone-lr 1e-4 \
    --late-backbone-lr 3e-4 \
    --head-lr 1e-3 \
    --epochs 300 \
    --no-mmap &
wait
