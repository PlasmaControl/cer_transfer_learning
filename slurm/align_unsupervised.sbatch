#!/bin/bash
#SBATCH --job-name=cer_align
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ps9551@princeton.edu
# Label-free choice of the target wavelength convention (no labels in the
# selection; labels only score the unmodified model output, raw R2). Array
# task 0: feature-distribution distance (MMD), task 1: model uncertainty.
#   sbatch slurm/align_unsupervised.sbatch
#   sbatch --export=ALL,CKPT=cer_ckpts/ladder_nstx_r1300.pt slurm/align_unsupervised.sbatch
module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs gallery
CKPT=${CKPT:-cer_ckpts/ladder_nstx_r650.pt}
SOURCE=${SOURCE:-nstx:splits/nstx_val.txt}
TARGET=${TARGET:-d3d_tangential:splits/d3d_val.txt}
N_SHOTS=${N_SHOTS:-20}
CRIT=(mmd sigma); C=${CRIT[${SLURM_ARRAY_TASK_ID:-0}]}
TAG=$(basename "$CKPT" .pt)_${TARGET%%:*}_$C
LOG="slurm/logs/${SLURM_JOB_NAME}_${TAG}"
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"
RUN="pixi run --frozen"

srun $RUN python -u -m cer_transfer.analysis.align_unsupervised --checkpoint "$CKPT" \
    --source $SOURCE --target $TARGET --n-shots $N_SHOTS --criterion $C \
    --out gallery/align_$TAG.csv
