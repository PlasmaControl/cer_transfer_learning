#!/bin/bash
#SBATCH --job-name=cer_axis_sweep
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ps9551@princeton.edu
# Wavelength-axis convention of a target machine from the data: evaluate an
# agnostic checkpoint on target discharges with the axis reversed / stretched
# and report the affine R2 per setting. Defaults: the quick NSTX model on
# DIII-D tangential chords. Override with --export=ALL,CKPT=...,MACHINE=...,LIST=...
module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs gallery
CKPT=${CKPT:-cer_ckpts/nstx_agnostic_quick.pt}
MACHINE=${MACHINE:-d3d_tangential}
LIST=${LIST:-splits/d3d_val.txt}
LIMIT=${LIMIT:-100}
TAG=$(basename "$CKPT" .pt)_on_$MACHINE
LOG="slurm/logs/${SLURM_JOB_NAME}_${TAG}"      # one log per checkpoint/machine
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"
RUN="pixi run --frozen"


srun $RUN python -u -m cer_transfer.analysis.axis_sweep --checkpoint "$CKPT" --machine $MACHINE \
    --list $LIST --limit $LIMIT --scales 0.25 0.33 0.5 0.7 1 1.4 2 3 4 --out gallery/axis_sweep_$TAG.csv
