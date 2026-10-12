#!/bin/bash
#SBATCH --job-name=cer_add_axis
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ps9551@princeton.edu
# Add the wavelength axis (nm per bin) to every DIII-D discharge file, rebuilt
# from the raw H5 files; 20 shards in parallel, CPU only. Existing keys stay
# unchanged; each file is replaced atomically after verification; files with
# a bin-count mismatch are left untouched and listed in the log.
#   DRY=1 sbatch slurm/add_wavelength_axis.sbatch     # check only, write nothing
module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs
LOG="slurm/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID:-0}"
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"
RAW=${RAW:-/scratch/gpfs/NSTX-U/passive_micer/raw}
DRYFLAG=""; [ "${DRY:-0}" = "1" ] && DRYFLAG="--dry-run"

pixi run --frozen python -u -m cer_transfer.datasets.add_wavelength_axis \
    --dirs training_set_30 test_set_30 --raw-dir "$RAW" \
    --shard ${SLURM_ARRAY_TASK_ID:-0} --n-shards ${SLURM_ARRAY_TASK_COUNT:-20} $DRYFLAG
