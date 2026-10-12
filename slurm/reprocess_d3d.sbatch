#!/bin/bash
#SBATCH --job-name=cer_reprocess_d3d
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ps9551@princeton.edu
# Rebuild the DIII-D joblib spectrograms on one fixed wavelength grid from the
# raw H5 files (CPU only; 20 shards). Output root must differ from CER_DATA_ROOT.
#   OUT=/scratch/gpfs/NSTX-U/passive_micer/fixed RAW=/scratch/gpfs/NSTX-U/passive_micer/raw   (-> fixed/training_set_30, fixed/test_set_30)
#   sbatch --export=ALL,OUT=$OUT,RAW=$RAW slurm/reprocess_d3d.sbatch
#   EXTRA="--delete-raw" removes each shot's raw H5 once its new file is verified
#   (frees space as it runs); DIRS="training_set_30" to do one set only
#   afterwards: ln -s $CER_DATA_ROOT/<nstx folders> $OUT/   and set CER_DATA_ROOT=$OUT
module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..
mkdir -p slurm/logs
LOG="slurm/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env ] && source .env
: "${CER_DATA_ROOT:?}"; : "${OUT:?OUT=<new data root>}"; : "${RAW:?RAW=<raw H5 dir>}"
pixi run --frozen python -u -m cer_transfer.datasets.reprocess_d3d \
    --dirs ${DIRS:-training_set_30 test_set_30} --raw-dir "$RAW" --out-root "$OUT" \
    --shard "$SLURM_ARRAY_TASK_ID" --n-shards "$SLURM_ARRAY_TASK_COUNT" ${EXTRA:-}
