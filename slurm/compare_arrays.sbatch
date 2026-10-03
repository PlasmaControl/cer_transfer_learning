#!/bin/bash
#SBATCH --job-name=cer_compare_arrays
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00

module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs
LOG="slurm/logs/${SLURM_JOB_NAME}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"
# Uses train+val lists; test lists untouched.
for M in nstx nstxu; do
  cat splits/${M}_active_train.txt  splits/${M}_active_val.txt  > /tmp/${M}_a.txt
  cat splits/${M}_passive_train.txt splits/${M}_passive_val.txt > /tmp/${M}_p.txt
  pixi run --frozen python -u -m cer_transfer.analysis.compare_arrays \
      --active-list /tmp/${M}_a.txt \
      --passive-list /tmp/${M}_p.txt \
      --out figs/compare_${M}
done
