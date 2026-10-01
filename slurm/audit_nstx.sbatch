#!/bin/bash
#SBATCH --job-name=cer_audit
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs
LOG="slurm/logs/${SLURM_JOB_NAME}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"
mkdir -p audit
for m in nstx_scratch nstx_ft_r325 nstx_scratch_r325; do
  echo "=== $m ==="
  pixi run --frozen python -u eval_checkpoint.py --checkpoint cer_ckpts/$m.pt \
      --list splits/nstx_val.txt --dump-preds audit/$m.npz
  pixi run --frozen python -u -m cer_transfer.analysis.label_audit --dump audit/$m.npz --vbound 1e9 \
      --out audit/${m}_audit.csv | tee audit/${m}_audit.txt
done
