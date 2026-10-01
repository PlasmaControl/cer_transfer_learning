#!/bin/bash
#SBATCH --job-name=cer_eval_zdf
#SBATCH --output=cer_eval_zdf.out
#SBATCH --error=cer_eval_zdf.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Evaluate all --zero-dark-frames retrains against the pre-registered
# acceptance criteria:
#   per checkpoint: (a) full-val eval -> score + per-chord CSV
#                   (b) small-list --dump-preds -> dark-behavior check
# Compare CSVs/scores against the official counterparts afterwards.
module purge
cd /scratch/gpfs/ps9551/passive_micer/cer_transfer
mkdir -p zdf_eval
head -30 splits/nstx_val.txt > zdf_eval/darkcheck.txt

for ck in nstx_ft_r325_zdf nstx_ft_r650_zdf nstx_ft_r1300_zdf \
          nstx_ft_r2600_zdf nstx_ft_r5200_zdf nstx_ft_zdf; do
  echo "=== $ck ==="
  pixi run python -u eval_checkpoint.py \
      --checkpoint cer_ckpts/${ck}.pt --list splits/nstx_val.txt \
      --out zdf_eval/eval_${ck}.csv
  pixi run python -u eval_checkpoint.py \
      --checkpoint cer_ckpts/${ck}.pt --list zdf_eval/darkcheck.txt \
      --dump-preds zdf_eval/${ck}.npz
  pixi run python -u -m cer_transfer.analysis.check_zdf --list zdf_eval/darkcheck.txt \
      --preds zdf_eval/${ck}.npz
done
echo "acceptance: compare zdf_eval/eval_*.csv + scores vs the official"
echo "runs; dark-behavior verdicts printed above per checkpoint."
