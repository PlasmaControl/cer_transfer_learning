#!/bin/bash
#SBATCH --job-name=cer_agnostic_eval_nstx
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
# Zero-shot evaluation (step 2): the DIII-D agnostic model on NSTX validation
# and test data, without any NSTX training; the NSTX fits serve only as
# evaluation. Writes prediction dumps and the affine diagnostic (slope/offset
# per target and the R2 remaining after that correction).
module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs gallery figs
LOG="slurm/logs/${SLURM_JOB_NAME}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"
RUN="pixi run --frozen"

CKPT=cer_ckpts/d3d_tangential_agnostic.pt
for split in val test; do
  echo "=== NSTX $split, zero-shot"
  srun $RUN python -u eval_checkpoint.py --checkpoint "$CKPT" --machine nstx \
      --list splits/nstx_$split.txt --dump-preds gallery/nstx_${split}_zero_shot.npz
  srun $RUN python -u -m cer_transfer.analysis.affine_fit \
      --preds gallery/nstx_${split}_zero_shot.npz --plot figs/affine_zero_shot_$split
done
echo "=== DIII-D tangential val (sanity: the agnostic model on its own machine)"
srun $RUN python -u eval_checkpoint.py --checkpoint "$CKPT" --list splits/d3d_val.txt --limit 200
