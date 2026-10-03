#!/bin/bash
#SBATCH --job-name=cer_eval_passive
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
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

# Passive evaluation: ACTIVE-trained checkpoints on PASSIVE spectrograms
# (same 48 flux-surface-aligned sightlines, same active-fit labels, same
# frames). Per-chord tables ARE the passive information-content result:
#   edge chords fine / core degraded  -> physics (expected)
#   uniform degradation incl. edge    -> low-count normalization problem
# Passive lists: full lists (train+val+test of passive files) are fine for
# a study eval, but the OFFICIAL numbers use only shots from the active
# TEST split to keep test discipline.
srun $RUN python -u eval_checkpoint.py \
    --checkpoint cer_ckpts/nstx_active_ft.pt \
    --list splits/nstx_passive_val.txt \
    --diagnose --out eval_nstx_passive.csv
srun $RUN python -u eval_checkpoint.py \
    --checkpoint cer_ckpts/nstxu_active_hop2.pt \
    --list splits/nstxu_passive_val.txt \
    --diagnose --out eval_nstxu_passive.csv
