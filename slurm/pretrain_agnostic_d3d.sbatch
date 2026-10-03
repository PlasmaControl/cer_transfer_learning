#!/bin/bash
#SBATCH --job-name=cer_agnostic_d3d
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ps9551@princeton.edu
# Machine-agnostic DIII-D pre-training (tangential chords only) (zero-shot transfer study, step 1):
# shared per-chord stem, chord-averaged trunk, shared readout; no parameter
# depends on the number of chords. The per-chord stem makes activations scale
# with the chord count, hence the small batch with gradient accumulation
# (effective batch 256, as in the default pre-training).
module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs
LOG="slurm/logs/${SLURM_JOB_NAME}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"
RUN="pixi run --frozen"

OUT=cer_ckpts/d3d_tangential_agnostic.pt
RESUME=""; [ -f "$OUT" ] && RESUME="--resume"   # requeue: continue from the saved checkpoint
trap 'echo "USR1: requeue"; scontrol requeue $SLURM_JOB_ID' USR1

srun $RUN python -u train.py $RESUME \
    --machine d3d_tangential --checkpoint "$OUT" \
    --agnostic --chord-attention --resample-w 256 \
    --feature-width 128 --hidden-dim 8 --kernel-size 3 --head-type mlp --norm group \
    --batch-size 8 --accumulation-steps 32 --subseq-len 256 \
    --num-workers 8 --no-mmap &
wait
