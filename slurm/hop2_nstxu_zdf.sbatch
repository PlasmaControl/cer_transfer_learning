#!/bin/bash
#SBATCH --job-name=cer_ft_hop2_nstxu_zdf
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@300          # signal 5 min before timeout (for requeue)
#SBATCH --requeue
#SBATCH --mail-type=all
#SBATCH --mail-user=ps9551@princeton.edu

# ZDF variant of the official recipe: only change is
# --zero-dark-frames (criterion: cer_transfer/dark.py).
# NSTX lists are certified by cer_transfer.analysis.check_dark_frames; for
# NSTX-U lists, run cer_transfer.analysis.check_dark_frames on that list once
# before first use (guard encodes breakdown-to-beam delay).
module purge
export PATH="$HOME/.pixi/bin:$PATH"
RUN="pixi run --frozen"

cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs
LOG="slurm/logs/${SLURM_JOB_NAME}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# Hop 2: NSTX -> NSTX-U. Everything warm-started (--transfer-readouts:
# trunk, head trunk, stem, per-chord readouts; 51/51 chords). Requires the
# hop-1 checkpoint.

SRC=cer_ckpts/nstx_ft.pt
OUT=cer_ckpts/nstxu_hop2_zdf.pt
RESUME=""; [ -f "$OUT" ] && RESUME="--resume"   # requeue: continue from the saved checkpoint
mkdir -p "$(dirname "$OUT")"

srun $RUN python -u finetune.py $RESUME \
    --source-checkpoint "$SRC" \
    --machine nstxu \
    --checkpoint "$OUT" \
    --train-list splits/nstxu_train.txt \
    --val-list   splits/nstxu_val.txt \
    --transfer-readouts \
    --subseq-len 64 \
    --head-type mlp \
    --probe-epochs 30 \
    --backbone-lr 1e-5 \
    --late-backbone-lr 1e-4 \
    --head-lr 1e-3 \
    --zero-dark-frames \
    --no-mmap &
wait
