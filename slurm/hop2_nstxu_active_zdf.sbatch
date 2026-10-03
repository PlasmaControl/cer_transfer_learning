#!/bin/bash
#SBATCH --job-name=cer_ft_hop2_nstxu_active_zdf
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@300
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
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"
trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# 48-chord ACTIVE lineage hop 2: nstx_active -> nstxu_active, everything
# warm (--transfer-readouts; same 48 sightlines-by-flux-surface). Scarce
# recipe (matches legacy hop 2).
SRC=cer_ckpts/nstx_active_ft.pt
OUT=cer_ckpts/nstxu_active_hop2_zdf.pt
RESUME=""; [ -f "$OUT" ] && RESUME="--resume"   # requeue: continue from the saved checkpoint
mkdir -p "$(dirname "$OUT")"

srun $RUN python -u finetune.py $RESUME \
    --source-checkpoint "$SRC" \
    --machine nstxu_active \
    --checkpoint "$OUT" \
    --train-list splits/nstxu_active_train.txt \
    --val-list   splits/nstxu_active_val.txt \
    --head-type mlp \
    --transfer-readouts \
    --subseq-len 64 \
    --probe-epochs 30 \
    --probe-lr 1e-3 \
    --backbone-lr 1e-5 \
    --late-backbone-lr 1e-4 \
    --head-lr 1e-3 \
    --epochs 300 \
    --zero-dark-frames \
    --no-mmap &
wait
