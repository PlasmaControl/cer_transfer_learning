#!/bin/bash
#SBATCH --job-name=cer_ft_nstx_passive_zdf
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
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
# (physics zero pseudo-labels on pre-discharge dark frames; criterion in
# cer_transfer/dark.py, certified by cer_transfer.analysis.check_dark_frames).
# Accept iff labeled-point val metrics match the official within noise
# AND dark-frame outputs -> ~0.

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# PASSIVE fine-tune: nstx_active_ft -> nstx_passive, everything warm
# (--transfer-readouts: same 48 flux-surface-aligned sightlines, same
# active-fit labels; only the INPUT array changes). This is the passive
# paper's inference model. Data-rich recipe (10k shots).
SRC=cer_ckpts/nstx_active_ft.pt
OUT=cer_ckpts/nstx_passive_ft_zdf.pt
RESUME=""; [ -f "$OUT" ] && RESUME="--resume"   # requeue: continue from the saved checkpoint
mkdir -p "$(dirname "$OUT")"

srun $RUN python -u finetune.py $RESUME \
    --source-checkpoint "$SRC" \
    --machine nstx_passive \
    --checkpoint "$OUT" \
    --train-list splits/nstx_passive_train.txt \
    --val-list   splits/nstx_passive_val.txt \
    --head-type mlp \
    --transfer-readouts \
    --subseq-len 64 \
    --probe-epochs 5 \
    --probe-lr 1e-3 \
    --backbone-lr 1e-4 \
    --late-backbone-lr 3e-4 \
    --head-lr 1e-3 \
    --epochs 300 \
    --zero-dark-frames \
    --no-mmap &
wait
