#!/bin/bash
#SBATCH --job-name=cer_ft_hop2_noD3D_nstxu
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

module purge
export PATH="$HOME/.pixi/bin:$PATH"
RUN="pixi run --frozen"

cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs
LOG="slurm/logs/${SLURM_JOB_NAME}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"
trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# Control: NSTX-scratch -> NSTX-U (no D3D anywhere in the chain).
# Isolates D3D's contribution vs hop 2 (nstx_ft source). Identical recipe.
SRC=cer_ckpts/nstx_scratch.pt
OUT=cer_ckpts/nstxu_hop2_noD3D.pt
RESUME=""; [ -f "$OUT" ] && RESUME="--resume"   # requeue: continue from the saved checkpoint
mkdir -p "$(dirname "$OUT")"

srun $RUN python -u finetune.py $RESUME \
    --source-checkpoint "$SRC" \
    --machine nstxu \
    --checkpoint "$OUT" \
    --train-list splits/nstxu_train.txt \
    --val-list   splits/nstxu_val.txt \
    --head-type mlp \
    --transfer-readouts \
    --subseq-len 64 \
    --probe-epochs 30 \
    --probe-lr 1e-3 \
    --backbone-lr 1e-5 \
    --late-backbone-lr 1e-4 \
    --head-lr 1e-3 \
    --epochs 300 \
    --no-mmap &
wait
