#!/bin/bash
#SBATCH --job-name=cer_ft_nstxu_pooled
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
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

# pooled union list (NSTX r325 + all NSTX-U train shots); both are 51-chord
# legacy-format datasets with identical targets/units — one stem serves both
# instruments, which is exactly what "pooled" tests.
cat splits/nstx_train_r325.txt splits/nstxu_train.txt > /tmp/nstxu_pooled_train.txt
SRC=cer_ckpts/d3d_transfer_source.pt
OUT=cer_ckpts/nstxu_pooled.pt
RESUME=""; [ -f "$OUT" ] && RESUME="--resume"   # requeue: continue from the saved checkpoint
mkdir -p "$(dirname "$OUT")"

srun $RUN python -u finetune.py $RESUME \
    --source-checkpoint "$SRC" \
    --machine nstxu \
    --checkpoint "$OUT" \
    --train-list /tmp/nstxu_pooled_train.txt \
    --val-list   splits/nstxu_val.txt \
    --head-type mlp \
    --subseq-len 64 \
    --probe-epochs 30 \
    --probe-lr 1e-3 \
    --backbone-lr 1e-5 \
    --late-backbone-lr 1e-4 \
    --head-lr 1e-3 \
    --epochs 300 \
    --no-mmap &
wait
