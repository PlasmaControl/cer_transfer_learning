#!/bin/bash
#SBATCH --job-name=cer_ft_d3d_r4000
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=90:00:00
#SBATCH --signal=B:USR1@300          # signal 5 min before timeout (for requeue)
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

# D3D scaling study, NSTX -> D3D transfer at 4000 training discharges
# (nested random subset splits/d3d_train_r4000.txt, cer_transfer.datasets.make_d3d_subsets).
# Scarce-data LP-FT recipe, identical to the NSTX scaling arms
# (finetune_nstx_r*): probe 30 epochs, backbone 1e-5 / late 1e-4 / head 1e-3.
# Val = D3D config val_dir (same for all sizes). Equal-data control:
# scratch_d3d_r4000.sbatch.

SRC=cer_ckpts/nstx_transfer_source.pt
OUT=cer_ckpts/d3d_ft_r4000.pt
RESUME=""; [ -f "$OUT" ] && RESUME="--resume"   # requeue: continue from the saved checkpoint
LIST=splits/d3d_train_r4000.txt
mkdir -p "$(dirname "$OUT")"
[ -f "$SRC" ]  || { echo "missing source checkpoint $SRC"; exit 1; }
[ -f "$LIST" ] || { echo "missing $LIST (run cer_transfer.datasets.make_d3d_subsets)"; exit 1; }

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

srun $RUN python -u finetune.py $RESUME \
    --source-checkpoint "$SRC" \
    --machine d3d \
    --checkpoint "$OUT" \
    --train-list "$LIST" \
    --head-type mlp \
    --subseq-len 256 \
    --probe-epochs 30 \
    --probe-lr 1e-3 \
    --backbone-lr 1e-5 \
    --late-backbone-lr 1e-4 \
    --head-lr 1e-3 \
    --epochs 300 \
    --no-mmap &
wait
