#!/bin/bash
#SBATCH --job-name=cer_ft_hop1_nstx_active
#SBATCH --output=cer_ft_hop1_nstx_active.out
#SBATCH --error=cer_ft_hop1_nstx_active.err
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=90:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --mail-type=all
#SBATCH --mail-user=ps9551@princeton.edu

module purge
export PATH="$HOME/.pixi/bin:$PATH"
RUN="pixi run --frozen"

cd /scratch/gpfs/ps9551/passive_micer/cer_transfer

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# 48-chord ACTIVE lineage hop 1: D3D -> nstx_active. Reuses the EXISTING
# transfer source (trunk is machine-agnostic; stem + 48 readouts fresh).
# Data-rich recipe (matches legacy hop 1).
SRC=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/d3d_transfer_source.pt
OUT=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstx_active_ft.pt
mkdir -p "$(dirname "$OUT")"

srun $RUN python -u finetune.py \
    --source-checkpoint "$SRC" \
    --machine nstx_active \
    --checkpoint "$OUT" \
    --train-list /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstx_active_train.txt \
    --val-list   /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstx_active_val.txt \
    --head-type mlp \
    --subseq-len 64 \
    --probe-epochs 5 \
    --probe-lr 1e-3 \
    --backbone-lr 1e-4 \
    --late-backbone-lr 3e-4 \
    --head-lr 1e-3 \
    --epochs 300 \
    --no-mmap &
wait
