#!/bin/bash
#SBATCH --job-name=cer_ft_hop1_nstx
#SBATCH --output=cer_ft_hop1_nstx.out
#SBATCH --error=cer_ft_hop1_nstx.err
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

cd /scratch/gpfs/ps9551/passive_micer/cer_transfer            # repo root

# Hop 1: D3D -> NSTX. Trunk + head trunk transferred; fresh 51-channel stem
# + readouts. 10.4k labeled shots => near-full-scale training: short probe,
# trunk LRs ~1/10 of full LR (not the scarce-data 1/100 defaults).

SRC=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/d3d_transfer_source.pt
OUT=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstx_ft.pt
mkdir -p "$(dirname "$OUT")"

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

srun $RUN python -u finetune.py \
    --source-checkpoint "$SRC" \
    --machine nstx \
    --checkpoint "$OUT" \
    --train-list /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstx_train.txt \
    --val-list   /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstx_val.txt \
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
