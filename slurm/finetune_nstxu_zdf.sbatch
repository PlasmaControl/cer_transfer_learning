#!/bin/bash
#SBATCH --job-name=cer_ft_hop1_nstxu_zdf
#SBATCH --output=cer_ft_hop1_nstxu_zdf.out
#SBATCH --error=cer_ft_hop1_nstxu_zdf.err
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

# ZDF variant of the official recipe: only change is
# --zero-dark-frames (criterion: cer_transfer/dark.py).
# NSTX lists are certified by cer_transfer.analysis.check_dark_frames; for
# NSTX-U lists, run cer_transfer.analysis.check_dark_frames on that list once
# before first use (guard encodes breakdown-to-beam delay).
module purge
export PATH="$HOME/.pixi/bin:$PATH"
RUN="pixi run --frozen"

cd /scratch/gpfs/ps9551/passive_micer/cer_transfer            # repo root

# Hop 1: D3D -> NSTX. Trunk + head trunk transferred; fresh 51-channel stem
# + readouts. 10.4k labeled shots => near-full-scale training: short probe,
# trunk LRs ~1/10 of full LR (not the scarce-data 1/100 defaults).

SRC=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/d3d_transfer_source.pt
OUT=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstxu_ft_zdf.pt
mkdir -p "$(dirname "$OUT")"

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

srun $RUN python -u finetune.py \
    --source-checkpoint "$SRC" \
    --machine nstxu \
    --checkpoint "$OUT" \
    --train-list /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstxu_train.txt \
    --val-list   /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstxu_val.txt \
    --head-type mlp \
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
