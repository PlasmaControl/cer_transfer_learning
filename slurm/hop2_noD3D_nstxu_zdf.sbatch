#!/bin/bash
#SBATCH --job-name=cer_ft_hop2_noD3D_nstxu_zdf
#SBATCH --output=cer_ft_hop2_noD3D_nstxu_zdf.out
#SBATCH --error=cer_ft_hop2_noD3D_nstxu_zdf.err
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

cd /scratch/gpfs/ps9551/passive_micer/cer_transfer

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# Control: NSTX-scratch -> NSTX-U (no D3D anywhere in the chain).
# Isolates D3D's contribution vs hop 2 (nstx_ft source). Identical recipe.
SRC=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstx_scratch.pt
OUT=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstxu_hop2_noD3D_zdf.pt
mkdir -p "$(dirname "$OUT")"

srun $RUN python -u finetune.py \
    --source-checkpoint "$SRC" \
    --machine nstxu \
    --checkpoint "$OUT" \
    --train-list /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstxu_train.txt \
    --val-list   /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstxu_val.txt \
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
