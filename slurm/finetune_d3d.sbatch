#!/bin/bash
#SBATCH --job-name=cer_ft_nstx_to_d3d
#SBATCH --output=cer_ft_nstx_to_d3d.out
#SBATCH --error=cer_ft_nstx_to_d3d.err
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

# Reverse hop: NSTX -> D3D. Source = NSTX from-scratch model (stem/trunk
# arch, feature_width 128, GroupNorm, mlp head). Trunk + head trunk
# transferred; fresh 80-channel D3D stem (stem_w_pools=2) + 80 readouts.
# D3D is the large dataset => same near-full-scale schedule as hop 1:
# short probe, trunk LRs ~1/10 of full LR.
# Data: no lists -> D3D config dirs (training_set_30 / test_set_30), exactly
# as the D3D source pretrain (pretrain_transfer_source.sbatch), so the
# from-scratch reference is cer_ckpts/d3d_transfer_source.pt.
# subseq-len 256 = D3D pretrain default (NSTX runs used 64 for short shots).

SRC=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstx_transfer_source.pt
OUT=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/d3d_ft.pt
mkdir -p "$(dirname "$OUT")"
[ -f "$SRC" ] || { echo "missing source checkpoint $SRC"; exit 1; }

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

srun $RUN python -u finetune.py \
    --source-checkpoint "$SRC" \
    --machine d3d \
    --checkpoint "$OUT" \
    --head-type mlp \
    --subseq-len 256 \
    --probe-epochs 5 \
    --probe-lr 1e-3 \
    --backbone-lr 1e-4 \
    --late-backbone-lr 3e-4 \
    --head-lr 1e-3 \
    --epochs 300 \
    --no-mmap &
wait
