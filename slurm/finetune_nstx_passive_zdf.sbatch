#!/bin/bash
#SBATCH --job-name=cer_ft_nstx_passive_zdf
#SBATCH --output=cer_ft_nstx_passive_zdf.out
#SBATCH --error=cer_ft_nstx_passive_zdf.err
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

cd /scratch/gpfs/ps9551/passive_micer/cer_transfer

# ZDF variant: identical to the official recipe + --zero-dark-frames
# (physics zero pseudo-labels on pre-discharge dark frames; criterion in
# cer_transfer/dark.py, certified by cer_transfer.analysis.check_dark_frames).
# Accept iff labeled-point val metrics match the official within noise
# AND dark-frame outputs -> ~0.

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# PASSIVE fine-tune: nstx_active_ft -> nstx_passive, everything warm
# (--transfer-readouts: same 48 flux-surface-aligned sightlines, same
# active-fit labels; only the INPUT array changes). This is the passive
# paper's inference model. Data-rich recipe (10k shots).
SRC=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstx_active_ft.pt
OUT=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstx_passive_ft_zdf.pt
mkdir -p "$(dirname "$OUT")"

srun $RUN python -u finetune.py \
    --source-checkpoint "$SRC" \
    --machine nstx_passive \
    --checkpoint "$OUT" \
    --train-list /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstx_passive_train.txt \
    --val-list   /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstx_passive_val.txt \
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
