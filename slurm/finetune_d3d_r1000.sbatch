#!/bin/bash
#SBATCH --job-name=cer_ft_d3d_r1000
#SBATCH --output=cer_ft_d3d_r1000.out
#SBATCH --error=cer_ft_d3d_r1000.err
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

# D3D scaling study, NSTX -> D3D transfer at 1000 training discharges
# (nested random subset splits/d3d_train_r1000.txt, cer_transfer.datasets.make_d3d_subsets).
# Scarce-data LP-FT recipe, identical to the NSTX scaling arms
# (finetune_nstx_r*): probe 30 epochs, backbone 1e-5 / late 1e-4 / head 1e-3.
# Val = D3D config val_dir (same for all sizes). Equal-data control:
# scratch_d3d_r1000.sbatch.

SRC=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstx_transfer_source.pt
OUT=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/d3d_ft_r1000.pt
LIST=/scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/d3d_train_r1000.txt
mkdir -p "$(dirname "$OUT")"
[ -f "$SRC" ]  || { echo "missing source checkpoint $SRC"; exit 1; }
[ -f "$LIST" ] || { echo "missing $LIST (run cer_transfer.datasets.make_d3d_subsets)"; exit 1; }

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

srun $RUN python -u finetune.py \
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
