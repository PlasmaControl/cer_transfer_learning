#!/bin/bash
#SBATCH --job-name=cer_ft_hop2_nstxu
#SBATCH --output=cer_ft_hop2_nstxu.out
#SBATCH --error=cer_ft_hop2_nstxu.err
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@300          # signal 5 min before timeout (for requeue)
#SBATCH --requeue
#SBATCH --mail-type=all
#SBATCH --mail-user=ps9551@princeton.edu

module purge
export PATH="$HOME/.pixi/bin:$PATH"
RUN="pixi run --frozen"

cd /scratch/gpfs/ps9551/passive_micer/cer_transfer            # repo root

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# Hop 2: NSTX -> NSTX-U. Everything warm-started (--transfer-readouts:
# trunk, head trunk, stem, per-chord readouts; 51/51 chords). Requires the
# hop-1 checkpoint.

SRC=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstx_ft.pt
OUT=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstxu_hop2.pt
mkdir -p "$(dirname "$OUT")"

srun $RUN python -u finetune.py \
    --source-checkpoint "$SRC" \
    --machine nstxu \
    --checkpoint "$OUT" \
    --train-list /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstxu_train.txt \
    --val-list   /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstxu_val.txt \
    --transfer-readouts \
    --subseq-len 64 \
    --head-type mlp \
    --probe-epochs 30 \
    --backbone-lr 1e-5 \
    --late-backbone-lr 1e-4 \
    --head-lr 1e-3 \
    --no-mmap &
wait
