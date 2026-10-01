#!/bin/bash
#SBATCH --job-name=cer_ft_hop2_nstxu_active
#SBATCH --output=cer_ft_hop2_nstxu_active.out
#SBATCH --error=cer_ft_hop2_nstxu_active.err
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

module purge
export PATH="$HOME/.pixi/bin:$PATH"
RUN="pixi run --frozen"

cd /scratch/gpfs/ps9551/passive_micer/cer_transfer

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# 48-chord ACTIVE lineage hop 2: nstx_active -> nstxu_active, everything
# warm (--transfer-readouts; same 48 sightlines-by-flux-surface). Scarce
# recipe (matches legacy hop 2).
SRC=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstx_active_ft.pt
OUT=/scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstxu_active_hop2.pt
mkdir -p "$(dirname "$OUT")"

srun $RUN python -u finetune.py \
    --source-checkpoint "$SRC" \
    --machine nstxu_active \
    --checkpoint "$OUT" \
    --train-list /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstxu_active_train.txt \
    --val-list   /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstxu_active_val.txt \
    --head-type mlp \
    --transfer-readouts \
    --subseq-len 64 \
    --probe-epochs 30 \
    --probe-lr 1e-3 \
    --backbone-lr 1e-5 \
    --late-backbone-lr 1e-4 \
    --head-lr 1e-3 \
    --epochs 300 \
    --no-mmap &
wait
