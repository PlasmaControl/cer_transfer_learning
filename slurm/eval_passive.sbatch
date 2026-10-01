#!/bin/bash
#SBATCH --job-name=cer_eval_passive
#SBATCH --output=cer_eval_passive.out
#SBATCH --error=cer_eval_passive.err
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --mail-type=all
#SBATCH --mail-user=ps9551@princeton.edu

module purge
export PATH="$HOME/.pixi/bin:$PATH"
RUN="pixi run --frozen"

cd /scratch/gpfs/ps9551/passive_micer/cer_transfer

trap 'echo "caught USR1, requeueing"; scontrol requeue $SLURM_JOB_ID' USR1

# Passive evaluation: ACTIVE-trained checkpoints on PASSIVE spectrograms
# (same 48 flux-surface-aligned sightlines, same active-fit labels, same
# frames). Per-chord tables ARE the passive information-content result:
#   edge chords fine / core degraded  -> physics (expected)
#   uniform degradation incl. edge    -> low-count normalization problem
# Passive lists: full lists (train+val+test of passive files) are fine for
# a study eval, but the OFFICIAL numbers use only shots from the active
# TEST split to keep test discipline.
srun $RUN python -u eval_checkpoint.py \
    --checkpoint /scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstx_active_ft.pt \
    --list /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstx_passive_val.txt \
    --diagnose --out /scratch/gpfs/ps9551/passive_micer/cer_transfer/eval_nstx_passive.csv
srun $RUN python -u eval_checkpoint.py \
    --checkpoint /scratch/gpfs/ps9551/passive_micer/cer_transfer/cer_ckpts/nstxu_active_hop2.pt \
    --list /scratch/gpfs/ps9551/passive_micer/cer_transfer/splits/nstxu_passive_val.txt \
    --diagnose --out /scratch/gpfs/ps9551/passive_micer/cer_transfer/eval_nstxu_passive.csv
