#!/bin/bash
#SBATCH --job-name=cer_compare_arrays
#SBATCH --output=cer_compare_arrays.out
#SBATCH --error=cer_compare_arrays.err
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00

module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd /scratch/gpfs/ps9551/passive_micer/cer_transfer

# whole-dataset active-vs-passive comparison (CPU/IO only, no GPU).
# Uses train+val lists; test lists untouched.
for M in nstx nstxu; do
  cat splits/${M}_active_train.txt  splits/${M}_active_val.txt  > /tmp/${M}_a.txt
  cat splits/${M}_passive_train.txt splits/${M}_passive_val.txt > /tmp/${M}_p.txt
  pixi run --frozen python -u -m cer_transfer.analysis.compare_arrays \
      --active-list /tmp/${M}_a.txt \
      --passive-list /tmp/${M}_p.txt \
      --out /scratch/gpfs/ps9551/passive_micer/cer_transfer/figs/compare_${M}
done
