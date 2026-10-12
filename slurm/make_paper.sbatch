#!/bin/bash
#SBATCH --job-name=cer_make_paper
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=8G --gres=gpu:1
#SBATCH --qos=gpu-short --time=06:00:00
#SBATCH --mail-type=END,FAIL --mail-user=ps9551@princeton.edu
# All evaluations, dumps and figures of the paper in one job:
#   sbatch slurm/make_paper.sbatch            # TEST=1 sbatch ... adds the one test-set evaluation
module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..
mkdir -p slurm/logs
exec >>"slurm/logs/${SLURM_JOB_NAME}.out" 2>>"slurm/logs/${SLURM_JOB_NAME}.err"
scripts/make_paper.sh
