#!/bin/bash
#SBATCH --job-name=cer_fixed_d3d
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ps9551@princeton.edu
# Machine-agnostic DIII-D model on the REBUILT files (one wavelength grid for
# every shot, uint16, t01-t48): machine d3d_tangential_fixed, DG/transfer
# recipe of the paper's d3d_full_dg model. Data root from .env.fixed
# (CER_DATA_ROOT=.../data_root_fixed), the regular .env stays untouched.
#   sbatch slurm/pretrain_fixed_d3d.sbatch
#   RECIPE=plain sbatch ... for the baseline without the transfer augmentations
module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs
LOG="slurm/logs/${SLURM_JOB_NAME}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env.fixed ] || { echo ".env.fixed missing (export CER_DATA_ROOT=<data_root_fixed>)"; exit 1; }
source .env.fixed
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set in .env.fixed}"
case "$CER_DATA_ROOT" in *fixed*) ;; *) echo "CER_DATA_ROOT=$CER_DATA_ROOT does not look like the fixed root"; exit 1;; esac
RUN="pixi run --frozen"

RECIPE=${RECIPE:-transfer}
OUT=cer_ckpts/d3d_fixed_dg.pt; [ "$RECIPE" = plain ] && OUT=cer_ckpts/d3d_fixed_plain.pt
AUG="--randconv randconv --randconv-max-shift 30 --consistency-w 5 --mirror-augment 0.5"
[ "$RECIPE" = plain ] && AUG=""
RESUME=""; [ -f "$OUT" ] && RESUME="--resume"   # requeue: continue from the saved checkpoint
trap 'echo "USR1: requeue"; scontrol requeue $SLURM_JOB_ID' USR1

echo "=== $OUT: machine d3d_tangential_fixed, recipe $RECIPE, data root $CER_DATA_ROOT"
srun $RUN python -u train.py $RESUME \
    --machine d3d_tangential_fixed --checkpoint "$OUT" \
    --agnostic --chord-attention --chord-attention-depth 2 --resample-w 256 \
    --feature-width 128 --hidden-dim 8 --kernel-size 3 --head-type mlp --norm group \
    $AUG \
    --batch-size 8 --accumulation-steps 32 --subseq-len 256 \
    --num-workers 8 --no-mmap &
wait
