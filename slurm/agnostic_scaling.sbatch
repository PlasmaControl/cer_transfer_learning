#!/bin/bash
#SBATCH --job-name=cer_agn_scaling
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=3-18:00:00
#SBATCH --array=0-6
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ps9551@princeton.edu
# Scaling study of the machine-agnostic model on NSTX, one array job per mode,
# array task = NSTX training size (0, 325, 650, 1300, 2600, 5200, all):
#   MODE=finetune (default): task 0 = zero-shot (DIII-D model SRC evaluated without
#     NSTX training); tasks 1-6 fine-tune SRC on the subsets (single phase).
#   MODE=scratch: tasks 1-6 train the same architecture from scratch on the
#     subsets (task 0 does not exist); RECIPE=transfer (default) uses the DIII-D
#     model's augmentations, RECIPE=plain none of them (the fair baseline).
# Every task evaluates on the complete NSTX validation list and writes
# gallery/agnostic_scaling/<tag>.csv; trained runs write cer_ckpts/<tag>.metrics.csv.
#   sbatch --array=0-6 --export=ALL,MODE=finetune,SRC=cer_ckpts/d3d_full_dg.pt slurm/agnostic_scaling.sbatch
#   sbatch --array=1-6 --export=ALL,MODE=scratch,SHIFT=30 slurm/agnostic_scaling.sbatch
#   sbatch --array=1-6 --export=ALL,MODE=scratch,RECIPE=plain slurm/agnostic_scaling.sbatch
#   track B: sbatch --array=0-6 --export=ALL,ENV=.env.fixed,MODE=finetune,SRC=cer_ckpts/d3d_fixed_plain_best.pt,PREFIX=nstx_fixed_ft slurm/agnostic_scaling.sbatch
module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs gallery/agnostic_scaling
MODE=${MODE:-finetune}
RECIPE=${RECIPE:-transfer}   # scratch only: transfer = DIII-D model's augmentations, plain = none
if [ "$MODE" = scratch ] && [ "$RECIPE" = plain ]; then PREFIX=${PREFIX:-nstx_agn_scratch_plain}
elif [ "$MODE" = scratch ]; then PREFIX=${PREFIX:-nstx_agn_scratch}; else
  SRC=${SRC:?SRC must be the DIII-D agnostic checkpoint}; PREFIX=${PREFIX:-nstx_agn_ft}; fi
SHIFT=${SHIFT:-30}   # random-shift range of the recipe (must match the DIII-D model)
RUNGS=(0 325 650 1300 2600 5200 full)
K=${RUNGS[${SLURM_ARRAY_TASK_ID:-0}]}
TAG=${PREFIX}_r${K}
[ "$K" = 0 ] && TAG=${PREFIX}_zero_shot
LOG="slurm/logs/${SLURM_JOB_NAME}_${TAG}"
exec >>"$LOG.out" 2>>"$LOG.err"
ENV=${ENV:-.env}; [ -f "$ENV" ] && source "$ENV"   # ENV=.env.fixed for the rebuilt DIII-D data (track B)
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"
RUN="pixi run --frozen"
[ "$MODE" = scratch ] && [ "$K" = 0 ] && { echo "MODE=scratch has no task 0"; exit 0; }
[ "$MODE" = scratch ] || [ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }

if [ "$MODE" = scratch ]; then
  LIST=splits/nstx_train_r$K.txt; [ "$K" = full ] && LIST=splits/nstx_train.txt
  N=$(grep -cv '^#' "$LIST")
  CKPT=cer_ckpts/$TAG.pt; MFLAG=""
  RESUME=""; [ -f "$CKPT" ] && RESUME="--resume"   # requeue: continue; finished runs are skipped
  trap 'echo "USR1: requeue"; scontrol requeue $SLURM_JOB_ID' USR1
  AUG="--randconv randconv --randconv-max-shift $SHIFT --consistency-w 5 --mirror-augment 0.5"
  [ "$RECIPE" = plain ] && AUG=""   # same architecture, no transfer augmentations
  echo "=== $TAG: train from scratch on $N NSTX discharges (recipe: $RECIPE${AUG:+, $AUG})"
  srun $RUN python -u train.py $RESUME --machine nstx --checkpoint "$CKPT" \
      --train-list $LIST --val-list splits/nstx_val.txt --epochs 300 \
      --agnostic --chord-attention --chord-attention-depth 2 --resample-w 256 \
      --feature-width 128 --hidden-dim 8 --kernel-size 3 --head-type mlp --norm group \
      $AUG \
      --batch-size 8 --accumulation-steps 32 --subseq-len 256 --num-workers 8 --no-mmap &
  wait
elif [ "$K" = 0 ]; then
  CKPT=$SRC; MFLAG="--machine nstx"; N=0
else
  LIST=splits/nstx_train_r$K.txt; [ "$K" = full ] && LIST=splits/nstx_train.txt
  N=$(grep -cv '^#' "$LIST")
  CKPT=cer_ckpts/$TAG.pt; MFLAG=""
  RESUME=""; [ -f "$CKPT" ] && RESUME="--resume"   # requeue: continue; finished runs are skipped
  trap 'echo "USR1: requeue"; scontrol requeue $SLURM_JOB_ID' USR1
  echo "=== $TAG: fine-tune $SRC on $N NSTX discharges"
  srun $RUN python -u finetune.py $RESUME \
      --source-checkpoint "$SRC" --machine nstx --checkpoint "$CKPT" \
      --train-list $LIST --val-list splits/nstx_val.txt \
      --subseq-len 64 --batch-size 32 --accumulation-steps 8 --num-workers 8 \
      --probe-epochs 0 --backbone-lr 1e-5 --late-backbone-lr 1e-4 --head-lr 1e-4 \
      --epochs 300 --no-mmap &
  wait
fi

echo "=== $TAG: evaluation on the complete NSTX validation list ($N training discharges)"
srun $RUN python -u eval_checkpoint.py --checkpoint "$CKPT" $MFLAG \
    --list splits/nstx_val.txt --dump-preds gallery/agnostic_scaling/$TAG.npz
srun $RUN python -u -m cer_transfer.analysis.affine_fit --preds gallery/agnostic_scaling/$TAG.npz \
    --csv gallery/agnostic_scaling/$TAG.csv \
    --meta tag=$TAG n_train=$N ckpt=$CKPT src=$SRC
