#!/bin/bash
#SBATCH --job-name=cer_agnostic_ladder
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=3-18:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ps9551@princeton.edu
# One rung of the agnostic scaling ladder: train the machine-agnostic model on
# a training subset to convergence (early stopping, resumes after requeue),
# evaluate it on its own validation set and zero-shot on the other machine,
# and write the affine diagnostic to gallery/ladder/<TAG>.csv. Submitted per
# rung by slurm/submit_agnostic_ladder.sh; collect with
# python -m cer_transfer.analysis.agnostic_ladder.
module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs gallery/ladder figs
: "${TAG:?TAG must be set (one per rung)}"
SRC_MACHINE=${SRC_MACHINE:?SRC_MACHINE must be set}
TGT_MACHINE=${TGT_MACHINE:?TGT_MACHINE must be set}
VAL_LIST=${VAL_LIST:?VAL_LIST must be set}
TGT_LIST=${TGT_LIST:?TGT_LIST must be set}
TRAIN_LIST=${TRAIN_LIST:-}          # empty: the machine's configured training directory
EPOCHS=${EPOCHS:-300}               # upper bound; early stopping (patience 50) ends earlier
LIMIT=${LIMIT:-200}                 # discharges used for the zero-shot evaluation
ATTN=${ATTN:-1}; ATTN_FLAG=""; [ "$ATTN" = "1" ] && ATTN_FLAG="--chord-attention"
MIRROR=${MIRROR:-0}                 # e.g. 0.5: mirror augmentation about the rest pixel
MIRROR_FLAG=""; [ "$MIRROR" != "0" ] && MIRROR_FLAG="--mirror-augment $MIRROR"
DG=${DG:-""}                        # extra domain-generalization flags, e.g.
                                    # "--randconv randconv --consistency-w 5 --chord-attention-depth 2"
OUT=cer_ckpts/$TAG.pt
LOG="slurm/logs/${SLURM_JOB_NAME}_${TAG}"      # one log per rung
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"
RUN="pixi run --frozen"

TRAIN_ARGS=""; [ -n "$TRAIN_LIST" ] && TRAIN_ARGS="--train-list $TRAIN_LIST"
N_TRAIN=$( [ -n "$TRAIN_LIST" ] && grep -cv '^#' "$TRAIN_LIST" || \
  $RUN python -c "from cer_transfer.configs import data_path, get_machine
print(len(list(data_path(get_machine('$SRC_MACHINE').train_dir).glob('*.joblib'))))" )
RESUME=""; [ -f "$OUT" ] && RESUME="--resume"   # requeue: continue; finished runs are skipped
trap 'echo "USR1: requeue"; scontrol requeue $SLURM_JOB_ID' USR1

echo "=== $TAG: train $SRC_MACHINE agnostic model on $N_TRAIN discharges (${TRAIN_LIST:-config dir})"
srun $RUN python -u train.py $RESUME --machine $SRC_MACHINE --checkpoint "$OUT" \
    $TRAIN_ARGS --val-list $VAL_LIST --epochs $EPOCHS \
    --agnostic $ATTN_FLAG $MIRROR_FLAG $DG --resample-w 256 \
    --feature-width 128 --hidden-dim 8 --kernel-size 3 --head-type mlp --norm group \
    --batch-size 8 --accumulation-steps 32 --subseq-len 256 --num-workers 8 --no-mmap &
wait

echo "=== $TAG: own machine ($SRC_MACHINE validation)"
srun $RUN python -u eval_checkpoint.py --checkpoint "$OUT" --list $VAL_LIST --limit $LIMIT \
    --dump-preds gallery/${TAG}_own.npz
srun $RUN python -u -m cer_transfer.analysis.affine_fit --preds gallery/${TAG}_own.npz \
    --csv gallery/ladder/${TAG}_own.csv \
    --meta tag=$TAG eval=own src=$SRC_MACHINE tgt=$SRC_MACHINE n_train=$N_TRAIN ckpt=$OUT

echo "=== $TAG: zero-shot on $TGT_MACHINE (first $LIMIT discharges of $TGT_LIST)"
srun $RUN python -u eval_checkpoint.py --checkpoint "$OUT" --machine $TGT_MACHINE \
    --list $TGT_LIST --limit $LIMIT --dump-preds gallery/${TAG}_zero_shot.npz
srun $RUN python -u -m cer_transfer.analysis.affine_fit --preds gallery/${TAG}_zero_shot.npz \
    --csv gallery/ladder/${TAG}_zero_shot.csv \
    --meta tag=$TAG eval=zero_shot src=$SRC_MACHINE tgt=$TGT_MACHINE n_train=$N_TRAIN ckpt=$OUT
