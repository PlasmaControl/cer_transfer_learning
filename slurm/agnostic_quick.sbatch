#!/bin/bash
#SBATCH --job-name=cer_agnostic_quick
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ps9551@princeton.edu
# Quick zero-shot test of the agnostic model: short training on a subset of
# one machine, then evaluation on its own validation data and zero-shot on the
# other machine, with the affine diagnostic. Hours instead of days.
#
#   NSTX -> DIII-D (fastest, ~1-2 h):
#     sbatch --export=ALL slurm/agnostic_quick.sbatch
#   DIII-D -> NSTX on 1000 discharges (~4-6 h):
#     sbatch --export=ALL,SRC_MACHINE=d3d_tangential,TRAIN_LIST=splits/d3d_train_r1000.txt,\
#            VAL_LIST=splits/d3d_val.txt,TGT_MACHINE=nstx,TGT_LIST=splits/nstx_val.txt,EPOCHS=30 \
#            slurm/agnostic_quick.sbatch
module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs gallery figs
SRC_MACHINE=${SRC_MACHINE:-nstx}
TRAIN_LIST=${TRAIN_LIST:-splits/nstx_train_r1300.txt}
VAL_LIST=${VAL_LIST:-splits/nstx_val.txt}
TGT_MACHINE=${TGT_MACHINE:-d3d_tangential}   # tangential chords only (0-47)
TGT_LIST=${TGT_LIST:-splits/d3d_val.txt}
EPOCHS=${EPOCHS:-20}
LIMIT=${LIMIT:-200}                 # discharges used for the zero-shot evaluation
ATTN=${ATTN:-1}; ATTN_FLAG=""; [ "$ATTN" = "1" ] && ATTN_FLAG="--chord-attention"
TAG=${TAG:-${SRC_MACHINE}_agnostic_quick}
OUT=cer_ckpts/$TAG.pt
LOG="slurm/logs/${SLURM_JOB_NAME}_${TAG}"      # one log per run tag
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"
RUN="pixi run --frozen"


echo "=== train agnostic model: $SRC_MACHINE, $TRAIN_LIST, $EPOCHS epochs"
srun $RUN python -u train.py --machine $SRC_MACHINE --checkpoint "$OUT" \
    --train-list $TRAIN_LIST --val-list $VAL_LIST --epochs $EPOCHS \
    --agnostic $ATTN_FLAG --resample-w 256 \
    --feature-width 128 --hidden-dim 8 --kernel-size 3 --head-type mlp --norm group \
    --batch-size 8 --accumulation-steps 32 --subseq-len 256 --num-workers 8 --no-mmap

echo "=== own machine: $SRC_MACHINE validation"
srun $RUN python -u eval_checkpoint.py --checkpoint "$OUT" --list $VAL_LIST

if [ ! -f "$TGT_LIST" ]; then   # e.g. DIII-D, whose runs use the config directory
  echo "building $TGT_LIST from the $TGT_MACHINE validation directory"
  $RUN python -c "from cer_transfer.configs import data_path, get_machine, list_entry
m = get_machine('$TGT_MACHINE')
print('\n'.join(list_entry(p) for p in sorted(data_path(m.val_dir).glob('*.joblib'))))" > "$TGT_LIST"
fi
echo "=== zero-shot: $TGT_MACHINE ($TGT_LIST, first $LIMIT discharges), no $TGT_MACHINE training"
srun $RUN python -u eval_checkpoint.py --checkpoint "$OUT" --machine $TGT_MACHINE \
    --list $TGT_LIST --limit $LIMIT --dump-preds gallery/${TAG}_zero_shot_$TGT_MACHINE.npz
srun $RUN python -u -m cer_transfer.analysis.affine_fit \
    --preds gallery/${TAG}_zero_shot_$TGT_MACHINE.npz --plot figs/affine_${TAG}_$TGT_MACHINE
