#!/bin/bash
#SBATCH --job-name=cer_ft_nstxu
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --account=kolemen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ps9551@princeton.edu
# Fine-tune a causal checkpoint SRC on the NSTX-U training discharges (325) with the recipe
# of the paper's fine-tuned NSTX models (nstx_rt_init_*: lr 1e-3 everywhere, no probe phase,
# subseq 64). NSTX-U files have the NSTX layout (51 chords, same window) and use the NSTX
# preprocessing (--machine nstx). Evaluates on the NSTX-U validation and test lists.
#   sbatch --export=ALL,SRC=cer_ckpts/d3d_fixed_dg_rt.pt,TAG=nstxu_ft_d3d_r325 slurm/finetune_nstxu.sbatch
#   sbatch --export=ALL,SRC=cer_ckpts/nstx_rt_init_rfull.pt,TAG=nstxu_ft_nstx_r325 slurm/finetune_nstxu.sbatch
# MODE=scratch: the baseline, the same (causal) architecture trained from scratch on the same
# discharges with the recipe of nstx_scratch64_* (agnostic_scaling.sbatch MODE=scratch RECIPE=plain)
#   sbatch --export=ALL,MODE=scratch,TAG=nstxu_scratch64_r325 slurm/finetune_nstxu.sbatch
#   seeds (nstxu_transfer figure): add SEED=1..5 -> <TAG>_s<n>
module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..
mkdir -p slurm/logs gallery/nstxu
MODE=${MODE:-finetune}
[ "$MODE" = scratch ] || SRC=${SRC:?SRC must be a causal checkpoint}
TAG=${TAG:?TAG names the run}
# SEED=n: training seed 42+n, tag suffix _s<n> (same convention as agnostic_scaling.sbatch;
# the NSTX-U training list is fixed, only the training seed changes)
SEED=${SEED:-}; TAG=$TAG${SEED:+_s$SEED}
SEEDFLAG=""; [ -n "$SEED" ] && SEEDFLAG="--seed $((42 + SEED))"
LIST=${LIST:-splits/nstxu_train.txt}
LOG="slurm/logs/${SLURM_JOB_NAME}_${TAG}"
exec >>"$LOG.out" 2>>"$LOG.err"
ENV=${ENV:-.env.fixed}; [ -f "$ENV" ] && source "$ENV"
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set}"
RUN="pixi run --frozen"
[ "$MODE" = scratch ] || [ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
CKPT=cer_ckpts/$TAG.pt
RESUME=""; [ -f "$CKPT" ] && RESUME="--resume"   # requeue: continue
trap 'echo "USR1: requeue"; scontrol requeue $SLURM_JOB_ID' USR1
if [ "$MODE" = scratch ]; then
  echo "=== $TAG: train from scratch on $(grep -cv '^#' $LIST) NSTX-U discharges (recipe: plain)"
  srun $RUN python -u train.py $RESUME $SEEDFLAG --machine nstx --checkpoint "$CKPT" \
      --train-list $LIST --val-list splits/nstxu_val.txt --epochs 300 \
      --agnostic --chord-attention --chord-attention-depth 2 --resample-w 256 \
      --feature-width 128 --hidden-dim 8 --kernel-size 3 --head-type mlp --norm group \
      --batch-size 32 --accumulation-steps 8 --subseq-len 64 --num-workers 8 --no-mmap &
  wait
else
echo "=== $TAG: fine-tune $SRC on $(grep -cv '^#' $LIST) NSTX-U discharges"
srun $RUN python -u finetune.py $RESUME $SEEDFLAG --source-checkpoint "$SRC" --machine nstx --checkpoint "$CKPT" \
    --train-list $LIST --val-list splits/nstxu_val.txt \
    --subseq-len 64 --batch-size 32 --accumulation-steps 8 --num-workers 8 \
    --probe-epochs 0 --backbone-lr 1e-3 --late-backbone-lr 1e-3 --head-lr 1e-3 \
    --epochs 300 --no-mmap &
wait
fi
for s in val test; do
  echo "=== $TAG: evaluation on the NSTX-U $s list"
  srun $RUN python -u eval_checkpoint.py --checkpoint "$CKPT" --list splits/nstxu_$s.txt \
      --dump-preds gallery/nstxu/${TAG}_$s.npz
  $RUN python -u -m cer_transfer.analysis.global_metrics --preds gallery/nstxu/${TAG}_$s.npz \
      > gallery/nstxu/${TAG}_$s.metrics.txt
done
