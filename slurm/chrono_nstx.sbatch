#!/bin/bash
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
# Chronological NSTX study: train on a window of consecutive campaigns, evaluate on
# every later campaign (splits from cer_transfer.datasets.make_chrono_splits).
#   WIN=FY2005-FY2006 EVAL="FY2007 FY2008 FY2009 FY2010" MODE=finetune|scratch
# Recipes as in the paper: finetune = d3d_fixed_dg_rt fine-tuned with lr 1e-3
# everywhere, no probe phase (nstx_rt_init_*); scratch = the same causal architecture
# without transfer augmentations (nstx_scratch64_*); subsequences of 64 frames.
# Outputs: cer_ckpts/nstx_chrono_<mode>_<WIN>.pt (+ .metrics.csv) and
# gallery/chrono/nstx_chrono_<mode>_<WIN>_on_<campaign>.{npz,metrics.txt}
#   sbatch --job-name=chrono_ft_FY2005 --export=ALL,WIN=FY2005,EVAL="FY2006 ...",MODE=finetune slurm/chrono_nstx.sbatch
module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..
mkdir -p slurm/logs gallery/chrono
WIN=${WIN:?WIN names the training window}; EVAL=${EVAL:?EVAL lists the later campaigns}
MODE=${MODE:-finetune}
SRC=${SRC:-cer_ckpts/d3d_fixed_dg_rt.pt}
SD=${SPLITS:-splits/chrono}
TAG=nstx_chrono_${MODE}_${WIN}
CKPT=cer_ckpts/$TAG.pt
exec >>"slurm/logs/$TAG.out" 2>>"slurm/logs/$TAG.err"
ENV=${ENV:-.env.fixed}; [ -f "$ENV" ] && source "$ENV"
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set}"
RUN="pixi run --frozen"
TRAIN=$SD/nstx_${WIN}_train.txt; VAL=$SD/nstx_${WIN}_val.txt
RESUME=""; [ -f "$CKPT" ] && RESUME="--resume"   # requeue: continue; finished runs are skipped
trap 'echo "USR1: requeue"; scontrol requeue $SLURM_JOB_ID' USR1
echo "=== $TAG: $MODE on $(grep -cv '^#' $TRAIN) discharges of $WIN (val $(grep -cv '^#' $VAL)), $(date -Is)"
if [ "$MODE" = scratch ]; then
  srun $RUN python -u train.py $RESUME --machine nstx --checkpoint "$CKPT" \
      --train-list $TRAIN --val-list $VAL --epochs 300 \
      --agnostic --chord-attention --chord-attention-depth 2 --resample-w 256 \
      --feature-width 128 --hidden-dim 8 --kernel-size 3 --head-type mlp --norm group \
      --batch-size 32 --accumulation-steps 8 --subseq-len 64 --num-workers 8 --no-mmap &
else
  srun $RUN python -u finetune.py $RESUME --source-checkpoint "$SRC" --machine nstx --checkpoint "$CKPT" \
      --train-list $TRAIN --val-list $VAL \
      --subseq-len 64 --batch-size 32 --accumulation-steps 8 --num-workers 8 \
      --probe-epochs 0 --backbone-lr 1e-3 --late-backbone-lr 1e-3 --head-lr 1e-3 \
      --epochs 300 --no-mmap &
fi
wait
for C in $EVAL; do
  OUT=gallery/chrono/${TAG}_on_$C
  [ -s $OUT.metrics.txt ] && continue
  echo "=== $TAG: evaluation on $C"
  srun $RUN python -u eval_checkpoint.py --checkpoint "$CKPT" --list $SD/nstx_$C.txt \
      --dump-preds $OUT.npz 2>&1 | grep -E "checkpoint:|overall"
  $RUN python -u -m cer_transfer.analysis.global_metrics --preds $OUT.npz > $OUT.metrics.tmp \
      && mv $OUT.metrics.tmp $OUT.metrics.txt
done
echo "=== $TAG: done $(date -Is)"
