#!/bin/bash
# Submit both agnostic scaling ladders, one job per rung (run from anywhere):
#   NSTX   -> DIII-D tangential: 325, 650, 1300, 2600, 5200 discharges, full set
#   DIII-D tangential -> NSTX:   500, 1000, 2000, 4000, 8000 discharges, full set
# Rungs whose training list is missing are skipped with a message; the DIII-D
# subsets come from python -m cer_transfer.datasets.make_d3d_subsets.
#   slurm/submit_agnostic_ladder.sh [nstx|d3d|both]     (default: both)
# Environment:
#   PREFIX=ladder        name prefix of the runs (checkpoints <PREFIX>_<machine>_r<K>)
#   RECIPE=dg            train exactly like the zero-shot DIII-D model: mirroring,
#                        RandConv, random shift (SHIFT bins, default 30), consistency
#                        loss, two attention blocks. Empty: the plain ladder recipe.
#   e.g. NSTX baselines for the zero-shot comparison:
#   PREFIX=base RECIPE=dg SHIFT=30 slurm/submit_agnostic_ladder.sh nstx
set -e
PREFIX=${PREFIX:-ladder}
if [ "${RECIPE:-}" = dg ]; then
  export MIRROR=0.5
  export DG="--randconv randconv --randconv-max-shift ${SHIFT:-30} --consistency-w 5 --chord-attention-depth 2"
  echo "recipe dg: MIRROR=$MIRROR DG=\"$DG\""
fi
cd "$(dirname "$0")/.."   # repository root
WHICH=${1:-both}
[ -f splits/d3d_val.txt ] || { echo "missing splits/d3d_val.txt"; exit 1; }
[ -f splits/nstx_val.txt ] || { echo "missing splits/nstx_val.txt"; exit 1; }

submit() {   # tag src tgt val_list tgt_list train_list
  local list=$6
  if [ -n "$list" ] && [ ! -f "$list" ]; then echo "skip $1: $list missing"; return; fi
  sbatch --export=ALL,TAG=$1,SRC_MACHINE=$2,TGT_MACHINE=$3,VAL_LIST=$4,TGT_LIST=$5,TRAIN_LIST=$list \
      slurm/agnostic_ladder.sbatch | sed "s/$/  ($1)/"
}
if [ "$WHICH" != d3d ]; then
  for K in 325 650 1300 2600 5200; do
    submit ${PREFIX}_nstx_r$K nstx d3d_tangential splits/nstx_val.txt splits/d3d_val.txt splits/nstx_train_r$K.txt
  done
  submit ${PREFIX}_nstx_full nstx d3d_tangential splits/nstx_val.txt splits/d3d_val.txt splits/nstx_train.txt
fi
if [ "$WHICH" != nstx ]; then
  for K in 500 1000 2000 4000 8000; do
    submit ${PREFIX}_d3d_r$K d3d_tangential nstx splits/d3d_val.txt splits/nstx_val.txt splits/d3d_train_r$K.txt
  done
  submit ${PREFIX}_d3d_full d3d_tangential nstx splits/d3d_val.txt splits/nstx_val.txt ""
fi
