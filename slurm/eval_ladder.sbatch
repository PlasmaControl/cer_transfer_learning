#!/bin/bash
#SBATCH --job-name=cer_eval_ladder
#SBATCH --output=/dev/null   # redirected below into slurm/logs/
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ps9551@princeton.edu
# Re-evaluate the agnostic ladder (no training): one array task per rung, in
# parallel. Own-machine and zero-shot evaluation with the current machine
# conventions, CSVs in gallery/ladder/ for cer_transfer.analysis.agnostic_ladder.
#   sbatch --export=ALL,SRC=nstx slurm/eval_ladder.sbatch       (6 NSTX rungs)
#   sbatch --export=ALL,SRC=d3d  --array=0-5 slurm/eval_ladder.sbatch
#   EVALS="zero_shot" to skip the own-machine evaluation
module purge
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f train.py ] || cd ..   # repo root (submit from root or slurm/)
mkdir -p slurm/logs gallery/ladder
SRC=${SRC:-nstx}
EVALS=${EVALS:-"own zero_shot"}
LIMIT=${LIMIT:-200}
if [ "$SRC" = nstx ]; then
  RUNGS=(r325 r650 r1300 r2600 r5200 full); NFULL=7307
  SRC_M=nstx; TGT_M=d3d_tangential; OWN_LIST=splits/nstx_val.txt; TGT_LIST=splits/d3d_val.txt
else
  RUNGS=(r500 r1000 r2000 r4000 r8000 full); NFULL=""
  SRC_M=d3d_tangential; TGT_M=nstx; OWN_LIST=splits/d3d_val.txt; TGT_LIST=splits/nstx_val.txt
fi
K=${RUNGS[$SLURM_ARRAY_TASK_ID]}
T=ladder_${SRC}_$K
LOG="slurm/logs/${SLURM_JOB_NAME}_${T}"
exec >>"$LOG.out" 2>>"$LOG.err"
[ -f .env ] && source .env            # CER_DATA_ROOT etc., written by scripts/setup_env.sh
: "${CER_DATA_ROOT:?CER_DATA_ROOT is not set: run scripts/setup_env.sh --data-root <dir> or export it}"
RUN="pixi run --frozen"
CKPT=cer_ckpts/$T.pt
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
if [ "$K" = full ]; then
  N=${NFULL:-$($RUN python -c "from cer_transfer.configs import data_path, get_machine
print(len(list(data_path(get_machine('$SRC_M').train_dir).glob('*.joblib'))))")}
else
  N=${K#r}
fi
echo "=== $T ($N training discharges), evaluations: $EVALS"
for EV in $EVALS; do
  if [ "$EV" = own ]; then M=$SRC_M; L=$OWN_LIST; MFLAG=""
  else M=$TGT_M; L=$TGT_LIST; MFLAG="--machine $TGT_M"; fi
  srun $RUN python -u eval_checkpoint.py --checkpoint "$CKPT" $MFLAG --list $L \
      --limit $LIMIT --dump-preds gallery/${T}_$EV.npz
  srun $RUN python -u -m cer_transfer.analysis.affine_fit --preds gallery/${T}_$EV.npz \
      --csv gallery/ladder/${T}_$EV.csv \
      --meta tag=$T eval=$EV src=$SRC_M tgt=$M n_train=$N ckpt=$CKPT
done
