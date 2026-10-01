#!/bin/bash
# run_passive_validation.sh -- beam-off / beam-free validation on the
# requested shots. CPU is fine (predict-only); run from the repo root.
#
#   bash run_passive_validation.sh              # everything
#   LOCK_SHOT=<ramp shot> T_LOCK=0.168 bash run_passive_validation.sh
#
# Inputs: joblib files in the three requested directories (48-chord
# background / foreground arrays; the 51-chord legacy files are listed
# for the sanity check only).
set -uo pipefail
cd "$(dirname "$0")/.."   # repository root

RUN="pixi run python -u"
BASE=${CER_DATA_ROOT:?set CER_DATA_ROOT to the data root}
D51=$BASE/chers_requested
DFG=$BASE/chers_requested_active
DBG=$BASE/chers_requested_passive
CK_BG=cer_ckpts/nstx_passive_ft.pt
CK_FG=cer_ckpts/nstx_active_ft.pt
OFF=-0.235
EDGE_SHOTS=$(seq 115517 115526)
LM_SHOTS="117192 117193"
mkdir -p gallery figs

find_file() {  # $1 dir, $2 shot, $3 split-list glob -> joblib path or empty
  local f
  f=$(ls "$1"/*"$2"*.joblib 2>/dev/null | head -1)
  if [ -z "$f" ] && [ -n "${3:-}" ]; then   # fallback: regular dataset
    f=$(grep -h "$2" $3 2>/dev/null | head -1)
  fi
  echo "$f"
}

echo "=== 0. split membership (training-set shots would leak into the edge test) ==="
for s in $EDGE_SHOTS $LM_SHOTS; do
  echo "$s: $(grep -l "$s" splits/nstx_*.txt 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')"
done

echo "=== 1. sanity: chords, frames, fitted points, plasma window ==="
for s in $EDGE_SHOTS $LM_SHOTS; do
  for pair in "$DBG|splits/nstx_passive_*.txt" "$DFG|splits/nstx_active_*.txt"; do
    d=${pair%%|*}; lg=${pair#*|}
    f=$(find_file "$d" "$s" "$lg")
    [ -n "$f" ] || { echo "$s: MISSING in $d and in $lg"; continue; }
    $RUN - "$f" <<'EOF'
import sys
import numpy as np
from joblib import load
p = sys.argv[1]
d = load(p, mmap_mode="r"); e = int(d["end_index"])
tot = np.asarray(d["input"][:, :e], np.float32).sum(axis=(0, 2))
dk = np.median(tot[:45]); nz = 1.4826 * np.median(np.abs(tot[:45] - dk))
act = np.where(tot > dk + 8 * nz)[0]; act = act[act >= 47]
win = (f"{act.min()/200-0.235:.3f}-{act.max()/200-0.235:.3f} s"
       if act.size else "none")
print(f"  {p.split('/')[-1]:32s} chords {d['input'].shape[0]:2d} | "
      f"end_index {e} | fitted points "
      f"{int(np.isfinite(d['target'][..., 0]).sum())} | plasma {win}")
EOF
  done
done

echo "=== 2. predictions (both arrays, predict-only) ==="
for s in $EDGE_SHOTS $LM_SHOTS; do
  fb=$(find_file "$DBG" "$s" "splits/nstx_passive_*.txt")
  ff=$(find_file "$DFG" "$s" "splits/nstx_active_*.txt")
  [ -n "$fb" ] && [ -n "$ff" ] || { echo "$s: skip (missing array file)"; continue; }
  echo "$fb" > gallery/rq_${s}_bg.txt
  echo "$ff" > gallery/rq_${s}_fg.txt
  [ -f gallery/rq_${s}_bg.npz ] || $RUN eval_checkpoint.py --checkpoint $CK_BG \
      --list gallery/rq_${s}_bg.txt --dump-preds gallery/rq_${s}_bg.npz --predict-only
  [ -f gallery/rq_${s}_fg.npz ] || $RUN eval_checkpoint.py --checkpoint $CK_FG \
      --list gallery/rq_${s}_fg.txt --dump-preds gallery/rq_${s}_fg.npz --predict-only
done

echo "=== 3. edge-frame test (115517-115526) ==="
rm -f gallery/edge_pool.csv
for s in $EDGE_SHOTS; do
  [ -f gallery/rq_${s}_bg.npz ] || continue
  $RUN -m cer_transfer.beamoff.edge_frame_test --shot $s --bg gallery/rq_${s}_bg.npz \
      --bg-file "$(cat gallery/rq_${s}_bg.txt)" --fg-file "$(cat gallery/rq_${s}_fg.txt)" \
      --csv gallery/edge_pool.csv
done
if [ -f gallery/edge_pool.csv ]; then
  $RUN -m cer_transfer.beamoff.pool_edge gallery/edge_pool.csv
else
  echo "NO edge-test results: none of $EDGE_SHOTS had both array files (see step 2)"
fi

echo "=== 4. locked-mode figures (117192/3) ==="
# T_LOCK marks the locking onset; LOCK_SHOT = the shot with the applied
# n=1 ramp (the reference shot does not lock and gets no marker)
if [ -n "${T_LOCK:-}" ] && [ -z "${LOCK_SHOT:-}" ]; then
  echo "WARNING: T_LOCK set without LOCK_SHOT -- marker omitted; set LOCK_SHOT to the ramp shot"
fi
for s in $LM_SHOTS; do
  [ -f gallery/rq_${s}_bg.npz ] && [ -f gallery/rq_${s}_fg.npz ] || continue
  MARK=()
  if [ -n "${T_LOCK:-}" ] && [ "${LOCK_SHOT:-}" = "$s" ]; then
    MARK=(--mark "${T_LOCK}:mode locking")
  fi
  $RUN -m cer_transfer.figures.beamoff_traces --bg gallery/rq_${s}_bg.npz --fg gallery/rq_${s}_fg.npz \
      --bg-file "$(cat gallery/rq_${s}_bg.txt)" --fg-file "$(cat gallery/rq_${s}_fg.txt)" \
      --t0 0.05 --t1 0.30 --t-offset $OFF --chords 5,15,25 --targets ti,vtor \
      ${MARK[@]+"${MARK[@]}"} --out figs --out-name lockedmode_${s}
done
echo "done: gallery/edge_pool.csv, figs/lockedmode_*.pdf"
