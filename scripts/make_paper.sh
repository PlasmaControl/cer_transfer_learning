#!/bin/bash
# make_paper.sh -- ONE command for every evaluation, prediction dump and figure of
# the paper, from the frozen checkpoints. Idempotent: dumps and evaluations that
# exist are reused (FORCE=1 recomputes them); figures are always redrawn.
#
#   scripts/make_paper.sh            # everything except the test-set evaluation
#   TEST=1 scripts/make_paper.sh     # also the ONE test-set evaluation (refuses to
#                                    # repeat it once gallery/paper/zero_shot_test.txt exists)
#   FORCE=1 scripts/make_paper.sh    # recompute dumps/evaluations (not the test set)
#
# Inputs (paper track, data root from .env; nothing from the 'fixed' track):
#   D3D   cer_ckpts/d3d_full_dg_e44.pt          DIII-D model (epoch-44 snapshot)
#   NSTX  cer_ckpts/nstx_agn_ft_rfull.pt        fine-tuned from D3D on all NSTX discharges
#   cer_ckpts/nstx_agn_ft_r*.metrics.csv, nstx_agn_scratch_plain_r*.metrics.csv   scaling
#   gallery/agnostic_scaling/nstx_agn_ft_zero_shot.csv                              zero-shot point
#   gallery/agnostic_scaling/nstx_agn_ft_rfull.npz   (complete val dump of NSTX, from the ladder job)
#   gallery/instrument_width_{d3d,nstx}.csv          (cer_transfer.analysis.instrument_width)
# Outputs: gallery/paper/*.txt (all numbers), gallery/*.npz dumps, figs/*.pdf, and
# gallery/paper/SUMMARY.md listing what was produced and what is still missing.
# Needs a GPU for the dumps/evaluations (an interactive GPU shell, or sbatch this script).
set -uo pipefail
cd "$(dirname "$0")/.."
ENV=${ENV:-.env}; [ -f "$ENV" ] && source "$ENV"   # ENV=.env.fixed with D3D=cer_ckpts/d3d_fixed_*.pt (track B)
: "${CER_DATA_ROOT:?CER_DATA_ROOT not set (source .env)}"

RUN="pixi run --frozen python -u"
D3D=${D3D:-cer_ckpts/d3d_full_dg_e44.pt}
NSTX=${NSTX:-cer_ckpts/nstx_agn_ft_rfull.pt}
# ladder prefix follows the NSTX model name (nstx_agn_ft_rfull -> nstx_agn_ft; nstx_fixed_ft_rfull -> nstx_fixed_ft)
FTP=$(basename "$NSTX" .pt); FTP=${FTP%_rfull}
FT325=cer_ckpts/${FTP}_r325.pt
SC325=cer_ckpts/nstx_agn_scratch_plain_r325.pt
# DIII-D convention of the zero-shot protocol: 1.4 for the old data (track A), 2.842 for the rebuilt files (track B)
case "$(basename "$D3D")" in d3d_fixed*) STRETCH=2.842; SIGSRC=gallery/instrument_width_d3d_fixed.csv;; *) STRETCH=1.4; SIGSRC=gallery/instrument_width_d3d.csv;; esac
case "$D3D$CER_DATA_ROOT" in *fixed*fixed*|*full_dg*data_root) ;; *) echo "D3D=$D3D does not match CER_DATA_ROOT=$CER_DATA_ROOT (track mix-up)"; exit 1;; esac
P=gallery/paper; mkdir -p $P gallery/137711 figs
SUMMARY=$P/SUMMARY.md; : > $SUMMARY
log() { echo "$*"; echo "$*" >> $SUMMARY; }
have() { for f in "$@"; do [ -e "$f" ] || { log "MISSING: $f"; return 1; }; done; }
fresh() { [ -n "${FORCE:-}" ] || [ ! -e "$1" ]; }   # compute if forced or absent
onelist() { # list file with the one entry of shot $1 (from any NSTX list)
  local f=$P/list_$1.txt; grep -h "$1" splits/nstx_val.txt splits/nstx_train.txt splits/nstx_test.txt 2>/dev/null | head -1 > $f
  [ -s $f ] && echo $f || { log "MISSING: discharge $1 in splits/"; return 1; }
}
GPU=1; command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || GPU=""
[ -z "$GPU" ] && log "note: no GPU visible - dumps/evaluations that do not exist yet are skipped"

log "# Paper build $(date -Is)"; log "D3D=$D3D  NSTX=$NSTX  data root $CER_DATA_ROOT"; log ""

# ---------------------------------------------------------------------------
log "## 1. Zero-shot DIII-D -> NSTX (complete validation list)"
# the ladder's task 0 (agnostic_scaling.sbatch --array=0) writes this dump; computed here if absent
ZSV=gallery/agnostic_scaling/${FTP}_zero_shot.npz
if have $D3D && [ -n "$GPU" ] && fresh $ZSV; then
  mkdir -p gallery/agnostic_scaling
  $RUN eval_checkpoint.py --checkpoint $D3D --machine nstx --list splits/nstx_val.txt \
      --dump-preds $ZSV 2>&1 | grep -E "checkpoint:|overall" | tee $P/zero_shot_val_raw.txt
fi
if have $ZSV gallery/instrument_width_d3d.csv gallery/instrument_width_nstx.csv; then
  $RUN -m cer_transfer.analysis.physics_correction --preds $ZSV \
      --src-window-nm 6.0 --src-stretch $STRETCH --tgt-window-nm 2.08 \
      --sigma-src $SIGSRC --sigma-tgt gallery/instrument_width_nstx.csv \
      --csv $P/zero_shot_val.csv | tee $P/zero_shot_val.txt
  grep -E "^(ti|vtor):" $P/zero_shot_val.txt | sed 's/^/    /' >> $SUMMARY
fi

# ---------------------------------------------------------------------------
log "## 2. Zero-shot on the NSTX TEST set (once per DIII-D model)"
DTAG=$(basename "$D3D" .pt)
ZST=gallery/zs_${DTAG}_test.npz
ZTXT=$P/zero_shot_test_${DTAG}.txt
if [ -n "${TEST:-}" ]; then
  if [ -e $ZTXT ]; then
    log "already done for $DTAG ($ZTXT) - not repeated; delete it deliberately to redo"
  elif have $D3D && [ -n "$GPU" ]; then
    $RUN eval_checkpoint.py --checkpoint $D3D --machine nstx --list splits/nstx_test.txt \
        --dump-preds $ZST 2>&1 | grep -E "checkpoint:|overall" | tee $P/zero_shot_test_raw_${DTAG}.txt
    $RUN -m cer_transfer.analysis.physics_correction --preds $ZST \
        --src-window-nm 6.0 --src-stretch $STRETCH --tgt-window-nm 2.08 \
        --sigma-src $SIGSRC --sigma-tgt gallery/instrument_width_nstx.csv \
        | tee $ZTXT
  fi
else
  log "skipped (TEST=1 runs it)"
fi
[ -e $P/zero_shot_test_raw_${DTAG}.txt ] && sed 's/^/    raw: /' $P/zero_shot_test_raw_${DTAG}.txt >> $SUMMARY
[ -e $ZTXT ] && grep -E "^(ti|vtor):" $ZTXT | sed 's/^/    /' >> $SUMMARY

# ---------------------------------------------------------------------------
log "## 3. Prediction dumps of the NSTX model (one discharge each)"
if have $NSTX && [ -n "$GPU" ]; then
  for spec in "137711:gallery/137711/agnostic_ft.npz" "116939:gallery/e116939.npz" "115600:gallery/e115600.npz"; do
    shot=${spec%%:*}; out=${spec#*:}
    fresh $out || continue
    L=$(onelist $shot) || continue
    $RUN eval_checkpoint.py --checkpoint $NSTX --list $L --dump-preds $out 2>&1 | grep -E "overall" | sed "s/^/$shot: /"
  done
  # the same three discharges with the NSTX-only model (from scratch, all discharges): alternative figures
  SCRF=cer_ckpts/nstx_agn_scratch_plain_rfull.pt
  if have $SCRF; then
    for spec in "137711:gallery/137711/scratch_full.npz" "116939:gallery/e116939_scratch.npz" "115600:gallery/e115600_scratch.npz"; do
      shot=${spec%%:*}; out=${spec#*:}
      fresh $out || continue
      L=$(onelist $shot) || continue
      $RUN eval_checkpoint.py --checkpoint $SCRF --list $L --dump-preds $out 2>&1 | grep -E "overall" | sed "s/^/$shot (NSTX only): /"
    done
  fi
  # transfer at 325 discharges on 116939
  for spec in "$FT325:gallery/e116939_${FTP}_r325.npz" "$SC325:gallery/e116939_nstx_agn_scratch_plain_r325.npz"; do
    ck=${spec%%:*}; out=${spec#*:}
    have $ck || continue; fresh $out || continue
    L=$(onelist 116939) || continue
    $RUN eval_checkpoint.py --checkpoint $ck --list $L --dump-preds $out 2>&1 | grep -E "overall" | sed "s|^|$(basename $ck): |"
  done
fi
# zero-shot DIII-D model on 115600 (rotation figure)
ZS115600=gallery/zs_$(basename "$D3D" .pt)_115600.npz
if have $D3D && [ -n "$GPU" ] && fresh $ZS115600; then
  L=$(onelist 115600) && $RUN eval_checkpoint.py --checkpoint $D3D --machine nstx --list $L --dump-preds $ZS115600 2>&1 | grep overall | sed 's/^/zero-shot 115600: /'
fi
for f in gallery/137711/agnostic_ft.npz gallery/e116939.npz gallery/e115600.npz gallery/e116939_nstx_agn_ft_r325.npz gallery/e116939_nstx_agn_scratch_plain_r325.npz $ZS115600; do
  [ -e $f ] && log "ok  $f" || log "MISSING: $f"
done

# ---------------------------------------------------------------------------
log "## 4. Calibration, coverage, label ceilings (NSTX model)"
CAL=gallery/137711/calibration.txt
if have $NSTX && [ -n "$GPU" ] && fresh $CAL; then
  $RUN -m cer_transfer.analysis.calibrate --checkpoint $NSTX --list splits/nstx_val.txt --limit 100 | tee $CAL
fi
[ -e $CAL ] && grep -E "^(ti|vtor):" $CAL | sed 's/^/    /' >> $SUMMARY
CAL2=gallery/137711/calibration_scratch.txt
if have cer_ckpts/nstx_agn_scratch_plain_rfull.pt && [ -n "$GPU" ] && fresh $CAL2; then
  $RUN -m cer_transfer.analysis.calibrate --checkpoint cer_ckpts/nstx_agn_scratch_plain_rfull.pt --list splits/nstx_val.txt --limit 100 | tee $CAL2
fi
VALDUMP=gallery/agnostic_scaling/${FTP}_rfull.npz
if have $VALDUMP $CAL; then
  KTI=$(awk '/^ti:/ {for (i=1;i<=NF;i++) if ($i=="k") print $(i+2)}' $CAL | tr -d ',')
  KVT=$(awk '/^vtor:/ {for (i=1;i<=NF;i++) if ($i=="k") print $(i+2)}' $CAL | tr -d ',')
  $RUN -m cer_transfer.analysis.coverage --preds $VALDUMP --k ${KTI:-1} ${KVT:-1} | tee $P/coverage.txt
  grep -A4 "LaTeX rows" $P/coverage.txt | sed 's/^/    /' >> $SUMMARY
fi
if have $VALDUMP; then
  $RUN -m cer_transfer.analysis.global_metrics --preds $VALDUMP | tee $P/global_metrics.txt
  grep -A6 "LaTeX rows" $P/global_metrics.txt | sed 's/^/    /' >> $SUMMARY
fi
if fresh $P/ceilings.txt && [ -n "$GPU" ]; then   # CPU-only but slow: run with the GPU session
  $RUN -m cer_transfer.analysis.ceiling_check --machine nstx --list splits/nstx_val.txt --honest | tee $P/ceilings.txt
fi
[ -e $P/ceilings.txt ] && log "ceilings: $P/ceilings.txt"

# ---------------------------------------------------------------------------
log "## 5. Scaling table (best val_score per rung)"
$RUN - <<'EOF' | tee $P/scaling_table.txt | sed 's/^/    /' >> $SUMMARY
import pandas as pd, os
rows = []
for k, n in (("325",325),("650",650),("1300",1300),("2600",2600),("5200",5200),("full",7307)):
    ft = f"cer_ckpts/{os.environ.get('FTP','nstx_agn_ft')}_r{k}.metrics.csv"; sc = f"cer_ckpts/nstx_agn_scratch_plain_r{k}.metrics.csv"
    def b(p):
        if not os.path.exists(p): return "  -  "
        d = pd.read_csv(p)
        col = next((c for c in ("val_score", "score", "val_combined", "combined_score") if c in d.columns), None)
        if col is None:  # fall back to the first column that looks like a score
            col = next((c for c in d.columns if "score" in c.lower()), None)
        return f"{d[col].max():.3f} [{col}]" if col else f"?? columns: {list(d.columns)}"
    rows.append(f"{n:>5} | fine-tuned {b(ft)} | plain scratch {b(sc)}")
z = "gallery/paper/zero_shot_val.csv"
if os.path.exists(z):
    d = pd.read_csv(z).set_index("target")
    rows.insert(0, f"    0 | zero-shot  {d.r2_protocol.mean():.3f} protocol (Ti {d.r2_protocol.ti:.3f}, vtor {d.r2_protocol.vtor:.3f}); "
                   f"raw mean {d.r2_raw.mean():.3f} (Ti {d.r2_raw.ti:.3f}, vtor {d.r2_raw.vtor:.3f})")
else:
    rows.insert(0, "    0 | zero-shot    -    (gallery/paper/zero_shot_val.csv missing)")
print("\n".join(rows))
EOF

# ---------------------------------------------------------------------------
log "## 6. Figures"
export BEAMOFF_CK=$D3D FTP   # lists and discharges: defaults in make_figs.sh (the paper's choice)
scripts/make_figs.sh 2>&1 | grep -E "wrote|SKIP|note|using:|no beam-off" | tee $P/figs.log | sed 's/^/    /' >> $SUMMARY
# zero-shot rotation figure: DIII-D model vs NSTX model on 115600
if have $ZS115600 gallery/e115600.npz; then
  R=gallery; [ -f figs/chords_Rtan_115600.csv ] && R=figs
  L=$(grep -h 115600 splits/nstx_val.txt | head -1)
  $RUN -m cer_transfer.figures.recon_composite --preds $ZS115600 --label1 "DIII-D model" \
      --preds2 gallery/e115600.npz --label2 "DIII-D+NSTX (7307 shots) model" --shot "$CER_DATA_ROOT/$L" \
      --t-offset -0.235 --unl-gap 2 ${R:+--profile-x $R/chords_Rtan_115600.csv} --profile-xlabel '$R_\mathrm{tan}$ (m)' \
      --out figs --out-name zero_shot_vtor_115600 2>&1 | grep -E "wrote|rror" | sed 's/^/    /' | tee -a $SUMMARY
fi
# (scaling figure incl. the zero-shot line and the baseline learning curves: drawn by make_figs.sh above)

log ""; log "## Result"
ls figs/*.pdf 2>/dev/null | sed 's/^/    /' >> $SUMMARY
log "$(grep -c MISSING $SUMMARY) missing inputs (grep MISSING $SUMMARY)"
echo; cat $SUMMARY
