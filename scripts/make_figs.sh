#!/bin/bash
# make_figs.sh -- reproduce the manuscript figures from existing prediction
# dumps (no GPU needed). Run from anywhere; it changes to the repo root.
# A figure whose inputs are missing is skipped with a message.
#
# Model: the machine-agnostic NSTX model fine-tuned from DIII-D on all NSTX
# discharges (cer_ckpts/nstx_agn_ft_rfull.pt). Prerequisite dumps
# (eval_checkpoint.py --dump-preds, one discharge each):
#   gallery/137711/agnostic_ft.npz          Fig. 2
#   gallery/137711/calibration.txt          output of cer_transfer.analysis.calibrate
#                                           for that model (calibration factors k)
#   gallery/e116939.npz gallery/e115600.npz events (same model)
#   gallery/e116939_nstx_agn_ft_r325.npz gallery/e116939_nstx_agn_scratch_plain_r325.npz
#                                           transfer at 325 discharges
#   gallery/137711/passive.npz, active.npz  background vs foreground (unchanged models)
#   beam-off examples: no dumps; the DIII-D agnostic model (cer_ckpts/d3d_full_dg.pt)
#   runs zero-shot on the NSTX discharges (BEAMOFF_CK=<checkpoint> to change;
#   --machine nstx is added for any d3d* checkpoint, BEAMOFF_MACHINE overrides;
#   discharges: the two around the median beam-off phase length, or
#   BEAMOFF_SHOTS="142239 135660" to pin them; BEAMOFF_LIST selects the list the
#   discharges are taken from, default splits/nstx_val.txt; several lists allowed)
#   cer_ckpts/nstx_agn_{ft,scratch_plain}_r*.metrics.csv   scaling figure
# Chord coordinates chords_Rtan_<shot>.csv: in figs/ or gallery/ (CHORD_X=<dir>).
set -uo pipefail
cd "$(dirname "$0")/.."   # repository root
[ -f .env ] && source .env

RUN="pixi run --frozen python -u"
OFF="-0.235"              # experimental time of frame 0 (recording pre-trigger)
XL='$R_\mathrm{tan}$ (m)'

# --- Legend names (one convention for every figure that compares models) ---
#   DIII-D model                   = trained on DIII-D, applied to NSTX without NSTX training (zero-shot)
#   NSTX model / NSTX (N shots) model = trained from scratch on NSTX (N discharges)
#   DIII-D+NSTX (N shots) model    = DIII-D model fine-tuned on N NSTX discharges
NAME_ZS="DIII-D model"; NAME_SCR="NSTX model"; NAME_FT_FULL="DIII-D+NSTX (7307 shots) model"; NAME_FT_325="DIII-D+NSTX (325 shots) model"; NAME_SCR_325="NSTX (325 shots) model"
H=gallery/137711
if [ -n "${CHORD_X:-}" ]; then R=$CHORD_X
elif [ -f figs/chords_Rtan_137711.csv ]; then R=figs; else R=gallery; fi

have() { for f in "$@"; do [ -e "$f" ] || { echo "SKIP: missing $f"; return 1; }; done; }
shotfile() { grep -h "$1" "$2" 2>/dev/null | head -1 || true; }
# discharge file of a shot for the time axis (time_s): the regenerated chers_set5 copy if listed, else any split list
shotany() { local f; f=$(grep -h "$1" splits/nstx_set5.txt 2>/dev/null | head -1); [ -n "$f" ] && { echo "$f"; return; }
            grep -h "$1" splits/nstx_val.txt splits/nstx_train.txt splits/nstx_test.txt 2>/dev/null | head -1 || true; }

# calibration factors of the model (from calibrate's output)
K=""
if [ -f $H/calibration.txt ]; then
  KTI=$(awk '/^ti:/ {for (i=1;i<=NF;i++) if ($i=="k") print $(i+2)}' $H/calibration.txt)
  KVT=$(awk '/^vtor:/ {for (i=1;i<=NF;i++) if ($i=="k") print $(i+2)}' $H/calibration.txt)
  [ -n "$KTI" ] && [ -n "$KVT" ] && K="--k $KTI $KVT"
fi
[ -z "$K" ] && echo "note: no calibration factors ($H/calibration.txt); uncertainties uncalibrated"

L137711=$(shotany 137711); L137711=${L137711:-chers_nstx_labeled/chers_137711.joblib}
B137711=$(shotfile 137711 splits/nstx_passive_val.txt)

# --- Fig 2: reconstruction of discharge 137711 -----------------------------
have $H/agnostic_ft.npz $R/chords_Rtan_137711.csv && \
$RUN -m cer_transfer.figures.recon_composite --preds $H/agnostic_ft.npz \
    --shot "${CER_DATA_ROOT:-.}/$L137711" --label1 "$NAME_FT_FULL" --t-offset $OFF --unl-gap 2 $K \
    --profile-x $R/chords_Rtan_137711.csv --profile-xlabel "$XL" \
    --out figs --out-name recon_legacy_137711

# --- Fig 3: background-array vs foreground-array model (unchanged) ---------
have $H/passive.npz $H/active.npz && [ -n "$B137711" ] && \
$RUN -m cer_transfer.figures.recon_composite --preds $H/passive.npz --label1 "background model" \
    --preds2 $H/active.npz --label2 "foreground model" --shot "${CER_DATA_ROOT:-.}/$B137711" \
    --t-offset $OFF --unl-gap 2 --no-unl --out figs --out-name recon_background_vs_active_137711

# --- Events --------------------------------------------------------------------
# EVENT_STYLE=panels (default): profiles at four fitted times (same visual language
# as Fig. 2) plus the time trace of one core chord through the window, model every
# 5 ms with its band. EVENT_STYLE=3d: the ribbon plot, bands off, chords with fits
# only (screen use).
ES=${EVENT_STYLE:-panels}
event() {  # $1 preds  $2 t0  $3 t1  $4 shot  $5 out-name  $6 trace R (m)  [$7 zlim "lo hi"]  [$8 min-fit-frac]  [$9 calibration flags]  [$10 legend name]  [$11 "--times t1 t2 t3"]
  have "$1" $R/chords_Rtan_$4.csv || return 0
  SF=$(shotany $4)   # discharge file for the time axis (time_s), preferring the regenerated chers_set5 copy
  EF="${EFIT_DIR:-${CER_DATA_ROOT:-.}/chers_set5}/efit_$4.joblib"; [ -f "$EF" ] || EF=""   # equilibrium row when the EFIT file exists
  if [ "$ES" = 3d ]; then
    $RUN -m cer_transfer.figures.event_3d --preds "$1" --t0 $2 --t1 $3 --t-offset $OFF --target vtor $K \
        --no-band --min-fit-frac 0.8 ${7:+--zlim $7} --profile-x $R/chords_Rtan_$4.csv --profile-xlabel "$XL" \
        --out figs --out-name "$5"
  else
    $RUN -m cer_transfer.figures.event_panels --preds "$1" --t0 $2 --t1 $3 --t-offset $OFF --target vtor ${9:-$K} \
        --label1 "${10:-$NAME_FT_FULL}" ${11:---n-times 4} --trace-x $6 ${7:+--ylim $7} --min-fit-frac ${8:-0.5} ${SF:+--shot $SF} ${EF:+--efit $EF} \
        --profile-x $R/chords_Rtan_$4.csv --profile-xlabel "$XL" --out figs --out-name "$5"
  fi
}
T116939="--times 0.345 0.375 0.425"; T115600="--times 0.315 0.365 0.395"   # three profiles: before, during, after
event gallery/e116939.npz 0.340 0.435 116939 event_ntv_116939 1.10 "" 0.5 "$K" "$NAME_FT_FULL" "$T116939"
event gallery/e115600.npz 0.315 0.400 115600 event_tm_115600 1.00 "" 0.7 "$K" "$NAME_FT_FULL" "$T115600"   # edge chords with few fits drop out

# --- Alternative versions of Fig. 2 and the two events with the NSTX-only model
# (trained from scratch on all 7307 discharges): dumps gallery/137711/scratch_full.npz,
# gallery/e116939_scratch.npz, gallery/e115600_scratch.npz; calibration gallery/137711/calibration_scratch.txt
KS=""
if [ -f $H/calibration_scratch.txt ]; then
  KTI2=$(awk '/^ti:/ {for (i=1;i<=NF;i++) if ($i=="k") print $(i+2)}' $H/calibration_scratch.txt)
  KVT2=$(awk '/^vtor:/ {for (i=1;i<=NF;i++) if ($i=="k") print $(i+2)}' $H/calibration_scratch.txt)
  [ -n "$KTI2" ] && [ -n "$KVT2" ] && KS="--k ${KTI2%,} ${KVT2%,}"
fi
have $H/scratch_full.npz $R/chords_Rtan_137711.csv && \
$RUN -m cer_transfer.figures.recon_composite --preds $H/scratch_full.npz \
    --shot "${CER_DATA_ROOT:-.}/$L137711" --label1 "NSTX (7307 shots) model" --t-offset $OFF --unl-gap 2 $KS \
    --profile-x $R/chords_Rtan_137711.csv --profile-xlabel "$XL" \
    --out figs --out-name recon_legacy_137711_scratch
event gallery/e116939_scratch.npz 0.340 0.435 116939 event_ntv_116939_scratch 1.10 "" 0.5 "$KS" "NSTX (7307 shots) model" "$T116939"
event gallery/e115600_scratch.npz 0.315 0.400 115600 event_tm_115600_scratch 1.00 "" 0.7 "$KS" "NSTX (7307 shots) model" "$T115600"

# --- Transfer at 325 discharges: fine-tuned vs from scratch in ONE figure ------
# (manuscript include: event_transfer_r325_116939)
FTP=${FTP:-nstx_agn_ft}   # ladder prefix (track B: nstx_fixed_ft)
have gallery/e116939_${FTP}_r325.npz gallery/e116939_nstx_agn_scratch_plain_r325.npz $R/chords_Rtan_116939.csv && \
$RUN -m cer_transfer.figures.event_panels --preds gallery/e116939_${FTP}_r325.npz \
    --preds2 gallery/e116939_nstx_agn_scratch_plain_r325.npz \
    --label1 "$NAME_FT_325" --label2 "$NAME_SCR_325" --no-reference \
    --t0 0.340 --t1 0.435 --t-offset $OFF --target vtor $T116939 --trace-x 1.10 --ylim -10 270 $(SF=$(shotany 116939); echo ${SF:+--shot $SF}) \
    $(EF="${EFIT_DIR:-${CER_DATA_ROOT:-.}/chers_set5}/efit_116939.joblib"; [ -f "$EF" ] && echo "--efit $EF") \
    --profile-x $R/chords_Rtan_116939.csv --profile-xlabel "$XL" --out figs --out-name event_transfer_r325_116939

# --- Beam-off examples: model on the regular spectra while the beam is off --
# (phases found from the fits and line brightness of the same files; the model
# runs directly, no stored dumps; BEAMOFF_CK selects the checkpoint)
BO_CK=${BEAMOFF_CK:-cer_ckpts/d3d_full_dg.pt}   # zero-shot: DIII-D model on NSTX
# a DIII-D checkpoint must be applied with the NSTX preprocessing (the files
# are NSTX): without --machine, load_model uses the checkpoint's machine and
# the DIII-D flip/stretch would invert the rotation sign; the calibration
# factors K belong to nstx_agn_ft_rfull and are not applied to other models
BO_M=${BEAMOFF_MACHINE:-}; BO_K=$K
# PAPER: the DIII-D model alone (the beam-off claim is zero-shot). BEAMOFF_CK2=<checkpoint> adds
# a second model in the same panels (e.g. cer_ckpts/nstx_fixed_ft_rfull.pt, orange, calibrated)
BO_CK2=${BEAMOFF_CK2:-}; [ -z "$BO_CK2" ] || [ -f "$BO_CK2" ] || BO_CK2=""
case "$(basename "$BO_CK")" in d3d*) BO_M=${BO_M:-nstx}; BO_K="";; esac
# PAPER: discharges 142239 (train) and 135660 (test) — the beam-off phases inside a rotating
# plasma; the validation list has essentially none (survey with --min-rotation 50: one candidate,
# whose plotted chord has no fit). BEAMOFF_SHOTS="" + BEAMOFF_MIN_ROT=50 reproduces that survey.
# default list: the duplicated files with NBI power (splits/nstx_set5.txt -> chers_set5/), else the split lists
BO_DEF="splits/nstx_train.txt splits/nstx_test.txt"; [ -f splits/nstx_set5.txt ] && BO_DEF=splits/nstx_set5.txt
BO_LIST=${BEAMOFF_LIST:-$BO_DEF}
BEAMOFF_SHOTS=${BEAMOFF_SHOTS-142239 135660}
have $BO_CK $BO_LIST && \
$RUN -m cer_transfer.figures.beamoff_examples --checkpoint $BO_CK --list $BO_LIST \
    --n 2 --chord 2 --target vtor --min-off-ms 50 --margin-before 0.20 0.06 \
    --min-rotation ${BEAMOFF_MIN_ROT:-0} --context-ms 50 --min-fits-side 3 --rotation-on-chord \
    --t-offset $OFF $BO_K ${BO_M:+--machine $BO_M} ${BEAMOFF_SHOTS:+--shots $BEAMOFF_SHOTS} --label "$NAME_ZS" \
    ${BO_CK2:+--checkpoint2 $BO_CK2 --label2 "$NAME_FT_FULL" ${K:+--k2 ${K#--k }}} \
    --out figs --out-name beamoff_examples

# --- Baseline learning curves: trained from scratch on each machine ---------
# DIII-D: the full-data DG run (frozen at its best epoch 44); NSTX: plain recipe, all discharges
# DIII-D curve = the run of the paper's DIII-D model (track B: d3d_fixed_plain; track A: d3d_full_dg)
D3D_METRICS=${D3D_METRICS:-cer_ckpts/d3d_fixed_plain.metrics.csv}; [ -f "$D3D_METRICS" ] || D3D_METRICS=cer_ckpts/d3d_full_dg.metrics.csv
have $D3D_METRICS cer_ckpts/nstx_agn_scratch_plain_rfull.metrics.csv && \
$RUN -m cer_transfer.figures.plots curves --csv $D3D_METRICS \
    cer_ckpts/nstx_agn_scratch_plain_rfull.metrics.csv --labels "DIII-D (13151 shots) model" "NSTX (7307 shots) model" \
    --mark-best --drop-outliers --out figs --out-name learning_curves_val_score_baseline

# --- Scaling figure: fine-tuned from DIII-D vs from scratch, zero-shot line ---
# (the zero-shot line is the label-free protocol score, gallery/paper/zero_shot_val.csv
#  from physics_correction --csv, written by scripts/make_paper.sh)
FT=(); SC=()
for k in 325 650 1300 2600 5200 full; do
  n=$k; [ $k = full ] && n=7307
  [ -f cer_ckpts/${FTP:-nstx_agn_ft}_r$k.metrics.csv ]      && FT+=("$n:cer_ckpts/${FTP:-nstx_agn_ft}_r$k.metrics.csv")
  [ -f cer_ckpts/nstx_agn_scratch_plain_r$k.metrics.csv ] && SC+=("$n:cer_ckpts/nstx_agn_scratch_plain_r$k.metrics.csv")
done
Z=gallery/paper/zero_shot_val.csv
if [ ${#FT[@]} -gt 0 ]; then
  $RUN -m cer_transfer.figures.plots scaling --out figs --runs-label "DIII-D+NSTX model" \
      --scratch-label "$NAME_SCR" --zero-shot-label "$NAME_ZS" --ylabel "NSTX validation score" \
      --runs "${FT[@]}" ${SC:+--scratch "${SC[@]}"} \
      $([ -f $Z ] && echo "--zero-shot-csv $Z --zero-shot-column r2_protocol")
  [ -f $Z ] || echo "note: $Z missing - scaling figure without the zero-shot line (run scripts/make_paper.sh)"
else
  echo "SKIP: no metrics of the scaling runs"
fi

ls -la figs/recon_legacy_137711.pdf figs/event_ntv_116939.pdf figs/event_tm_115600.pdf \
    figs/event_transfer_r325_116939.pdf figs/beamoff_examples.pdf figs/scaling_val_score.pdf \
    figs/learning_curves_val_score_baseline.pdf 2>/dev/null
