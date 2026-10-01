#!/bin/bash
# make_figs.sh -- reproduce the manuscript figures from existing prediction
# dumps (no GPU needed). Run from anywhere; it changes to the repo root.
#
# Prerequisite dumps (eval_checkpoint.py --dump-preds, one discharge each):
#   gallery/137711/legacy.npz          51-chord NSTX model (nstx_ft)
#   gallery/137711/passive.npz         background-array model (nstx_passive_ft)
#   gallery/137711/active.npz          foreground-array model (nstx_active_ft)
#   gallery/e116939.npz  gallery/e115600.npz           events, nstx_ft
#   gallery/e116939_nstx_ft_r325.npz  gallery/e116939_nstx_scratch_r325.npz
#   gallery/bo_<shot>_bg.npz + _bg.txt + _fg.txt       beam-off pool
#     (scripts/run_passive_validation.sh produces them)
#   gallery/beamoff_verified*.csv, gallery/edge_pool_*.csv
# Tangency radii of the lines of sight: data/chord_radii/chords_Rtan_<shot>.csv
set -euo pipefail
cd "$(dirname "$0")/.."   # repository root

RUN="pixi run python -u"
OFF="-0.235"              # experimental time of frame 0 (recording pre-trigger)
K51="--k 0.992 1.171"     # calibration factors of the 51-chord NSTX model
XL='$R_\mathrm{tan}$ (m)'
H=gallery/137711
R=data/chord_radii

L137711=$(grep 137711 splits/nstx_val.txt)
B137711=$(grep 137711 splits/nstx_passive_val.txt)

# --- Fig 2: reconstruction of discharge 137711 (51-chord model) ---------
$RUN -m cer_transfer.figures.recon_composite --preds $H/legacy.npz --shot "$L137711" \
    --label1 model --t-offset $OFF --unl-gap 2 $K51 \
    --profile-x $R/chords_Rtan_137711.csv --profile-xlabel "$XL" \
    --out figs --out-name recon_legacy_137711

# --- Fig 3: background-array vs foreground-array model ------------------
$RUN -m cer_transfer.figures.recon_composite --preds $H/passive.npz --label1 "background model" \
    --preds2 $H/active.npz --label2 "foreground model" --shot "$B137711" \
    --t-offset $OFF --unl-gap 2 --no-unl --out figs --out-name recon_background_vs_active_137711

# --- Events: 3D waterfalls (model every frame, fits with error bars) -----
$RUN -m cer_transfer.figures.event_3d --preds gallery/e116939.npz --t0 0.340 --t1 0.435 \
    --t-offset $OFF --target vtor $K51 --profile-x $R/chords_Rtan_116939.csv \
    --profile-xlabel "$XL" --out figs --out-name event_ntv_116939
$RUN -m cer_transfer.figures.event_3d --preds gallery/e115600.npz --t0 0.315 --t1 0.400 \
    --t-offset $OFF --target vtor $K51 --profile-x $R/chords_Rtan_115600.csv \
    --profile-xlabel "$XL" --out figs --out-name event_tm_115600

# --- Transfer at 325 discharges: scratch vs fine-tuned, identical axes ---
for m in nstx_scratch_r325 nstx_ft_r325; do
  $RUN -m cer_transfer.figures.event_3d --preds gallery/e116939_$m.npz --t0 0.340 --t1 0.435 \
      --t-offset $OFF --target vtor --zlim -10 270 --profile-x $R/chords_Rtan_116939.csv \
      --profile-xlabel "$XL" --out figs --out-name event_${m/_r325/}_r325_116939
done

# --- Beam-off examples: representative discharges, phases >= 50 ms -------
$RUN -m cer_transfer.figures.beamoff_grid --verified gallery/beamoff_verified.csv \
    gallery/beamoff_verified_test.csv --pick median --min-notch-ms 50 \
    --edge-csv gallery/edge_pool_val.csv gallery/edge_pool_test.csv \
    --n 2 --chord 2 --target vtor --out figs --out-name beamoff_examples

# --- Per-discharge beam-off comparison (supplement) -----------------------
$RUN -m cer_transfer.figures.edge_scatter gallery/edge_pool_val.csv gallery/edge_pool_test.csv \
    --target vtor --log --out figs --out-name edge_scatter

# --- Scaling figure (needs the metrics CSVs of the scaling runs) ----------
# $RUN -m cer_transfer.figures.plots scaling --annotate --out figs \
#     --runs 325:<r325.csv> 650:<r650.csv> 1300:<r1300.csv> 2600:<r2600.csv> 5200:<r5200.csv> \
#     --scratch 325:0.664 650:0.734 1300:0.744 2600:0.755 5200:0.757 7307:0.764

ls -la figs/recon_legacy_137711.pdf figs/recon_background_vs_active_137711.pdf \
    figs/event_ntv_116939.pdf figs/event_tm_115600.pdf figs/beamoff_examples.pdf 2>/dev/null
