#!/bin/bash
# make_handoff.sh -- one self-contained ZIP for a colleague: code, trained models
# (inference-only), example discharges of DIII-D and NSTX, and a quick start.
#
#   scripts/make_handoff.sh                       # -> cer_handoff_<date>.zip in the repo root
#   D3D_SHOTS="190123 190456" scripts/make_handoff.sh   # choose the DIII-D examples
#
# Contents of the ZIP:
#   cer_transfer_learning/   the repository at HEAD (git archive), incl. pixi.toml/pixi.lock
#   models/                  d3d_fixed_plain_best.pt, nstx_fixed_ft_rfull.pt, nstx_fixed_ft_r325.pt
#   data/nstx/               the five NSTX discharges of the paper (+ EFIT files)
#   data/d3d/                a few DIII-D discharges (rebuilt files, one wavelength grid)
#   data/chords_Rtan_*.csv   chord tangency radii for the profile plots
#   lists/                   nstx_demo.txt, d3d_demo.txt (paths relative to data/)
#   QUICKSTART.md
set -eu
trap 'echo "make_handoff.sh: failed at line $LINENO" >&2' ERR
cd "$(dirname "$0")/.."
[ -f .env.fixed ] || { echo "missing .env.fixed (data root of the rebuilt DIII-D files)" >&2; exit 1; }
source .env.fixed   # DIII-D examples come from the rebuilt files

STAMP=$(date +%Y%m%d)
OUT=${OUT:-$PWD/cer_handoff_$STAMP}
NSTX_SRC=${NSTX_SRC:-/scratch/gpfs/EKOLEMEN/ps9551/chers_set5}
# default DIII-D examples: the first three discharges of the rebuilt DIII-D test folder
D3D_DIR=$CER_DATA_ROOT/test_set_30
[ -d "$D3D_DIR" ] || { echo "no DIII-D folder $D3D_DIR" >&2; exit 1; }
if [ -z "${D3D_SHOTS:-}" ]; then
  D3D_SHOTS=$(find -L "$D3D_DIR" -maxdepth 1 -name '[0-9]*.joblib' -printf '%f\n' | sort | sed -n '1,3p' | sed 's/\.joblib$//' | tr '\n' ' ')
fi
echo "output:   $OUT.zip"
echo "NSTX:     $NSTX_SRC"
echo "DIII-D:   $D3D_DIR  ($D3D_SHOTS)"

rm -rf "$OUT"; mkdir -p "$OUT"/{models,data/nstx,data/d3d,lists}

echo "== code"
git archive --format=tar --prefix=cer_transfer_learning/ HEAD | tar -x -C "$OUT"

echo "== models (weights, configuration and normalization only)"
pixi run --frozen python - "$OUT/models" <<'EOF'
import sys
from cer_transfer.inference import export_inference_checkpoint as ex
out = sys.argv[1]
for n in ("d3d_fixed_plain_best", "nstx_fixed_ft_rfull", "nstx_fixed_ft_r325"):
    ex(f"cer_ckpts/{n}.pt", f"{out}/{n}.pt")
    print("  ", n)
EOF

echo "== data: NSTX (paper discharges + EFIT)"
cp "$NSTX_SRC"/chers_*.joblib "$NSTX_SRC"/efit_*.joblib "$OUT/data/nstx/"
ls "$OUT/data/nstx" | grep '^chers_' | sed 's|^|nstx/|' > "$OUT/lists/nstx_demo.txt"
cp figs/chords_Rtan_*.csv "$OUT/data/" 2>/dev/null || true

echo "== data: DIII-D ($D3D_SHOTS)"
for s in $D3D_SHOTS; do cp "$D3D_DIR/$s.joblib" "$OUT/data/d3d/"; echo "d3d/$s.joblib" >> "$OUT/lists/d3d_demo.txt"; done

cat > "$OUT/QUICKSTART.md" <<'EOF'
# Quick start

Needs: a Linux machine with pixi (https://pixi.sh), e.g. Stellar; a GPU is optional.

```bash
cd cer_transfer_learning
pixi install                                            # exact environment from pixi.lock
echo "export CER_DATA_ROOT=$(cd ../data && pwd)" > .env && source .env

# NSTX: the fine-tuned model on the five paper discharges (R2, RMSE per target)
pixi run --frozen python -u eval_checkpoint.py --checkpoint ../models/nstx_fixed_ft_rfull.pt \
    --list ../lists/nstx_demo.txt 2>&1 | grep -E "checkpoint:|overall"

# NSTX without any NSTX training: the DIII-D model applied directly (zero-shot)
pixi run --frozen python -u eval_checkpoint.py --checkpoint ../models/d3d_fixed_plain_best.pt \
    --machine nstx --list ../lists/nstx_demo.txt 2>&1 | grep -E "checkpoint:|overall"

# DIII-D: the DIII-D model on the example discharges
pixi run --frozen python -u eval_checkpoint.py --checkpoint ../models/d3d_fixed_plain_best.pt \
    --list ../lists/d3d_demo.txt 2>&1 | grep -E "checkpoint:|overall"

# FIGURES: time traces and profiles of one discharge -> PNG
pixi run --frozen python examples/plot_discharge.py --checkpoint ../models/nstx_fixed_ft_rfull.pt \
    --shot nstx/chers_137711.joblib --profile-x ../data/chords_Rtan_137711.csv --out nstx_137711.png
pixi run --frozen python examples/plot_discharge.py --checkpoint ../models/d3d_fixed_plain_best.pt \
    --machine nstx --shot nstx/chers_116939.joblib --profile-x ../data/chords_Rtan_116939.csv \
    --out nstx_116939_zero_shot.png                      # DIII-D model on NSTX, no NSTX training
pixi run --frozen python examples/plot_discharge.py --checkpoint ../models/d3d_fixed_plain_best.pt \
    --shot $(head -1 ../lists/d3d_demo.txt) --chords 5 20 --out d3d_example.png

# interactive: the notebook
pixi run jupyter lab notebooks/demo.ipynb
```

In the notebook, set MODE = "local", the checkpoint to `../models/nstx_fixed_ft_rfull.pt`
and the discharge to `nstx/chers_137711.joblib` (relative to CER_DATA_ROOT).

Models
- `d3d_fixed_plain_best.pt`: trained on 13 143 DIII-D discharges (validation score 0.876).
- `nstx_fixed_ft_rfull.pt`: the DIII-D model fine-tuned on all 7307 NSTX training discharges.
- `nstx_fixed_ft_r325.pt`: the same, fine-tuned on 325 NSTX discharges.

Data files are joblib dicts: `input` (chords x frames x wavelength bins, raw counts),
`target` and `target_error` (chords x frames x [Ti in keV for NSTX / eV for DIII-D, v_tor km/s],
NaN where no conventional fit exists), `end_index`; NSTX files also carry `time_s` and
`nbi_power_MW`, DIII-D files `wavelength_nm`. `efit_<shot>.joblib`: EFIT01 equilibria.
How the paper was built: `cer_transfer_learning/docs/WORKFLOW.md`.
The data are DIII-D and NSTX data: do not redistribute.
EOF

echo "== zip"
( cd "$(dirname "$OUT")" && zip -qr "$(basename "$OUT").zip" "$(basename "$OUT")" )
du -sh "$OUT.zip"
echo "wrote $OUT.zip"
