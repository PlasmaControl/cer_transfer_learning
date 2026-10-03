#!/bin/bash
# setup_env.sh -- set up the cer-transfer environment on any cluster.
#
#   scripts/setup_env.sh --data-root /path/to/data [options]
#
# Steps: install pixi if missing -> create the environment (pixi install)
# -> check the data root against the split lists -> run the smoke tests
# -> register a Jupyter kernel that starts through pixi (optional)
# -> print the two lines to add to your shell profile.
#
# Options:
#   --data-root DIR    directory holding the discharge-file folders (needed for
#                      training/evaluation; the demo notebook works without it)
#   --pixi-cache DIR   pixi package cache; use node-local or home disk, not a
#                      parallel filesystem (default: $HOME/.cache/pixi)
#   --kernel           register the Jupyter kernel "cer-transfer (pixi)"
#   --no-tests         skip the smoke tests
set -euo pipefail
cd "$(dirname "$0")/.."   # repository root
ROOT=$(pwd)

DATA_ROOT=""; PIXI_CACHE="${PIXI_CACHE_DIR:-$HOME/.cache/pixi}"; KERNEL=0; TESTS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --data-root)  DATA_ROOT=$2; shift 2 ;;
    --pixi-cache) PIXI_CACHE=$2; shift 2 ;;
    --kernel)     KERNEL=1; shift ;;
    --no-tests)   TESTS=0; shift ;;
    -h|--help)    sed -n 2,20p "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done
if [ -n "$DATA_ROOT" ]; then
  [ -d "$DATA_ROOT" ] || { echo "error: data root $DATA_ROOT does not exist"; exit 1; }
  DATA_ROOT=$(cd "$DATA_ROOT" && pwd)
  export CER_DATA_ROOT="$DATA_ROOT"
fi
export PIXI_CACHE_DIR="$PIXI_CACHE"

echo "== 1. pixi"
if ! command -v pixi >/dev/null 2>&1; then
  echo "pixi not found: installing to ~/.pixi/bin"
  curl -fsSL https://pixi.sh/install.sh | bash
  export PATH="$HOME/.pixi/bin:$PATH"
fi
echo "pixi $(pixi --version | awk '{print $2}') at $(command -v pixi); cache: $PIXI_CACHE_DIR"

echo "== 2. environment"
pixi install          # re-solves pixi.lock if pixi.toml changed; commit the lock afterwards
mkdir -p cer_ckpts slurm/logs gallery figs

if [ -n "$DATA_ROOT" ]; then
echo "== 3. data root: $CER_DATA_ROOT"
missing=0
for lst in splits/nstx_val.txt splits/nstx_passive_val.txt splits/d3d_val.txt; do
  [ -f "$lst" ] || continue
  first=$(grep -v '^#' "$lst" | head -1)
  if [ -f "$CER_DATA_ROOT/$first" ]; then echo "  ok      $lst -> $first"
  else echo "  MISSING $lst -> $CER_DATA_ROOT/$first"; missing=1; fi
done
[ $missing -eq 0 ] || echo "  some split lists do not resolve under this root (symlink the folders or fix --data-root)"
else
echo "== 3. no --data-root given: demo-only setup (training/evaluation need the data root)"
fi

if [ $TESTS -eq 1 ]; then
  echo "== 4. tests"
  pixi run test
  pixi run smoke || echo "  (torch/CUDA check failed: expected on a node without GPU; rerun on a GPU node)"
fi

if [ $KERNEL -eq 1 ]; then
  echo "== 5. Jupyter kernel"
  KDIR="$HOME/.local/share/jupyter/kernels/cer-transfer"
  mkdir -p "$KDIR"
  cat > "$KDIR/kernel.json" <<EOF
{
  "display_name": "cer-transfer (pixi)",
  "language": "python",
  "argv": ["$(command -v pixi)", "run", "--frozen", "--manifest-path", "$ROOT/pixi.toml",
           "python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
  "env": {"CER_DATA_ROOT": "${CER_DATA_ROOT:-}"}
}
EOF
  echo "  registered $KDIR/kernel.json (visible to any JupyterLab run as this user)"
fi

# repo-local environment file sourced by every job (git-ignored)
{
  [ -n "$DATA_ROOT" ] && echo "export CER_DATA_ROOT=$CER_DATA_ROOT"
  echo "export PIXI_CACHE_DIR=$PIXI_CACHE_DIR"
} > .env
echo "== wrote .env (sourced by the job files)"

echo "== done. Add to your shell profile (~/.bashrc), or 'source .env':"
[ -n "$DATA_ROOT" ] && echo "  export CER_DATA_ROOT=$CER_DATA_ROOT"
echo "  export PIXI_CACHE_DIR=$PIXI_CACHE_DIR"
echo "Then: sbatch slurm/<job>.sbatch from the repository root; notebooks/demo.ipynb with the kernel 'cer-transfer (pixi)'."
