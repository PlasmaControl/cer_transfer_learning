# cer-transfer

Neural reconstruction of ion temperature (T_i) and toroidal rotation (v_tor)
profiles from charge-exchange recombination (CER) spectrograms, with
transfer learning between tokamaks.

A convolutional network reads the spectrograms of all lines of sight of a
CER system at once and returns per-chord T_i and v_tor with uncertainties,
frame by frame. It is pre-trained on DIII-D, fine-tuned on NSTX (a
different machine class), and adapted to the NSTX background spectrometer
array, whose lines of sight do not intersect the diagnostic beam. The
background model also reconstructs profiles while the beam is off, where
the conventional analysis provides none.

## Getting started

Try the model on a discharge in five minutes (CPU is enough):
```bash
git clone https://github.com/PlasmaControl/cer_transfer_learning.git && cd cer_transfer_learning
scripts/setup_env.sh --kernel          # installs pixi if needed, builds the environment, registers a Jupyter kernel
pixi run jupyter lab notebooks/demo.ipynb   # or open it in any JupyterLab with the kernel "cer-transfer (pixi)"
```
In the notebook's first cell set `RELEASE` to the release URL; it downloads an
inference checkpoint and one NSTX discharge and plots the reconstruction against
the conventional fits. Training, evaluation and the figure pipeline additionally
need the data archive (`CER_DATA_ROOT`) and the split lists: see Installation and
Running on another cluster.

## Layout

```
train.py             pre-train a model on one machine
finetune.py          transfer a model to another machine (probe, then full fine-tune)
tune.py              hyper-parameter search (Ray Tune)
eval_checkpoint.py   metrics and per-point prediction dumps

cer_transfer/        package
  configs.py         machine definitions (chords, wavelength bins, data dirs) and model/training configs
  data.py            dataset over per-discharge joblib files
  models.py          backbone (machine-specific stem + shared trunk) and heads
  training.py        trainer, checkpoints, normalization
  losses.py, augment.py, dark.py, logging_utils.py
  beamstate.py       beam on/off detection from the spectra of both arrays
  inference.py       load a checkpoint and run it on one discharge; inference-only export
  figures/           manuscript figures (run as python -m cer_transfer.figures.<name>)
  analysis/          calibration, label-noise ceilings, coverage, audits, receptive field
  beamoff/           beam-off validation: scanning, edge-frame test, pooling
  datasets/          split lists and nested training subsets

notebooks/demo.ipynb minimal demo: model on one discharge vs conventional fits
scripts/             shell drivers (figures, beam-off validation)
slurm/               job files for every training/evaluation run of the study
splits/              discharge lists (not in git: generated per data set, see Data format)
tests/smoke.py       end-to-end check of the non-torch modules on synthetic data
docs/                notes
```

Everything runs from the repository root; the package needs no installation.

## Installation

One command sets everything up on any cluster:
```bash
scripts/setup_env.sh --data-root /path/to/data --pixi-cache /local/disk/pixi-cache --kernel
```
It installs [pixi](https://pixi.sh) if missing, creates the environment (`pixi
install`, PyTorch with CUDA 12.8 wheels, no module loads), checks that the split
lists resolve under the data root, runs the smoke tests, and with `--kernel`
registers the Jupyter kernel "cer-transfer (pixi)" for JupyterLab portals such as
Open OnDemand (the kernel starts through `pixi run`, so it follows the environment
wherever it lives, and carries `CER_DATA_ROOT`). Finally it prints the two `export`
lines for your shell profile.

Afterwards prefix commands with `pixi run`. Tasks: `pixi run train`, `pixi run
finetune`, `pixi run eval`, `pixi run figs`, `pixi run test`, `pixi run smoke`
(torch/CUDA check on a GPU node).

## Data format

One joblib file per discharge with

| key | shape | content |
|---|---|---|
| `input` | (chords, frames, wavelength bins) | spectrograms, one per line of sight |
| `target` | (chords, frames, 2) | conventional fits of T_i and v_tor; NaN where none exists |
| `target_error` | (chords, frames, 2) | their quoted 1σ uncertainties |
| `end_index` | int | number of valid frames; ≤ 0 means the whole recording |

File paths in lists and on the command line are resolved against the environment
variable `CER_DATA_ROOT` unless absolute (`cer_transfer.configs.data_path`). The
lists themselves (`splits/*.txt`, one path per line, `#` comments allowed) are not
part of the repository: they are generated from the local data set with
`python -m cer_transfer.datasets.make_splits` (train/val/test by discharge) and
`make_d3d_subsets` (nested label-efficiency subsets), and the job files refer to
them by name (`splits/nstx_train_r325.txt` etc.). The lists used for the paper's
runs are archived with the data, since the same lists are needed to reproduce the
reported numbers.
Machines are defined in `cer_transfer/configs.py`: `d3d` (80 channels),
`nstx` (51 chords, one array), `nstx_active` / `nstx_passive` (48
flux-surface-aligned chords of the foreground / background array; the labels
are always the active-array fits), and their NSTX-U counterparts. Lists of
files (`splits/*.txt`, one path per line) select the data for every run.

## Demo

`notebooks/demo.ipynb` downloads an inference checkpoint and one discharge from
the release, runs the model on CPU and plots traces and profiles against the
conventional fits. In code:
```python
from cer_transfer.inference import load_model, predict_file
m = load_model("cer_ckpts/nstx_ft.pt")
r = predict_file(m, "chers_nstx_labeled/chers_137711.joblib")   # relative to CER_DATA_ROOT
r.pred[chord, frame, k]   # k = 0: T_i (eV), 1: v_tor (km/s); r.pred_sigma, r.y, r.sigma
```
In JupyterLab choose the kernel "cer-transfer (pixi)" registered by
`scripts/setup_env.sh --kernel`; it sets `CER_DATA_ROOT` for the kernel, which a
portal-launched JupyterLab does not take from your shell profile.

Checkpoints for sharing are exported without optimizer state (about 16 MB):
`python -m cer_transfer.inference cer_ckpts/nstx_ft.pt nstx_ft_inference.pt`.

## Workflows

**Pre-training** (`slurm/pretrain_transfer_source.sbatch`):
```
python -u train.py --machine d3d --checkpoint cer_ckpts/d3d_transfer_source.pt \
    --head-type mlp --norm group --feature-width 128 --hidden-dim 8 --kernel-size 3
```
`--feature-width` enables the stem/trunk architecture: the first block (stem)
is machine-specific, the rest (trunk) transfers across machines with different
channel counts.

**Transfer** (`slurm/finetune_*.sbatch`): a short probe phase trains the fresh
stem and the head while the trunk is frozen, then everything is fine-tuned with
discriminative learning rates:
```
python -u finetune.py --source-checkpoint cer_ckpts/d3d_transfer_source.pt \
    --machine nstx --checkpoint cer_ckpts/nstx_ft.pt --train-list splits/nstx_train_r325.txt
```
Label-efficiency studies use nested subsets of the training list
(`splits/nstx_train_r*.txt`, `splits/d3d_train_r*.txt`;
`python -m cer_transfer.datasets.make_d3d_subsets`), with from-scratch
counterparts in `slurm/scratch_*.sbatch`.

**Evaluation and prediction dumps**:
```
python -u eval_checkpoint.py --checkpoint cer_ckpts/nstx_ft.pt --list splits/nstx_test.txt \
    --dump-preds gallery/nstx_test.npz        # add --predict-only for shots without fits
```
The dump holds per-point `chord, y, pred, sigma, pred_sigma` and is the input
of the figure and analysis modules.

**Uncertainty and labels**: `cer_transfer.analysis.calibrate` (scale factor
of the predicted σ), `ceiling_check` (label-noise ceilings of the R² score),
`coverage` (1σ/2σ coverage), `label_audit` and `dark_audit` (label-only
quality flags), `receptive_field` (temporal receptive field of a checkpoint).

**Beam-off validation** (`scripts/run_passive_validation.sh` shows the chain):
`cer_transfer.beamoff.scan_verified` finds phases in which the diagnostic beam
is off from the brightness of both arrays; `edge_frame_test` compares the
background-array reconstruction in those phases with the nearest conventional
fit across the beam switch, and the same comparison with the beam on;
`pool_edge` pools the per-discharge results.

**Figures**: `scripts/make_figs.sh` renders the manuscript figures from the
dumps listed in its header. The per-chord plot coordinate is the tangency
radius of each line of sight, given as a `chord,x` CSV per discharge
(`gallery/chords_Rtan_<shot>.csv`, not part of the repository); without a coordinate file
the chord index is used.

## Running on another cluster

Job files are location-independent: all paths inside the repository (`cer_ckpts/`,
`splits/`) are relative, and the jobs find the repository root whether submitted from
it or from `slurm/`, and write their logs to `slurm/logs/<job-name>.out` / `.err`
themselves, so the submission directory does not matter.
Fine-tuning and pre-training jobs continue from their checkpoint when requeued
(`--resume` is added automatically if the output checkpoint exists). A run that
finished (early stopping or epoch limit) is marked in its checkpoint and is not
resumed; `--force-resume` overrides that for a deliberate continuation. What has to be adapted:

1. Set `CER_DATA_ROOT` to the directory that holds the discharge-file folders
   (`chers_nstx_labeled/`, `chers_nstx_active/`, `chers_nstx_passive/`,
   `training_set_30/`, `test_set_30/`, ...). All split lists and the default
   DIII-D directories are relative to it; `sbatch` passes the variable on to
   the jobs. Copy the archived split lists into `splits/`; only regenerate
   them for a new data set, since new lists reshuffle train/val/test.
2. `#SBATCH` headers: add `--partition`/`--account` as required, change
   `--mail-user`. Keep `--signal=B:USR1@300 --requeue`; training resumes from
   the checkpoint after preemption.
4. GPU: the runs assume an A100 (`--batch-size 256`, `--subseq-len 256`); on a
   smaller card halve the batch size and set `--accumulation-steps 2`.
4. `scripts/setup_env.sh --data-root … --pixi-cache …` does the environment,
   the data-root check and the tests; on a GPU node `pixi run smoke` must report
   `torch.cuda.is_available() True`.

## Conventions

- NSTX CHERS records at 200 Hz and the recording starts 235 ms before the
  experimental clock zero, so experimental time is `frame / 200 - 0.235`.
  Both are options of the figure modules (`--fs`, `--t-offset`).
- `--k k_Ti k_vtor` scales the predicted σ by the calibration factors of the
  model being plotted (`cer_transfer.analysis.calibrate`). The factors are
  model-specific and must not be reused across checkpoints.
- Phases without the diagnostic beam are identified from the spectra only:
  the foreground-array line brightness drops to the background level, and
  only switches at which the background brightness stays unchanged count.

## Code style

Formatting follows scikit-learn's conventions: `black` and `isort` at 88 columns
(`pyproject.toml` holds the settings) and numpydoc docstrings. `pyflakes` is clean.

## Tests

`python -m tests.smoke` builds a synthetic discharge and runs every figure,
beam-off and analysis module that does not need PyTorch.
