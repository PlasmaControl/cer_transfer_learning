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
  figures/           manuscript figures (run as python -m cer_transfer.figures.<name>)
  analysis/          calibration, label-noise ceilings, coverage, audits, receptive field
  beamoff/           beam-off validation: scanning, edge-frame test, pooling
  datasets/          split lists and nested training subsets

scripts/             shell drivers (figures, beam-off validation)
slurm/               job files for every training/evaluation run of the study
splits/              discharge lists defining the train/val/test splits and subsets
data/chord_radii/    tangency radii of the NSTX lines of sight (plot coordinate)
tests/smoke.py       end-to-end check of the non-torch modules on synthetic data
docs/                notes
```

Everything runs from the repository root; the package needs no installation.

## Installation

The environment is managed with [pixi](https://pixi.sh): `pixi install`, then
prefix commands with `pixi run`. Tasks: `pixi run train`, `pixi run finetune`,
`pixi run eval`, `pixi run figs`, `pixi run test`.

## Data format

One joblib file per discharge with

| key | shape | content |
|---|---|---|
| `input` | (chords, frames, wavelength bins) | spectrograms, one per line of sight |
| `target` | (chords, frames, 2) | conventional fits of T_i and v_tor; NaN where none exists |
| `target_error` | (chords, frames, 2) | their quoted 1σ uncertainties |
| `end_index` | int | number of valid frames; ≤ 0 means the whole recording |

Machines are defined in `cer_transfer/configs.py`: `d3d` (80 channels),
`nstx` (51 chords, one array), `nstx_active` / `nstx_passive` (48
flux-surface-aligned chords of the foreground / background array; the labels
are always the active-array fits), and their NSTX-U counterparts. Lists of
files (`splits/*.txt`, one path per line) select the data for every run.

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
radius of each line of sight (`data/chord_radii/`); without a coordinate file
the chord index is used.

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

## Tests

`python -m tests.smoke` builds a synthetic discharge and runs every figure,
beam-off and analysis module that does not need PyTorch.
