# Workflow (state of 2026-10-13)

Rule 1: every job that runs is listed here, with its data root, inputs and outputs.
A job that is not in this file is not submitted.
Rule 2: a track never reads the other track's data root or checkpoints.
Rule 3: paper numbers and figures come from ONE command (`slurm/make_paper.sbatch`);
hand-run evaluations are for looking, never for the manuscript.

## The paper: track B (rebuilt DIII-D data, plain recipe)

| | |
|---|---|
| data root | `.env.fixed` -> `/scratch/gpfs/EKOLEMEN/ps9551/data_root_fixed`; DIII-D -> `/scratch/gpfs/EKOLEMEN/ps9551/passive_micer/fixed/{training_set_30,test_set_30}` (one grid 532 -> 526 nm / 536 bins, uint16, t01-t48; 11 shots excluded: `splits/d3d_fixed_excluded.txt`); NSTX folders identical to track A |
| figure-only NSTX files | `chers_set5/chers_<shot>.joblib` (+ `time_s`, `nbi_power_MW`) and `chers_set5/efit_<shot>.joblib` for 137711, 116939, 115600, 142239, 135660; list `splits/nstx_set5.txt`; symlinked into both data roots |
| how the DIII-D data were made | `cer_transfer.datasets.reprocess_d3d` from the raw H5 (`passive_micer/raw`), per-chord time interpolation at t0 + 5 i ms, lag-checked against the old files (`fixed/reprocess_*.csv`) |
| DIII-D convention | `d3d_tangential_fixed` (2.842 / -29 / 104.6, metadata only; `tests/convention_check.py --machine d3d_tangential_fixed`) |
| DIII-D model | `cer_ckpts/d3d_fixed_plain_best.pt` - plain recipe (no RandConv/shift/consistency/mirror), epoch 71, DIII-D validation 0.876, 13 143 training discharges. **Never overwritten.** |
| NSTX models | `cer_ckpts/nstx_fixed_ft_r{325,650,1300,2600,5200,full}.pt` - fine-tuned from the DIII-D model, all parameters, lr 1e-5 / 1e-4 |
| baselines | `cer_ckpts/nstx_agn_scratch_plain_r*.pt` (NSTX from scratch; NSTX files unchanged, valid for both tracks) |
| time axis | measured CHERS frame times (`time_s`); the old axis frame/200 Hz - 0.235 s was 30 ms early |

Numbers (all in `gallery/paper/SUMMARY.md`):
- zero-shot, NSTX test set, no correction: R2 Ti 0.372, v_tor 0.588 (validation 0.37 / 0.58)
- scaling, NSTX validation score: fine-tuned 0.756 0.757 0.772 0.775 0.777 0.775;
  from scratch 0.660 0.672 0.699 0.713 0.720 0.727 (325 ... 7307 discharges)
- all-data fine-tuned model, all labels: R2 0.776 / 0.774, RMSE 196 eV / 30.2 km/s,
  median 41.6 eV / 5.01 km/s; consistent labels (83.3 % / 84.4 %): R2 0.889 / 0.956,
  RMSE 124 eV / 12.0 km/s; best possible R2 0.896 / 0.992 (all), 0.963 / 0.997 (consistent);
  flagged labels carry 66 % / 87 % of the squared error
- core / edge median errors: 73 eV, 6.8 km/s (core) and 33 eV, 4.4 km/s (edge)
- calibration factors of the NSTX model: `gallery/137711/calibration.txt`

### Build (one job; repeat after any code or data change)
```
ENV=.env.fixed D3D=cer_ckpts/d3d_fixed_plain_best.pt NSTX=cer_ckpts/nstx_fixed_ft_rfull.pt sbatch slurm/make_paper.sbatch
```
- `FORCE=1` recomputes dumps/evaluations; `TEST=1` runs the test set, once per DIII-D model
  (guard: `gallery/paper/zero_shot_test_<model>.txt`) - done for `d3d_fixed_plain_best`.
- Login-node redraw of the figures that read dumps: `FTP=nstx_fixed_ft scripts/make_figs.sh`.
  The beam-off figure and the zero-shot comparison run models (GPU, i.e. the job).
- The beam-off figure's DIII-D model is printed on its first line:
  `BEAMOFF_CK`, else `D3D`, else `d3d_fixed_plain_best.pt`.

### Figures of the manuscript (all written by the build into `figs/`)
| manuscript | file | content |
|---|---|---|
| Fig. 1 | `hero_layout_mockup.pdf` | drawn by hand |
| Fig. 2 | `recon_legacy_137711.pdf` | DIII-D+NSTX (7307 shots) model, v_tor and Ti (keV), calibrated bands |
| Fig. 3, 4 | `event_ntv_116939.pdf`, `event_tm_115600.pdf` | equilibrium at t0, v_tor and Ti rows at three times, traces of one core chord |
| Fig. 5 | `beamoff_examples.pdf` | DIII-D model zero-shot, 135660 / 142239, NBI power row |
| Fig. 6 | `learning_curves_val_score_baseline.pdf` | DIII-D (plain) and NSTX from scratch; resume dips interpolated |
| Fig. 7 | `scaling_val_score.pdf` | fine-tuned vs from scratch, zero-shot line |
| Fig. 8 | `event_transfer_r325_116939.pdf` | 325-shot fine-tuned vs from scratch, both targets |
| Fig. 9 | `model_architecture_new.png` | drawn by hand |
| supplement (optional) | `*_scratch.pdf`, `zero_shot_vtor_115600.pdf` | NSTX-only alternatives; zero-shot comparison |

All paper PDFs are 6.5 in wide (= `\textwidth`, `cer_transfer/figures/paperfig.py`).
Legend names: "DIII-D model", "NSTX (N shots) model", "DIII-D+NSTX (N shots) model".

### Open items
1. Beam-off figure: decide the DIII-D model (plain, as now, or the DG e44 model of track A,
   named as such in the caption).
2. Manuscript placeholders: Fig. 2 profile time (`\todoval{0.265}`, from the figure title);
   116939 median errors (`\todoval` x3, from `gallery/e116939*.npz`); NSTX discharges per day.
3. DIII-D dataset table (11 411 / 1 740 / 1 740) vs the lists actually used
   (`training_set_30`: 13 143, `test_set_30`: 1 737).
4. Code cleanup (below), then the release package for external users.

## The comparison: track A (old DIII-D data, DG recipe; frozen)

`.env` -> `data_root`; DIII-D model `cer_ckpts/d3d_full_dg_e44.pt` (stretch 1.4 chosen on NSTX
validation); ladder `nstx_agn_ft_r*`; results frozen in `gallery/paper_trackA/`, `figs_trackA/`.
Zero-shot test: Ti 0.59 / v_tor 0.32 with the metadata correction (v_tor 0.46 raw).
Used in the Discussion only. Nothing of track A is rerun.

## Retired experiments (kept in git history, tag `pre-cleanup`)
- DG recipe on the rebuilt data (`d3d_fixed_dg.pt`): converged at Ti 0.31 / v_tor 0.20 raw.
- Background subtraction (`subtract_passive.py`): hurts every model.
- Label-selected stretch 1.4 and the v_tor correction; self-consistency ceiling analysis.

## Code that the paper runs
- entry points: `train.py`, `finetune.py`, `eval_checkpoint.py`, `tune.py`
- core: `configs`, `data`, `dark`, `augment`, `losses`, `models`, `labram_blocks`, `randconv`,
  `training`, `inference`, `logging_utils`
- analysis: `affine_fit`, `calibrate`, `ceiling_check`, `coverage`, `global_metrics`,
  `physics_correction`, `instrument_width`
- datasets: `reprocess_d3d`, `wavelength_axes`, `make_splits`, `make_d3d_subsets`
- figures: `beamoff_examples`, `common`, `event_panels`, `event_3d`, `paperfig`, `plots`,
  `recon_composite`, `timeaxis`
- tests: `convention_check`
- scripts: `make_paper.sh`, `make_figs.sh`, `setup_env.sh`
- job files (6): `pretrain_fixed_d3d`, `agnostic_scaling`, `make_paper`, `reprocess_d3d`,
  `agnostic_ladder` (track A's DIII-D model), `tune_d3d`

Style: black, 88 columns, numpydoc docstrings (`pyproject.toml`). Check with
`ruff check --line-length 88 --select E,W,F,D <files>`.

## Naming
- `d3d_full_dg*` / `nstx_agn_*` = track A; `d3d_fixed_*` / `nstx_fixed_*` = track B;
  `nstx_agn_scratch_plain_*` = baselines (both tracks).
- Logs: `slurm/logs/<job-name>[_<tag>].out`; one job name per purpose (`--job-name=` when
  reusing a job file).

## Checks before any submission
```
squeue -u ps9551 -o "%.10i %.24j %.9T %.11M %R"     # what runs, nothing stale
grep CER_DATA_ROOT .env .env.fixed                   # the two roots, unchanged
ls cer_ckpts/*.pt | sed 's|cer_ckpts/||'             # every checkpoint appears in this file
```
