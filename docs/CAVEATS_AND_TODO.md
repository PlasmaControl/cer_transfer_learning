# CER transfer: caveats, status, TODO

Status snapshot (2026-09-17): 51-chord NATIVE matrix COMPLETE.

RESULTS (full val lists, matched configs, official; per-chord = median):
- D3D pretrain (transfer source): FINAL 0.8691 (ep 144; last LR level paid;
  manual scancel after >50 flat epochs = executing the pre-registered
  patience rule). Equals legacy 0.8696 => "the transferable architecture
  costs nothing". PROVENANCE: legacy 51-ch hops trained from the ep-105
  0.8640 snapshot (same path, since overwritten); 48-lineage trains from
  0.8691 automatically. Tangential-48 eval: pooled ti 0.929 / vtor 0.965;
  medians 0.856 / 0.870 — at/above nominal ceilings (D3D sigmas
  conservative).
- NSTX (native): scratch 0.7636 (0.499 / 0.413); hop 1 0.7708
  (0.511 / 0.439). Transfer wins EVERY metric; the old hop-1 vtor-median
  anomaly (0.24 vs 0.43) was an interpolated-era / unmatched-subset
  artifact — DEAD.
- NSTX-U (ti-med / vtor-med):
    scratch        0.6438 (0.160 / 0.218)
    direct D3D     0.6773 (0.283 / 0.280)
    hop 2          0.6825 (0.289 / 0.288)
    hop 2 no-D3D   0.6851 (0.284 / 0.277)
  ATTRIBUTION FINAL: hop2 ~= noD3D ~= direct at every granularity — the
  middle hop adds nothing measurable; ANY data-rich CER source delivers
  the benefit. Paper claim: "use the nearest data-rich machine;
  cross-class transfer still delivers most of the benefit." Per-chord
  dynamics +75% via transfer; chi2 roughly halves (ti 11-15 vs 26).
- Scaling (warm hop-2 recipe): n25 0.6664 | n50 0.6709 | n100 0.6734 |
  n200 0.6815 | full 0.6825. HEADLINE: 25 labeled shots via transfer beat
  325 from scratch (0.6438); saturates by n200.
- Context-length observation: hop 2 @ subseq 128 scored 0.6934 despite
  excluding 38% of shots — official numbers @ 64; see caveat 4.

THE PLATEAU, RESOLVED (was: both NSTX models stuck at 0.765, far below
nominal ceilings):
- Eliminated by experiment: preprocessing (native re-extraction: no
  change), 200 Hz label fabrication, imbalance (tail sampling), objective
  (per-chord norm 0.5997 with medians DOWN; noise-aware NLL), hyperparams
  (sweep null), transfer, explicit moment features ("always sits below or
  at non-momentum"), per-shot offsets (correction test: vtor +0.015 only,
  Ti moved MORE => not wavelength zero-points), label timing (shift scan
  peaks at 0).
- Root cause measured (label self-consistency, consecutive labeled frames
  / combined sigma): NSTX vtor sigmas OVERCONFIDENT at high line amplitude
  by ~4-5x (deciles 1.16 -> 5.5); Ti milder (0.69 -> 2.0). Model |z| vs
  amplitude rises 1.5 -> 9.8 (vtor) — constant absolute error against
  shrinking quoted sigma.
- HONEST CEILINGS COMPUTED (ceiling_check --honest: sigma inflated by
  per-amplitude-decile self-consistency factor, normalized to lowest
  decile; mild lower bounds where bright chords evolve faster; envelope
  arithmetic earlier in this file's history was wrong — k is large exactly
  where sigma is small, so pooled ceilings deflate less than expected):
    nstx   ti 0.938 -> 0.898   vtor 0.995 -> 0.990
    nstxu  ti 0.865 -> 0.791   vtor 0.969 -> 0.966
    (d3d tangential pending — rerun after --first-chords amp-crop fix)
  Gap table (val, pooled, vs honest): NSTX ti 85%, NSTX vtor 77%,
  NSTX-U hop2 ti 93% (essentially done), NSTX-U hop2 vtor 65%.
- Model |z| vs amplitude DIVIDED by k => uniform ~1.2-2.2 across all
  amplitudes: the residual gap is DIFFUSE, a flat factor ~1.5-2 above
  measured label reproducibility. Gate fired for the last architectural
  lever: FULL-RESOLUTION LEARNED BRANCH (no-pooling 1D conv over native W
  per chord/frame, ~16-dim into readouts; strictly generalizes the null
  moments) — TO BUILD. Its result doubles as the systematics test: gain =>
  extraction was the limit; null => residual is slow-drift label
  systematics (consistency only catches frame-to-frame noise, honest
  ceiling itself an upper bound) and the claim finalizes at "within
  ~1.5-2x measured label reproducibility".
- Paper framing: "reproduces the standard analysis to within its measured
  reproducibility"; label-honesty-vs-amplitude is a diagnostician-facing
  result (method: cer_transfer.analysis.ceiling_check --label-consistency).

CEILING REFERENCE (nominal, target_error taken at face value; per-chord =
median; top-bin = highest |vtor| quintile):
  machine      pooled ti/vtor   per-chord ti/vtor   top-bin vtor
  d3d tang48   0.850 / 0.967    0.842 / 0.966       0.934
  nstx         0.898 / 0.992    0.758 / 0.987       0.987
  nstxu        0.865 / 0.969    0.715 / 0.952       0.970
Nominal ceilings are LOWER BOUNDS where sigmas are conservative (D3D) and
OVERSTATED where sigmas are photon-statistics-only (NSTX high amplitude).

## Dual-array lineage (second extraction generation; 2026-09-16)
- NSTX(-U) CHERS has TWO arrays viewing the same flux surfaces: active
  (beam-intersecting) + passive. New extraction: 48 chords/array, labels
  always the active fits, frames synchronized => (passive spectrogram ->
  active label) pairs beat the beam-modulation adjacency protocol.
- Four configs registered: nstx_active / nstx_passive / nstxu_active /
  nstxu_passive (48 ch, list-enforced). 48 != 51 => NEW stem/readout
  lineage; legacy 51-ch checkpoints/numbers frozen as the transfer paper.
- Training on ACTIVE only; passive is EVALUATION-ONLY (information-content
  study). Passive photon counts much lower; working assumption: log
  preprocess + GroupNorm absorbs gain (additive-offset argument).
  Falsification test = zero-shot per-chord eval of active model on passive
  list: edge-fine/core-degraded => physics (the paper figure);
  uniform degradation => low-count normalization, then build
  --input-norm robust (deterministic per-spectrum median/IQR) — NOT gain
  augmentation (user rejects as unclean; log+GN argument accepted for now).
- Scripts ready: scratch_nstx_active, finetune_nstx_active (hop 1 from the
  EXISTING d3d_transfer_source.pt — no D3D retrain), hop2_nstxu_active
  (--transfer-readouts, 48==48 flux-aligned), eval_passive (both machines,
  --diagnose, per-chord CSVs). Expect splits/nstx_active_*.txt etc. from
  make_splits. Official passive numbers: active-TEST-split shots only.
- Open config question: passive array's window / native W assumed equal to
  active — confirm before validator run.

Dual-array updates (2026-09-17):
- Passive dataset REBUILT after NaN spectra found in some shots (fix at
  extraction; a make_splits NaN filter was declined). Rebuilt set contains
  sub-count values ~1e-5 — compare_arrays now WARNS on counts < 1; check
  extraction units/offsets.
- INSTRUMENT COMPARISON, dataset-wide (cer_transfer.analysis.compare_arrays, train+val):
    machine  cont A/P   line A/P   A/P ratio (raw)  distinct fibers
    nstx     214 / 54   313 / 895  0.62             31 of 48 slots
    nstxu    201 / 44   181 / 762  0.39             31 of 48 slots
  Passive lines 5-8x brighter than active PER CCD BIN on both machines
  (after the ~x3 instrumental factor: array A sums ~3 CCD bins, array B
  selects one; raw pedestal, no dark subtraction). The passive challenge
  is "bright photons from the wrong place" (edge emission dominates the
  window), NOT photon starvation. NSTX ACTIVE saturation 3.9e-3 (>> other
  arrays) — per-chord glance pending. Caption caveat everywhere: 48
  passive slots = ~31 distinct fibers.
- Early NSTX-U 48-lineage results (325 shots — direction only):
  active-48 scratch ~= legacy-51 at pooled AND medians (0.201/0.190 vs
  0.176/0.172) => lineage continuity benign. Passive scratch 0.5894
  pooled, medians 0.144/0.098. Active-passive delta RADIALLY ORDERED:
  vtor deficit +0.145 (chords 0-11) -> +0.053 (36-47); edge chords ~=
  active; vtor-collapsed chords #1/#3/#6. This is the
  edge-measured/core-degraded physics signature (GroupNorm assumption
  survived its first test). OPEN QUESTION to user: confirm low index =
  core-viewing for the PASSIVE array ordering.
- In flight: scratch_nstx_active + scratch_nstx_passive + 
  finetune_nstx_active (submitted); hop2_nstxu_active manual after hop 1;
  then eval_passive. finetune_nstx_passive.sbatch (nstx_active_ft ->
  nstx_passive warm) PROMISED, not yet written. User has NOT yet rerun
  compare_arrays with the 2-panel figure.

## Passive framing (PI wants passive in THIS paper)
- Transfer = spine; passive = payoff section. Claim: "profile inference
  from passive spectra, validated against simultaneous active analysis"
  — NEVER "passive CER measurement".
- Per-chord/flux-surface performance curve is reported AS the measured
  radial information content. Core values = "edge-constrained,
  prior-informed estimates"; calibrated sigma(rho) widening toward core
  carries the core claim (=> cer_transfer.analysis.calibrate REQUIRED).
- Deployment caveat via continuity check across beam transitions (eval
  figure once a passive checkpoint exists).
- Shared-embedding caveat: chord c's readout sees a GLOBAL embedding, so
  a nonzero passive per-chord median is not yet proof that sightline's
  light carries the info; chord-occlusion test (~30 lines) is the referee
  answer if demanded — parked per user.
- Core-inference mechanism taxonomy (if core skill appears):
  line-integration through core / profile stiffness / scenario ID /
  cross-chord reconstruction — separable via per-chord curve shape,
  medians-vs-pooled, occlusion.

## Plan state (two-hop + controls) — 51-lineage COMPLETE
- [x] Transfer source final (0.8691)
- [x] NSTX native pair: scratch + hop 1 (transfer wins every metric)
- [x] NSTX-U: scratch / direct / hop 2 / no-D3D (attribution final)
- [x] Scaling array n25..full @ 64
- [x] Full-list evals of all of the above (per-chord tables in eval_*.csv)
- [ ] Test-set finalization: one eval per official checkpoint on the
      UNTOUCHED test lists; then numbers are citable
- [ ] cer_transfer.analysis.calibrate (temperature scaling) — REQUIRED for the passive
      section's sigma claims, next code item
- [ ] D3D --honest ceiling rerun (after --first-chords amp-crop fix)
- [ ] Opportunistic: tail-run partial-checkpoint binned eval; NSTX active
      saturation (3.9e-3) per-chord CSV glance

## Caveats (live ones; resolved ones struck to history below)
1. NSTX-U labeled set has low-rotation selection bias (p90 ~41 km/s vs
   NSTX per-frame range +-368): high-Mach evaluation on NSTX-U is
   label-limited; rotation-binned reporting mandatory.
2. High-Mach vtor is the one open performance frontier: NSTX top-bin 0.33
   (ceiling 0.99 nominal), NSTX-U top-bin 0.14. Levers: tail-oversample +
   noise-aware (designed pairing, untested at NSTX scale — tail run
   aborted for queue priority, partial ckpt evaluable), Doppler-shift
   augmentation (BLOCKED on per-chord vtor<->px geometry factors; native
   wavelength key now stored), C III blend question to diagnosticians
   (does the fitter deblend at high shift?).
3. Hop-2 stem was re-initialized despite 51==51 chords (transfer logic
   keys on machine identity, not shape) — possible free gain; code look
   pending.
4. Context length: subseq 128 beat 64 on NSTX-U pooled (0.6934 vs 0.6807)
   despite excluding 38% of shots — worth a controlled look (e.g. train 64,
   eval full-shot; or variable-length batching) before final runs.
5. Low-|vtor| bins have little/no label information (NSTX-U lowest-bin
   ceiling -1.0): negative binned R2 there is the metric, not the model.
   Judge bins by R2-vs-ceiling and RMSE-vs-floor.
6. Edge chords (#44-50 NSTX-family; #10-15 D3D tangential) are weak with
   LOW CEILINGS => label quality, not model. Report per-chord medians
   excluding ceiling<0.2 chords, rule stated.
7. Short sequences: NSTX-family trains at subseq 64 (D3D trunk saw 256);
   median NSTX-U shot yields ~2 chunks => tiny effective batches (caveat
   interacts with #4).
8. eval_checkpoint --limit N uses the FIRST N list files (earliest
   campaigns): subset-matched comparisons only; full-list evals for
   official numbers.
9. val_loss is objective-dependent (noise-aware runs incomparable to
   plain); val_score (pooled R2) is the cross-run language, per-chord
   medians the honest one.

## Environment
- pixi.toml synced into repo (torch 2.4-2.6 pypi + bundled CUDA, seaborn,
  ray[tune]); pixi.lock is the ABI guard — always `pixi run --frozen`.

## Script hygiene (adopted after three in-place repurposings)
- Baseline scripts are IMMUTABLE and reproduce their published numbers;
  experiments get COPIES with own job names + checkpoint paths.
- grep -E "noise-aware|per-chord-norm|tail-oversample|moment-features"
  *.sbatch must return empty across the official matrix.
- Official matrix scripts: scratch_nstx, scratch_nstxu, finetune_nstx
  (hop 1), hop2_nstxu, finetune_nstxu (direct; matched scarce recipe),
  scaling_nstxu (@64), pretrain_transfer_source. Suggested renames:
  hop1_nstx / direct_d3d_nstxu.
- Version-skew rule: every zip message names changed files; cluster syncs
  via unzip -o -j before running anything new.

## Tool inventory (all runtime- or regression-tested unless noted)
- cer_transfer.analysis.check_dataset (validator), cer_transfer.datasets.make_splits (stratified splits)
- cer_transfer.analysis.ceiling_check: pooled / --per-chord / --vtor-bins /
  --label-consistency (sigma honesty) / --first-chords
- eval_checkpoint.py: pooled + per-chord + binned + --diagnose (time-shift,
  |z| vs amplitude, frame bimodality, offset-correction) / --first-chords
- cer_transfer.figures.plots: hist / recon / curves (dual talk-PNG + paper-PDF)
- Training flags (documented negative results kept as options):
  --per-chord-norm, --noise-aware-nll, --tail-oversample, --moment-features,
  --max-label-relerr (label QC; untested in anger)
- ceiling_check --honest: honest ceilings (verified vs constructed case)
- cer_transfer.analysis.compare_arrays + slurm/compare_arrays.sbatch: dataset-wide
  active/passive comparison (pairs shots by stem; per-chord
  continuum/line/saturation/ratio; fiber dedup; sub-count warning;
  2-panel figure: dataset-median spectrum + per-chord line amplitude;
  OOM view-pinning fixed with .copy())
- cer_transfer.analysis.inspect_shot: single-shot multi-dataset spectra comparison (stats
  table + 3-panel figure)
- cer_transfer.figures.plots: dual-context rendering — talk PNG slide-sized, paper PDF
  column-sized (curves 5.5x3.4 in, verified 390 pt)
- cer_transfer.analysis.calibrate: NOT YET WRITTEN (next code item — required for passive)
- full-resolution learned branch: NOT YET BUILT (optional polish /
  systematics test; gate fired but superseded in priority by passive)

## Negative results ledger (paper's eliminated-alternatives table)
- NSTX-U scratch improvements all null: hp sweep, per-chord norm +
  noise-aware (0.5997, medians dropped), tail oversample (0.637, top-bin
  0.123), moment features (below or at baseline)
- NSTX: moment features null; native re-extraction no change; per-shot
  offset correction +0.015 only
- Transfer at data-rich target (hop 1): speed, not ceiling

## Physics / paper threads (parked, reopen conditions noted)
- Passive CER: scoped to edge-measured / core-estimated with adjacency
  validation + continuity + occlusion evidence; beam_on per-frame array
  wanted in extraction (gets more expensive later); tearing-mode
  correlation from old abstract = strongest passive evidence, reproduce
  on current models when passive paper starts.
- Two-paper split: transfer paper (Nuclear Fusion floor / Nat. Comms
  stretch) + passive follow-on. Old abstract's 0.95/0.93 were
  inflated-metric era — never cite.
- C III edge line in NSTX window: fiducial asset + blending hazard at
  high shift; wavelength unconfirmed.
- 4,111,116 params total; ~16% machine-specific (stem+readouts).

## Dataset sizes (cer_transfer.analysis.count_samples, 2026-09-18; labeled pts = finite
## target+sigma (chord,frame) points; subseqs @ 64)
  list                    shots      frames    labeled pts   subseqs
  d3d train / val        13,151 / 1,740   (subseq 256 for pretrain; row pending)
  nstx train             7,307   1,205,124  12,404,631      15,328
  nstx val / test        1,565 each; ~2.67-2.71M pts each
  nstx r-subsets         325/650/1300/2600/5200 nested, seed 42, random
                         (593k / 1.12M / 2.22M / 4.45M / 8.78M pts)
  nstx n-subsets         25..400 stratified (legacy scaling, unused in paper)
  nstx_active train      7,306   1,208,858  11,752,541      15,409
  nstx_active val/test   1,565 each, ~2.52M pts each
  nstx_passive           IDENTICAL counts to nstx_active (same labels,
                         same shots; passive spectra as input)
  nstxu (archived)       325 / 70 / 70; train 617,092 pts
  Label density ~20% of (chord, frame) cells on both machines.

Reference numbers: D3D windows 526-532 nm (~400-600 native bins,
tangential = first 48 of 80); NSTX(-U) 528.25-531.25 nm, W 79-87 native
(pre-crop), 51 tangential chords; 100 Hz native. Splits: NSTX
7307/1565/1565 (+n25..n400), NSTX-U 325/70/70 (+n25..n200); test lists
UNTOUCHED until final evaluation.
