# Instrument-function widths for NSTX CHERS and DIII-D CER

Compiled 2026-10-19. Read-only lookup: nothing in the CHERS analysis tree or the
calibration archive was modified.

Deliverables:

| file | rows | status |
|---|---|---|
| `instrument_function_nstx_chers.csv` | 459 (9 epochs x 51 chords) | **measured values, read from calibration** |
| `instrument_function_d3d_cer.csv` | 3 | **method + provenance only; no numeric widths** |

---

## 1. NSTX CHERS, active (toroidal) array A, 51 chords

### Where the numbers come from

The CHERS fit does **not** subtract an instrument width in quadrature. It
**convolves** a Gaussian line shape with a measured instrument function that is
represented as a **sum of two Gaussians**, inside `cgauss.pro` / `clgauss.pro`
(R.E. Bell, PPPL, 19-Aug-1999):

```
/u/mlampert/rbell/chers_analysis/cgauss.pro
    conv[3i+0] = AMPLITUDE, conv[3i+1] = POSITION, conv[3i+2] = FWHM
    ww  = a_fwhm^2 + wj^2                      ; convolution of two Gaussians
```

The per-fiber instrument function is stored in the IDL save files
`disp_conv_<A|B><nnn>.dat` under `/u/mlampert/rbell/chers/`, tag
`cala.CONV[51,6]`, **in detector pixels**. `map_br_5.pro` (the routine that
produces the CHERS brightness spectra) converts it to wavelength:

```
/u/mlampert/rbell/chers_analysis/nstx/map_br_5.pro  (lines 226-236)
    convp = cala.conv[*,ifiber]
    dldp  = cala.disp[ifiber]        ; Angstrom / pixel, per fiber
    widl  = widp * dldp              ; instrument FWHM in Angstrom
```

Those two lines are exactly what the CSV reproduces.

### How it was determined

`chers_cal_files_2016.pro`, line 190:

> `; Specify dispersion/instrumental width files from Neon Glow Calibrations`

A neon glow-discharge lamp is recorded through the CHERS fibers. The two Ne I
reference lines are stored in the same file as `cala.LAMBDA = [5298.190,
5304.756] A`. Each line is modelled as the **same** two-Gaussian instrument
shape (shared amplitude ratio, shared separation, shared widths) with free
amplitude and position, over a linear background -- 10 free parameters, which is
exactly the length of the `cala.FIT[51,10]` vector stored alongside. The
reduced chi-square of each fiber's fit is in `cala.CHISQ`.

Verification that the stored `FIT` vector really is that model, for
`disp_conv_A019.dat` fiber 0:
`FIT[9] - FIT[7] = 102.379 - 77.902 = 24.477 px`, and
`24.477 px x 0.26825 A/px = 6.566 A = LAMBDA[1] - LAMBDA[0]` exactly.
The intensity-weighted centroid of the two instrument Gaussians,
`(a1 w1 p1 + a2 w2 p2)/(a1 w1 + a2 w2) = 78.885 px`, is the true position of
Ne I 5298.190 A -- which is the normalisation `cgauss.pro` applies
(`dj = dj - total(aj*wj*dj)/total(aj*wj)`).

### What is in the CSV

Per chord, per calibration epoch:

- `sigma_inst_nm` -- **standard deviation of the composite two-Gaussian
  instrument profile**, computed as the area-weighted mixture variance
  `sigma^2 = sum_j f_j (sigma_j^2 + (p_j - pbar)^2)` with `f_j` proportional to
  `a_j w_j` (the area weighting `cgauss.pro` itself uses). This is the quantity
  to use if you want a single-Gaussian equivalent: it reproduces the correct
  second moment, so `Ti_true = Ti_apparent - Ti_inst` holds in a variance sense.
- `fwhm_inst_nm` -- the **true numerical FWHM of the composite profile**, found
  by evaluating the two-Gaussian sum on a fine grid. This is *not*
  `2.3548 x sigma_inst_nm`, because the profile is not a single Gaussian (the
  two components sit ~1.8 px apart). Both columns are given so neither is
  implied by the other.
- `g1_*`, `g2_*` -- the two components themselves (amplitude, pixel position,
  FWHM in nm). **Use these if you want to reproduce the fit exactly** rather
  than a Gaussian approximation.
- `dispersion_nm_per_px` -- `cala.DISP[fiber]/10`.
- `lambda0_nm` -- `cala.WCAL[fiber,0]/10`, the wavelength at pixel 0. The
  wavelength axis is `lambda(px) = WCAL[0] + WCAL[1]*px`, with
  `WCAL[1] == DISP`.
- `R_m` -- tangency radius from `spatiala.RT` in `chers_radii_*.dat`,
  converted cm -> m. Cross-checked against the previously delivered
  `chord_radii_{137711,115600,116939}.csv`: max |difference| 5e-6 m over 51
  chords (rounding only).
- `px_CVI`, `px_cal_line` -- pixel of C VI 529.0525 nm and of the calibration
  line. See caveat 2.

### Summary of the values

| epoch | shots | median sigma | median FWHM(2.3548 sigma) | equivalent apparent Ti(C) |
|---|---|---|---|---|
| FY2004a | 111933-112493 | 41.5 pm | 97.7 pm | ~69 eV (54-102) |
| FY2005  | 114754-118926 | 41.6 pm | 98.0 pm | ~69 eV (53-108) |
| FY2010  | 137000-143699 | 42.3 pm | 99.6 pm | ~71 eV (54-120) |
| NSTX-Ua | 200000-204936 | 39.0 pm | 91.8 pm | ~61 eV (52-86)  |

Across all 459 rows: `sigma_inst` 0.0361-0.0565 nm, composite FWHM
0.0894-0.1517 nm, dispersion 0.02443-0.02691 nm/px. The width tends to grow from the
inboard to the outboard chords -- chord-index/sigma correlation +0.30 to +0.69
depending on epoch -- but it is **not** monotonic in any epoch, so use the
per-chord value, not a fit in chord index.
The "equivalent apparent Ti(C)" column is a sanity check, not a deliverable:
it is the carbon temperature that would produce the same Doppler width.

### Reliability

All NSTX rows are **measured**, read directly out of the calibration files the
production fitting code loads. Caveats, in order of importance:

1. **The composite instrument profile is not Gaussian.** The two components are
   separated by ~1.8 px (~0.05 nm), so the profile is slightly asymmetric and
   broader in the wings than a Gaussian of the same sigma. For an ML comparison
   against DIII-D a single sigma is probably what you want, but the exact fit
   used `g1_*`/`g2_*`.
2. **The instrument function is measured ~29 px away from the C VI line.** Ne I
   5298.190 A lands ~29 px (~0.77 nm) redward of C VI 5290.525 A; the offset is
   27.4-31.0 px across the 51 chords. The analysis code applies one instrument
   function per fiber regardless of position on the detector, so any variation
   of the instrument width across that 29 px is not captured -- by the code, and
   therefore not by this CSV either.
3. **Epoch granularity, not shot granularity.** One calibration per run
   campaign; there is no per-shot instrument function for CHERS.
4. Two epochs in `CAL_EPOCHS` are **not** in the CSV, deliberately:
   - `FY2011+` (shots >= 143700): the highest NSTX shot in the archive is
     142524, so no shot falls in it, and its radii file `chers_radii_A9.dat` is
     missing from the archive anyway.
   - `NSTX-Ub` (shots >= 204937): `disp_conv_A202.dat` is absent from the
     archive. The highest NSTX-U shot is 204742, so again no shot falls in it.
   Every shot in the dataset is covered by the 9 epochs that are present.
5. `chers_radii_a5.dat` is stored as `chers_radii_A5.dat`; the script resolves
   calibration filenames case-insensitively, as `make_dataset.py` does.

---

## 2. DIII-D CER, tangential chords

**No per-chord instrument width could be obtained, and none is filled in.**
The width columns in `instrument_function_d3d_cer.csv` are empty on purpose.

This machine has no DIII-D data access and no DIII-D code; everything below is
from the published literature. What is established:

**Method** (K.H. Burrell, B.A. Grierson, W.M. Solomon, E.A. Belli, OSTI 1352315):

> "We measure the instrumental response by detecting the Xe II line at 5292.22
> A, which is excited by neutral beam injection in to the tokamak vacuum vessel
> filled with xenon gas to a pressure around 0.1 mTorr."

> "The shape of this line is fit with a model which is the sum of Gaussians of
> various widths, locations and amplitudes."

> "This model is the convolution of a Gaussian line shape radiated by the plasma
> with the measured instrumental response. Since the instrumental response was
> fit with a sum of Gaussians, the convolution integral is trivial to do."

So **DIII-D uses the same scheme as NSTX**: a sum-of-Gaussians instrument
function, convolved (not quadrature-subtracted) inside the fitting code. The
two systems are directly comparable in form. The paper also notes the shape
"is influenced by the precise position of the fibers on the entrance slit and
the width of the entrance slit (typically 250 to 300 um)", i.e. it is genuinely
per-channel, and the real-time CER paper (Rev. Sci. Instrum. 97, 103501 (2026))
adds that "the instrumental response and dispersion are a function of
wavelength", so there is one set per wavelength setting.

**Wavelength/dispersion** (P. Gohil et al., Rev. Sci. Instrum. 70, 878 (1999);
restated in the two papers above): Ne I lines from neon pen lamps in small
integrating spheres, moved in front of the fibers **after every discharge**.
Ne I quartet near 5290.5 A. Quoted accuracy: <1% on dispersion, ~0.01 px on the
pixel-to-wavelength conversion.

**Dispersion, the one published number** (C. Chrystal et al., OSTI 1287042,
HTPD 2016, Sec. II): for the 2016 edge upgrade -- 2/3 m scanning Czerny-Turner,
1400 g/mm grating, SRI Avanti-768 CCD (768x256, 18 um pixels), 8 view chords per
spectrometer -- "The dispersion is ~9 pixels/Angstrom", i.e. **0.0111 nm/px**.
It is written as approximate in the paper, and I have entered it as approximate.

**Chord count**: the same paper states the impurity CER system has **80 view
chords**, which matches your 80 channels. 38 of them are the edge set (half
tangential, half vertical), spaced ~8 mm in major radius covering **R =
2.18-2.32 m**. That range is the only DIII-D radial information I can give; it
is a coverage range for a chord subset, not a per-chord R.

**Where the per-chord geometry lives** (verified in OMFIT source,
`omfit_classes/omfit_omas_d3d.py :: setup_charge_exchange_hardware_description_d3d`,
omfit_classes 3.2026.39.2): MDSplus server `DIII-D`, tree `IONS`, path
`CER.<CERQUICK|CERAUTO|CERFIT>.<TANGENTIAL|VERTICAL>.CHANNEL<NN>`, nodes `R`,
`Z`, `VIEW_PHI`, `TIME`. Note `R` is **time-dependent** -- it is where the
sightline crosses the active beam, not a fixed tangency radius, so it is not
the same quantity as the NSTX `R_m` column.

**What is missing and how to get it.** The instrument function is CERFIT *input*
calibration, not an MDSplus output, and no published number exists. To fill the
DIII-D columns someone with GA access needs to, for the shot range you are
using:

1. locate the CERFIT calibration set for the relevant wavelength setting
   (529 nm) and campaign -- the Xe II 5292.22 A beam-into-gas instrument
   response fit, per channel;
2. read out the sum-of-Gaussians parameters per channel (amplitude, position,
   width), the same three-per-component form NSTX stores;
3. read the per-shot Ne I dispersion and wavelength zero point for the same
   channels.

C. Chrystal (General Atomics) is the author on both the edge-CER upgrade and the
real-time CER papers and is the natural contact.

**Reliability of the DIII-D rows**: the method, the calibration lines, the
cadence, the chord count and the MDSplus node path are all read from sources and
are citable. The single dispersion value is read from a paper but published as
approximate and applies only to the 2016 edge spectrometers. **No instrument
width is given, because none could be read from a calibration -- it has not been
estimated.**

---

## Reproducing

- NSTX: `instrument_function_nstx_chers.csv` is produced by the scratch script
  `nstx_inst.py` (session scratchpad), which reads only
  `/u/mlampert/rbell/chers/*.dat` via `scipy.io.readsav`.
- DIII-D: `instrument_function_d3d_cer.csv` is hand-assembled from the cited
  sources; `d3d_inst.py` in the same scratchpad just writes it out.
