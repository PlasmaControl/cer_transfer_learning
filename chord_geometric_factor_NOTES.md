# NSTX CHERS active array: geometric factor g, v_los = g * v_tor

Compiled 2026-10-19. Read-only: nothing in the CHERS analysis tree, the
calibration archive or the MDSplus mirrors was modified.

Deliverable: `chord_geometric_factor_nstx.csv` -- 1377 rows
= 9 calibration epochs x 51 chords x 3 beam sources.

**Short answer.** `g = R_tangency / R_intersection`, and the measurement point
the analysis uses is the intersection with **NBI source B**. So use the rows
with `is_analysis_radius = 1`. `g` runs from **0.99659 to 1.00000**, i.e. the
sightline is within **0.00 to 4.74 degrees** of toroidal everywhere -- the
correction is at most 0.34%.

---

## 1. All tags of `spatiala` in `chers_radii_*.dat`

Built by `/u/mlampert/rbell/chers_analysis/cx/compute_spatial_a8.pro`.
All lengths in **cm**, all angles in **radians**. IDL shape `[51,3]` reads back
through `scipy.io.readsav` as `(3,51)`.

| tag | shape | meaning |
|---|---|---|
| `RT` | (51,) | tangency radius of the sightline |
| `RINT` | (3,51) | **major radius where the sightline crosses beam source A/B/C** |
| `VELANG` | (3,51) | angle between sightline and the toroidal velocity vector |
| `RB` | (3,51) | distance along the beam from the crossover point to the intersection |
| `BEAMANG` | (3,51) | angle between sightline and beam (`delta` in the source) |
| `LB` | (51,3) | path length through each radial zone, per beam |
| `LB2` | (51,3) | same, measured from the zone's tangency radius |
| `FILENAME` | scalar | the file's own path as written |

Beam geometry constants hard-coded in `compute_spatial_a8.pro`:

```
thc        = 10.702 deg
Rcrossover = 158.58 cm
Rt_beam    = [69.38, 59.13, 48.59] cm   ; sources A, B, C
nsources   = 3
```

This is the original NSTX NBI-1 box; the second box (NSTX-U) is not in these
files.

## 2. Which beam source the analysis assumes

`/u/mlampert/rbell/ca/cxf_active_2.pro:68` and `cxf_p4_2016.pro:118`:

```idl
radius = SpatialA.Rint[*,1]     ; radius at intersection of Source B
```

and `corr_cx_adas_4.pro:162` just inherits it (`radius = cin.radius`).

I checked this against the archive rather than trusting the 2016 code alone.
`\ACTIVESPEC::TOP.CHERS.ANALYSIS.CT1:RADIUS` (units `cm`, 51 points) equals
`RINT[:,1]` to **0.0000 cm** for shot 115600 (FY2005), 137711 (FY2010) and
204202 (NSTX-U) -- while differing from `RT` by up to 0.54 cm and from
`RINT[:,0]`/`RINT[:,2]` by 0.24-0.62 cm. So source B is the measurement point
in every epoch, not only after the 2016 rewrite.

The *correction* step is a different matter: `corr_cx_adas_4.pro:307` passes
`velang[ir,knabc]` -- **all sources with beam voltage > 10 kV** -- and the
forward model sums over them weighted by the CX cross-section. So the effective
projection on a multi-source shot is a weighted blend, even though the radius
label is always source B's.

## 3. Where the line shift becomes a velocity

`/u/mlampert/rbell/ca/cxf_active_2.pro:226`:

```idl
Vt2[ifiber,i] = ((a2[3])*dispa[ifiber])/lambda.lambda0*3e5
```

`a2[3]` is the fitted line position in pixels relative to the C VI rest pixel;
times the per-fiber dispersion gives the shift in Angstrom; `/lambda0*c` gives
km/s. **No geometric factor is applied here.** Despite the name, `ca.Vt` is the
raw line-of-sight velocity.

The projection enters only in the atomic-physics correction.
`/u/mlampert/rbell/chers_analysis/cx/cxpr_vh.pro:88` (the forward model):

```idl
vrx = rVphi*cos(VELang[nb]/!radeg)
```

`CorrTV_VH` then iterates `rVi` until the forward model's apparent velocity
matches the measured one. So **`cos(VELANG)` is never applied as a standalone
division** -- it is folded into the same inversion that removes the
cross-section weighting.

This is visible in the tree. Both quantities are stored:

- `...CT1:AVT` -- `avt = cin.vt` (`corr_cx_adas_4.pro:155`), i.e. **the raw
  Doppler line-of-sight velocity**, saved before the Zeeman step;
- `...CT1:VT` -- the corrected toroidal velocity.

Their ratio is **not** g: median `AVT/VT` is 0.952 (137711), 0.971 (204202),
0.985 (115600), where g alone would be ~0.997. The rest is the
cross-section weighting. If you want the purely geometric factor, use the CSV;
if you want what the pipeline actually did end-to-end, use `AVT` and `VT`.

## 4. A bug in the stored `VELANG` for sources A and B

`compute_spatial_a8.pro` computes `delta` inside the loop over sources:

```idl
for i=0,nsources-1 do begin
   delta = (thetam_main+alpham_main-(thc+alpha_crossover[i]))
   beamang[*,i] = delta
   Rint[*,i] = ...
endfor                                  ; <- delta now holds SOURCE C's value

velang = fltarr(nch,3)
for j=0,2 do begin
  for i=0,nch-1 do begin
     ...
     velang[i,j] = acos(...) - delta[i] - !pi/2      ; delta is never reindexed
  endfor
endfor
```

`delta` is not recomputed in the second loop, so **the stored `VELANG` for
sources A and B carries source C's beam angle**. `BEAMANG` was written per
source correctly inside the first loop, so the intended value is recoverable as
`VELANG[j] + BEAMANG[2] - BEAMANG[j]`.

Evidence, over all five distinct radii files in the archive
(`a3, a4, A7, A8, A11`):

| source | max abs(cos(VELANG_stored) - Rt/R) | max abs(cos(VELANG_fixed) - Rt/R) |
|---|---|---|
| A | 2.1e-02 | 1e-06 |
| B | 8.3e-03 | 1e-06 |
| C | 1e-06 | 1e-06 |

Source C agrees with the exact geometry already -- because C is the source whose
`delta` was left in the variable. Applying each source's own `delta` makes all
three agree to float precision. That is not a coincidence over five independent
files.

Size of the effect on an inferred `v_tor` (`= v_los/g`), using `chers_radii_A8.dat`:

| source | v_tor relative error | at v_tor = 200 km/s |
|---|---|---|
| A | +0.55% .. +2.16% | +1.1 .. +4.3 km/s |
| B | -0.28% .. +0.83% | -0.6 .. +1.7 km/s |
| C | 0 | 0 |

So the archived `VT` carries a systematic of order **1 km/s at 200 km/s** on
source-B-dominated shots, more if source A dominates. It is small, but it is
systematic in chord index and therefore does not average out in a radial
profile. I have **not** changed any file -- the CSV reports both numbers side by
side and you can decide.

## 5. Columns in `chord_geometric_factor_nstx.csv`

| column | meaning |
|---|---|
| `epoch`, `shot_from`, `shot_to` | calibration epoch and its shot range |
| `chord` | 0-51, same ordering as the 51-chord joblib `input` axis |
| `beam_source` | A, B or C |
| `is_analysis_radius` | 1 for source B -- the rows to use by default |
| `R_tangency_m` | `spatiala.RT`, cm -> m |
| `R_intersect_m` | `spatiala.RINT[source]`, cm -> m: the measurement point |
| **`g`** | **`R_tangency_m / R_intersect_m` -- the factor you asked for** |
| `g_as_coded` | `cos(spatiala.VELANG[source])` as stored, i.e. what the IDL correction actually used (see section 4) |
| `angle_deg` | `arccos(g)`, degrees |
| `velang_deg_stored` / `velang_deg_fixed` | `VELANG` as stored / with each source's own `delta` |
| `beamang_deg` | `spatiala.BEAMANG[source]`, degrees |
| `Rb_m` | `spatiala.RB[source]`, cm -> m |
| `Rt_beam_m` | beam tangency radius of that source |

`g` and `g_as_coded` are identical for source C and differ for A and B by the
amounts in section 4.

### Values, source B

| epoch | g, chord 0 | g, chord 50 |
|---|---|---|
| FY2004a/b | 0.99904 | 0.99664 |
| FY2005 | 0.99874 | 0.99699 |
| FY2006 | 0.99890 | 0.99720 |
| FY2007 | 0.99886 | 0.99716 |
| FY2008 | 0.99889 | 0.99709 |
| FY2009 | 0.99884 | 0.99668 |
| FY2010 | 0.99884 | 0.99668 |
| NSTX-Ua | 0.99891 | 0.99659 |

g is not monotonic in chord index: it rises to 1.00000 for the chords whose
sightline is tangent at the beam crossing, then falls again.

## 6. Caveats

1. `g` is a pure geometric projection at the source-B intersection. It is *not*
   the full apparent-to-true conversion: the energy-dependent CX cross-section
   and the gyro-orbit effect are separate corrections, both already inside the
   archived `VT`.
2. `R_intersect_m` is where the sightline crosses the beam *centre line*. The
   beam has finite width (`beam_width = 12.16 cm` in `corr_cx_ict1_vh.pro`), so
   the emission is a line integral over a range of R and the single `g` is an
   on-axis value.
3. The three sources give different `g` for the same chord. On a shot where the
   dominant source is not B, the pipeline's radius label is still B's.
4. `FY2011+` and `NSTX-Ub` are absent for the same reasons as in
   `instrument_function_NOTES.md`: no shot in the archive falls in either.

Produced by `nstx_geom.py` in the session scratchpad; it reads only
`/u/mlampert/rbell/chers/*.dat`.
