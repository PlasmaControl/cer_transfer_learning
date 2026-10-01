"""Pool cer_transfer.beamoff.passive_moments --csv rows across shots.

    pixi run python -u -m cer_transfer.beamoff.pool_moments gallery/moments_pool.csv
"""
import csv
import sys

import numpy as np

rows = list(csv.DictReader(open(sys.argv[1])))
f = lambda k: np.array([float(r[k]) for r in rows if r.get(k)])
print(f"{len(rows)} discharges | verified beam-off frames {int(f('n_off').sum())} "
      f"| chords with a valid moment reference: median {np.median(f('n_chords')):.0f}")
for tn, u in (("vtor", "km/s"), ("ti", "eV")):
    ro, no, do, ra = f(f"{tn}_ref_on"), f(f"{tn}_nn_on"), f(f"{tn}_d_off"), f(f"{tn}_ratio")
    if not ra.size:
        print(f"{tn}: no shot with a valid reference"); continue
    print(f"{tn} ({u}), n={ra.size} discharges:")
    print(f"  beam on (held-out fits): moment reference error {np.median(ro):.3g}, "
          f"network error {np.median(no):.3g}")
    print(f"  beam off: |network - moment reference| {np.median(do):.3g} = "
          f"x{np.median(ra):.2f} the reference's own beam-on error "
          f"(range {ra.min():.2f}-{ra.max():.2f}); within 1.5x in "
          f"{int((ra < 1.5).sum())}/{ra.size}")
