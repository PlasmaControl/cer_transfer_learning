"""Pool per-shot beam-off verdicts (cer_transfer.beamoff.analyze --csv) into the
summary numbers.

    pixi run python -u -m cer_transfer.beamoff.pool_beamoff gallery/beamoff_pool.csv
"""
import csv
import sys

import numpy as np


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    f = lambda k: np.array([float(r[k]) for r in rows if r.get(k)])
    n = len(rows)
    print(f"{n} discharges | verified beam-off frames: "
          f"{int(f('n_off').sum())} total ({int(f('n_notch').sum())} notch, "
          f"{int(f('n_tail').sum())} tail, {int(f('n_head').sum())} head)")
    for tn, u in (("vtor", "km/s"), ("ti", "eV")):
        print(f"\n{tn}:")
        for k, lab in (("diff", f"inter-array median |diff| ({u})"),
                       ("diff_ratio", "  ... relative to beam-on"),
                       ("z", "median |z| (0.674 = calibrated)"),
                       ("sig_infl", "sigma inflation beam-off"),
                       ("reentry_ratio", "re-entry / global error"),
                       ("step_bg", "step at transitions: background"),
                       ("step_fg", "step at transitions: foreground")):
            v = f(f"{tn}_{k}")
            if v.size:
                print(f"  {lab:34s} median {np.median(v):7.3g}  "
                      f"range {v.min():.3g}-{v.max():.3g}  (n={v.size})")
        rr = f(f"{tn}_reentry_ratio")
        if rr.size:
            print(f"  re-entry within +-25% of global: "
                  f"{int(((rr > 0.75) & (rr < 1.25)).sum())}/{rr.size}")
        sb, sf = f(f"{tn}_step_bg"), f(f"{tn}_step_fg")
        if sb.size and sb.size == sf.size:
            print(f"  background steps less than foreground at transitions:"
                  f" {int((sb < sf).sum())}/{sb.size}")


if __name__ == "__main__":
    main()
