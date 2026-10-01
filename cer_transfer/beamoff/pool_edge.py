"""Pool cer_transfer.beamoff.edge_frame_test --csv rows across shots, overall and per
campaign, plus a headline pool over selected campaigns.

    pixi run python -u -m cer_transfer.beamoff.pool_edge gallery/edge_pool_val.csv [more.csv ...] \
        [--select 2005-06,2010]

Campaign boundaries are APPROXIMATE shot-number ranges -- verify against
the NSTX run logs and adjust CAMPAIGNS if needed.
"""
import argparse
import csv

import numpy as np

CAMPAIGNS = [  # (label, first shot, last shot) -- approximate, verify
    ("2005-06", 0, 121999),
    ("2007", 122000, 126499),
    ("2008", 126500, 130999),
    ("2009", 131000, 136999),
    ("2010", 137000, 999999),
]


def campaign(shot):
    s = int(shot)
    for lab, a, b in CAMPAIGNS:
        if a <= s <= b:
            return lab
    return "?"


def summarize(rows, title):
    f = lambda k: np.array([float(r[k]) for r in rows
                            if r.get(k) not in (None, "", "nan")])
    print(f"{title}: {len(rows)} discharges, "
          f"{int(sum(float(r['n_pairs']) for r in rows))} edge pairs")
    for tn, u in (("vtor", "km/s"), ("ti", "eV")):
        o, n = f(f"{tn}_off"), f(f"{tn}_on")
        ra, fc = f(f"{tn}_ratio"), f(f"{tn}_fitchange")
        if not ra.size:
            print(f"  {tn}: no data")
            continue
        print(f"  {tn} ({u}), n={ra.size}: beam off {np.median(o):.3g} | "
              f"beam on {np.median(n):.3g} | ratio median "
              f"{np.median(ra):.2f} (IQR {np.percentile(ra, 25):.2f}-"
              f"{np.percentile(ra, 75):.2f}) | change of fits themselves "
              f"{np.nanmedian(fc) if fc.size else float('nan'):.3g}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+")
    p.add_argument("--select", type=str, default=None,
                   help="comma-separated campaigns for a headline pool")
    args = p.parse_args()
    rows, seen = [], set()
    for path in args.csv:
        for r in csv.DictReader(open(path)):
            if r["shot"] not in seen:  # a shot counted once across files
                seen.add(r["shot"])
                rows.append(r)
    summarize(rows, "ALL")
    for lab, _, _ in CAMPAIGNS:
        sub = [r for r in rows if campaign(r["shot"]) == lab]
        if sub:
            summarize(sub, lab)
    if args.select:
        sel = {s.strip() for s in args.select.split(",")}
        summarize([r for r in rows if campaign(r["shot"]) in sel],
                  "SELECTED " + "+".join(sorted(sel)))


if __name__ == "__main__":
    main()
