r"""Chronological NSTX splits: train on a window of consecutive campaigns, evaluate later.

Campaigns are the CHERS calibration periods of ``chord_geometric_factor_nstx.csv``
(FY2005 ... FY2010); every boundary between them lies inside an operational break of
760-1600 shot numbers without a usable discharge, so no run day is split. The
discharges are those of the curated random split (union of splits/nstx_{train,val,test}.txt).

For every window of consecutive campaigns that leaves at least one later campaign
(FY2005, FY2005-06, ..., FY2006, ..., FY2009: 15 windows) this writes

* ``<out>/nstx_<window>_train.txt``: the window without its most recent discharges,
* ``<out>/nstx_<window>_val.txt``: the most recent ``--val-frac`` of the window
  (the last discharges before the cut; model selection on the newest data),

and for every campaign ``<out>/nstx_<campaign>.txt`` (all its discharges, the
evaluation lists), plus ``<out>/README.txt`` with the sizes.

    pixi run python -m cer_transfer.datasets.make_chrono_splits --out splits/chrono
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def shot_of(entry: str) -> int:
    """Discharge number of a list entry (six digits in the file name)."""
    return int(re.search(r"(\d{6})", Path(entry).name).group(1))


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lists", nargs="+",
                   default=["splits/nstx_train.txt", "splits/nstx_val.txt", "splits/nstx_test.txt"])
    p.add_argument("--geometry-csv", default="chord_geometric_factor_nstx.csv",
                   help="campaign (epoch) shot ranges")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--out", type=Path, default=Path("splits/chrono"))
    args = p.parse_args()

    entries = sorted({ln.strip() for f in args.lists for ln in open(f)
                      if ln.strip() and not ln.startswith("#")}, key=shot_of)
    epochs = sorted({(int(r["shot_from"]), int(r["shot_to"]), r["epoch"])
                     for r in csv.DictReader(open(args.geometry_csv))})
    camp = {}
    for e in entries:
        s = shot_of(e)
        name = next((n for lo, hi, n in epochs if lo <= s <= hi), None)
        if name is None:
            raise SystemExit(f"{e}: shot {s} in no campaign of {args.geometry_csv}")
        camp.setdefault(name, []).append(e)
    names = [n for _, _, n in epochs if n in camp]  # chronological
    args.out.mkdir(parents=True, exist_ok=True)

    report = [f"Chronological NSTX splits ({len(entries)} discharges, validation = most recent "
              f"{100 * args.val_frac:.0f} % of each training window)", "",
              "campaign lists (evaluation):"]
    for n in names:
        (args.out / f"nstx_{n}.txt").write_text("\n".join(camp[n]) + "\n")
        report.append(f"  nstx_{n}.txt: {len(camp[n])} discharges, shots "
                      f"{shot_of(camp[n][0])}-{shot_of(camp[n][-1])}")
    report += ["", "training windows (train / val; evaluated on every later campaign):"]
    for i in range(len(names) - 1):
        for j in range(i, len(names) - 1):
            win = [e for n in names[i:j + 1] for e in camp[n]]  # chronological
            n_val = max(1, round(args.val_frac * len(win)))
            tag = names[i] if i == j else f"{names[i]}-{names[j]}"
            (args.out / f"nstx_{tag}_train.txt").write_text("\n".join(win[:-n_val]) + "\n")
            (args.out / f"nstx_{tag}_val.txt").write_text("\n".join(win[-n_val:]) + "\n")
            report.append(f"  {tag:15s} train {len(win) - n_val:5d}  val {n_val:4d} "
                          f"(shots {shot_of(win[-n_val])}-{shot_of(win[-1])})  "
                          f"-> evaluate on {', '.join(names[j + 1:])}")
    (args.out / "README.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
