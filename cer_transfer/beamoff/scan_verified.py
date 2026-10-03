"""Find VERIFIED beam-off phases across all discharges present in both
arrays' validation lists (beam state from the spectra; only phases
bounded by clean beam switches are kept, see cer_transfer/beamstate.py).

    pixi run python -u -m cer_transfer.beamoff.scan_verified \
        --bg-list splits/nstx_passive_val.txt \
        --fg-list splits/nstx_active_val.txt \
        --out gallery/beamoff_verified.csv [--top 30]
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from cer_transfer.beamstate import beam_state
from cer_transfer.figures.common import FRAME_HZ

FS = FRAME_HZ  # overridden by --fs
T_OFF = -0.235


def shotmap(list_path):
    """Map shot number -> joblib path for every entry of a file list."""
    m = {}
    for ln in Path(list_path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        g = re.findall(r"(\d{6})", Path(ln).stem)
        if g:
            m[g[-1]] = ln
    return m


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--fs", type=float, default=FRAME_HZ, help="frame rate (Hz)")
    p.add_argument("--bg-list", type=Path, required=True)
    p.add_argument("--fg-list", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("gallery/beamoff_verified.csv"))
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    global FS
    FS = args.fs

    bgm, fgm = shotmap(args.bg_list), shotmap(args.fg_list)
    shots = sorted(set(bgm) & set(fgm))
    if args.limit:
        shots = shots[: args.limit]
    print(f"{len(shots)} discharges in both validation lists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    n_rej = {"not a beam switch": 0, "no bounding transition": 0}
    n_fg, n_fg_flat = 0, 0
    for s in shots:
        try:
            st = beam_state(bgm[s], fgm[s])
        except Exception as e:  # corrupt/odd file: skip, report
            print(f"  skip {s}: {e}")
            continue
        n_fg += st.get("n_fg_steps", 0)
        n_fg_flat += st.get("n_fg_steps_bg_flat", 0)
        ver = {"head": 0, "notch": 0, "tail": 0}
        for a, b, cls, ok, why in st["segments"]:
            rows.append(
                dict(
                    shot=s,
                    start=a,
                    end=b,
                    cls=cls,
                    verified=ok,
                    why=why,
                    t0=f"{a / FS + T_OFF:.3f}",
                    t1=f"{b / FS + T_OFF:.3f}",
                )
            )
            if ok:
                ver[cls] += b - a + 1
            else:
                n_rej[why] += 1
        if sum(ver.values()):
            summary.append((ver["notch"], ver["tail"], ver["head"], s))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["shot"])
        w.writeheader()
        w.writerows(rows)
    summary.sort(reverse=True)
    print(
        f"{'notch':>6} {'tail':>5} {'head':>5}  shot   (verified "
        "beam-off frames, 5 ms each)"
    )
    for no, ta, he, s in summary[: args.top]:
        print(f"{no:6d} {ta:5d} {he:5d}  {s}")
    tot = (
        [sum(x) for x in zip(*[(a, b, c) for a, b, c, _ in summary])]
        if summary
        else [0, 0, 0]
    )
    print(
        f"\n{len(summary)} discharges with verified beam-off phases: "
        f"{tot[0]} notch / {tot[1]} tail / {tot[2]} head frames. "
        f"Rejected segments: {n_rej['not a beam switch']} not a beam "
        f"switch, {n_rej['no bounding transition']} without bounding "
        f"transition. Segments -> {args.out}"
    )
    print(
        f"foreground-brightness steps: {n_fg}; background flat at "
        f"{n_fg_flat} of them ({100 * n_fg_flat / max(n_fg, 1):.0f}%) -- "
        "NOT a beam-specific test: includes plasma events that brighten "
        "both arrays; use beam_switch_test.py with NB-power switch times"
    )


if __name__ == "__main__":
    main()
