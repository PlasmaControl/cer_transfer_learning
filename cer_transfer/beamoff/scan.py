"""Find beam-off candidate windows: plasma-active frames without fits.

Fits require the neutral beam, so long label gaps with bright spectra
are beam-off phases to first order (confirm top candidates against the
NB power trace). Ranks shots by the longest such gap that is bracketed
by fitted frames (enables re-entry validation).

    pixi run python -u -m cer_transfer.beamoff.scan --list splits/nstx_passive_val.txt \
        [--min-gap 20] [--top 25] [--limit N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from joblib import load
from cer_transfer.beamstate import valid_end


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", type=Path, required=True)
    p.add_argument("--min-gap", type=int, default=20,
                   help="minimum gap length in frames (20 = 100 ms)")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    files = [ln.strip() for ln in args.list.read_text().splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    if args.limit:
        files = files[:args.limit]

    rows = []
    for fp in files:
        d = load(fp, mmap_mode="r")
        end = valid_end(d)
        spec = np.asarray(d["input"][:, :end, :], dtype=np.float32)
        tgt = np.asarray(d["target"][:, :end, 0], dtype=np.float32)
        lab = np.isfinite(tgt).any(axis=0)
        if lab.sum() < 4:
            continue
        # plasma-active proxy: total line-band signal above the shot's
        # dark level (first frames), per frame
        tot = spec.sum(axis=(0, 2))
        # recordings begin 235 ms (47 frames) before the experimental
        # clock zero: those frames are pre-plasma by construction and
        # provide the dark reference; plasma cannot be active there
        PRE = min(47, end - 1)
        dw = tot[:max(8, PRE - 2)]
        dark = np.median(dw)
        noise = 1.4826 * np.median(np.abs(dw - dark))
        active = tot > dark + 8 * max(noise, 1e-6 * abs(dark) + 1e-6)
        active[:PRE] = False
        lo, hi = np.where(lab)[0].min(), np.where(lab)[0].max()
        # gaps INSIDE the fitted span (bracketed) + post-beam tail
        best = None
        run = 0
        for f in range(lo, hi + 1):
            if active[f] and not lab[f]:
                run += 1
            else:
                if run >= args.min_gap and (best is None or run > best[0]):
                    best = (run, f - run, f - 1)
                run = 0
        if run >= args.min_gap:
            best = max(best or (0, 0, 0), (run, hi + 1 - run, hi))
        # pre-beam startup (active before first fit): beam-off in
        # nearly every shot, validated at beam turn-on
        head = 0
        f = lo - 1
        while f >= 0 and active[f]:
            head += 1
            f -= 1
        # post-beam decay tail (active after last fit)
        tail = 0
        f = hi + 1
        while f < end and active[f]:
            tail += 1
            f += 1
        if best or tail >= args.min_gap or head >= args.min_gap:
            shot = Path(fp).stem
            rows.append((best[0] if best else 0, head, tail, shot,
                         best[1] if best else -1, best[2] if best else -1,
                         int(lab.sum())))
    rows.sort(reverse=True)
    print(f"{'gap':>5} {'head':>5} {'tail':>5} {'shot':>28} "
          f"{'gap frames':>16} {'#fits':>6}   (gap: bracketed; head: "
          "pre-beam startup; tail: post-beam)")
    for g, h, t, s, a, b, n in rows[:args.top]:
        w = f"[{a}..{b}]" if a >= 0 else "-"
        print(f"{g:5d} {h:5d} {t:5d} {s:>28} {w:>16} {n:6d}")
    print(f"\n{len(rows)} shots with candidate windows "
          f">= {args.min_gap} frames ({args.min_gap*5} ms). Confirm the "
          "top candidates against the NB power trace before use.")


if __name__ == "__main__":
    main()
