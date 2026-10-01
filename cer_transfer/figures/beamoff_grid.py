"""Several discharges through phases without the diagnostic beam, one
panel each: reconstruction from the passive spectra (1-sigma band) and
conventional fits, with the phase without conventional measurement in
grey. Discharges are selected objectively: the N longest verified
mid-discharge beam-off phases in the scanner CSVs (startup excluded).

    pixi run python -u -m cer_transfer.figures.beamoff_grid \
        --verified gallery/beamoff_verified.csv gallery/beamoff_verified_test.csv \
        --n 4 --chord 2 --target vtor --out figs --out-name beamoff_grid
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cer_transfer.beamstate import beam_state

sns.set_style("whitegrid")
from cer_transfer.figures.common import FRAME_HZ

FS = FRAME_HZ  # overridden by --fs


GALLERY = Path("gallery")   # overridden by --gallery


def files_for(shot):
    """Prediction dump and file lists of a shot: <gallery>/<prefix>_<shot>_bg.npz
    plus _bg.txt / _fg.txt, prefix 'bo' (validation) or 'bt' (test)."""
    for p in ("bo", "bt"):
        npz = GALLERY / f"{p}_{shot}_bg.npz"
        if npz.exists():
            return (npz, (GALLERY / f"{p}_{shot}_bg.txt").read_text().strip(),
                    (GALLERY / f"{p}_{shot}_fg.txt").read_text().strip())
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", type=float, default=FRAME_HZ, help="frame rate (Hz)")
    p.add_argument("--gallery", type=Path, default=Path("gallery"),
                   help="folder with the per-shot prediction dumps and file lists")
    p.add_argument("--verified", nargs="+", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--pick", choices=("longest", "median"), default="longest",
                   help="longest: longest verified notches; median: discharges "
                        "whose beam-off/beam-on ratio (edge test) is closest "
                        "to the median of all discharges -- representative cases")
    p.add_argument("--edge-csv", nargs="*", default=[],
                   help="cer_transfer.beamoff.edge_frame_test CSVs (needed for --pick median)")
    p.add_argument("--min-notch-ms", type=float, default=0.0,
                   help="only discharges whose longest verified notch is at "
                        "least this long (ms)")
    p.add_argument("--shots", type=str, default=None,
                   help="override the selection: comma-separated shots")
    p.add_argument("--chord", type=int, default=2)
    p.add_argument("--target", choices=("ti", "vtor"), default="vtor")
    p.add_argument("--t-offset", type=float, default=-0.235)
    p.add_argument("--margin", type=float, default=0.06)
    p.add_argument("--out", type=Path, default=Path("figs"))
    p.add_argument("--out-name", type=str, default="beamoff_grid")
    args = p.parse_args()
    global GALLERY
    GALLERY = args.gallery
    global FS
    FS = args.fs
    args.out.mkdir(parents=True, exist_ok=True)

    # longest verified notch per shot
    best = {}
    for path in args.verified:
        for r in csv.DictReader(open(path)):
            if r["verified"] != "True" or r["cls"] != "notch":
                continue
            L = int(r["end"]) - int(r["start"]) + 1
            if L > best.get(r["shot"], (0,))[0]:
                best[r["shot"]] = (L, int(r["start"]), int(r["end"]))
    ratio = {}
    for path in args.edge_csv:
        for r in csv.DictReader(open(path)):
            key = f"{args.target}_ratio"
            if r.get(key) not in (None, "", "nan"):
                ratio.setdefault(r["shot"], float(r[key]))
    if args.shots:
        chosen = [s.strip() for s in args.shots.split(",")]
        how = "given"
    elif args.pick == "median":
        if not ratio:
            raise SystemExit("--pick median needs --edge-csv")
        med = float(np.median(list(ratio.values())))
        cand = [s for s in ratio if s in best and files_for(s) is not None
                and best[s][0] * 5 >= args.min_notch_ms]
        chosen = sorted(cand, key=lambda s: abs(np.log(ratio[s] / med)))[:args.n]
        how = (f"closest to the median ratio {med:.2f} of {len(ratio)} discharges"
               + (f", notch >= {args.min_notch_ms:g} ms" if args.min_notch_ms else ""))
    else:
        chosen = [s for s, _ in sorted(best.items(), key=lambda kv: -kv[1][0])
                  if files_for(s) is not None][:args.n]
        how = "longest verified notch"
    print(f"selected ({how}): " + ", ".join(
        f"{s} ({best.get(s, (0,))[0] * 5} ms"
        + (f", ratio {ratio[s]:.2f})" if s in ratio else ")") for s in chosen))

    k = 1 if args.target == "vtor" else 0
    unit = "$v_{tor}$ (km/s)" if k else "$T_i$ (eV)"
    n = len(chosen)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    for ctx, ext in (("talk", "png"), ("paper", "pdf")):
        with sns.plotting_context(ctx):
            size = (7.2, 2.6 * nrow) if ctx == "paper" else (13, 4.5 * nrow)
            fig, axes = plt.subplots(nrow, ncol, figsize=size, squeeze=False)
            for ax in axes.ravel()[n:]:
                ax.set_visible(False)
            for ax, s in zip(axes.ravel(), chosen):
                fz = files_for(s)
                if fz is None:
                    ax.set_title(f"{s}: files missing", fontsize="small")
                    continue
                npz, fb, ff = fz
                d = np.load(npz)
                ch = d["chord"]
                C = int(ch.max()) + 1
                T = len(ch) // C
                pr = d["pred"].reshape(C, T, -1)[args.chord, :, k]
                ps = d["pred_sigma"].reshape(C, T, -1)[args.chord, :, k]
                yy = d["y"].reshape(C, T, -1)
                fitted = np.isfinite(yy[..., k]).any(axis=0)
                y = yy[args.chord, :, k]
                st = beam_state(fb, ff)
                fr = np.where(fitted)[0]
                spans = []
                for a, b, cls, ok, _ in st["segments"]:
                    if not ok or cls != "notch" or a >= T:
                        continue
                    left, right = fr[fr < a], fr[fr > b]
                    s0 = int(left[-1]) + 1 if left.size else a
                    s1 = int(right[0]) - 1 if right.size else min(b, T - 1)
                    if spans and s0 <= spans[-1][1] + 1:
                        spans[-1] = (spans[-1][0], max(spans[-1][1], s1))
                    else:
                        spans.append((s0, s1))
                if not spans:
                    ax.set_title(f"{s}: no verified notch", fontsize="small")
                    continue
                a, b = max(spans, key=lambda ab: ab[1] - ab[0])
                t = np.arange(T) / FS + args.t_offset
                m = int(round(args.margin * FS))
                f0, f1 = max(a - m, 0), min(b + m, T - 1)
                w = np.arange(f0, f1 + 1)
                ax.axvspan(t[a] - 0.5 / FS, t[b] + 0.5 / FS, color="0.88",
                           zorder=0, lw=0)
                ax.fill_between(t[w], pr[w] - ps[w], pr[w] + ps[w], color="C0",
                                alpha=0.22, lw=0)
                ax.plot(t[w], pr[w], "-", color="C0", lw=1.5,
                        label="reconstruction from passive spectra")
                mm = np.isfinite(y[w])
                ax.plot(t[w][mm], y[w][mm], "o", color="k", ms=3, zorder=5,
                        label="conventional fit")
                ax.set_xlim(t[f0], t[f1])
                ax.set_title(f"discharge {s}", fontsize="small")
                ax.set_xlabel("time (s)")
                ax.set_ylabel(f"chord {args.chord}\n{unit}", fontsize="small")
            axes.ravel()[0].legend(fontsize="xx-small", loc="best")
            fig.tight_layout()
            fig.savefig(args.out / f"{args.out_name}.{ext}", dpi=300,
                        bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {args.out}/{args.out_name}.png/.pdf")


if __name__ == "__main__":
    main()
