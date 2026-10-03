"""Hero-shot gallery: sample N shots common to the legacy, active, and
passive validation lists; run the three one-shot evals per shot; render
the Fig-2-style (legacy) and Fig-3-style (passive vs active) composites.

    pixi run python -u -m cer_transfer.figures.gallery --n 20 --seed 42 \
        [--out gallery] [--dry-run]

GPU recommended (60 one-shot evals); use gallery.sbatch for batch mode.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

RECIPES = {  # one consistent training recipe across all three arrays
    "ft": {
        "legacy": "cer_ckpts/nstx_ft.pt",
        "active": "cer_ckpts/nstx_active_ft.pt",
        "passive": "cer_ckpts/nstx_passive_ft.pt",
    },
    "scratch": {
        "legacy": "cer_ckpts/nstx_scratch.pt",
        "active": "cer_ckpts/nstx_active_scratch.pt",
        "passive": "cer_ckpts/nstx_passive_scratch.pt",
    },
}
LISTS = {
    "legacy": "splits/nstx_val.txt",
    "active": "splits/nstx_active_val.txt",
    "passive": "splits/nstx_passive_val.txt",
}


def shot_ids(list_path: Path) -> dict[str, str]:
    """Map shot number -> joblib path for every entry of a file list."""
    out = {}
    for ln in list_path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        m = re.search(r"(\d{6})", Path(ln).stem)
        if m:
            out[m.group(1)] = ln
    return out


def run(cmd, dry):
    """Run a command (or print it with ``dry``); return success."""
    print("$", " ".join(str(x) for x in cmd))
    if dry:
        return True
    r = subprocess.run([str(x) for x in cmd])
    if r.returncode:
        print(
            f"!! FAILED (continuing with next shot): "
            f"{' '.join(str(x) for x in cmd)}"
        )
        return False
    return True


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("gallery"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--recipe",
        choices=sorted(RECIPES),
        default="ft",
        help="checkpoint family: same training recipe for all "
        "three arrays (no mixed comparisons)",
    )
    for k in ("legacy", "active", "passive"):
        p.add_argument(
            f"--ckpt-{k}", type=Path, default=None, help=f"override the {k} checkpoint"
        )
    args = p.parse_args()
    ckpt = dict(RECIPES[args.recipe])
    for k in ckpt:
        ov = getattr(args, f"ckpt_{k}")
        if ov is not None:
            ckpt[k] = str(ov)
    missing = [k for k, v in ckpt.items() if not Path(v).exists()]
    if missing and not args.dry_run:
        sys.exit(
            f"missing checkpoints for recipe '{args.recipe}': "
            f"{[ckpt[k] for k in missing]} -- override with "
            f"--ckpt-<name> or pick the other recipe"
        )

    maps = {k: shot_ids(Path(v)) for k, v in LISTS.items()}
    common = sorted(set(maps["legacy"]) & set(maps["active"]) & set(maps["passive"]))
    print(f"common shots across the three val lists: {len(common)}")
    if not common:
        sys.exit("no common shots -- check list paths / id pattern")
    rng = np.random.default_rng(args.seed)
    picks = sorted(
        rng.choice(common, size=min(args.n, len(common)), replace=False).tolist()
    )
    print("selected:", ", ".join(picks))

    args.out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    failed = []
    for s in picks:
        d = args.out / s
        d.mkdir(exist_ok=True)
        ok = True
        for kind in ("legacy", "active", "passive"):
            lst = d / f"{kind}.txt"
            if not args.dry_run:
                lst.write_text(maps[kind][s] + "\n")
            npz = d / f"{kind}.npz"
            ok &= run(
                [
                    py,
                    "-u",
                    "eval_checkpoint.py",
                    "--checkpoint",
                    ckpt[kind],
                    "--list",
                    lst,
                    "--dump-preds",
                    npz,
                ],
                args.dry_run,
            )
        renders = (
            (
                "recon_legacy",
                [
                    "--preds",
                    d / "legacy.npz",
                    "--label1",
                    "model (51-chord)",
                    "--shot",
                    maps["legacy"][s],
                ],
            ),
            (
                "recon_active",
                [
                    "--preds",
                    d / "active.npz",
                    "--label1",
                    "active model",
                    "--shot",
                    maps["active"][s],
                ],
            ),
            (
                "recon_background_vs_active",
                [
                    "--preds",
                    d / "passive.npz",
                    "--label1",
                    "background model",
                    "--preds2",
                    d / "active.npz",
                    "--label2",
                    "active model",
                    "--shot",
                    maps["passive"][s],
                ],
            ),
        )
        for name, extra in renders:
            ok &= run(
                [
                    py,
                    "-u",
                    "-m",
                    "cer_transfer.figures.recon_composite",
                    *extra,
                    "--t-offset",
                    "-0.235",
                    "--unl-gap",
                    "2",
                    "--out",
                    d,
                    "--out-name",
                    f"{name}_{s}",
                ],
                args.dry_run,
            )
        if not ok:
            failed.append(s)
    if failed:
        print(
            f"shots with failures (renders may be incomplete): " f"{', '.join(failed)}"
        )
    print(
        f"gallery under {args.out}/<shot>/: recon_legacy_*, "
        f"recon_active_*, recon_passive_vs_active_* (png+pdf)"
    )


if __name__ == "__main__":
    main()
