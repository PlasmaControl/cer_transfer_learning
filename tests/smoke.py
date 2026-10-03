"""Smoke test for the figure, beam-off and analysis modules.

Builds a small synthetic discharge (both spectrometer arrays, conventional
fits on every other frame, one beam notch), matching prediction dumps and
a chord-coordinate file, then runs every module that does not need
PyTorch and checks that it exits cleanly. No cluster data required.

    python -m tests.smoke            # from the repository root
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]


def build(tmp: Path):
    """Write the synthetic discharge, prediction dumps and lists into ``tmp``."""
    rng = np.random.default_rng(0)
    C, T, W = 51, 240, 40
    t = np.arange(T)
    x = np.arange(C)
    plasma = (t >= 55) & (t < 225)
    beam = plasma & (t >= 70) & ~((t >= 100) & (t <= 140))  # notch
    vt = 60 + 60 * np.tanh((t - 120) / 40.0)
    prof_v = 0.3 + 0.7 / (1 + np.exp((x - 22) / 3.0))
    prof_t = 250 + 850 / (1 + np.exp((x - 22) / 4.0))
    v = prof_v[:, None] * vt[None, :]
    ti = prof_t[:, None] * np.clip((t - 55) / 30.0, 0, 1)[None, :] + 50
    pix = np.arange(W)
    passive = 150 * np.exp(-0.5 * ((pix[None, :] - 20) / 2.0) ** 2) * plasma[:, None]
    active = 700 * np.exp(-0.5 * ((pix[None, :] - 24) / 3.0) ** 2) * beam[:, None]
    bg = 20 + passive[None].repeat(C, 0) + rng.normal(0, 2, (C, T, W))
    fg = 20 + (passive + active)[None].repeat(C, 0) + rng.normal(0, 2, (C, T, W))
    tgt = np.full((C, T, 2), np.nan, np.float32)
    for f in range(72, 222, 2):
        if beam[f]:
            tgt[:33, f, 0] = ti[:33, f] + rng.normal(0, 25, 33)
            tgt[:33, f, 1] = v[:33, f] + rng.normal(0, 3, 33)
    err = np.where(
        np.isfinite(tgt), np.where(np.arange(2) == 0, 25.0, 3.0), np.nan
    ).astype(np.float32)
    for nm, sp in (("bg", bg), ("fg", fg)):
        dump(
            {
                "input": sp.astype(np.float32),
                "target": tgt,
                "target_error": err,
                "end_index": -1,
            },
            tmp / f"chers_{nm}.joblib",
        )
    psig = np.where(x[:, None] >= 33, 12.0, 1.0)
    for nm, nt, nv in (("bg", 15, 2), ("fg", 8, 1)):
        pr = np.stack(
            [ti + rng.normal(0, nt, (C, T)), v + rng.normal(0, nv, (C, T))], -1
        )
        ps = np.stack([40 * psig * np.ones((C, T)), 5 * psig * np.ones((C, T))], -1)
        np.savez(
            tmp / f"{nm}.npz",
            chord=np.repeat(np.arange(C), T),
            pred=pr.reshape(-1, 2).astype(np.float32),
            pred_sigma=ps.reshape(-1, 2).astype(np.float32),
            y=tgt.reshape(-1, 2),
            sigma=err.reshape(-1, 2),
            targets=np.array(["ti", "vtor"]),
        )
    radii = 0.91 + 0.0131 * x
    with open(tmp / "radii.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["chord", "x"])
        for i, r in enumerate(radii):
            w.writerow([i, f"{r:.4f}"])
    # a shot-numbered copy for the scanner -> grid chain
    for nm in ("bg", "fg"):
        (tmp / nm).mkdir()
        (tmp / nm / "chers_141710.joblib").write_bytes(
            (tmp / f"chers_{nm}.joblib").read_bytes()
        )
        (tmp / f"{nm}.txt").write_text(str(tmp / nm / "chers_141710.joblib"))
    # lists with CER_DATA_ROOT-relative entries (as in splits/)
    (tmp / "bg_rel.txt").write_text("bg/chers_141710.joblib")
    (tmp / "fg_rel.txt").write_text("fg/chers_141710.joblib")
    (tmp / "gallery").mkdir()
    (tmp / "gallery" / "bo_141710_bg.npz").write_bytes((tmp / "bg.npz").read_bytes())
    (tmp / "gallery" / "bo_141710_bg.txt").write_text(
        str(tmp / "bg" / "chers_141710.joblib")
    )
    (tmp / "gallery" / "bo_141710_fg.txt").write_text(
        str(tmp / "fg" / "chers_141710.joblib")
    )
    # file-list layout expected by the beam-off scanner
    with open(tmp / "edge.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "shot",
                "n_pairs",
                "vtor_off",
                "vtor_on",
                "vtor_ratio",
                "vtor_fitchange",
                "ti_off",
                "ti_on",
                "ti_ratio",
                "ti_fitchange",
            ],
        )
        w.writeheader()
        for i in range(6):
            w.writerow(
                {
                    "shot": str(138000 + 10 * i),
                    "n_pairs": 2,
                    "vtor_off": 12 + i,
                    "vtor_on": 10,
                    "vtor_ratio": (12 + i) / 10,
                    "vtor_fitchange": 6,
                    "ti_off": 120,
                    "ti_on": 70,
                    "ti_ratio": 1.7,
                    "ti_fitchange": 40,
                }
            )


def main():
    """Command-line entry point."""
    tmp = Path(tempfile.mkdtemp(prefix="cer_smoke_"))
    build(tmp)
    bgf, fgf = str(tmp / "chers_bg.joblib"), str(tmp / "chers_fg.joblib")
    out = str(tmp / "figs")
    common = ["--t-offset", "-0.235"]
    cmds = {
        "figures.event_3d": [
            "--preds",
            f"{tmp}/bg.npz",
            "--t0",
            "0.1",
            "--t1",
            "0.5",
            "--profile-x",
            f"{tmp}/radii.csv",
            "--profile-xlabel",
            "R (m)",
            "--out",
            out,
        ],
        "figures.event_panels": [
            "--preds",
            f"{tmp}/bg.npz",
            "--t0",
            "0.1",
            "--t1",
            "0.5",
            "--out",
            out,
        ],
        "figures.event_waterfall": [
            "--preds",
            f"{tmp}/bg.npz",
            "--t0",
            "0.1",
            "--t1",
            "0.5",
            "--out",
            out,
        ],
        "figures.event_profiles": [
            "--preds",
            f"{tmp}/bg.npz",
            "--t0",
            "0.1",
            "--t1",
            "0.5",
            "--out",
            out,
        ],
        "figures.recon_composite": [
            "--preds",
            f"{tmp}/bg.npz",
            "--preds2",
            f"{tmp}/fg.npz",
            "--shot",
            bgf,
            "--label1",
            "background model",
            "--label2",
            "foreground model",
            "--unl-gap",
            "2",
            "--k",
            "1",
            "1",
            "--profile-x",
            f"{tmp}/radii.csv",
            "--profile-xlabel",
            "R (m)",
            "--out",
            out,
        ],
        "figures.beamoff_traces": [
            "--bg",
            f"{tmp}/bg.npz",
            "--fg",
            f"{tmp}/fg.npz",
            "--bg-file",
            bgf,
            "--fg-file",
            fgf,
            "--t0",
            "0.0",
            "--t1",
            "0.9",
            "--chords",
            "2",
            "--targets",
            "vtor",
            "--hero",
            "--no-fg",
            "--no-strip",
            "--out",
            out,
        ],
        "figures.passive_spectra": [
            "--bg",
            f"{tmp}/bg.npz",
            "--bg-file",
            bgf,
            "--fg-file",
            fgf,
            "--chord",
            "2",
            "--t0",
            "0.0",
            "--t1",
            "0.9",
            "--out",
            out,
        ],
        "figures.shot_compare": [
            "--a",
            f"{tmp}/bg.npz:a",
            "--b",
            f"{tmp}/fg.npz:b",
            "--t0",
            "0.1",
            "--t1",
            "0.5",
            "--zwin",
            "0.2,0.3",
            "--out",
            out,
        ],
        "figures.edge_scatter": [
            f"{tmp}/edge.csv",
            "--log",
            "--label-top",
            "2",
            "--out",
            out,
        ],
        "beamoff.edge_frame_test": [
            "--shot",
            "S",
            "--bg",
            f"{tmp}/bg.npz",
            "--bg-file",
            bgf,
            "--fg-file",
            fgf,
            "--verbose",
            "--csv",
            f"{tmp}/edge_out.csv",
        ],
        "beamoff.pool_edge": [f"{tmp}/edge.csv", "--select", "2010"],
        "beamoff.scan_verified": [
            "--bg-list",
            f"{tmp}/bg.txt",
            "--fg-list",
            f"{tmp}/fg.txt",
            "--out",
            f"{tmp}/verified.csv",
        ],
        "figures.beamoff_grid": [
            "--verified",
            f"{tmp}/verified.csv",
            "--n",
            "1",
            "--gallery",
            f"{tmp}/gallery",
            "--out",
            out,
        ],
        "beamoff.passive_moments": [
            "--shot",
            "S",
            "--bg",
            f"{tmp}/bg.npz",
            "--bg-file",
            bgf,
            "--fg-file",
            fgf,
        ],
        "beamoff.analyze": [
            "--shot",
            "S",
            "--bg",
            f"{tmp}/bg.npz",
            "--fg",
            f"{tmp}/fg.npz",
            "--bg-file",
            bgf,
            "--fg-file",
            fgf,
        ],
        "analysis.coverage": ["--preds", f"{tmp}/bg.npz", "--k", "1", "1"],
        "analysis.affine_fit": [
            "--preds",
            f"{tmp}/bg.npz",
            "--plot",
            f"{tmp}/figs/affine",
        ],
        # relative list entries resolved against CER_DATA_ROOT
        "beamoff.scan_verified@relative": [
            "--bg-list",
            f"{tmp}/bg_rel.txt",
            "--fg-list",
            f"{tmp}/fg_rel.txt",
            "--out",
            f"{tmp}/verified_rel.csv",
        ],
    }
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "MPLBACKEND": "Agg",
        "CER_DATA_ROOT": str(tmp),
    }
    failed = []
    for key, args in cmds.items():
        mod = key.split("@")[0]
        extra = (
            common
            if mod
            in (
                "figures.event_3d",
                "figures.event_panels",
                "figures.event_waterfall",
                "figures.event_profiles",
                "figures.recon_composite",
                "figures.beamoff_traces",
                "figures.passive_spectra",
                "figures.shot_compare",
            )
            else []
        )
        r = subprocess.run(
            [sys.executable, "-m", f"cer_transfer.{mod}", *args, *extra],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )
        status = "ok  " if r.returncode == 0 else "FAIL"
        print(f"{status} cer_transfer.{key}")
        if r.returncode:
            failed.append(key)
            print(r.stderr[-1500:])
    print(f"\n{len(cmds) - len(failed)}/{len(cmds)} modules passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
