"""End-to-end check of training, evaluation and zero-shot transfer on
synthetic data (needs PyTorch; CPU is fine, about a minute).

1. synthetic "DIII-D" (80 chords, 394 bins) and "NSTX" (51 chords, 82 bins)
   discharges with conventional fits on every other frame
2. default model: one epoch on the synthetic DIII-D data (nothing may break)
3. agnostic model: one epoch on DIII-D, then ``eval_checkpoint.py --machine
   nstx`` on the NSTX files (zero-shot path) with a prediction dump
4. ``inference.load_model(..., machine="nstx")`` on one NSTX file

    python -m tests.train_smoke        # from the repository root
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]


def make_shot(path: Path, C: int, W: int, T: int = 120, seed: int = 0):
    """Write one synthetic discharge with a Gaussian line and labels."""
    rng = np.random.default_rng(seed)
    pix = np.arange(W)
    v = (40 + 80 * np.exp(-(((np.arange(C) - 0.15 * C) / (0.25 * C)) ** 2)))[:, None]
    ti = (300 + 900 * np.exp(-(((np.arange(C)) / (0.4 * C)) ** 2)))[:, None]
    cen = 0.45 * W + 0.0008 * W * v
    sig = 0.01 * W * np.sqrt(1 + ti / 400.0)
    spec = 50 + 400 * np.exp(
        -0.5 * ((pix[None, None, :] - cen[..., None]) / sig[..., None]) ** 2
    )
    spec = spec * np.ones((C, T, 1)) + rng.normal(0, 3, (C, T, W))
    tgt = np.full((C, T, 2), np.nan, np.float32)
    for f in range(10, T - 10, 2):
        tgt[:, f, 0] = ti[:, 0] + rng.normal(0, 20, C)
        tgt[:, f, 1] = v[:, 0] + rng.normal(0, 2, C)
    err = np.where(np.isfinite(tgt), np.where(np.arange(2) == 0, 20.0, 2.0), np.nan)
    dump(
        {
            "input": spec.astype(np.float32),
            "target": tgt,
            "end_index": T,
            "target_error": err.astype(np.float32),
        },
        path,
    )


def run(cmd, env):
    print("$", " ".join(str(c) for c in cmd))
    r = subprocess.run(
        [str(c) for c in cmd], cwd=ROOT, env=env, capture_output=True, text=True
    )
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit(f"FAILED: {cmd[1]}")
    return r.stdout


def main():
    tmp = Path(tempfile.mkdtemp(prefix="cer_train_smoke_"))
    (tmp / "training_set_30").mkdir()
    (tmp / "test_set_30").mkdir()
    (tmp / "chers_nstx_labeled").mkdir()
    for i in range(3):
        make_shot(tmp / "training_set_30" / f"d3d_{i}.joblib", 80, 394, seed=i)
    make_shot(tmp / "test_set_30" / "d3d_val.joblib", 80, 394, seed=9)
    for i in range(2):
        make_shot(tmp / "chers_nstx_labeled" / f"chers_{i}.joblib", 51, 82, seed=20 + i)
    (tmp / "nstx.txt").write_text(
        "\n".join(f"chers_nstx_labeled/chers_{i}.joblib" for i in range(2))
    )
    env = {**os.environ, "CER_DATA_ROOT": str(tmp), "PYTHONPATH": str(ROOT)}
    py = sys.executable
    common = [
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--subseq-len",
        "32",
        "--num-workers",
        "0",
        "--hidden-dim",
        "8",
        "--feature-width",
        "32",
        "--head-type",
        "mlp",
        "--norm",
        "group",
        "--no-mmap",
    ]

    print("== default model, one epoch on synthetic DIII-D")
    run(
        [
            py,
            "train.py",
            "--machine",
            "d3d",
            "--checkpoint",
            tmp / "d3d_default.pt",
            *common,
        ],
        env,
    )

    print("== agnostic model, one epoch on synthetic DIII-D")
    run(
        [
            py,
            "train.py",
            "--machine",
            "d3d",
            "--checkpoint",
            tmp / "d3d_agnostic.pt",
            *common,
            "--agnostic",
            "--chord-attention",
            "--resample-w",
            "256",
        ],
        env,
    )

    print("== zero-shot evaluation on synthetic NSTX (--machine nstx)")
    out = run(
        [
            py,
            "eval_checkpoint.py",
            "--checkpoint",
            tmp / "d3d_agnostic.pt",
            "--list",
            tmp / "nstx.txt",
            "--machine",
            "nstx",
            "--dump-preds",
            tmp / "zero_shot.npz",
        ],
        env,
    )
    print(
        "\n".join(
            l for l in out.splitlines() if l.startswith("overall") or "checkpoint:" in l
        )
    )
    d = np.load(tmp / "zero_shot.npz")
    assert int(d["chord"].max()) + 1 == 51, "dump should have 51 NSTX chords"

    print("== --machine on a default checkpoint must be refused")
    r = subprocess.run(
        [
            py,
            "eval_checkpoint.py",
            "--checkpoint",
            str(tmp / "d3d_default.pt"),
            "--list",
            str(tmp / "nstx.txt"),
            "--machine",
            "nstx",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "agnostic" in (
        r.stdout + r.stderr
    ), "override should be refused"

    print("== inference API with machine override")
    out = run(
        [
            py,
            "-c",
            f"""
from cer_transfer.inference import load_model, predict_file
m = load_model(r'{tmp / "d3d_agnostic.pt"}', machine='nstx')
r = predict_file(m, 'chers_nstx_labeled/chers_0.joblib')
print('pred', r.pred.shape, 'finite', bool(__import__('numpy').isfinite(r.pred).all()))
""",
        ],
        env,
    )
    print(out.strip())
    print("\nall training/eval checks passed")


if __name__ == "__main__":
    main()
