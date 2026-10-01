"""Minimal inference API: load a checkpoint, run it on one discharge file.

    from cer_transfer.inference import load_model, predict_file
    model = load_model("cer_ckpts/nstx_ft.pt")
    r = predict_file(model, "chers_137711.joblib")
    r.pred[chord, frame, k]        # k = 0: T_i (eV), 1: v_tor (km/s)

``export_inference_checkpoint`` strips the optimizer/scheduler state for
sharing (weights, config, normalization statistics only).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cer_transfer.configs import ModelConfig, data_path, get_machine
from cer_transfer.data import ShotDataset
from cer_transfer.models import build_model
from cer_transfer.training import denormalize_mu_sigma, load_checkpoint


def load_model(checkpoint, device: str = "cpu") -> SimpleNamespace:
    """Model, machine config and normalization statistics of a checkpoint."""
    ckpt = load_checkpoint(Path(checkpoint))
    mk = ckpt["machine"]
    machine = get_machine(mk["name"] if isinstance(mk, dict) else mk)
    cfg = ModelConfig(**ckpt["model_config"])
    model = build_model(machine, cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    stats = {k: v.to(device) for k, v in ckpt["norm_stats"].items()}
    return SimpleNamespace(model=model, machine=machine, config=cfg, stats=stats,
                           device=device, best_score=ckpt.get("best_score"),
                           epoch=ckpt.get("epoch"))


@torch.no_grad()
def predict_file(m: SimpleNamespace, path) -> SimpleNamespace:
    """Run the model on one discharge file (CER_DATA_ROOT-relative or absolute).

    Returns arrays of shape (chords, frames, targets): ``pred`` and
    ``pred_sigma`` from the model, ``y`` and ``sigma`` the conventional fits
    and their uncertainties (NaN where none exists), plus ``targets``.
    """
    ds = ShotDataset([data_path(path)], m.machine, subseq_len=-1, mmap=False)
    spec, tgt, err, moments = ds[0]
    mu_n, sigma_n = m.model(spec.unsqueeze(0).to(m.device),
                            moments.unsqueeze(0).to(m.device))
    mu, psig = denormalize_mu_sigma(mu_n, sigma_n, m.stats, m.machine.targets)
    C, n_t = m.machine.n_chords, len(m.machine.targets)
    return SimpleNamespace(
        pred=mu[0].cpu().numpy().reshape(C, -1, n_t),
        pred_sigma=psig[0].cpu().numpy().reshape(C, -1, n_t),
        y=tgt.numpy().reshape(C, -1, n_t), sigma=err.numpy().reshape(C, -1, n_t),
        targets=list(m.machine.targets))


def export_inference_checkpoint(src, dst) -> Path:
    """Copy a checkpoint without optimizer/scheduler state (for sharing)."""
    ckpt = load_checkpoint(Path(src))
    keep = {k: v for k, v in ckpt.items()
            if k not in ("optimizer_state", "scheduler_state")}
    torch.save(keep, dst)
    return Path(dst)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="export an inference-only checkpoint")
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    a = p.parse_args()
    out = export_inference_checkpoint(a.src, a.dst)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
