r"""Test whether the edges of the wavelength window drive the predictions.

The trained model is run on each discharge in five variants and every variant is
compared with the unmodified run:

1. ``baseline``: unchanged input.
2. ``mask_low``: the lowest ``--bins`` bins of real data (the wing of the line
   at 528.05 nm) replaced by the window median, per chord and frame.
3. ``mask_high``: the ``--bins`` bins of real data below the upper data edge
   replaced the same way.
4. ``mask_both``: 2 and 3 together.
5. ``floor_256``: the input floor of the preprocessing taken per 256-frame
   window, as in training, instead of over the whole discharge (the model then
   also sees 256-frame windows, as in training).

The masking acts on the preprocessed 256-bin input. For the rebuilt DIII-D files
the real data end ``--high-pad`` bins below the top of the window (the last bins
are edge padding of the wavelength shift); the high mask sits below them. For
NSTX all 256 bins are data (``--high-pad 0``).

Per variant and group (all shots, edge chords, core chords, and the shot groups
given by ``--groups``) the script prints R2 and median error per target against
the conventional fits, and the median and 95th percentile of the change of the
prediction relative to the baseline.

    pixi run --frozen python -u -m cer_transfer.analysis.edge_ablation \
        --checkpoint cer_ckpts/d3d_fixed_plain_best.pt \
        --shots 185761 189950 189979 189422 189958 \
        --groups 185761:normal 189950:low 189979:low 189422:ohmic 189958:ohmic

``--shots`` are looked up in ``--dirs`` (default: the machine's training and
validation folders) and the split of each shot is printed. ``--list`` adds the
entries of split lists, ``--limit`` caps their number.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cer_transfer.configs import data_path
from cer_transfer.data import ShotDataset
from cer_transfer.inference import load_model
from cer_transfer.training import denormalize_mu_sigma

VARIANTS = ("baseline", "mask_low", "mask_high", "mask_both", "floor_256")


def find_shots(shots, dirs):
    """Return {shot: list entry} for the shots found in ``dirs``."""
    found = {}
    for d in dirs:
        root = data_path(d)
        if not root.exists():
            continue
        for s in shots:
            p = root / f"{s}.joblib"
            if p.exists() and s not in found:
                found[s] = f"{d}/{p.name}"
    return found


def mask_bins(x, lo, hi):
    """Replace bins [lo, hi) of x (..., W) by the median over the window."""
    x = x.clone()
    med = x.median(dim=-1, keepdim=True).values
    x[..., lo:hi] = med.expand_as(x[..., lo:hi])
    return x


@torch.no_grad()
def run(m, spec, moments):
    """Forward pass; return mu and sigma of shape (C, T, n_targets)."""
    mu_n, sg_n = m.model(
        spec.unsqueeze(0).to(m.device), moments.unsqueeze(0).to(m.device)
    )
    mu, sg = denormalize_mu_sigma(mu_n, sg_n, m.stats, m.machine.targets)
    C, n_t = m.machine.n_chords, len(m.machine.targets)
    return (
        mu[0].cpu().numpy().reshape(C, -1, n_t),
        sg[0].cpu().numpy().reshape(C, -1, n_t),
    )


def predict_variants(m, entry, n_bins, high_pad):
    """Predictions of all variants for one discharge, labels and sigmas."""
    ds = ShotDataset(
        [data_path(entry)],
        m.machine,
        subseq_len=-1,
        mmap=False,
        resample_w=m.config.resample_w,
        causal=m.config.causal,
    )
    spec, tgt, err, mom = ds[0]
    W = spec.shape[-1]
    hi_end = W - high_pad
    out = {"baseline": run(m, spec, mom)[0]}
    out["mask_low"] = run(m, mask_bins(spec, 0, n_bins), mom)[0]
    out["mask_high"] = run(m, mask_bins(spec, hi_end - n_bins, hi_end), mom)[0]
    both = mask_bins(mask_bins(spec, 0, n_bins), hi_end - n_bins, hi_end)
    out["mask_both"] = run(m, both, mom)[0]
    # training-like: floor and context per 256-frame window
    ds256 = ShotDataset(
        [data_path(entry)],
        m.machine,
        subseq_len=256,
        mmap=False,
        resample_w=m.config.resample_w,
        causal=m.config.causal,
    )
    T = out["baseline"].shape[1]
    w = np.full_like(out["baseline"], np.nan)
    for i in range(len(ds256)):
        _, start = ds256.subseq_info[i]
        s_i, _, _, m_i = ds256[i]
        mu_i = run(m, s_i, m_i)[0]
        w[:, start : start + mu_i.shape[1]] = mu_i[:, : max(0, T - start)]
    out["floor_256"] = w
    y = tgt.numpy().reshape(out["baseline"].shape)
    return out, y


def stats(pred, y, base):
    """Return R2 and median error vs labels, and the change vs baseline.

    The change is given as median and 95th percentile of |pred - base|.
    """
    d = np.abs(pred - base)
    d = d[np.isfinite(d)]
    dm, d95 = (np.median(d), np.percentile(d, 95)) if d.size else (np.nan, np.nan)
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() < 10:  # e.g. ohmic: no labels, but the change is still meaningful
        return np.nan, np.nan, dm, d95
    e = pred[ok] - y[ok]
    r2 = 1.0 - (e**2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
    return r2, np.median(np.abs(e)), dm, d95


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--machine", default=None, help="apply an agnostic model to this machine"
    )
    ap.add_argument("--shots", nargs="*", default=[], help="shot numbers")
    ap.add_argument(
        "--dirs", nargs="*", default=None, help="folders to search (default: machine)"
    )
    ap.add_argument("--list", nargs="*", default=[], help="split lists")
    ap.add_argument("--limit", type=int, default=20, help="entries taken from --list")
    ap.add_argument("--groups", nargs="*", default=[], help="shot:group labels")
    ap.add_argument("--bins", type=int, default=15, help="bins masked per edge")
    ap.add_argument(
        "--high-pad",
        type=int,
        default=None,
        help="padded bins at the top (default: |wavelength_shift| of the machine)",
    )
    ap.add_argument(
        "--edge-chords",
        type=int,
        nargs="*",
        default=None,
        help="chord indices counted as edge (default: last third)",
    )
    args = ap.parse_args()

    m = load_model(
        args.checkpoint,
        machine=args.machine,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    mach = m.machine
    high_pad = args.high_pad
    if high_pad is None:
        high_pad = (
            abs(int(round(mach.wavelength_shift))) if mach.wavelength_shift < 0 else 0
        )
    dirs = args.dirs or [str(p) for p in (mach.train_dir, mach.val_dir) if p]

    entries = {}
    if args.shots:
        found = find_shots(args.shots, dirs)
        for s in args.shots:
            print(
                f"shot {s}: "
                + (f"{found[s]}" if s in found else f"not found in {dirs}")
            )
        entries.update(found)
    for lst in args.list:
        lines = [
            ln.strip()
            for ln in Path(lst).read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        for e in lines[: args.limit]:
            entries[Path(e).stem] = e
    if not entries:
        raise SystemExit("no discharges found")
    group_of = dict(g.split(":", 1) for g in args.groups)
    print(
        f"model {Path(args.checkpoint).name} on {mach.name}; {len(entries)} "
        "discharges; "
        f"{args.bins} bins masked per edge, high edge below {high_pad} padded bins"
    )

    acc = defaultdict(
        lambda: defaultdict(lambda: ([], [], []))
    )  # group -> variant -> (pred, y, base)
    names = list(mach.targets)
    for shot, entry in entries.items():
        out, y = predict_variants(m, entry, args.bins, high_pad)
        C = y.shape[0]
        edge = (
            set(args.edge_chords)
            if args.edge_chords is not None
            else set(range(2 * C // 3, C))
        )
        sel = {
            "all": np.arange(C),
            "edge chords": np.array(sorted(edge)),
            "core chords": np.array([c for c in range(C) if c not in edge]),
        }
        if shot in group_of:
            sel[f"group {group_of[shot]}"] = np.arange(C)
        # held-out vs seen in training: the folder of the file
        sel[f"split {Path(entry).parent.name}"] = np.arange(C)
        for g, idx in sel.items():
            for v in VARIANTS:
                p, yy, b = acc[g][v]
                p.append(out[v][idx])
                yy.append(y[idx])
                b.append(out["baseline"][idx])
        print(f"  done {shot}")

    for g in acc:
        print(f"\n== {g}")
        for k, tn in enumerate(names):
            u = "eV" if tn == "ti" else "km/s"
            print(
                f"{tn}: {'variant':10s} {'R2':>7s} {'med|err|':>9s} "
                f"{'med|change|':>12s} {'p95|change|':>12s}  ({u})"
            )
            for v in VARIANTS:
                p = np.concatenate([a[..., k].ravel() for a in acc[g][v][0]])
                yy = np.concatenate([a[..., k].ravel() for a in acc[g][v][1]])
                b = np.concatenate([a[..., k].ravel() for a in acc[g][v][2]])
                r = stats(p, yy, b)
                print(f"     {v:10s} {r[0]:7.3f} {r[1]:9.3g} {r[2]:12.3g} {r[3]:12.3g}")


if __name__ == "__main__":
    main()
