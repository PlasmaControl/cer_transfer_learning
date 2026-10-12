"""Compare two machines in the model's input space (the preprocessed,
resampled spectra the agnostic model actually sees) and derive the
wavelength convention that makes them match.

For every labeled chord-frame of each machine the line is measured by its
moments on the linear spectrum (continuum = median of the lowest 20 % of
bins): centroid (pixel), width (pixel, second moment) and amplitude. Then

* ``centroid = p0 + p1 * vtor``  ->  pixels per km/s (p1, its sign is the
  axis orientation) and the pixel of the unshifted line (p0)
* ``width**2 = q0 + q1 * Ti``    ->  pixels**2 per eV (q1) and the
  instrument width (sqrt(q0))

and the distributions of centroid, width and amplitude. With the reference
machine (NSTX) the script prints the convention for the other machine that
would equalize the shift scale and the line position:
``wavelength_flip``, ``wavelength_scale`` and ``wavelength_shift``, and how
far the width scale and the instrument width still differ after that.

    pixi run python -u -m cer_transfer.analysis.domain_check \\
        --ref nstx:splits/nstx_val.txt --other d3d_tangential:splits/d3d_val.txt \\
        --limit 50 --resample-w 256
"""

from __future__ import annotations

import argparse

import numpy as np

from cer_transfer.configs import data_path, get_machine
from cer_transfer.data import ShotDataset


def line_moments(spec_log10, half_window: int = 16):
    """Centroid, width, amplitude per chord-frame of (C, T, W) log10 spectra.

    Continuum: median of the lowest 20 % of bins. Moments are taken within
    +-``half_window`` bins of the peak, so noise and continuum wings far from
    the line do not bias them.
    """
    lin = np.power(10.0, spec_log10.astype(np.float64))
    w = lin.shape[-1]
    base = np.median(np.sort(lin, axis=-1)[..., : max(w // 5, 1)], axis=-1)
    line = np.clip(lin - base[..., None], 0, None)
    pix = np.arange(w)
    peak = line.argmax(-1)
    win = np.abs(pix - peak[..., None]) <= half_window
    lw = np.where(win, line, 0.0)
    tot = lw.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cen = (lw * pix).sum(-1) / tot
        wid = np.sqrt((lw * (pix - cen[..., None]) ** 2).sum(-1) / tot)
    return cen, wid, line.max(-1)


def fit(x, y):
    """Least-squares y = c0 + c1*x in float64; returns (c0, c1, r2)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    xm, ym = x.mean(), y.mean()
    c1 = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    c0 = ym - c1 * xm
    r2 = 1 - ((y - c0 - c1 * x) ** 2).mean() / y.var()
    return float(c0), float(c1), float(r2)


def measure(spec: str, limit: int, resample_w: int):
    """Line-moment statistics of one machine (``name:list``)."""
    name, lst = spec.split(":", 1)
    machine = get_machine(name)
    files = [
        data_path(ln.strip())
        for ln in open(lst).read().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ][:limit]
    ds = ShotDataset(files, machine, subseq_len=-1, mmap=False, resample_w=resample_w)
    cens, wids, amps, vts, tis = [], [], [], [], []
    for i in range(len(ds)):
        x, tgt, _, _ = ds[i]
        cen, wid, amp = line_moments(x.numpy())
        y = tgt.numpy().reshape(cen.shape[0], cen.shape[1], -1)
        ok = np.isfinite(y).all(-1) & np.isfinite(cen) & np.isfinite(wid)
        cens.append(cen[ok]), wids.append(wid[ok]), amps.append(amp[ok])
        vts.append(y[..., 1][ok]), tis.append(y[..., 0][ok])
    cen, wid, amp = map(np.concatenate, (cens, wids, amps))
    vt, ti = np.concatenate(vts), np.concatenate(tis)
    # robust: drop the outer 1 % of moments
    keep = (
        (cen > np.percentile(cen, 1))
        & (cen < np.percentile(cen, 99))
        & (wid > np.percentile(wid, 1))
        & (wid < np.percentile(wid, 99))
    )
    p0, p1, r2v = fit(vt[keep], cen[keep])
    q0, q1, r2t = fit(ti[keep], wid[keep] ** 2)
    return dict(
        name=name,
        n=int(keep.sum()),
        p0=p0,
        p1=p1,
        r2v=r2v,
        q0=q0,
        q1=q1,
        r2t=r2t,
        cen=np.percentile(cen[keep], [5, 50, 95]),
        wid=np.percentile(wid[keep], [5, 50, 95]),
        amp=np.percentile(amp[keep], [5, 50, 95]),
        vt=np.percentile(vt[keep], [5, 50, 95]),
        ti=np.percentile(ti[keep], [5, 50, 95]),
        conv=(
            machine.wavelength_flip,
            machine.wavelength_scale,
            machine.wavelength_shift,
        ),
    )


def report(m):
    """Print one machine's statistics."""
    f, s, sh = m["conv"]
    print(
        f"\n{m['name']}: {m['n']:,} labeled chord-frames "
        f"(stored convention flip={f} scale={s} shift={sh})"
    )
    print(
        f"  centroid = {m['p0']:.1f} px + {m['p1']:.4f} px/(km/s) * vtor   (R2 {m['r2v']:.2f})"
    )
    print(
        f"  width^2  = {m['q0']:.1f} px^2 + {m['q1']:.5f} px^2/eV * Ti   (R2 {m['r2t']:.2f})"
        f"  -> instrument width ~{np.sqrt(max(m['q0'], 0)):.1f} px"
    )
    for k, u in (
        ("cen", "px"),
        ("wid", "px"),
        ("amp", ""),
        ("vt", "km/s"),
        ("ti", "eV"),
    ):
        a = m[k]
        print(f"  {k:4s} 5/50/95 %: {a[0]:9.3g} {a[1]:9.3g} {a[2]:9.3g} {u}")


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="reference machine:list (e.g. NSTX)")
    p.add_argument("--other", required=True, help="other machine:list")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--resample-w", type=int, default=256)
    args = p.parse_args()
    ref = measure(args.ref, args.limit, args.resample_w)
    oth = measure(args.other, args.limit, args.resample_w)
    report(ref), report(oth)

    f, s, sh = oth["conv"]
    ratio = ref["p1"] / oth["p1"]  # stretch still needed (signed)
    flip_new = f != (ratio < 0)
    scale_new = s * abs(ratio)
    # line position after the new stretch (stretch about the window centre)
    c = args.resample_w / 2
    p0_new = c + (oth["p0"] - c) * abs(ratio) * (-1 if ratio < 0 else 1)
    shift_new = sh + (ref["p0"] - p0_new)
    q1_new = oth["q1"] * ratio**2
    print(
        f"\nshift scale: {ref['name']} {ref['p1']:.4f} vs {oth['name']} {oth['p1']:.4f} "
        f"px/(km/s) -> mismatch factor {ratio:.3f}"
        + ("  (opposite sign: orientation differs)" if ratio < 0 else "")
    )
    print(
        f"line position: {ref['name']} {ref['p0']:.1f} px vs {oth['name']} {oth['p0']:.1f} px"
    )
    print(f"\nsuggested convention for {oth['name']} (relative to {ref['name']}):")
    print(
        f"    wavelength_flip={flip_new}, wavelength_scale={scale_new:.3f}, "
        f"wavelength_shift={shift_new:.1f}"
    )
    print(
        f"after that, width scale {ref['name']} {ref['q1']:.5f} vs {oth['name']} "
        f"{q1_new:.5f} px^2/eV (ratio {q1_new / ref['q1']:.2f}); instrument width "
        f"{np.sqrt(max(ref['q0'], 0)):.1f} vs {np.sqrt(max(oth['q0'], 0)) * abs(ratio):.1f} px"
    )
    print(
        "  (a width-scale ratio far from 1 or a large instrument-width difference "
        "means T_i cannot transfer through the shift convention alone)"
    )


if __name__ == "__main__":
    main()
