"""Check the metadata-derived DIII-D wavelength convention against NSTX.

A spectral line of a given wavelength must land on the same pixel, with the
same width, as on NSTX (needs PyTorch; CPU, seconds).

    python -m tests.convention_check
"""

from __future__ import annotations

import numpy as np

from cer_transfer.configs import get_machine
from cer_transfer.data import Preprocessor

LAM_NSTX = np.linspace(528.20, 530.28, 83)
LAM_D3D = np.linspace(532.0, 526.0, 536)


def _line(lam, l0, sig=0.06):
    return (100 + 1000 * np.exp(-0.5 * ((lam - l0) / sig) ** 2)).astype(np.float32)[
        None, None, :
    ]


def _peak(x):
    v = 10 ** x[0, 0].numpy().astype(float)
    v -= np.median(v)
    k = int(v.argmax())
    y0, y1, y2 = v[k - 1], v[k], v[k + 1]
    return k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)


def _fwhm(x):
    v = 10 ** x[0, 0].numpy().astype(float)
    v -= np.median(v)
    return int((v >= v.max() / 2).sum())


def main():
    """Assert pixel and width agreement for lines across the NSTX window."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--machine",
        default="d3d_tangential_meta",
        help="DIII-D convention to test on the 532 -> 526 nm / 536-bin grid",
    )
    args = ap.parse_args()
    d3d = get_machine(args.machine)
    print(
        f"testing {d3d.name}: flip {d3d.wavelength_flip}, stretch "
        f"{d3d.wavelength_scale}, "
        f"shift {d3d.wavelength_shift}, rest pixel {d3d.rest_pixel}"
    )
    pre_n = Preprocessor(get_machine("nstx"), 256)
    pre_d = Preprocessor(d3d, 256)
    worst = 0.0
    for l0 in np.linspace(528.45, 529.95, 7):
        pn, pd = _peak(pre_n(_line(LAM_NSTX, l0))), _peak(pre_d(_line(LAM_D3D, l0)))
        worst = max(worst, abs(pd - pn))
        print(f"{l0:.2f} nm: NSTX pixel {pn:6.2f}, DIII-D pixel {pd:6.2f}")
    wn, wd = _fwhm(pre_n(_line(LAM_NSTX, 529.05))), _fwhm(pre_d(_line(LAM_D3D, 529.05)))
    rest = _peak(pre_n(_line(LAM_NSTX, 529.05)))
    print(
        f"worst pixel offset {worst:.2f} | FWHM NSTX {wn} px, DIII-D {wd} px | C VI "
        f"at {rest:.1f}"
    )
    assert worst < 1.0, "line positions differ by more than one pixel"
    assert abs(wn - wd) <= 1, "line widths differ"
    assert abs(rest - d3d.rest_pixel) < 1.0
    assert abs(rest - get_machine("nstx").rest_pixel) < 1.0
    print("convention check passed")


if __name__ == "__main__":
    main()
