"""Rebuild DIII-D spectrograms on ONE fixed wavelength grid, frame-aligned to the targets.

The existing joblib files were made per shot on a master grid taken from one
chord group, so their bin count (390, 536, 538, ...) and window vary between
shots while the pixel pitch is the same. Training resampled every file to a
fixed number of bins, i.e. to different nm per bin -- a label-inconsistent
stretch of up to 1.37 between files. This tool rebuilds the ``input`` array of
every file from the raw ``<shot>_cer_spectra.h5`` on the same grid for all
shots and leaves everything the loader reads about the targets untouched:

* grid: 532.00 -> 526.00 nm, 536 bins (0.011215 nm/bin, descending), the
  convention ``d3d_tangential_meta`` already assumes; stored as ``wavelength_nm``
* channels: t01..t48 (``--all-chords``: plus v01..v32, as in the old files, for
  the 80-channel 'd3d' config); stored as uint16 counts (``--dtype float32``
  to keep floats; 4x smaller together, the tangential model reads nothing
  else); each chord is
  interpolated linearly from its own ``feature_axis`` onto the grid, bins
  outside the chord's window hold the edge value (np.interp, as the notebook did)
* frames: the joblib grid is 5 ms steps from the first to the last time of
  the t09-t16 spectrometer (T == steps + 1 on every shot checked; that start
  is 0 ms for most shots but 250, 300 or 2000 ms for some). Joblib frame i is
  each chord's spectrum linearly interpolated in time at t0 + 5 i ms on that
  chord's own ``time_axis``, held at the first/last sample outside its record
  (np.interp). Cadences differ per spectrometer within a shot (0.5-10 ms
  seen); where the grid coincides with a sample this is the sample itself
  (verified on 189998: raw frame 2 i, every other mapping >100x worse;
  166530: odd frames of 2 ms chords are interpolated). The lag check below
  guards every file against a time shift.
* chords missing in the raw file (1-element datasets) keep the OLD channel,
  resampled onto the new grid by wavelength when ``--fit-old-axis`` is given
  (slow fit of the old grid; needed only then), else filled with the old
  channel's median (such channels carry no spectrum)
* ``target``, ``target_error``, ``end_index`` are copied byte for byte

Alignment check (always, before writing): per chord, the per-frame brightness
of the old and the new spectrogram over the labeled range must correlate best
at lag 0 (a non-zero lag counts only if it beats lag 0 by > 0.05 in r);
chords without temporal structure or with r < 0.5 at every lag (dark
spectrometer, noise) carry no timing information and are skipped. A file
fails when > 10 % of the informative chords are misaligned or >= 3 chords
share one non-zero lag (a shifted spectrometer); the lag histogram is logged. Optionally ``--fit-old-axis`` adds a spectral residual at lag 0 (RMS
difference old vs new on frames with a clear line, relative to the line
peak; expect ~0.001-0.02) -- a diagnostic of the old-grid fit, logged, not a
gate. A file that fails the lag check is reported and NOT written
(``--force`` writes it anyway, flagged).

``--delete-raw`` removes a shot's raw H5 right after its new file has been
written, re-read and verified (never on failure, never with --force, never in
a dry run): the run then frees ~1.6 GB per shot as it goes. Irreversible --
the grid, frame mapping and channel set are final for that shot afterwards.

Files are written uncompressed (the loader uses mmap) under ``--out-root`` in
a folder of the same name as the input folder, via a temporary name that is
re-read and compared before being renamed. Re-runs skip existing outputs.

    pixi run --frozen python -u -m cer_transfer.datasets.reprocess_d3d \\
        --dirs training_set_30 test_set_30 --raw-dir /scratch/gpfs/NSTX-U/passive_micer/raw \\
        --out-root /scratch/gpfs/EKOLEMEN/ps9551/data_root_fixed \\
        [--shard 0 --n-shards 20] [--fit-old-axis --limit 20] [--dry-run]

Then: point CER_DATA_ROOT at data_root_fixed (symlink the NSTX folders into
it) and train with ``--machine d3d_tangential_fixed`` (48 raw channels).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path

import numpy as np
from joblib import dump, load

from cer_transfer.configs import data_path

GRID_FIRST, GRID_LAST, GRID_BINS = 532.0, 526.0, 536
GRID = np.linspace(GRID_FIRST, GRID_LAST, GRID_BINS)
T_CHANNELS = [f"t{i:02d}" for i in range(1, 49)]
V_CHANNELS = [f"v{i:02d}" for i in range(1, 33)]
CHANNELS = T_CHANNELS + V_CHANNELS  # order of the OLD files; the output keeps a prefix of it
REF_CHORDS = [f"t{i:02d}" for i in range(9, 17)]  # spectrometer whose time axis set the notebook grid
FRAME_MS = 5.0  # joblib cadence
LOADER_KEYS = ("input", "target", "target_error", "end_index")


def interp_weights(axis_sorted, grid):
    """Indices and fractions for linear interpolation of rows sampled on
    ``axis_sorted`` (ascending) onto ``grid``; outside -> edge value."""
    j = np.searchsorted(axis_sorted, grid, side="right") - 1
    j = np.clip(j, 0, len(axis_sorted) - 2)
    x0, x1 = axis_sorted[j], axis_sorted[j + 1]
    frac = (grid - x0) / np.where(x1 > x0, x1 - x0, 1.0)
    frac = np.clip(frac, 0.0, 1.0)  # edge padding
    return j, frac.astype(np.float64)


def resample_rows(rows, j, frac):
    """rows (T, Wc) on the sorted axis -> (T, len(grid))."""
    return rows[:, j] * (1.0 - frac) + rows[:, j + 1] * frac


def brightness_lags(old_b, new_b, max_lag=6):
    """Correlation of old vs new per-frame brightness at each lag.

    Returns dict lag -> r, where old[i + lag] is compared with new[i]; the
    aligned case has its maximum at lag 0.
    """
    n = len(old_b)
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        a = old_b[max(0, lag) : n + min(0, lag)]
        b = new_b[max(0, -lag) : n - max(0, lag)]
        if len(a) < 10 or a.std() == 0 or b.std() == 0:
            continue
        out[lag] = float(np.corrcoef(a, b)[0, 1])
    return out


def brightness_lag(old_b, new_b, max_lag=3):
    """(lag, r) of the best lag; kept for diagnostics."""
    rs = brightness_lags(old_b, new_b, max_lag)
    if not rs:
        return 0, -np.inf
    lag = max(rs, key=rs.get)
    return lag, rs[lag]


def process_file(entry, raw_dir, out_root, args):
    """Reprocess one file; returns a dict for the log."""
    src = data_path(entry)
    shots = re.findall(r"\d{6}", src.stem)
    rec = {"file": entry, "shot": shots[-1] if shots else "", "status": "", "note": ""}
    if not shots:
        rec["status"] = "skip"; rec["note"] = "no shot number in name"; return rec
    out_dir = out_root / src.parent.name
    out = out_dir / src.name
    if out.exists():
        rec["status"] = "exists"; return rec
    raw = raw_dir / f"{shots[-1]}_cer_spectra.h5"
    if not raw.exists():
        rec["status"] = "fail"; rec["note"] = "raw file missing"; return rec

    import h5py

    old = load(src, mmap_mode="r")
    x_old = old["input"]
    C_old, T, W_old = x_old.shape
    end = int(old["end_index"])
    if C_old != len(CHANNELS):
        rec["status"] = "fail"; rec["note"] = f"{C_old} channels in joblib"; return rec
    C = C_old if args.all_chords else len(T_CHANNELS)  # output channels (prefix of CHANNELS)

    new = np.empty((C, T, GRID_BINS), dtype=np.float32)
    missing, filled_median, cadences = [], [], {}
    with h5py.File(str(raw), "r") as h5:
        ch_group = h5["chords"]
        # the notebook's time grid: 5 ms steps from the first to the last time
        # of the reference spectrometer (t09-t16), NOT from 0 ms; shots whose
        # reference record starts late (250, 300, 2000 ms seen) have their
        # joblib frame 0 there. Verified by T == steps + 1 on every shot checked.
        t_ref = None
        for name in REF_CHORDS:
            if name in ch_group and ch_group[name]["time_axis"].shape[0] > 2:
                t_ref = np.asarray(ch_group[name]["time_axis"][()], np.float64)
                rec["ref_chord"] = name
                break
        if t_ref is None:
            rec["status"] = "fail"; rec["note"] = "no reference chord (t09-t16) with a time axis"; return rec
        t0 = float(t_ref[0])
        n_expect = int(round((float(t_ref[-1]) - t0) / FRAME_MS)) + 1
        if abs(n_expect - T) > 1:
            rec["status"] = "fail"
            rec["note"] = (f"joblib T={T} but reference {rec['ref_chord']} spans {t0:.1f}..{t_ref[-1]:.1f} ms "
                           f"= {n_expect} frames at 5 ms: grid rule does not apply")
            return rec
        rec["t0_ms"] = f"{t0:.1f}"
        want = t0 + FRAME_MS * np.arange(T)  # joblib frame i at t0 + 5 i ms
        for c, name in enumerate(CHANNELS[:C]):
            ok = False
            if name in ch_group:
                g = ch_group[name]
                ax = np.asarray(g["feature_axis"][()], np.float64)
                tax = np.asarray(g["time_axis"][()], np.float64)
                if (ax.size >= 8 and not np.all(np.mod(ax, 1) == 0) and tax.size >= 2
                        and g["data"].shape[0] == tax.size and np.all(np.diff(tax) > 0)):
                    d = np.asarray(g["data"][()], np.float64)
                    if np.all(np.diff(ax) < 0):
                        ax, d = ax[::-1], d[:, ::-1]
                    elif not np.all(np.diff(ax) > 0):
                        o = np.argsort(ax); ax, d = ax[o], d[:, o]
                    if np.diff(ax).min() > 0:
                        dt = float(np.median(np.diff(tax)))
                        cadences[round(dt, 3)] = cadences.get(round(dt, 3), 0) + 1
                        # time: linear interpolation of this chord's frames at 5 i ms,
                        # held at the first/last sample outside the record (np.interp
                        # semantics, as the notebook did; exact sample where 5 i ms
                        # coincides with one, i.e. for 2.5 and 5 ms cadences)
                        k = np.clip(np.searchsorted(tax, want) - 1, 0, tax.size - 2)
                        wgt = np.clip((want - tax[k]) / (tax[k + 1] - tax[k]), 0.0, 1.0)
                        rows = d[k] * (1.0 - wgt)[:, None] + d[k + 1] * wgt[:, None]
                        # wavelength: linear interpolation onto the fixed grid, edge-padded
                        j, frac = interp_weights(ax, GRID)
                        new[c] = resample_rows(rows, j, frac).astype(np.float32)
                        ok = True
            if not ok:
                missing.append(name)
                oc = np.asarray(x_old[c], np.float64)
                if args.fit_old_axis and np.nanstd(oc) > 1e-6 * (np.nanmean(np.abs(oc)) + 1e-9):
                    rec.setdefault("_need_axis", True)
                    new[c] = np.nan  # filled below once the old axis is known
                else:
                    new[c] = np.float32(np.nanmedian(oc))
                    filled_median.append(name)
    if not cadences:
        rec["status"] = "fail"; rec["note"] = "no usable raw chord"; return rec
    rec["frame_map"] = (f"interp@{t0:.0f}+5i ms; raw dt "
                        + ",".join(f"{k}x{v}" for k, v in sorted(cadences.items())))

    # old grid (only if asked for or needed): the notebook's grid, fitted
    old_axis = None
    if args.fit_old_axis or rec.get("_need_axis"):
        from cer_transfer.datasets.wavelength_axes import estimate_axis

        fit = estimate_axis(src, raw)
        if fit is None:
            rec["status"] = "fail"; rec["note"] = "old axis fit failed"; return rec
        first, last, corr = fit
        old_axis = np.linspace(first, last, W_old)
        rec["old_axis"] = f"{first:.3f}->{last:.3f} r={corr:.3f}"
        if corr < 0.9:
            rec["status"] = "fail"; rec["note"] = "old axis fit corr < 0.9"; return rec
        if rec.get("_need_axis"):
            oa = old_axis[::-1] if old_axis[0] > old_axis[-1] else old_axis
            j, frac = interp_weights(oa, GRID)
            for c, name in enumerate(CHANNELS[:C]):
                if name in missing and name not in filled_median:
                    oc = np.asarray(x_old[c], np.float64)
                    oc = oc[:, ::-1] if old_axis[0] > old_axis[-1] else oc
                    new[c] = resample_rows(oc, j, frac).astype(np.float32)
    rec.pop("_need_axis", None)
    rec["missing"] = ",".join(missing)

    # ---- alignment check: brightness lag per chord over the labeled range.
    # The magnitude of r is not the criterion (old and new grids cover
    # different parts of a chord's window, so the brightness sums differ in
    # background content); the LAG is. Chords whose best r is below 0.5
    # carry no timing information (dark spectrometer, noise) and are ignored.
    informative, misaligned, lag_hist = 0, 0, {}
    for c, name in enumerate(CHANNELS[:C]):
        if name in missing:
            continue
        ob = np.asarray(x_old[c, :end], np.float64).sum(-1)
        nb = new[c, :end].astype(np.float64).sum(-1)
        if ob.std() < 0.02 * (abs(ob.mean()) + 1e-9):  # no temporal structure
            continue
        rs = brightness_lags(ob, nb)
        if not rs:
            continue
        best = max(rs, key=rs.get)
        if rs[best] < 0.5:
            continue
        informative += 1
        r0 = rs.get(0, -np.inf)
        if best != 0 and rs[best] - r0 > 0.05:
            misaligned += 1
            lag_hist[best] = lag_hist.get(best, 0) + 1
    rec["chords_checked"] = informative
    rec["misaligned_chords"] = misaligned
    rec["lag_histogram"] = ";".join(f"{k:+d}x{v}" for k, v in sorted(lag_hist.items()))  # lag histogram
    same_lag = max(lag_hist.values()) if lag_hist else 0
    align_ok = informative > 0 and misaligned <= max(1, int(0.1 * informative)) and same_lag < 3
    if informative == 0:
        rec["note"] += " no chord with timing information (dark shot?);"
    # ---- optional spectral check at lag 0 on the overlap
    if old_axis is not None:
        oa = old_axis[::-1] if old_axis[0] > old_axis[-1] else old_axis
        j, frac = interp_weights(oa, GRID)
        inside = (GRID >= oa[0]) & (GRID <= oa[-1])
        errs = []
        for c, name in enumerate(CHANNELS[: min(C, 48)]):
            if name in missing:
                continue
            fr = np.arange(min(end, 1500) // 10, min(end, 1500), max(1, min(end, 1500) // 12))
            oc = np.asarray(x_old[c][fr], np.float64)
            oc = oc[:, ::-1] if old_axis[0] > old_axis[-1] else oc
            o_on_new = resample_rows(oc, j, frac)[:, inside]
            n_on_new = new[c][fr][:, inside].astype(np.float64)
            # only frames with a real line: peak above background > 10 x noise
            bg = np.percentile(o_on_new, 10, axis=-1, keepdims=True)
            line = o_on_new - bg
            noise = 1.4826 * np.median(np.abs(line - np.median(line, -1, keepdims=True)), -1) + 1e-9
            strong = line.max(-1) > 10 * noise
            if strong.sum() < 3:
                continue
            o, nn = o_on_new[strong], n_on_new[strong]
            errs.append(np.median(np.sqrt(((o - nn) ** 2).mean(-1)) / (line[strong].max(-1) + 1e-9)))
        rec["spectral_nrmse_median"] = f"{np.median(errs):.4f}" if errs else ""
        if errs and np.median(errs) > args.max_nrmse:
            rec["note"] += f" spectral residual {np.median(errs):.3f} > {args.max_nrmse} (diagnostic);"

    if not align_ok and not args.force:
        rec["status"] = "fail"
        rec["note"] += " alignment check failed"
        return rec
    rec["status"] = "ok" if align_ok else "forced"
    if args.dry_run:
        rec["status"] += "(dry)"; return rec
    if args.delete_raw and args.force and not align_ok:
        rec["status"] = "fail"; rec["note"] += " not written: --delete-raw refuses --force"; return rec

    out_dir.mkdir(parents=True, exist_ok=True)
    stored = new
    if args.dtype == "uint16":
        mx = float(np.nanmax(new))
        if np.isfinite(mx) and mx <= 65535.0:
            # camera counts: integers plus interpolation fractions far below
            # the photon noise -> lossless for the model (loader casts to float32)
            stored = np.rint(np.clip(new, 0.0, 65535.0)).astype(np.uint16)
        else:
            rec["note"] += f" max {mx:.0f} > 65535: kept float32;"
    rec["dtype"] = str(stored.dtype)
    payload = {
        "input": stored,
        "target": np.asarray(old["target"]),
        "target_error": np.asarray(old["target_error"]),
        "end_index": old["end_index"],
        "wavelength_nm": GRID.astype(np.float64),
        "channels": np.array(CHANNELS[:C]),
        "source": f"reprocess_d3d: raw {raw.name}, frame i = raw frame at {FRAME_MS} i ms, "
                  f"grid {GRID_FIRST}->{GRID_LAST} nm/{GRID_BINS}; missing chords: {rec['missing'] or 'none'}",
    }
    for k in ("target", "target_error", "end_index"):
        if k not in old:
            payload.pop(k)
    tmp = out.with_name(out.name + ".tmp_reproc")
    dump(payload, tmp)  # uncompressed: the loader mmaps
    chk = load(tmp, mmap_mode="r")
    good = (
        chk["input"].shape == stored.shape
        and np.array_equal(np.asarray(chk["target"]), np.asarray(old["target"]), equal_nan=True)
        and np.array_equal(np.asarray(chk["target_error"]), np.asarray(old["target_error"]), equal_nan=True)
        and int(chk["end_index"]) == end
        and np.array_equal(np.asarray(chk["input"][0, :5]), stored[0, :5])
    )
    del chk
    if not good:
        os.remove(tmp)
        rec["status"] = "fail"; rec["note"] += " re-read mismatch"; return rec
    os.replace(tmp, out)
    rec["out"] = str(out)
    if args.delete_raw:
        # only here: the new file is written, re-read and verified
        try:
            chk = load(out, mmap_mode="r")
            if chk["input"].shape == stored.shape and int(chk["end_index"]) == end:
                del chk
                os.remove(raw)
                rec["raw_deleted"] = "yes"
            else:
                rec["raw_deleted"] = "no: re-read failed"
        except Exception as exc:
            rec["raw_deleted"] = f"no: {exc!r}"
    return rec


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dirs", nargs="*", default=[], help="folders under CER_DATA_ROOT")
    ap.add_argument("--list", nargs="*", default=[], help="split lists (CER_DATA_ROOT-relative entries)")
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True, help="new data root (never the old one)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-corr", type=float, default=0.99, help="(unused; kept for compatibility)")
    ap.add_argument("--max-nrmse", type=float, default=0.05,
                    help="spectral residual (relative to line peak) flagged in the log, with --fit-old-axis; not a gate")
    ap.add_argument("--fit-old-axis", action="store_true", help="also check spectra (slow: fits the old grid)")
    ap.add_argument("--dtype", choices=("uint16", "float32"), default="uint16",
                    help="storage type of 'input' (uint16: rounded counts, half the size)")
    ap.add_argument("--all-chords", action="store_true",
                    help="keep v01-v32 as well (default: t01-t48 only, 48 channels)")
    ap.add_argument("--delete-raw", action="store_true",
                    help="remove the shot's raw H5 after its new file is written and verified "
                    "(frees ~1.7 GB per shot; irreversible, never on fail/exists/dry-run)")
    ap.add_argument("--force", action="store_true", help="write files that fail the check (flagged)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", type=Path, default=None, help="CSV log (default <out-root>/reprocess_<shard>.csv)")
    args = ap.parse_args()

    entries = []
    for d in args.dirs:
        p = data_path(d)
        entries += [f"{d}/{f}" for f in sorted(os.listdir(p)) if f.endswith(".joblib")]
    for lst in args.list:
        entries += [ln.strip() for ln in Path(lst).read_text().splitlines()
                    if ln.strip() and not ln.strip().startswith("#")]
    entries = sorted(set(entries))[args.shard :: args.n_shards]
    if args.limit:
        entries = entries[: args.limit]
    if str(args.out_root.resolve()) == str(Path(os.environ.get("CER_DATA_ROOT", "")).resolve()):
        raise SystemExit("--out-root must differ from CER_DATA_ROOT")
    log = args.log or args.out_root / f"reprocess_{args.shard}.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "shot", "status", "dtype", "ref_chord", "t0_ms", "frame_map", "missing", "chords_checked",
              "misaligned_chords", "lag_histogram", "spectral_nrmse_median", "old_axis", "note", "out",
              "raw_deleted"]
    counts = {}
    t0 = time.time()
    with open(log, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if fh.tell() == 0:
            w.writeheader()
        for i, e in enumerate(entries):
            try:
                rec = process_file(e, args.raw_dir, args.out_root, args)
            except Exception as exc:  # keep the shard running
                rec = {"file": e, "status": "error", "note": repr(exc)}
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            w.writerow(rec); fh.flush()
            if rec["status"] not in ("ok", "exists") or (i + 1) % 50 == 0:
                print(f"[{i + 1}/{len(entries)}] {rec['status']:8s} {e} "
                      f"{rec.get('misaligned_chords', '')}/{rec.get('chords_checked', '')} {rec.get('lag_histogram', '')} "
                      f"{rec.get('spectral_nrmse_median', '')} {rec.get('note', '')}")
    print(f"done in {time.time() - t0:.0f} s: {counts} -> {log}")


if __name__ == "__main__":
    main()
