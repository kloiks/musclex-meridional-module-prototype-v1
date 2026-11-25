"""Naming tracks after the expected peaks (config/expected_peaks.csv); used by multiscale.py and unet.py."""
import csv
import os

import numpy as np


def read_expected_peaks(path):
    """[(name, d_nm, tol_pct or None)] from a name,d_nm,tol_pct CSV (# lines are comments)."""
    with open(path) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return [(r["name"].strip(), float(r["d_nm"]),
             float(r["tol_pct"]) if (r.get("tol_pct") or "").strip() else None) for r in rows]


def label_windows(peaks, tol_pct):
    """[(name, d0, half-window in nm)] sorted by d0, capped at half the gap to the neighbours."""
    peaks = sorted(peaks, key=lambda x: x[1])
    windows = []
    for i, (name, d0, tol) in enumerate(peaks):
        tol = d0 * (tol if tol is not None else tol_pct) / 100  # % of d0 -> nm
        gaps = [abs(d0 - peaks[j][1]) for j in (i - 1, i + 1) if 0 <= j < len(peaks)]
        windows.append((name, d0, min([tol] + [g / 2 for g in gaps])))
    return windows


def label_tracks(rows, inv_nm, peaks, tol_pct, min_frames, share_names=False):
    """{track: (name, d_nm, d - d0)}, closest pair first.  Each name is given once or, with
    ``share_names`` (U-Net), also to further tracks that have no frame in common with its others."""
    r_of, frames_of = {}, {}
    for row in rows:
        r_of.setdefault(row["track"], []).append(row["r"])
        frames_of.setdefault(row["track"], set()).add(row["frame"])
    d_of = {a: 1.0 / (np.median(rr) * inv_nm) for a, rr in r_of.items() if len(rr) >= min_frames}
    windows = label_windows(peaks, tol_pct)
    pairs = sorted((abs(d - d0), a, name, d, d - d0) for a, d in d_of.items()
                   for name, d0, tol in windows if abs(d - d0) <= tol)
    out, used = {}, {}  # used: name -> frames already carrying it
    for _, a, name, d, dd in pairs:
        free = name not in used or (share_names and not used[name] & frames_of[a])
        if a not in out and free:
            out[a] = (name, d, dd)
            used.setdefault(name, set()).update(frames_of[a])
    return out


def unmatched_names(r, labels):
    """{track: "X1", "X2", ...} for the tracks without a label, numbered from the centre out
    (by median r); ``r`` is (track x frame)."""
    rest = sorted((a for a in range(len(r)) if a not in labels), key=lambda a: np.nanmedian(r[a]))
    return {a: f"X{i + 1}" for i, a in enumerate(rest)}


def write_labels(out_dir, labels):
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["anchor", "label", "d_nm", "dd_nm"])
        for a, (name, d, dd) in sorted(labels.items(), key=lambda kv: kv[1][1]):
            w.writerow([a, name, f"{d:.4f}", f"{dd:+.4f}"])
