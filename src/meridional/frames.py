"""Frames left out of peak detection (e.g. shutter closed): a frame list like "0-1, 40-60".

Frames are numbered from 0, as in tracks.csv.  By default the list is found from the
kymograph (blank_frames); a list typed in the GUI is kept per recording in
<output>/excluded_frames.json.
"""
import json
import os
import re

import numpy as np
from scipy.ndimage import uniform_filter1d


def blank_frames(totals, fraction, pad):
    """Boolean mask of beam-off frames: column total < ``fraction`` of the median exposed
    total, padded by ``pad`` frames either side (partly exposed frames)."""
    totals = np.asarray(totals, dtype=float)
    ref = np.median(totals)
    exposed = totals > fraction * totals.max()
    if not ref > fraction * totals.max() and exposed.sum() >= 3:  # most of the record is blank
        ref = np.median(totals[exposed])
    dark = totals < fraction * ref
    if pad > 0 and dark.any():
        dark = uniform_filter1d(dark.astype(float), 2 * pad + 1, mode="constant") > 1e-9
    return dark


def parse_frames(text):
    """Sorted frame numbers of "0-1, 40-60"; ValueError if the text is not such a list."""
    out = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:\s*[-–]\s*(\d+))?", part)
        if not m:
            raise ValueError(f"cannot read {part!r}; use frames and ranges like 0-1, 40-60")
        lo, hi = int(m[1]), int(m[2] or m[1])
        if hi < lo or hi > 10 ** 6:
            raise ValueError(f"bad range {part!r}")
        out.update(range(lo, hi + 1))
    return sorted(out)


def format_frames(frames):
    """The inverse of parse_frames."""
    runs = []
    for f in sorted(int(f) for f in frames):
        if runs and f == runs[-1][1] + 1:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def excluded_path(out_dir):
    return os.path.join(out_dir, "excluded_frames.json")


def load_excluded(out_dir):
    """The frames typed in for this recording, or None (automatic)."""
    try:
        with open(excluded_path(out_dir)) as f:
            return parse_frames(json.load(f)["exclude"])
    except FileNotFoundError:
        return None


def save_excluded(out_dir, frames):
    """Keep ``frames`` for this recording; None goes back to automatic."""
    path = excluded_path(out_dir)
    if frames is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"exclude": format_frames(frames)}, f)
    elif os.path.exists(path):
        os.remove(path)


def excluded_mask(kymo, p, exclude=None):
    """Boolean mask of the columns left out: ``exclude``, or the blank frames if it is None."""
    n = kymo.shape[1]
    if exclude is None:
        drop = blank_frames(kymo.sum(axis=0), p["dark_fraction"], p["drop_pad"])
    else:
        drop = np.zeros(n, dtype=bool)
        drop[[f for f in exclude if f < n]] = True
    if drop.all():
        raise ValueError("every frame is excluded; check 'Frames to exclude' in Parameters")
    return drop
