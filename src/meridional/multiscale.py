#!/usr/bin/env python3
"""Coarse-to-fine (multiscale) peak tracking on a per-frame meridional kymograph.

K_s(r, t) is the sliding mean of s frames for s = 1, 2, 4, ... and the whole
record.  Peaks are detected in every column of every level (baseline removed,
local noise from second differences).  The baseline is the convex hull
(bg_method = hull) or a fitted smooth background (bg_method = fit: MuscleX's
general background, baseline + exponential + bg_comp2 = lorentzian, powerlaw
or stretched exponential, with oversubtraction weighted bg_neg_weight times
harder), of each column (background = grouped) or once of the record's mean
profile and subtracted from every column (background = averaged):

* the coarsest level keeps peaks with SNR >= snr_coarse: the *anchors*;
* each finer level keeps a peak within ``corridor`` px of an accepted peak of
  the level above and of the nearest anchor, which it joins, if its SNR clears
  a threshold relaxed linearly from snr_coarse to snr_fine.  Of several peaks
  that join an anchor in one frame the nearest to the anchor is its peak.

Per anchor, the finest scale at which it is found in >= min_cov of the frames
is its *resolution scale*; its track is read off that scale.  Excluded frames (given, or
the beam-off frames: column sum < dark_fraction of the median, padded by drop_pad)
are removed before the pyramid is built; frame numbers written out are the input's.

Anchors are then named after the expected peaks (config/expected_peaks.csv:
name, d_nm, tol_pct), one-to-one and closest first; the window is tol_pct % of
d_nm (blank = ``tol_pct``, 1 % by default), capped at half the gap to the
neighbouring peak.

Writes to ``out_dir``: run.json, multiscale.npz (per-scale r/height of every
anchor), tracks.csv, labels.csv.

Example:
    python multiscale.py qf_results/kymo/meridian_w20_g1.npy --peaks config/expected_peaks.csv
"""
import csv
import os

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_widths

from . import tracking
from .background import fitted_backgrounds, profile_noise, shared_background
from .frames import format_frames
from .labels import label_tracks, read_expected_peaks, write_labels

# the multiscale_v3_c4_sc3 settings
DEFAULTS = dict(
    rmin=40, rmax=1400, background="grouped", bg_method="hull", bg_comp2="lorentzian", bg_neg_weight=10.0,
    smooth=1, snr_floor=1.0, noise_win=51, noise_pct=25.0, min_sep=3,
    snr_coarse=3.0, snr_fine=1.1, corridor=4.0, base=2, min_cov=0.5, seed_anchors="yes",
    dark_fraction=0.05, drop_pad=1,
    tol_pct=1.0, min_frames=5,
)
# the keys that change the tracking (labelling is redone on every run)
TRACK_KEYS = [k for k in DEFAULTS if k not in ("tol_pct", "min_frames")]


# ------------------------------------------------------------------ detection

def parabolic_offset(y, i):
    """Sub-pixel offset of each (interior) maximum i of y."""
    bend = y[i - 1] - 2.0 * y[i] + y[i + 1]
    offset = np.divide(0.5 * (y[i - 1] - y[i + 1]), bend, out=np.zeros(len(i)), where=bend != 0)
    return np.clip(offset, -1.0, 1.0)


def detect(profile, r0, p, background=None):
    """Peaks of one profile: dict of r, height, fwhm, snr (of the height) and snr_min (of the height
    or the prominence, whichever is lower) arrays."""
    flat, sigma = profile_noise(profile, p, background)
    thr = p["snr_floor"] * sigma
    pk, props = find_peaks(flat, height=thr, prominence=thr, distance=max(1, p["min_sep"]))
    fwhm = np.maximum(peak_widths(flat, pk, rel_height=0.5)[0], 1.0) if len(pk) else np.zeros(0)
    return dict(r=r0 + pk + parabolic_offset(flat, pk), height=flat[pk], fwhm=fwhm,
                snr=flat[pk] / sigma[pk], snr_min=np.minimum(flat[pk], props["prominences"]) / sigma[pk])


def detect_level(ks, pos, p, background=None):
    """Peaks of every column, as one dict of arrays with the column number in ``t``."""
    fits = fitted_backgrounds(ks, pos, p) if background is None and p["bg_method"] == "fit" else None
    dets = []
    for t in range(ks.shape[1]):
        d = detect(ks[:, t], pos[0], p, background if fits is None else fits[:, t])
        d["t"] = np.full(len(d["r"]), t)
        dets.append(d)
    return {k: np.concatenate([d[k] for d in dets]) for k in dets[0]}


# ------------------------------------------------------------------ pyramid / corridor

def scale_ladder(n, base=2):
    """Coarse to fine: n, ..., base^2, base, 1."""
    if base < 2:  # the scales would never reach n
        raise ValueError(f"the scale ratio (base) is {base}; it must be at least 2. Change it in Parameters")
    s, out = 1, []
    while s < n:
        out.append(s)
        s *= base
    return (out + [n])[::-1]


def accept_snr(p, k, n_scales):
    """SNR a peak needs at level k, relaxed linearly from coarse (0) to fine (n_scales - 1)."""
    return p["snr_coarse"] + (p["snr_fine"] - p["snr_coarse"]) * k / max(n_scales - 1, 1)


def bin_time(kymo, s):
    """Sliding mean of s frames, window kept inside the record."""
    n = kymo.shape[1]
    if s >= n:
        return np.repeat(kymo.mean(axis=1, keepdims=True), n, axis=1)
    if s <= 1:
        return kymo.copy()
    c = np.concatenate([np.zeros((kymo.shape[0], 1)), np.cumsum(kymo, axis=1)], axis=1)
    start = np.clip(np.arange(n) - s // 2, 0, n - s)
    return (c[:, start + s] - c[:, start]) / s


def row_of(r, r0, n_pos):
    return np.clip(np.round(r - r0).astype(int), 0, n_pos - 1)


def corridor_mask(r, t, shape, r0, corridor):
    """(position x frame) mask of the pixels within ``corridor`` of a point (r, t)."""
    m = np.zeros(shape, dtype=bool)
    m[row_of(r, r0, shape[0]), t] = True
    return uniform_filter1d(m.astype(float), 2 * int(corridor) + 1, axis=0, mode="constant") > 0


def accepted_level(det, accept, anchor, anchor_r):
    """The accepted peaks with their anchor; ``selected`` marks one per (anchor, frame), the
    nearest to the anchor."""
    lev = {key: v[accept] for key, v in det.items()}
    lev["anchor"] = a = anchor[accept]
    best = {}
    for i in np.argsort(-np.abs(lev["r"] - anchor_r[a]), kind="stable"):  # nearest last: it wins
        best[(a[i], lev["t"][i])] = i
    lev["selected"] = np.zeros(len(a), dtype=bool)
    lev["selected"][list(best.values())] = True
    return lev


def nearest_anchor(anchor_r, r):
    """(anchor, distance) of the anchor nearest to each r."""
    dist = np.abs(anchor_r[:, None] - r[None, :])
    anchor = np.argmin(dist, axis=0)
    return anchor, dist[anchor, np.arange(len(r))]


def find_anchors(kymo, pos, p, background, seed_r):
    """(coarsest level, anchor radii, number of seeded anchors): the peaks of the mean profile, which
    is every column of this level, then the seeds further than ``corridor`` from all of them."""
    n = kymo.shape[1]
    one = detect_level(bin_time(kymo, n)[:, :1], pos, p, background)
    det = {key: np.tile(v, n) for key, v in one.items()}
    det["t"] = np.repeat(np.arange(n), len(one["t"]))
    accept = det["snr_min"] >= p["snr_coarse"]
    det_r = np.sort(det["r"][accept & (det["t"] == 0)])
    seeds = [r for r in seed_r if not len(det_r) or np.abs(det_r - r).min() > p["corridor"]]
    anchor_r = np.concatenate([det_r, seeds])
    if not len(anchor_r):
        raise ValueError("no anchors at the coarsest scale; lower snr_coarse")
    anchor, _ = nearest_anchor(anchor_r, det["r"])
    return accepted_level(det, accept, anchor, anchor_r), anchor_r, len(seeds)


def follow_level(det, parent, anchor_r, n_seeded, thr, shape, r0, corridor):
    """One finer level: the peaks inside the parent's corridor (or a seeded anchor's, open at every
    level), within ``corridor`` of an anchor and with SNR >= thr."""
    n = shape[1]
    mask = corridor_mask(parent["r"], parent["t"], shape, r0, corridor)
    if n_seeded:
        seeded = anchor_r[-n_seeded:]
        mask |= corridor_mask(np.repeat(seeded, n), np.tile(np.arange(n), n_seeded), shape, r0, corridor)
    anchor, dist = nearest_anchor(anchor_r, det["r"])
    inside = mask[row_of(det["r"], r0, shape[0]), det["t"]] & (dist <= corridor)
    return accepted_level(det, inside & (det["snr_min"] >= thr), anchor, anchor_r)


def run_corridor(kymo, pos, scales, p, seed_r=()):
    """({scale: accepted peaks}, anchor radii, number of seeded anchors).

    ``seed_r`` (px) adds anchors at expected positions the coarsest scale did not find, so a peak
    too weak at the coarse scales can still be picked up at the fine ones; they come last."""
    background = shared_background(kymo, pos, p)  # sliding means keep the scale, so one fits all levels
    lev, anchor_r, n_seeded = find_anchors(kymo, pos, p, background, seed_r)
    levels = {scales[0]: lev}
    for k, s in enumerate(scales[1:], 1):
        det = detect_level(bin_time(kymo, s), pos, p, background)
        lev = follow_level(det, lev, anchor_r, n_seeded, accept_snr(p, k, len(scales)),
                           kymo.shape, pos[0], p["corridor"])
        levels[s] = lev
    return levels, anchor_r, n_seeded


def anchor_series(lev, n_anchor, n):
    """{key: (anchor x frame)} of the selected peak, NaN where none."""
    out = {k: np.full((n_anchor, n), np.nan) for k in ("r", "height", "fwhm", "snr")}
    sel = lev["selected"]
    for k in out:
        out[k][lev["anchor"][sel], lev["t"][sel]] = lev[k][sel]
    return out


def resolution_scales(series, scales, min_cov):
    """Finest scale per anchor with coverage >= min_cov (0 if never)."""
    res = np.zeros(len(series[scales[0]]["r"]), dtype=int)
    for s in scales:  # coarse -> fine: the last that passes wins
        r = series[s]["r"]
        res[np.isfinite(r).sum(axis=1) >= min_cov * r.shape[1]] = s
    return res


def build_tracks(series, res, frames):
    """Rows (anchor, frame): each anchor read at its resolution scale, gaps interpolated.

    An anchor only resolved at the whole record has a flat r(t): one row at its middle frame."""
    rows = []
    for a, s in enumerate(res):
        if s == 0:
            continue
        r, height, fwhm, snr = (series[s][k][a] for k in ("r", "height", "fwhm", "snr"))
        obs = np.flatnonzero(np.isfinite(r))
        if not len(obs):
            continue
        whole = s >= len(frames)
        span = [obs[len(obs) // 2]] if whole else np.arange(obs[0], obs[-1] + 1)
        for t, fill in zip(span, np.interp(span, obs, r[obs])):
            seen = bool(np.isfinite(r[t]))
            rows.append(dict(track=a, frame=int(frames[t]), r=float(r[t] if seen else fill),
                             height=height[t], fwhm=fwhm[t], snr=snr[t], scale=int(s), gap=not seen))
    return rows


# ------------------------------------------------------------------ run

def track(inp, p, seed_d):
    """(arrays of multiscale.npz, rows of tracks.csv, facts for run.json)."""
    keep = (inp.pos >= p["rmin"]) & (inp.pos <= p["rmax"])
    kymo, pos = inp.kymo[keep][:, inp.frames], inp.pos[keep]
    scales = scale_ladder(kymo.shape[1], p["base"])
    levels, anchor_r, n_seeded = run_corridor(kymo, pos, scales, p, [1.0 / (d * inp.inv_nm) for d in seed_d])
    series = {s: anchor_series(levels[s], len(anchor_r), kymo.shape[1]) for s in scales}
    res = resolution_scales(series, scales, p["min_cov"])
    arrays = dict(scales=scales, frames=inp.frames, r=np.stack([series[s]["r"] for s in scales]),
                  height=np.stack([series[s]["height"] for s in scales]))
    facts = dict(scales=scales, n_anchors=len(anchor_r), n_seeded=n_seeded, anchor_r=anchor_r.tolist(),
                 resolution_scale=res.tolist())
    return arrays, build_tracks(series, res, inp.frames), facts


def write_tracks(path, rows, inp):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "frame", "r", "d_nm", "height", "fwhm", "snr", "scale", "gap"])
        for row in rows:
            numbers = [row["r"], inp.d_nm(row["r"]), row["height"], row["fwhm"], row["snr"]]
            w.writerow([row["track"], row["frame"]] + [f"{v:.5g}" for v in numbers] + [row["scale"], row["gap"]])


def read_tracks(path):
    """The columns of tracks.csv that the labelling needs."""
    with open(path) as f:
        return [dict(track=int(r["track"]), frame=int(r["frame"]), r=float(r["r"])) for r in csv.DictReader(f)]


def process(npy, out_dir, params=None, peaks_csv=None, exclude=None):
    """Track (or reuse the saved tracking in out_dir) and label; returns (summary, cached).
    ``exclude``: frames (from 0) to leave out; None = the blank frames."""
    p = dict(DEFAULTS, **(params or {}))
    inp = tracking.load_input(npy, p, exclude)
    peaks = read_expected_peaks(peaks_csv) if peaks_csv else []
    seed_d = sorted(d for _, d, _ in peaks if inp.inv_nm and p["seed_anchors"] == "yes"
                    and p["rmin"] <= 1.0 / (d * inp.inv_nm) <= p["rmax"])
    key = dict({k: p[k] for k in TRACK_KEYS}, **inp.key(), seed_d=seed_d)

    os.makedirs(out_dir, exist_ok=True)
    tracks_csv = os.path.join(out_dir, "tracks.csv")
    run = tracking.saved_run(out_dir, key, ("multiscale.npz", "tracks.csv"))
    cached = run is not None
    if cached:
        rows = read_tracks(tracks_csv)
    else:
        arrays, rows, facts = track(inp, p, seed_d)
        np.savez_compressed(os.path.join(out_dir, "multiscale.npz"), **arrays)
        write_tracks(tracks_csv, rows, inp)
        run = dict(key, **facts)
        tracking.save_run(out_dir, run)

    labels = label_tracks(rows, inp.inv_nm, peaks, p["tol_pct"], p["min_frames"])
    write_labels(out_dir, labels)
    summary = (f"multiscale: {run['n_anchors']} anchors ({run['n_seeded']} seeded), scales {run['scales']}"
               + (f", excluded frames {format_frames(inp.dropped)}" if inp.dropped else "")
               + f"; labelled {len(labels)}: " + ", ".join(sorted(v[0] for v in labels.values())))
    return summary, cached


def main():
    tracking.run_cli(process, "multiscale", __doc__)


if __name__ == "__main__":
    main()
