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
  the level above (and, with ref = coarse, of the anchor's r(t) at the
  coarsest level) whose SNR clears a threshold relaxed linearly from
  snr_coarse to snr_fine; it joins that anchor.

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
import argparse
import csv
import json
import os

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_widths

import cache
from background import fitted_backgrounds, profile_noise, shared_background
from frames import excluded_mask, format_frames, parse_frames
from labels import label_tracks, read_expected_peaks, write_labels

# the multiscale_v3_c4_sc3 settings
DEFAULTS = dict(
    rmin=40, rmax=1400, background="grouped", bg_method="hull", bg_comp2="lorentzian", bg_neg_weight=10.0,
    smooth=1, snr_floor=1.0, noise_win=51, noise_pct=25.0, min_sep=3,
    snr_coarse=3.0, snr_fine=1.1, corridor=4.0, base=2, min_cov=0.5,
    select="nearest", ref="coarse", seed_anchors="yes",
    dark_fraction=0.05, drop_pad=1,
    tol_pct=1.0, min_frames=5,
)
# the keys that change the tracking (labelling is redone on every run)
TRACK_KEYS = [k for k in DEFAULTS if k not in ("tol_pct", "min_frames")]


# ------------------------------------------------------------------ detection

def parabolic_offset(y, idx):
    out = np.zeros(len(idx))
    for n, i in enumerate(idx):
        if 0 < i < len(y) - 1:
            den = y[i - 1] - 2.0 * y[i] + y[i + 1]
            if den != 0:
                out[n] = np.clip(0.5 * (y[i - 1] - y[i + 1]) / den, -1.0, 1.0)
    return out


def detect(profile, r0, p, background=None):
    """Peaks of one profile: dict of r, height, fwhm, snr, snr_prom arrays."""
    flat, sigma = profile_noise(profile, p, background)
    thr = p["snr_floor"] * sigma
    pk, props = find_peaks(flat, height=thr, prominence=thr, distance=max(1, p["min_sep"]))
    fwhm = np.maximum(peak_widths(flat, pk, rel_height=0.5)[0], 1.0) if len(pk) else np.zeros(0)
    return dict(r=r0 + pk + parabolic_offset(flat, pk), height=flat[pk], fwhm=fwhm,
                snr=flat[pk] / sigma[pk], snr_prom=props["prominences"] / sigma[pk])


def detect_level(ks, pos, p, background=None):
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


def corridor_mask(r, t, n_pos, n, r0, corridor):
    m = np.zeros((n_pos, n), dtype=bool)
    m[np.clip(np.round(r - r0).astype(int), 0, n_pos - 1), t] = True
    return uniform_filter1d(m.astype(float), 2 * int(corridor) + 1, axis=0, mode="constant") > 0


def select_peaks(lev, rule, ref):
    """One peak per (anchor, frame): highest SNR, or nearest the anchor's reference r(t)."""
    a, t = lev["anchor"], lev["t"]
    key = -np.abs(lev["r"] - ref[a, t]) if rule == "nearest" else lev["snr"]
    best = {}
    for i in np.argsort(key, kind="stable"):  # ascending: the last write wins
        best[(a[i], t[i])] = i
    sel = np.zeros(len(a), dtype=bool)
    sel[list(best.values())] = True
    return sel


def ref_matrix(lev, anchor_r, n):
    """(anchor x frame) r(t) of the selected peaks, gaps interpolated."""
    out = np.repeat(anchor_r[:, None], n, axis=1)
    for a in range(len(anchor_r)):
        m = lev["selected"] & (lev["anchor"] == a)
        if m.any():
            o = np.argsort(lev["t"][m])
            out[a] = np.interp(np.arange(n), lev["t"][m][o], lev["r"][m][o])
    return out


def run_corridor(kymo, pos, scales, p, seed_r=()):
    """({scale: accepted peaks}, anchor radii, number of seeded anchors).

    ``seed_r`` (px, needs ref = coarse) adds anchors at expected positions the coarsest
    scale did not find within ``corridor``; they come after the detected ones and keep
    a corridor open at every scale, so a peak too weak at the coarse scales can still
    be picked up at the fine ones."""
    n_pos, n = kymo.shape
    levels, parent, fixed = {}, None, None
    background = shared_background(kymo, pos, p)  # sliding means keep the scale, so one fits all levels
    for k, s in enumerate(scales):
        det = detect_level(bin_time(kymo, s), pos, p, background)
        snr = np.minimum(det["snr"], det["snr_prom"])
        if parent is None:  # every column of the top level is the same: anchors from column 0
            accept = snr >= p["snr_coarse"]
            det_r = np.sort(det["r"][accept & (det["t"] == 0)])
            seeds = [r for r in seed_r if not len(det_r) or np.abs(det_r - r).min() > p["corridor"]] \
                if p["ref"] == "coarse" else []
            anchor_r, n_det = np.concatenate([det_r, seeds]), len(det_r)
            if not len(anchor_r):
                raise ValueError("no anchors at the coarsest scale; lower snr_coarse")
            anchor = np.argmin(np.abs(anchor_r[:, None] - det["r"][None, :]), axis=0)
            ref = np.repeat(anchor_r[:, None], n, axis=1)
        else:
            thr = p["snr_coarse"] + (p["snr_fine"] - p["snr_coarse"]) * k / max(len(scales) - 1, 1)
            pm = corridor_mask(parent["r"], parent["t"], n_pos, n, pos[0], p["corridor"])
            if fixed is not None and len(anchor_r) > n_det:
                seeded = fixed[n_det:]
                pm |= corridor_mask(seeded.ravel(), np.tile(np.arange(n), len(seeded)),
                                    n_pos, n, pos[0], p["corridor"])
            inside = pm[np.clip(np.round(det["r"] - pos[0]).astype(int), 0, n_pos - 1), det["t"]]
            if fixed is not None:  # also within corridor of the anchor's coarse r(t), which it joins
                dist = np.abs(fixed[:, det["t"]] - det["r"][None, :])
                anchor = np.argmin(dist, axis=0)
                inside &= dist[anchor, np.arange(len(anchor))] <= p["corridor"]
                ref = fixed
            else:  # joins the anchor of the nearest parent peak in the same frame
                anchor = np.full(len(snr), -1)
                for i in np.flatnonzero(inside):
                    cand = np.flatnonzero(parent["t"] == det["t"][i])
                    anchor[i] = parent["anchor"][cand[np.argmin(np.abs(parent["r"][cand] - det["r"][i]))]]
                ref = ref_matrix(parent, anchor_r, n) if p["select"] == "nearest" else None
            accept = inside & (snr >= thr)
        lev = {key: v[accept] for key, v in det.items()}
        lev["anchor"] = anchor[accept]
        lev["selected"] = select_peaks(lev, p["select"], ref)
        if p["ref"] == "coarse" and k == 0:
            fixed = ref_matrix(lev, anchor_r, n)
        levels[s] = lev
        parent = lev
    return levels, anchor_r, len(anchor_r) - n_det


def anchor_series(lev, n_anchor, n):
    """{key: (anchor x frame)} of the selected peak, NaN where none."""
    out = {k: np.full((n_anchor, n), np.nan) for k in ("r", "height", "fwhm", "snr")}
    sel = lev["selected"]
    for k in out:
        out[k][lev["anchor"][sel], lev["t"][sel]] = lev[k][sel]
    return out


def resolution_scales(levels, scales, n_anchor, n, min_cov):
    """Finest scale per anchor with coverage >= min_cov (0 if never)."""
    res = np.zeros(n_anchor, dtype=int)
    for s in scales:  # coarse -> fine: the last that passes wins
        lev = levels[s]
        for a in range(n_anchor):
            if len(np.unique(lev["t"][lev["anchor"] == a])) >= min_cov * n:
                res[a] = s
    return res


def build_tracks(series, scales, res, n, frames):
    """Rows (anchor, frame): each anchor read at its resolution scale, gaps interpolated.

    An anchor only resolved at the whole record has a flat r(t): one row at its middle frame."""
    rows = []
    for a, s in enumerate(res):
        if s == 0:
            continue
        ser = {k: v[a] for k, v in series[s].items()}
        obs = np.flatnonzero(np.isfinite(ser["r"]))
        if not len(obs):
            continue
        span = obs[len(obs) // 2:len(obs) // 2 + 1] if s >= n else np.arange(obs[0], obs[-1] + 1)
        fill = np.interp(span, obs, ser["r"][obs])
        for i, t in enumerate(span):
            seen = bool(np.isfinite(ser["r"][t]))
            rows.append(dict(track=a, frame=int(frames[t]), r=float(ser["r"][t]) if seen else float(fill[i]),
                             height=ser["height"][t], fwhm=ser["fwhm"][t], snr=ser["snr"][t],
                             scale=int(s), gap=not seen))
    return rows


# ------------------------------------------------------------------ run

def _fmt(v):
    return f"{v:.5g}" if isinstance(v, float) else v


def process(npy, out_dir, params=None, peaks_csv=None, exclude=None):
    """Track (or reuse the cached tracking in out_dir) and label; returns (summary, cached).
    ``exclude``: frames (from 0) to leave out; None = the blank frames."""
    p = dict(DEFAULTS, **(params or {}))
    kymo = np.load(npy)
    with open(os.path.splitext(npy)[0] + ".json") as f:
        meta = json.load(f)
    inv_nm = meta.get("inv_nm_per_pixel")
    pos_all = np.arange(kymo.shape[0]) + meta.get("pos_start", 0)
    os.makedirs(out_dir, exist_ok=True)
    run_json = os.path.join(out_dir, "run.json")
    peaks = read_expected_peaks(peaks_csv) if peaks_csv else []
    seed_d = sorted(d for _, d, _ in peaks
                    if inv_nm and p["seed_anchors"] == "yes" and p["rmin"] <= 1.0 / (d * inv_nm) <= p["rmax"])
    key = dict({k: p[k] for k in TRACK_KEYS}, kymo=os.path.abspath(npy), kymo_sha256=cache.array_hash(kymo),
               n_frames_source=int(kymo.shape[1]), inv_nm_per_pixel=inv_nm, seed_d=seed_d,
               exclude=exclude)

    run = cache.read_json(run_json)
    cached = cache.matches(run, key) and all(os.path.exists(os.path.join(out_dir, name))
                                             for name in ("multiscale.npz", "tracks.csv"))
    if cached:
        with open(os.path.join(out_dir, "tracks.csv")) as f:
            rows = [dict(r, track=int(r["track"]), r=float(r["r"])) for r in csv.DictReader(f)]
    else:
        keep_r = (pos_all >= p["rmin"]) & (pos_all <= p["rmax"])
        drop = excluded_mask(kymo, p, exclude)
        frames = np.flatnonzero(~drop)
        k, pos = kymo[keep_r][:, frames], pos_all[keep_r]
        n = k.shape[1]
        scales = scale_ladder(n, p["base"])
        levels, anchor_r, n_seeded = run_corridor(k, pos, scales, p, [1.0 / (d * inv_nm) for d in seed_d])
        series = {s: anchor_series(levels[s], len(anchor_r), n) for s in scales}
        res = resolution_scales(levels, scales, len(anchor_r), n, p["min_cov"])
        rows = build_tracks(series, scales, res, n, frames)
        np.savez_compressed(os.path.join(out_dir, "multiscale.npz"), scales=scales, frames=frames,
                            anchor_r=anchor_r, res=res,
                            r=np.stack([series[s]["r"] for s in scales]),
                            height=np.stack([series[s]["height"] for s in scales]))
        with open(os.path.join(out_dir, "tracks.csv"), "w", newline="") as f:
            w = csv.writer(f)
            fields = ["track", "frame", "r", "d_nm", "height", "fwhm", "snr", "scale", "gap"]
            w.writerow(fields)
            for r in rows:
                d = 1.0 / (r["r"] * inv_nm) if inv_nm and r["r"] > 0 else np.nan
                w.writerow([_fmt(v) for v in (r["track"], r["frame"], r["r"], d, r["height"],
                                              r["fwhm"], r["snr"], r["scale"], r["gap"])])
        run = dict(key, scales=scales, dropped_frames=np.flatnonzero(drop).tolist(),
                   n_anchors=len(anchor_r), n_seeded=n_seeded, anchor_r=anchor_r.tolist(), resolution_scale=res.tolist())
        with open(run_json, "w") as f:
            json.dump(run, f, indent=2)

    labels = {}
    if inv_nm and peaks:
        labels = label_tracks(rows, inv_nm, peaks, p["tol_pct"], p["min_frames"])
    write_labels(out_dir, labels)

    dropped = run["dropped_frames"]
    summary = (f"multiscale: {run['n_anchors']} anchors ({run.get('n_seeded', 0)} seeded), scales {run['scales']}"
               + (f", excluded frames {format_frames(dropped)}" if dropped else "")
               + f"; labelled {len(labels)}: " + ", ".join(sorted(v[0] for v in labels.values())))
    return summary, cached


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kymo", help="per-frame kymograph .npy (meridian_w<W>_g1.npy)")
    ap.add_argument("--out-dir", help="default <kymo dir>/multiscale_w<W>")
    ap.add_argument("--params", help="parameters.json (missing keys use the defaults)")
    ap.add_argument("--peaks", help="expected_peaks.csv for the labels")
    ap.add_argument("--exclude", help='frames to exclude, from 0, e.g. "0-1, 40-60" (default: blank frames)')
    args = ap.parse_args()
    params = {}
    if args.params:
        with open(args.params) as f:
            params = json.load(f)
    width = os.path.basename(args.kymo).split("_")[1]
    out = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.kymo)), f"multiscale_{width}")
    exclude = None if args.exclude is None else parse_frames(args.exclude)
    print(process(args.kymo, out, params, args.peaks, exclude)[0])


if __name__ == "__main__":
    main()
