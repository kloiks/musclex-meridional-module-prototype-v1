"""Intensity normalisation of the kymographs: within a sequence (incident beam I0 from the beamline
log) and between sequences (one scale factor per sequence).

Within a sequence
    BioCAT writes ``<run>/<run>.log`` beside each recording: one row per frame (frame k = kymograph
    column k) with the I0 (incident) and I1 (transmitted) ion-chamber counts.  With "Normalize by I0",
    column k is multiplied by mean(I0) / I0[k] (mean over the kept frames), so the overall level stays
    in counts.  Flicker = frame-to-frame jitter of the column totals: RMS of the differences of
    log(total) between consecutive kept frames, / sqrt(2), in %.  Slow changes (contraction, stretch)
    hardly enter it.

Between sequences (as in meridional-simulation results/intensity_norm/passive_vs_active/
scale_methods.md; the scale factors have geometric mean 1)
    "ls"      least squares on log10 of the mean profiles of the 20 px meridional (r 50-599 px) and
              equatorial (r 50-329 px) strips: minimise sum_proj mean_pairs sum_r w(r)
              (x_i + a_i - x_j - a_j)^2, w = p / sum p within each projection, p(r) = 1 rising to 2
              at M3, M6 (sigma 3 px) and I1,0, I1,1 (sigma 6 px); s_i = 10^a_i.
    "eq_bg"   background (hull or fit, as in Parameters/Background) of the mean equatorial profile,
              summed over r 90-600 px; s_i = geometric mean(N) / N_i.
    "mer_bg"  the same on the meridional profile.
The mean profiles are over the frames used (default 2-20, the frames before the stimulus; empty = all
kept frames; dropped frames are always left out), after
the I0 normalisation if it is on.  The equatorial strip is read from the quadrant-folded images once
and saved next to the kymograph (equator_w<W>_g1.npy).
"""
import glob
import json
import os
from typing import NamedTuple

import numpy as np
from PIL import Image

import cache
from background import column_backgrounds, convex_hull_subtract, smoothed
from frames import parse_frames
from kymograph import find_frames, load_kymograph

BETWEEN = (("none", "None (no normalization)"),
           ("ls", "Least squares (log mean profiles, meridian + equator)"),
           ("eq_bg", "Equatorial background, integrated r 90–600 px"),
           ("mer_bg", "Meridional background, integrated r 90–600 px"))
LS_RANGE = {"mer": (50, 600), "eq": (50, 330)}  # px, end excluded
BG_RANGE = (90, 600)                             # px, both included
I10_PX, I10_SEARCH = 113, 30                     # expected I1,0 radius and search half-width (px)
SIGMA = {"mer": 3.0, "eq": 6.0}                  # peak weight widths (px)


# ------------------------------------------------------------------ beamline log

class BeamLog(NamedTuple):
    path: str
    header: dict
    i0: np.ndarray
    i1: np.ndarray


def find_beamlog(qf):
    """The BioCAT log of the recording whose quadrant-folded images are in ``qf``, or "".
    Looks in the run folder (the parent of qf_results) and in qf itself: ``<run>.log`` first, then
    any .log whose first line is a BioCAT header."""
    qf = os.path.normpath(qf)
    for folder in (os.path.dirname(qf), qf):
        own = os.path.join(folder, os.path.basename(folder) + ".log")
        if os.path.isfile(own):
            return own
        for path in sorted(glob.glob(os.path.join(folder, "*.log"))):
            try:
                with open(path, errors="replace") as f:
                    if f.readline().startswith("#Instrument"):
                        return path
            except OSError:
                pass
    return ""


def read_beamlog(path):
    """BeamLog of a BioCAT log; ValueError if it has no I0 / I1 columns."""
    header, rows, columns = {}, [], None
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                if "\t" in line and "I0" in line.split("\t"):
                    columns = [c.strip() for c in line[1:].split("\t")]
                else:
                    key, _, value = line[1:].partition(":")
                    header[key.strip()] = value.strip()
            elif line.strip():
                rows.append(line.rstrip("\n").split("\t"))
    if columns is None or "I1" not in columns:
        raise ValueError(f"{os.path.basename(path)}: no I0 / I1 columns")
    i0, i1 = (np.array([float(r[columns.index(c)]) for r in rows]) for c in ("I0", "I1"))
    return BeamLog(path, header, i0, i1)


def i0_factors(i0, frames):
    """Per kept frame: mean(I0 over the kept frames) / I0, or None if the log is too short."""
    frames = np.asarray(frames)
    if len(i0) <= frames.max() or np.any(i0[frames] <= 0):
        return None
    v = i0[frames]
    return v.mean() / v


def flicker(totals, frames):
    """Frame-to-frame jitter of ``totals`` (one per kept frame) in %: RMS of the differences of the
    log between consecutive frames, / sqrt(2).  NaN with fewer than two such pairs."""
    totals, frames = np.asarray(totals, dtype=float), np.asarray(frames)
    pair = (np.diff(frames) == 1) & (totals[:-1] > 0) & (totals[1:] > 0)
    if pair.sum() < 2:
        return np.nan
    d = np.diff(np.log(np.clip(totals, 1e-300, None)))[pair]
    return 100.0 * np.sqrt(np.mean(d ** 2) / 2.0)


# ------------------------------------------------------------------ equatorial strip

_STRIPS = {}  # in memory too, for output folders that cannot be written


def equator_strips(qf, out, width):
    """(n_frames, radius) equatorial strip sums (``width`` px tall, positive half), cached in
    <out>/equator_w<W>_g1.npy (reused while the image files are unchanged)."""
    frames = find_frames(qf)
    key = {"frames_hash": cache.frames_hash(frames), "width": width}
    memo = (os.path.abspath(qf), width, key["frames_hash"])
    if memo in _STRIPS:
        return _STRIPS[memo]
    base = os.path.join(out, f"equator_w{width}_g1")
    strips = None
    if cache.matches(cache.read_json(base + ".json"), key) and os.path.exists(base + ".npy"):
        strips = np.load(base + ".npy")
        if strips.shape[0] != len(frames):
            strips = None
    if strips is None:
        rows = []
        for f in frames:
            img = np.asarray(Image.open(f), dtype=float)
            cy, cx = img.shape[0] // 2, img.shape[1] // 2
            rows.append(img[cy - width // 2:cy - width // 2 + width, cx:].sum(axis=0))
        strips = np.array(rows)
        try:
            os.makedirs(out, exist_ok=True)
            np.save(base + ".npy", strips)
            with open(base + ".json", "w") as fh:
                json.dump(dict(key, source=os.path.abspath(qf)), fh, indent=2)
        except OSError:
            pass  # kept in memory only
    _STRIPS[memo] = strips
    return strips


# ------------------------------------------------------------------ between sequences

class Sequence(NamedTuple):
    """What the between-sequence scales need of one recording; profiles indexed by px from centre."""
    label: str
    mer: np.ndarray   # mean meridional profile
    eq: np.ndarray    # mean equatorial profile (None if not needed)
    m3: float         # M3 radius (px)


def mean_profiles(kymo_npy, qf, frames, factors, need_eq):
    """(meridional, equatorial or None) mean over ``frames`` (source frame numbers), each frame
    multiplied by its factor."""
    kymo, pos, meta = load_kymograph(kymo_npy)
    full = np.zeros((int(pos[-1]) + 1, kymo.shape[1]))
    full[pos.astype(int)] = kymo
    w = np.asarray(factors, dtype=float)
    mer = (full[:, frames] * w).mean(axis=1)
    eq = None
    if need_eq:
        strips = equator_strips(qf, os.path.dirname(kymo_npy), meta["width"])
        eq = (strips[frames] * w[:, None]).mean(axis=0)
    return mer, eq


def peak_weights(r, centres, sigma):
    return 1.0 + sum(np.exp(-(r - c) ** 2 / (2 * sigma ** 2)) for c in centres)


def i10_radius(eq):
    """I1,0: the highest point of the hull-subtracted equatorial profile near I10_PX."""
    lo, hi = I10_PX - I10_SEARCH, I10_PX + I10_SEARCH
    seg = convex_hull_subtract(smoothed(eq[lo - 20:hi + 20], 3), 0)[20:-20]
    return float(lo + np.argmax(seg)) if seg.size and seg.max() > 0 else float(I10_PX)


def ls_regions(seqs):
    """{projection: (X (n, R) log10 profiles, w (R,) weights summing to 1)} over the pixels positive
    in every sequence."""
    out = {}
    i10 = np.mean([i10_radius(s.eq) for s in seqs])
    m3 = np.mean([s.m3 for s in seqs])
    for key, centres in (("mer", (m3, 2 * m3)), ("eq", (i10, np.sqrt(3) * i10))):
        lo, hi = LS_RANGE[key]
        I = np.array([getattr(s, key)[lo:hi] for s in seqs])
        ok = np.all(np.isfinite(I) & (I > 0), axis=0)
        p = peak_weights(np.arange(lo, hi)[ok], centres, SIGMA[key])
        out[key] = (np.log10(I[:, ok]), p / p.sum())
    return out


def ls_scales(regions, n):
    """log10 scale per sequence (mean 0).  Each projection weighs half and the pairs equally, so the
    solution equalises the mean over projections of the weighted mean log level."""
    mu = sum(X @ w for X, w in regions.values()) / len(regions)
    return -(mu - mu.mean())


def level_spread(regions, a):
    """{projection: spread of the weighted mean log10 level between the sequences after adding a, %}
    = 100 (10^SD - 1), SD over the sequences (ddof 1).  Differences in shape are not counted: no
    scale factor can remove them."""
    return {key: 100.0 * (10 ** np.std(X @ w + a, ddof=1) - 1.0) for key, (X, w) in regions.items()}


def background_area(profile, p):
    """Background (hull or fit, as in ``p``) of a mean profile, summed over BG_RANGE."""
    lo, hi = BG_RANGE
    y = profile[lo:hi + 1]
    return float(column_backgrounds(y[:, None], np.arange(lo, hi + 1.0), p)[:, 0].sum())


class Between(NamedTuple):
    scales: np.ndarray   # per sequence
    metric: np.ndarray   # the background areas (bg methods), else None
    before: dict         # level_spread() without scaling
    after: dict          # ... with the scales


def between_scales(seqs, method, p):
    """Between(...) for ``method`` (a key of BETWEEN other than none)."""
    n = len(seqs)
    regions = ls_regions(seqs)
    metric = None
    if method == "ls":
        a = ls_scales(regions, n)
    else:
        metric = np.array([background_area(s.eq if method == "eq_bg" else s.mer, p) for s in seqs])
        if np.any(~np.isfinite(metric) | (metric <= 0)):
            raise ValueError("a background area is zero or negative")
        a = -(np.log10(metric) - np.log10(metric).mean())
    return Between(10 ** a, metric, level_spread(regions, np.zeros(n)), level_spread(regions, a))


def parse_used(text, frames):
    """The kept frames (source numbers) in the typed list ``text`` (empty: all kept frames); dropped frames are skipped."""
    frames = np.asarray(frames)
    if not text.strip():
        return frames
    want = set(parse_frames(text))
    used = np.array([f for f in frames if f in want], dtype=int)
    if not used.size:
        raise ValueError(f"none of the frames {text!r} is kept")
    return used


# ------------------------------------------------------------------ both, for the loaded recordings

class Within(NamedTuple):
    log: str             # path used ("" if none)
    status: str          # what was read, or why not
    i0: np.ndarray       # factors per kept frame, or None
    before: float        # flicker (%) as measured
    after: float         # ... after the I0 normalisation (NaN without one)
    beam: object = None  # the BeamLog read, or None


class Plan(NamedTuple):
    factors: list        # per recording: factor per kept frame (all the normalisation applied)
    within: list         # per recording: Within
    between: object      # Between, None (method none) or an error message (str)
    used: list           # per recording: the frames the between-sequence profiles average


def within(rec, log_path):
    """Within(...) of a Recording with the log at ``log_path`` ("" = none)."""
    totals = rec.raw_base.sum(axis=0)
    before = flicker(totals, rec.frames)
    if not log_path:
        return Within("", "no log found", None, before, np.nan)
    try:
        log = read_beamlog(log_path)
    except (OSError, ValueError) as e:
        return Within(log_path, f"cannot read: {e}", None, before, np.nan)
    f = i0_factors(log.i0, rec.frames)
    if f is None:
        return Within(log_path, f"{len(log.i0)} rows for {rec.n_source} frames, or I0 ≤ 0", None, before, np.nan, log)
    cv = 100 * np.std(log.i0[rec.frames]) / np.mean(log.i0[rec.frames])
    return Within(log_path, f"{len(log.i0)} frames, I0 varies {cv:.2f}%", f, before, flicker(totals * f, rec.frames),
                  log)


def plan(recs, logs, use_i0, method, frames_text, p):
    """The normalisation of the Recordings ``recs``: ``logs`` = log path per recording, ``use_i0``,
    ``method`` (BETWEEN key), ``frames_text`` (frames the between profiles use; "" = all kept), ``p``
    the parameters (background method).  Between-sequence errors become a message, not an exception."""
    wi = [within(r, path) for r, path in zip(recs, logs)]
    factors = [w.i0 if use_i0 and w.i0 is not None else np.ones(len(r.frames)) for r, w in zip(recs, wi)]
    used, between = [r.frames for r in recs], None
    if method != "none":
        try:
            if len(recs) < 2:
                raise ValueError("needs two or more sequences")
            used = [parse_used(frames_text, r.frames) for r in recs]
            seqs = []
            for r, f, u in zip(recs, factors, used):
                w = f[np.searchsorted(r.frames, u)]
                mer, eq = mean_profiles(r.npy, r.qf, u, w, need_eq=True)
                m3 = cache.read_json(os.path.join(os.path.dirname(r.npy), "calibration.json")).get("n_m3_pixels")
                seqs.append(Sequence(r.label, mer, eq, float(m3) if m3 else 1.0 / (14.34 * r.inv)))
            between = between_scales(seqs, method, p)
            factors = [f * s for f, s in zip(factors, between.scales)]
        except Exception as e:  # shown in the dialog
            between = f"{type(e).__name__}: {e}" if not isinstance(e, ValueError) else str(e)
    return Plan(factors, wi, between, used)
