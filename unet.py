#!/usr/bin/env python3
"""U-Net peak detection and tracking on a per-frame meridional kymograph (alternative to multiscale.py).

Excluded frames (frames.py) are cut out and the rest joined, the kymograph is normalized as
in training, and the net (models/new_w61.pt) gives a 0-1 score per pixel.  Maxima along r with
score >= unet_threshold are linked over time (Hungarian, unmatched allowed) within unet_gate px,
bridging up to unet_memory missing frames; tracks with fewer than unet_min_length detected
frames are dropped.  Model, normalization, peak finding and linking follow the training code
(train_unet.py, link_detections.py in meridional-simulation); the defaults are the settings
chosen on validation for new_w61.

Tracks are named after the expected peaks with the multiscale labelling rules, except that
tracks which never share a frame may share a name (a reflection can be found in pieces).
Tracks that match no expected peak are named X1, X2, ... from the centre out.

Writes to ``out_dir``: run.json, unet.npz, tracks.csv, labels.csv.

Example:
    python unet.py qf_results/kymo/meridian_w20_g1.npy --peaks config/expected_peaks.csv
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment

import cache
import multiscale
from background import column_backgrounds, shared_background, smoothed
from frames import excluded_mask, format_frames, parse_frames
from labels import label_tracks, read_expected_peaks, unmatched_names, write_labels

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "new_w61.pt")
DEFAULTS = dict(unet_threshold=0.4, unet_gate=10, unet_memory=10, unet_min_length=20)
# the keys that change tracks or heights; the blank-frame settings count through dropped_frames
KEYS = list(DEFAULTS) + ["rmin", "rmax", "background", "bg_method", "bg_comp2", "bg_neg_weight", "smooth"]


# ------------------------------------------------------------------ model

def normalize(a):
    """Log intensity, background along r removed, noise scaled to 1 (as in training)."""
    x = np.log1p(np.maximum(a, 0.0))
    z = x - median_filter(np.median(x, axis=1), size=101, mode="nearest")[:, None]
    spread = median_filter(np.median(np.abs(z), axis=1), size=101, mode="nearest") * 1.4826
    return np.clip(z / np.maximum(spread, 0.02)[:, None], -5, 15).astype(np.float32)


def build_net(base, levels):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def block(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.SiLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            ch = [base * 2 ** i for i in range(levels + 1)]
            self.first = nn.Conv2d(1, ch[0], 3, padding=1)
            self.down = nn.ModuleList(block(ch[max(i - 1, 0)], ch[i]) for i in range(levels + 1))
            self.reduce = nn.ModuleList(nn.Conv2d(ch[i + 1], ch[i], 1) for i in range(levels))
            self.up = nn.ModuleList(block(ch[i] * 2, ch[i]) for i in range(levels))
            self.out = nn.Conv2d(ch[0], 1, 1)
            self.factor = 2 ** levels

        def forward(self, x):
            skips = []
            x = self.first(x)
            for i, d in enumerate(self.down):
                x = d(x)
                if i < len(self.down) - 1:
                    skips.append(x)
                    x = F.max_pool2d(x, 2)
            for i in reversed(range(len(self.up))):
                x = F.interpolate(x, size=skips[i].shape[-2:], mode="bilinear", align_corners=False)
                x = self.up[i](torch.cat([self.reduce[i](x), skips[i]], 1))
            return self.out(x)

    return UNet()


def heatmap(x, model_path=MODEL):
    """Whole-kymograph score map; on the GPU if there is one with room, else the CPU."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        raise RuntimeError("U-Net detection needs PyTorch: pip install -r requirements.txt") from e

    def run(device):
        c = torch.load(model_path, map_location=device, weights_only=True)
        net = build_net(c["base"], c["levels"]).to(device)
        net.load_state_dict(c["model"])
        net.eval()
        n_pos, n_fr = x.shape
        t = torch.from_numpy(x)[None, None].to(device)
        t = F.pad(t, (0, -n_fr % net.factor, 0, -n_pos % net.factor), mode="replicate")
        with torch.inference_mode(), torch.autocast(device, enabled=device == "cuda"):
            out = torch.sigmoid(net(t))
        return out[0, 0, :n_pos, :n_fr].float().cpu().numpy()

    if torch.cuda.is_available():
        try:
            return run("cuda")
        except torch.OutOfMemoryError:
            pass
    return run("cpu")


# ------------------------------------------------------------------ peaks and tracks

def find_peaks(h, min_score):
    """(frame, r, score) arrays: local maxima along r in each frame, sub-pixel positions."""
    is_max = (h[1:-1] >= h[:-2]) & (h[1:-1] > h[2:]) & (h[1:-1] >= min_score)
    i, f = np.nonzero(is_max)
    i = i + 1
    left, mid, right = h[i - 1, f], h[i, f], h[i + 1, f]
    bend = left - 2 * mid + right
    step = np.zeros(len(i))
    ok = np.abs(bend) > 1e-6
    step[ok] = 0.5 * (left - right)[ok] / bend[ok]
    return f, i + np.clip(step, -0.5, 0.5), mid


def match(cost, gate):
    """Lowest-cost (track, peak) pairs; leaving either unmatched costs gate / 2."""
    n, m = cost.shape
    big = 1e6
    full = np.full((n + m, n + m), big)
    full[:n, :m] = np.where(cost <= gate, cost, big)
    full[np.arange(n), m + np.arange(n)] = gate / 2
    full[n + np.arange(m), np.arange(m)] = gate / 2
    full[n:, m:] = 0.0
    return [(a, b) for a, b in zip(*linear_sum_assignment(full))
            if a < n and b < m and cost[a, b] <= gate]


def link(frame, r, score, gate, memory, min_length):
    """Tracks, each a list of (frame, r, score), in order of their first detection."""
    tracks, live = [], []  # live: tracks that may still be continued
    for f in np.unique(frame):
        at = np.flatnonzero(frame == f)
        at = at[np.argsort(r[at])]
        live = [t for t in live if f - t[-1][0] <= memory + 1]
        pairs = match(np.abs(np.array([t[-1][1] for t in live])[:, None] - r[at][None, :]), gate) if live else []
        for a, b in pairs:
            live[a].append((int(f), float(r[at[b]]), float(score[at[b]])))
        for b in sorted(set(range(len(at))) - {b for _, b in pairs}):
            live.append([(int(f), float(r[at[b]]), float(score[at[b]]))])
            tracks.append(live[-1])
    return [t for t in tracks if len(t) >= min_length]


# ------------------------------------------------------------------ run

def detect(kymo, pos, p, frames):
    """(r, height, score), each (track x kept frame), NaN where not detected."""
    k = kymo[:, frames]
    f, i, s = find_peaks(heatmap(normalize(k)), p["unet_threshold"])
    tracks = link(f, i, s, p["unet_gate"], p["unet_memory"], p["unet_min_length"])
    r, score = np.full((2, len(tracks), len(frames)), np.nan)
    for a, t in enumerate(tracks):
        for fr, x, sc in t:
            r[a, fr], score[a, fr] = x + pos[0], sc
    inside = (r >= p["rmin"]) & (r <= p["rmax"])
    r[~inside] = score[~inside] = np.nan
    keep = np.isfinite(r).any(axis=1)
    r, score = r[keep], score[keep]

    # heights above the same baseline the viewer subtracts
    keep_r = (pos >= p["rmin"]) & (pos <= p["rmax"])
    base, pos = k[keep_r], pos[keep_r]
    shared = shared_background(base, pos, p)
    bg = column_backgrounds(base, pos, p) if shared is None else shared[:, None]
    flat = np.column_stack([smoothed(base[:, t], p["smooth"]) for t in range(len(frames))]) - bg
    height = np.full(r.shape, np.nan)
    for t in range(len(frames)):
        ok = np.isfinite(r[:, t])
        height[ok, t] = np.interp(r[ok, t], pos, flat[:, t])
    return r, height, score


def process(npy, out_dir, params=None, peaks_csv=None, exclude=None):
    """Detect and track (or reuse the cached result in out_dir) and label; returns (summary, cached)."""
    p = dict(multiscale.DEFAULTS, **DEFAULTS)
    p.update(params or {})
    kymo = np.load(npy)
    with open(os.path.splitext(npy)[0] + ".json") as f:
        meta = json.load(f)
    inv_nm = meta.get("inv_nm_per_pixel")
    pos = np.arange(kymo.shape[0]) + meta.get("pos_start", 0)
    drop = excluded_mask(kymo, p, exclude)
    frames = np.flatnonzero(~drop)
    key = dict({k: p[k] for k in KEYS}, detector="unet", model=os.path.basename(MODEL),
               model_sha256=cache.file_hash(MODEL), kymo=os.path.abspath(npy), kymo_sha256=cache.array_hash(kymo),
               n_frames_source=int(kymo.shape[1]), inv_nm_per_pixel=inv_nm,
               dropped_frames=np.flatnonzero(drop).tolist())

    os.makedirs(out_dir, exist_ok=True)
    run_json, npz = os.path.join(out_dir, "run.json"), os.path.join(out_dir, "unet.npz")
    cached = cache.matches(cache.read_json(run_json), key) and os.path.exists(npz)
    if cached:
        with np.load(npz) as data:
            r, height, score = data["r"][0], data["height"][0], data["score"]
    else:
        r, height, score = detect(kymo, pos, p, frames)
        # one "scale" (per frame), in the layout the viewer reads
        np.savez_compressed(npz, scales=[1], frames=frames, r=r[None], height=height[None], score=score)
        with open(run_json, "w") as f:
            json.dump(key, f, indent=2)

    rows = [dict(track=int(a), frame=int(frames[t]), r=float(r[a, t]), height=float(height[a, t]),
                 score=float(score[a, t])) for a, t in zip(*np.nonzero(np.isfinite(r)))]
    peaks = read_expected_peaks(peaks_csv) if peaks_csv else []
    labels = (label_tracks(rows, inv_nm, peaks, p["tol_pct"], p["min_frames"], share_names=True)
              if inv_nm and peaks else {})
    write_labels(out_dir, labels)
    names = {**{a: v[0] for a, v in labels.items()}, **unmatched_names(r, labels)}
    with open(os.path.join(out_dir, "tracks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "label", "frame", "r", "d_nm", "height", "score"])
        for x in rows:
            d = 1.0 / (x["r"] * inv_nm) if inv_nm and x["r"] > 0 else np.nan
            w.writerow([x["track"], names[x["track"]], x["frame"]] + [f"{v:.5g}" for v in (x["r"], d, x["height"], x["score"])])

    summary = (f"U-Net ({os.path.basename(MODEL)}): {len(r)} tracks, excluded frames "
               f"{format_frames(np.flatnonzero(drop)) or 'none'}; labelled {len(labels)}, unmatched {len(r) - len(labels)}: "
               + ", ".join(sorted({v[0] for v in labels.values()})))
    if len(frames) < p["unet_min_length"]:
        summary += (f"\nWARNING: only {len(frames)} frames kept, fewer than the minimum track length "
                    f"({p['unet_min_length']}): no track can be found. Lower it in Parameters.")
    return summary, cached


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kymo", help="per-frame kymograph .npy (meridian_w<W>_g1.npy)")
    ap.add_argument("--out-dir", help="default <kymo dir>/unet_w<W>")
    ap.add_argument("--params", help="parameters.json (missing keys use the defaults)")
    ap.add_argument("--peaks", help="expected_peaks.csv for the labels")
    ap.add_argument("--exclude", help='frames to exclude, from 0, e.g. "0-1, 40-60" (default: blank frames)')
    args = ap.parse_args()
    params = {}
    if args.params:
        with open(args.params) as f:
            params = json.load(f)
    width = os.path.basename(args.kymo).split("_")[1]
    out = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.kymo)), f"unet_{width}")
    exclude = None if args.exclude is None else parse_frames(args.exclude)
    print(process(args.kymo, out, params, args.peaks, exclude)[0])


if __name__ == "__main__":
    main()
