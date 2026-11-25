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
import csv
import os

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment

import cache
import multiscale
import tracking
from background import column_backgrounds, shared_background, smoothed
from frames import format_frames
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
        points = [(int(f), float(r[i]), float(score[i])) for i in at]
        pairs = match(np.abs(np.array([t[-1][1] for t in live])[:, None] - r[at][None, :]), gate) if live else []
        for a, b in pairs:
            live[a].append(points[b])
        for b in sorted(set(range(len(at))) - {b for _, b in pairs}):
            live.append([points[b]])
            tracks.append(live[-1])
    return [t for t in tracks if len(t) >= min_length]


# ------------------------------------------------------------------ run

def peak_heights(kymo, pos, r, p):
    """Height of each (track x frame) position above the baseline the viewer subtracts."""
    shared = shared_background(kymo, pos, p)
    bg = column_backgrounds(kymo, pos, p) if shared is None else shared[:, None]
    flat = np.column_stack([smoothed(col, p["smooth"]) for col in kymo.T]) - bg
    height = np.full(r.shape, np.nan)
    for t in range(r.shape[1]):
        ok = np.isfinite(r[:, t])
        height[ok, t] = np.interp(r[ok, t], pos, flat[:, t])
    return height


def detect(inp, p):
    """(r, height, score), each (track x kept frame), NaN where not detected."""
    kymo, pos = inp.kymo[:, inp.frames], inp.pos
    frame, row, sc = find_peaks(heatmap(normalize(kymo)), p["unet_threshold"])
    tracks = link(frame, row, sc, p["unet_gate"], p["unet_memory"], p["unet_min_length"])
    r, score = np.full((2, len(tracks), len(inp.frames)), np.nan)
    for a, points in enumerate(tracks):
        for t, x, s in points:
            r[a, t], score[a, t] = x + pos[0], s
    inside = (r >= p["rmin"]) & (r <= p["rmax"])
    r[~inside] = score[~inside] = np.nan
    found = np.isfinite(r).any(axis=1)
    r, score = r[found], score[found]
    keep = (pos >= p["rmin"]) & (pos <= p["rmax"])
    return r, peak_heights(kymo[keep], pos[keep], r, p), score


def write_tracks(path, rows, names, inp):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "label", "frame", "r", "d_nm", "height", "score"])
        for row in rows:
            numbers = (row["r"], inp.d_nm(row["r"]), row["height"], row["score"])
            w.writerow([row["track"], names[row["track"]], row["frame"]] + [f"{v:.5g}" for v in numbers])


def process(npy, out_dir, params=None, peaks_csv=None, exclude=None):
    """Detect and track (or reuse the saved result in out_dir) and label; returns (summary, cached)."""
    p = dict(multiscale.DEFAULTS, **DEFAULTS)
    p.update(params or {})
    inp = tracking.load_input(npy, p, exclude)
    key = dict({k: p[k] for k in KEYS}, **inp.key(), detector="unet", model=os.path.basename(MODEL),
               model_sha256=cache.file_hash(MODEL))

    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, "unet.npz")
    cached = tracking.saved_run(out_dir, key, ("unet.npz",)) is not None
    if cached:
        with np.load(npz) as data:
            r, height, score = data["r"][0], data["height"][0], data["score"]
    else:
        r, height, score = detect(inp, p)
        # one "scale" (per frame), in the layout the viewer reads
        np.savez_compressed(npz, scales=[1], frames=inp.frames, r=r[None], height=height[None], score=score)
        tracking.save_run(out_dir, key)

    rows = [dict(track=int(a), frame=int(inp.frames[t]), r=float(r[a, t]), height=float(height[a, t]),
                 score=float(score[a, t])) for a, t in zip(*np.nonzero(np.isfinite(r)))]
    peaks = read_expected_peaks(peaks_csv) if peaks_csv else []
    labels = label_tracks(rows, inp.inv_nm, peaks, p["tol_pct"], p["min_frames"], share_names=True)
    write_labels(out_dir, labels)
    names = {**{a: v[0] for a, v in labels.items()}, **unmatched_names(r, labels)}
    write_tracks(os.path.join(out_dir, "tracks.csv"), rows, names, inp)
    return summarize(inp, p, len(r), labels), cached


def summarize(inp, p, n_tracks, labels):
    text = (f"U-Net ({os.path.basename(MODEL)}): {n_tracks} tracks, "
            f"excluded frames {format_frames(inp.dropped) or 'none'}; "
            f"labelled {len(labels)}, unmatched {n_tracks - len(labels)}: "
            + ", ".join(sorted({v[0] for v in labels.values()})))
    if len(inp.frames) < p["unet_min_length"]:
        text += (f"\nWARNING: only {len(inp.frames)} frames kept, fewer than the minimum track length "
                 f"({p['unet_min_length']}): no track can be found. Lower it in Parameters.")
    return text


if __name__ == "__main__":
    tracking.run_cli(process, "unet", __doc__)
