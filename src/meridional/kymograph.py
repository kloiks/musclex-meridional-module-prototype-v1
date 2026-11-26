#!/usr/bin/env python3
"""Meridional kymograph from a folder of quadrant-folded TIFF frames.

Each frame's meridian strip (``--width`` px, centred on the image centre) is
summed across, the positive half kept, and the profile becomes one column of a
(position x frame) image.

Writes to ``--out-dir`` (default <qf folder>/kymo):
    meridian_w<W>_g1.npy / .json   the kymograph and its sidecar (g1: one frame per column)
    meridian_w<W>_g1_log.png       log(1+I), contrast at the 1-99.5 percentiles

Example:
    python kymograph.py .../RigA_WT9EDL_LoActive1_0040/qf_results --width 20
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from PIL import Image
from matplotlib.figure import Figure

from . import cache

PERCENTILES = (1.0, 99.5)


def natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(path))]


def find_frames(folder):
    files = glob.glob(os.path.join(folder, "*.tif")) or glob.glob(os.path.join(folder, "*.tiff"))
    if not files:
        raise ValueError(f"no TIFF frames in {folder}")
    return sorted(files, key=natural_key)


def recording_name(qf_folder):
    return os.path.basename(os.path.dirname(os.path.abspath(qf_folder)))


def build_kymograph(frames, width):
    """(kymo, pos): meridian strip sums, positive half of the axial axis."""
    cols = []
    for f in frames:
        img = np.asarray(Image.open(f), dtype=float)
        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        cols.append(img[cy:, cx - width // 2:cx - width // 2 + width].sum(axis=1))
    kymo = np.column_stack(cols)
    return kymo, np.arange(kymo.shape[0])


def load_kymograph(npy):
    """(kymo, pos, sidecar dict) of a saved kymograph."""
    kymo = np.load(npy)
    with open(os.path.splitext(npy)[0] + ".json") as f:
        meta = json.load(f)
    return kymo, np.arange(kymo.shape[0]) + meta.get("pos_start", 0), meta


def plot_kymograph(kymo, pos, title, out_png, inv_nm_per_pixel=None):
    disp = np.log1p(np.clip(kymo, 0, None))
    vmin, vmax = np.percentile(disp, PERCENTILES)
    fig = Figure(figsize=(11, 6))
    ax = fig.subplots()
    im = ax.imshow(disp, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                   extent=[0.5, kymo.shape[1] + 0.5, pos[0], pos[-1]])
    ax.set_xlabel("frame")
    ax.set_ylabel("axial position from centre (pixels)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, pad=0.10 if inv_nm_per_pixel else 0.02)
    cb.set_label("log(1+I) (sum over strip)")
    if inv_nm_per_pixel:
        s = inv_nm_per_pixel
        ax.secondary_yaxis("right", functions=(lambda p: p * s, lambda q: q / s)) \
          .set_ylabel("q (nm$^{-1}$)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def kymograph_base(out_dir, width):
    """Path of the saved kymograph without its extension."""
    return os.path.join(out_dir, f"meridian_w{width}_g1")


def cached_kymograph(frames, out_dir, width):
    """(kymo, pos, cached); reuses the saved kymograph if it was made from the same image files
    (cache.frames_hash)."""
    base = kymograph_base(out_dir, width)
    if cache.matches(cache.read_json(base + ".json"), {"frames_hash": cache.frames_hash(frames)}):
        try:
            kymo, pos, _ = load_kymograph(base + ".npy")
            if kymo.shape[1] == len(frames):
                return kymo, pos, True
        except (OSError, ValueError):
            pass
    return (*build_kymograph(frames, width), False)


def make_kymograph(qf_folder, out_dir, width, inv_nm_per_pixel=None):
    """Build (or load) the kymograph and write what is out of date; returns (npy path, cached)."""
    frames = find_frames(qf_folder)
    frames_hash = cache.frames_hash(frames)  # before reading: images that change meanwhile are read again
    os.makedirs(out_dir, exist_ok=True)
    kymo, pos, cached = cached_kymograph(frames, out_dir, width)
    base = kymograph_base(out_dir, width)
    meta = {
        "source": os.path.abspath(qf_folder),
        "width": width,
        "n_source_frames": len(frames),
        "pos_start": int(pos[0]),
        "first_frame": os.path.basename(frames[0]),
        "last_frame": os.path.basename(frames[-1]),
        "frames_hash": frames_hash,
        "inv_nm_per_pixel": inv_nm_per_pixel,
    }
    if not cached:
        np.save(base + ".npy", kymo)
    elif cache.read_json(base + ".json") == meta and os.path.exists(base + "_log.png"):
        return base + ".npy", cached
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    plot_kymograph(kymo, pos, f"{recording_name(qf_folder)}  meridian w={width} px  ({len(frames)} frames)",
                   base + "_log.png", inv_nm_per_pixel)
    return base + ".npy", cached


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("qf_folder")
    p.add_argument("--width", type=int, default=20, help="meridian strip width in px (default 20)")
    p.add_argument("--calibration", help="calibration JSON for a q axis")
    p.add_argument("--out-dir", help="default <qf_folder>/kymo")
    args = p.parse_args()

    scale = None
    if args.calibration:
        with open(args.calibration) as f:
            scale = json.load(f).get("inv_nm_per_pixel")
    out_dir = args.out_dir or os.path.join(args.qf_folder, "kymo")
    npy, cached = make_kymograph(args.qf_folder, out_dir, args.width, scale)
    print(f"wrote {npy}" + (" (cached)" if cached else ""))


if __name__ == "__main__":
    main()
