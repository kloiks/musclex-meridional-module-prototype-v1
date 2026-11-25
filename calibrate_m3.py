#!/usr/bin/env python3
"""M3 internal calibration from a centred, rotated (quadrant-folded) TIF.

The M3 meridional reflection is the 3rd order of the ~42.9 nm myosin repeat,
d_M3 = 14.34 nm.  Its pixel radius n_M3 from the centre fixes the
small-angle scale  C = n_M3 * d_M3  [nm * px]  (MuscleX lambda_sdd),
so q = n / C  [nm^-1].

Example:
    python calibrate_m3.py frame_00001_folded_compressed.tif --gap 20 --rmin 250 --debug
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from background import convex_hull_subtract

D_M3_NM = 14.34  # 42.9 / 3


def meridional_profile(img, gap):
    """Meridian strip of width ``gap`` summed across, folded about the centre by radius."""
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    mer = img[:, cx - gap // 2:cx + gap // 2].sum(axis=1)
    radii = np.arange(min(cy, len(mer) - 1 - cy) + 1)
    return radii, mer[cy - radii] + mer[cy + radii]


def calibrate(img, gap, rmin, d_m3=D_M3_NM):
    """Return (calibration dict, radii, background-subtracted profile)."""
    radii, folded = meridional_profile(img, gap)
    folded = convex_hull_subtract(folded, rmin)
    n_m3 = int(radii[np.argmax(np.where(radii >= rmin, folded, 0))])
    data = dict(from_m3(n_m3, d_m3), rmin_pixels=rmin, gap_pixels=gap)
    return data, radii, folded


def from_m3(n_m3, d_m3=D_M3_NM):
    """Calibration keys for an M3 at ``n_m3`` px."""
    c = n_m3 * d_m3
    return {"n_m3_pixels": n_m3, "d_m3_nm": d_m3, "lambda_sdd": c,
            "inv_nm_per_pixel": 1.0 / c if c else None}


def save_projection_png(radii, folded, n_m3, rmin, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(radii, folded, lw=1.0, color="black", label="meridional projection (hull subtracted)")
    ax.axvspan(0, rmin, color="grey", alpha=0.2, label=f"rmin = {rmin}")
    ax.axvline(n_m3, color="red", ls="--", label=f"M3 @ {n_m3} px")
    ax.set_xlabel("radius from centre [pixels]")
    ax.set_ylabel("summed intensity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def calibrate_file(image, gap, rmin, out_json, debug_png=None, d_m3=D_M3_NM):
    img = np.asarray(Image.open(image), dtype=float)
    data, radii, folded = calibrate(img, gap, rmin, d_m3)
    data["image"] = os.path.abspath(image)
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    if debug_png:
        save_projection_png(radii, folded, data["n_m3_pixels"], rmin, debug_png)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image")
    p.add_argument("--gap", type=int, default=20, help="meridian strip width in px (default 20)")
    p.add_argument("--rmin", type=int, default=250, help="min radius in px for M3 (default 250)")
    p.add_argument("-o", "--output", help="calibration JSON (default <image>_calibration.json)")
    p.add_argument("--debug", action="store_true", help="also save the projection PNG")
    args = p.parse_args()

    stem = os.path.splitext(args.image)[0]
    data = calibrate_file(args.image, args.gap, args.rmin,
                          args.output or stem + "_calibration.json",
                          stem + "_projection.png" if args.debug else None)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
