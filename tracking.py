"""What both trackers (multiscale.py, unet.py) share: their input, the reuse of a saved run, the command line."""
import argparse
import json
import os
from dataclasses import dataclass

import numpy as np

import cache
from frames import excluded_mask, parse_frames
from kymograph import load_kymograph

VERSION = 2  # raise when the tracking changes, so saved runs are made again


@dataclass
class TrackingInput:
    npy: str
    kymo: np.ndarray    # position x frame, every frame
    pos: np.ndarray     # radius (px) of each row
    inv_nm: float       # nm^-1 per px, or None
    frames: np.ndarray  # the frames kept
    dropped: list       # the frames left out

    def key(self):
        """The part of a run's cache key that describes its input."""
        return dict(version=VERSION, kymo=os.path.abspath(self.npy), kymo_sha256=cache.array_hash(self.kymo),
                    n_frames_source=int(self.kymo.shape[1]), inv_nm_per_pixel=self.inv_nm,
                    dropped_frames=self.dropped)

    def d_nm(self, r):
        return 1.0 / (r * self.inv_nm) if self.inv_nm and r > 0 else np.nan


def load_input(npy, p, exclude=None):
    """``exclude``: frames (from 0) to leave out; None = the blank frames."""
    kymo, pos, meta = load_kymograph(npy)
    drop = excluded_mask(kymo, p, exclude)
    return TrackingInput(npy, kymo, pos, meta.get("inv_nm_per_pixel"), np.flatnonzero(~drop),
                         np.flatnonzero(drop).tolist())


def saved_run(out_dir, key, files):
    """The saved run.json if it was made with ``key`` and its files are there, else None."""
    run = cache.read_json(os.path.join(out_dir, "run.json"))
    complete = all(os.path.exists(os.path.join(out_dir, name)) for name in files)
    return run if complete and cache.matches(run, key) else None


def save_run(out_dir, run):
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=2)


def run_cli(process, name, doc):
    """Command line of a tracker: ``process`` one kymograph into <kymo dir>/<name>_w<W>."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kymo", help="per-frame kymograph .npy (meridian_w<W>_g1.npy)")
    ap.add_argument("--out-dir", help=f"default <kymo dir>/{name}_w<W>")
    ap.add_argument("--params", help="parameters.json (missing keys use the defaults)")
    ap.add_argument("--peaks", help="expected_peaks.csv for the labels")
    ap.add_argument("--exclude", help='frames to exclude, from 0, e.g. "0-1, 40-60" (default: blank frames)')
    args = ap.parse_args()
    params = {}
    if args.params:
        with open(args.params) as f:
            params = json.load(f)
    width = os.path.basename(args.kymo).split("_")[1]
    out = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.kymo)), f"{name}_{width}")
    exclude = None if args.exclude is None else parse_frames(args.exclude)
    print(process(args.kymo, out, params, args.peaks, exclude)[0])
