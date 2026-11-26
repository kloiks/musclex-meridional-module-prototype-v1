"""One tracked recording as the viewer shows it: its kymograph, tracks and, per grouping, backgrounds."""
import csv
import os
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from . import cache
from . import multiscale
from .background import column_backgrounds, profile_noise, shared_background, smoothed
from .ddf import column_time, from_trigger, load_traces
from .kymograph import load_kymograph, recording_name
from .labels import unmatched_names

GROUPS = (1, 2, 4, 8, 16)  # frame groupings always on offer, whatever the tracker and its scales


class Job(NamedTuple):
    """What to run for one recording."""
    qf: str
    ddf: str  # "" if none
    out: str

    @property
    def label(self):
        return recording_name(self.qf)


class Grouping(NamedTuple):
    """Position x frame arrays at one grouping."""
    grouped: np.ndarray      # sliding mean, as the tracker saw it
    profiles: np.ndarray     # ... smoothed along r
    backgrounds: np.ndarray  # each column's own hull or fit
    flats: np.ndarray        # profiles minus the background the tracker subtracted


class Column(NamedTuple):
    """One frame at one grouping, along r."""
    profile: np.ndarray
    background: np.ndarray  # the one subtracted
    flat: np.ndarray
    sigma: np.ndarray


class Peak(NamedTuple):
    x: float       # background-subtracted intensity at the peak
    q: float
    name: str
    height: float  # as tracked


@dataclass
class Recording:
    label: str
    color: str
    traces: list          # force and length, time from the trigger; [] without a .ddf
    dt: float             # s per frame
    inv: float            # nm^-1 per px
    detector: str         # "multiscale", "unet" or "maxflow"
    params: dict
    n_source: int         # frames, the excluded ones included
    frames: np.ndarray    # the frames kept
    pos: np.ndarray       # radius (px) of each row
    base: np.ndarray      # kymograph, position x kept frame
    scales: np.ndarray    # the tracker's
    r: np.ndarray         # scale x track x kept frame
    height: np.ndarray
    labels: dict          # track -> name
    unmatched: dict       # track -> X1, X2, ... (U-Net, max-flow)
    npy: str = ""         # the kymograph file (its folder is the output folder)
    qf: str = ""          # the quadrant-folded images
    s: int = 1            # frames grouped in the current view
    k: int = 0            # index of the tracker scale whose tracks it shows
    _groupings: dict = field(default_factory=dict)

    def __post_init__(self):
        self.q = self.pos * self.inv
        self.raw_base, self.raw_height = self.base, self.height  # before normalisation
        self.norm_note = ""                                       # for the kymograph title
        self.beam = None                                          # BeamLog while normalised, else None
        self.factors = np.ones(len(self.frames))                  # per kept frame, applied
        self.times = (self.frames + 0.5) * self.dt
        self.t_end = self.n_source * self.dt
        self.shared = shared_background(self.base, self.pos, self.params)  # None: one per column
        self._column = {int(f): i for i, f in enumerate(self.frames)}

    # ---------------------------------------------------------------- normalisation

    def normalize(self, factors):
        """Multiply each kept frame of the kymograph (and the tracked heights) by its factor;
        ones = as measured."""
        self.factors = np.asarray(factors, dtype=float)
        self.base = self.raw_base * self.factors
        self.height = self.raw_height * self.factors
        self.shared = shared_background(self.base, self.pos, self.params)
        self._groupings = {}

    # ---------------------------------------------------------------- grouping

    def scale_for(self, want):
        """(s, k) when ``want`` is selected: s frames grouped and the tracker scale nearest to that
        (U-Net has only the per-frame one)."""
        s = want if want in GROUPS else int(self.scales[np.argmin(np.abs(self.scales - want))])
        s = min(s, len(self.frames))
        return s, int(np.argmin(np.abs(self.scales - s)))

    def group_by(self, want):
        self.s, self.k = self.scale_for(want)

    @property
    def view(self):
        """The arrays at the current grouping, computed once."""
        if self.s not in self._groupings:
            grouped = multiscale.bin_time(self.base, self.s)
            profiles = np.column_stack([smoothed(col, self.params["smooth"]) for col in grouped.T])
            backgrounds = column_backgrounds(grouped, self.pos, self.params)
            subtracted = backgrounds if self.shared is None else self.shared[:, None]
            self._groupings[self.s] = Grouping(grouped, profiles, backgrounds, profiles - subtracted)
        return self._groupings[self.s]

    # ---------------------------------------------------------------- at one time

    def column_at(self, t):
        """Index of the kept frame at time t, or None (excluded, or outside)."""
        return self._column.get(int(np.floor(t / self.dt)))

    def column(self, col):
        own = self.view.backgrounds[:, col]
        subtracted = own if self.shared is None else self.shared
        per_column_hull = self.shared is None and self.params["bg_method"] != "fit"
        flat, sigma = profile_noise(self.view.grouped[:, col], self.params,
                                    None if per_column_hull else subtracted)
        return Column(self.view.profiles[:, col], subtracted, flat, sigma)

    def peaks_at(self, col, flat, tracks, q_range):
        """The Peaks of ``tracks`` in that frame whose q is in range."""
        out = []
        for a, name in tracks:
            r = self.r[self.k, a, col]
            if np.isfinite(r) and q_range[0] <= r * self.inv <= q_range[1]:
                out.append(Peak(float(np.interp(r, self.pos, flat)), r * self.inv, name,
                                float(self.height[self.k, a, col])))
        return out

    # ---------------------------------------------------------------- tracks

    def tracks(self, q_range, wanted=None, unmatched=True):
        """[(track, name)] with the median q in range and the name matching the pattern ``wanted``."""
        names = {**self.labels, **(self.unmatched if unmatched else {})}
        out = []
        for a, name in sorted(names.items()):
            r = self.r[self.k, a]
            if (wanted and not wanted.match(name)) or not np.isfinite(r).any():
                continue
            if q_range[0] <= np.nanmedian(r) * self.inv <= q_range[1]:
                out.append((a, name))
        return out

    @property
    def threshold(self):
        """Acceptance SNR of multiscale at the current scale."""
        return multiscale.accept_snr(self.params, self.k, len(self.scales))

    @property
    def background_mode(self):
        p = self.params
        return f"{'fit, ' + p['bg_comp2'] if p['bg_method'] == 'fit' else 'hull'}, {p['background']}"


def load_recording(label, npy, ddf_path, track_dir, color):
    """From the kymograph, the .ddf ("" if none) and the tracker's output folder."""
    run = cache.read_json(os.path.join(track_dir, "run.json"))
    detector = run.get("detector", "multiscale")
    with open(os.path.join(track_dir, "labels.csv")) as f:
        labels = {int(r["anchor"]): r["label"] for r in csv.DictReader(f)}
    with np.load(os.path.join(track_dir, f"{detector}.npz")) as data:
        scales, frames, r, height = (data[k] for k in ("scales", "frames", "r", "height"))
    kymo, pos, meta = load_kymograph(npy)
    params = dict(multiscale.DEFAULTS, **{k: run[k] for k in multiscale.TRACK_KEYS if k in run})
    keep = (pos >= params["rmin"]) & (pos <= params["rmax"])

    ddf, traces = load_traces(ddf_path) if ddf_path else (None, [])
    time = column_time(meta, ddf)
    return Recording(label=label, color=color, traces=from_trigger(time.trigger, traces, [])[0], dt=time.dt,
                     inv=meta["inv_nm_per_pixel"], detector=detector, params=params, n_source=kymo.shape[1],
                     frames=frames, pos=pos[keep], base=kymo[keep][:, frames], scales=scales, r=r, height=height,
                     labels=labels, unmatched=unmatched_names(r[0], labels) if detector != "multiscale" else {},
                     npy=npy, qf=images_folder(npy, meta))


def images_folder(npy, meta):
    """The quadrant-folded images of a kymograph: as saved in its sidecar, or, if that folder is gone
    (data moved), the folder holding the output folder (the default output is <qf folder>/kymo)."""
    source = meta.get("source", "")
    return source if os.path.isdir(source) else os.path.dirname(os.path.dirname(os.path.abspath(npy)))


def group_options(recs):
    """Groupings offered: GROUPS (as far as the longest recording goes) and the tracker's scales."""
    longest = max(len(r.frames) for r in recs)
    return sorted({g for g in GROUPS if g <= longest} | {int(s) for r in recs for s in r.scales})
