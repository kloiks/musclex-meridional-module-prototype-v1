"""The app's parameters (config/parameters.json) and the paths of its config files."""
import json
import os

from . import ROOT, cache
from . import maxflow
from . import multiscale
from . import unet
from .calibrate_m3 import D_M3_NM

CONFIG = os.path.join(ROOT, "config")
PARAMS_JSON = os.path.join(CONFIG, "parameters.json")
PEAKS_CSV = os.path.join(CONFIG, "expected_peaks.csv")

DEFAULTS = dict(multiscale.DEFAULTS, **unet.DEFAULTS, **maxflow.DEFAULTS,
                width=20, m3_rmin=250, d_m3=D_M3_NM, pos_lo=0, pos_hi=1200, stretch=1.4,
                norm_i0=False, norm_between="none", norm_frames="2-20")  # Normalization dialog


def load_params():
    """The saved parameters; missing keys use the defaults, unknown ones are dropped."""
    saved = cache.read_json(PARAMS_JSON)
    return {k: saved.get(k, default) for k, default in DEFAULTS.items()}


def save_params(values):
    """Update config/parameters.json with ``values``, keeping the other keys."""
    os.makedirs(CONFIG, exist_ok=True)
    with open(PARAMS_JSON, "w") as f:
        json.dump(dict(load_params(), **values), f, indent=2)
