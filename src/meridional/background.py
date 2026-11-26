"""Background (convex hull or fitted smooth curve) and noise of meridional profiles.

Shared by calibrate_m3.py, multiscale.py, unet.py and the viewer.  ``p`` is the parameter dict
(bg_method, bg_comp2, bg_neg_weight, background, smooth, noise_win, noise_pct).
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import percentile_filter, uniform_filter1d
from scipy.optimize import least_squares
from scipy.stats import norm


def smoothed(profile, smooth):
    p = np.asarray(profile, dtype=float)
    return uniform_filter1d(p, smooth, mode="nearest") if smooth > 1 else p


# ------------------------------------------------------------------ convex hull

def convex_hull_subtract(y, start):
    """Subtract the lower convex hull of y[start:] (MuscleX convexHull); zero below start."""
    y = np.asarray(y, dtype=float)
    seg = y[start:]
    out = np.zeros_like(y)
    if seg.size < 5:
        return out
    hull, i = [0], 0
    while i < seg.size - 1:  # next hull vertex: the point of minimum slope from i
        i += 1 + int(np.argmin((seg[i + 1:] - seg[i]) / np.arange(1, seg.size - i)))
        hull.append(i)
    background = PchipInterpolator(hull, seg[hull])(np.arange(seg.size))
    out[start:] = np.clip(seg - background, 0, None)
    return out


def flatten(profile, smooth, background=None):
    """Profile minus its own convex hull, or minus a given background."""
    p = smoothed(profile, smooth)
    return convex_hull_subtract(p, 0) if background is None else p - background


def hull_background(profile, smooth):
    """The convex-hull background that flatten() removes from ``profile``."""
    p = smoothed(profile, smooth)
    return p - convex_hull_subtract(p, 0)


# ------------------------------------------------------------------ fit

# MuscleX bg_fitting general background, along the meridian:
#   B(r) = baseline + A1 (s_exp(r) - s_exp(r_edge)) + A2 (s_comp2(r) - s_comp2(r_edge))
# each kernel zeroed at the last radius so the baseline is the floor there.

def radial_shape(r, scale, kernel, shape=None):
    x = r / scale
    if kernel == "exponential":
        return np.exp(-x)
    if kernel == "lorentzian":
        return 1.0 / (1.0 + x * x)
    if kernel == "powerlaw":  # shape = n
        return (1.0 + x) ** -shape
    if kernel == "stretched":  # shape = beta
        return np.exp(-x ** shape)
    raise ValueError(f"unknown kernel {kernel!r}")


def background_model(params, r, comp2, edge):
    """B(r) for params [baseline, A1, D1, A2, D2(, n | beta)], kernels zeroed at r = edge."""
    b, a1, d1, a2, d2 = params[:5]
    shape = params[5] if len(params) > 5 else None
    out = b + a1 * (radial_shape(r, d1, "exponential") - radial_shape(edge, d1, "exponential"))
    return out + a2 * (radial_shape(r, d2, comp2, shape) - radial_shape(edge, d2, comp2, shape))


def initial_background(r, y, comp2):
    """Baseline from the tail, exponential amplitude and decay from the head and log-slope
    (MuscleX init_from_wedge); comp2 starts at 0.3x the amplitude and 3x the decay length."""
    n_tail, n_head = max(3, len(y) // 5), max(2, len(y) // 20)
    base = max(min(np.median(y[-n_tail:]), np.percentile(y, 2)), 0.0)
    amp = max(np.median(y[:n_head]) - base, 1e-6)
    above = y - base
    ok = above > 0.02 * amp
    slope = np.polyfit(r[ok], np.log(above[ok]), 1)[0] if ok.sum() >= 5 else 0.0
    decay = -1.0 / slope if slope < 0 else 0.25 * (r[-1] - r[0])
    decay = float(np.clip(decay, 1.0, 0.5 * r[-1]))
    extra = {"powerlaw": [4.0], "stretched": [1.0]}.get(comp2, [])
    return np.array([base, amp, decay, 0.3 * amp, min(3.0 * decay, r[-1])] + extra)


def fit_background(profile, r, comp2, neg_weight, init=None):
    """(background, params): the general background fitted to one profile with the
    oversubtraction (model > data) residuals weighted ``neg_weight`` times.
    ``init`` (params of a similar profile) warm-starts the fit."""
    r_all = np.asarray(r, dtype=float)
    y = np.asarray(profile, dtype=float)
    use = np.isfinite(y) & (y > 0)  # detector gaps and masked pixels read 0
    if use.sum() < 10:
        return np.zeros(len(y)), init
    norm = float(y[use].max())
    r, y = r_all[use], y[use] / norm
    shape_lo, shape_hi = {"powerlaw": ([0.5], [8.0]), "stretched": ([0.1], [3.0])}.get(comp2, ([], []))
    lo = np.array([0.0, 0.0, 1.0, 0.0, 1.0] + shape_lo)
    hi = np.array([np.inf, np.inf, 10.0 * r[-1], np.inf, 10.0 * r[-1]] + shape_hi)
    if init is None:
        x0 = initial_background(r, y, comp2)
    else:  # amplitudes back into this profile's units
        x0 = np.array(init, dtype=float)
        x0[[0, 1, 3]] /= norm
    x0 = np.clip(x0, lo + 1e-9, np.where(np.isfinite(hi), hi - 1e-9, np.inf))

    def residuals(x):
        res = y - background_model(x, r, comp2, r_all[-1])
        return np.where(res < 0, neg_weight * res, res)

    x = least_squares(residuals, x0, bounds=(lo, hi), max_nfev=600).x
    x[[0, 1, 3]] *= norm
    return background_model(x, r_all, comp2, r_all[-1]), x


def fitted_backgrounds(ks, pos, p):
    """Fitted background of every column of ``ks`` (smoothed as detect() sees it), each fit
    warm-started from the previous column's; identical neighbouring columns are fitted once."""
    out = np.empty(ks.shape)
    params, last = None, None
    for t in range(ks.shape[1]):
        col = smoothed(ks[:, t], p["smooth"])
        if last is not None and np.array_equal(col, last):
            out[:, t] = out[:, t - 1]
            continue
        out[:, t], params = fit_background(col, pos, p["bg_comp2"], p["bg_neg_weight"], params)
        last = col
    return out


# ------------------------------------------------------------------ hull or fit, as the parameters say

def column_backgrounds(ks, pos, p):
    """The background each column's own baseline removes (hull or fit), as columns."""
    if p["bg_method"] == "fit":
        return fitted_backgrounds(ks, pos, p)
    return np.column_stack([hull_background(ks[:, t], p["smooth"]) for t in range(ks.shape[1])])


def shared_background(kymo, pos, p):
    """With background = averaged, the background of the mean profile; else None (per column)."""
    if p["background"] != "averaged":
        return None
    return column_backgrounds(kymo.mean(axis=1, keepdims=True), pos, p)[:, 0]


# ------------------------------------------------------------------ noise

def local_noise(flat, win, pct, ref=None, smooth=1):
    """Rolling noise sigma from |second differences| (estimated off the unsmoothed ``ref``)."""
    src = flat if ref is None else ref
    d2 = np.abs(np.diff(src, n=2, prepend=src[0], append=src[-1]))
    sigma = percentile_filter(d2, pct, size=win, mode="nearest") / (norm.ppf((1 + pct / 100) / 2) * np.sqrt(6))
    if ref is not None and smooth > 1:
        sigma = sigma / np.sqrt(smooth)
    positive = sigma[sigma > 0]
    return np.maximum(sigma, max(np.percentile(positive, 5) if positive.size else 0.0, 1e-9))


def profile_noise(profile, p, background=None):
    """(flat, sigma): the baseline-removed profile and its local noise, as detect() sees them."""
    flat = flatten(profile, p["smooth"], background)
    ref = flatten(profile, 1, background) if p["smooth"] > 1 else None
    return flat, local_noise(flat, p["noise_win"], p["noise_pct"], ref, p["smooth"])
