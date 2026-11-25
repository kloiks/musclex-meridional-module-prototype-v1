"""Plotting helpers of the viewer (matplotlib only, no Qt)."""
import warnings

import numpy as np
from matplotlib.ticker import MaxNLocator, NullFormatter

TIME_COLOR = "#d93025"
LABEL_PT = 7  # font size of the peak labels on the projections
NM_TICKS = (200, 100, 70, 50, 40, 30, 25, 20, 15, 14.5, 12, 10, 8, 7, 6,
            5.9, 5.1, 5, 4, 3.5, 3, 2.7, 2.5, 2, 1.5, 1)


def d_ticks(ax, q_lo, q_hi, min_gap=0.05):
    """Ticks on a vertical q (nm^-1) axis labelled with d = 1/q in nm, thinned so labels do not overlap."""
    d_lo, d_hi = 1.0 / q_hi, 1.0 / q_lo
    cands = sorted({d for d in NM_TICKS} | {float(d) for d in MaxNLocator(6).tick_values(d_lo, d_hi)},
                   reverse=True)  # descending d = ascending q
    ticks, gap = [], min_gap * (q_hi - q_lo)
    for d in cands:
        if d > 0 and q_lo <= 1.0 / d <= q_hi and (not ticks or 1.0 / d - ticks[-1] >= gap):
            ticks.append(1.0 / d)
    ax.set_yticks(ticks, [f"{1.0 / q:.4g}" for q in ticks])


def broken(frames, x, y):
    """Insert NaNs where the frames jump, so a line is not drawn across a gap."""
    cut = np.flatnonzero(np.diff(frames) > 1) + 1
    return np.insert(x, cut, np.nan), np.insert(y, cut, np.nan)


def smooth_series(h, frames, window):
    """Centred moving mean of one value per kept frame over ``window`` frames, NaNs (not detected)
    skipped and kept; never averaged across excluded frames."""
    out = h.copy()
    half = window // 2
    for seg in np.split(np.arange(len(h)), np.flatnonzero(np.diff(frames) > 1) + 1):
        y = h[seg]
        for i in np.flatnonzero(np.isfinite(y)) if half else ():
            out[seg[i]] = np.nanmean(y[max(i - half, 0):i + half + 1])
    return out


def time_line(ax):
    """The line that follows the time slider."""
    return ax.axvline(0.0, color=TIME_COLOR, lw=1)


def style_time_axis(ax, t_max):
    ax.set_xlabel("time (s)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0.0, t_max + 0.12 * t_max)  # room for labels
    ax.grid(axis="y", lw=0.6, color="#e9e9e9")


def panel_row(fig, n, q_range, share_q):
    """(left, right): a wide time panel beside n narrow vertical (q up) panels that share q, with the
    time panel too if ``share_q``."""
    with warnings.catch_warnings():  # tearing down a shared log axis warns harmlessly
        warnings.simplefilter("ignore", UserWarning)
        fig.clear()
    gs = fig.add_gridspec(1, 1 + n, width_ratios=[3.2] + [1.0] * n)
    left = fig.add_subplot(gs[0, 0])
    first = fig.add_subplot(gs[0, 1], sharey=left if share_q else None)
    right = [first] + [fig.add_subplot(gs[0, 1 + i], sharey=first) for i in range(1, n)]
    if not share_q:  # the left panel has no q axis: label the first projection
        first.set_ylim(*q_range)
        d_ticks(first, *q_range)
        first.set_ylabel("d (nm)", fontsize=8)
    return left, right


def spread_labels(ys, min_gap, lo=0.0, hi=1.0):
    """Label positions at least ``min_gap`` apart inside [lo, hi], in the order of ``ys`` and moved
    as little as possible (least squares); a gap that cannot fit is shrunk until the labels do."""
    ys = np.asarray(ys, float)
    n = len(ys)
    if n == 0:
        return ys
    gap = min(min_gap, (hi - lo) / max(n - 1, 1))
    order = np.argsort(ys, kind="stable")
    blocks = []  # isotonic regression of y - i * gap: overlapping labels merge into evenly spaced runs
    for z in ys[order] - gap * np.arange(n):
        blocks.append([z, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            z2, w2 = blocks.pop()
            blocks[-1] = [(blocks[-1][0] * blocks[-1][1] + z2 * w2) / (blocks[-1][1] + w2), blocks[-1][1] + w2]
    z = np.repeat([b[0] for b in blocks], [b[1] for b in blocks])
    out = np.empty(n)
    out[order] = np.clip(z, lo, max(hi - gap * (n - 1), lo)) + gap * np.arange(n)
    return out


class Projection:
    """A recording's profile at the slider time (intensity across, q up).  ``what``: "d" or "height"
    (background-subtracted, its peaks labelled with that), or "bg" (with its background)."""

    def __init__(self, ax, rec, what, values, log, xlim):
        """x is log or linear, from ``xlim`` or (None) fitted to ``values``."""
        self.ax, self.what, self.notes = ax, what, []
        self.auto = self.fitted_xlim(values[np.isfinite(values)].ravel(), log)
        if log:
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())  # minor labels crowd a narrow panel
        lo, hi = xlim or self.auto
        self.floor = lo if log else None  # log: clip to the left edge
        ax.set_xlim(lo, hi)
        ax.set_title(f"{rec.label}\n(s = {rec.s})", color=rec.color, fontsize=8)
        ax.set_xlabel("I" if what == "bg" else "I − baseline", fontsize=8)
        ax.tick_params(labelsize=7, labelleft=ax.get_ylabel() != "")
        self.line = ax.plot([], [], lw=1, color=rec.color)[0]
        self.extra = ax.plot([], [], lw=1, color="grey", ls="--" if what == "bg" else ":")[0]
        self.marks = ax.scatter([], [], s=28, color=rec.color, edgecolors="black", linewidths=0.6, zorder=5)
        self.message = ax.text(0.5, 0.5, "", transform=ax.transAxes, ha="center", color="grey", fontsize=9)

    def fitted_xlim(self, values, log):
        bg = self.what == "bg"
        top = float(np.percentile(values, 99.9)) if values.size else 1.0  # ignore the odd hot pixel
        if not log:
            low = min(0.0, float(np.percentile(values, 0.1))) if values.size else 0.0
            return low, max(top, 1e-9) * (1.05 if bg else 1.5)  # room for the peak labels
        positive = values[values > 0]
        floor = float(np.percentile(positive, 1)) * (0.8 if bg else 1.0) if positive.size else 1e-3
        return floor, max(top * (1.5 if bg else 8.0), floor * 10)

    def clip(self, x):
        return x if self.floor is None else np.clip(x, self.floor, None)

    def clear(self, message=""):
        for note in self.notes:
            note.remove()
        self.notes = []
        for line in (self.line, self.extra):
            line.set_data([], [])
        self.marks.set_offsets(np.empty((0, 2)))
        self.message.set_text(message)

    def show(self, q, x, extra=None, peaks=()):
        """The profile x(q), a second line (the threshold or the background) and the labelled peaks."""
        self.clear()
        self.line.set_data(self.clip(x), q)
        if extra is not None:
            self.extra.set_data(self.clip(extra), q)
        peaks = [p._replace(x=self.clip(p.x)) for p in peaks]
        self.marks.set_offsets(np.array([(p.x, p.q) for p in peaks]).reshape(-1, 2))
        self.label_peaks(peaks)

    def label_peaks(self, peaks):
        """Label every peak in view; a label pushed away from its dot by its neighbours gets a leader
        line back to it."""
        ax = self.ax
        (x_lo, x_hi), (y_lo, y_hi) = sorted(ax.get_xlim()), ax.get_ylim()
        peaks = [p for p in peaks if x_lo <= p.x <= x_hi and y_lo <= p.q <= y_hi]
        point = ax.figure.dpi / 72 / max(ax.bbox.width, 1.0)  # one point, in axes widths
        line = min(LABEL_PT * 1.25 * ax.figure.dpi / 72 / max(ax.bbox.height, 1.0), 0.5)  # in axes heights
        to_axes = ax.transScale + ax.transLimits
        home = to_axes.transform([(p.x, p.q) for p in peaks]).reshape(-1, 2)  # dots in axes fractions (log x too)
        rows = spread_labels(home[:, 1], line, line / 2, 1 - line / 2)
        for p, (x0, y0), y1 in zip(peaks, home, rows):
            clear = home[np.abs(home[:, 1] - y1) < line, 0].max(initial=x0)  # start right of any dot in this row
            moved = abs(y1 - y0) > 0.25 * line or clear > x0
            text = f"{p.name} {1.0 / p.q:.2f}" if self.what == "d" else f"{p.name} {p.height:.3g}"
            leader = dict(arrowstyle="-", lw=0.5, color="grey", relpos=(0, 0.5), shrinkA=1, shrinkB=2)
            self.notes.append(ax.annotate(
                text, (p.x, p.q), xytext=(max(clear, x0) + (14 if moved else 5) * point, y1),
                textcoords="axes fraction", va="center", fontsize=LABEL_PT, arrowprops=leader if moved else None))
