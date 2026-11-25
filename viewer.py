"""The viewer: force/length traces with a time slider, kymographs, tracked peaks, peak heights and backgrounds."""
import csv
import json
import os
import re
import warnings

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, NullFormatter
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QSpinBox,
                               QStyle, QStyleOptionSlider, QToolButton, QVBoxLayout, QWidget)

import multiscale
from background import column_backgrounds, profile_noise, shared_background, smoothed
from kymo_force import column_time, from_trigger, load_traces
from labels import unmatched_names

SLIDER_STYLE = """
QSlider { max-height: 14px; }
QSlider::groove:horizontal { height: 4px; background: #d5d9de; border-radius: 2px; }
QSlider::sub-page:horizontal { height: 4px; background: #8ab4f8; border-radius: 2px; }
QSlider::handle:horizontal { width: 10px; margin: -4px 0; border-radius: 5px; background: #1a73e8; }
"""
STEPS = (("<<", -2), ("<", -1), (">", 1), (">>", 2))  # button, groups moved
GROUPS = (1, 2, 4, 8, 16)  # frame groupings always on offer, whatever the tracker and its scales
COLORS = ("#2b7fff", "#e8590c", "#2f9e44")
BG_D_NM = (80.0, 3.0)  # d range (nm) over which the background panel integrates the background
NM_TICKS = (200, 100, 70, 50, 40, 30, 25, 20, 15, 14.5, 12, 10, 8, 7, 6,
            5.9, 5.1, 5, 4, 3.5, 3, 2.7, 2.5, 2, 1.5, 1)


def d_ticks(ax, q_lo, q_hi, axis="y", min_gap=0.05):
    """Ticks on a q (nm^-1) axis labelled with d = 1/q in nm, thinned so labels do not overlap."""
    d_lo, d_hi = 1.0 / q_hi, 1.0 / q_lo
    cands = sorted({d for d in NM_TICKS} | {float(d) for d in MaxNLocator(6).tick_values(d_lo, d_hi)},
                   reverse=True)  # descending d = ascending q
    ticks, gap = [], min_gap * (q_hi - q_lo)
    for d in cands:
        if d > 0 and q_lo <= 1.0 / d <= q_hi and (not ticks or 1.0 / d - ticks[-1] >= gap):
            ticks.append(1.0 / d)
    (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks, [f"{1.0 / q:.4g}" for q in ticks])


def canvas(height):
    c = FigureCanvasQTAgg(Figure(layout="constrained"))
    c.setMinimumHeight(height)
    return c


def load_multiscale(ms_dir):
    """Tracking arrays (multiscale or U-Net), labels and the analysed per-frame kymograph (rows rmin..rmax, kept frames)."""
    with open(os.path.join(ms_dir, "run.json")) as f:
        run = json.load(f)
    detector = run.get("detector", "multiscale")
    data = np.load(os.path.join(ms_dir, f"{detector}.npz"))
    with open(os.path.join(ms_dir, "labels.csv")) as f:
        labels = {int(r["anchor"]): r["label"] for r in csv.DictReader(f)}
    with open(os.path.splitext(run["kymo"])[0] + ".json") as f:
        pos_start = json.load(f).get("pos_start", 0)
    kymo = np.load(run["kymo"])
    pos = np.arange(kymo.shape[0]) + pos_start
    keep = (pos >= run["rmin"]) & (pos <= run["rmax"])
    frames = data["frames"]
    p = dict(multiscale.DEFAULTS, **{k: run[k] for k in multiscale.TRACK_KEYS if k in run})
    base = kymo[keep][:, frames]
    unmatched = unmatched_names(data["r"][0], labels) if detector == "unet" else {}
    return dict(detector=detector, unmatched=unmatched, scales=data["scales"], frames=frames, r=data["r"], height=data["height"], labels=labels,
                base=base, pos=pos[keep], n_source=kymo.shape[1], p=p, shared=shared_background(base, pos[keep], p),
                rmin=run["rmin"], rmax=run["rmax"], col_of={int(f): i for i, f in enumerate(frames)},
                grouped={}, profiles={}, backgrounds={}, flats={})


def group_options(recs):
    """Grouping scales offered: GROUPS (as far as the longest recording goes) and the tracker's scales."""
    longest = max(len(r["ms"]["frames"]) for r in recs)
    return sorted({g for g in GROUPS if g <= longest} | {int(s) for r in recs for s in r["ms"]["scales"]})


def scale_for(ms, want):
    """(s, k): frames grouped when ``want`` is selected, and the index of the tracker scale nearest
    to s, whose tracks are drawn (U-Net has only the per-frame one)."""
    scales = ms["scales"]
    s = want if want in GROUPS else int(scales[np.argmin(np.abs(scales - want))])
    s = min(s, len(ms["frames"]))
    return s, int(np.argmin(np.abs(scales - s)))


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


LABEL_PT = 7  # font size of the peak labels on the projections


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


def label_peaks(pr, vis):
    """Annotate every peak in view of a projection panel; a label pushed away from its dot by its
    neighbours gets a leader line back to it. ``vis`` = [(x, q, name, height)]; adds to pr["notes"]."""
    ax = pr["ax"]
    (x_lo, x_hi), (y_lo, y_hi) = sorted(ax.get_xlim()), ax.get_ylim()
    vis = [v for v in vis if x_lo <= v[0] <= x_hi and y_lo <= v[1] <= y_hi]
    point = ax.figure.dpi / 72 / max(ax.bbox.width, 1.0)  # one point, in axes widths
    line = min(LABEL_PT * 1.25 * ax.figure.dpi / 72 / max(ax.bbox.height, 1.0), 0.5)  # in axes heights
    to_axes = ax.transScale + ax.transLimits
    home = to_axes.transform([v[:2] for v in vis]).reshape(-1, 2)  # dots in axes fractions (log x too)
    rows = spread_labels(home[:, 1], line, line / 2, 1 - line / 2)
    for (x, y, name, h), (x0, y0), y1 in zip(vis, home, rows):
        clear = home[np.abs(home[:, 1] - y1) < line, 0].max(initial=x0)  # start right of any dot in this row
        moved = abs(y1 - y0) > 0.25 * line or clear > x0
        text = f"{name} {1.0 / y:.2f}" if pr["what"] == "d" else f"{name} {h:.3g}"
        leader = dict(arrowstyle="-", lw=0.5, color="grey", relpos=(0, 0.5), shrinkA=1, shrinkB=2) if moved else None
        pr["notes"].append(ax.annotate(text, (x, y), xytext=(max(clear, x0) + (14 if moved else 5) * point, y1),
                                       textcoords="axes fraction", va="center", fontsize=LABEL_PT,
                                       arrowprops=leader))


class ClickSlider(QSlider):
    """Horizontal slider that jumps to the clicked point (then drags) instead of paging."""

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
            handle = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
            x = event.position().toPoint().x()
            if not handle.contains(event.position().toPoint()):
                self.setValue(QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(), x - groove.x() - handle.width() // 2,
                    max(1, groove.width() - handle.width())))
        super().mousePressEvent(event)


class Viewer(QWidget):
    """Force/length overlay with a time slider; kymographs; tracked peaks (d and height) beside each
    recording's grouped projection at the slider time.  Every position axis is q = r * inv_nm,
    labelled in d (nm), so recordings with different calibrations line up."""

    def __init__(self):
        super().__init__()
        self.force_canvas, self.kymo_canvas = canvas(320), canvas(420)
        self.track_canvas, self.height_canvas = canvas(620), canvas(620)
        self.background_canvas = canvas(520)
        self.sliders, self.time_labels = [], []
        self.peaks_edit = QLineEdit()
        self.peaks_edit.setPlaceholderText("all peaks")
        self.peaks_edit.setToolTip("Comma-separated names or prefixes (M2 = M2, M2L, M2-H-21.2 ...)")
        self.scale_box = QComboBox()
        self.scale_box.setToolTip("Frames averaged (sliding mean) in the kymographs, projections and background. "
                                  "U-Net tracks and heights stay per frame.")
        self.peaks_edit.editingFinished.connect(self.redraw)
        self.scale_box.currentTextChanged.connect(self.redraw)

        controls = QHBoxLayout()
        for label, w in (("Peaks", self.peaks_edit), ("Grouping scale", self.scale_box)):
            controls.addWidget(QLabel(label))
            controls.addWidget(w)
        self.unmatched_box = QCheckBox("Show unmatched tracks")
        self.unmatched_box.setChecked(True)
        self.unmatched_box.setToolTip("U-Net tracks that match no expected peak, named X1, X2, ... from the "
                                      "centre out in each recording (dashed)")
        self.unmatched_box.toggled.connect(lambda _: self.redraw())
        controls.addWidget(self.unmatched_box)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.force_canvas)
        layout.addLayout(self.time_slider())
        layout.addLayout(controls)
        # every section repeats the d range and (beside a projection) the projection x controls;
        # the copies are linked, so applying one updates the rest
        self.zoom_rows, self.xrows, self.xlim, self.xautos = [], [], None, []
        for c in (self.kymo_canvas, self.track_canvas, self.height_canvas, self.background_canvas):
            if c is self.height_canvas:
                row, self.height_smooth = self.smoothing_row(
                    "Smooth U-Net peak heights", "tracks.csv and the projection labels keep the measured heights")
                layout.addLayout(row)
            elif c is self.background_canvas:
                row, self.background_smooth = self.smoothing_row(
                    "Smooth background", "the subtracted background stays the same")
                layout.addLayout(row)
            layout.addWidget(c)
            if c is self.kymo_canvas:  # intensity cutoffs, then the d range on the right; slider below
                row = QHBoxLayout()
                row.addLayout(self.contrast_row())
                row.addLayout(self.range_row(projection=False))
                layout.addLayout(row)
                layout.addLayout(self.time_slider())
            else:
                layout.addLayout(self.time_slider())
                layout.addLayout(self.range_row(projection=True))
        self.recs, self.t_min, self.t_max, self.t_cur, self.dt = [], 0.0, 1.0, 0.0, 1.0
        self.time_lines, self.kymo_lines, self.panel_lines = [], [], []
        self.track_canvas.mpl_connect("resize_event", self.respace_labels)
        self.hide()

    def time_slider(self):
        """A time slider row; all of them are linked, so moving one moves the others."""
        slider = ClickSlider(Qt.Horizontal)
        slider.setStyleSheet(SLIDER_STYLE)
        slider.valueChanged.connect(lambda v, me=slider: self.slide(me, v))
        label = QLabel()
        self.sliders.append(slider)
        self.time_labels.append(label)
        row = QHBoxLayout()
        row.setContentsMargins(6, 0, 6, 0)
        row.addWidget(QLabel("time"))
        buttons = []
        for text, n in STEPS:
            b = QToolButton(text=text, autoRaise=True)
            b.setToolTip(f"{'back' if n < 0 else 'forward'} {abs(n)} group{'s' * (abs(n) > 1)} "
                         "at the grouping scale")
            b.clicked.connect(lambda _=False, n=n: self.step(n))
            buttons.append(b)
        for w in buttons[:2] + [slider] + buttons[2:] + [label]:
            row.addWidget(w)
        return row

    def range_row(self, projection):
        """One section's d range (Enter or "Apply range"; "Show whole" clears it) and, if it has
        projections, their scale and x range (Enter or "Apply" fixes it; "Auto", or changing the
        scale, fits it again and shows the fitted range)."""
        row = QHBoxLayout()
        row.setContentsMargins(6, 0, 6, 0)
        lo, hi, unit = QLineEdit(), QLineEdit(), QComboBox()
        lo.setPlaceholderText("radius from")
        hi.setPlaceholderText("radius to")
        unit.addItems(["d (nm)", "px"])
        unit.currentTextChanged.connect(lambda text: [u.setCurrentText(text) for _, _, u in self.zoom_rows])
        apply_button, whole_button = QPushButton("Apply range"), QPushButton("Show whole")
        apply_button.clicked.connect(lambda: self.apply_zoom(lo, hi))
        whole_button.clicked.connect(self.show_whole)
        self.zoom_rows.append((lo, hi, unit))
        widgets = [("Range", lo), ("to", hi), ("", unit), ("", apply_button), ("", whole_button)]
        for w in (lo, hi):
            w.returnPressed.connect(lambda: self.apply_zoom(lo, hi))
        if projection:
            scale, xlo, xhi = QComboBox(), QLineEdit(), QLineEdit()
            scale.addItems(["log", "linear"])
            xlo.setPlaceholderText("x from")
            xhi.setPlaceholderText("x to")
            xapply, xauto = QPushButton("Apply"), QPushButton("Auto")
            scale.currentTextChanged.connect(self.set_xscale)
            xauto.clicked.connect(lambda: self.set_xlim(None))
            xapply.clicked.connect(lambda: self.set_xlim(self.typed_xlim(xlo, xhi)))
            for w in (xlo, xhi):
                w.returnPressed.connect(lambda: self.set_xlim(self.typed_xlim(xlo, xhi)))
            self.xrows.append((scale, xlo, xhi))
            widgets += [(None, None), ("projections x", scale), ("from", xlo), ("to", xhi),
                        ("", xapply), ("", xauto)]
        for label, w in widgets:
            if w is None:
                row.addStretch()
                continue
            if label:
                row.addWidget(QLabel(label))
            if isinstance(w, QLineEdit):
                w.setMaximumWidth(90)
            row.addWidget(w)
        return row

    def smoothing_row(self, label, what):
        """(row, spin box) smoothing one panel's lines over time (1 frame = off)."""
        box = QSpinBox(minimum=1, maximum=101, singleStep=2, value=1, suffix=" frames")
        box.setSpecialValueText("off")
        box.setKeyboardTracking(False)
        box.setToolTip("Moving mean over this many frames (odd), separately on each side of "
                       f"excluded frames.\nOnly the plotted lines change; {what}.")
        # a centred window is odd: an even value moves up one, which triggers the redraw
        box.valueChanged.connect(lambda v: self.redraw() if v % 2 else box.setValue(v + 1))
        row = QHBoxLayout()
        row.setContentsMargins(6, 0, 6, 0)
        row.addWidget(QLabel(label))
        row.addWidget(box)
        row.addStretch()
        return row, box

    def contrast_row(self):
        """Kymograph intensity scale (log(1+I) or counts) and cutoffs in that scale, as on the
        colorbar: values at or below "min" are white, at or above "max" black.  An empty field is
        automatic (per-recording percentile)."""
        row = QHBoxLayout()
        row.setContentsMargins(6, 0, 6, 0)
        self.vmin_edit, self.vmax_edit = QLineEdit(), QLineEdit()
        self.intensity_scale = QComboBox()
        self.intensity_scale.addItems(["log(1+I)", "counts"])
        self.intensity_scale.currentTextChanged.connect(self.set_intensity_scale)
        apply_button, auto_button = QPushButton("Apply"), QPushButton("Auto")
        apply_button.clicked.connect(self.redraw_kymos)
        auto_button.clicked.connect(lambda: [w.clear() for w in (self.vmin_edit, self.vmax_edit)]
                                    or self.redraw_kymos())
        row.addWidget(QLabel("Kymograph intensity"))
        row.addWidget(self.intensity_scale)
        row.addWidget(QLabel("min"))
        for w, label in ((self.vmin_edit, "to max"), (self.vmax_edit, None)):
            w.setPlaceholderText("auto")
            w.setMaximumWidth(90)
            w.returnPressed.connect(self.redraw_kymos)
            row.addWidget(w)
            if label:
                row.addWidget(QLabel(label))
        row.addWidget(apply_button)
        row.addWidget(auto_button)
        row.addStretch()
        return row

    def set_intensity_scale(self, text):
        """Convert any typed cutoffs to the new scale, then redraw the kymographs."""
        for w in (self.vmin_edit, self.vmax_edit):
            try:
                v = float(w.text())
            except ValueError:
                continue
            w.setText(f"{np.expm1(v) if text == 'counts' else np.log1p(max(v, 0.0)):.4g}")
        self.redraw_kymos()

    def intensity_limits(self, auto_lo, auto_hi):
        """(vmin, vmax): the typed cutoffs, each falling back to its automatic value."""
        vals = []
        for w, auto in ((self.vmin_edit, auto_lo), (self.vmax_edit, auto_hi)):
            try:
                vals.append(float(w.text()))
            except ValueError:
                vals.append(auto)
        return tuple(vals) if vals[0] < vals[1] else (auto_lo, auto_hi)

    def redraw_kymos(self):
        if self.recs:
            self.draw_kymos()
            self.set_time(self.t_cur)

    def apply_zoom(self, lo, hi):
        """Copy one row's d range to every row, then redraw."""
        for a, b, _ in self.zoom_rows:
            a.setText(lo.text())
            b.setText(hi.text())
        self.redraw()

    def xscale(self):
        return self.xrows[0][0].currentText()

    def set_xscale(self, text):
        for scale, _, _ in self.xrows:
            scale.blockSignals(True)
            scale.setCurrentText(text)
            scale.blockSignals(False)
        self.set_xlim(None)

    def typed_xlim(self, lo, hi):
        """The typed (from, to), or None (auto) if unusable on the current scale."""
        try:
            a, b = float(lo.text()), float(hi.text())
        except ValueError:
            return None
        return (a, b) if a < b and (a > 0 or self.xscale() == "linear") else None

    def set_xlim(self, lim):
        self.xlim = lim
        if lim is not None:
            for _, a, b in self.xrows:
                a.setText(f"{lim[0]:.4g}")
                b.setText(f"{lim[1]:.4g}")
        self.redraw()

    def step(self, n):
        """Move every slider n groups (of the grouping scale's frames)."""
        s = int(self.scale_box.currentText() or 1)
        slider = self.sliders[0]
        slider.setValue(min(max(slider.value() + n * s, 0), slider.maximum()))

    def slide(self, moved, v):
        for s in self.sliders:
            if s is not moved:
                s.blockSignals(True)
                s.setValue(v)
                s.blockSignals(False)
        self.set_time(self.t_min + (v + 0.5) * self.dt)  # slider value = frame

    def load(self, results):
        self.recs = []
        for (label, npy, ddf, ms_dir), color in zip(results, COLORS):
            with open(os.path.splitext(npy)[0] + ".json") as f:
                meta = json.load(f)
            d, traces = load_traces(ddf) if ddf else (None, [])
            t0, _, raw_dt = column_time(meta, d)[:3]
            traces = from_trigger(t0, traces, [])[0]
            self.recs.append(dict(label=label, color=color, traces=traces, t0=0.0, raw_dt=raw_dt,
                                  inv=meta["inv_nm_per_pixel"], ms=load_multiscale(ms_dir)))
        self.refresh()

    def remove(self, i):
        """Drop recording ``i``, keeping the time and grouping scale; the others are not reloaded."""
        del self.recs[i]
        self.refresh(self.t_cur)

    def refresh(self, t=None):
        """Fit the sliders and grouping scales to the recordings and redraw, at time ``t`` and the
        current scale if given, else at the start and the finest scale.  No recordings: hide."""
        if not self.recs:
            self.hide()
            return
        for w in (self.unmatched_box, self.height_smooth):
            w.setEnabled(any(r["ms"]["detector"] == "unet" for r in self.recs))
        self.t_min = min(r["t0"] for r in self.recs)
        self.t_max = max(r["t0"] + r["ms"]["n_source"] * r["raw_dt"] for r in self.recs)
        self.dt = min(r["raw_dt"] for r in self.recs)
        last = max(0, round((self.t_max - self.t_min) / self.dt) - 1)
        value = 0 if t is None else min(max(int((t - self.t_min) / self.dt), 0), last)
        for slider in self.sliders:
            slider.blockSignals(True)
            slider.setMaximum(last)
            slider.setValue(value)
            slider.blockSignals(False)
        scales = [str(s) for s in group_options(self.recs)]
        current = self.scale_box.currentText()
        self.scale_box.blockSignals(True)
        self.scale_box.clear()
        self.scale_box.addItems(scales)
        self.scale_box.setCurrentText(current if t is not None and current in scales
                                      else scales[0])  # the finest: per frame
        self.scale_box.blockSignals(False)
        self.draw_force()
        self.show()
        self.redraw(self.t_min + (value + 0.5) * self.dt)

    def show_whole(self):
        for lo, hi, _ in self.zoom_rows:
            lo.clear()
            hi.clear()
        self.redraw()

    def q_range(self):
        """Shared (q_lo, q_hi): the zoom fields, else every recording's radius from/to."""
        lo, hi, unit = self.zoom_rows[0]  # the rows are kept in step
        try:
            a, b = float(lo.text()), float(hi.text())
        except ValueError:
            return self.q_range_default()
        if unit.currentText() == "px":
            qs = [v * r["inv"] for r in self.recs for v in (a, b)]
        else:
            qs = [1.0 / v for v in (a, b) if v > 0]
        return (min(qs), max(qs)) if len(qs) > 1 and min(qs) < max(qs) else self.q_range_default()

    def q_range_default(self):
        return (min(r["ms"]["rmin"] * r["inv"] for r in self.recs),
                max(r["ms"]["rmax"] * r["inv"] for r in self.recs))

    def redraw(self, t=None):
        if not self.recs:
            return
        self.q_lo, self.q_hi = self.q_range()
        self.s_want = int(self.scale_box.currentText() or 1)
        for r in self.recs:
            ms = r["ms"]
            r["s"], r["k"] = s, _ = scale_for(ms, self.s_want)
            if s not in ms["grouped"]:  # sliding mean of s frames, as the tracker saw it
                g = ms["grouped"][s] = multiscale.bin_time(ms["base"], s)
                cols = range(g.shape[1])
                prof = ms["profiles"][s] = np.column_stack([smoothed(g[:, t], ms["p"]["smooth"])
                                                            for t in cols])
                ms["backgrounds"][s] = column_backgrounds(g, ms["pos"], ms["p"])
                ms["flats"][s] = prof - (ms["backgrounds"][s] if ms["shared"] is None else ms["shared"][:, None])
            r["shown"] = self.shown_anchors(r, r["k"])
            r["times"] = r["t0"] + (ms["frames"] + 0.5) * r["raw_dt"]
        self.draw_kymos()
        self.xautos = []
        self.draw_panels()
        self.draw_background()
        if self.xlim is None and self.xautos:  # auto: show the fitted range (union over all panels)
            for _, lo, hi in self.xrows:
                lo.setText(f"{min(a for a, _ in self.xautos):.4g}")
                hi.setText(f"{max(b for _, b in self.xautos):.4g}")
        self.set_time(self.t_cur if t is None or isinstance(t, str) else t)

    def draw_force(self):
        fig = self.force_canvas.figure
        fig.clear()
        keys = [k for k in ("force", "length") if any(tr["key"] == k for r in self.recs for tr in r["traces"])]
        self.time_lines = []
        if not keys:
            fig.text(0.5, 0.5, "no .ddf loaded", ha="center", va="center", color="grey")
            return
        axes = fig.subplots(len(keys), 1, sharex=True, squeeze=False)[:, 0]
        for ax, key in zip(axes, keys):
            for r in self.recs:
                for tr in r["traces"]:
                    if tr["key"] == key:
                        ax.plot(tr["t"], tr["y"], lw=1.2, color=r["color"], label=r["label"])
                        ylabel = f"{key} ({tr['units']})"
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", lw=0.6, color="#e9e9e9")
            self.time_lines.append(ax.axvline(self.t_min, color="#d93025", lw=1))
        axes[0].legend(fontsize=8, frameon=False)
        axes[-1].set_xlabel("time from trigger (s)")
        axes[-1].set_xlim(self.t_min, self.t_max)

    def draw_kymos(self):
        fig = self.kymo_canvas.figure
        fig.clear()
        axes = fig.subplots(1, len(self.recs), sharey=True, squeeze=False)[0]
        self.kymo_lines = []
        for ax, r in zip(axes, self.recs):
            ms = r["ms"]
            disp = np.full((len(ms["pos"]), ms["n_source"]), np.nan)  # dropped frames stay blank
            counts = self.intensity_scale.currentText() == "counts"
            raw = np.clip(ms["grouped"][r["s"]], 0, None)
            disp[:, ms["frames"]] = raw if counts else np.log1p(raw)
            q = ms["pos"] * r["inv"]
            inside = (q >= self.q_lo) & (q <= self.q_hi)
            lo, hi = self.intensity_limits(
                *np.nanpercentile(disp[inside] if inside.any() else disp, (1.0, 99.5)))
            im = ax.imshow(disp, origin="lower", aspect="auto", cmap="gray_r", vmin=lo, vmax=hi,
                           interpolation="nearest",
                           extent=(r["t0"], r["t0"] + ms["n_source"] * r["raw_dt"], q[0], q[-1]))
            for a, _ in r["shown"]:  # the tracks, faint so the kymograph stays readable
                rr = ms["r"][r["k"], a]
                ok = np.isfinite(rr)
                ax.plot(*broken(ms["frames"][ok], r["times"][ok], rr[ok] * r["inv"]),
                        **(dict(color="#7a3fb5", ls="--") if a in ms["unmatched"] else dict(color=r["color"])),
                        lw=0.8, alpha=0.45)
            bar = fig.colorbar(im, ax=ax, pad=0.13, fraction=0.05)  # own scale, clear of the px axis
            bar.set_label("I (counts)" if counts else "log(1+I)", fontsize=8)
            bar.ax.tick_params(labelsize=7)
            ax.set_title(f"{r['label']}  (s = {r['s']})", color=r["color"], fontsize=9)
            ax.set_xlabel("time (s)")
            ax.set_xlim(self.t_min, self.t_max)
            ax.set_ylim(self.q_lo, self.q_hi)
            inv = r["inv"]
            ax.secondary_yaxis("right", functions=(lambda v, s=inv: v / s, lambda p, s=inv: p * s)) \
              .tick_params(labelsize=7)
            self.kymo_lines.append(ax.axvline(self.t_min, color="#d93025", lw=1))
        d_ticks(axes[0], self.q_lo, self.q_hi)
        axes[0].set_ylabel("d (nm)   [right: px]")

    def shown_anchors(self, r, k):
        """[(anchor, name)] whose name matches the peaks field and whose median q is in range;
        the unmatched tracks (X1, X2, ...) only with their box checked."""
        names = dict(r["ms"]["labels"])
        if self.unmatched_box.isChecked():
            names.update(r["ms"]["unmatched"])
        tokens = [t.strip() for t in self.peaks_edit.text().split(",") if t.strip()]
        wanted = re.compile("|".join(rf"{re.escape(t)}(?!\d)" for t in tokens)) if tokens else None
        out = []
        for a, name in sorted(names.items()):
            rr = r["ms"]["r"][k, a]
            if (wanted and not wanted.match(name)) or not np.isfinite(rr).any():
                continue
            if self.q_lo <= np.nanmedian(rr) * r["inv"] <= self.q_hi:
                out.append((a, name))
        return out

    def panel_row(self, c):
        """A wide time panel beside one narrow vertical (q up) panel per recording, sharing q."""
        n = len(self.recs)
        fig = c.figure
        with warnings.catch_warnings():  # tearing down a shared log axis warns harmlessly
            warnings.simplefilter("ignore", UserWarning)
            fig.clear()
        gs = fig.add_gridspec(1, 1 + n, width_ratios=[3.2] + [1.0] * n)
        left = fig.add_subplot(gs[0, 0])
        first = fig.add_subplot(gs[0, 1], sharey=left if c is self.track_canvas else None)
        right = [first] + [fig.add_subplot(gs[0, 1 + i], sharey=first) for i in range(1, n)]
        if c is not self.track_canvas:  # the left panel has no q axis: label the first projection
            first.set_ylim(self.q_lo, self.q_hi)
            d_ticks(first, self.q_lo, self.q_hi)
            first.set_ylabel("d (nm)", fontsize=8)
        return left, right

    def projection(self, r, ax, what, values):
        """Set up one projection panel, log or linear x fitted to ``values`` unless the section has a
        typed range; returns its artists."""
        values = values[np.isfinite(values)].ravel()
        top = float(np.percentile(values, 99.9)) if values.size else 1.0  # ignore the odd hot pixel
        room = (1.5, 1.05) if what == "bg" else (8.0, 1.5)  # log, linear: room for the peak labels
        log = self.xscale() == "log"
        if log:
            positive = values[values > 0]
            floor = float(np.percentile(positive, 1)) * (0.8 if what == "bg" else 1.0) if positive.size else 1e-3
            auto = (floor, max(top * room[0], floor * 10))
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())  # minor labels crowd a narrow panel
        else:
            auto = (min(0.0, float(np.percentile(values, 0.1))) if values.size else 0.0,
                    max(top, 1e-9) * room[1])
        self.xautos.append(auto)
        lim = self.xlim or auto
        floor = lim[0] if log else None  # log: clip to the left edge
        ax.set_xlim(*lim)
        ax.set_title(f"{r['label']}\n(s = {r['s']})", color=r["color"], fontsize=8)
        ax.set_xlabel("I − baseline" if what != "bg" else "I", fontsize=8)
        ax.tick_params(labelsize=7, labelleft=ax.get_ylabel() != "")
        return dict(ax=ax, what=what, floor=floor, notes=[],
                    line=ax.plot([], [], lw=1, color=r["color"])[0],
                    extra=ax.plot([], [], lw=1, color="grey", ls=":" if what != "bg" else "--")[0],
                    marks=ax.scatter([], [], s=28, color=r["color"], edgecolors="black", linewidths=0.6, zorder=5),
                    miss=ax.text(0.5, 0.5, "", transform=ax.transAxes, ha="center", color="grey", fontsize=9))

    def draw_panels(self):
        """Tracks (q, labelled in d) and, equally sized below, their heights; each beside one grouped
        projection per recording (background removed as the tracker did, q up), the detected peaks
        marked with their d above and their height below; dotted: the acceptance threshold."""
        (ax_t, ax_p), (ax_h, ax_hp) = self.panel_row(self.track_canvas), self.panel_row(self.height_canvas)
        s_want = self.s_want
        heights = []
        for r, ax, axh in zip(self.recs, ax_p, ax_hp):
            ms, k = r["ms"], r["k"]
            times = r["times"]
            for a, name in r["shown"]:
                rr, h = ms["r"][k, a], ms["height"][k, a]
                ok = np.isfinite(rr)
                if ms["detector"] == "unet":
                    h = smooth_series(h, ms["frames"], self.height_smooth.value())
                hh = np.where(h[ok] > 0, h[ok], np.nan)
                heights.append(hh[np.isfinite(hh)])
                for axis, y in ((ax_t, rr[ok] * r["inv"]), (ax_h, hh)):
                    axis.plot(*broken(ms["frames"][ok], times[ok], y), lw=1.3, color=r["color"],
                              ls="--" if a in ms["unmatched"] else "-")
                fin = np.flatnonzero(np.isfinite(hh))
                ax_t.text(times[ok][-1], rr[ok][-1] * r["inv"], f" {name}", fontsize=7, color=r["color"],
                          va="center", clip_on=True)
                if len(fin):
                    ax_h.text(times[ok][fin[-1]], hh[fin[-1]], f" {name}", fontsize=7,
                              color=r["color"], va="center", clip_on=True)
            p, n_s = ms["p"], len(ms["scales"])  # acceptance SNR at this scale, relaxed coarse to fine
            r["thr"] = p["snr_coarse"] + (p["snr_fine"] - p["snr_coarse"]) * k / max(n_s - 1, 1)
            q = ms["pos"] * r["inv"]
            vals = ms["flats"][r["s"]][(q >= self.q_lo) & (q <= self.q_hi)]
            r["proj"] = [self.projection(r, ax, "d", vals), self.projection(r, axh, "height", vals)]
        unet_run = any(r["ms"]["detector"] == "unet" for r in self.recs)
        for pr in self.recs[0]["proj"] if not unet_run else ():
            pr["ax"].legend([pr["extra"]], ["noise × SNR threshold"], fontsize=6, loc="lower right",
                            frameon=False)

        ax_t.set_ylim(self.q_lo, self.q_hi)
        d_ticks(ax_t, self.q_lo, self.q_hi)
        ax_t.set_ylabel("d (nm)", fontsize=9)
        ax_t.set_title(("U-Net tracks (per frame)" if unet_run else f"tracked peaks at grouping scale {s_want}"), fontsize=9)
        ax_h.set_yscale("log")
        heights = np.concatenate(heights) if heights else np.empty(0)
        if heights.size:
            ax_h.set_ylim(heights.min() / 2.0, heights.max() * 2.0)
        ax_h.set_ylabel("peak height above baseline", fontsize=9)
        ax_h.set_title(("U-Net peak heights (per frame)" + (f" (mean of {self.height_smooth.value()} frames)"
                                               if self.height_smooth.value() > 1 else "") if unet_run else f"peak heights at grouping scale {s_want}"), fontsize=9)
        for axis in (ax_t, ax_h):
            axis.set_xlabel("time (s)", fontsize=9)
            axis.tick_params(labelsize=8)
            axis.set_xlim(self.t_min, self.t_max + 0.12 * (self.t_max - self.t_min))  # room for labels
            axis.grid(axis="y", lw=0.6, color="#e9e9e9")
        if not any(r["shown"] for r in self.recs):
            for axis in (ax_t, ax_h):
                axis.text(0.5, 0.5, "no tracks to show in range", transform=axis.transAxes,
                          ha="center", color="grey")
        self.panel_lines = [axis.axvline(self.t_min, color="#d93025", lw=1) for axis in (ax_t, ax_h)]

    def draw_background(self):
        """The background over time (integrated over q between d = 80 and 3 nm, from each grouped
        profile's own hull or fit) beside each recording's grouped profile at the slider time with the
        background the tracker subtracted from it (dashed)."""
        ax_b, ax_r = self.panel_row(self.background_canvas)
        modes = set()
        for r, ax in zip(self.recs, ax_r):
            ms, s = r["ms"], r["s"]
            q = ms["pos"] * r["inv"]
            inside = (q >= self.q_lo) & (q <= self.q_hi)
            inside = inside if inside.any() else np.ones_like(inside)
            band = (q >= 1.0 / BG_D_NM[0]) & (q <= 1.0 / BG_D_NM[1])  # area under the background
            area = smooth_series(np.trapezoid(ms["backgrounds"][s][band], q[band], axis=0), ms["frames"],
                                 self.background_smooth.value())
            ax_b.plot(*broken(ms["frames"], r["times"], area),
                      lw=1.3, color=r["color"], label=r["label"])
            if ms["shared"] is not None:
                ax_b.axhline(np.trapezoid(ms["shared"][band], q[band]), color=r["color"], ls="--", lw=1,
                             label=f"{r['label']}: averaged profile's (subtracted)")
            p = ms["p"]
            modes.add(f"{'fit, ' + p['bg_comp2'] if p['bg_method'] == 'fit' else 'hull'}, {p['background']}")
            r["proj"].append(self.projection(r, ax, "bg", ms["profiles"][s][inside]))
        pr = self.recs[0]["proj"][-1]
        pr["ax"].legend([pr["extra"]], ["subtracted background"], fontsize=6, loc="lower right", frameon=False)
        ax_b.set_ylabel(f"background area, d {BG_D_NM[0]:g}–{BG_D_NM[1]:g} nm (I·nm⁻¹)", fontsize=9)
        ax_b.set_title(f"background at grouping scale {self.s_want}  "
                       f"(tracker subtracts: {', '.join(sorted(modes))})"
                       + (f"  (mean of {self.background_smooth.value()} frames)"
                          if self.background_smooth.value() > 1 else ""), fontsize=9)
        ax_b.set_xlabel("time (s)", fontsize=9)
        ax_b.tick_params(labelsize=8)
        ax_b.set_xlim(self.t_min, self.t_max + 0.12 * (self.t_max - self.t_min))
        ax_b.grid(axis="y", lw=0.6, color="#e9e9e9")
        ax_b.legend(fontsize=7, frameon=False)
        self.panel_lines.append(ax_b.axvline(self.t_min, color="#d93025", lw=1))

    def respace_labels(self, _event):
        """The label spacing depends on the panel height; a resize can come before the first draw."""
        if self.recs and all("proj" in r for r in self.recs):
            self.set_time(self.t_cur)

    def set_time(self, t):
        if not self.recs:
            return
        self.t_cur = t
        for line in self.time_lines + self.kymo_lines + self.panel_lines:
            line.set_xdata([t, t])
        for r in self.recs:
            ms, s = r["ms"], r["s"]
            col = ms["col_of"].get(int(np.floor((t - r["t0"]) / r["raw_dt"])))
            vis = []  # (subtracted intensity at the peak, q, name, tracked height)
            if col is not None:
                bg = ms["backgrounds"][s][:, col] if ms["shared"] is None else ms["shared"]
                fitted = ms["shared"] is None and ms["p"]["bg_method"] == "fit"  # else as the tracker did
                flat, sigma = profile_noise(ms["grouped"][s][:, col], ms["p"],
                                            bg if fitted else ms["shared"])
                prof = ms["profiles"][s][:, col]
                for a, name in r["shown"]:
                    rr = ms["r"][r["k"], a, col]
                    if np.isfinite(rr) and self.q_lo <= rr * r["inv"] <= self.q_hi:
                        vis.append((float(np.interp(rr, ms["pos"], flat)), rr * r["inv"], name,
                                    float(ms["height"][r["k"], a, col])))
            q = ms["pos"] * r["inv"]
            for pr in r["proj"]:
                for note in pr["notes"]:
                    note.remove()
                pr["notes"].clear()
                pr["miss"].set_text("" if col is not None else "frame dropped\nor outside")
                if col is None:
                    for artist in (pr["line"], pr["extra"]):
                        artist.set_data([], [])
                    pr["marks"].set_offsets(np.empty((0, 2)))
                    continue
                clip = (lambda v, f=pr["floor"]: np.clip(v, f, None)) if pr["floor"] is not None else (lambda v: v)
                if pr["what"] == "bg":
                    pr["line"].set_data(clip(prof), q)
                    pr["extra"].set_data(clip(bg), q)
                    continue
                pr["line"].set_data(clip(flat), q)
                if ms["detector"] != "unet":  # the U-Net has no SNR threshold
                    pr["extra"].set_data(clip(r["thr"] * sigma), q)
                pr["marks"].set_offsets(np.array([(clip(x), y) for x, y, _, _ in vis]).reshape(-1, 2))
                label_peaks(pr, [(clip(x), y, name, h) for x, y, name, h in vis])
        for label in self.time_labels:
            label.setText(f"{t:.4f} s")
        for c in (self.force_canvas, self.kymo_canvas, self.track_canvas, self.height_canvas,
                  self.background_canvas):
            c.draw_idle()
