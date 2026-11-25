"""The viewer: force/length traces with a time slider, kymographs, tracked peaks, peak heights and backgrounds."""
import re

import numpy as np
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from controls import ContrastRow, RangeRows, SmoothingRow, TimeSliders, canvas
from plots import Projection, broken, d_ticks, panel_row, smooth_series, style_time_axis, time_line
from recording import group_options, load_recording

COLORS = ("#2b7fff", "#e8590c", "#2f9e44")
UNMATCHED_COLOR = "#7a3fb5"
SMOOTH_FRAMES = 10  # default window of the plotted peak heights and background
BG_D_NM = (80.0, 3.0)  # d range (nm) over which the background panel integrates the background


class Viewer(QWidget):
    """Force/length overlay with a time slider; kymographs; tracked peaks (d and height) beside each
    recording's grouped projection at the slider time.  Every position axis is q = r * inv,
    labelled in d (nm), so recordings with different calibrations line up."""

    def __init__(self):
        super().__init__()
        self.recs = []          # Recordings
        self.shown = []         # per recording: the [(track, name)] drawn
        self.projections = []   # per recording: its three Projections (d, height, background)
        self.force_lines, self.kymo_lines, self.panel_lines = [], [], []  # they follow the slider
        self.t_cur, self.t_max, self.dt = 0.0, 1.0, 1.0
        self.make_controls()
        self.make_layout()
        self.track_canvas.mpl_connect("resize_event", self.respace_labels)
        self.hide()

    # ---------------------------------------------------------------- controls and layout

    def make_controls(self):
        self.force_canvas, self.kymo_canvas = canvas(320), canvas(420)
        self.track_canvas, self.height_canvas = canvas(620), canvas(620)
        self.background_canvas = canvas(520)
        self.peaks_edit = QLineEdit(placeholderText="all peaks")
        self.peaks_edit.setToolTip("Comma-separated names or prefixes (M2 = M2, M2L, M2-H-21.2 ...; M1 is not M10; "
                                   "Tn = every Tn). Capitals matter.")
        self.peaks_edit.editingFinished.connect(self.redraw)
        self.scale_box = QComboBox()
        self.scale_box.setToolTip("Frames averaged (sliding mean) in the kymographs, projections and background. "
                                  "U-Net tracks and heights stay per frame.")
        self.scale_box.currentTextChanged.connect(lambda _: self.redraw())
        self.unmatched_box = QCheckBox("Show unmatched tracks", checked=True)
        self.unmatched_box.setToolTip("U-Net tracks that match no expected peak, named X1, X2, ... from the "
                                      "centre out in each recording (dashed)")
        self.unmatched_box.toggled.connect(lambda _: self.redraw())

        self.time, self.range, self.contrast = TimeSliders(), RangeRows(), ContrastRow()
        self.height_smooth = SmoothingRow("Smooth peak heights",
                                          "tracks.csv and the projection labels keep the measured heights",
                                          SMOOTH_FRAMES)
        self.background_smooth = SmoothingRow("Smooth background", "the subtracted background stays the same",
                                              SMOOTH_FRAMES)
        self.time.moved.connect(lambda frame: self.set_time((frame + 0.5) * self.dt))
        self.time.stepped.connect(lambda n: self.time.move(n * self.grouping()))
        self.contrast.changed.connect(self.redraw_kymos)
        for control in (self.range, self.height_smooth, self.background_smooth):
            control.changed.connect(self.redraw)

    def make_layout(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        for label, w in (("Peaks", self.peaks_edit), ("Grouping scale", self.scale_box)):
            controls.addWidget(QLabel(label))
            controls.addWidget(w)
        controls.addWidget(self.unmatched_box)

        layout.addWidget(self.force_canvas)
        layout.addLayout(self.time.new_row())
        layout.addLayout(controls)
        layout.addWidget(self.kymo_canvas)
        kymo_row = QHBoxLayout()  # intensity cutoffs, then the d range on the right
        kymo_row.addLayout(self.contrast.row)
        kymo_row.addLayout(self.range.new_row(projection=False))
        layout.addLayout(kymo_row)
        layout.addLayout(self.time.new_row())
        for c, smoothing in ((self.track_canvas, None), (self.height_canvas, self.height_smooth),
                             (self.background_canvas, self.background_smooth)):
            if smoothing:
                layout.addLayout(smoothing.row)
            layout.addWidget(c)
            layout.addLayout(self.time.new_row())
            layout.addLayout(self.range.new_row(projection=True))

    # ---------------------------------------------------------------- recordings

    def load(self, results):
        """``results``: (label, kymograph .npy, .ddf or "", tracking folder) of each recording."""
        self.recs = [load_recording(*result, color) for result, color in zip(results, COLORS)]
        self.t_cur = 0.0
        self.refresh(keep_grouping=False)

    def remove(self, i):
        """Drop recording ``i``, keeping the time and grouping scale; the others are not reloaded."""
        del self.recs[i]
        self.refresh(keep_grouping=True)

    def refresh(self, keep_grouping):
        """Fit the sliders and grouping scales to the recordings and redraw.  No recordings: hide."""
        if not self.recs:
            self.hide()
            return
        self.projections = []  # the old ones, until the redraw
        self.unmatched_box.setEnabled(self.any_unet())
        self.t_max = max(r.t_end for r in self.recs)
        self.dt = min(r.dt for r in self.recs)
        last = max(0, round(self.t_max / self.dt) - 1)
        frame = min(max(int(self.t_cur / self.dt), 0), last)
        self.time.set(frame, last)
        self.t_cur = (frame + 0.5) * self.dt

        options = [str(s) for s in group_options(self.recs)]
        current = self.scale_box.currentText()
        with QSignalBlocker(self.scale_box):
            self.scale_box.clear()
            self.scale_box.addItems(options)
            self.scale_box.setCurrentText(current if keep_grouping and current in options else options[0])
        self.draw_force()
        self.show()
        self.redraw()

    def any_unet(self):
        return any(r.detector == "unet" for r in self.recs)

    def grouping(self):
        return int(self.scale_box.currentText() or 1)

    def q_range(self):
        """(q_lo, q_hi) shown: the typed range, else every recording's radius from/to."""
        whole = (min(r.params["rmin"] * r.inv for r in self.recs), max(r.params["rmax"] * r.inv for r in self.recs))
        typed = self.range.typed_range()
        if typed is None:
            return whole
        a, b, unit = typed
        if unit == "px":
            qs = [v * r.inv for r in self.recs for v in (a, b)]
        else:
            qs = [1.0 / v for v in (a, b) if v > 0]
        return (min(qs), max(qs)) if len(qs) > 1 and min(qs) < max(qs) else whole

    def wanted_names(self):
        """Pattern of the names in the peaks field, or None (all).  A name matches if it starts with one
        of them; after a final digit no other digit may follow (M1 is not M10, but M is every M)."""
        tokens = [t.strip() for t in self.peaks_edit.text().split(",") if t.strip()]
        patterns = [re.escape(t) + (r"(?!\d)" if t[-1].isdigit() else "") for t in tokens]
        return re.compile("|".join(patterns)) if patterns else None

    # ---------------------------------------------------------------- drawing

    def redraw(self):
        if not self.recs:
            return
        self.q_shown = self.q_range()
        wanted, unmatched = self.wanted_names(), self.unmatched_box.isChecked()
        for r in self.recs:
            r.group_by(self.grouping())
        self.shown = [r.tracks(self.q_shown, wanted, unmatched) for r in self.recs]
        self.projections = [[] for _ in self.recs]
        self.draw_kymos()
        self.draw_tracks()
        self.draw_background()
        if self.range.xlim is None:  # show the fitted range (union over all panels)
            autos = [p.auto for ps in self.projections for p in ps]
            self.range.show_xlim((min(lo for lo, _ in autos), max(hi for _, hi in autos)))
        self.set_time(self.t_cur)

    def redraw_kymos(self):
        if self.recs:
            self.draw_kymos()
            self.set_time(self.t_cur)

    def draw_force(self):
        fig = self.force_canvas.figure
        fig.clear()
        traces = [(r, tr) for r in self.recs for tr in r.traces]
        keys = [k for k in ("force", "length") if any(tr["key"] == k for _, tr in traces)]
        self.force_lines = []
        if not keys:
            fig.text(0.5, 0.5, "no .ddf loaded", ha="center", va="center", color="grey")
            return
        axes = fig.subplots(len(keys), 1, sharex=True, squeeze=False)[:, 0]
        for ax, key in zip(axes, keys):
            for r, tr in traces:
                if tr["key"] == key:
                    ax.plot(tr["t"], tr["y"], lw=1.2, color=r.color, label=r.label)
                    ax.set_ylabel(f"{key} ({tr['units']})")
            ax.grid(axis="y", lw=0.6, color="#e9e9e9")
            self.force_lines.append(time_line(ax))
        axes[0].legend(fontsize=8, frameon=False)
        axes[-1].set_xlabel("time from trigger (s)")
        axes[-1].set_xlim(0.0, self.t_max)

    def draw_kymos(self):
        fig = self.kymo_canvas.figure
        fig.clear()
        axes = fig.subplots(1, len(self.recs), sharey=True, squeeze=False)[0]
        for ax, r, shown in zip(axes, self.recs, self.shown):
            self.draw_kymo(fig, ax, r, shown)
        self.kymo_lines = [time_line(ax) for ax in axes]
        d_ticks(axes[0], *self.q_shown)
        axes[0].set_ylabel("d (nm)   [right: px]")

    def draw_kymo(self, fig, ax, r, shown):
        counts = self.contrast.counts
        raw = np.clip(r.view.grouped, 0, None)
        disp = np.full((len(r.pos), r.n_source), np.nan)  # dropped frames stay blank
        disp[:, r.frames] = raw if counts else np.log1p(raw)
        inside = (r.q >= self.q_shown[0]) & (r.q <= self.q_shown[1])
        lo, hi = self.contrast.limits(np.nanpercentile(disp[inside] if inside.any() else disp, (1.0, 99.5)))
        im = ax.imshow(disp, origin="lower", aspect="auto", cmap="gray_r", vmin=lo, vmax=hi,
                       interpolation="nearest", extent=(0.0, r.t_end, r.q[0], r.q[-1]))
        for a, _ in shown:  # the tracks, faint so the kymograph stays readable
            unmatched = a in r.unmatched
            self.plot_track(ax, r, r.r[r.k, a], r.r[r.k, a] * r.inv, lw=0.8, alpha=0.45,
                            color=UNMATCHED_COLOR if unmatched else r.color, ls="--" if unmatched else "-")
        bar = fig.colorbar(im, ax=ax, pad=0.13, fraction=0.05)  # own scale, clear of the px axis
        bar.set_label("I (counts)" if counts else "log(1+I)", fontsize=8)
        bar.ax.tick_params(labelsize=7)
        ax.set_title(f"{r.label}  (s = {r.s})", color=r.color, fontsize=9)
        ax.set_xlabel("time (s)")
        ax.set_xlim(0.0, self.t_max)
        ax.set_ylim(*self.q_shown)
        ax.secondary_yaxis("right", functions=(lambda q: q / r.inv, lambda px: px * r.inv)).tick_params(labelsize=7)

    @staticmethod
    def plot_track(ax, r, rr, y, **style):
        """y where the track rr is detected, broken at excluded frames."""
        ok = np.isfinite(rr)
        ax.plot(*broken(r.frames[ok], r.times[ok], y[ok]), **style)

    def add_projections(self, axes, what, values):
        """One projection per recording; x is fitted to the rows of ``values(r)`` in the shown range."""
        for r, ax, projections in zip(self.recs, axes, self.projections):
            inside = (r.q >= self.q_shown[0]) & (r.q <= self.q_shown[1])
            if what == "bg" and not inside.any():
                inside[:] = True
            projections.append(Projection(ax, r, what, values(r)[inside], self.range.log, self.range.xlim))

    def draw_tracks(self):
        """Tracks (q, labelled in d) and, equally sized below, their heights; each beside one grouped
        projection per recording (background removed as the tracker did, q up), the detected peaks
        marked with their d above and their height below; dotted: the acceptance threshold."""
        n, unet = len(self.recs), self.any_unet()
        ax_t, right_t = panel_row(self.track_canvas.figure, n, self.q_shown, share_q=True)
        ax_h, right_h = panel_row(self.height_canvas.figure, n, self.q_shown, share_q=False)
        heights = np.concatenate([np.empty(0)] + [self.draw_track(ax_t, ax_h, r, a, name)
                                                  for r, shown in zip(self.recs, self.shown) for a, name in shown])
        self.add_projections(right_t, "d", lambda r: r.view.flats)
        self.add_projections(right_h, "height", lambda r: r.view.flats)
        for pr in [] if unet else self.projections[0]:
            pr.ax.legend([pr.extra], ["noise × SNR threshold"], fontsize=6, loc="lower right", frameon=False)

        ax_t.set_ylim(*self.q_shown)
        d_ticks(ax_t, *self.q_shown)
        ax_t.set_ylabel("d (nm)", fontsize=9)
        ax_h.set_yscale("log")
        if heights.size:
            ax_h.set_ylim(heights.min() / 2.0, heights.max() * 2.0)
        ax_h.set_ylabel("peak height above baseline", fontsize=9)
        at = f"at grouping scale {self.grouping()}"
        ax_t.set_title("U-Net tracks (per frame)" if unet else f"tracked peaks {at}", fontsize=9)
        ax_h.set_title(("U-Net peak heights (per frame)" if unet else f"peak heights {at}") + self.height_smooth.note,
                       fontsize=9)
        for ax in (ax_t, ax_h):
            style_time_axis(ax, self.t_max, left_labels=True)
            if not any(self.shown):
                ax.text(0.5, 0.5, "no tracks to show in range", transform=ax.transAxes, ha="center", color="grey")
        self.panel_lines = [time_line(ax) for ax in (ax_t, ax_h)]

    def draw_track(self, ax_t, ax_h, r, a, name):
        """One track's position and height, named at both ends; returns the plotted heights."""
        rr = r.r[r.k, a]
        h = smooth_series(r.height[r.k, a], r.frames, self.height_smooth.window)
        h = np.where(np.isfinite(rr) & (h > 0), h, np.nan)
        for ax, y in ((ax_t, rr * r.inv), (ax_h, h)):
            self.plot_track(ax, r, rr, y, lw=1.3, color=r.color, ls="--" if a in r.unmatched else "-")
            seen = np.flatnonzero(np.isfinite(y))
            for i, text, side in ((seen[0], f"{name} ", "right"), (seen[-1], f" {name}", "left")) if len(seen) else ():
                ax.text(r.times[i], y[i], text, ha=side, va="center", fontsize=7, color=r.color, clip_on=True)
        return h[np.isfinite(h)]

    def draw_background(self):
        """The background over time (integrated over q between d = 80 and 3 nm, from each grouped
        profile's own hull or fit) beside each recording's grouped profile at the slider time with the
        background the tracker subtracted from it (dashed)."""
        ax_b, right = panel_row(self.background_canvas.figure, len(self.recs), self.q_shown, share_q=False)
        for r in self.recs:
            band = (r.q >= 1.0 / BG_D_NM[0]) & (r.q <= 1.0 / BG_D_NM[1])  # area under the background
            area = np.trapezoid(r.view.backgrounds[band], r.q[band], axis=0)
            area = smooth_series(area, r.frames, self.background_smooth.window)
            ax_b.plot(*broken(r.frames, r.times, area), lw=1.3, color=r.color, label=r.label)
            if r.shared is not None:
                ax_b.axhline(np.trapezoid(r.shared[band], r.q[band]), color=r.color, ls="--", lw=1,
                             label=f"{r.label}: averaged profile's (subtracted)")
        self.add_projections(right, "bg", lambda r: r.view.profiles)
        pr = self.projections[0][-1]
        pr.ax.legend([pr.extra], ["subtracted background"], fontsize=6, loc="lower right", frameon=False)
        modes = ", ".join(sorted({r.background_mode for r in self.recs}))
        note = self.background_smooth.note
        ax_b.set_ylabel(f"background area, d {BG_D_NM[0]:g}–{BG_D_NM[1]:g} nm (I·nm⁻¹)", fontsize=9)
        ax_b.set_title(f"background at grouping scale {self.grouping()}  (tracker subtracts: {modes})"
                       + (f" {note}" if note else ""), fontsize=9)
        style_time_axis(ax_b, self.t_max)
        ax_b.legend(fontsize=7, frameon=False)
        self.panel_lines.append(time_line(ax_b))

    # ---------------------------------------------------------------- time

    def respace_labels(self, _event):
        """The label spacing depends on the panel height; a resize can come before the first draw."""
        if self.recs and len(self.projections) == len(self.recs):
            self.set_time(self.t_cur)

    def set_time(self, t):
        if not self.recs:
            return
        self.t_cur = t
        for line in self.force_lines + self.kymo_lines + self.panel_lines:
            line.set_xdata([t, t])
        for r, shown, projections in zip(self.recs, self.shown, self.projections):
            self.show_projections(r, shown, projections, r.column_at(t))
        self.time.set_text(f"{t:.4f} s")
        for c in (self.force_canvas, self.kymo_canvas, self.track_canvas, self.height_canvas,
                  self.background_canvas):
            c.draw_idle()

    def show_projections(self, r, shown, projections, col):
        if col is None:
            for pr in projections:
                pr.clear("frame dropped\nor outside")
            return
        c = r.column(col)
        peaks = r.peaks_at(col, c.flat, shown, self.q_shown)
        threshold = None if r.detector == "unet" else r.threshold * c.sigma  # the U-Net has no SNR threshold
        for pr in projections:
            if pr.what == "bg":
                pr.show(r.q, c.profile, c.background)
            else:
                pr.show(r.q, c.flat, threshold, peaks)
