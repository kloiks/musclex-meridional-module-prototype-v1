"""The viewer: force/length traces with a time slider, kymographs, tracked peaks, peak heights and backgrounds."""
import re

import numpy as np
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from controls import ContrastRow, RangeRows, SmoothingRow, TimeSliders, canvas
from plots import Projection, broken, d_ticks, panel_row, smooth_series, style_time_axis, time_line
from normalization import find_beamlog, plan
from params import load_params
from recording import group_options, load_recording

COLORS = ("#2b7fff", "#e8590c", "#2f9e44")
UNMATCHED_COLOR = "#7a3fb5"
NO_FRAME = "frame dropped\nor outside"  # shown on the projections
SMOOTH_FRAMES = 10  # default window of the plotted peak heights and background
BG_D_NM = ("3", "80")  # default d range (nm) of the background section
trapezoid = getattr(np, "trapezoid", None) or np.trapz  # np.trapezoid is NumPy >= 2.0


class Viewer(QWidget):
    """Force/length overlay with a time slider; kymographs; tracked peaks (d and height) beside each
    recording's grouped projection at the slider time.  Every position axis is q = r * inv,
    labelled in d (nm), so recordings with different calibrations line up.  The background section has
    its own range and projection controls; it shares the time and the grouping."""

    def __init__(self):
        super().__init__()
        self.recs = []          # Recordings
        self.shown = []         # per recording: the [(track, name)] drawn
        self.projections = []   # per recording: its Projections of the tracks and the heights
        self.bg_projections = []  # per recording: its Projection of the background section
        self.force_lines, self.kymo_lines, self.track_lines, self.bg_lines = [], [], [], []  # at the slider time
        self.t_cur, self.t_max, self.dt = 0.0, 1.0, 1.0
        p = load_params()  # the Normalization dialog's settings; logs: kymograph .npy -> log chosen
        self.norm = dict(logs={}, use_i0=bool(p["norm_i0"]), method=p["norm_between"], frames=p["norm_frames"])
        self.norm_error = ""  # why the between-sequence normalisation could not be applied
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
        self.peaks_edit.editingFinished.connect(self.redraw_tracks)
        self.scale_box = QComboBox()
        self.scale_box.setToolTip("Frames averaged (sliding mean) in the kymographs, projections and background. "
                                  "U-Net and max-flow tracks and heights stay per frame.")
        self.scale_box.currentTextChanged.connect(lambda _: self.redraw())
        self.unmatched_box = QCheckBox("Show unmatched tracks", checked=True)
        self.unmatched_box.setToolTip("U-Net and max-flow tracks that match no expected peak, named X1, X2, ... from the "
                                      "centre out in each recording (dashed)")
        self.unmatched_box.toggled.connect(lambda _: self.redraw_tracks())

        self.time, self.contrast = TimeSliders(), ContrastRow()
        self.range, self.bg_range = RangeRows(), RangeRows(BG_D_NM)
        self.height_smooth = SmoothingRow("Smooth peak heights",
                                          "tracks.csv and the projection labels keep the measured heights",
                                          SMOOTH_FRAMES)
        self.background_smooth = SmoothingRow("Smooth background", "the subtracted background stays the same",
                                              SMOOTH_FRAMES)
        self.time.moved.connect(lambda frame: self.set_time((frame + 0.5) * self.dt))
        self.time.stepped.connect(lambda n: self.time.move(n * self.grouping()))
        self.contrast.changed.connect(self.redraw_kymos)
        for control in (self.range, self.height_smooth):
            control.changed.connect(self.redraw_tracks)
        for control in (self.bg_range, self.background_smooth):
            control.changed.connect(self.redraw_background)

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
        for c, smoothing in ((self.track_canvas, None), (self.height_canvas, self.height_smooth)):
            if smoothing:
                layout.addLayout(smoothing.row)
            layout.addWidget(c)
            layout.addLayout(self.time.new_row())
            layout.addLayout(self.range.new_row(projection=True))
        self.add_background_section(layout)

    def add_background_section(self, layout):
        """Set apart by a line and a subtitle; its range, at the top, is not linked to the others."""
        subtitle = QLabel("Background intensity")
        subtitle.setStyleSheet("font-weight: bold; font-size: 13pt")
        layout.addSpacing(12)
        layout.addWidget(QFrame(frameShape=QFrame.HLine, frameShadow=QFrame.Sunken))
        layout.addWidget(subtitle)
        layout.addLayout(self.bg_range.new_row(projection=True))
        layout.addLayout(self.background_smooth.row)
        layout.addWidget(self.background_canvas)
        layout.addLayout(self.time.new_row())

    # ---------------------------------------------------------------- recordings

    def load(self, results):
        """``results``: (label, kymograph .npy, .ddf or "", tracking folder) of each recording."""
        self.recs = [load_recording(*result, color) for result, color in zip(results, COLORS)]
        self.t_cur = 0.0
        self.renormalize()
        self.refresh(keep_grouping=False)

    def remove(self, i):
        """Drop recording ``i``, keeping the time and grouping scale; the others are not reloaded
        (but renormalised: the between-sequence scales depend on which recordings there are)."""
        del self.recs[i]
        self.renormalize()
        self.refresh(keep_grouping=True)

    # ---------------------------------------------------------------- normalisation

    def renormalize(self):
        """Apply the normalisation settings to the recordings (logs not chosen: found next to the images)."""
        if not self.recs:
            return
        s = self.norm
        if not s["use_i0"] and s["method"] == "none":
            result = None
        else:
            logs = [s["logs"].get(r.npy) or find_beamlog(r.qf) for r in self.recs]
            result = plan(self.recs, logs, s["use_i0"], s["method"], s["frames"], load_params())
        self.apply_plan(result)

    def set_normalization(self, settings, result):
        """From the Normalization dialog: its settings and the plan it computed; redraws."""
        self.norm = dict(settings, logs={**self.norm["logs"], **settings["logs"]})
        self.apply_plan(result)
        if self.recs:
            self.refresh(keep_grouping=True)

    def apply_plan(self, result):
        self.norm_error = result.between if result is not None and isinstance(result.between, str) else ""
        for i, r in enumerate(self.recs):
            r.normalize(np.ones(len(r.frames)) if result is None else result.factors[i])
            r.norm_note = "" if result is None else self.norm_note(result, i)
            r.beam = None if result is None else result.within[i].beam

    def norm_note(self, result, i):
        """Short description of what was applied to recording i, for the kymograph title."""
        parts = []
        if self.norm["use_i0"] and result.within[i].i0 is not None:
            parts.append("÷ I0")
        if not isinstance(result.between, (str, type(None))):
            parts.append(f"× {result.between.scales[i]:.3f}")
        return f", {' '.join(parts)}" if parts else ""

    def refresh(self, keep_grouping):
        """Fit the sliders and grouping scales to the recordings and redraw.  No recordings: hide."""
        if not self.recs:
            self.hide()
            return
        self.projections, self.bg_projections = [], []  # the old ones, until the redraw
        self.unmatched_box.setEnabled(self.any_per_frame())
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

    def any_per_frame(self):
        """Whether a recording was tracked per frame (U-Net or max-flow) rather than by multiscale."""
        return any(r.detector != "multiscale" for r in self.recs)

    def per_frame_name(self):
        names = {"unet": "U-Net", "maxflow": "max-flow"}
        return " / ".join(sorted({names[r.detector] for r in self.recs if r.detector in names}))

    def grouping(self):
        return int(self.scale_box.currentText() or 1)

    def q_range(self, rows):
        """(q_lo, q_hi) of the range applied in ``rows`` (RangeRows), else every recording's radius from/to."""
        whole = (min(r.params["rmin"] * r.inv for r in self.recs), max(r.params["rmax"] * r.inv for r in self.recs))
        if rows.range is None:
            return whole
        a, b, unit = rows.range
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
        """Everything, at the grouping scale."""
        for r in self.recs:
            r.group_by(self.grouping())
        self.redraw_background()
        self.redraw_tracks()

    def redraw_tracks(self):
        """The kymographs, tracks and heights."""
        if not self.recs:
            return
        self.q_shown = self.q_range(self.range)
        wanted, unmatched = self.wanted_names(), self.unmatched_box.isChecked()
        self.shown = [r.tracks(self.q_shown, wanted, unmatched) for r in self.recs]
        self.draw_kymos()
        self.draw_tracks()
        self.show_fitted_x(self.range, [p for ps in self.projections for p in ps])
        self.set_time(self.t_cur)

    def redraw_background(self):
        """The background section."""
        if not self.recs:
            return
        self.draw_background()
        self.show_fitted_x(self.bg_range, self.bg_projections)
        self.set_time(self.t_cur)

    @staticmethod
    def show_fitted_x(rows, projections):
        """Without an x range set, show the fitted one (union over the projections)."""
        if rows.xlim is None:
            rows.show_xlim((min(p.auto[0] for p in projections), max(p.auto[1] for p in projections)))

    def redraw_kymos(self):
        if self.recs:
            self.draw_kymos()
            self.set_time(self.t_cur)

    def draw_force(self):
        """Force and length from the .ddf and, while a normalisation is applied, I0 and I1 from the
        beamline log (frame k at the middle of its exposure, from the trigger)."""
        fig = self.force_canvas.figure
        fig.clear()
        traces = [(r, tr) for r in self.recs for tr in r.traces]
        keys = [k for k in ("force", "length") if any(tr["key"] == k for _, tr in traces)]
        beams = [r for r in self.recs if r.beam is not None]
        keys += ["I0", "I1"] if beams else []
        self.force_canvas.setMinimumHeight(320 if len(keys) <= 2 else 150 * len(keys))
        self.force_lines = []
        if not keys:
            fig.text(0.5, 0.5, "no .ddf loaded", ha="center", va="center", color="grey")
            return
        axes = fig.subplots(len(keys), 1, sharex=True, squeeze=False)[:, 0]
        for ax, key in zip(axes, keys):
            if key in ("I0", "I1"):
                for r in beams:
                    y = (r.beam.i0 if key == "I0" else r.beam.i1)[:r.n_source]
                    ax.plot((np.arange(len(y)) + 0.5) * r.dt, y, lw=1.2, color=r.color, label=r.label)
                ax.set_ylabel(f"{key} ({'incident' if key == 'I0' else 'transmitted'}, counts)", fontsize=8)
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
        ax.set_title(f"{r.label}  (s = {r.s}{r.norm_note})", color=r.color, fontsize=9)
        ax.set_xlabel("time (s)")
        ax.set_xlim(0.0, self.t_max)
        ax.set_ylim(*self.q_shown)
        ax.secondary_yaxis("right", functions=(lambda q: q / r.inv, lambda px: px * r.inv)).tick_params(labelsize=7)

    @staticmethod
    def plot_track(ax, r, rr, y, **style):
        """y where the track rr is detected, broken at excluded frames."""
        ok = np.isfinite(rr)
        ax.plot(*broken(r.frames[ok], r.times[ok], y[ok]), **style)

    def make_projections(self, axes, what, rows, q_range, values):
        """One projection per recording with the x settings of ``rows``; x is fitted to ``values(r)``
        in the q range."""
        return [Projection(ax, r, what, values(r)[(r.q >= q_range[0]) & (r.q <= q_range[1])], rows.log, rows.xlim)
                for r, ax in zip(self.recs, axes)]

    def draw_tracks(self):
        """Tracks (q, labelled in d) and, equally sized below, their heights; each beside one grouped
        projection per recording (background removed as the tracker did, q up), the detected peaks
        marked with their d above and their height below; dotted: the acceptance threshold."""
        n, per_frame = len(self.recs), self.any_per_frame()
        ax_t, right_t = panel_row(self.track_canvas.figure, n, self.q_shown, share_q=True)
        ax_h, right_h = panel_row(self.height_canvas.figure, n, self.q_shown, share_q=False)
        heights = np.concatenate([np.empty(0)] + [self.draw_track(ax_t, ax_h, r, a, name)
                                                  for r, shown in zip(self.recs, self.shown) for a, name in shown])
        self.projections = list(zip(*(
            self.make_projections(axes, what, self.range, self.q_shown, lambda r: r.view.flats)
            for axes, what in ((right_t, "d"), (right_h, "height")))))
        for pr in [] if per_frame else self.projections[0]:
            pr.ax.legend([pr.extra], ["noise × SNR threshold"], fontsize=6, loc="lower right", frameon=False)

        ax_t.set_ylim(*self.q_shown)
        d_ticks(ax_t, *self.q_shown)
        ax_t.set_ylabel("d (nm)", fontsize=9)
        ax_h.set_yscale("log")
        if heights.size:
            ax_h.set_ylim(heights.min() / 2.0, heights.max() * 2.0)
        ax_h.set_ylabel("peak height above baseline", fontsize=9)
        at = f"at grouping scale {self.grouping()}"
        ax_t.set_title(f"{self.per_frame_name()} tracks (per frame)" if per_frame else f"tracked peaks {at}", fontsize=9)
        ax_h.set_title((f"{self.per_frame_name()} peak heights (per frame)" if per_frame else f"peak heights {at}") + self.height_smooth.note,
                       fontsize=9)
        for ax in (ax_t, ax_h):
            style_time_axis(ax, self.t_max, left_labels=True)
            if not any(self.shown):
                ax.text(0.5, 0.5, "no tracks to show in range", transform=ax.transAxes, ha="center", color="grey")
        self.track_lines = [time_line(ax) for ax in (ax_t, ax_h)]

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
        """The background over time (integrated over the section's range, from each grouped profile's
        own hull or fit) beside each recording's grouped profile at the slider time with the
        background the tracker subtracted from it (dashed)."""
        q_lo, q_hi = q_range = self.q_range(self.bg_range)
        ax_b, right = panel_row(self.background_canvas.figure, len(self.recs), q_range, share_q=False)
        for r in self.recs:
            band = (r.q >= q_lo) & (r.q <= q_hi)  # area under the background
            area = trapezoid(r.view.backgrounds[band], r.q[band], axis=0)
            area = smooth_series(area, r.frames, self.background_smooth.window)
            ax_b.plot(*broken(r.frames, r.times, area), lw=1.3, color=r.color, label=r.label)
            if r.shared is not None:
                ax_b.axhline(trapezoid(r.shared[band], r.q[band]), color=r.color, ls="--", lw=1,
                             label=f"{r.label}: averaged profile's (subtracted)")
        self.bg_projections = self.make_projections(right, "bg", self.bg_range, q_range, lambda r: r.view.profiles)
        pr = self.bg_projections[0]
        pr.ax.legend([pr.extra], ["subtracted background"], fontsize=6, loc="lower right", frameon=False)
        modes = ", ".join(sorted({r.background_mode for r in self.recs}))
        note = self.background_smooth.note
        ax_b.set_ylabel(f"background area, d {1 / q_hi:.4g}–{1 / q_lo:.4g} nm (I·nm⁻¹)", fontsize=9)
        ax_b.set_title(f"background at grouping scale {self.grouping()}  (tracker subtracts: {modes})"
                       + (f" {note}" if note else ""), fontsize=9)
        style_time_axis(ax_b, self.t_max)
        ax_b.legend(fontsize=7, frameon=False)
        self.bg_lines = [time_line(ax_b)]

    # ---------------------------------------------------------------- time

    def respace_labels(self, _event):
        """The label spacing depends on the panel height; a resize can come before the first draw."""
        if self.recs and len(self.projections) == len(self.recs):
            self.set_time(self.t_cur)

    def set_time(self, t):
        if not self.recs:
            return
        self.t_cur = t
        for line in self.force_lines + self.kymo_lines + self.track_lines + self.bg_lines:
            line.set_xdata([t, t])
        for r, shown, projections in zip(self.recs, self.shown, self.projections):
            self.show_projections(r, shown, projections, r.column_at(t))
        for r, pr in zip(self.recs, self.bg_projections):
            self.show_background(r, pr, r.column_at(t))
        self.time.set_text(f"{t:.4f} s")
        for c in (self.force_canvas, self.kymo_canvas, self.track_canvas, self.height_canvas,
                  self.background_canvas):
            c.draw_idle()

    def show_projections(self, r, shown, projections, col):
        if col is None:
            for pr in projections:
                pr.clear(NO_FRAME)
            return
        c = r.column(col)
        peaks = r.peaks_at(col, c.flat, shown, self.q_shown)
        threshold = r.threshold * c.sigma if r.detector == "multiscale" else None  # the multiscale acceptance SNR
        for pr in projections:
            pr.show(r.q, c.flat, threshold, peaks)

    @staticmethod
    def show_background(r, pr, col):
        if col is None:
            pr.clear(NO_FRAME)
            return
        c = r.column(col)
        pr.show(r.q, c.profile, c.background)
