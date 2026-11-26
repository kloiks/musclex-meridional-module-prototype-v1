"""The dialogs: expected peaks, parameters, background, calibration and normalization."""
import csv
import json
import os

import numpy as np
from PIL import Image
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QRadioButton, QScrollArea, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from . import cache
from .calibrate_m3 import calibrate, from_m3
from .controls import canvas
from .frames import blank_frames, format_frames, load_excluded, parse_frames, save_excluded
from .kymograph import cached_kymograph, find_frames
from .normalization import BETWEEN, find_beamlog, plan
from .params import DEFAULTS, PEAKS_CSV, load_params, save_params

# (group, [(key, label, choices or None)]); the widget type follows the default's type
PARAM_GROUPS = [
    ("Kymograph", [("width", "Meridional strip width (px)", None),
                   ("m3_rmin", "M3 search from radius (px)", None),
                   ("d_m3", "M3 spacing (nm)", None),
                   ("pos_lo", "Force figure: range from (px)", None),
                   ("pos_hi", "Force figure: range to (px)", None),
                   ("stretch", "Force figure: display px per position row", None)]),
    ("Blank frames (automatic detection)", [("dark_fraction", "Blank if column sum < fraction of median", None),
                                            ("drop_pad", "Extra frames excluded either side", None)]),
    ("U-Net detection", [("unet_threshold", "Score threshold (0-1)", None),
                         ("unet_gate", "Max step between linked peaks (px)", None),
                         ("unet_memory", "Missing frames a track may bridge", None),
                         ("unet_min_length", "Min detected frames of a track", None)]),
    ("Max-flow detection (guided by multiscale anchors)", [
        ("maxflow_min_snr", "Detection SNR in single frames", None),
        ("maxflow_smooth", "Smoothing (px)", None),
        ("maxflow_guide_max_res", "Guide anchors: max resolution scale", None),
        ("maxflow_guide_scale", "Guide anchors: r(t) read at scale", None),
        ("maxflow_corridor", "Corridor half-width around anchors (px)", None),
        ("maxflow_det_half", "Detection reward: log SNR where it changes sign", None),
        ("maxflow_max_shift", "Max step per frame (px)", None),
        ("maxflow_shift_scale", "Step costing one unit (px)", None),
        ("maxflow_max_gap", "Missing frames a link may bridge", None),
        ("maxflow_c_miss", "Cost per missing frame", None),
        ("maxflow_c_enter", "Cost of starting a track", None),
        ("maxflow_c_exit", "Cost of ending a track", None),
        ("maxflow_min_length", "Min detected frames of a track", None)]),
    ("Multiscale detection", [("rmin", "Radius from (px)", None),
                              ("rmax", "Radius to (px)", None),
                              ("smooth", "Smoothing (px)", None),
                              ("snr_floor", "Detection SNR floor", None),
                              ("noise_win", "Noise window (px)", None),
                              ("noise_pct", "Noise percentile", None),
                              ("min_sep", "Min peak separation (px)", None)]),
    ("Multiscale corridor", [("snr_coarse", "SNR at the coarsest scale (anchors)", None),
                             ("snr_fine", "SNR at the finest scale", None),
                             ("corridor", "Corridor half-width (px)", None),
                             ("base", "Scale ratio", None),
                             ("min_cov", "Coverage that counts as resolved", None),
                             ("seed_anchors", "Seed anchors at expected peaks", ("yes", "no"))]),
    ("Labelling", [("tol_pct", "Default tolerance (% of d)", None),
                   ("min_frames", "Min frames of a labelled track", None)]),
]
MINIMUMS = {"base": 2}  # the other numbers start at 0


def note(text):
    label = QLabel(text)
    label.setStyleSheet("color: grey")
    return label


def save_cancel(dialog, *more):
    """The Save and Cancel (and ``more``) buttons of a dialog."""
    flags = QDialogButtonBox.Save | QDialogButtonBox.Cancel
    for button in more:
        flags |= button
    buttons = QDialogButtonBox(flags)
    buttons.accepted.connect(dialog.save)
    buttons.rejected.connect(dialog.reject)
    return buttons


def radio_group(title, choices, current):
    """(box, {key: radio button}) of (key, text) choices; an unknown ``current`` selects the first."""
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    buttons = {}
    for key, text in choices:
        buttons[key] = QRadioButton(text)
        layout.addWidget(buttons[key])
    buttons.get(current, buttons[choices[0][0]]).setChecked(True)
    return box, buttons


def checked(buttons):
    return next(key for key, b in buttons.items() if b.isChecked())


class ExpectedPeaksDialog(QDialog):
    """Edit config/expected_peaks.csv: name, d (nm), tolerance (% of d, blank = tol_pct)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Expected peaks")
        with open(PEAKS_CSV) as f:
            lines = f.readlines()
        self.header = [line for line in lines if line.startswith("#")]
        rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
        self.default_tol = f"{load_params()['tol_pct']:g}"
        self.table = QTableWidget(len(rows), 3)
        self.table.setHorizontalHeaderLabels(["name", "d (nm)", "tolerance (%)"])
        for i, r in enumerate(rows):
            for j, key in enumerate(("name", "d_nm")):
                self.table.setItem(i, j, QTableWidgetItem(r.get(key) or ""))
            self.set_tol(i, (r.get("tol_pct") or "").strip())
        self.table.itemChanged.connect(self.tol_edited)
        self.table.horizontalHeader().setStretchLastSection(True)

        add, remove = QPushButton("Add peak"), QPushButton("Remove selected")
        add.clicked.connect(self.add_row)
        remove.clicked.connect(self.remove_selected)
        edit_row = QHBoxLayout()
        edit_row.addWidget(add)
        edit_row.addWidget(remove)
        edit_row.addStretch()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Tolerance is a percentage of d. Grey = the 'Default tolerance (% of d)' "
                                f"parameter ({self.default_tol} %);\ntype a value to override it. "
                                "Windows are capped at half the gap to the neighbouring peak."))
        layout.addWidget(self.table)
        layout.addLayout(edit_row)
        layout.addWidget(save_cancel(self))
        self.resize(480, 640)

    def set_tol(self, i, text):
        """Tolerance cell: ``text``, or the default in grey (saved blank) when empty."""
        item = QTableWidgetItem(text or self.default_tol)
        if not text:
            item.setData(Qt.UserRole, True)
            item.setForeground(Qt.gray)
        self.table.setItem(i, 2, item)

    def tol_edited(self, item):
        """An edited tolerance is an override: no longer grey."""
        if item.column() == 2 and item.data(Qt.UserRole) and item.text().strip() != self.default_tol:
            item.setData(Qt.UserRole, False)
            item.setForeground(self.table.palette().text())

    def add_row(self):
        i = self.table.rowCount()
        self.table.insertRow(i)
        self.table.setItem(i, 0, QTableWidgetItem())
        self.set_tol(i, "")
        self.table.scrollToBottom()
        self.table.editItem(self.table.item(i, 0))

    def remove_selected(self):
        for i in sorted({ix.row() for ix in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(i)

    def cell(self, i, j):
        item = self.table.item(i, j)
        return item.text().strip() if item else ""

    def read_row(self, i):
        """(name, d, tolerance or None) of a row, None if it is empty; ValueError if it is incomplete."""
        name, d, tol = (self.cell(i, j) for j in range(3))
        if not name and not d:
            return None
        if not name:
            raise ValueError("no name")
        if self.table.item(i, 2).data(Qt.UserRole) and tol == self.default_tol:
            tol = ""  # still the default: keep following the parameter
        return name, float(d), float(tol) if tol else None

    def save(self):
        rows = []
        for i in range(self.table.rowCount()):
            try:
                rows.append(self.read_row(i))
            except ValueError:
                QMessageBox.warning(self, "Expected peaks", f"Row {i + 1}: needs a name, a number for d "
                                                            "and an empty or numeric tolerance.")
                return
        with open(PEAKS_CSV, "w") as f:
            f.writelines(self.header)
            f.write("name,d_nm,tol_pct\n")
            for name, d, tol in sorted((r for r in rows if r), key=lambda r: -r[1]):
                f.write(f"{name},{d:g},{'' if tol is None else f'{tol:g}'}\n")
        self.accept()


class ParametersDialog(QDialog):
    """Edit config/parameters.json (missing keys use the defaults) and, per selected recording,
    the frames to exclude: filled in with the blank frames found; an edited list is kept in
    <output>/excluded_frames.json."""

    def __init__(self, parent, jobs=()):
        super().__init__(parent)
        self.setWindowTitle("Parameters")
        self.widgets = {}  # parameter -> its widget
        self.recs = []     # per recording; r["auto"] = the list follows the detection
        content = QWidget()
        columns = QHBoxLayout(content)
        sides = [QVBoxLayout(), QVBoxLayout()]
        for side in sides:
            columns.addLayout(side)
        sides[0].addWidget(self.exclude_box(jobs, load_params()["width"]))
        for title, items in PARAM_GROUPS:
            sides[title.startswith(("Multiscale", "Labelling"))].addWidget(self.param_box(title, items))
        for side in sides:
            side.addStretch()

        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(content)
        buttons = save_cancel(self, QDialogButtonBox.RestoreDefaults)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        layout.addWidget(buttons)
        self.set_values(load_params())
        self.fill_auto()
        for key in ("dark_fraction", "drop_pad"):  # the automatic lists follow these settings
            self.widgets[key].valueChanged.connect(self.fill_auto)
        for side in sides:  # the scroll area has no size of its own: fit the dialog to its content
            side.activate()
        columns.activate()
        size = columns.sizeHint()
        self.resize(size.width() + 60, min(size.height() + 100, 900))

    def param_box(self, title, items):
        box = QGroupBox(title)
        form = QFormLayout(box)
        for key, label, choices in items:
            self.widgets[key] = self.widget(key, choices)
            form.addRow(label, self.widgets[key])
        return box

    def exclude_box(self, jobs, width):
        """One "frames to exclude" field per recording."""
        box = QGroupBox("Frames to exclude (all detectors)")
        form = QFormLayout(box)
        QApplication.setOverrideCursor(Qt.WaitCursor)  # without a cached kymograph the images are read
        try:
            for job in jobs:
                try:
                    kymo = cached_kymograph(find_frames(job.qf), job.out, width)[0]
                    saved = load_excluded(job.out)
                except Exception as e:
                    form.addRow(job.label, QLabel(f"cannot read: {e}"))
                else:
                    form.addRow(job.label, self.exclude_row(job, kymo.sum(axis=0), saved))
        finally:
            QApplication.restoreOverrideCursor()
        form.addRow(note("Frames from 0, e.g. 0-1, 40-60. Filled in with the blank frames found."
                         if jobs else "Select a QF folder first."))
        return box

    def exclude_row(self, job, totals, saved):
        """The field, filled with the ``saved`` frames (None = automatic), and its Auto button."""
        r = dict(out=job.out, label=job.label, totals=totals, auto=saved is None,
                 edit=QLineEdit(format_frames(saved or [])), info=QLabel())
        self.recs.append(r)
        r["edit"].setPlaceholderText("none")
        r["edit"].setMinimumWidth(180)
        r["edit"].textEdited.connect(lambda _: self.set_auto(r, False))
        auto = QPushButton("Auto")
        auto.setToolTip("Use the blank frames found in this recording")
        auto.clicked.connect(lambda: self.set_auto(r, True))
        row = QHBoxLayout()
        for w in (r["edit"], auto, r["info"]):
            row.addWidget(w)
        return row

    def set_auto(self, r, auto):
        r["auto"] = auto
        self.fill_auto()

    def fill_auto(self, *_):
        for r in self.recs:
            if r["auto"]:
                blank = blank_frames(r["totals"], self.widgets["dark_fraction"].value(),
                                     self.widgets["drop_pad"].value())
                r["edit"].setText(format_frames(np.flatnonzero(blank)))
            r["info"].setText(f"of {len(r['totals'])} frames, " + ("automatic" if r["auto"] else "edited"))

    def restore_defaults(self):
        self.set_values(DEFAULTS)
        for r in self.recs:
            self.set_auto(r, True)

    @staticmethod
    def widget(key, choices):
        if choices:
            w = QComboBox()
            w.addItems(choices)
        elif isinstance(DEFAULTS[key], int):
            w = QSpinBox(minimum=MINIMUMS.get(key, 0), maximum=100000)
        else:
            w = QDoubleSpinBox(minimum=0.0, maximum=1e6, decimals=3, singleStep=0.1)
        return w

    def set_values(self, values):
        for key, w in self.widgets.items():
            if isinstance(w, QComboBox):
                w.setCurrentText(str(values[key]))
            else:
                w.setValue(values[key])

    @staticmethod
    def excluded(r):
        """The frames typed for a recording, None if automatic; ValueError if they cannot be used."""
        if r["auto"]:
            return None
        frames, n = parse_frames(r["edit"].text()), len(r["totals"])
        if frames and frames[-1] >= n:
            raise ValueError(f"frame {frames[-1]} is past the last frame ({n - 1})")
        if len(frames) == n:
            raise ValueError("every frame is excluded")
        return frames

    def save(self):
        lists = []
        for r in self.recs:
            try:
                lists.append((r["out"], self.excluded(r)))
            except ValueError as e:
                QMessageBox.warning(self, "Frames to exclude", f"{r['label']}: {e}")
                return
        for out, frames in lists:
            save_excluded(out, frames)
        save_params({key: w.currentText() if isinstance(w, QComboBox) else w.value()
                     for key, w in self.widgets.items()})
        self.accept()


class BackgroundDialog(QDialog):
    """Choose the background the tracker (and the viewer) subtracts: the convex hull, or a fitted
    smooth background (MuscleX bg_fitting's general background: baseline + exponential + a second
    component), of every grouped projection or once from the record's averaged profile."""

    METHODS = (("hull", "Convex hull"),
               ("fit", "Fit: baseline + exponential + component 2 (MuscleX general background)"))
    COMP2 = (("lorentzian", "Lorentzian  1 / (1 + (r/D)²)"),
             ("powerlaw", "Power law  (1 + r/D)^−n"),
             ("stretched", "Stretched exponential  exp(−(r/D)^β)"))
    CHOICES = (("grouped", "Each grouped projection (background changes with time)"),
               ("averaged", "The averaged profile, the same background removed everywhere"))

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Background subtraction")
        params = load_params()
        method_box, self.methods = radio_group("Background", self.METHODS, params["bg_method"])
        choice_box, self.choices = radio_group("Computed from", self.CHOICES, params["background"])
        fit_box = self.fit_box(params)
        method_box.layout().addWidget(fit_box)
        self.methods["fit"].toggled.connect(fit_box.setEnabled)
        fit_box.setEnabled(self.methods["fit"].isChecked())
        layout = QVBoxLayout(self)
        for w in (method_box, choice_box,
                  note("Applies on the next run (tracking is recomputed; the kymographs are reused)."),
                  save_cancel(self)):
            layout.addWidget(w)

    def fit_box(self, params):
        self.comp2 = QComboBox()
        for key, text in self.COMP2:
            self.comp2.addItem(text, key)
        self.comp2.setCurrentIndex(max(self.comp2.findData(params["bg_comp2"]), 0))
        self.neg_weight = QDoubleSpinBox(minimum=1.0, maximum=1000.0, decimals=1, singleStep=1.0)
        self.neg_weight.setValue(params["bg_neg_weight"])
        box = QGroupBox("Fit")
        form = QFormLayout(box)
        form.addRow("Component 2", self.comp2)
        form.addRow("Oversubtraction penalty (× weight)", self.neg_weight)
        return box

    def save(self):
        save_params({"background": checked(self.choices), "bg_method": checked(self.methods),
                     "bg_comp2": self.comp2.currentData(), "bg_neg_weight": self.neg_weight.value()})
        self.accept()


class CalibrationDialog(QDialog):
    """Set the M3 radius (hence px -> nm) per recording on its background-subtracted projection.

    Click a plot (with the toolbar's zoom/pan off) or type the radius: Save then writes
    <output>/calibration.json marked manual, which later runs keep.  "Auto" on frame 1 goes back to
    the automatic calibration; on another frame it counts as manual (runs calibrate on frame 1).
    Recordings whose radius was not set here are not written.
    """

    def __init__(self, parent, jobs, width, rmin, d_m3):
        super().__init__(parent)
        self.setWindowTitle("Pixel to nm calibration (M3)")
        self.width, self.rmin, self.d_m3 = width, rmin, d_m3
        self.recs = [self.read_job(job) for job in jobs]
        self.frame_box = QSpinBox(minimum=1, maximum=min(len(r["frames"]) for r in self.recs), value=1)
        self.frame_box.valueChanged.connect(lambda _: self.draw())
        self.canvas = canvas(260 * len(self.recs))
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.canvas.mpl_connect("button_press_event", self.click)

        form = QGridLayout()
        form.addWidget(QLabel("Frame"), 0, 0)
        form.addWidget(self.frame_box, 0, 1)
        for i, r in enumerate(self.recs, 1):
            self.add_row(form, i, r)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"M3 = {d_m3:g} nm; strip width {width} px, auto search from rmin {rmin} px. "
                                "Click a plot to place M3."))
        layout.addLayout(form)
        for w in (self.toolbar, self.canvas, save_cancel(self)):
            layout.addWidget(w)
        self.resize(1000, 300 + 260 * len(self.recs))
        self.draw()

    def read_job(self, job):
        """A recording: r["n_m3"] and r["manual"] as saved for this width (None = not saved);
        r["frame"] = the frame the radius was set on in this window (None = not set here)."""
        path = os.path.join(job.out, "calibration.json")
        cal = cache.read_json(path)
        saved = cal.get("gap_pixels") == self.width
        return dict(label=job.label, frames=find_frames(job.qf), path=path, frame=None,
                    n_m3=float(cal["n_m3_pixels"]) if saved else None, manual=bool(saved and cal.get("manual")))

    def add_row(self, form, i, r):
        r["box"] = QDoubleSpinBox(minimum=1.0, maximum=1e5, decimals=1, suffix=" px")
        r["box"].setValue(self.profile(r)[2] if r["n_m3"] is None else r["n_m3"])
        r["box"].valueChanged.connect(lambda _: self.mark_set(r, manual=True))
        r["info"] = QLabel()
        auto = QPushButton("Auto")
        auto.clicked.connect(lambda: self.set_auto(r))
        for j, w in enumerate((QLabel(r["label"]), r["box"], auto, r["info"])):
            form.addWidget(w, i, j)

    def profile(self, r):
        """(radii, background-subtracted profile, auto M3) of the frame shown."""
        img = np.asarray(Image.open(r["frames"][self.frame_box.value() - 1]), dtype=float)
        data, radii, folded = calibrate(img, self.width, self.rmin, self.d_m3)
        return radii, folded, float(data["n_m3_pixels"])

    def draw(self):
        fig = self.canvas.figure
        fig.clear()
        axes = fig.subplots(len(self.recs), 1, squeeze=False)[:, 0]
        for ax, r in zip(axes, self.recs):
            radii, folded, r["auto"] = self.profile(r)
            ax.plot(radii, folded, lw=1, color="black")
            ax.axvspan(0, self.rmin, color="grey", alpha=0.15, lw=0)
            ax.axvline(r["auto"], color="grey", ls=":", lw=1)
            r["line"] = ax.axvline(r["box"].value(), color="#d93025", ls="--", lw=1.2)
            ax.set_xlim(0, min(radii[-1], 2.5 * max(r["box"].value(), r["auto"])))
            ax.set_title(f"{r['label']}  (grey dotted: auto M3 in this frame)", fontsize=9)
            ax.set_ylabel("I − convex hull", fontsize=8)
            ax.tick_params(labelsize=8)
            r["ax"] = ax
            self.update_marker(r)
        axes[-1].set_xlabel("radius from centre (px)")

    def mark_set(self, r, manual):
        r["manual"], r["frame"] = manual, self.frame_box.value()
        self.update_marker(r)

    def set_auto(self, r):
        r["box"].setValue(r["auto"])
        self.mark_set(r, manual=self.frame_box.value() != 1)

    def update_marker(self, r):
        n = r["box"].value()
        r["line"].set_xdata([n, n])
        cal = from_m3(n, self.d_m3)
        r["info"].setText(f"lambda_sdd = {cal['lambda_sdd']:.1f} nm·px,  {cal['inv_nm_per_pixel']:.4g} nm⁻¹/px  "
                          f"({'manual' if r['manual'] else 'automatic'})")
        self.canvas.draw_idle()

    def click(self, event):
        if self.toolbar.mode or event.xdata is None:
            return
        for r in self.recs:
            if event.inaxes is r["ax"]:
                r["box"].setValue(round(event.xdata, 1))

    def save(self):
        for r in self.recs:
            if r["frame"] is None:
                continue
            os.makedirs(os.path.dirname(r["path"]), exist_ok=True)
            cal = dict(from_m3(r["box"].value(), self.d_m3), rmin_pixels=self.rmin, gap_pixels=self.width,
                       image=os.path.abspath(r["frames"][0]), manual=r["manual"], set_on_frame=r["frame"])
            with open(r["path"], "w") as f:
                json.dump(cal, f, indent=2)
        self.accept()


GOOD, BAD, OFF = "#2f9e44", "#d93025", "grey"


def change_text(before, after, what):
    """'<what> reduced by x %' (green) or 'increased' (red), as HTML; '' if either is missing."""
    if not (np.isfinite(before) and np.isfinite(after) and before > 0):
        return ""
    change = 100.0 * (after - before) / before
    color = GOOD if change <= 0 else BAD
    word = "reduced" if change <= 0 else "increased"
    return (f"<span style='color:{color}'>{what} {word} by {abs(change):.1f}%</span> "
            f"<span style='color:grey'>({before:.2f}% → {after:.2f}%)</span>")


class NormalizationDialog(QDialog):
    """Normalisation of the loaded recordings (normalization.py): per frame by I0 from the beamline log,
    and one scale factor per recording.  The information is recomputed on every change; Apply / OK
    put it on the viewer, Cancel leaves the viewer as it was."""

    def __init__(self, parent, viewer):
        super().__init__(parent)
        self.setWindowTitle("Normalization")
        self.viewer, self.recs = viewer, viewer.recs
        settings = viewer.norm
        self.logs, self.status, self.flicker = [], [], []
        layout = QVBoxLayout(self)
        layout.addWidget(self.within_box(settings))
        layout.addWidget(self.between_box(settings))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
        layout.addWidget(buttons)
        self.resize(1100, 560)
        self.update_info()

    # ---------------------------------------------------------------- sections

    def within_box(self, settings):
        box = QGroupBox("Within sequence alignment")
        grid = QGridLayout()
        for j, title in enumerate(("Recording", "Beamline log (I0, I1 per frame)", "", "", "")):
            grid.addWidget(QLabel(f"<b>{title}</b>"), 0, j)
        for i, r in enumerate(self.recs, 1):
            edit = QLineEdit(settings["logs"].get(r.npy) or find_beamlog(r.qf))
            edit.setPlaceholderText("no log found next to the images")
            edit.setMinimumWidth(380)
            edit.editingFinished.connect(self.update_info)
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _=False, r=r, e=edit: self.pick_log(r, e))
            status, info = note(""), QLabel()
            info.setMinimumWidth(330)
            name = QLabel(r.label)
            name.setStyleSheet(f"color: {r.color}")
            for j, w in enumerate((name, edit, browse, status, info)):
                grid.addWidget(w, i, j)
            self.logs.append(edit)
            self.status.append(status)
            self.flicker.append(info)
        self.i0_box = QCheckBox("Normalize by I0 incident intensity")
        self.i0_box.setChecked(settings["use_i0"])
        self.i0_box.setToolTip("Each frame (kymograph column, tracked heights) × mean(I0) / I0 of that frame, "
                               "to reduce frame-to-frame flicker from the incident beam")
        self.i0_box.toggled.connect(lambda _: self.update_info())
        layout = QVBoxLayout(box)
        layout.addLayout(grid)
        layout.addWidget(self.i0_box)
        layout.addWidget(note("The log is found in the recording folder (<run>/<run>.log) or the image folder. "
                              "Flicker = frame-to-frame jitter of the kymograph column totals:\nRMS of the "
                              "differences of log(total) between consecutive kept frames / √2. Slow changes "
                              "(contraction, stretch) hardly count."))
        return box

    def between_box(self, settings):
        box = QGroupBox("Between sequence alignment")
        self.method = QComboBox()
        for key, text in BETWEEN:
            self.method.addItem(text, key)
        self.method.setCurrentIndex(max(self.method.findData(settings["method"]), 0))
        self.method.currentIndexChanged.connect(lambda _: self.update_info())
        self.frames_edit = QLineEdit(settings["frames"])
        self.frames_edit.setPlaceholderText("empty = all frames except the dropped ones")
        self.frames_edit.setToolTip("Frames (from 0) whose mean profiles set the scale factors; 2-20 (default) = "
                                    "the frames before the stimulus; empty = all frames except the dropped ones "
                                    "(dropped frames are always skipped)")
        self.frames_edit.editingFinished.connect(self.update_info)
        self.between_info = QLabel()
        self.between_info.setTextFormat(Qt.RichText)
        self.between_info.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.between_info.setMinimumWidth(480)
        form = QFormLayout()
        form.addRow("Align by", self.method)
        form.addRow("Frames used", self.frames_edit)
        form.addRow(note("Least squares: log10 of the mean 20 px meridional (r 50–599 px) and equatorial\n"
                         "(r 50–329 px) profiles, each projection weighs half, peak weights at M3, M6, I1,0,\n"
                         "I1,1. Backgrounds: hull or fit as set in Background…, of the mean profile.\n"
                         "Scale factors have geometric mean 1. The equatorial strip is read from the images\n"
                         "once (equator_w<W>_g1.npy beside the kymograph)."))
        row = QHBoxLayout(box)
        row.addLayout(form, 1)
        row.addWidget(self.between_info, 1)
        return box

    def pick_log(self, r, edit):
        start = edit.text() or os.path.dirname(os.path.normpath(r.qf))
        path, _ = QFileDialog.getOpenFileName(self, f"Beamline log of {r.label}", start,
                                              "Log files (*.log);;All files (*)")
        if path:
            edit.setText(path)
            self.update_info()

    # ---------------------------------------------------------------- information

    def settings(self):
        return dict(logs={r.npy: e.text().strip() for r, e in zip(self.recs, self.logs)},
                    use_i0=self.i0_box.isChecked(), method=self.method.currentData(),
                    frames=self.frames_edit.text().strip())

    def compute(self):
        s = self.settings()
        QApplication.setOverrideCursor(Qt.WaitCursor)  # the first between-sequence run reads the images
        try:
            return s, plan(self.recs, [s["logs"][r.npy] for r in self.recs], s["use_i0"], s["method"],
                           s["frames"], load_params())
        finally:
            QApplication.restoreOverrideCursor()

    def update_info(self):
        s, self.plan = self.compute()
        for w, status, info in zip(self.plan.within, self.status, self.flicker):
            status.setText(w.status)
            text = change_text(w.before, w.after, "Flicker")
            if not text:
                info.setText(f"<span style='color:grey'>flicker {w.before:.2f}%</span>")
            elif s["use_i0"]:
                info.setText(text)
            else:
                info.setText(f"<span style='color:{OFF}'>if on: </span>" + text)
        self.between_info.setText(self.between_text(self.plan.between))

    def between_text(self, b):
        if b is None:
            return "<span style='color:grey'>No between-sequence normalization.</span>"
        if isinstance(b, str):
            return f"<span style='color:{BAD}'>Cannot align: {b}</span>"
        rows = "".join(
            f"<tr><td style='color:{r.color}'>{r.label}</td><td align='right'>&nbsp;× {s:.3f}</td>"
            + (f"<td align='right' style='color:grey'>&nbsp;bg {m:,.0f}</td>" if b.metric is not None else "")
            + "</tr>" for r, s, m in zip(self.recs, b.scales, b.metric if b.metric is not None else b.scales))
        spread = "".join(f"<br>{name}: " + change_text(b.before[key], b.after[key], "spread")
                         for key, name in (("mer", "Meridian (r 50–599 px)"), ("eq", "Equator (r 50–329 px)")))
        return (f"<b>Scale factors</b><table>{rows}</table>"
                f"<br><b>Spread of the mean log level between sequences</b>{spread}")

    # ---------------------------------------------------------------- apply

    def apply(self):
        s, self.plan = self.compute()
        self.viewer.set_normalization(s, self.plan)
        save_params({"norm_i0": s["use_i0"], "norm_between": s["method"], "norm_frames": s["frames"]})

    def save(self):
        self.apply()
        self.accept()
