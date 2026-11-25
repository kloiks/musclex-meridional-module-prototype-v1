#!/usr/bin/env python3
"""GUI: M3 calibration, meridional kymographs, force overlay and peak tracking
(multiscale or U-Net) for one to three qf folders.

Each row: a quadrant-folded (qf_results) folder, an optional .ddf force file and
an output folder (default <qf folder>/kymo); empty rows are skipped.  "Clear second
folder" empties row 2 and drops its recording from the viewer.  "Create kymographs
and process multiscale" runs, per row:
    1. M3 calibration on the first frame  -> calibration.json, calibration_projection.png
    2. kymograph.make_kymograph           -> meridian_w<W>_g1.npy/.json/_log.png (one column per frame)
    3. kymo_force.kymo_force (if a .ddf)  -> ..._force.png, _events.csv, _traces.csv
    4. multiscale.process on the per-frame kymograph -> multiscale_w<W>/
       (tracks.csv, labels.csv named after config/expected_peaks.csv)
"Run Unet Detection" runs the same steps with unet.process -> unet_w<W>/ instead of 4.

"Expected peaks" edits config/expected_peaks.csv, "Parameters" config/parameters.json.

Afterwards the viewer shows the force/length traces overlaid with a time slider,
the kymographs side by side and, below them, the tracked peaks of all recordings
overlaid (d in nm, and peak height) next to each recording's projection at the
slider time (all on the DMC clock; a row without a .ddf starts at t = 0).
"""
import csv
import json
import os
import re
import sys
import traceback
import warnings

import numpy as np
from PIL import Image
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, NullFormatter
from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QRadioButton, QScrollArea,
                               QSlider, QSpinBox, QStyle, QStyleOptionSlider, QTableWidget, QTableWidgetItem,
                               QToolButton, QVBoxLayout, QWidget)

import multiscale
import unet
from calibrate_m3 import D_M3_NM, calibrate, calibrate_file, from_m3
from frames import blank_frames, format_frames, load_excluded, parse_frames, save_excluded
from kymograph import cached_frames_kymograph, find_frames, make_kymograph
from kymo_force import column_time, from_trigger, kymo_force, load_traces

MAX_ROWS = 3
SLIDER_STYLE = """
QSlider { max-height: 14px; }
QSlider::groove:horizontal { height: 4px; background: #d5d9de; border-radius: 2px; }
QSlider::sub-page:horizontal { height: 4px; background: #8ab4f8; border-radius: 2px; }
QSlider::handle:horizontal { width: 10px; margin: -4px 0; border-radius: 5px; background: #1a73e8; }
"""
STEPS = (("<<", -2), ("<", -1), (">", 1), (">>", 2))  # button, groups moved
COLORS = ("#2b7fff", "#e8590c", "#2f9e44")
BG_D_NM = (80.0, 3.0)  # d range (nm) over which the background panel integrates the background
NM_TICKS = (200, 100, 70, 50, 40, 30, 25, 20, 15, 14.5, 12, 10, 8, 7, 6,
            5.9, 5.1, 5, 4, 3.5, 3, 2.7, 2.5, 2, 1.5, 1)

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
PARAMS_JSON = os.path.join(CONFIG, "parameters.json")
PEAKS_CSV = os.path.join(CONFIG, "expected_peaks.csv")

DEFAULTS = dict(multiscale.DEFAULTS, **unet.DEFAULTS, width=20, m3_rmin=250, d_m3=D_M3_NM, pos_lo=0, pos_hi=1200, stretch=1.4)
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
                             ("select", "Peak kept per frame", ("nearest", "snr")),
                             ("ref", "Corridor reference", ("coarse", "parent")),
                             ("seed_anchors", "Seed anchors at expected peaks (needs coarse)", ("yes", "no"))]),
    ("Labelling", [("tol_pct", "Default tolerance (% of d)", None),
                   ("min_frames", "Min frames of a labelled track", None)]),
]


def save_params(values):
    """Update config/parameters.json with ``values``, keeping the other keys."""
    params = load_params()
    params.update(values)
    os.makedirs(CONFIG, exist_ok=True)
    with open(PARAMS_JSON, "w") as f:
        json.dump(params, f, indent=2)


def load_params():
    params = dict(DEFAULTS)
    try:
        with open(PARAMS_JSON) as f:
            params.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return params


class Worker(QThread):
    log = Signal(str)
    result = Signal(int, str, str, str, str)  # job index, label, npy, ddf ("" if none), tracking dir

    def __init__(self, jobs, width, rmin, params, detector=multiscale):
        """detector: the module (multiscale or unet) that tracks the peaks."""
        super().__init__()
        self.detector = detector
        self.jobs, self.width, self.rmin, self.params = jobs, width, rmin, params

    def run(self):
        p = self.params
        for i, (qf, ddf, out) in enumerate(self.jobs):
            self.log.emit(f"=== {qf}")
            try:
                os.makedirs(out, exist_ok=True)
                cal, cached = self.calibration(qf, out)
                self.log.emit(f"M3 at {cal['n_m3_pixels']:g} px, lambda_sdd = {cal['lambda_sdd']:.1f}"
                              + (" (manual)" if cal.get("manual") else " (cached)" if cached else ""))
                npy, cached = make_kymograph(qf, out, self.width, 1, cal["inv_nm_per_pixel"])
                self.log.emit(f"wrote {npy}" + (" (from cached per-frame kymograph)" if cached else ""))
                if ddf:
                    self.log.emit(kymo_force(npy, ddf, (p["pos_lo"], p["pos_hi"]), p["stretch"]))
                ms_dir = os.path.join(out, f"{self.detector.__name__}_w{self.width}")
                summary, cached = self.detector.process(npy, ms_dir, p, PEAKS_CSV, load_excluded(out))
                self.log.emit(summary + (" (cached tracking)" if cached else ""))
                label = os.path.basename(os.path.dirname(os.path.abspath(qf)))
                self.result.emit(i, label, npy, ddf or "", ms_dir)
            except Exception:
                self.log.emit("FAILED:\n" + traceback.format_exc())
        self.log.emit("done")

    def calibration(self, qf, out):
        """(calibration, cached): reuse out/calibration.json if made from the same image and width
        (and, unless M3 was set by hand in the Calibration window, the same rmin and d_M3)."""
        path, image = os.path.join(out, "calibration.json"), os.path.abspath(find_frames(qf)[0])
        d_m3 = self.params["d_m3"]
        try:
            with open(path) as f:
                cal = json.load(f)
            if (cal["image"], cal["gap_pixels"]) == (image, self.width):
                if cal.get("manual"):
                    cal.update(from_m3(cal["n_m3_pixels"], d_m3))
                    return cal, True
                if (cal["rmin_pixels"], cal["d_m3_nm"]) == (self.rmin, d_m3):
                    return cal, True
        except (OSError, KeyError, ValueError):
            pass
        return calibrate_file(image, self.width, self.rmin, path,
                              os.path.join(out, "calibration_projection.png"), d_m3), False


class Row:
    """qf folder, .ddf and output folder pickers for one recording."""

    def __init__(self, parent, grid, i, qf_start=lambda: ""):
        """qf_start: callable giving the folder the QF dialog opens in while this row is empty."""
        self.parent, self.qf_start = parent, qf_start
        self.qf, self.ddf, self.out = QLineEdit(), QLineEdit(), QLineEdit()
        self.out.setPlaceholderText("<qf folder>/kymo")
        buttons = [("QF folder…", self.pick_qf), ("DDF file…", self.pick_ddf), ("Output…", self.pick_out)]
        label = QLabel(f"{i + 1}")
        grid.addWidget(label, 3 * i, 0)
        self.widgets = [label, self.qf, self.ddf, self.out]
        for j, (edit, (text, slot)) in enumerate(zip((self.qf, self.ddf, self.out), buttons)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            grid.addWidget(button, 3 * i + j, 1)
            grid.addWidget(edit, 3 * i + j, 2)
            self.widgets.append(button)

    def remove(self, grid):
        for w in self.widgets:
            grid.removeWidget(w)
            w.deleteLater()

    def pick_qf(self):
        path = QFileDialog.getExistingDirectory(self.parent, "Quadrant-folded images folder",
                                                self.qf.text() or self.qf_start())
        if path:
            self.qf.setText(path)

    def pick_ddf(self):
        """Opens in the parent of this row's qf folder; falls back to the last .ddf folder
        (remembered across sessions)."""
        settings = QSettings("musclex", "meridional-kymo")
        qf = self.qf.text().strip()
        start = (self.ddf.text() or (os.path.dirname(os.path.normpath(qf)) if qf else "")
                 or settings.value("last_ddf_dir", ""))
        path, _ = QFileDialog.getOpenFileName(self.parent, "Force file", start,
                                              "DDF files (*.ddf);;All files (*)")
        if path:
            self.ddf.setText(path)
            settings.setValue("last_ddf_dir", os.path.dirname(path))

    def pick_out(self):
        path = QFileDialog.getExistingDirectory(self.parent, "Output folder", self.out.text() or self.qf.text())
        if path:
            self.out.setText(path)

    def job(self):
        qf = self.qf.text().strip()
        if not qf:
            return None
        return qf, self.ddf.text().strip() or None, self.out.text().strip() or os.path.join(qf, "kymo")


# ------------------------------------------------------------------ dialogs

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
        remove.clicked.connect(lambda: [self.table.removeRow(i) for i in
                                        sorted({ix.row() for ix in self.table.selectedIndexes()}, reverse=True)])
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
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
        layout.addWidget(buttons)
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

    def save(self):
        rows = []
        for i in range(self.table.rowCount()):
            name, d, tol = (self.table.item(i, j).text().strip() if self.table.item(i, j) else ""
                            for j in range(3))
            if not name and not d:
                continue
            if self.table.item(i, 2).data(Qt.UserRole) and tol == self.default_tol:
                tol = ""  # still the default: keep following the parameter
            try:
                rows.append((name, float(d), float(tol) if tol else None))
                if not name:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Expected peaks", f"Row {i + 1}: needs a name, a number for d "
                                                            "and an empty or numeric tolerance.")
                return
        with open(PEAKS_CSV, "w") as f:
            f.writelines(self.header)
            f.write("name,d_nm,tol_pct\n")
            for name, d, tol in sorted(rows, key=lambda r: -r[1]):
                f.write(f"{name},{d:g},{'' if tol is None else f'{tol:g}'}\n")
        self.accept()


class ParametersDialog(QDialog):
    """Edit config/parameters.json (missing keys use the defaults) and, per selected recording,
    the frames to exclude: filled in with the blank frames found; an edited list is kept in
    <output>/excluded_frames.json."""

    def __init__(self, parent, jobs=()):
        super().__init__(parent)
        self.setWindowTitle("Parameters")
        self.widgets = {}
        content = QWidget()
        columns = QHBoxLayout(content)
        sides = [QVBoxLayout(), QVBoxLayout()]
        for side in sides:
            columns.addLayout(side)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(content)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        sides[0].addWidget(self.exclude_box(jobs, load_params()["width"]))
        for title, items in PARAM_GROUPS:
            box = QGroupBox(title)
            form = QFormLayout(box)
            for key, label, choices in items:
                w = self.widget(DEFAULTS[key], choices)
                self.widgets[key] = w
                form.addRow(label, w)
            sides[title.startswith(("Multiscale", "Labelling"))].addWidget(box)
        for side in sides:
            side.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel
                                   | QDialogButtonBox.RestoreDefaults)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)
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

    def exclude_box(self, jobs, width):
        """One "frames to exclude" field per recording; r["auto"] = it follows the detection."""
        box = QGroupBox("Frames to exclude (multiscale and U-Net)")
        form = QFormLayout(box)
        self.recs = []
        QApplication.setOverrideCursor(Qt.WaitCursor)  # without a cached kymograph the images are read
        try:
            for qf, _, out in jobs:
                label = os.path.basename(os.path.dirname(os.path.abspath(qf)))
                try:
                    kymo = cached_frames_kymograph(find_frames(qf), out, width)[0]
                    saved = load_excluded(out)
                except Exception as e:
                    form.addRow(label, QLabel(f"cannot read: {e}"))
                    continue
                r = dict(out=out, label=label, totals=kymo.sum(axis=0), auto=saved is None,
                         edit=QLineEdit("" if saved is None else format_frames(saved)), info=QLabel())
                r["edit"].setPlaceholderText("none")
                r["edit"].setMinimumWidth(180)
                r["edit"].textEdited.connect(lambda _="", r=r: self.set_auto(r, False))
                auto = QPushButton("Auto")
                auto.setToolTip("Use the blank frames found in this recording")
                auto.clicked.connect(lambda _=False, r=r: self.set_auto(r, True))
                row = QHBoxLayout()
                for w in (r["edit"], auto, r["info"]):
                    row.addWidget(w)
                form.addRow(label, row)
                self.recs.append(r)
        finally:
            QApplication.restoreOverrideCursor()
        note = QLabel("Frames from 0, e.g. 0-1, 40-60. Filled in with the blank frames found."
                      if jobs else "Select a QF folder first.")
        note.setStyleSheet("color: grey")
        form.addRow(note)
        return box

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
    def widget(default, choices):
        if choices:
            w = QComboBox()
            w.addItems(choices)
        elif isinstance(default, int):
            w = QSpinBox(minimum=0, maximum=100000)
        else:
            w = QDoubleSpinBox(minimum=0.0, maximum=1e6, decimals=3, singleStep=0.1)
        return w

    def set_values(self, values):
        for key, w in self.widgets.items():
            if isinstance(w, QComboBox):
                w.setCurrentText(str(values[key]))
            else:
                w.setValue(values[key])

    def save(self):
        lists = []
        for r in self.recs:
            try:
                frames = None if r["auto"] else parse_frames(r["edit"].text())
                n = len(r["totals"])
                if frames and frames[-1] >= n:
                    raise ValueError(f"frame {frames[-1]} is past the last frame ({n - 1})")
                if frames is not None and len(frames) == n:
                    raise ValueError("every frame is excluded")
            except ValueError as e:
                QMessageBox.warning(self, "Frames to exclude", f"{r['label']}: {e}")
                return
            lists.append((r["out"], frames))
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
        method_box = QGroupBox("Background")
        method_layout = QVBoxLayout(method_box)
        self.methods = {}
        for key, text in self.METHODS:
            self.methods[key] = QRadioButton(text)
            method_layout.addWidget(self.methods[key])
        self.methods.get(params["bg_method"], self.methods["hull"]).setChecked(True)
        self.fit_box = QGroupBox("Fit")
        fit_form = QFormLayout(self.fit_box)
        self.comp2 = QComboBox()
        for key, text in self.COMP2:
            self.comp2.addItem(text, key)
        self.comp2.setCurrentIndex(max(self.comp2.findData(params["bg_comp2"]), 0))
        self.neg_weight = QDoubleSpinBox(minimum=1.0, maximum=1000.0, decimals=1, singleStep=1.0)
        self.neg_weight.setValue(params["bg_neg_weight"])
        fit_form.addRow("Component 2", self.comp2)
        fit_form.addRow("Oversubtraction penalty (× weight)", self.neg_weight)
        method_layout.addWidget(self.fit_box)
        self.methods["fit"].toggled.connect(self.fit_box.setEnabled)
        self.fit_box.setEnabled(self.methods["fit"].isChecked())
        box = QGroupBox("Computed from")
        form = QVBoxLayout(box)
        self.buttons = {}
        for key, text in self.CHOICES:
            self.buttons[key] = QRadioButton(text)
            form.addWidget(self.buttons[key])
        self.buttons.get(params["background"], self.buttons["grouped"]).setChecked(True)
        note = QLabel("Applies on the next run (tracking is recomputed; the kymographs are reused).")
        note.setStyleSheet("color: grey")
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        for w in (method_box, box, note, buttons):
            layout.addWidget(w)

    def save(self):
        save_params({"background": next(k for k, b in self.buttons.items() if b.isChecked()),
                     "bg_method": next(k for k, b in self.methods.items() if b.isChecked()),
                     "bg_comp2": self.comp2.currentData(), "bg_neg_weight": self.neg_weight.value()})
        self.accept()


class CalibrationDialog(QDialog):
    """Set the M3 radius (hence px -> nm) per recording on its background-subtracted projection.

    Click a plot (with the toolbar's zoom/pan off) or type the radius; Save writes
    <output>/calibration.json marked manual, which later runs keep.
    """

    def __init__(self, parent, jobs, width, rmin, d_m3):
        super().__init__(parent)
        self.setWindowTitle("Pixel to nm calibration (M3)")
        self.width, self.rmin, self.d_m3 = width, rmin, d_m3
        self.recs = []
        for qf, _, out in jobs:
            frames = find_frames(qf)
            path = os.path.join(out, "calibration.json")
            try:
                with open(path) as f:
                    cal = json.load(f)
            except (OSError, ValueError):
                cal = None
            self.recs.append(dict(label=os.path.basename(os.path.dirname(os.path.abspath(qf))),
                                  frames=frames, path=path, cal=cal))
        n = min(len(r["frames"]) for r in self.recs)
        self.frame_box = QSpinBox(minimum=1, maximum=n, value=1)
        self.frame_box.valueChanged.connect(self.draw)
        self.canvas = canvas(260 * len(self.recs))
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.canvas.mpl_connect("button_press_event", self.click)

        form = QGridLayout()
        form.addWidget(QLabel("Frame"), 0, 0)
        form.addWidget(self.frame_box, 0, 1)
        for i, r in enumerate(self.recs):
            r["box"] = QDoubleSpinBox(minimum=1.0, maximum=1e5, decimals=1, suffix=" px")
            r["info"] = QLabel()
            auto = QPushButton("Auto")
            auto.clicked.connect(lambda _=False, r=r: r["box"].setValue(r["auto"]))
            form.addWidget(QLabel(r["label"]), i + 1, 0)
            form.addWidget(r["box"], i + 1, 1)
            form.addWidget(auto, i + 1, 2)
            form.addWidget(r["info"], i + 1, 3)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"M3 = {d_m3:g} nm; strip width {width} px, auto search from rmin {rmin} px. "
                                "Click a plot to place M3."))
        layout.addLayout(form)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(buttons)
        self.resize(1000, 300 + 260 * len(self.recs))
        self.draw(init=True)

    def draw(self, *_, init=False):
        fig = self.canvas.figure
        fig.clear()
        axes = fig.subplots(len(self.recs), 1, squeeze=False)[:, 0]
        for ax, r in zip(axes, self.recs):
            img = np.asarray(Image.open(r["frames"][self.frame_box.value() - 1]), dtype=float)
            data, radii, folded = calibrate(img, self.width, self.rmin, self.d_m3)
            r["auto"] = float(data["n_m3_pixels"])
            if init:
                manual = r["cal"] and r["cal"].get("gap_pixels") == self.width
                r["box"].setValue(float(r["cal"]["n_m3_pixels"]) if manual else r["auto"])
                r["box"].valueChanged.connect(lambda _=0, r=r: self.update_marker(r))
            ax.plot(radii, folded, lw=1, color="black")
            ax.axvspan(0, self.rmin, color="grey", alpha=0.15, lw=0)
            ax.axvline(r["auto"], color="grey", ls=":", lw=1)
            r["line"] = ax.axvline(r["box"].value(), color="#d93025", ls="--", lw=1.2)
            ax.set_xlim(0, min(radii[-1], 2.5 * max(r["box"].value(), r["auto"])))
            ax.set_title(f"{r['label']}  (grey dotted: auto M3 in this frame)", fontsize=9)
            ax.set_ylabel("I − convex hull", fontsize=8)
            ax.tick_params(labelsize=8)
            r["ax"] = ax
            self.update_marker(r, redraw=False)
        axes[-1].set_xlabel("radius from centre (px)")
        self.canvas.draw_idle()

    def update_marker(self, r, redraw=True):
        n = r["box"].value()
        r["line"].set_xdata([n, n])
        cal = from_m3(n, self.d_m3)
        r["info"].setText(f"lambda_sdd = {cal['lambda_sdd']:.1f} nm·px,  {cal['inv_nm_per_pixel']:.4g} nm⁻¹/px")
        if redraw:
            self.canvas.draw_idle()

    def click(self, event):
        if self.toolbar.mode or event.xdata is None:
            return
        for r in self.recs:
            if event.inaxes is r["ax"]:
                r["box"].setValue(round(event.xdata, 1))

    def save(self):
        for r in self.recs:
            os.makedirs(os.path.dirname(r["path"]), exist_ok=True)
            cal = dict(from_m3(r["box"].value(), self.d_m3), rmin_pixels=self.rmin, gap_pixels=self.width,
                       image=os.path.abspath(r["frames"][0]), manual=r["box"].value() != r["auto"]
                       or bool(r["cal"] and r["cal"].get("manual")),
                       set_on_frame=self.frame_box.value())
            with open(r["path"], "w") as f:
                json.dump(cal, f, indent=2)
        self.accept()


# ------------------------------------------------------------------ viewer

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
    unmatched = unet.unmatched_names(data["r"][0], labels) if detector == "unet" else {}
    return dict(detector=detector, unmatched=unmatched, scales=data["scales"], frames=frames, r=data["r"], height=data["height"], labels=labels,
                base=base, pos=pos[keep], n_source=kymo.shape[1], p=p, shared=multiscale.shared_background(base, pos[keep], p),
                rmin=run["rmin"], rmax=run["rmax"], col_of={int(f): i for i, f in enumerate(frames)},
                grouped={}, profiles={}, backgrounds={}, flats={})


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
        scales = [str(s) for s in sorted({int(s) for r in self.recs for s in r["ms"]["scales"]})]
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
            r["k"] = int(np.argmin(np.abs(ms["scales"] - self.s_want)))
            r["s"] = s = int(ms["scales"][r["k"]])
            if s not in ms["grouped"]:  # sliding mean of s frames, as the tracker saw it
                g = ms["grouped"][s] = multiscale.bin_time(ms["base"], s)
                cols = range(g.shape[1])
                prof = ms["profiles"][s] = np.column_stack([multiscale.smoothed(g[:, t], ms["p"]["smooth"])
                                                            for t in cols])
                ms["backgrounds"][s] = multiscale.column_backgrounds(g, ms["pos"], ms["p"])
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
        ax_t.set_title(("U-Net tracks" if unet_run else f"tracked peaks at grouping scale {s_want}"), fontsize=9)
        ax_h.set_yscale("log")
        heights = np.concatenate(heights) if heights else np.empty(0)
        if heights.size:
            ax_h.set_ylim(heights.min() / 2.0, heights.max() * 2.0)
        ax_h.set_ylabel("peak height above baseline", fontsize=9)
        ax_h.set_title(("U-Net peak heights" + (f" (mean of {self.height_smooth.value()} frames)"
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
                flat, sigma = multiscale.profile_noise(ms["grouped"][s][:, col], ms["p"],
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


class KymoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meridional kymographs")
        self.grid = QGridLayout()
        self.rows = [Row(self, self.grid, i, self.first_qf_parent if i else (lambda: "")) for i in range(2)]
        self.rows[1].qf.setPlaceholderText("optional")
        self.loaded = []  # the Row of each recording in the viewer, in viewer order
        self.add_button = QPushButton("+ Add third folder")
        self.add_button.clicked.connect(self.toggle_third_row)
        self.clear_button = QPushButton("Clear second folder")
        self.clear_button.setToolTip("Empty the second row and remove its kymograph from the viewer")
        self.clear_button.clicked.connect(self.clear_second_row)
        peaks_button, params_button = QPushButton("Expected peaks…"), QPushButton("Parameters…")
        cal_button, bg_button = QPushButton("Calibration…"), QPushButton("Background…")
        bg_button.clicked.connect(lambda: BackgroundDialog(self).exec())
        peaks_button.clicked.connect(lambda: ExpectedPeaksDialog(self).exec())
        params_button.clicked.connect(lambda: self.open_parameters())
        cal_button.clicked.connect(self.open_calibration)

        self.run_button = QPushButton("Create kymographs and process multiscale")
        self.run_button.clicked.connect(lambda: self.run(multiscale))
        settings = QHBoxLayout()
        settings.addWidget(self.run_button)
        self.unet_button = QPushButton("Run Unet Detection")
        self.unet_button.setToolTip("Create the kymographs and find the peaks with the new_w61 U-Net "
                                    "instead of multiscale")
        self.unet_button.clicked.connect(lambda: self.run(unet))
        settings.addWidget(self.unet_button)
        settings.addStretch()
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumHeight(120)
        self.viewer = Viewer()

        side = QVBoxLayout()
        side.addWidget(peaks_button)
        side.addWidget(params_button)
        side.addWidget(cal_button)
        side.addWidget(bg_button)
        paths = QHBoxLayout()  # no stretch inside: it would make the path rows spread out vertically
        paths.addLayout(self.grid, 1)
        paths.addLayout(side)
        paths.setAlignment(side, Qt.AlignTop)
        top = QHBoxLayout()
        top.addWidget(self.add_button)
        top.addWidget(self.clear_button)
        top.addStretch()
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addLayout(paths)
        layout.addLayout(top)
        layout.addLayout(settings)
        layout.addWidget(self.log)
        layout.addWidget(self.viewer)
        layout.addStretch()
        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(content)
        QVBoxLayout(self).addWidget(scroll)
        self.resize(1400, 1000)

    def first_qf_parent(self):
        """Start folder for rows 2 and 3: the grandparent of the first row's qf folder."""
        qf = self.rows[0].qf.text().strip()
        return os.path.dirname(os.path.dirname(os.path.normpath(qf))) if qf else ""

    def toggle_third_row(self):
        if len(self.rows) < MAX_ROWS:
            self.rows.append(Row(self, self.grid, len(self.rows), self.first_qf_parent))
            self.add_button.setText("− Remove third folder")
        else:
            self.rows.pop().remove(self.grid)
            self.add_button.setText("+ Add third folder")

    def clear_second_row(self):
        row = self.rows[1]
        for edit in (row.qf, row.ddf, row.out):
            edit.clear()
        if row in self.loaded:
            i = self.loaded.index(row)
            del self.loaded[i]
            try:
                self.viewer.remove(i)
            except Exception:
                self.log.appendPlainText("Viewer FAILED:\n" + traceback.format_exc())

    def open_parameters(self):
        ParametersDialog(self, [j for j in (r.job() for r in self.rows) if j]).exec()

    def open_calibration(self):
        jobs = [j for j in (r.job() for r in self.rows) if j]
        if not jobs:
            self.log.appendPlainText("Select at least one QF folder.")
            return
        try:
            p = load_params()
            CalibrationDialog(self, jobs, p["width"], p["m3_rmin"], p["d_m3"]).exec()
        except Exception:
            self.log.appendPlainText("Calibration FAILED:\n" + traceback.format_exc())

    def run(self, detector):
        rows = [r for r in self.rows if r.job()]
        if not rows:
            self.log.appendPlainText("Select at least one QF folder.")
            return
        self.run_button.setEnabled(False)
        self.unet_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.results = []
        p = load_params()
        self.worker = Worker([r.job() for r in rows], p["width"], p["m3_rmin"], p, detector)
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.result.connect(lambda i, *r: self.results.append((rows[i], r)))
        self.worker.finished.connect(self.done)
        self.worker.start()

    def done(self):
        self.run_button.setEnabled(True)
        self.unet_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.loaded = [row for row, _ in self.results]
        try:
            self.viewer.load([r for _, r in self.results])
        except Exception:
            self.log.appendPlainText("Viewer FAILED:\n" + traceback.format_exc())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KymoWindow()
    window.show()
    sys.exit(app.exec())
