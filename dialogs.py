"""Parameters (config/parameters.json) and the dialogs that edit them, the expected peaks and the calibration."""
import csv
import json
import os

import numpy as np
from PIL import Image
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QRadioButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

import multiscale
import unet
from calibrate_m3 import D_M3_NM, calibrate, from_m3
from frames import blank_frames, format_frames, load_excluded, parse_frames, save_excluded
from kymograph import cached_frames_kymograph, find_frames
from viewer import canvas

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

    Click a plot (with the toolbar's zoom/pan off) or type the radius: Save then writes
    <output>/calibration.json marked manual, which later runs keep.  "Auto" on frame 1 goes back to
    the automatic calibration; on another frame it counts as manual (runs calibrate on frame 1).
    Recordings whose radius was not set here are not written.
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
            auto.clicked.connect(lambda _=False, r=r: self.set_auto(r))
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
                saved = r["cal"] and r["cal"].get("gap_pixels") == self.width
                r["box"].setValue(float(r["cal"]["n_m3_pixels"]) if saved else r["auto"])
                # frame: the one the radius was set on in this window, None = not set here
                r["manual"], r["frame"] = bool(saved and r["cal"].get("manual")), None
                r["box"].valueChanged.connect(lambda _=0, r=r: self.mark_set(r, manual=True))
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

    def mark_set(self, r, manual):
        r["manual"], r["frame"] = manual, self.frame_box.value()
        self.update_marker(r)

    def set_auto(self, r):
        r["box"].setValue(r["auto"])
        self.mark_set(r, manual=self.frame_box.value() != 1)

    def update_marker(self, r, redraw=True):
        n = r["box"].value()
        r["line"].set_xdata([n, n])
        cal = from_m3(n, self.d_m3)
        r["info"].setText(f"lambda_sdd = {cal['lambda_sdd']:.1f} nm·px,  {cal['inv_nm_per_pixel']:.4g} nm⁻¹/px  "
                          f"({'manual' if r['manual'] else 'automatic'})")
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
            if r["frame"] is None:
                continue
            os.makedirs(os.path.dirname(r["path"]), exist_ok=True)
            cal = dict(from_m3(r["box"].value(), self.d_m3), rmin_pixels=self.rmin, gap_pixels=self.width,
                       image=os.path.abspath(r["frames"][0]), manual=r["manual"], set_on_frame=r["frame"])
            with open(r["path"], "w") as f:
                json.dump(cal, f, indent=2)
        self.accept()
