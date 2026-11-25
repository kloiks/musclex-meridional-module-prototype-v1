#!/usr/bin/env python3
"""GUI: M3 calibration, meridional kymographs, force overlay and peak tracking
(multiscale or U-Net) for one to three qf folders.

Each row: a quadrant-folded (qf_results) folder, an optional .ddf force file and
an output folder (default <qf folder>/kymo); empty rows are skipped.  "Clear second
folder" empties row 2 and drops its recording from the viewer.  "Create kymographs
and process multiscale" runs, per row:
    1. M3 calibration on the first frame  -> calibration.json, calibration_projection.png
    2. kymograph.make_kymograph           -> meridian_w<W>_g1.npy/.json/_log.png (one column per frame)
    3. kymo_force.kymo_force (if a .ddf)  -> ..._force.png/.json, _events.csv, _traces.csv
    4. multiscale.process on the per-frame kymograph -> multiscale_w<W>/
       (tracks.csv, labels.csv named after config/expected_peaks.csv)
"Run Unet Detection" runs the same steps with unet.process -> unet_w<W>/ instead of 4.
A step whose data and settings have not changed (cache.py) is not repeated.

"Expected peaks" edits config/expected_peaks.csv, "Parameters" config/parameters.json (dialogs.py).

Afterwards the viewer (viewer.py) shows the force/length traces overlaid with a time slider,
the kymographs side by side and, below them, the tracked peaks of all recordings
overlaid (d in nm, and peak height) next to each recording's projection at the
slider time (all from the trigger; a row without a .ddf starts at t = 0).
"""
import json
import os
import sys
import traceback

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (QApplication, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget)

import multiscale
import unet
from calibrate_m3 import calibrate_file, from_m3
from dialogs import (PEAKS_CSV, BackgroundDialog, CalibrationDialog, ExpectedPeaksDialog, ParametersDialog,
                     load_params)
from frames import load_excluded
from kymo_force import kymo_force
from kymograph import find_frames, make_kymograph
from viewer import Viewer

MAX_ROWS = 3


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
                    self.log.emit(kymo_force(npy, ddf, (p["pos_lo"], p["pos_hi"]), p["stretch"],
                                             dark_fraction=p["dark_fraction"]))
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
