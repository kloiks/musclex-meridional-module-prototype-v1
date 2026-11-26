#!/usr/bin/env python3
"""GUI: M3 calibration, meridional kymographs, force overlay and peak tracking
(multiscale or U-Net) for one to three qf folders.

Each row: a quadrant-folded (qf_results) folder, an optional .ddf force file and
an output folder (default <qf folder>/kymo); empty rows are skipped.  "Clear second
folder" empties row 2 and drops its recording from the viewer.  "Process patterns using" one of three
detectors runs, per row:
    1. M3 calibration on the first frame  -> calibration.json, calibration_projection.png
    2. kymograph.make_kymograph           -> meridian_w<W>_g1.npy/.json/_log.png (one column per frame)
    3. kymo_force.kymo_force (if a .ddf)  -> ..._force.png/.json, _events.csv, _traces.csv
    4. the peak detection on the per-frame kymograph (tracks.csv, labels.csv named after
       config/expected_peaks.csv):
       "Multiscale peak detection"  multiscale.process -> multiscale_w<W>/
       "U-Net peak detection"       unet.process       -> unet_w<W>/
       "MaxFlow peak detection"     maxflow.process    -> maxflow_w<W>/ (guided by the
                                    multiscale anchors, so multiscale_w<W>/ is made too)
A step whose data and settings have not changed (cache.py) is not repeated.

"Expected peaks" edits config/expected_peaks.csv, "Parameters" config/parameters.json (dialogs.py).
"Normalization" scales the kymographs in the viewer: each frame by I0 from the beamline log, and
each recording by one factor (least squares or a background integral; normalization.py).

Afterwards the viewer (viewer.py) shows the force/length traces overlaid with a time slider,
the kymographs side by side and, below them, the tracked peaks of all recordings
overlaid (d in nm, and peak height) next to each recording's projection at the
slider time (all from the trigger; a row without a .ddf starts at t = 0).
"""
import os
import sys
import traceback

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (QApplication, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget)

import maxflow
import multiscale
import unet
from calibrate_m3 import load_or_calibrate
from dialogs import BackgroundDialog, CalibrationDialog, ExpectedPeaksDialog, NormalizationDialog, ParametersDialog
from frames import load_excluded
from kymo_force import kymo_force
from kymograph import find_frames, make_kymograph
from params import PEAKS_CSV, load_params
from recording import Job
from viewer import Viewer

MAX_ROWS = 3


class Worker(QThread):
    log = Signal(str)
    result = Signal(int, str, str, str, str)  # job index, label, npy, ddf ("" if none), tracking dir

    def __init__(self, jobs, params, detector):
        """detector: the module (multiscale, unet or maxflow) that tracks the peaks."""
        super().__init__()
        self.jobs, self.params, self.detector = jobs, params, detector

    def run(self):
        for i, job in enumerate(self.jobs):
            self.log.emit(f"=== {job.qf}")
            try:
                npy, track_dir = self.run_job(job)
                self.result.emit(i, job.label, npy, job.ddf, track_dir)
            except Exception:
                self.log.emit("FAILED:\n" + traceback.format_exc())
        self.log.emit("done")

    def run_job(self, job):
        """Calibration, kymograph, force figure and tracking; returns (kymograph .npy, tracking dir)."""
        p, width = self.params, self.params["width"]
        os.makedirs(job.out, exist_ok=True)
        cal, cached = load_or_calibrate(find_frames(job.qf)[0], job.out, width, p["m3_rmin"], p["d_m3"])
        self.log.emit(f"M3 at {cal['n_m3_pixels']:g} px, lambda_sdd = {cal['lambda_sdd']:.1f}"
                      + (" (manual)" if cal.get("manual") else " (cached)" if cached else ""))
        npy, cached = make_kymograph(job.qf, job.out, width, cal["inv_nm_per_pixel"])
        self.log.emit(f"wrote {npy}" + (" (cached)" if cached else ""))
        if job.ddf:
            self.log.emit(kymo_force(npy, job.ddf, (p["pos_lo"], p["pos_hi"]), p["stretch"],
                                     dark_fraction=p["dark_fraction"]))
        track_dir = os.path.join(job.out, f"{self.detector.__name__}_w{width}")
        summary, cached = self.detector.process(npy, track_dir, p, PEAKS_CSV, load_excluded(job.out))
        self.log.emit(summary + (" (cached tracking)" if cached else ""))
        return npy, track_dir


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
        """The Job, or None while there is no qf folder."""
        qf = self.qf.text().strip()
        return Job(qf, self.ddf.text().strip(), self.out.text().strip() or os.path.join(qf, "kymo")) if qf else None


def button(text, slot, tip=""):
    b = QPushButton(text)
    b.clicked.connect(lambda: slot())
    b.setToolTip(tip)
    return b


def lay_out(layout, *items):
    """Add widgets to a layout, with a stretch at the end."""
    for w in items:
        layout.addWidget(w)
    layout.addStretch()
    return layout


class KymoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meridional kymographs")
        self.loaded = []  # the Row of each recording in the viewer, in viewer order
        self.grid = QGridLayout()
        self.rows = [Row(self, self.grid, 0), Row(self, self.grid, 1, self.first_qf_parent)]
        self.rows[1].qf.setPlaceholderText("optional")
        self.add_button = button("+ Add third folder", self.toggle_third_row)
        self.clear_button = button("Clear second folder", self.clear_second_row,
                                   "Empty the second row and remove its kymograph from the viewer")
        self.detector_buttons = [
            button("Multiscale peak detection", lambda: self.run(multiscale),
                   "Create the kymographs and track the peaks by averaging frames to find weak peaks, "
                   "then following them down to single frames"),
            button("U-Net peak detection", lambda: self.run(unet),
                   "Create the kymographs and find the peaks in single frames with the new_w61 U-Net"),
            button("MaxFlow peak detection", lambda: self.run(maxflow),
                   "Create the kymographs, find the multiscale anchors, then detect peaks in single frames "
                   "inside corridors around the anchors and link them by one global min-cost flow"),
        ]
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumHeight(120)
        self.viewer = Viewer()
        self.make_layout()
        self.resize(1400, 1000)

    def make_layout(self):
        side = QVBoxLayout()
        for text, slot in (("Expected peaks…", lambda: ExpectedPeaksDialog(self).exec()),
                           ("Parameters…", lambda: ParametersDialog(self, self.jobs()).exec()),
                           ("Calibration…", self.open_calibration),
                           ("Background…", lambda: BackgroundDialog(self).exec()),
                           ("Normalization…", self.open_normalization)):
            side.addWidget(button(text, slot))
        paths = QHBoxLayout()  # no stretch inside: it would make the path rows spread out vertically
        paths.addLayout(self.grid, 1)
        paths.addLayout(side)
        paths.setAlignment(side, Qt.AlignTop)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addLayout(paths)
        layout.addLayout(lay_out(QHBoxLayout(), self.add_button, self.clear_button))
        layout.addLayout(lay_out(QHBoxLayout(), QLabel("Process patterns using"), *self.detector_buttons))
        lay_out(layout, self.log, self.viewer)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(content)
        QVBoxLayout(self).addWidget(scroll)

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
            self.attempt("Viewer", lambda: self.viewer.remove(i))

    def attempt(self, what, action):
        """Run ``action``; a failure goes to the log."""
        try:
            action()
        except Exception:
            self.log.appendPlainText(f"{what} FAILED:\n" + traceback.format_exc())

    def jobs(self):
        return [row.job() for row in self.rows if row.job()]

    def selected_jobs(self):
        """The jobs; without any, says so in the log."""
        jobs = self.jobs()
        if not jobs:
            self.log.appendPlainText("Select at least one QF folder.")
        return jobs

    def open_calibration(self):
        jobs, p = self.selected_jobs(), load_params()
        if jobs:
            self.attempt("Calibration", lambda: CalibrationDialog(self, jobs, p["width"], p["m3_rmin"],
                                                                  p["d_m3"]).exec())

    def open_normalization(self):
        """Needs the recordings in the viewer (it normalises their kymographs and heights)."""
        if not self.viewer.recs:
            self.log.appendPlainText("Normalization: run a peak detection first, so the recordings are in the viewer.")
            return
        self.attempt("Normalization", lambda: NormalizationDialog(self, self.viewer).exec())
        if self.viewer.norm_error:
            self.log.appendPlainText(f"Between-sequence normalization not applied: {self.viewer.norm_error}")

    def run(self, detector):
        jobs = self.selected_jobs()
        if not jobs:
            return
        rows = [row for row in self.rows if row.job()]
        self.set_running(True)
        self.results = []  # (Row, what Viewer.load takes)
        self.worker = Worker(jobs, load_params(), detector)
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.result.connect(lambda i, *result: self.results.append((rows[i], result)))
        self.worker.finished.connect(self.done)
        self.worker.start()

    def done(self):
        self.set_running(False)
        self.loaded = [row for row, _ in self.results]
        self.attempt("Viewer", lambda: self.viewer.load([result for _, result in self.results]))

    def set_running(self, running):
        for b in self.detector_buttons + [self.clear_button]:
            b.setEnabled(not running)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KymoWindow()
    window.show()
    sys.exit(app.exec())
