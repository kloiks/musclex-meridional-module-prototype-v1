#!/usr/bin/env python3
"""Regression tests: the whole pipeline on two recordings, against tests/reference/.

The images and .ddf are read from STRUCTURE (or $KYMO_TEST_DATA; the tests that need them are
skipped without it); everything is written to a temporary folder.  The config is the snapshot in
tests/reference/config, so editing the app's parameters does not change the result.

    python -m unittest discover tests -v
    python tests/test_outputs.py --update    # accept the current outputs as the new reference
"""
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
from PIL import Image

import kymo_force
import kymograph
import multiscale
import unet
from calibrate_m3 import calibrate_file, meridional_profile
from frames import blank_frames
from recording import Job

STRUCTURE = os.environ.get("KYMO_TEST_DATA", "/home/irina/data/2026_09_22_Hessel/EDL_Lo/Passive/structure")
RECORDINGS = ("WT4_RigA_LoPassive2_0071", "WT4_RigA_LoPassive1_0070")
REFERENCE = os.path.join(HERE, "reference")
PEAKS = os.path.join(REFERENCE, "config", "expected_peaks.csv")
with open(os.path.join(REFERENCE, "config", "parameters.json")) as _f:
    PARAMS = json.load(_f)
W = PARAMS["width"]
FILES = ["calibration.json", f"meridian_w{W}_g1.npy", f"meridian_w{W}_g1.json",
         f"meridian_w{W}_g1_events.csv", f"meridian_w{W}_g1_traces.csv"] + [
    f"{d}_w{W}/{name}" for d in ("multiscale", "unet")
    for name in ("tracks.csv", "labels.csv", f"{d}.npz", "run.json")]
# paths, and hashes that depend on them or on file times
IGNORED_KEYS = {"kymo", "image", "source", "frames_hash", "kymo_sha256"}
try:
    import torch
    HAVE_GPU = torch.cuda.is_available()
except ImportError:
    HAVE_GPU = False
# the U-Net runs in half precision on the GPU, so another GPU may differ slightly
UNET_TOL = dict(rtol=2e-2, atol=2e-2)
TOL = dict(rtol=1e-5, atol=1e-8)


APP = []  # the one QApplication, kept alive until every widget is gone


def qt_app():
    from PySide6.QtWidgets import QApplication
    if not APP:
        APP.append(QApplication.instance() or QApplication([]))


def delete_widget(widget):
    """Delete a widget now: left to the interpreter's exit, the order can crash Qt."""
    import shiboken6
    APP[0].processEvents()  # pending redraws first
    if shiboken6.isValid(widget):
        shiboken6.delete(widget)


def have_data(rec):
    return os.path.isdir(os.path.join(STRUCTURE, rec, "qf_results"))


def run_pipeline(rec, out):
    """What the GUI does for one recording, both detectors; returns {step: cached}."""
    folder = os.path.join(STRUCTURE, rec)
    qf = os.path.join(folder, "qf_results")
    cal = calibrate_file(kymograph.find_frames(qf)[0], W, PARAMS["m3_rmin"],
                         os.path.join(out, "calibration.json"), None, PARAMS["d_m3"])
    cached = {}
    npy, cached["kymograph"] = kymograph.make_kymograph(qf, out, W, cal["inv_nm_per_pixel"])
    ddf = sorted(glob.glob(os.path.join(folder, "*.ddf")))[0]
    summary = kymo_force.kymo_force(npy, ddf, (PARAMS["pos_lo"], PARAMS["pos_hi"]), PARAMS["stretch"])
    cached["force"] = summary.endswith("(cached)")
    for detector in (multiscale, unet):
        name = detector.__name__
        _, cached[name] = detector.process(npy, os.path.join(out, f"{name}_w{W}"), PARAMS, PEAKS)
    return cached


class TestRecordings(unittest.TestCase):
    """Every output equals the reference (numbers within a tolerance, text exactly)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.tmp)
        cls.out = {}
        for rec in RECORDINGS:
            if have_data(rec):
                cls.out[rec] = os.path.join(cls.tmp, rec)
                os.makedirs(cls.out[rec])
                run_pipeline(rec, cls.out[rec])

    def each_file(self, check, *endings):
        """Run check(reference path, new path, tolerance) on every file with one of the endings."""
        for rec in RECORDINGS:
            for name in FILES:
                if name.endswith(endings):
                    with self.subTest(recording=rec, file=name):
                        if rec not in self.out:
                            self.skipTest(f"no data for {rec} in {STRUCTURE}")
                        if name.startswith("unet") and not HAVE_GPU:
                            self.skipTest("the U-Net reference was made on a GPU; a CPU finds slightly other peaks")
                        check(os.path.join(REFERENCE, rec, name), os.path.join(self.out[rec], name),
                              UNET_TOL if name.startswith("unet") else TOL)

    def check_numbers(self, ref, new, tol, what=""):
        self.assertEqual(ref.shape, new.shape, what)
        np.testing.assert_allclose(new, ref, equal_nan=True, err_msg=what, **tol)

    def test_arrays(self):
        def check(ref, new, tol):
            if ref.endswith(".npy"):
                np.testing.assert_array_equal(np.load(new), np.load(ref))
                return
            with np.load(ref) as a, np.load(new) as b:
                self.assertEqual(set(a), set(b))
                for key in a:
                    self.check_numbers(a[key], b[key], tol, key)
        self.each_file(check, ".npy", ".npz")

    def test_csv(self):
        def check(ref, new, tol):
            with open(ref) as f, open(new) as g:
                a, b = list(csv.reader(f)), list(csv.reader(g))
            self.assertEqual(len(a), len(b), "number of rows")
            for i, (ra, rb) in enumerate(zip(a, b), 1):
                self.assertEqual(len(ra), len(rb), f"row {i}")
                for x, y in zip(ra, rb):
                    if x != y:
                        try:
                            xf, yf = float(x), float(y)
                        except ValueError:
                            self.fail(f"row {i}: {x!r} became {y!r}")
                        # values near 0 (dd_nm) have few digits: compare to their last digit as well
                        np.testing.assert_allclose(yf, xf, rtol=tol["rtol"], atol=max(tol["atol"], 1e-4),
                                                   err_msg=f"row {i}")
        self.each_file(check, ".csv")

    def test_json(self):
        def check(ref, new, _tol):
            with open(ref) as f, open(new) as g:
                a, b = json.load(f), json.load(g)
            for key in set(a) - IGNORED_KEYS:
                self.assertIn(key, b)
                if isinstance(a[key], list) and a[key] and isinstance(a[key][0], float):
                    self.check_numbers(np.array(a[key]), np.array(b[key]), TOL, key)
                else:
                    self.assertEqual(a[key], b[key], key)
        self.each_file(check, ".json")

    def test_second_run_is_cached(self):
        for rec, out in self.out.items():
            with self.subTest(recording=rec):
                paths = [os.path.join(out, n) for n in FILES if n.endswith((".npy", ".npz", "events.csv"))]
                before = [os.stat(p).st_mtime_ns for p in paths]
                self.assertEqual(set(run_pipeline(rec, out).values()), {True})
                self.assertEqual(before, [os.stat(p).st_mtime_ns for p in paths])
        if not self.out:
            self.skipTest(f"no data in {STRUCTURE}")

    def test_viewer_grouping(self):
        """Both detectors offer groupings 1, 2, 4, 8, 16, and each one is shown."""
        if len(self.out) < len(RECORDINGS):
            self.skipTest(f"no data in {STRUCTURE}")
        from recording import GROUPS
        from viewer import Viewer
        qt_app()
        for detector in ("multiscale", "unet"):
            view = Viewer()
            self.addCleanup(delete_widget, view)
            view.load([(rec, os.path.join(out, f"meridian_w{W}_g1.npy"), "", os.path.join(out, f"{detector}_w{W}"))
                       for rec, out in self.out.items()])
            offered = [int(view.scale_box.itemText(i)) for i in range(view.scale_box.count())]
            self.assertEqual(GROUPS, (1, 2, 4, 8, 16))
            self.assertLessEqual(set(GROUPS), set(offered))
            for s in offered:
                with self.subTest(detector=detector, grouping=s):
                    view.scale_box.setCurrentText(str(s))
                    for r in view.recs:
                        self.assertEqual(r.s, s)
                        np.testing.assert_allclose(r.view.grouped, multiscale.bin_time(r.base, s))
                        self.assertEqual(r.scales[r.k], s if detector == "multiscale" else 1)
            view.close()


class TestKymographCache(unittest.TestCase):
    """Synthetic frames: what is rebuilt when the images or the calibration change."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.qf, self.out = os.path.join(self.tmp, "rec", "qf_results"), os.path.join(self.tmp, "out")
        os.makedirs(self.qf)
        for i in range(32):
            self.write(i, i + 1)

    def write(self, i, value):
        path = os.path.join(self.qf, f"f_{i:05d}.tif")
        Image.fromarray(np.full((60, 60), value, np.float32)).save(path)
        os.utime(path, ns=(10 ** 18 + int(value), 10 ** 18 + int(value)))  # a change always shows

    def make(self, **kw):
        npy, cached = kymograph.make_kymograph(self.qf, self.out, 20, **kw)
        return np.load(npy), cached

    def test_changed_image_is_read_again(self):
        self.assertFalse(self.make()[1])
        self.assertTrue(self.make()[1])
        self.write(3, 1000)
        kymo, cached = self.make()
        self.assertFalse(cached)
        self.assertEqual(kymo[0, 3], 20 * 1000)

    def test_new_calibration_keeps_the_array(self):
        self.make(inv_nm_per_pixel=1e-4)
        npy = os.path.join(self.out, "meridian_w20_g1.npy")
        before = os.stat(npy).st_mtime_ns
        self.assertTrue(self.make(inv_nm_per_pixel=2e-4)[1])
        self.assertEqual(before, os.stat(npy).st_mtime_ns)
        with open(os.path.join(self.out, "meridian_w20_g1.json")) as f:
            self.assertEqual(json.load(f)["inv_nm_per_pixel"], 2e-4)

    def test_damaged_array_is_rebuilt(self):
        good, _ = self.make()
        with open(os.path.join(self.out, "meridian_w20_g1.npy"), "wb") as f:
            f.write(b"broken")
        again, cached = self.make()
        self.assertFalse(cached)
        np.testing.assert_array_equal(good, again)


class TestUnetCache(unittest.TestCase):
    """The U-Net is rerun for the settings it uses, and only for those (the net itself is replaced)."""

    def test_settings(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        npy = os.path.join(tmp, "meridian_w20_g1.npy")
        np.save(npy, np.random.default_rng(0).poisson(100, (200, 30)).astype(float))
        with open(npy[:-4] + ".json", "w") as f:
            json.dump(dict(pos_start=0, inv_nm_per_pixel=1e-3), f)
        calls = []

        def heatmap(x):
            calls.append(1)
            return np.exp(-0.5 * ((np.arange(len(x)) - 100) / 2.0) ** 2)[:, None] * np.ones(x.shape[1])

        with unittest.mock.patch.object(unet, "heatmap", heatmap):
            def rerun(**changes):
                n = len(calls)
                unet.process(npy, os.path.join(tmp, "unet"), dict(PARAMS, **changes), PEAKS)
                return len(calls) > n
            self.assertTrue(rerun())
            for same in (dict(), dict(snr_coarse=9.0), dict(corridor=9.0), dict(noise_win=9), dict(base=3),
                         dict(dark_fraction=0.06), dict(tol_pct=3.0)):
                self.assertFalse(rerun(**same), same)
            for other in (dict(unet_threshold=0.5), dict(rmax=150), dict(smooth=3), dict(background="averaged")):
                self.assertTrue(rerun(**other), other)


class TestStripAndBlankFrames(unittest.TestCase):
    def test_calibration_and_kymograph_use_the_same_strip(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = np.random.default_rng(0).random((60, 60)).astype(np.float32)
            path = os.path.join(tmp, "f_1.tif")
            Image.fromarray(img).save(path)
            for width in (1, 20, 21):
                kymo, _ = kymograph.build_kymograph([path], width)
                radii, folded = meridional_profile(np.asarray(img, dtype=float), width)
                below = img[30 - radii, 30 - width // 2:30 - width // 2 + width].sum(axis=1)
                np.testing.assert_allclose(folded - kymo[radii, 0], below, rtol=1e-5)

    def test_beam_off_spans_follow_blank_frames(self):
        for dark, fraction in ((slice(8, 12), 0.05), (slice(0, 14), 0.05), (slice(8, 12), 0.5)):
            kymo = np.full((10, 20), 100.0)
            kymo[:, dark] = 0.1 if fraction == 0.05 else 30.0
            spans = kymo_force.dark_spans(kymo, 1.0, fraction)
            self.assertEqual(spans, [(dark.start, dark.stop)])
            blank = blank_frames(kymo.sum(axis=0), fraction, 0)
            self.assertEqual(np.flatnonzero(blank).tolist(), list(range(dark.start, dark.stop)))


class TestTrackingCache(unittest.TestCase):
    """Multiscale on a synthetic kymograph, and the command lines."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.npy = os.path.join(self.tmp, "meridian_w20_g1.npy")
        r = np.arange(300)[:, None]
        rng = np.random.default_rng(0)
        np.save(self.npy, rng.poisson(300.0 - r * 0.5 + 400 * np.exp(-0.5 * ((r - 150) / 2.0) ** 2) * np.ones(30))
                .astype(float))
        with open(self.npy[:-4] + ".json", "w") as f:
            json.dump(dict(pos_start=0, inv_nm_per_pixel=1e-3, source=self.tmp), f)

    def process(self, exclude=None, **changes):
        return multiscale.process(self.npy, os.path.join(self.tmp, "ms"), dict(PARAMS, **changes), PEAKS, exclude)[1]

    def test_settings(self):
        self.assertFalse(self.process())
        self.assertTrue(self.process())
        self.assertTrue(self.process(tol_pct=2.0))
        self.assertFalse(self.process(snr_fine=1.5))
        self.assertFalse(self.process(snr_fine=1.5, exclude=[3]))

    def test_runs_of_older_versions_are_not_reused(self):
        self.process()
        path = os.path.join(self.tmp, "ms", "run.json")
        with open(path) as f:
            run = json.load(f)
        del run["version"]  # as saved when there were corridor options
        with open(path, "w") as f:
            json.dump(dict(run, ref="parent", select="snr"), f)
        self.assertFalse(self.process())

    def test_command_lines(self):
        root = os.path.dirname(HERE)
        qf = os.path.join(self.tmp, "rec", "qf_results")
        os.makedirs(qf)
        tif = os.path.join(qf, "f_1.tif")
        r = np.abs(np.arange(401) - 200)[:, None]
        Image.fromarray((500.0 - r + 90 * np.exp(-0.5 * ((r - 150) / 2.0) ** 2) * np.ones(401))
                        .astype(np.float32)).save(tif)
        cal = os.path.join(self.tmp, "cal.json")
        for args in (["calibrate_m3.py", tif, "--rmin", "100", "-o", cal, "--debug"],
                     ["kymograph.py", qf, "--calibration", cal],
                     ["multiscale.py", self.npy, "--peaks", PEAKS, "--exclude", "0-1"]):
            run = subprocess.run([sys.executable, os.path.join(root, args[0])] + args[1:],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
        with open(os.path.join(qf, "kymo", "meridian_w20_g1.json")) as f:
            self.assertAlmostEqual(json.load(f)["inv_nm_per_pixel"], 1 / (150 * 14.34))
        with open(os.path.join(self.tmp, "multiscale_w20", "run.json")) as f:
            self.assertEqual(json.load(f)["dropped_frames"], [0, 1])

    def test_unet_and_force_command_lines(self):
        rec = RECORDINGS[0]
        if not have_data(rec):
            self.skipTest(f"no data in {STRUCTURE}")
        ddf = glob.glob(os.path.join(STRUCTURE, rec, "*.ddf"))[0]
        for args in (["kymo_force.py", self.npy, ddf, "--pos-range", "0", "200"],
                     ["unet.py", self.npy, "--peaks", PEAKS]):
            run = subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), args[0])] + args[1:],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(os.path.exists(self.npy[:-4] + "_force.png"))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "unet_w20", "tracks.csv")))


class TestSmoothing(unittest.TestCase):
    def test_series(self):
        from plots import smooth_series
        h, frames = np.arange(12.0), np.r_[0:6, 10:16]  # frames 6-9 excluded
        h[2] = np.nan
        np.testing.assert_array_equal(smooth_series(h, frames, 1), h)
        odd, even = smooth_series(h, frames, 3), smooth_series(h, frames, 4)
        self.assertTrue(np.isnan(odd[2]) and np.isnan(even[2]))
        self.assertEqual((odd[4], odd[5], odd[6]), (4.0, 4.5, 6.5))  # not across the gap
        self.assertEqual((even[4], even[9]), (4.0, 8.5))  # frames i-2 ... i+1, without the NaN

    def test_viewer_defaults(self):
        from viewer import Viewer
        qt_app()
        view = Viewer()
        self.addCleanup(delete_widget, view)
        for row in (view.height_smooth, view.background_smooth):
            self.assertEqual((row.window, row.note), (10, " (mean of 10 frames)"))


class TestBackgroundSection(unittest.TestCase):
    """Its range and x controls are its own, and set the area; tracks and heights have theirs."""

    def test_independent(self):
        from recording import Recording
        from viewer import Viewer
        qt_app()
        view = Viewer()
        self.addCleanup(delete_widget, view)
        pos, n = np.arange(40, 400), 30
        base = np.random.default_rng(0).poisson(500 * np.exp(-pos / 100.0)[:, None] + 50, (len(pos), n)) * 1.0
        view.recs = [Recording(
            label="a", color="k", traces=[], dt=0.01, inv=1e-3, detector="unet", n_source=n, pos=pos, base=base,
            params=dict(multiscale.DEFAULTS, rmax=399), frames=np.arange(n), scales=np.array([1]),
            r=np.full((1, 1, n), 150.0), height=np.full((1, 1, n), 5.0), labels={0: "M5"}, unmatched={})]
        view.refresh(keep_grouping=False)
        rec, d_pr = view.recs[0], view.projections[0][0]

        def check_area(q_lo, q_hi):
            band = (rec.q >= q_lo) & (rec.q <= q_hi)
            expected = np.trapezoid(rec.view.backgrounds[band], rec.q[band], axis=0)
            line = view.background_canvas.figure.axes[0].lines[0]
            np.testing.assert_allclose(line.get_ydata(), expected)

        view.background_smooth.box.setValue(1)
        self.assertEqual(view.bg_range.range, (3.0, 80.0, "d (nm)"))
        check_area(1 / 80, 1 / 3)
        self.assertIs(view.projections[0][0], d_pr)  # the other panels were not redrawn
        bg_pr = view.bg_projections[0]

        view.range.set_range("20", "5")  # the other panels' range and scale: the section stays
        view.range.set_scale("linear")
        view.range.set_xlim((1.0, 9.0))
        self.assertIs(view.bg_projections[0], bg_pr)
        self.assertEqual(bg_pr.ax.get_xscale(), "log")
        check_area(1 / 80, 1 / 3)

        d_pr = view.projections[0][0]
        view.bg_range.set_range("10", "4")  # the section's own range: view and area follow, the others stay
        view.bg_range.set_xlim((2.0, 700.0))
        check_area(1 / 10, 1 / 4)
        bg_pr = view.bg_projections[0]
        np.testing.assert_allclose(bg_pr.ax.get_ylim(), (0.1, 0.25))
        self.assertEqual(bg_pr.ax.get_xlim(), (2.0, 700.0))
        self.assertIn("d 4–10 nm", view.background_canvas.figure.axes[0].get_ylabel())
        self.assertIs(view.projections[0][0], d_pr)
        self.assertEqual(d_pr.ax.get_xlim(), (1.0, 9.0))

        view.bg_range.units[0].setCurrentText("px")  # a unit counts only once applied
        view.scale_box.setCurrentText("4")
        check_area(1 / 10, 1 / 4)
        view.bg_range.set_range("100", "300")
        check_area(0.1, 0.3)
        view.bg_range.clear_range()
        check_area(0.0, 1.0)
        view.time.move(10)  # time is shared
        self.assertEqual(view.bg_projections[0].line.get_xdata().tolist(),
                         np.clip(rec.column(rec.column_at(view.t_cur)).profile, 2.0, None).tolist())


class TestPeaksField(unittest.TestCase):
    def test_prefixes(self):
        from labels import read_expected_peaks
        from viewer import Viewer
        qt_app()
        view = Viewer()
        self.addCleanup(delete_widget, view)
        names = [name for name, _, _ in read_expected_peaks(PEAKS)] + ["X1", "X12"]

        def shown(text):
            view.peaks_edit.setText(text)
            wanted = view.wanted_names()
            return [n for n in names if wanted is None or wanted.match(n)]

        self.assertEqual(shown(""), names)
        self.assertEqual(shown("M1"), ["M1L-45.5", "M1"])
        self.assertEqual(shown("X1"), ["X1"])
        self.assertEqual(shown("X"), ["X1", "X12"])
        self.assertEqual(shown("Tn"), ["Tn1", "Tn2", "Tn3", "Tn4"])
        self.assertEqual(shown("M"), [n for n in names if n[0] == "M"])
        self.assertEqual(shown("Z-15."), ["Z-15.48", "Z-15.4", "Z-15.15"])
        self.assertEqual(shown(" M6, , Star"), ["Star", "M6L-45.5", "M6"])
        self.assertEqual(shown("m3"), [])


class TestRecording(unittest.TestCase):
    def test_scale_for(self):
        from recording import Recording
        n = 40
        rec = Recording(label="", color="", traces=[], dt=0.005, inv=1e-3, detector="multiscale",
                        params=multiscale.DEFAULTS, n_source=50, frames=np.arange(5, 45), pos=np.arange(20),
                        base=np.ones((20, n)), scales=np.array([40, 27, 9, 3, 1]), r=np.empty((5, 0, n)),
                        height=np.empty((5, 0, n)), labels={}, unmatched={})
        self.assertEqual(rec.scale_for(1), (1, 4))
        self.assertEqual(rec.scale_for(8), (8, 2))    # a group on offer: kept, tracks of scale 9
        self.assertEqual(rec.scale_for(27), (27, 1))
        self.assertEqual(rec.scale_for(59), (40, 0))  # a longer recording's whole record
        self.assertEqual(rec.column_at(0.026), 0)
        self.assertIsNone(rec.column_at(0.01))


class TestOtherDialogs(unittest.TestCase):
    """Opening and saving against a temporary config folder changes nothing."""

    def setUp(self):
        import dialogs
        import params
        qt_app()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        for name in ("expected_peaks.csv", "parameters.json"):
            shutil.copy(os.path.join(REFERENCE, "config", name), self.tmp)
        for module, name, value in ((params, "PARAMS_JSON", "parameters.json"), (params, "CONFIG", ""),
                                    (dialogs, "PEAKS_CSV", "expected_peaks.csv")):
            patch = unittest.mock.patch.object(module, name, os.path.join(self.tmp, value))
            patch.start()
            self.addCleanup(patch.stop)

    def read(self, name):
        with open(os.path.join(self.tmp, name)) as f:
            return f.read()

    def test_save_unchanged(self):
        from dialogs import BackgroundDialog, ExpectedPeaksDialog, ParametersDialog
        before = self.read("expected_peaks.csv"), json.loads(self.read("parameters.json"))
        for kind in (BackgroundDialog, ExpectedPeaksDialog, ParametersDialog):
            d = kind(None)
            self.addCleanup(delete_widget, d)
            d.save()
        from params import DEFAULTS
        self.assertEqual(before[0], self.read("expected_peaks.csv"))
        self.assertEqual({k: v for k, v in before[1].items() if k in DEFAULTS},
                         json.loads(self.read("parameters.json")))

    def test_background_choice(self):
        from dialogs import BackgroundDialog
        d = BackgroundDialog(None)
        self.addCleanup(delete_widget, d)
        d.methods["fit"].setChecked(True)
        d.choices["averaged"].setChecked(True)
        d.comp2.setCurrentIndex(2)
        d.save()
        p = json.loads(self.read("parameters.json"))
        self.assertEqual((p["bg_method"], p["background"], p["bg_comp2"]), ("fit", "averaged", "stretched"))

    def test_excluded_frames(self):
        from dialogs import ParametersDialog
        from frames import load_excluded
        qf, out = os.path.join(self.tmp, "rec", "qf_results"), os.path.join(self.tmp, "out")
        os.makedirs(qf)
        for i in range(10):
            Image.fromarray(np.full((60, 60), 0.0 if i == 4 else 9.0, np.float32)).save(os.path.join(qf, f"f_{i}.tif"))
        d = ParametersDialog(None, [Job(qf, "", out)])
        self.addCleanup(delete_widget, d)
        r = d.recs[0]
        self.assertEqual((r["edit"].text(), r["auto"]), ("3-5", True))
        r["edit"].setText("1, 7-8")
        r["edit"].textEdited.emit("1, 7-8")
        d.save()
        self.assertEqual(load_excluded(out), [1, 7, 8])
        d.set_auto(r, True)
        d.save()
        self.assertIsNone(load_excluded(out))


class TestScaleRatio(unittest.TestCase):
    def test_ladder(self):
        self.assertEqual(multiscale.scale_ladder(59, 2), [59, 32, 16, 8, 4, 2, 1])
        self.assertEqual(multiscale.scale_ladder(59, 3), [59, 27, 9, 3, 1])
        for base in (1, 0, -1):
            with self.assertRaisesRegex(ValueError, "at least 2"):
                multiscale.scale_ladder(59, base)

    def test_dialog_minimum(self):
        from dialogs import ParametersDialog
        qt_app()
        d = ParametersDialog(None)
        self.addCleanup(delete_widget, d)
        d.widgets["base"].setValue(1)
        self.assertEqual(d.widgets["base"].value(), 2)
        self.assertEqual(d.widgets["drop_pad"].minimum(), 0)


class TestCalibrationDialog(unittest.TestCase):
    """What Save writes after each way of setting M3 (auto M3: 60 px in frame 1, 70 px in frame 2)."""

    def setUp(self):
        qt_app()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        qf = os.path.join(self.tmp, "rec", "qf_results")
        os.makedirs(qf)
        r = np.abs(np.arange(201) - 100)
        for i, peak in enumerate((60, 70), 1):
            profile = 200.0 - r + 50 * np.exp(-0.5 * ((r - peak) / 2.0) ** 2)
            Image.fromarray(np.repeat(profile[:, None], 201, axis=1).astype(np.float32)).save(
                os.path.join(qf, f"f_{i}.tif"))
        self.job = Job(qf, "", os.path.join(self.tmp, "out"))
        self.path = os.path.join(self.job.out, "calibration.json")

    def dialog(self, frame=1):
        from dialogs import CalibrationDialog
        d = CalibrationDialog(None, [self.job], 20, 30, 14.34)
        self.addCleanup(delete_widget, d)
        d.frame_box.setValue(frame)
        return d, d.recs[0]

    def saved(self, d):
        d.save()
        with open(self.path) as f:
            cal = json.load(f)
        return cal["n_m3_pixels"], cal["manual"], cal["set_on_frame"]

    def test_untouched_is_not_written(self):
        self.dialog(frame=2)[0].save()
        self.assertFalse(os.path.exists(self.path))

    def test_typed_then_auto_on_frame_1(self):
        d, r = self.dialog()
        r["box"].setValue(65.0)
        self.assertEqual(self.saved(d), (65.0, True, 1))
        d, r = self.dialog()  # opened again: shown as saved, and Auto undoes it
        self.assertEqual((r["box"].value(), r["manual"]), (65.0, True))
        d.set_auto(r)
        self.assertEqual(self.saved(d), (60.0, False, 1))

    def test_auto_on_another_frame_is_manual(self):
        d, r = self.dialog(frame=2)
        d.set_auto(r)
        d.frame_box.setValue(1)  # looking at another frame afterwards changes nothing
        self.assertEqual(self.saved(d), (70.0, True, 2))

    def test_typing_the_automatic_value_is_manual(self):
        d, r = self.dialog()
        r["box"].setValue(61.0)
        r["box"].setValue(60.0)
        self.assertEqual(self.saved(d), (60.0, True, 1))


def update():
    for rec in RECORDINGS:
        if not have_data(rec):
            sys.exit(f"no data for {rec} in {STRUCTURE}")
        with tempfile.TemporaryDirectory() as out:
            run_pipeline(rec, out)
            for name in FILES:
                dest = os.path.join(REFERENCE, rec, name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(os.path.join(out, name), dest)
        print("updated", os.path.join(REFERENCE, rec))


if __name__ == "__main__":
    if "--update" in sys.argv:
        update()
    else:
        unittest.main()
