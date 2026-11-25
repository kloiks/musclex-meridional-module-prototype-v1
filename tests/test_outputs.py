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
    npy, cached["kymograph"] = kymograph.make_kymograph(qf, out, W, 1, cal["inv_nm_per_pixel"])
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
        from viewer import GROUPS, Viewer
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
                        ms = r["ms"]
                        self.assertEqual(r["s"], s)
                        np.testing.assert_allclose(ms["grouped"][s], multiscale.bin_time(ms["base"], s))
                        self.assertEqual(ms["scales"][r["k"]], s if detector == "multiscale" else 1)
            view.close()


class TestKymographCache(unittest.TestCase):
    """Synthetic frames: what is rebuilt when the images, the calibration or the grouping change."""

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

    def test_grouped_files(self):
        one, _ = self.make()
        for g in (2, 4, 8, 16):
            grouped, _ = self.make(group=g)
            self.assertEqual(grouped.shape[1], 32 // g)
            np.testing.assert_array_equal(grouped[:, 0], one[:, :g].sum(axis=1))
        self.write(0, 500)
        self.assertEqual(self.make(group=2)[0][0, 0], 20 * (500 + 2))  # not the old grouped file


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
            spans = kymo_force.dark_spans(kymo, 0.0, 1.0, fraction)
            self.assertEqual(spans, [(dark.start, dark.stop)])
            blank = blank_frames(kymo.sum(axis=0), fraction, 0)
            self.assertEqual(np.flatnonzero(blank).tolist(), list(range(dark.start, dark.stop)))


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
        self.job = (qf, None, os.path.join(self.tmp, "out"))
        self.path = os.path.join(self.job[2], "calibration.json")

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
