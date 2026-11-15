import itertools
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

import camid
import dartboard as db
import live_darts
from engine import LiveEngine
from tipcommon import load_calib

CALIB = Path(__file__).resolve().parent / "calib_board" / "board_calib.json"


def fake_board_image(calib, cam, seed=0):
    rng = np.random.default_rng(seed + cam)
    img = np.full((720, 1280, 3), 60, np.uint8)
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    th = np.linspace(0, 2 * np.pi, 360)
    for r in db.RING_RADII:
        p = live_darts.project_raw(calib, cam, np.column_stack([r * np.sin(th), -r * np.cos(th)]))
        cv2.polylines(img, [p.astype(np.int32).reshape(-1, 1, 2)], False, (230, 230, 230), 3, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (3, 3), 0)


class CamIdTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calib = load_calib(str(CALIB))
        cls.imgs = {c: fake_board_image(cls.calib, c) for c in range(3)}

    def test_all_permutations(self):
        for perm in itertools.permutations(range(3)):
            shuffled = {d: self.imgs[perm[d]] for d in range(3)}
            devices, rep = camid.identify(shuffled, self.calib, [0, 1, 2])
            self.assertEqual(devices, [perm.index(c) for c in range(3)], perm)
            self.assertTrue(rep["ok"], rep)

    def test_blank_images_not_ok(self):
        blank = {d: np.zeros((720, 1280, 3), np.uint8) for d in range(3)}
        _, rep = camid.identify(blank, self.calib, [0, 1, 2])
        self.assertFalse(rep["ok"])

    def test_one_dead_camera_not_ok(self):
        imgs = dict(self.imgs)
        imgs[2] = np.zeros((720, 1280, 3), np.uint8)
        _, rep = camid.identify(imgs, self.calib, [0, 1, 2])
        self.assertFalse(rep["ok"])

    def test_extra_device_ignored(self):
        imgs = {0: np.full((720, 1280, 3), 90, np.uint8), 1: self.imgs[1], 2: self.imgs[0], 3: self.imgs[2]}
        devices, rep = camid.identify(imgs, self.calib, [0, 1, 2])
        self.assertEqual(devices, [2, 1, 3])


class EnginePickTest(unittest.TestCase):
    def make(self, **kw):
        args = types.SimpleNamespace(auto_devices=True, devices=[2, 1, 0], cams=[0, 1, 2], device_candidates=None, **kw)
        e = LiveEngine(lambda e: None, lambda s: None, args)
        e._calib = None
        return e

    def run_with(self, result, **kw):
        orig = camid.auto_devices
        camid.auto_devices = lambda *a, **k: result
        try:
            e = self.make(**kw)
            return e._pick_devices(), e.warning
        finally:
            camid.auto_devices = orig

    def test_ok_uses_detected(self):
        dev, warn = self.run_with(([1, 0, 2], {"ok": True, "message": "rendben"}))
        self.assertEqual((dev, warn), ([1, 0, 2], ""))

    def test_uncertain_uses_manual_and_warns(self):
        dev, warn = self.run_with(([1, 0, 2], {"ok": False, "message": "bizonytalan"}))
        self.assertEqual(dev, [2, 1, 0])
        self.assertIn("Kamerák", warn)

    def test_failure_uses_manual_and_warns(self):
        dev, warn = self.run_with(None)
        self.assertEqual(dev, [2, 1, 0])
        self.assertTrue(warn)

    def test_disabled_returns_manual(self):
        e = self.make()
        e.args.auto_devices = False
        self.assertEqual(e._pick_devices(), [2, 1, 0])


if __name__ == "__main__":
    unittest.main()
