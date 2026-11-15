import tempfile
import time
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

import dartboard as db
import live_darts
from engine import LiveEngine
from server import AppState

HERE = Path(__file__).resolve().parent
CALIB = HERE / "calib_board" / "board_calib.json"

WORLD = {"darts": [], "hand_until": 0.0}
_calib = None


class FakeSource:
    def __init__(self, cams, *a, **k):
        self.cams = list(cams)
        self.n = 0

    def read(self):
        time.sleep(0.03)
        self.n += 1
        out = {}
        for c in self.cams:
            f = np.full((720, 1280, 3), 40, np.uint8)
            f[0, 0] = (c, 0, 0)
            if time.monotonic() < WORLD["hand_until"]:
                p = live_darts.project_raw(_calib, c, [(0, 0)])[0]
                x, y = int(p[0]) + (60 if self.n % 2 else -60), int(p[1])
                cv2.rectangle(f, (x - 80, y - 80), (x + 80, y + 80), (255, 255, 255), -1)
            out[c] = f
        return out

    def release(self):
        pass


class FakeDetector:
    def __init__(self, *a, **k):
        pass

    def __call__(self, frame):
        cam = int(frame[0, 0, 0])
        pts = [db.label_to_xy(lab) for lab in WORLD["darts"]]
        pts = [p for p in pts if p is not None]
        if not pts:
            return []
        raw = live_darts.project_raw(_calib, cam, pts)
        return [(float(x), float(y), 0.9) for x, y in raw]


def wait_for(cond, timeout=12.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global _calib
        _calib = live_darts.load_calib(str(CALIB))
        cls._orig = (live_darts.CameraSource, live_darts.YoloTipDetector)
        live_darts.CameraSource, live_darts.YoloTipDetector = FakeSource, FakeDetector
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.S = S = AppState(cls.tmp.name, sim=False)
        args = types.SimpleNamespace(calib=str(CALIB), weights="nincs", cams=[0, 1, 2], imgsz=960, conf=0.05,
                                     device="cpu", iou=0.35, settle=0.3, anchor=0.25, recheck=0.0,
                                     max_spread=15.0, weak_spread=8.0)
        S.engine = LiveEngine(S.on_event, S.on_status, args)
        S.engine.start()
        S.new_game(["Anna", "Béla"], 501, True, False)
        assert wait_for(lambda: S.engine.status == "figyel", 10), S.engine.status

    @classmethod
    def tearDownClass(cls):
        cls.S.engine.stop()
        live_darts.CameraSource, live_darts.YoloTipDetector = cls._orig
        cls.tmp.cleanup()

    def hand(self, secs=0.4):
        WORLD["hand_until"] = time.monotonic() + secs

    def test_full_round(self):
        S = self.S
        g = lambda: S.state()["game"]

        WORLD["darts"] = ["T20"]
        self.hand()
        self.assertTrue(wait_for(lambda: len(g()["turn"]) == 1), "T20 nem érkezett meg")
        self.assertEqual(g()["turn"][0]["label"], "T20")
        self.assertEqual(g()["players"][0]["remaining"], 441)

        WORLD["darts"] = ["T20", "S5"]
        self.hand()
        self.assertTrue(wait_for(lambda: len(g()["turn"]) == 2), "S5 nem érkezett meg")
        WORLD["darts"] = ["T20", "S5", "D16"]
        self.hand()
        self.assertTrue(wait_for(lambda: g()["turn_ended"]), "a kör nem zárult le")
        s = S.state()
        self.assertEqual([d["label"] for d in g()["last_turn"]["darts"]], ["T20", "S5", "D16"])
        self.assertEqual(g()["players"][0]["remaining"], 501 - 60 - 5 - 32)
        self.assertEqual(g()["current"], 1)
        self.assertTrue(s["awaiting_clear"])
        self.assertEqual(len(s["board"]), 3)

        WORLD["darts"] = []
        self.hand()
        self.assertTrue(wait_for(lambda: S.state()["board"] == []), "a kivétel nem jött át")
        self.assertFalse(S.state()["awaiting_clear"])
        self.assertEqual(g()["current"], 1)

        WORLD["darts"] = ["D20"]
        self.hand()
        self.assertTrue(wait_for(lambda: len(g()["turn"]) == 1))
        self.assertEqual(g()["players"][1]["remaining"], 461)

        jpg = S.engine.view_jpeg()
        self.assertTrue(jpg and jpg[:2] == b"\xff\xd8")

        lines = (Path(self.tmp.name) / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 7)

    def test_zz_camera_failure_recovers(self):
        S = self.S
        orig = live_darts.CameraSource
        calls = {"n": 0}

        def flaky(cams, *a, **k):
            calls["n"] += 1
            raise RuntimeError("szimulált kamera hiba")

        live_darts.CameraSource = flaky
        try:

            S.engine._frames = None
            FakeSource.read, saved = (lambda self: None), FakeSource.read
            self.assertTrue(wait_for(lambda: S.engine.status.startswith("kamera hiba"), 8))
            FakeSource.read = saved
            live_darts.CameraSource = orig
            self.assertTrue(wait_for(lambda: S.engine.status == "figyel", 12), S.engine.status)
        finally:
            live_darts.CameraSource = orig


class FallbackTest(unittest.TestCase):
    def test_falls_back_to_secondary_model(self):
        global _calib
        _calib = live_darts.load_calib(str(CALIB))
        orig = (live_darts.CameraSource, live_darts.YoloTipDetector)
        loaded = []

        def factory(weights, imgsz, conf, device, iou, *a, **k):
            loaded.append((weights, device))
            if weights == "rossz":
                raise RuntimeError("nincs ilyen eszkoz")
            return FakeDetector()

        live_darts.CameraSource, live_darts.YoloTipDetector = FakeSource, factory
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        S = AppState(tmp.name, sim=False)
        args = types.SimpleNamespace(calib=str(CALIB), weights="rossz", device="intel:gpu", fallback_weights="jo",
                                     fallback_device="cpu", cams=[0, 1, 2], imgsz=960, conf=0.05, iou=0.35,
                                     settle=0.3, anchor=0.25, recheck=0.0, max_spread=15.0, weak_spread=8.0)
        S.engine = LiveEngine(S.on_event, S.on_status, args)
        try:
            S.engine.start()
            self.assertTrue(wait_for(lambda: S.engine.status == "figyel", 10), S.engine.status)
            self.assertEqual(loaded, [("rossz", "intel:gpu"), ("jo", "cpu")])
        finally:
            S.engine.stop()
            live_darts.CameraSource, live_darts.YoloTipDetector = orig
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
