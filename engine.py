import threading
import time
import traceback

from tracker import DartTracker


class BaseEngine:
    mode = "base"

    def __init__(self, on_event, on_status, tracker=None):
        self.on_event = on_event
        self.on_status = on_status
        self.tracker = tracker or DartTracker()
        self._status = "indul"
        self.warning = ""

    @property
    def status(self):
        return self._status

    def _set_status(self, s):
        if s != self._status:
            self._status = s
            self.on_status(s)

    def _set_warning(self, w):
        if w != self.warning:
            self.warning = w
            self.on_status(self._status)

    def reset_board(self):
        self.tracker.reset()
        self.on_event({"type": "reset"})

    def view_jpeg(self):
        return None

    def start(self):
        pass

    def stop(self):
        pass


class SimEngine(BaseEngine):
    mode = "sim"

    def __init__(self, on_event, on_status):
        super().__init__(on_event, on_status)
        self._set_status("szimuláció")
        self._lock = threading.Lock()

    def _feed(self, extra=None):
        with self._lock:
            found = [((d["x"], d["y"]), 3, 0.5, 0.9) for d in self.tracker.snapshot()]
            if extra:
                found.append(extra)
            events = []
            for _ in range(self.tracker.confirm):
                events += self.tracker.update(found)
        return events

    def throw(self, label, uncertain=False):
        import dartboard as db
        xy = db.label_to_xy(label)
        if xy is None:
            raise ValueError("Mellé dobásnak nincs helye a táblán (használd a 'Mellé' gombot).")
        events = self._feed((xy, 3, 0.5, 0.2 if uncertain else 0.9))
        for e in events:
            if e["type"] == "throw" and uncertain:
                e["uncertain"] = True
            self.on_event(e)
        return events

    def clear(self):
        with self._lock:
            events = []
            for _ in range(self.tracker.confirm):
                events += self.tracker.update([])
        for e in events:
            self.on_event(e)
        return events


class LiveEngine(BaseEngine):
    mode = "live"

    def __init__(self, on_event, on_status, args):
        super().__init__(on_event, on_status,
                         DartTracker(fast_conf=getattr(args, "fast_confirm", None)))
        self.args = args
        self._stop = threading.Event()
        self._thread = None
        self._frames = None
        self._frames_t = 0.0
        self._calib = None
        self.detail = ""

    def start(self):
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _on_frame(self, frames):
        self._frames, self._frames_t = frames, time.monotonic()

    def _on_detail(self, text):
        self.detail = text

    def _on_eval(self, ms, st):
        d = "  ".join(f"c{c}:{v}" for c, v in sorted(st["dets"].items()))
        print(f"[kiertekeles] {ms:.0f} ms | {d} -> egyesitett: {st['fused']}", flush=True)

    def view_jpeg(self):
        import cv2
        import live_darts
        frames = self._frames
        if frames is None or time.monotonic() - self._frames_t > 5 or self._calib is None:
            return None
        try:
            img = live_darts.draw(frames, self._calib, self.tracker, self.detail or self._status, tile_w=420)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return buf.tobytes() if ok else None
        except Exception:
            return None

    def _pick_devices(self):
        a = self.args
        manual = getattr(a, "devices", None)
        if not getattr(a, "auto_devices", False):
            return manual
        try:
            import camid
            res = camid.auto_devices(self._calib, a.cams, candidates=getattr(a, "device_candidates", None))
        except Exception as ex:
            print(f"[camid] hiba: {ex}", flush=True)
            res = None
        if res is None:
            self._set_warning("A kamerák automatikus azonosítása nem sikerült, a megadott sorrendet használom.")
            return manual
        found, rep = res
        if rep["ok"]:
            self._set_warning("")
            return found

        self._set_warning("Kamerák: " + rep["message"] + ". Ellenőrizd a Kamerák fület!")
        return manual or found

    def _run(self):
        a = self.args
        try:
            import live_darts
            from tipcommon import load_calib
        except Exception as ex:
            self._set_status(f"hiba: {ex}")
            return

        detect = None
        candidates = [(a.weights, a.device)]
        if getattr(a, "fallback_weights", None):
            candidates.append((a.fallback_weights, getattr(a, "fallback_device", "cpu")))
        while not self._stop.is_set() and detect is None:
            try:
                self._calib = load_calib(a.calib)
            except Exception as ex:
                self._set_status(f"hiba a kalibráció betöltésénél: {ex}")
                print(f"[engine] kalibrációs hiba: {ex}", flush=True)
                traceback.print_exc()
                self._stop.wait(10)
                continue
            for i, (weights, device) in enumerate(candidates):
                try:
                    self._set_status("modell betöltése...")
                    detect = live_darts.YoloTipDetector(weights, a.imgsz, a.conf, device, a.iou)
                    if i > 0:
                        print(f"[engine] FIGYELEM: tartalék modell fut: {weights} ({device})", flush=True)
                    break
                except Exception as ex:
                    self._set_status(f"hiba a betöltésnél: {ex}")
                    print(f"[engine] betöltési hiba ({weights}, {device}): {ex}", flush=True)
                    traceback.print_exc()
            if detect is None:
                self._stop.wait(10)
        rois = {c: live_darts.make_roi(self._calib, c) for c in a.cams} if detect else {}

        while not self._stop.is_set():
            src = None
            try:
                devices = self._pick_devices()
                src = live_darts.CameraSource(a.cams, devices=devices)
                gate = live_darts.MotionGate(rois)
                self._set_status("figyel")
                live_darts.run_loop(src, detect, self._calib, self.tracker, gate, settle=a.settle,
                                    on_event=self.on_event, show=False, anchor=a.anchor,
                                    recheck=a.recheck, max_spread=a.max_spread, weak_spread=a.weak_spread,
                                    should_stop=self._stop.is_set, on_frame=self._on_frame,
                                    on_status=self._on_detail, on_eval=self._on_eval)
                if not self._stop.is_set():
                    raise RuntimeError("a kamera nem ad képet")
            except Exception as ex:
                self._set_status(f"kamera hiba, újrapróbálom ({ex})")
                print(f"[engine] kamera hiba: {ex}", flush=True)
                traceback.print_exc()
            finally:
                if src is not None:
                    try:
                        src.release()
                    except Exception:
                        pass
            self._frames = None
            self._stop.wait(4)
