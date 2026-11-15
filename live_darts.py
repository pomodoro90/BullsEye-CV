import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import dartboard as db
from tipcommon import associate, board_dets, load_calib
from tracker import DartTracker

SMALL = (320, 180)


def project_raw(calib, cam, pts_mm):
    v = calib[f"cam{cam}"]
    pts = np.asarray(pts_mm, float).reshape(-1, 2)
    obj = np.column_stack([pts, np.zeros(len(pts))])
    P, _ = cv2.projectPoints(obj, np.array(v["rvec"]), np.array(v["tvec"]),
                             np.array(v["camera_matrix"]), np.array(v["dist_coeffs"]))
    return P.reshape(-1, 2)


def make_roi(calib, cam, height=190.0, radius=215.0):
    v = calib[f"cam{cam}"]
    w, h = v["image_size"]
    ang = np.radians(np.arange(0, 360, 3))
    pts3 = []
    for z in (0.0, -height):
        pts3 += [(radius * np.sin(a), -radius * np.cos(a), z) for a in ang]
    P, _ = cv2.projectPoints(np.array(pts3, np.float64), np.array(v["rvec"]), np.array(v["tvec"]),
                             np.array(v["camera_matrix"]), np.array(v["dist_coeffs"]))
    hull = cv2.convexHull(P.reshape(-1, 2).astype(np.float32)).astype(np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    return cv2.resize(m, SMALL, interpolation=cv2.INTER_NEAREST) > 0


def draw(frames, calib, tracker, status, tile_w=480):
    tiles = []
    for cam in sorted(frames):
        f = frames[cam].copy()
        h, w = f.shape[:2]
        k = max(1, int(round(w / tile_w)))
        th = np.radians(np.arange(0, 361, 4))
        for r, col in ((db.R_BULL_OUT, (255, 255, 0)), (db.R_TRIPLE_OUT, (255, 255, 0)), (db.R_DOUBLE_OUT, (255, 0, 255))):
            ring = project_raw(calib, cam, np.column_stack([r * np.sin(th), -r * np.cos(th)]))
            cv2.polylines(f, [ring.astype(np.int32).reshape(-1, 1, 2)], False, col, k, cv2.LINE_AA)
        for d in tracker.snapshot():
            p = project_raw(calib, cam, [(d["x"], d["y"])])[0]
            cv2.circle(f, (int(p[0]), int(p[1])), 7 * k, (0, 255, 0), max(2, k), cv2.LINE_AA)
            cv2.putText(f, d["label"], (int(p[0]) + 9 * k, int(p[1]) - 9 * k), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8 * k / 1.5, (0, 255, 0), max(2, k), cv2.LINE_AA)
        cv2.putText(f, f"CAM {cam}", (12, 34 * k // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * k / 1.5, (255, 255, 255),
                    max(2, k), cv2.LINE_AA)
        tiles.append(cv2.resize(f, (tile_w, int(h * tile_w / w)), interpolation=cv2.INTER_AREA))
    row = np.hstack(tiles)
    on_board = ", ".join(d["label"] for d in tracker.snapshot()) or "-"
    total = sum(d["points"] for d in tracker.snapshot())
    bar = np.zeros((54, row.shape[1], 3), np.uint8)
    cv2.putText(bar, f"Allapot: {status}", (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, f"Tablan: {on_board}   (osszesen {total})    [q] kilepes  [r] nullazas",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([row, bar])


class MotionGate:
    def __init__(self, rois, min_px=25, pix_thr=22):
        self.rois, self.min_px, self.pix_thr = rois, min_px, pix_thr
        self.prev = {}

    def update(self, frames):
        total = 0
        for cam, f in frames.items():
            g = cv2.cvtColor(cv2.resize(f, SMALL, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32)
            p = self.prev.get(cam)
            if p is not None:
                d = g - p
                d -= np.median(d)
                total += int(((np.abs(d) > self.pix_thr) & self.rois[cam]).sum())
            self.prev[cam] = g
        return total >= self.min_px, total


class CameraSource:
    def __init__(self, cams, width=1280, height=720, devices=None):
        self.cams = list(cams)
        self.devices = list(devices) if devices else list(cams)
        if len(self.devices) != len(self.cams):
            raise ValueError("A --devices számának meg kell egyeznie a --cams számával.")
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.caps = []
        for dev in self.devices:
            c = cv2.VideoCapture(dev, backend)
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not c.isOpened():
                raise RuntimeError(f"A {dev}. kamera nem nyitható meg.")
            self.caps.append(c)

    def read(self):
        for c in self.caps:
            c.grab()
        out = {}
        for i, c in zip(self.cams, self.caps):
            ok, f = c.retrieve()
            if ok:
                out[i] = f
        return out if len(out) == len(self.cams) else None

    def release(self):
        for c in self.caps:
            c.release()


class YoloTipDetector:
    def __init__(self, weights, imgsz=960, conf=0.25, device="auto", iou=0.35, warmup=True):
        from ultralytics import YOLO
        self.model = YOLO(weights, task="detect")
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.device = self._pick(device)
        print(f"Detektor eszköze: {self.device}", flush=True)
        if warmup:
            t0 = time.monotonic()
            self(np.zeros((720, 1280, 3), np.uint8))
            print(f"Bemelegítés kész ({time.monotonic() - t0:.1f} s)", flush=True)

    @staticmethod
    def _pick(device):
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                x = torch.randn(1, 3, 32, 32, device="cuda")
                torch.nn.functional.conv2d(x, torch.randn(4, 3, 3, 3, device="cuda")).sum().item()
                return 0
        except Exception:
            pass
        return "cpu"

    def __call__(self, frame):
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device, verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, float(b.conf[0])))
        return out


def evaluate(frames, detect, calib, tracker, anchor=0.25, max_spread=15.0, weak_spread=8.0):
    dets = {cam: board_dets(calib, cam, detect(f)) for cam, f in frames.items()}
    found = associate(dets, max_spread=max_spread, anchor_conf=anchor, weak_spread=weak_spread, nms_mm=6.0)
    evaluate.last = {"raw": {cam: len(v) for cam, v in dets.items()}, "found": len(found),
                     "dets": {cam: [(round(x), round(y), round(c, 2)) for x, y, c in v] for cam, v in dets.items()},
                     "fused": [(round(float(f[0][0])), round(float(f[0][1])), int(f[1]), round(float(f[2]), 1),
                                round(float(f[3]), 2)) for f in found]}
    return tracker.update(found), found


evaluate.last = {"raw": {}, "found": 0}


def run_loop(src, detect, calib, tracker, gate, settle=0.8, n_eval=2, eval_gap=0.15,
             on_event=print, show=True, max_seconds=None, periodic=0.0, sleep=0.0, on_frames=None,
             anchor=0.25, recheck=2.5, on_save=None, max_spread=15.0, weak_spread=8.0,
             should_stop=None, on_frame=None, on_status=None, on_eval=None):
    t0 = time.monotonic()
    last_motion, last_eval = t0 - settle, t0
    dirty, status = True, "indul"
    recheck_at, is_recheck = None, False
    n_frames, fps_t, fps = 0, t0, 0.0
    while True:
        now = time.monotonic()
        if max_seconds and now - t0 > max_seconds:
            break
        if should_stop and should_stop():
            break
        frames = src.read()
        if frames is None:
            break
        if on_frame:
            on_frame(frames)
        n_frames += 1
        if now - fps_t >= 2.0:
            fps, n_frames, fps_t = n_frames / (now - fps_t), 0, now
        moved, cnt = gate.update(frames)
        if moved:
            dirty, last_motion, status, recheck_at = True, now, f"mozgas ({cnt})", None
        elif recheck_at and now >= recheck_at:
            dirty, last_motion, recheck_at, is_recheck = True, now - settle, None, True
        elif periodic and now - last_eval >= periodic:
            dirty, last_motion = True, now - settle

        if dirty and now - last_motion >= settle:
            status = "kiertekeles..."
            t_eval = time.monotonic()
            for k in range(n_eval):
                if k > 0:
                    time.sleep(eval_gap)
                    frames = src.read() or frames
                events, found = evaluate(frames, detect, calib, tracker, anchor, max_spread, weak_spread)
                for e in events:
                    if e.get("type") == "throw":
                        e["latency_ms"] = round((time.monotonic() - last_motion) * 1000)
                    on_event(e)
                if events and on_frames:
                    on_frames(events, frames, found)
            ms = (time.monotonic() - t_eval) * 1000.0 / max(n_eval, 1)
            dirty, last_eval, status = False, time.monotonic(), f"figyel (egy kiertekeles: {ms:.0f} ms)"
            if on_eval:
                on_eval(ms, dict(evaluate.last))
            if recheck > 0 and not is_recheck:
                recheck_at = time.monotonic() + recheck
            is_recheck = False

        if on_status:
            on_status(f"{status} | {fps:.0f} kep/mp")
        if show:
            cv2.imshow("darts", draw(frames, calib, tracker, status))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r"):
                tracker.reset()
                on_event({"type": "reset"})
            if k == ord("s") and on_save:
                status = on_save(frames)
        if sleep:
            time.sleep(sleep)


def format_event(e):
    if e["type"] == "throw":
        s = (f"DOBAS: {e['label']} ({e['points']} pont)   hely: x={e['x']:.1f}, y={e['y']:.1f} mm   "
             f"kamerak: {e['cams']}   vezetektol: {e['margin_mm']:.1f} mm   bizonyossag: {e.get('conf', 1):.2f}")
        if e["uncertain"]:
            s += "   [BIZONYTALAN]"
        return s
    if e["type"] == "removed":
        return f"KIVETEL: {e['label']}   (a tablan maradt: {e['darts_on_board']})"
    return "NULLAZAS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--calib", default="calib_board/board_calib.json")
    ap.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--devices", type=int, nargs="+", default=None,
                    help="USB-eszközindexek a --cams sorrendjében (pl. --devices 2 1 0, ha a cam0 és cam2 fel van cserélve)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.05, help="a detektor alsó küszöbe (gyenge találatokat is megtart)")
    ap.add_argument("--anchor", type=float, default=0.25,
                    help="legalább egy kamerának ennyire biztosnak kell lennie; a gyengébbek csak csatlakozhatnak")
    ap.add_argument("--max-spread", type=float, default=15.0, help="a kamerák közti legnagyobb eltérés (mm)")
    ap.add_argument("--weak-spread", type=float, default=8.0, help="ugyanez, ha gyenge találat is van a csoportban")
    ap.add_argument("--fast-confirm", type=float, default=None, metavar="CONF",
                    help="gyors megerősítés: az új nyilat egy kiértékelés után elfogadja, ha mind a 3 kamera legalább ennyire "
                         "biztos benne (pl. 0.75); alapból kikapcsolva")
    ap.add_argument("--iou", type=float, default=0.35, help="egymást átfedő dobozok összevonása (kisebb = erősebb)")
    ap.add_argument("--recheck", type=float, default=2.5, help="második ránézés ennyi mp-cel a kiértékelés után (0 = ki)")
    ap.add_argument("--save-dir", default="captures/live_extra",
                    help="az 's' billentyűvel ide menti a pillanatnyi képeket (utólagos bejelöléshez)")
    ap.add_argument("--first-index", type=int, default=101)
    ap.add_argument("--device", default="auto", help="auto | cpu | 0")
    ap.add_argument("--settle", type=float, default=0.8, help="ennyi másodpercnyi nyugalom után értékel ki")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--log", default=None, help="események mentése (JSON soronként)")
    ap.add_argument("--debug-dir", default=None,
                    help="minden eseménynél elmenti a kamerák képét ide (hibakereséshez)")
    args = ap.parse_args()

    calib = load_calib(args.calib)
    detect = YoloTipDetector(args.weights, args.imgsz, args.conf, args.device, args.iou)
    src = CameraSource(args.cams, devices=args.devices)
    tracker = DartTracker(fast_conf=args.fast_confirm)
    gate = MotionGate({c: make_roi(calib, c) for c in args.cams})
    logf = open(args.log, "a", encoding="utf-8") if args.log else None

    def on_event(e):
        print(format_event(e), flush=True)
        if logf:
            logf.write(json.dumps({"t": time.time(), **e}, ensure_ascii=False) + "\n")
            logf.flush()

    on_frames = None
    if args.debug_dir:
        dbg = Path(args.debug_dir)
        dbg.mkdir(parents=True, exist_ok=True)
        print(f"Hibakereso kepek mentese ide: {dbg.resolve()}")

        def on_frames(events, frames, found):
            tag = "_".join(f"{e['type']}-{e.get('label', '')}" for e in events)
            stamp = time.strftime("%H%M%S")
            for cam, f in frames.items():
                cv2.imwrite(str(dbg / f"{stamp}_{tag}_cam{cam}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])

    save_folder = Path(args.save_dir) / "darts"
    counter = {"n": args.first_index - 1}
    if save_folder.exists():
        for p in save_folder.glob("darts_*_cam0.png"):
            counter["n"] = max(counter["n"], int(p.name.split("_")[1]))

    def on_save(frames):
        save_folder.mkdir(parents=True, exist_ok=True)
        counter["n"] += 1
        for cam, f in frames.items():
            cv2.imwrite(str(save_folder / f"darts_{counter['n']:04d}_cam{cam}.png"), f)
        print(f"Kepek mentve (bejeloleshez): darts_{counter['n']:04d}", flush=True)
        return f"mentve: #{counter['n']:04d}"

    print("Figyelem a tablat. Dobj! (kilepes: q az ablakban, vagy Ctrl+C)  [s] = kepek mentese bejeloleshez")
    try:
        run_loop(src, detect, calib, tracker, gate, settle=args.settle, on_event=on_event, show=args.show,
                 on_frames=on_frames, anchor=args.anchor, recheck=args.recheck, on_save=on_save,
                 max_spread=args.max_spread, weak_spread=args.weak_spread)
    except KeyboardInterrupt:
        pass
    finally:
        src.release()
        cv2.destroyAllWindows()
        if logf:
            logf.close()


if __name__ == "__main__":
    main()
