import itertools
import sys

import cv2
import numpy as np

import dartboard as db

MIN_MARGIN = 1.25
MIN_EACH = 1.3
MIN_RATIO = 1.6


def _ring_points(calib, cam, n=240):
    import live_darts
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = []
    for r in db.RING_RADII[2:]:
        pts.append(live_darts.project_raw(calib, cam, np.column_stack([r * np.sin(th), -r * np.cos(th)])))
    return np.vstack(pts)


def pair_score(calib, cam, img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    mag = cv2.dilate(mag, np.ones((5, 5), np.uint8))
    h, w = mag.shape
    p = np.round(_ring_points(calib, cam)).astype(int)
    ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    if ok.sum() < 0.5 * len(p):
        return 0.0
    vals = mag[p[ok, 1], p[ok, 0]]
    return float(vals.mean() / (mag.mean() + 1e-6)) * float(ok.mean())


def identify(images, calib, cams):
    devs = sorted(images)
    S = np.array([[pair_score(calib, c, images[d]) for d in devs] for c in cams])
    best, second = (None, -1.0), -1.0
    for perm in itertools.permutations(range(len(devs)), len(cams)):
        s = float(sum(S[i, j] for i, j in enumerate(perm)))
        if best[0] is None or s > best[1]:
            best, second = (perm, s), best[1] if best[0] is not None else -1.0
        elif s > second:
            second = s
    perm, total = best
    devices = [devs[j] for j in perm]
    margin = total / second if second > 0 else float("inf")
    ratio = total / len(cams)
    weakest = min(S[i, j] for i, j in enumerate(perm))
    good = margin >= MIN_MARGIN and ratio >= MIN_RATIO and weakest >= MIN_EACH
    msg = "rendben" if good else (
        "bizonytalan azonosítás: a gyűrűk nem illenek elég jól egyik sorrendhez sem (elmozdult a tábla/kamera, vagy nagyon sötét a kép)")
    return devices, {"ok": bool(good), "scores": S.round(2).tolist(), "devices_order": devs,
                     "margin": round(margin, 2), "mean_score": round(ratio, 2), "message": msg}


def capture_all(indices, width=1280, height=720, warm=8):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    out = {}
    for d in indices:
        cap = cv2.VideoCapture(d, backend)
        try:
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            f = None
            for _ in range(warm):
                ok, fr = cap.read()
                if ok:
                    f = fr
            if f is not None:
                out[d] = f
        finally:
            cap.release()
    return out


def auto_devices(calib, cams, candidates=None, log=print):
    candidates = list(candidates) if candidates is not None else list(range(len(cams) + 2))
    imgs = capture_all(candidates)
    if len(imgs) < len(cams):
        log(f"[camid] csak {len(imgs)} kamera nyitható meg ({sorted(imgs)}), kell: {len(cams)}")
        return None
    devices, rep = identify(imgs, calib, cams)
    log(f"[camid] eszközök: {devices} | pontszámok: {rep['scores']} | előny: {rep['margin']}x | {rep['message']}")
    return devices, rep


if __name__ == "__main__":
    import argparse
    from tipcommon import load_calib
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="calib_board/board_calib.json")
    ap.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    res = auto_devices(load_calib(a.calib), a.cams)
    if res:
        print("\nAjánlott kapcsoló:  --devices " + " ".join(map(str, res[0])))
