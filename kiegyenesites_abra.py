import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def grab(dev, w=1280, h=720, warm=10):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(dev, backend)
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        f = None
        for _ in range(warm):
            ok, fr = cap.read()
            if ok:
                f = fr
        return f
    finally:
        cap.release()


def grid(img, step=80, color=(0, 255, 255)):
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), color, 1)
    return out


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (340, 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intrinsics", default="calib_simple/intrinsics.json")
    ap.add_argument("--devices", type=int, nargs=3, default=[0, 1, 2])
    ap.add_argument("--images", nargs=3, help="3 kepfajl (cam0 cam1 cam2) kamera helyett")
    ap.add_argument("--out", default="abra_kiegyenesites")
    a = ap.parse_args()

    calib = json.load(open(a.intrinsics, encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        name = f"cam{i}"
        img = cv2.imread(a.images[i]) if a.images else grab(a.devices[i])
        if img is None:
            print(f"{name}: nem sikerult kepet szerezni")
            continue
        K = np.array(calib[name]["camera_matrix"], dtype=np.float64)
        d = np.array(calib[name]["dist_coeffs"], dtype=np.float64)
        und = cv2.undistort(img, K, d)
        cv2.imwrite(str(out / f"{name}_elotte.png"), img)
        cv2.imwrite(str(out / f"{name}_utana.png"), und)
        both = np.hstack([label(grid(img), f"{name} - eredeti"), label(grid(und), f"{name} - kiegyenesitett")])
        cv2.imwrite(str(out / f"{name}_osszehasonlitas.png"), both)
        print(f"{name}: kesz ({img.shape[1]}x{img.shape[0]}, rms={calib[name].get('rms', '?')})")
    print("Kepek:", out.resolve())


if __name__ == "__main__":
    main()
