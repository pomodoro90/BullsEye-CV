import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

MODES = {1: "chess", 2: "empty", 3: "darts"}
MODE_TEXT = {1: "SAKKTABLA", 2: "URES TABLA", 3: "NYILAK"}


def open_cam(index, width, height):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def grab_all(caps, discard=2):
    frames = None
    for _ in range(discard + 1):
        for c in caps:
            c.grab()
        frames = []
        for c in caps:
            ok, f = c.retrieve()
            frames.append(f if ok else None)
    return frames


def find_chess(gray, pattern, fast=False):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags |= cv2.CALIB_CB_FAST_CHECK
    return cv2.findChessboardCorners(gray, pattern, flags)


def scan_counts(session):
    counts = {1: 0, 2: 0, 3: 0}
    for m, name in MODES.items():
        for p in (session / name).glob(f"{name}_*_cam*.png"):
            try:
                counts[m] = max(counts[m], int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return counts


def make_preview(frames, mode, pattern, counts, status, label_buf, tile_w=426):
    tiles = []
    for i, f in enumerate(frames):
        if f is None:
            t = np.zeros((240, tile_w, 3), np.uint8)
            cv2.putText(t, f"CAM {i}: NINCS KEP", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            h, w = f.shape[:2]
            t = cv2.resize(f, (tile_w, int(h * tile_w / w)))
            color = (0, 255, 0)
            if mode == 1:
                g = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
                found, corners = find_chess(g, pattern, fast=True)
                if found:
                    cv2.drawChessboardCorners(t, pattern, corners, found)
                else:
                    color = (0, 165, 255)
            cv2.putText(t, f"CAM {i}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        tiles.append(t)
    row = np.hstack(tiles)

    bar_h = 56
    cv2.rectangle(row, (0, row.shape[0] - bar_h), (row.shape[1], row.shape[0]), (0, 0, 0), -1)
    if label_buf is not None:
        line1 = f"CIMKE: {label_buf}_"
        line2 = "Enter = kesz   Backspace = torles   Esc = kihagyas   (pl. T20,D6,S3)"
        c1 = (0, 255, 255)
    else:
        line1 = f"MOD: {MODE_TEXT[mode]}  |  mentve: {counts[mode]}  |  {status}"
        line2 = "[1] sakktabla  [2] ures tabla  [3] nyilak  [s] mentes  [q] kilepes"
        c1 = (255, 255, 255)
    y = row.shape[0] - bar_h
    cv2.putText(row, line1, (10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c1, 1)
    cv2.putText(row, line2, (10, y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return row


def write_label(path, n, label):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["index", "label"])
        wr.writerow([f"{n:04d}", label])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--chess", default="9x6", help="belső sarkok száma, pl. 9x6")
    ap.add_argument("--out", default="captures")
    ap.add_argument("--session", default=None, help="meglévő session mappa neve (folytatáshoz)")
    args = ap.parse_args()

    cols, rows = (int(x) for x in args.chess.lower().split("x"))
    pattern = (cols, rows)

    caps = [open_cam(i, args.width, args.height) for i in args.cams]
    for idx, c in zip(args.cams, caps):
        if not c.isOpened():
            print(f"HIBA: a {idx}. kamera nem nyitható meg.")
            sys.exit(1)
        w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Kamera {idx}: {w}x{h}")

    name = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    session = Path(args.out) / name
    for m in MODES.values():
        (session / m).mkdir(parents=True, exist_ok=True)
    labels_path = session / "darts" / "labels.csv"
    counts = scan_counts(session)
    print(f"Mentés ide: {session.resolve()}")
    print(f"Meglévő mentések: {counts}")

    mode = 1
    status = "kesz"
    label_buf = None
    pending_n = None

    try:
        while True:
            frames = grab_all(caps, discard=0)
            cv2.imshow("capture", make_preview(frames, mode, pattern, counts, status, label_buf))
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            if label_buf is not None:
                if key == 13:
                    write_label(labels_path, pending_n, label_buf)
                    status = f"#{pending_n:04d} cimke: {label_buf}"
                    print(f"[darts] #{pending_n:04d} címke: {label_buf}")
                    label_buf, pending_n = None, None
                elif key == 27:
                    write_label(labels_path, pending_n, "")
                    status = f"#{pending_n:04d} cimke nelkul"
                    label_buf, pending_n = None, None
                elif key == 8:
                    label_buf = label_buf[:-1]
                elif 32 <= key < 127:
                    label_buf += chr(key).upper()
                continue

            if key == ord("q"):
                break
            if key in (ord("1"), ord("2"), ord("3")):
                mode = int(chr(key))
                status = "kesz"
            elif key == ord("s"):
                frames = grab_all(caps, discard=2)
                folder = session / MODES[mode]
                n = counts[mode] + 1
                saved = []
                for i, f in enumerate(frames):
                    if f is None:
                        continue
                    if mode == 1:
                        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                        found, _ = find_chess(g, pattern)
                        if not found:
                            continue
                    fname = f"{MODES[mode]}_{n:04d}_cam{i}.png"
                    cv2.imwrite(str(folder / fname), f)
                    saved.append(i)

                if not saved:
                    status = "NEM mentett (sakktabla nem latszik)"
                    print("Nem mentettem semmit (a sakktábla egyik kamerán sem látszik).")
                    continue
                counts[mode] = n
                print(f"[{MODES[mode]}] #{n:04d} mentve, kamerák: {saved}")
                status = f"#{n:04d} mentve, kamerak: {saved}"
                if mode == 3:
                    label_buf, pending_n = "", n
    finally:
        for c in caps:
            c.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
