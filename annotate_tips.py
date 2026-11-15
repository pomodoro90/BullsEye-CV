import argparse
import csv
import re
from pathlib import Path

import cv2

ZOOM_HALF = 16
ZOOM_FACTOR = 8


def load_tips(path):
    tips = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                lst = tips.setdefault(row["image"], [])
                x, y = float(row["x"]), float(row["y"])
                if x >= 0:
                    lst.append((x, y))
    return tips


def save_tips(path, tips):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "x", "y"])
        for name in sorted(tips):
            pts = tips[name]
            if not pts:
                wr.writerow([name, -1, -1])
            for x, y in pts:
                wr.writerow([name, f"{x:.1f}", f"{y:.1f}"])


def render(img, pts, mouse, info):
    vis = img.copy()
    for i, (x, y) in enumerate(pts):
        cv2.drawMarker(vis, (int(round(x)), int(round(y))), (0, 255, 255), cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
        cv2.circle(vis, (int(round(x)), int(round(y))), 9, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, str(i + 1), (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if mouse is not None:
        mx, my = mouse
        h, w = img.shape[:2]
        x0, y0 = int(mx) - ZOOM_HALF, int(my) - ZOOM_HALF
        pad = cv2.copyMakeBorder(img, ZOOM_HALF * 2, ZOOM_HALF * 2, ZOOM_HALF * 2, ZOOM_HALF * 2,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
        crop = pad[y0 + ZOOM_HALF * 2: y0 + ZOOM_HALF * 2 + ZOOM_HALF * 2 + 1,
                   x0 + ZOOM_HALF * 2: x0 + ZOOM_HALF * 2 + ZOOM_HALF * 2 + 1]
        zoom = cv2.resize(crop, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR, interpolation=cv2.INTER_CUBIC)
        c = zoom.shape[0] // 2
        cv2.line(zoom, (c - 25, c), (c - 6, c), (0, 0, 255), 1)
        cv2.line(zoom, (c + 6, c), (c + 25, c), (0, 0, 255), 1)
        cv2.line(zoom, (c, c - 25), (c, c - 6), (0, 0, 255), 1)
        cv2.line(zoom, (c, c + 6), (c, c + 25), (0, 0, 255), 1)
        zh, zw = zoom.shape[:2]

        zx = w - zw - 8 if mx < w / 2 else 8
        zy = 8
        vis[zy:zy + zh, zx:zx + zw] = zoom
        cv2.rectangle(vis, (zx, zy), (zx + zw, zy + zh), (255, 255, 255), 1)
    cv2.rectangle(vis, (0, vis.shape[0] - 30), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
    cv2.putText(vis, info, (10, vis.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", default=None, help="a tips.csv helye (alapból a session mappában)")
    ap.add_argument("--redo", action="store_true", help="már bejelölt képeket is újra mutat")
    args = ap.parse_args()

    folder = Path(args.session) / "darts"
    files = sorted(folder.glob("darts_*_cam*.png"),
                   key=lambda p: tuple(int(v) for v in re.search(r"darts_(\d+)_cam(\d)", p.name).groups()))
    if not files:
        print(f"Nem találok képeket itt: {folder}")
        return
    csv_path = Path(args.out) if args.out else Path(args.session) / "tips.csv"
    tips = load_tips(csv_path)

    todo = [f for f in files if args.redo or f.name not in tips]
    print(f"{len(files)} kép van összesen, ebből {len(todo)} még bejelölésre vár.")
    if not todo:
        print("Minden kép kész.")
        return

    win = "nyilhegy bejeloles"
    state = {"mouse": None, "pts": []}

    def on_mouse(ev, x, y, flags, param):
        state["mouse"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["pts"].append((float(x), float(y)))
        elif ev == cv2.EVENT_RBUTTONDOWN and state["pts"]:
            state["pts"].pop()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    i = 0
    while 0 <= i < len(todo):
        f = todo[i]
        img = cv2.imread(str(f))
        if img is None:
            i += 1
            continue
        state["pts"] = list(tips.get(f.name, []))
        done_here = False
        while True:
            info = (f"{f.name}   [{i + 1}/{len(todo)}]   nyilhegyek: {len(state['pts'])}   "
                    f"Enter=kovetkezo  b=vissza  q=kilepes  jobb klikk=torles")
            cv2.imshow(win, render(img, state["pts"], state["mouse"], info))
            k = cv2.waitKey(15) & 0xFF
            if k in (13, 32):
                tips[f.name] = list(state["pts"])
                save_tips(csv_path, tips)
                i += 1
                break
            if k == ord("b"):
                tips[f.name] = list(state["pts"])
                save_tips(csv_path, tips)
                i = max(0, i - 1)
                break
            if k == ord("q"):
                if state["pts"]:
                    tips[f.name] = list(state["pts"])
                    save_tips(csv_path, tips)
                done_here = True
                break
        if done_here:
            break

    cv2.destroyAllWindows()
    n_img = len(tips)
    n_tip = sum(len(v) for v in tips.values())
    print(f"Mentve: {csv_path.resolve()}  ({n_img} kép, {n_tip} nyílhegy)")


if __name__ == "__main__":
    main()
