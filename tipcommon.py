import csv
import itertools
import json
import re
from collections import defaultdict

import cv2
import numpy as np

import dartboard as db

BOX = 32
VAL_SETS = set(range(9, 13)) | set(range(21, 25))


def load_tips_csv(path):
    out = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = re.search(r"darts_(\d+)_cam(\d)", r["image"])
            x, y = float(r["x"]), float(r["y"])
            s, c = int(m.group(1)), int(m.group(2))
            if x >= 0:
                out[s][c].append((x, y))
            else:
                out[s][c]
    return out


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return {int(r["index"]): r["label"] for r in csv.DictReader(fh)}


def load_calib(path):
    return json.load(open(path, encoding="utf-8"))


def to_board(calib, cam, pts_px):
    if len(pts_px) == 0:
        return np.zeros((0, 2))
    v = calib[f"cam{cam}"]
    K = np.array(v["camera_matrix"])
    d = np.array(v["dist_coeffs"])
    u = cv2.undistortPoints(np.array(pts_px, np.float64).reshape(-1, 1, 2), K, d, P=K).reshape(-1, 2)
    return db.apply_h(np.array(v["H_img_to_board"]), u)


def board_dets(calib, cam, dets_px):
    if not dets_px:
        return []
    pts = to_board(calib, cam, [(d[0], d[1]) for d in dets_px])
    return [(float(p[0]), float(p[1]), float(d[2])) for p, d in zip(pts, dets_px)]


def associate(dets, max_spread=25.0, anchor_conf=0.0, weak_spread=None, nms_mm=0.0):
    cams = sorted(dets)
    opts = [list(range(len(dets[c]))) + [None] for c in cams]
    cands = []
    for combo in itertools.product(*opts):
        members = [dets[c][i] for c, i in zip(cams, combo) if i is not None]
        if len(members) < 2:
            continue
        confs = [m[2] if len(m) > 2 else 1.0 for m in members]
        if max(confs) < anchor_conf:
            continue
        pts = [np.asarray(m[:2], float) for m in members]
        med = np.median(pts, axis=0)
        spread = max(np.linalg.norm(p - med) for p in pts)
        limit = max_spread
        if weak_spread is not None and min(confs) < anchor_conf:
            limit = min(limit, weak_spread)
        if spread <= limit:
            cands.append((-len(pts), spread, combo, med, max(confs)))
    cands.sort(key=lambda t: (t[0], t[1]))
    used, out = set(), []
    for neg, spread, combo, med, mc in cands:
        keys = [(c, i) for c, i in zip(cams, combo) if i is not None]
        if any(k in used for k in keys):
            continue
        used.update(keys)
        if nms_mm and any(np.linalg.norm(med - o[0]) < nms_mm for o in out):
            continue
        out.append((med, -neg, spread, mc))
    return out


def human_ground_truth(calib, tips_for_set):
    dets = {c: [tuple(p) for p in to_board(calib, c, pts)] for c, pts in tips_for_set.items()}
    return associate(dets)
