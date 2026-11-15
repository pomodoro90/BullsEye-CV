import threading

import numpy as np

import dartboard as db


def margin_mm(x, y):
    r = float(np.hypot(x, y))
    d_ring = min(abs(r - R) for R in db.RING_RADII)
    if r <= db.R_BULL_OUT:
        return d_ring
    ang = float(db.board_angle_deg(x, y))
    k = (ang + 9.0) % 18.0
    d_wire = np.radians(min(k, 18.0 - k)) * r
    return float(min(d_ring, d_wire))


def _greedy_match(a, b, max_d):
    pairs = sorted((float(np.linalg.norm(p - q)), i, j) for i, p in enumerate(a) for j, q in enumerate(b))
    ua, ub, out = set(), set(), []
    for d, i, j in pairs:
        if d <= max_d and i not in ua and j not in ub:
            ua.add(i); ub.add(j); out.append((i, j, d))
    return out


class DartTracker:
    def __init__(self, match_mm=6.0, confirm=2, uncertain_mm=1.5, near_mm=20.0, near_conf=0.5,
                 fast_conf=None, fast_spread=8.0):
        self.match_mm = match_mm
        self.near_mm = near_mm
        self.near_conf = near_conf
        self.confirm = confirm
        self.uncertain_mm = uncertain_mm

        self.fast_conf = fast_conf
        self.fast_spread = fast_spread
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        with self._lock:
            self.known = []
            self.gone = []
            self.pending = []

    def update(self, found):
        with self._lock:
            return self._update(found)

    def _update(self, found):
        events = []
        pos = [np.asarray(f[0], float) for f in found]
        pairs = _greedy_match([d["pos"] for d in self.known], pos, self.match_mm)
        matched_known = {i for i, _, _ in pairs}
        matched_found = {j for _, j, _ in pairs}

        for i in range(len(self.known)):
            self.gone[i] = 0 if i in matched_known else self.gone[i] + 1

        removed = [i for i in range(len(self.known)) if self.gone[i] >= self.confirm]
        for i in sorted(removed, reverse=True):
            d = self.known.pop(i)
            self.gone.pop(i)
            events.append({"type": "removed", "label": d["label"], "points": d["points"],
                           "x": float(d["pos"][0]), "y": float(d["pos"][1]),
                           "darts_on_board": len(self.known)})

        used, new_pending = set(), []
        for j in range(len(found)):
            if j in matched_found:
                continue

            conf_j = float(found[j][3]) if len(found[j]) > 3 else 1.0
            if self.known and conf_j < self.near_conf and \
                    min(np.linalg.norm(pos[j] - d["pos"]) for d in self.known) < self.near_mm:
                continue
            n = 1
            for k, c in enumerate(self.pending):
                if k not in used and np.linalg.norm(c["pos"] - pos[j]) <= self.match_mm:
                    n = c["n"] + 1
                    used.add(k)
                    break
            new_pending.append({"pos": pos[j], "n": n, "ncams": found[j][1], "spread": found[j][2],
                                "conf": float(found[j][3]) if len(found[j]) > 3 else 1.0})

        self.pending = [c for c in new_pending if not self._ready(c)]
        for c in new_pending:
            if self._ready(c):
                label, points = db.score_from_xy(*c["pos"])
                m = margin_mm(*c["pos"])
                self.known.append({"pos": c["pos"], "label": label, "points": points})
                self.gone.append(0)
                events.append({"type": "throw", "label": label, "points": points,
                               "x": float(c["pos"][0]), "y": float(c["pos"][1]),
                               "cams": int(c["ncams"]), "spread_mm": float(c["spread"]),
                               "margin_mm": m, "conf": c["conf"],
                               "uncertain": bool(m < self.uncertain_mm or c["ncams"] < 3 or c["conf"] < 0.4),
                               "darts_on_board": len(self.known)})
        return events

    def _ready(self, c):
        if c["n"] >= self.confirm:
            return True
        return (self.fast_conf is not None and c["ncams"] >= 3 and c["conf"] >= self.fast_conf
                and c["spread"] <= self.fast_spread)

    def snapshot(self):
        with self._lock:
            return self._snapshot()

    def _snapshot(self):
        return [{"label": d["label"], "points": d["points"], "x": float(d["pos"][0]), "y": float(d["pos"][1])}
                for d in self.known]
