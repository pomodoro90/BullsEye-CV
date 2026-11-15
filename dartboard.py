import numpy as np


SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

BLACK_SECTORS = {20, 18, 13, 10, 2, 3, 7, 8, 14, 12}

R_BULL_IN = 6.35
R_BULL_OUT = 15.9
R_TRIPLE_IN = 99.0
R_TRIPLE_OUT = 107.0
R_DOUBLE_IN = 162.0
R_DOUBLE_OUT = 170.0

RING_RADII = [R_BULL_IN, R_BULL_OUT, R_TRIPLE_IN, R_TRIPLE_OUT, R_DOUBLE_IN, R_DOUBLE_OUT]


def board_angle_deg(x, y):
    return np.degrees(np.arctan2(x, -y)) % 360.0


def sector_index(angle_deg):
    return np.floor(((np.asarray(angle_deg) + 9.0) % 360.0) / 18.0).astype(int)


def score_from_xy(x, y):
    r = float(np.hypot(x, y))
    if r <= R_BULL_IN:
        return "50", 50
    if r <= R_BULL_OUT:
        return "25", 25
    if r > R_DOUBLE_OUT:
        return "MISS", 0
    n = SECTORS[int(sector_index(board_angle_deg(x, y)))]
    if R_TRIPLE_IN <= r <= R_TRIPLE_OUT:
        return f"T{n}", 3 * n
    if R_DOUBLE_IN <= r <= R_DOUBLE_OUT:
        return f"D{n}", 2 * n
    return f"S{n}", n


def label_to_xy(label):
    s = str(label).strip().upper()
    if s == "MISS":
        return None
    if s == "50":
        return (0.0, 0.0)
    if s == "25":
        return (0.0, -(R_BULL_IN + R_BULL_OUT) / 2.0)
    kind, num = s[0], int(s[1:])
    idx = SECTORS.index(num)
    a = np.radians(idx * 18.0)
    r = {"S": (R_TRIPLE_OUT + R_DOUBLE_IN) / 2.0,
         "T": (R_TRIPLE_IN + R_TRIPLE_OUT) / 2.0,
         "D": (R_DOUBLE_IN + R_DOUBLE_OUT) / 2.0}[kind]
    return (float(r * np.sin(a)), float(-r * np.cos(a)))


def render_signed_template(scale=2.0, margin=12.0):
    ext = R_DOUBLE_OUT + margin
    n = int(round(2 * ext * scale))
    S = np.array([[scale, 0, ext * scale],
                  [0, scale, ext * scale],
                  [0, 0, 1.0]])
    v, u = np.mgrid[0:n, 0:n].astype(np.float64)
    x = u / scale - ext
    y = v / scale - ext
    r = np.hypot(x, y)
    idx = sector_index(board_angle_deg(x, y))
    sign = np.array([1.0 if s in BLACK_SECTORS else -1.0 for s in SECTORS])
    val = np.zeros((n, n), np.float32)
    ring = ((r >= R_TRIPLE_IN) & (r <= R_TRIPLE_OUT)) | ((r >= R_DOUBLE_IN) & (r <= R_DOUBLE_OUT))
    val[ring] = sign[idx][ring]
    val[r < R_BULL_IN] = 1.0
    val[(r >= R_BULL_IN) & (r < R_BULL_OUT)] = -1.0
    return val, S


def apply_h(H, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, np.float64).T
    return p[:, :2] / p[:, 2:3]


def rot_cw(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
