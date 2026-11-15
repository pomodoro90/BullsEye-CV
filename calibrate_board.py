import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import dartboard as db


def segment(und):
    hsv = cv2.cvtColor(und, cv2.COLOR_BGR2HSV)
    h, s, v = [hsv[..., i].astype(int) for i in range(3)]
    red = ((h < 10) | (h > 165)) & (s > 110) & (v > 90)
    green = (h > 45) & (h < 95) & (s > 90) & (v > 50)
    return red, green


def signed_image(red, green, ell):
    (cx, cy), a, b, ang = ell
    mask = np.zeros(red.shape, np.uint8)
    cv2.ellipse(mask, ((cx, cy), (2 * a * 1.05, 2 * b * 1.05), ang), 255, -1)
    inside = mask > 0
    img = np.zeros(red.shape, np.float32)
    img[red & inside] = 1.0
    img[green & inside] = -1.0
    return img


def ring_geometry(red, green):
    m = (red | green).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        raise RuntimeError("Nem találtam színes gyűrűt a képen.")
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    comp = (lab == big).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)

    outer = cv2.convexHull(cnt)

    best = None
    for i in range(1, n):
        if i == big:
            continue
        area = st[i, cv2.CC_STAT_AREA]
        cx, cy = cen[i]
        if 300 <= area <= 5000 and cv2.pointPolygonTest(outer, (float(cx), float(cy)), False) > 0:
            if best is None or area > best[0]:
                best = (area, (float(cx), float(cy)))
    if best is None:
        raise RuntimeError("A bull-t nem találtam meg.")
    return outer.reshape(-1, 2).astype(np.float64), best[1]


def pose_residuals(params, K, pts_o, q):
    R, _ = cv2.Rodrigues(params[:3])
    H = K @ np.column_stack([R[:, 0], R[:, 1], params[3:6]])
    Hi = np.linalg.inv(H)
    Po = db.apply_h(Hi, pts_o)
    Pq = db.apply_h(Hi, [q])[0]
    return np.concatenate([np.hypot(Po[:, 0], Po[:, 1]) - db.R_DOUBLE_OUT, 3.0 * Pq])


def initial_pose(outer, q, K):
    from scipy.optimize import least_squares
    (cx, cy), (w, h), ang = cv2.fitEllipse(outer.astype(np.float32).reshape(-1, 1, 2))

    if w >= h:
        a, b, mu = w / 2, h / 2, np.radians(ang)
    else:
        a, b, mu = h / 2, w / 2, np.radians(ang + 90.0)
    tilt = np.arccos(np.clip(b / a, 0.0, 1.0))
    f = 0.5 * (K[0, 0] + K[1, 1])
    Z = f * db.R_DOUBLE_OUT / a
    e1 = np.array([np.cos(mu), np.sin(mu), 0.0])

    perp = np.array([-np.sin(mu), np.cos(mu)])
    d = np.array([q[0] - cx, q[1] - cy])
    far = perp if np.dot(perp, d) >= 0 else -perp
    e2 = np.array([np.cos(tilt) * far[0], np.cos(tilt) * far[1], np.sin(tilt)])
    yb = e2 if np.cross(e1, e2)[2] > 0 else -e2
    Rm = np.column_stack([e1, yb, np.cross(e1, yb)])
    rvec0, _ = cv2.Rodrigues(Rm)
    t0 = Z * np.linalg.inv(K) @ np.array([q[0], q[1], 1.0])
    x0 = np.concatenate([rvec0.ravel(), t0])
    sol = least_squares(pose_residuals, x0, args=(K, outer, q), loss="soft_l1", f_scale=2.0)
    R, _ = cv2.Rodrigues(sol.x[:3])
    H = K @ np.column_stack([R[:, 0], R[:, 1], sol.x[3:6]])
    res = pose_residuals(sol.x, K, outer, q)
    return H / H[2, 2], float(np.median(np.abs(res[:-2]))), tilt


def search_orientation(tmpl, img_s, H0, S, step=2):
    h, w = img_s.shape
    sig_img = 5.0
    ratio = float(np.sqrt(abs(np.linalg.det((H0 @ np.linalg.inv(S))[:2, :2]))))
    tb = cv2.GaussianBlur(tmpl, (0, 0), sig_img / ratio)
    ib = cv2.GaussianBlur(img_s, (0, 0), sig_img)
    ni = np.linalg.norm(ib) + 1e-9
    Sinv = np.linalg.inv(S)
    best = (-2.0, None, 0.0)
    for th in range(0, 360, step):
        Hb = H0 @ db.rot_cw(th)
        W0 = Hb @ Sinv
        warped = cv2.warpPerspective(tb, W0, (w, h))
        sc = float((warped * ib).sum() / ((np.linalg.norm(warped) + 1e-9) * ni))
        if sc > best[0]:
            best = (sc, Hb, float(th))
    return best[0], best[1]


def h_from_params(x, K):
    R, _ = cv2.Rodrigues(np.asarray(x[:3], np.float64))
    H = K @ np.column_stack([R[:, 0], R[:, 1], x[3:6]])
    return H / H[2, 2]


def params_from_h(H, K):
    A = np.linalg.inv(K) @ H
    lam = 1.0 / np.linalg.norm(A[:, 0])
    r1, r2, t = A[:, 0] * lam, A[:, 1] * lam, A[:, 2] * lam
    r1 /= np.linalg.norm(r1)
    r2 = r2 - r1 * np.dot(r1, r2)
    r2 /= np.linalg.norm(r2)
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    rvec, _ = cv2.Rodrigues(R)
    return np.concatenate([rvec.ravel(), t])


def refine_pose(tmpl, img_s, H0, K, S, sigmas=(5.0, 3.0, 2.0)):
    from scipy.optimize import minimize
    h, w = img_s.shape
    Sinv = np.linalg.inv(S)
    x = params_from_h(H0, K)
    D = np.array([0.01, 0.01, 0.01, 2.0, 2.0, 2.0])
    corr = -1.0
    for sg in sigmas:
        W0 = H0 @ Sinv
        ratio = float(np.sqrt(abs(np.linalg.det(W0[:2, :2]))))
        tb = cv2.GaussianBlur(tmpl, (0, 0), sg / ratio)
        ib = cv2.GaussianBlur(img_s, (0, 0), sg)
        ni = np.linalg.norm(ib) + 1e-9

        def cost(z):
            H = h_from_params(x + D * z, K)
            wp = cv2.warpPerspective(tb, H @ Sinv, (w, h))
            return -float((wp * ib).sum() / ((np.linalg.norm(wp) + 1e-9) * ni))

        res = minimize(cost, np.zeros(6), method="Powell",
                       options={"xtol": 1e-2, "ftol": 1e-6, "maxiter": 6})
        x = x + D * res.x
        H0 = h_from_params(x, K)
        corr = -res.fun
    return h_from_params(x, K), float(corr)


def pose_from_h(Hb, K):
    objs = []
    for r in (db.R_BULL_OUT, db.R_TRIPLE_IN, db.R_TRIPLE_OUT, db.R_DOUBLE_IN, db.R_DOUBLE_OUT):
        for ang in range(0, 360, 10):
            t = np.radians(ang)
            objs.append((r * np.sin(t), -r * np.cos(t)))
    objs = np.array(objs, np.float64)
    img = db.apply_h(Hb, objs)
    obj3 = np.hstack([objs, np.zeros((len(objs), 1))])
    ok, rvec, tvec = cv2.solvePnP(obj3, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("solvePnP sikertelen")
    rvec, tvec = cv2.solvePnPRefineLM(obj3, img, K, None, rvec, tvec)
    Rm, _ = cv2.Rodrigues(rvec)
    Hp = K @ np.column_stack([Rm[:, 0], Rm[:, 1], tvec.ravel()])
    Hp = Hp / Hp[2, 2]
    resid = np.linalg.norm(db.apply_h(Hp, objs) - img, axis=1)
    return Hp, rvec.ravel(), tvec.ravel(), float(resid.mean()), float(resid.max())


def ask_click(img, cam):
    pt = {}
    win = f"CAM {cam}: kattints a 20-as szeletbe (szam es kozeppont kozott), majd Enter"

    def cb(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pt["p"] = (x, y)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cb)
    while True:
        vis = img.copy()
        if "p" in pt:
            cv2.circle(vis, pt["p"], 8, (255, 0, 255), 2)
        cv2.putText(vis, f"CAM {cam}: kattints a 20-as szeletbe, majd Enter", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 32) and "p" in pt:
            break
    cv2.destroyWindow(win)
    return pt["p"]


def draw_overlay(und, H, click=None):
    vis = und.copy()
    th = np.radians(np.arange(0, 361, 2))
    for r in db.RING_RADII:
        pts = db.apply_h(H, np.column_stack([r * np.sin(th), -r * np.cos(th)]))
        cv2.polylines(vis, [pts.astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 1, cv2.LINE_AA)
    for k in range(20):
        a = np.radians(9 + 18 * k)
        p = db.apply_h(H, [(db.R_BULL_OUT * np.sin(a), -db.R_BULL_OUT * np.cos(a)),
                           (db.R_DOUBLE_OUT * np.sin(a), -db.R_DOUBLE_OUT * np.cos(a))])
        cv2.line(vis, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (255, 255, 0), 1, cv2.LINE_AA)
    for k, n in enumerate(db.SECTORS):
        a = np.radians(18 * k)
        p = db.apply_h(H, [(188 * np.sin(a), -188 * np.cos(a))])[0]
        cv2.putText(vis, str(n), (int(p[0]) - 10, int(p[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 0, 255), 2, cv2.LINE_AA)
    c = db.apply_h(H, [(0, 0)])[0]
    cv2.drawMarker(vis, (int(c[0]), int(c[1])), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
    if click is not None:
        cv2.circle(vis, (int(click[0]), int(click[1])), 6, (0, 165, 255), 2)
    return vis


def parse_clicks(items):
    out = {}
    for it in items or []:
        cam, xy = it.split(":")
        x, y = xy.split(",")
        out[int(cam)] = (float(x), float(y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--empty-index", type=int, default=1, help="melyik üres-tábla felvételt használja")
    ap.add_argument("--clicks", nargs="*", help="kamera:x,y a 20-as szeletben, pl. 0:886,343")
    ap.add_argument("--out", default="calib_board")
    args = ap.parse_args()

    session = Path(args.session)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    intr = json.load(open(args.intrinsics, encoding="utf-8"))
    clicks = parse_clicks(args.clicks)
    tmpl, S = db.render_signed_template(scale=2.0)
    results = {}

    for cam in args.cams:
        print(f"\n=== CAM {cam} ===")
        path = session / "empty" / f"empty_{args.empty_index:04d}_cam{cam}.png"
        img = cv2.imread(str(path))
        if img is None:
            print(f"   HIBA: nem találom: {path}")
            continue
        K = np.array(intr[f"cam{cam}"]["camera_matrix"], np.float64)
        dist = np.array(intr[f"cam{cam}"]["dist_coeffs"], np.float64)
        und = cv2.undistort(img, K, dist, None, K)

        red, green = segment(und)
        outer, bull = ring_geometry(red, green)
        H0, med_res, tilt = initial_pose(outer, bull, K)
        print(f"   kezdeti póz: dőlés {np.degrees(tilt):.0f} fok, kontúr-illeszkedés mediánja {med_res:.2f} mm")
        ell = cv2.fitEllipse(outer.astype(np.float32).reshape(-1, 1, 2))
        (ecx, ecy), (ew, eh), eang = ell
        img_s = signed_image(red, green, ((ecx, ecy), ew / 2, eh / 2, eang))
        sc0, Hb = search_orientation(tmpl, img_s, H0, S)
        print(f"   szín-korreláció a kezdeti pózzal: {sc0:.3f}")
        Hb, cc = refine_pose(tmpl, img_s, Hb, K, S)
        print(f"   korreláció a póz-finomítás után: {cc:.3f}   (0.9 felett jó)")

        click = clicks.get(cam)
        if click is None:
            click = ask_click(und, cam)
        p = db.apply_h(np.linalg.inv(Hb), [click])[0]
        k = int(db.sector_index(db.board_angle_deg(p[0], p[1])))
        if k % 2 == 1:
            print("   FIGYELEM: a kattintott szelet szín-paritása nem egyezik a sablonnal.")
            print("             Lehet, hogy rossz helyre kattintottál, vagy a szín-kiosztás eltér.")
        Hb = Hb @ db.rot_cw(k * 18.0)

        Hp, rvec, tvec, rmean, rmax = pose_from_h(Hb, K)
        Rm, _ = cv2.Rodrigues(rvec)
        cam_pos = (-Rm.T @ tvec).tolist()
        print(f"   kamera helyzete a tábla koordinátarendszerében (mm): "
              f"x={cam_pos[0]:.0f}, y={cam_pos[1]:.0f}, z={cam_pos[2]:.0f}")

        vis = draw_overlay(und, Hp, click)
        cv2.imwrite(str(out / f"overlay_cam{cam}.png"), vis)
        cv2.imwrite(str(out / f"overlay_cam{cam}_ecc.png"), draw_overlay(und, Hb, click))

        results[f"cam{cam}"] = {
            "image_size": [und.shape[1], und.shape[0]],
            "camera_matrix": K.tolist(),
            "dist_coeffs": dist.tolist(),
            "H_board_to_img": Hp.tolist(),
            "H_img_to_board": np.linalg.inv(Hp).tolist(),
            "H_ecc_board_to_img": Hb.tolist(),
            "rvec": rvec.tolist(),
            "tvec": tvec.tolist(),
            "camera_pos_mm": cam_pos,
            "ecc_correlation": cc,
            "pose_resid_mean_px": rmean,
            "pose_resid_max_px": rmax,
            "click": list(click),
        }

    with open(out / "board_calib.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nKész. Mentve: {(out / 'board_calib.json').resolve()}")
    print(f"Ellenőrző képek: {out.resolve()} (overlay_cam*.png)")


if __name__ == "__main__":
    main()
