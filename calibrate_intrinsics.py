import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_corners(files, pattern, square):
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    obj_pts, img_pts, used = [], [], []
    size = None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print(f"   kihagyva (nincs sarok): {f.name}")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        obj_pts.append(objp)
        img_pts.append(corners)
        used.append(f)
    return obj_pts, img_pts, used, size


def per_image_errors(obj_pts, img_pts, rvecs, tvecs, K, dist):
    errs = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        errs.append(float(np.sqrt(np.mean(np.sum((i - proj) ** 2, axis=2)))))
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--chess", default="9x6")
    ap.add_argument("--square", type=float, default=25.0, help="négyzet oldala mm-ben")
    ap.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="calib")
    ap.add_argument("--fix-aspect", action="store_true", help="fx = fy rögzítése (négyzet pixelek)")
    ap.add_argument("--simple", action="store_true",
                    help="egyszerűbb modell: fx=fy, nincs tangenciális torzítás, nincs k3")
    args = ap.parse_args()

    flags = 0
    if args.fix_aspect or args.simple:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if args.simple:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3

    cols, rows = (int(x) for x in args.chess.lower().split("x"))
    pattern = (cols, rows)
    session = Path(args.session)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    result = {}

    for cam in args.cams:
        files = sorted((session / "chess").glob(f"chess_*_cam{cam}.png"))
        print(f"\n=== CAM {cam}: {len(files)} kép ===")
        obj_pts, img_pts, used, size = load_corners(files, pattern, args.square)
        if len(used) < 8:
            print(f"   HIBA: csak {len(used)} használható kép, legalább 8-10 kellene.")
            continue

        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None, flags=flags)
        errs = per_image_errors(obj_pts, img_pts, rvecs, tvecs, K, dist)

        keep = [k for k, e in enumerate(errs) if e < max(2.5 * np.median(errs), 0.6)]
        if 8 <= len(keep) < len(errs):
            dropped = [used[k].name for k in range(len(errs)) if k not in keep]
            print(f"   kiugró képek kizárva: {dropped}")
            obj_pts = [obj_pts[k] for k in keep]
            img_pts = [img_pts[k] for k in keep]
            used = [used[k] for k in keep]
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None, flags=flags)
            errs = per_image_errors(obj_pts, img_pts, rvecs, tvecs, K, dist)

        worst = sorted(zip(errs, [u.name for u in used]), reverse=True)[:3]
        print("   legrosszabb képek: " + ", ".join(f"{n} ({e:.2f} px)" for e, n in worst))
        print(f"   használt képek: {len(used)}")
        print(f"   RMS reprojekciós hiba: {rms:.3f} px   (0.5 alatt jó, 1.0 felett gyanús)")
        print(f"   fókusz: fx={K[0,0]:.1f} fy={K[1,1]:.1f}   közép: cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
        print(f"   torzítás (k1,k2,p1,p2,k3): {np.round(dist.ravel(), 4).tolist()}")

        allp = np.vstack([p.reshape(-1, 2) for p in img_pts])
        cov = (allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max())
        print(f"   sarkok lefedése: x {cov[0]:.0f}-{cov[1]:.0f} / {size[0]},  y {cov[2]:.0f}-{cov[3]:.0f} / {size[1]}")

        result[f"cam{cam}"] = {
            "image_size": list(size),
            "rms": float(rms),
            "camera_matrix": K.tolist(),
            "dist_coeffs": dist.ravel().tolist(),
            "n_images": len(used),
        }

        ref = sorted((session / "empty").glob(f"empty_*_cam{cam}.png")) or used
        img = cv2.imread(str(ref[0]))
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, 0)
        und = cv2.undistort(img, K, dist, None, newK)
        both = np.hstack([cv2.resize(img, None, fx=0.6, fy=0.6),
                          cv2.resize(und, None, fx=0.6, fy=0.6)])
        cv2.putText(both, "eredeti", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(both, "kiegyenesitett", (both.shape[1] // 2 + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(str(out / f"undistort_cam{cam}.png"), both)

    with open(out / "intrinsics.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nKész. Mentve: {(out / 'intrinsics.json').resolve()}")


if __name__ == "__main__":
    main()
