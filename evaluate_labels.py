import argparse
import csv
import re
from pathlib import Path

import numpy as np

import dartboard as db
from evaluate_tips import make_yolo_detector
from tipcommon import associate, board_dets, load_calib, load_labels


def margin_mm(x, y):
    r = float(np.hypot(x, y))
    d_ring = min(abs(r - R) for R in db.RING_RADII)
    if r <= db.R_BULL_OUT:
        return d_ring
    ang = float(db.board_angle_deg(x, y))
    k = (ang + 9.0) % 18.0
    d_wire = np.radians(min(k, 18.0 - k)) * r
    return float(min(d_ring, d_wire))


def run(detect, session, calib, labels, cams=(0, 1, 2), anchor=0.0, max_spread=25.0, weak_spread=None, nms_mm=0.0):
    sets = sorted({int(re.search(r"darts_(\d+)_cam", p.name).group(1))
                   for p in (Path(session) / "darts").glob("darts_*_cam0.png")})
    rows = []
    for s in sets:
        dets = {}
        for cam in cams:
            p = Path(session) / "darts" / f"darts_{s:04d}_cam{cam}.png"
            if not p.exists():
                continue
            dets[cam] = board_dets(calib, cam, detect(p))
        found = associate(dets, max_spread=max_spread, anchor_conf=anchor, weak_spread=weak_spread, nms_mm=nms_mm)
        lab = labels.get(s, "?")
        pred = [db.score_from_xy(*f[0])[0] for f in found]
        row = {"set": f"{s:04d}", "label": lab, "talalt_nyilak": len(found), "talalt_pontszamok": " ".join(pred),
               "eredmeny": "", "tavolsag_vezetektol_mm": "", "kamerak": ""}
        if len(found) == 1:
            row["eredmeny"] = "jo" if pred[0] == lab else "HIBA"
            row["tavolsag_vezetektol_mm"] = round(margin_mm(*found[0][0]), 1)
            row["kamerak"] = found[0][1]
        elif len(found) == 0:
            row["eredmeny"] = "NEM TALALT"
        else:
            row["eredmeny"] = "tobb nyil (" + ("cimke benne" if lab in pred else "cimke NINCS benne") + ")"
        rows.append(row)
    return rows


def summarize(rows):
    single = [r for r in rows if r["eredmeny"] in ("jo", "HIBA")]
    ok = sum(r["eredmeny"] == "jo" for r in single)
    print("\n===== ÖSSZEGZÉS =====")
    print(f"felvételek: {len(rows)} | egy nyilat talált: {len(single)} | jó pontszám: {ok}/{len(single)}"
          + (f"  ({100 * ok / len(single):.0f}% )" if single else ""))
    other = [r for r in rows if r["eredmeny"] not in ("jo", "HIBA")]
    if other:
        print(f"nem egyértelmű ({len(other)} db): " + ", ".join(f"{r['set']}:{r['eredmeny']}" for r in other))
    errs = [r for r in single if r["eredmeny"] == "HIBA"]
    if errs:
        print("\nHibák (a hegy távolsága a legközelebbi vezetéktől):")
        for r in errs:
            print(f"  {r['set']}: címke {r['label']}, talált {r['talalt_pontszamok']}, "
                  f"{r['tavolsag_vezetektol_mm']} mm a vezetéktől")
    near = [r for r in single if r["tavolsag_vezetektol_mm"] != "" and r["tavolsag_vezetektol_mm"] < 3]
    if near:
        okn = sum(r["eredmeny"] == "jo" for r in near)
        print(f"\nvezetékhez nagyon közeli (<3 mm) nyilak: {okn}/{len(near)} jó")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--calib", default="calib_board/board_calib.json")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--anchor", type=float, default=0.25)
    ap.add_argument("--max-spread", type=float, default=15.0)
    ap.add_argument("--weak-spread", type=float, default=8.0)
    ap.add_argument("--iou", type=float, default=0.35)
    ap.add_argument("--out", default="eval_labels.csv")
    args = ap.parse_args()

    labels = load_labels(Path(args.session) / "darts" / "labels.csv")
    calib = load_calib(args.calib)
    detect = make_yolo_detector(args.weights, args.imgsz, args.conf, args.iou)
    rows = run(detect, args.session, calib, labels, anchor=args.anchor, max_spread=args.max_spread,
               weak_spread=args.weak_spread, nms_mm=6.0)
    for r in rows:
        print(r["set"], "címke", r["label"], "| talált:", r["talalt_pontszamok"] or "-", "|", r["eredmeny"],
              "| vezetéktől:", r["tavolsag_vezetektol_mm"], "mm")
    summarize(rows)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print(f"\nMentve: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
