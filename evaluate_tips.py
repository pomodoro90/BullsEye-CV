import argparse
import csv
from pathlib import Path

import numpy as np

import dartboard as db
from tipcommon import VAL_SETS, associate, board_dets, human_ground_truth, load_calib, load_labels, load_tips_csv


def make_yolo_detector(weights, imgsz, conf, iou=0.35):
    from ultralytics import YOLO
    model = YOLO(weights)

    def detect(img_path):
        r = model.predict(str(img_path), imgsz=imgsz, conf=conf, iou=iou, device="cpu", verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, float(b.conf[0])))
        return out
    return detect


def match(gt, det, max_d=30.0):
    pairs = sorted(((np.linalg.norm(g[0] - d[0]), i, j) for i, g in enumerate(gt) for j, d in enumerate(det)))
    ui, uj, out = set(), set(), []
    for dist, i, j in pairs:
        if dist <= max_d and i not in ui and j not in uj:
            ui.add(i); uj.add(j); out.append((i, j, dist))
    return out


def run_eval(detect, session, calib, tips, labels, sets, anchor=0.0):
    rows = []
    for s in sets:
        gt = human_ground_truth(calib, tips[s])
        dets = {}
        for cam in sorted(tips[s]):
            p = Path(session) / "darts" / f"darts_{s:04d}_cam{cam}.png"
            dets[cam] = board_dets(calib, cam, detect(p))
        found = associate(dets, anchor_conf=anchor)
        pairs = match(gt, found)
        errs = [d for _, _, d in pairs]
        sc_ok = sum(db.score_from_xy(*gt[i][0])[0] == db.score_from_xy(*found[j][0])[0] for i, j, _ in pairs)
        rows.append({
            "set": f"{s:04d}", "label": labels.get(s, "?"),
            "nyilak_valodi": len(gt), "nyilak_talalt": len(found), "talalt_jo": len(pairs),
            "pontszam_jo": sc_ok,
            "atlag_hiba_mm": round(float(np.mean(errs)), 2) if errs else "",
            "max_hiba_mm": round(float(np.max(errs)), 2) if errs else "",
            "talalt_pontszamok": " ".join(db.score_from_xy(*f[0])[0] for f in found),
            "valodi_pontszamok": " ".join(db.score_from_xy(*g[0])[0] for g in gt),
        })
    return rows


def summarize(rows):
    n_gt = sum(r["nyilak_valodi"] for r in rows)
    n_det = sum(r["nyilak_talalt"] for r in rows)
    n_ok = sum(r["talalt_jo"] for r in rows)
    n_sc = sum(r["pontszam_jo"] for r in rows)
    errs = [r["atlag_hiba_mm"] for r in rows if r["atlag_hiba_mm"] != ""]
    print("\n===== ÖSSZEGZÉS =====")
    print(f"valódi nyilak: {n_gt} | talált: {n_det} | helyesen megtalált: {n_ok}  ({100 * n_ok / max(n_gt, 1):.0f}% )")
    if n_ok:
        print(f"helyes pontszám a megtaláltak közül: {n_sc}/{n_ok}  ({100 * n_sc / n_ok:.0f}% )")
    else:
        print("nincs megtalált nyíl, ezért a pontszám nem értékelhető")
    print(f"helyes pontszám az összes valódi nyílra: {n_sc}/{n_gt}  ({100 * n_sc / max(n_gt, 1):.0f}% )")
    if errs:
        print(f"átlagos helyhiba: {np.mean(errs):.2f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--calib", default="calib_board/board_calib.json")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--anchor", type=float, default=0.0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    session = Path(args.session)
    tips = load_tips_csv(session / "tips.csv")
    labels = load_labels(session / "darts" / "labels.csv")
    calib = load_calib(args.calib)
    sets = sorted(tips) if args.all else sorted(s for s in tips if s in VAL_SETS)
    print(f"Kiértékelés {len(sets)} felvételen ({'összes' if args.all else 'félretett'}).")
    try:
        detect = make_yolo_detector(args.weights, args.imgsz, args.conf)
    except ImportError:
        print("Hiányzik az ultralytics. Telepítsd:  pip install ultralytics")
        return
    rows = run_eval(detect, session, calib, tips, labels, sets, anchor=args.anchor)
    for r in rows:
        print(r["set"], "címke", r["label"], "| valódi:", r["valodi_pontszamok"], "| talált:", r["talalt_pontszamok"],
              "| hiba mm:", r["atlag_hiba_mm"])
    summarize(rows)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print(f"\nMentve: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
