import argparse
import csv
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def load_frames(folder, n):
    import cv2
    frames = []
    if folder:
        files = sorted(Path(folder).glob("*.png")) + sorted(Path(folder).glob("*.jpg"))
        for f in files[:n]:
            im = cv2.imread(str(f))
            if im is not None:
                frames.append(im)
    if frames:
        return frames, False
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)], True


def exported_path(weights, fmt):
    w = Path(weights)
    return w.with_name(w.stem + "_openvino_model") if fmt == "openvino" else w.with_suffix(".onnx")


def get_exported(weights, fmt, imgsz):
    p = exported_path(weights, fmt)
    if p.exists():
        return str(p)
    from ultralytics import YOLO
    print(f"  [{fmt}] export készül (egyszer kell)...", flush=True)
    out = YOLO(weights, task="detect").export(format=fmt, imgsz=imgsz, half=False)
    return str(out)


def make_predictor(kind, weights, imgsz, conf, iou):
    from ultralytics import YOLO
    if kind == "pytorch":
        model, device = YOLO(weights, task="detect"), "cpu"
    elif kind in ("openvino", "openvino-gpu"):
        model = YOLO(get_exported(weights, "openvino", imgsz), task="detect")
        device = "intel:cpu" if kind == "openvino" else "intel:gpu"
    elif kind == "onnx":
        model, device = YOLO(get_exported(weights, "onnx", imgsz), task="detect"), "cpu"
    else:
        raise ValueError(f"ismeretlen backend: {kind}")
    state = {"device": device}

    def predict(frame):
        kw = dict(imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        if state["device"] is not None:
            kw["device"] = state["device"]
        try:
            r = model.predict(frame, **kw)[0]
        except (TypeError, ValueError, RuntimeError):
            if kind == "openvino-gpu" or state["device"] is None:
                raise
            state["device"] = None
            kw.pop("device", None)
            r = model.predict(frame, **kw)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, float(b.conf[0])))
        return out
    return predict


def agreement(base, other, tol=10.0):
    total = matched = 0
    offs = []
    for b_dets, o_dets in zip(base, other):
        for bx, by, _ in b_dets:
            total += 1
            if o_dets:
                d = min(np.hypot(bx - ox, by - oy) for ox, oy, _ in o_dets)
                if d <= tol:
                    matched += 1
                    offs.append(d)
    return (100.0 * matched / total if total else None), (float(np.mean(offs)) if offs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.35)
    ap.add_argument("--runs", type=int, default=30, help="mért futások száma backendenként")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--images", default=None)
    ap.add_argument("--backends", nargs="+", default=["pytorch", "openvino", "onnx"])
    ap.add_argument("--label", default=platform.node(), help="a gép neve a CSV-ben")
    ap.add_argument("--out", default="bench_results.csv")
    args = ap.parse_args()

    try:
        import psutil
        proc = psutil.Process()
    except ImportError:
        psutil = proc = None
        print("(psutil nincs telepítve: a CPU- és memóriahasználatot nem mérem)")

    frames, synthetic = load_frames(args.images, 10)
    print(f"Gép: {args.label} | CPU: {platform.processor() or '?'} | {len(frames)} kép"
          f"{' (zajos teszt-kép: csak sebesség)' if synthetic else ''} | imgsz {args.imgsz}")

    rows, baseline = [], None
    for kind in args.backends:
        print(f"\n== {kind} ==", flush=True)
        try:
            predict = make_predictor(kind, args.weights, args.imgsz, args.conf, args.iou)
            for i in range(args.warmup):
                predict(frames[i % len(frames)])
            if proc:
                proc.cpu_percent(None)
            times, dets = [], []
            for i in range(args.runs):
                f = frames[i % len(frames)]
                t0 = time.perf_counter()
                d = predict(f)
                times.append((time.perf_counter() - t0) * 1000.0)
                if i < len(frames):
                    dets.append(d)
        except Exception as ex:
            print(f"  kihagyva: {type(ex).__name__}: {ex}")
            continue
        cpu = rss = None
        if proc:
            cpu = proc.cpu_percent(None) / (psutil.cpu_count() or 1)
            rss = proc.memory_info().rss / 1e6
        if kind == "pytorch":
            baseline = dets
        match, off = (agreement(baseline, dets) if (baseline and not synthetic and kind != "pytorch") else (None, None))
        mean, med = statistics.mean(times), statistics.median(times)
        p95 = sorted(times)[int(0.95 * (len(times) - 1))]
        rows.append({"gep": args.label, "backend": kind, "imgsz": args.imgsz, "futasok": args.runs,
                     "atlag_ms": round(mean, 1), "median_ms": round(med, 1), "p95_ms": round(p95, 1),
                     "harom_kamera_ms": round(3 * mean, 1),
                     "cpu_szazalek": None if cpu is None else round(cpu, 1),
                     "memoria_mb": None if rss is None else round(rss),
                     "talalat_egyezes_szazalek": None if match is None else round(match, 1),
                     "atlag_elteres_px": None if off is None else round(off, 2),
                     "zajos_kep": synthetic})
        print(f"  egy kép: átlag {mean:.0f} ms, medián {med:.0f} ms, p95 {p95:.0f} ms | 3 kamera ≈ {3 * mean:.0f} ms"
              + (f" | egyezés a PyTorch-csal: {match:.0f}% (átl. eltérés {off:.2f} px)" if match is not None else ""))

    if rows:
        with open(args.out, "a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if fh.tell() == 0:
                wr.writeheader()
            wr.writerows(rows)
        print(f"\nEredmények hozzáfűzve: {Path(args.out).resolve()}")
        best = min(rows, key=lambda r: r["atlag_ms"])
        print(f"Leggyorsabb: {best['backend']} ({best['atlag_ms']} ms/kép)")
    else:
        print("Egyik backend sem futott le.")
        sys.exit(1)


if __name__ == "__main__":
    main()
