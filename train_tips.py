import argparse
import shutil
from pathlib import Path

import cv2

from tipcommon import BOX, VAL_SETS, load_tips_csv


def build_dataset(sessions, out, final):
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for f in (out / sub).glob("*"):
            f.unlink()
    n = {"train": 0, "val": 0}
    missing = 0
    for k, session in enumerate(sessions):
        tips = load_tips_csv(Path(session) / "tips.csv")
        for s in sorted(tips):
            for cam, pts in tips[s].items():
                src = Path(session) / "darts" / f"darts_{s:04d}_cam{cam}.png"
                if not src.exists():
                    missing += 1
                    continue
                is_val = (k == 0 and s in VAL_SETS)
                splits = ["train"] if (final or not is_val) else ["val"]
                if final and is_val:
                    splits = ["train", "val"]
                img = cv2.imread(str(src))
                h, w = img.shape[:2]
                lines = []
                for (x, y) in pts:
                    x0, y0 = max(0.0, x - BOX / 2), max(0.0, y - BOX / 2)
                    x1, y1 = min(float(w), x + BOX / 2), min(float(h), y + BOX / 2)
                    lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                                 f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
                name = f"s{k}_{src.name}"
                for sp in splits:
                    shutil.copy(src, out / f"images/{sp}" / name)
                    (out / f"labels/{sp}" / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n")
                    n[sp] += 1
    yaml = out / "data.yaml"
    yaml.write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: tip\n")
    print(f"Adatkészlet kész: {n['train']} tanító kép, {n['val']} kiértékelő kép"
          + (f" ({missing} kép hiányzik)" if missing else "") + f"  [{len(sessions)} mappából]")
    return yaml


def pick_device():
    import torch
    if not torch.cuda.is_available():
        print("Nincs használható CUDA (a PyTorch CPU-s változata van telepítve, vagy nincs illesztőprogram).")
        return "cpu", False
    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        x = torch.randn(1, 3, 32, 32, device="cuda")
        w = torch.randn(4, 3, 3, 3, device="cuda")
        torch.nn.functional.conv2d(x, w).sum().item()

        amp = cap[0] >= 7
        print(f"Eszköz: {name} (compute capability {cap[0]}.{cap[1]}), fp16 {'be' if amp else 'ki'}")
        return 0, amp
    except Exception as e:
        print(f"A videókártya nem használható ezzel a PyTorch-csal ({str(e).splitlines()[0][:100]}).")
        print("Folytatom CPU-n (lassabb).")
        return "cpu", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="további, bejelölt (tips.csv-s) mappák a tanításhoz")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--final", action="store_true", help="minden adatból tanul (nincs félretett rész)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--data-dir", default="tipdata")
    args = ap.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        print("Hiányzik az ultralytics. Telepítsd:  pip install ultralytics")
        return

    yaml = build_dataset([args.session] + list(args.extra), Path(args.data_dir), args.final)
    device, amp = pick_device()
    if device == "cpu":
        print("Eszköz: CPU (lassabb, türelem)")

    name = args.name or ("tips_final" if args.final else "tips")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device, amp=amp,
        project=str(Path("runs").resolve()), name=name,
        patience=30, workers=0, close_mosaic=10,
        degrees=3, translate=0.1, scale=0.3, fliplr=0.5, flipud=0.0, hsv_v=0.4,
        exist_ok=True, verbose=True,
    )
    print(f"\nKész. A modell itt van:  runs\\{name}\\weights\\best.pt")


if __name__ == "__main__":
    main()
