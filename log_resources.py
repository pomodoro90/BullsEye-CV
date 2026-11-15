import argparse
import csv
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    sys.exit("Hiányzik a psutil. Telepítés:  .venv\\Scripts\\python.exe -m pip install psutil")


def find_server(match="server.py"):
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = p.info["cmdline"] or []
            if any(c.endswith(match) for c in cmd) and any("python" in c.lower() for c in cmd[:1]):
                return p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


class ProcTracker:
    def __init__(self, root, ncpu):
        self.root, self.ncpu, self.procs = root, ncpu, {}
        self._refresh()

    def _refresh(self):
        try:
            alive = [self.root] + self.root.children(recursive=True)
        except psutil.Error:
            alive = []
        ids = {p.pid for p in alive}
        for p in alive:
            if p.pid not in self.procs:
                self.procs[p.pid] = p
                try:
                    p.cpu_percent(None)
                except psutil.Error:
                    pass
        for pid in list(self.procs):
            if pid not in ids:
                del self.procs[pid]

    def sample(self):
        self._refresh()
        cpu = ram = 0.0
        for p in self.procs.values():
            try:
                cpu += p.cpu_percent(None) / self.ncpu
                ram += p.memory_info().rss / 1e6
            except psutil.Error:
                pass
        return cpu, ram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="meres", help="a mérés neve a fájlnévben (pl. iGPU-jatek)")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--interval", type=float, default=2.0, help="mintavételezés másodpercenként")
    ap.add_argument("--note", default="", help="szabad jegyzet (pl. hőmérséklet, fogyasztás, fényviszony)")
    ap.add_argument("--out-dir", default="measurements")
    a = ap.parse_args()

    srv = find_server()
    if srv is None:
        print("FIGYELEM: nem találom a futó server.py folyamatot - csak a rendszer-adatokat naplózom.")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"resources_{a.label}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    ncpu = psutil.cpu_count() or 1
    psutil.cpu_percent(None)
    tracker = ProcTracker(srv, ncpu) if srv else None

    rows = []
    t0 = time.monotonic()
    end = t0 + a.minutes * 60
    print(f"Naplózás: {path}  ({a.minutes} perc, {a.interval} mp-ként)  Ctrl+C = korai leállítás")
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["mp", "rendszer_cpu_%", "szerver_cpu_%", "szerver_ram_MB", "rendszer_ram_%", "rendszer_ram_MB"])
        try:
            while time.monotonic() < end:
                time.sleep(a.interval)
                sys_cpu = psutil.cpu_percent(None)
                s_cpu, s_ram = tracker.sample() if tracker else (0.0, 0.0)
                vm = psutil.virtual_memory()
                row = [round(time.monotonic() - t0), sys_cpu, round(s_cpu, 1), round(s_ram), vm.percent, round(vm.used / 1e6)]
                rows.append(row)
                w.writerow([str(x).replace(".", ",") for x in row])
                fh.flush()
        except KeyboardInterrupt:
            print("\nLeállítva.")
        if a.note:
            w.writerow([])
            w.writerow(["jegyzet", a.note])

    if not rows:
        return
    col = lambda i: [r[i] for r in rows]
    p95 = lambda v: sorted(v)[min(len(v) - 1, int(0.95 * (len(v) - 1) + 0.5))]
    print(f"\nÖsszegzés ({len(rows)} minta, {rows[-1][0]} mp):")
    for name, i, unit in (("Rendszer CPU", 1, "%"), ("Szerver CPU", 2, "%"), ("Szerver RAM", 3, "MB"), ("Rendszer RAM", 4, "%")):
        v = col(i)
        print(f"  {name:13s} átlag {statistics.mean(v):7.1f}{unit}   p95 {p95(v):7.1f}{unit}   max {max(v):7.1f}{unit}")
    if len(rows) > 10:
        d = rows[-1][3] - rows[0][3]
        print(f"  Szerver RAM változása az elejétől a végéig: {d:+d} MB  (tartós növekedés = memóriaszivárgás gyanú)")
    print(f"Fájl: {path}")


if __name__ == "__main__":
    main()
