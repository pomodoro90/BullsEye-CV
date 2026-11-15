import argparse
import asyncio
import json
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from game import X01Game, normalize_label
from stats import StatsDB

HERE = Path(__file__).resolve().parent


POWER_ACTIONS = ("shutdown", "restart", "cancel", "quit")
QUIT_EXIT_CODE = 42
POWER_DELAY = 15


def power_argv(action, windows=None):
    win = (os.name == "nt") if windows is None else windows
    if win:
        return {"shutdown": ["shutdown", "/s", "/t", str(POWER_DELAY)],
                "restart": ["shutdown", "/r", "/t", str(POWER_DELAY)],
                "cancel": ["shutdown", "/a"]}[action]
    mins = max(1, round(POWER_DELAY / 60))
    return {"shutdown": ["shutdown", "-h", f"+{mins}"],
            "restart": ["shutdown", "-r", f"+{mins}"],
            "cancel": ["shutdown", "-c"]}[action]


def run_power(action):
    import subprocess
    argv = power_argv(action)
    print(f"[power] {' '.join(argv)}", flush=True)
    subprocess.Popen(argv)


class Hub:
    def __init__(self):
        self.queues = set()
        self.loop = None

    def subscribe(self):
        q = asyncio.Queue(maxsize=1)
        self.queues.add(q)
        return q

    def unsubscribe(self, q):
        self.queues.discard(q)

    def _push(self, msg):
        for q in list(self.queues):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(msg)

    def publish(self, msg):
        if self.loop is None:
            return
        try:
            self.loop.call_soon_threadsafe(self._push, msg)
        except RuntimeError:
            pass


class AppState:
    def __init__(self, data_dir, sim):
        self.lock = threading.RLock()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.game_path = self.data_dir / "game.json"
        self.log_path = self.data_dir / "events.jsonl"
        self.hub = Hub()
        self.sim = sim
        self.measure = {"on": False, "session": "", "condition": ""}
        self.power_enabled = False
        self.quit_hook = None
        try:
            self.stats = StatsDB(self.data_dir / "darts.db")
        except Exception as ex:
            print(f"[server] a statisztikai adatbázis nem nyitható meg: {ex}", flush=True)
            self.stats = None
        self.engine = None
        self.game = None
        self.log = deque(maxlen=40)
        self.seq = 0
        self._load_game()

    def _load_game(self):
        try:
            if self.game_path.exists():
                self.game = X01Game.from_json(json.loads(self.game_path.read_text(encoding="utf-8")))
        except Exception as ex:
            print(f"[server] a mentett játék nem tölthető be: {ex}", flush=True)
            self.game = None

    def _stats_sync(self):
        if not self.stats or not self.game:
            return
        try:
            if self.game.legs > 1:
                self.stats.save_match(self.game)
            elif self.game.over:
                self.stats.save_game(self.game, "finished")
            else:
                self.stats.delete_finished(self.game.uid)
        except Exception as ex:
            print(f"[server] statisztika mentési hiba: {ex}", flush=True)

    def _archive_game(self):
        if not self.stats or not self.game or self.game.over:
            return
        try:

            self.stats.save_game(self.game, "abandoned", leg=self.game.leg_index if self.game.legs > 1 else None)
        except Exception as ex:
            print(f"[server] statisztika mentési hiba: {ex}", flush=True)

    def _save_game(self):
        self._stats_sync()
        try:
            tmp = self.game_path.with_suffix(".tmp")
            data = self.game.to_json() if self.game else {}
            if self.game:
                tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, self.game_path)
            elif self.game_path.exists():
                self.game_path.unlink()
        except Exception as ex:
            print(f"[server] mentési hiba: {ex}", flush=True)

    def _log_file(self, rec):
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def state(self):
        with self.lock:
            board = self.engine.tracker.snapshot() if self.engine else []
            g = self.game.state() if self.game else None

            awaiting = bool(g and g["turn_ended"] and g["winner"] is None and len(board) > 0)
            return {"seq": self.seq, "time": time.time(), "mode": self.engine.mode if self.engine else "none",
                    "engine_status": self.engine.status if self.engine else "nincs motor",
                    "warning": self.engine.warning if self.engine else "", "game": g, "board": board, "awaiting_clear": awaiting, "log": list(self.log)}

    def publish(self):
        with self.lock:
            self.seq += 1
            msg = json.dumps(self.state(), ensure_ascii=False)
        self.hub.publish(msg)

    def on_event(self, e):
        with self.lock:
            rec = {"t": time.time(), **e}
            entry = {"t": rec["t"], "type": e["type"], "label": e.get("label"), "points": e.get("points"),
                     "uncertain": e.get("uncertain", False), "applied": False}
            if e["type"] == "throw":
                if self.game is not None and self.game.add_dart(e["label"], e.get("uncertain", False), "camera",
                                                                round(e["x"], 1), round(e["y"], 1)):
                    entry["applied"] = True
                    self._save_game()
            if e["type"] == "throw" and self.measure["on"] and self.stats:
                try:
                    self.stats.m_add_pending(e, self.measure["session"], self.measure["condition"])
                except Exception as ex:
                    print(f"[server] mérés mentési hiba: {ex}", flush=True)
            self.log.appendleft(entry)
            self._log_file(rec)
        print(f"[esemeny] {e['type']} {e.get('label', '')} {'(alkalmazva)' if entry['applied'] else ''}", flush=True)
        self.publish()

    def on_status(self, _):
        self.publish()

    def new_game(self, players, start, double_out, double_in, legs=1):
        with self.lock:
            self._archive_game()
            self.game = X01Game(players, start, double_out, double_in, legs)
            self._save_game()
        self.publish()

    def command(self, name, **kw):
        with self.lock:
            g = self.game
            if name != "new" and g is None:
                raise ValueError("Nincs folyamatban játék.")
            if name == "undo":
                if not g.undo():
                    raise ValueError("Nincs mit visszavonni.")
            elif name == "end_turn":
                if not g.end_turn():
                    raise ValueError("A játék véget ért.")
            elif name == "edit":
                g.edit(int(kw["index"]), kw["label"])
            elif name == "add":
                if not g.add_dart(kw["label"], False, "manual"):
                    raise ValueError("A játék véget ért.")
            elif name == "end_game":
                self._archive_game()
                self.game = None
            self._save_game()
        self.publish()


async def read_json(request: Request):
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def err(msg, code=400):
    return JSONResponse({"error": msg}, status_code=code)


def make_app(app_state: AppState):
    S = app_state

    async def index(request):
        return FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-cache"})

    async def api_state(request):
        return JSONResponse(S.state())

    async def api_events(request):
        q = S.hub.subscribe()

        async def gen():
            try:
                yield f"retry: 2000\ndata: {json.dumps(S.state(), ensure_ascii=False)}\n\n"
                while True:
                    try:
                        msg = await asyncio.wait_for(q.get(), 15)
                        yield f"data: {msg}\n\n"
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
            finally:
                S.hub.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def api_new(request):
        d = await read_json(request)
        players = d.get("players") or []
        try:
            start = int(d.get("start", 501))
            legs = int(d.get("legs", 1))
            S.new_game(players, start, bool(d.get("double_out", True)), bool(d.get("double_in", False)), legs)
        except (ValueError, TypeError) as ex:
            return err(str(ex))
        return JSONResponse({"ok": True})

    def simple(name):
        async def handler(request):
            d = await read_json(request)
            try:
                if name in ("edit", "add") and "label" in d:
                    d["label"] = normalize_label(d["label"])
                S.command(name, **d)
            except (ValueError, KeyError, TypeError) as ex:
                return err(str(ex))
            return JSONResponse({"ok": True})
        return handler

    async def api_board_reset(request):
        S.engine.reset_board()
        return JSONResponse({"ok": True})

    async def api_power(request):
        d = await read_json(request)
        action = d.get("action")
        if action not in POWER_ACTIONS:
            return err("ismeretlen művelet")
        if action == "quit":
            if not S.quit_hook:
                return err("a program leállítása ebben az indításban nem támogatott")
            print("[power] a program leállítása a weboldalról", flush=True)
            S.quit_hook()
            return JSONResponse({"ok": True})
        if not S.power_enabled:
            return err("a gép leállítása ki van kapcsolva ezen a szerveren (szimuláció vagy --no-power)")
        run_power(action)
        return JSONResponse({"ok": True, "delay": POWER_DELAY if action != "cancel" else 0})

    def need_stats():
        if not S.stats:
            raise ValueError("A statisztikai adatbázis nem érhető el.")
        return S.stats

    async def api_stats_players(request):
        try:
            return JSONResponse({"players": need_stats().players(), "games": need_stats().recent_games(10)})
        except ValueError as ex:
            return err(str(ex), 503)

    async def api_stats_player(request):
        try:
            p = need_stats().player(request.query_params.get("name", ""))
        except ValueError as ex:
            return err(str(ex), 503)
        return JSONResponse(p) if p else err("Nincs ilyen játékos.", 404)

    async def api_stats_delete_game(request):
        d = await read_json(request)
        try:
            ok = need_stats().delete_game(int(d.get("id")))
        except (ValueError, TypeError) as ex:
            return err(str(ex))
        return JSONResponse({"ok": ok})

    async def api_stats_delete_player(request):
        d = await read_json(request)
        try:
            ok = need_stats().delete_player(str(d.get("name", "")))
        except ValueError as ex:
            return err(str(ex))
        return JSONResponse({"ok": ok})

    def measure_view(session=None):
        st = need_stats()
        return {"on": S.measure["on"], "session": S.measure["session"], "condition": S.measure["condition"],
                "pending": st.m_pending(), "sessions": st.m_sessions(),
                "summary": st.m_summary(session or None)}

    async def api_measure(request):
        try:
            return JSONResponse(measure_view(request.query_params.get("session")))
        except ValueError as ex:
            return err(str(ex), 503)

    async def api_measure_setup(request):
        d = await read_json(request)
        with S.lock:
            if "on" in d:
                S.measure["on"] = bool(d["on"])
            if "session" in d:
                S.measure["session"] = str(d["session"]).strip()[:40]
            if "condition" in d:
                S.measure["condition"] = str(d["condition"]).strip()[:60]
        return JSONResponse({"ok": True})

    async def api_measure_label(request):
        d = await read_json(request)
        try:
            tl = normalize_label(d["true_label"]) if d.get("true_label") else None
            need_stats().m_label(d.get("id"), str(d.get("outcome", "")), tl)
        except (ValueError, KeyError, TypeError) as ex:
            return err(str(ex))
        return JSONResponse({"ok": True})

    async def api_measure_all_correct(request):
        try:
            return JSONResponse({"ok": True, "n": need_stats().m_label_all_correct()})
        except ValueError as ex:
            return err(str(ex), 503)

    async def api_measure_missed(request):
        d = await read_json(request)
        try:
            need_stats().m_add_missed(normalize_label(d.get("true_label", "")), S.measure["session"], S.measure["condition"])
        except (ValueError, KeyError, TypeError) as ex:
            return err(str(ex))
        return JSONResponse({"ok": True})

    async def api_measure_delete(request):
        d = await read_json(request)
        try:
            if d.get("all_pending"):
                return JSONResponse({"ok": True, "n": need_stats().m_clear_pending()})
            return JSONResponse({"ok": need_stats().m_delete(int(d.get("id")))})
        except (ValueError, TypeError) as ex:
            return err(str(ex))

    async def api_measure_export(request):
        import csv
        import io
        try:
            session = request.query_params.get("session") or None
            rows = need_stats().m_export(session)
        except ValueError as ex:
            return err(str(ex), 503)
        hu = request.query_params.get("fmt", "hu") != "std"
        buf = io.StringIO()
        cols = ["sorszam", "id", "t", "session", "condition", "predicted", "uncertain", "cams", "spread", "conf", "margin",
                "x", "y", "latency_ms", "true_label", "outcome"]
        w = csv.writer(buf, delimiter=";" if hu else ",")
        w.writerow(cols)
        for n, r in enumerate(rows, 1):
            r = dict(r, sorszam=n)
            vals = []
            for c in cols:
                v = r[c]
                if v is None:
                    v = ""
                elif hu and isinstance(v, float):
                    v = str(round(v, 3)).replace(".", ",")
                vals.append(v)
            w.writerow(vals)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (session or "osszes"))
        return Response("\ufeff" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="meresek_{safe}.csv"'})

    async def api_view(request):
        jpg = S.engine.view_jpeg() if S.engine else None
        if jpg is None:
            return Response(status_code=204)
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def sim_throw(request):
        d = await read_json(request)
        try:
            label = normalize_label(d.get("label", ""))
            S.engine.throw(label, bool(d.get("uncertain", False)))
        except (ValueError, KeyError) as ex:
            return err(str(ex))
        return JSONResponse({"ok": True})

    async def sim_clear(request):
        S.engine.clear()
        return JSONResponse({"ok": True})

    async def healthz(request):
        return JSONResponse({"ok": True, "mode": S.engine.mode, "status": S.engine.status,
                             "detail": getattr(S.engine, "detail", "")})

    routes = [
        Route("/", index),
        Route("/healthz", healthz),
        Route("/api/state", api_state),
        Route("/api/events", api_events),
        Route("/api/view.jpg", api_view),
        Route("/api/new", api_new, methods=["POST"]),
        Route("/api/undo", simple("undo"), methods=["POST"]),
        Route("/api/end_turn", simple("end_turn"), methods=["POST"]),
        Route("/api/edit", simple("edit"), methods=["POST"]),
        Route("/api/add", simple("add"), methods=["POST"]),
        Route("/api/end_game", simple("end_game"), methods=["POST"]),
        Route("/api/board_reset", api_board_reset, methods=["POST"]),
        Route("/api/power", api_power, methods=["POST"]),
        Route("/api/measure", api_measure),
        Route("/api/measure/setup", api_measure_setup, methods=["POST"]),
        Route("/api/measure/label", api_measure_label, methods=["POST"]),
        Route("/api/measure/all_correct", api_measure_all_correct, methods=["POST"]),
        Route("/api/measure/missed", api_measure_missed, methods=["POST"]),
        Route("/api/measure/delete", api_measure_delete, methods=["POST"]),
        Route("/api/measure/export.csv", api_measure_export),
        Route("/api/stats/players", api_stats_players),
        Route("/api/stats/player", api_stats_player),
        Route("/api/stats/delete_game", api_stats_delete_game, methods=["POST"]),
        Route("/api/stats/delete_player", api_stats_delete_player, methods=["POST"]),
    ]
    if S.sim:
        routes += [Route("/api/sim/throw", sim_throw, methods=["POST"]),
                   Route("/api/sim/clear", sim_clear, methods=["POST"])]

    @asynccontextmanager
    async def lifespan(app):
        S.hub.loop = asyncio.get_running_loop()
        S.engine.start()
        try:
            yield
        finally:
            S.engine.stop()

    return Starlette(routes=routes, lifespan=lifespan)


def build(args):
    from engine import LiveEngine, SimEngine
    S = AppState(args.data_dir, args.sim)
    S.power_enabled = not args.sim and not args.no_power
    if args.sim:
        S.engine = SimEngine(S.on_event, S.on_status)
    else:
        if not args.weights:
            raise SystemExit("Add meg a modellt: --weights runs\\tips\\weights\\best.pt   (vagy használd a --sim kapcsolót)")
        S.engine = LiveEngine(S.on_event, S.on_status, args)
    return S, make_app(S)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="kamera nélküli szimuláció")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--calib", default="calib_board/board_calib.json")
    ap.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--devices", type=int, nargs="+", default=None,
                    help="USB-eszközindexek a --cams sorrendjében; pl. --devices 2 1 0 (ha a cam0 és cam2 fel van cserélve)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--anchor", type=float, default=0.25)
    ap.add_argument("--max-spread", type=float, default=15.0)
    ap.add_argument("--weak-spread", type=float, default=8.0)
    ap.add_argument("--fast-confirm", type=float, default=None, metavar="CONF",
                    help="gyors megerősítés: az új nyilat egy kiértékelés után elfogadja, ha mind a 3 kamera legalább ennyire "
                         "biztos benne (pl. 0.75); alapból kikapcsolva")
    ap.add_argument("--auto-devices", action="store_true",
                    help="indításkor a képek alapján kitalálja, melyik USB-kamera melyik (a --devices csak tartalék)")
    ap.add_argument("--device-candidates", type=int, nargs="+", default=None,
                    help="mely USB-eszközindexeket nézze az automatikus azonosításkor (alap: 0 .. kamerák száma+1)")
    ap.add_argument("--iou", type=float, default=0.35)
    ap.add_argument("--recheck", type=float, default=2.5)
    ap.add_argument("--settle", type=float, default=0.8)
    ap.add_argument("--device", default="auto", help="auto | cpu | 0 | intel:gpu | intel:cpu")
    ap.add_argument("--fallback-weights", default=None,
                    help="tartalék modell, ha az elsődleges nem tölthető be (pl. a .pt, ha az elsődleges az OpenVINO-mappa)")
    ap.add_argument("--fallback-device", default="cpu")
    ap.add_argument("--no-power", action="store_true", help="a weboldalról ne lehessen leállítani a gépet")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", default="data", help="ide menti a játékot és az események naplóját")
    return ap.parse_args(argv)


def main():
    import uvicorn
    args = parse_args()
    os.chdir(HERE)
    S, app = build(args)
    print(f"Szerver: http://{args.host}:{args.port}   mód: {'SZIMULÁCIÓ' if args.sim else 'ÉLES'}", flush=True)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="warning",
                                           timeout_graceful_shutdown=2))
    quit_requested = []

    def quit_now():
        quit_requested.append(True)
        threading.Timer(0.8, lambda: setattr(server, "should_exit", True)).start()

    S.quit_hook = quit_now
    server.run()
    if quit_requested:
        print("[power] a program leállt (a weboldalról kérték)", flush=True)
        raise SystemExit(QUIT_EXIT_CODE)


if __name__ == "__main__":
    main()
