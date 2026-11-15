import re
import sqlite3
import threading
import time
from pathlib import Path

DEFAULT_NAME = re.compile(r"^j[aá]t[eé]kos\s*[1-4]$", re.IGNORECASE)

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY,
    uid TEXT NOT NULL UNIQUE,
    started REAL,
    saved REAL NOT NULL,
    status TEXT NOT NULL,                 -- finished | abandoned
    start_score INTEGER NOT NULL,
    double_out INTEGER NOT NULL,
    double_in INTEGER NOT NULL,
    winner_id INTEGER REFERENCES players(id)
);
CREATE TABLE IF NOT EXISTS game_players (
    game_id INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    seat INTEGER NOT NULL,
    player_id INTEGER REFERENCES players(id),     -- NULL: alapértelmezett nevű vendég
    name TEXT NOT NULL,
    remaining INTEGER NOT NULL,
    darts INTEGER NOT NULL,
    points INTEGER NOT NULL,
    PRIMARY KEY (game_id, seat)
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    game_id INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    seat INTEGER NOT NULL,
    player_id INTEGER REFERENCES players(id),
    total INTEGER NOT NULL,
    bust INTEGER NOT NULL,
    start_rem INTEGER NOT NULL,
    end_rem INTEGER NOT NULL,
    won INTEGER NOT NULL,
    n_darts INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS darts (
    id INTEGER PRIMARY KEY,
    game_id INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    turn_id INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    player_id INTEGER REFERENCES players(id),
    seq INTEGER NOT NULL,
    label TEXT NOT NULL,
    points INTEGER NOT NULL,
    effective INTEGER NOT NULL,
    uncertain INTEGER NOT NULL,
    src TEXT,
    x REAL,
    y REAL
);
CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY,
    t REAL NOT NULL,
    session TEXT NOT NULL DEFAULT '',
    condition TEXT NOT NULL DEFAULT '',
    predicted TEXT,                      -- amit a rendszer felismert (NULL: a rendszer nem érzékelte)
    uncertain INTEGER,
    cams INTEGER,
    spread REAL,
    conf REAL,
    margin REAL,
    x REAL,
    y REAL,
    latency_ms REAL,
    true_label TEXT,                     -- a valódi mező (NULL: szellem)
    outcome TEXT                         -- NULL = még nincs megerősítve | correct | wrong | ghost | missed
);
CREATE INDEX IF NOT EXISTS idx_meas_outcome ON measurements(outcome);
CREATE INDEX IF NOT EXISTS idx_turns_player ON turns(player_id);
CREATE INDEX IF NOT EXISTS idx_darts_player ON darts(player_id);
CREATE INDEX IF NOT EXISTS idx_gp_player ON game_players(player_id);
"""


def is_registered(name):
    name = (name or "").strip()
    return bool(name) and not DEFAULT_NAME.match(name)


def wilson(k, n, z=1.96):
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(100 * max(0.0, c - h), 1), round(100 * min(1.0, c + h), 1)]


def field_class(label):
    if not label:
        return None
    if label in ("25", "50"):
        return "bull"
    return {"S": "egyes", "D": "dupla", "T": "tripla"}.get(label[0])


def _pct(k, n):
    return round(100.0 * k / n, 1) if n else None


def _quantile(vals, q):
    v = sorted(vals)
    if not v:
        return None
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return round(v[i])


class MeasureMixin:
    OUTCOMES = ("correct", "wrong", "ghost", "missed")

    def m_add_pending(self, e, session="", condition=""):
        with self.lock, self.db:
            return self.db.execute(
                "INSERT INTO measurements(t, session, condition, predicted, uncertain, cams, spread, conf, margin, x, y, latency_ms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), session, condition, e.get("label"), int(bool(e.get("uncertain"))), e.get("cams"),
                 e.get("spread_mm"), e.get("conf"), e.get("margin_mm"), e.get("x"), e.get("y"),
                 e.get("latency_ms"))).lastrowid

    def m_label(self, mid, outcome, true_label=None):
        if outcome not in ("correct", "wrong", "ghost"):
            raise ValueError("ismeretlen eredmény")
        with self.lock, self.db:
            row = self.db.execute("SELECT predicted FROM measurements WHERE id = ?", (int(mid),)).fetchone()
            if not row:
                raise ValueError("nincs ilyen mérés")
            if outcome == "correct":
                true_label = row["predicted"]
            elif outcome == "ghost":
                true_label = None
            elif not true_label:
                raise ValueError("add meg a valódi mezőt")
            elif true_label == row["predicted"]:
                outcome = "correct"
            self.db.execute("UPDATE measurements SET outcome = ?, true_label = ? WHERE id = ?", (outcome, true_label, int(mid)))

    def m_label_all_correct(self):
        with self.lock, self.db:
            return self.db.execute("UPDATE measurements SET outcome = 'correct', true_label = predicted "
                                   "WHERE outcome IS NULL AND predicted IS NOT NULL").rowcount

    def m_add_missed(self, true_label, session="", condition=""):
        with self.lock, self.db:
            return self.db.execute(
                "INSERT INTO measurements(t, session, condition, true_label, outcome) VALUES (?,?,?,?, 'missed')",
                (time.time(), session, condition, true_label)).lastrowid

    def m_delete(self, mid):
        with self.lock, self.db:
            return self.db.execute("DELETE FROM measurements WHERE id = ?", (int(mid),)).rowcount > 0

    def m_clear_pending(self):
        with self.lock, self.db:
            return self.db.execute("DELETE FROM measurements WHERE outcome IS NULL").rowcount

    def m_pending(self, limit=12):
        with self.lock:
            return [dict(r) for r in self.db.execute(
                "SELECT id, t, predicted, uncertain, cams, spread, conf, latency_ms FROM measurements "
                "WHERE outcome IS NULL ORDER BY id DESC LIMIT ?", (limit,))]

    def m_export(self, session=None):
        q, args = "SELECT * FROM measurements WHERE outcome IS NOT NULL", ()
        if session:
            q, args = q + " AND session = ?", (session,)
        with self.lock:
            return [dict(r) for r in self.db.execute(q + " ORDER BY id", args)]

    def m_sessions(self):
        with self.lock:
            return [r["session"] for r in self.db.execute(
                "SELECT DISTINCT session FROM measurements WHERE outcome IS NOT NULL ORDER BY session")]

    def m_summary(self, session=None):
        with self.lock:
            q = "SELECT * FROM measurements WHERE outcome IS NOT NULL"
            args = ()
            if session:
                q += " AND session = ?"
                args = (session,)
            rows = [dict(r) for r in self.db.execute(q, args)]
        return summarize(rows)


def _acc_block(rows):
    c = sum(r["outcome"] == "correct" for r in rows)
    w = sum(r["outcome"] == "wrong" for r in rows)
    m = sum(r["outcome"] == "missed" for r in rows)
    g = sum(r["outcome"] == "ghost" for r in rows)
    n = c + w + m
    return {"n": n, "correct": c, "wrong": w, "missed": m, "ghost": g,
            "accuracy": _pct(c, n), "accuracy_ci": wilson(c, n),
            "detection": _pct(c + w, n),
            "precision": _pct(c, c + w + g)}


def summarize(rows):
    out = {"overall": _acc_block(rows)}
    pred = [r for r in rows if r["outcome"] in ("correct", "wrong")]

    fl = [r for r in pred if r["uncertain"]]
    nf = [r for r in pred if not r["uncertain"]]
    out["flag"] = {"flagged": len(fl), "flagged_wrong": sum(r["outcome"] == "wrong" for r in fl),
                   "unflagged": len(nf), "unflagged_wrong": sum(r["outcome"] == "wrong" for r in nf)}
    out["by_condition"] = {k: _acc_block([r for r in rows if r["condition"] == k])
                           for k in sorted({r["condition"] for r in rows})}
    out["by_cams"] = {str(k): _acc_block([r for r in pred if r["cams"] == k]) for k in sorted({r["cams"] for r in pred if r["cams"]})}
    classes = {}
    for r in rows:
        k = field_class(r["true_label"])
        if k:
            classes.setdefault(k, []).append(r)
    out["by_class"] = {k: _acc_block(v) for k, v in sorted(classes.items())}
    lat = [r["latency_ms"] for r in pred if r["latency_ms"] is not None]
    out["latency"] = {"n": len(lat), "mean": round(sum(lat) / len(lat)) if lat else None,
                      "median": _quantile(lat, 0.5), "p95": _quantile(lat, 0.95), "max": round(max(lat)) if lat else None}
    errs = {}
    for r in rows:
        if r["outcome"] == "wrong":
            errs[(r["predicted"], r["true_label"])] = errs.get((r["predicted"], r["true_label"]), 0) + 1
    out["errors"] = [{"predicted": a, "true": b, "n": n} for (a, b), n in sorted(errs.items(), key=lambda kv: -kv[1])[:10]]
    sp_c = [r["spread"] for r in pred if r["outcome"] == "correct" and r["spread"] is not None]
    sp_w = [r["spread"] for r in pred if r["outcome"] == "wrong" and r["spread"] is not None]
    out["spread"] = {"correct_mean": round(sum(sp_c) / len(sp_c), 2) if sp_c else None,
                     "wrong_mean": round(sum(sp_w) / len(sp_w), 2) if sp_w else None}
    return out


class StatsDB(MeasureMixin):
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        try:
            self.db.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:
            pass
        with self.lock, self.db:
            self.db.executescript(SCHEMA)

    def close(self):
        with self.lock:
            self.db.close()

    def _player_id(self, name):
        row = self.db.execute("SELECT id FROM players WHERE name = ?", (name,)).fetchone()
        if row:
            return row["id"]
        return self.db.execute("INSERT INTO players(name, created) VALUES (?, ?)", (name, time.time())).lastrowid

    def save_match(self, game):
        base = game.uid + "-L"
        with self.lock, self.db:
            self.db.execute("DELETE FROM games WHERE uid LIKE ? AND status = 'finished'", (base + "%",))
        for leg in range(len(game.leg_winners) + (1 if game.over else 0)):
            self.save_game(game, "finished", leg=leg)

    def save_game(self, game, status, leg=None):
        names = game.players
        if not any(is_registered(n) for n in names):
            return None
        turns = list(game.turns_log)
        part = game.partial_turn()
        if part:
            turns.append(part)
        uid, started, winner_idx = game.uid, game.started, game.winner
        if leg is not None or getattr(game, "legs", 1) > 1:

            if leg is None:
                leg = game.leg_index
            turns = [t for t in turns if t.get("leg", 0) == leg]
            uid = f"{game.uid}-L{leg}"
            if leg < len(game.leg_winners):
                winner_idx = game.leg_winners[leg]
            elif game.over and leg == game.leg_index:
                winner_idx = game.winner
            else:
                winner_idx = None
        if not any(t["darts"] for t in turns):
            return None
        if leg is None:
            st = game.state()
            leg_rem = [p["remaining"] for p in st["players"]]
            leg_darts = list(game.darts_thrown)
            leg_pts = list(game.points_scored)
        else:
            leg_rem = [game.start] * len(names)
            leg_darts = [0] * len(names)
            leg_pts = [0] * len(names)
            for t in turns:
                if t is not part:
                    leg_darts[t["player"]] += len(t["darts"])
                    if not t["bust"]:
                        leg_pts[t["player"]] += sum(d["effective"] for d in t["darts"])
                leg_rem[t["player"]] = t["end"]
        with self.lock, self.db:
            self.db.execute("DELETE FROM games WHERE uid = ?", (uid,))
            pid = [self._player_id(n.strip()) if is_registered(n) else None for n in names]
            winner = pid[winner_idx] if winner_idx is not None else None
            gid = self.db.execute(
                "INSERT INTO games(uid, started, saved, status, start_score, double_out, double_in, winner_id) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (uid, started, time.time(), status, game.start, int(game.double_out),
                 int(game.double_in), winner)).lastrowid
            for seat, name in enumerate(names):
                self.db.execute("INSERT INTO game_players VALUES (?,?,?,?,?,?,?)",
                                (gid, seat, pid[seat], name, leg_rem[seat], leg_darts[seat], leg_pts[seat]))
            for seq, t in enumerate(turns):
                seat = t["player"]
                tid = self.db.execute(
                    "INSERT INTO turns(game_id, seq, seat, player_id, total, bust, start_rem, end_rem, won, n_darts) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (gid, seq, seat, pid[seat], t["total"], int(t["bust"]), t["start"], t["end"],
                     int(t["won"]), len(t["darts"]))).lastrowid
                for k, d in enumerate(t["darts"]):
                    h = game.history[d["index"]] if 0 <= d.get("index", -1) < len(game.history) else {}
                    self.db.execute(
                        "INSERT INTO darts(game_id, turn_id, player_id, seq, label, points, effective, uncertain, src, x, y) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (gid, tid, pid[seat], k, d["label"], d["points"], d["effective"], int(d["uncertain"]),
                         h.get("src"), h.get("x"), h.get("y")))
            return gid

    def delete_finished(self, uid):
        with self.lock, self.db:
            self.db.execute("DELETE FROM games WHERE uid = ? AND status = 'finished'", (uid,))

    def delete_game(self, game_id):
        with self.lock, self.db:
            return self.db.execute("DELETE FROM games WHERE id = ?", (int(game_id),)).rowcount > 0

    def delete_player(self, name):
        with self.lock, self.db:
            row = self.db.execute("SELECT id FROM players WHERE name = ?", (name,)).fetchone()
            if not row:
                return False
            pid = row["id"]
            self.db.execute("DELETE FROM games WHERE id IN (SELECT game_id FROM game_players WHERE player_id = ?)", (pid,))
            self.db.execute("DELETE FROM players WHERE id = ?", (pid,))
            return True

    def _summary(self, pid):
        q = self.db.execute
        games = q("SELECT COUNT(*) c FROM game_players gp JOIN games g ON g.id = gp.game_id "
                  "WHERE gp.player_id = ? AND g.status = 'finished'", (pid,)).fetchone()["c"]
        wins = q("SELECT COUNT(*) c FROM games WHERE winner_id = ?", (pid,)).fetchone()["c"]
        abandoned = q("SELECT COUNT(*) c FROM game_players gp JOIN games g ON g.id = gp.game_id "
                      "WHERE gp.player_id = ? AND g.status = 'abandoned'", (pid,)).fetchone()["c"]
        t = q("SELECT COALESCE(SUM(total),0) pts, COALESCE(SUM(n_darts),0) darts, COALESCE(MAX(total),0) best, "
              "SUM(total = 180) n180, SUM(total >= 140 AND total < 180) n140, SUM(total >= 100 AND total < 140) n100, "
              "SUM(bust) busts, COUNT(*) turns FROM turns WHERE player_id = ?", (pid,)).fetchone()
        co = q("SELECT MAX(start_rem) c FROM turns WHERE player_id = ? AND won = 1", (pid,)).fetchone()["c"]
        leg = q("SELECT MIN(gp.darts) c FROM game_players gp JOIN games g ON g.id = gp.game_id "
                "WHERE gp.player_id = ? AND g.winner_id = ? AND g.start_score = 501", (pid, pid)).fetchone()["c"]
        darts = t["darts"]
        return {"games": games, "wins": wins, "abandoned": abandoned,
                "win_pct": round(100.0 * wins / games, 1) if games else None,
                "darts": darts, "avg": round(3.0 * t["pts"] / darts, 1) if darts else None,
                "best_turn": t["best"], "n180": t["n180"] or 0, "n140": t["n140"] or 0, "n100": t["n100"] or 0,
                "busts": t["busts"] or 0, "turns": t["turns"], "best_checkout": co, "best_leg_501": leg}

    def players(self):
        with self.lock:
            rows = self.db.execute("SELECT id, name FROM players ORDER BY name COLLATE NOCASE").fetchall()
            out = [{"name": r["name"], **self._summary(r["id"])} for r in rows]
        out.sort(key=lambda p: (-(p["games"] + p["abandoned"]), p["name"].lower()))
        return out

    def player(self, name, recent=15, points=400):
        with self.lock:
            row = self.db.execute("SELECT id, name FROM players WHERE name = ?", (name,)).fetchone()
            if not row:
                return None
            pid = row["id"]
            q = self.db.execute
            out = {"name": row["name"], **self._summary(pid)}
            out["fields"] = [dict(r) for r in q(
                "SELECT label, COUNT(*) n FROM darts WHERE player_id = ? GROUP BY label ORDER BY n DESC, label LIMIT 8", (pid,))]
            out["misses"] = q("SELECT COUNT(*) c FROM darts WHERE player_id = ? AND label = 'MISS'", (pid,)).fetchone()["c"]
            out["points"] = [[r["x"], r["y"]] for r in q(
                "SELECT x, y FROM darts WHERE player_id = ? AND x IS NOT NULL AND y IS NOT NULL ORDER BY id DESC LIMIT ?",
                (pid, points))]
            games = []
            for g in q("SELECT g.id, g.started, g.saved, g.status, g.start_score, g.winner_id, gp.darts, gp.points, gp.remaining "
                       "FROM game_players gp JOIN games g ON g.id = gp.game_id WHERE gp.player_id = ? "
                       "ORDER BY g.saved DESC LIMIT ?", (pid, recent)).fetchall():
                opp = [r["name"] for r in q("SELECT name FROM game_players WHERE game_id = ? AND (player_id IS NOT ? ) ORDER BY seat",
                                            (g["id"], pid))]
                games.append({"id": g["id"], "time": g["started"] or g["saved"], "status": g["status"],
                              "start": g["start_score"], "won": g["winner_id"] == pid,
                              "darts": g["darts"], "avg": round(3.0 * g["points"] / g["darts"], 1) if g["darts"] else None,
                              "remaining": g["remaining"], "opponents": opp})
            out["games_recent"] = games
            out["avg_history"] = [a for a in (g["avg"] for g in reversed(games)) if a is not None]
            return out

    def recent_games(self, limit=20):
        with self.lock:
            rows = self.db.execute("SELECT id, started, saved, status, start_score, winner_id FROM games "
                                   "ORDER BY saved DESC LIMIT ?", (limit,)).fetchall()
            out = []
            for g in rows:
                pl = self.db.execute("SELECT name, player_id, remaining, darts FROM game_players WHERE game_id = ? ORDER BY seat",
                                     (g["id"],)).fetchall()
                win = next((p["name"] for p in pl if p["player_id"] is not None and p["player_id"] == g["winner_id"]), None)
                out.append({"id": g["id"], "time": g["started"] or g["saved"], "status": g["status"], "start": g["start_score"],
                            "winner": win, "players": [{"name": p["name"], "remaining": p["remaining"], "darts": p["darts"]} for p in pl]})
            return out
