import tempfile
import unittest
from pathlib import Path

from game import X01Game
from stats import StatsDB, is_registered


def play(game, labels):
    for lab in labels:
        game.add_dart(lab, False, "camera", 1.0, 2.0)


class StatsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = StatsDB(Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_default_names_not_registered(self):
        self.assertFalse(is_registered("Játékos 1"))
        self.assertFalse(is_registered("jatekos 3"))
        self.assertTrue(is_registered("Tomi"))
        self.assertTrue(is_registered("Játékos 12"))

    def test_guest_only_game_not_saved(self):
        g = X01Game(["Játékos 1", "Játékos 2"], 101, True, False)
        play(g, ["T20", "T20", "S1"])
        self.assertIsNone(self.db.save_game(g, "abandoned"))
        self.assertEqual(self.db.players(), [])

    def test_finished_game_stats(self):
        g = X01Game(["Tomi", "Játékos 2"], 101, True, False)

        play(g, ["T20", "S1", "S1", "S1", "S1", "S1", "S7", "D16"])
        self.assertTrue(g.over)
        self.assertIsNotNone(self.db.save_game(g, "finished"))
        p = self.db.player("Tomi")
        self.assertEqual((p["games"], p["wins"], p["win_pct"]), (1, 1, 100.0))
        self.assertEqual((p["darts"], p["best_turn"], p["best_checkout"]), (5, 62, 39))
        self.assertEqual(p["avg"], round(3 * 101 / 5, 1))
        self.assertEqual(self.db.players()[0]["name"], "Tomi")
        self.assertEqual(len(self.db.players()), 1)
        self.assertEqual(self.db.recent_games()[0]["winner"], "Tomi")
        self.assertEqual(p["fields"][0]["label"], "S1")
        self.assertEqual(len(p["points"]), 5)

    def test_name_is_case_insensitive(self):
        g1 = X01Game(["Tomi"], 40, True, False); play(g1, ["D20"])
        g2 = X01Game(["tomi"], 40, True, False); play(g2, ["D20"])
        self.db.save_game(g1, "finished"); self.db.save_game(g2, "finished")
        self.assertEqual(len(self.db.players()), 1)
        self.assertEqual(self.db.players()[0]["games"], 2)

    def test_replace_same_uid_and_delete_on_undo(self):
        g = X01Game(["Tomi"], 40, True, False)
        play(g, ["D20"])
        self.assertTrue(g.over)
        self.db.save_game(g, "finished")
        self.db.save_game(g, "finished")
        self.assertEqual(self.db.player("Tomi")["games"], 1)
        g.undo()
        self.db.delete_finished(g.uid)
        self.assertEqual(self.db.player("Tomi")["games"], 0)

    def test_abandoned_has_partial_turn(self):
        g = X01Game(["Tomi", "Anna"], 501, True, False)
        play(g, ["T20", "T20"])
        self.db.save_game(g, "abandoned")
        p = self.db.player("Tomi")
        self.assertEqual((p["games"], p["abandoned"], p["darts"]), (0, 1, 2))
        self.assertEqual(p["avg"], 180.0)

    def test_180_and_busts(self):
        g = X01Game(["Tomi"], 501, True, False)
        play(g, ["T20", "T20", "T20"])
        play(g, ["T20", "T20", "T20"])
        play(g, ["T20", "T20", "T20"])
        self.db.save_game(g, "abandoned")
        p = self.db.player("Tomi")
        self.assertEqual((p["n180"], p["busts"], p["best_turn"]), (2, 1, 180))

    def test_delete_player_removes_games(self):
        g = X01Game(["Tomi"], 40, True, False)
        play(g, ["D20"])
        self.db.save_game(g, "finished")
        self.assertTrue(self.db.delete_player("Tomi"))
        self.assertEqual(self.db.recent_games(), [])
        self.assertEqual(self.db.players(), [])


class ServerStatsTest(unittest.TestCase):
    def test_game_flow_through_server(self):
        import server
        from starlette.testclient import TestClient
        with tempfile.TemporaryDirectory() as tmp:
            S, app = server.build(server.parse_args(["--sim", "--data-dir", tmp]))
            with TestClient(app) as c:
                post = lambda path, **kw: c.post(path, json=kw)
                post("/api/new", players=["Tomi", "Játékos 2"], start=40, double_out=True, double_in=False)
                post("/api/sim/throw", label="D20")
                self.assertTrue(c.get("/api/state").json()["game"]["winner_name"] == "Tomi")
                r = c.get("/api/stats/players").json()
                self.assertEqual([p["name"] for p in r["players"]], ["Tomi"])
                self.assertEqual(r["players"][0]["wins"], 1)

                post("/api/undo")
                self.assertEqual(c.get("/api/stats/players").json()["players"][0]["games"], 0)

                post("/api/add", label="S20")
                post("/api/new", players=["Tomi"], start=501, double_out=True, double_in=False)
                p = c.get("/api/stats/player", params={"name": "Tomi"}).json()
                self.assertEqual((p["abandoned"], p["darts"]), (1, 1))
                self.assertEqual(c.get("/api/stats/player", params={"name": "Nincs"}).status_code, 404)
                gid = c.get("/api/stats/players").json()["games"][0]["id"]
                self.assertTrue(post("/api/stats/delete_game", id=gid).json()["ok"])
                self.assertTrue(post("/api/stats/delete_player", name="Tomi").json()["ok"])
                self.assertEqual(c.get("/api/stats/players").json()["players"], [])
            S.stats.close()


class MeasureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = StatsDB(Path(self.tmp.name) / "m.db")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def throw(self, label, **kw):
        e = {"label": label, "uncertain": False, "cams": 3, "spread_mm": 1.0, "conf": 0.9, "margin_mm": 5.0,
             "x": 1.0, "y": 2.0, "latency_ms": 1000}
        e.update(kw)
        return self.db.m_add_pending(e, "s1", "LED")

    def test_summary_numbers(self):
        for _ in range(8):
            self.db.m_label(self.throw("T20"), "correct")
        self.db.m_label(self.throw("S20", uncertain=True, cams=2, latency_ms=2000), "wrong", "S5")
        self.db.m_label(self.throw("D16"), "ghost")
        self.db.m_add_missed("T19", "s1", "LED")
        sm = self.db.m_summary()
        o = sm["overall"]
        self.assertEqual((o["n"], o["correct"], o["wrong"], o["missed"], o["ghost"]), (10, 8, 1, 1, 1))
        self.assertEqual(o["accuracy"], 80.0)
        self.assertEqual(o["detection"], 90.0)
        self.assertEqual(o["precision"], 80.0)
        lo, hi = o["accuracy_ci"]
        self.assertTrue(lo < 80.0 < hi and 0 <= lo and hi <= 100)
        self.assertEqual(sm["flag"], {"flagged": 1, "flagged_wrong": 1, "unflagged": 8, "unflagged_wrong": 0})
        self.assertEqual(sm["by_cams"]["2"]["n"], 1)
        self.assertEqual(sm["by_condition"]["LED"]["n"], 10)
        self.assertEqual(sm["errors"], [{"predicted": "S20", "true": "S5", "n": 1}])
        self.assertEqual(sm["latency"]["n"], 9)
        self.assertEqual(sm["latency"]["max"], 2000)
        self.assertEqual(sm["by_class"]["tripla"]["n"], 9)

    def test_pending_rules(self):
        a = self.throw("T20")
        self.assertEqual(len(self.db.m_pending()), 1)
        with self.assertRaises(ValueError):
            self.db.m_label(a, "wrong")
        with self.assertRaises(ValueError):
            self.db.m_label(a, "bogus")
        self.db.m_label(a, "wrong", "T20")
        self.assertEqual(self.db.m_summary()["overall"]["correct"], 1)
        self.throw("S1"); self.throw("S2")
        self.assertEqual(self.db.m_label_all_correct(), 2)
        self.assertEqual(self.db.m_pending(), [])
        self.assertEqual(self.db.m_summary()["overall"]["n"], 3)

    def test_empty_and_session_filter(self):
        self.assertEqual(self.db.m_summary()["overall"]["n"], 0)
        self.assertIsNone(self.db.m_summary()["overall"]["accuracy"])
        e = {"label": "S1", "cams": 3}
        i = self.db.m_add_pending(e, "A", "x"); self.db.m_label(i, "correct")
        j = self.db.m_add_pending(e, "B", "x"); self.db.m_label(j, "ghost")
        self.assertEqual(self.db.m_summary("A")["overall"]["ghost"], 0)
        self.assertEqual(self.db.m_summary("B")["overall"]["ghost"], 1)
        self.assertEqual(self.db.m_sessions(), ["A", "B"])

    def test_wilson(self):
        from stats import wilson
        self.assertEqual(wilson(0, 0), None)
        lo, hi = wilson(50, 100)
        self.assertAlmostEqual(lo, 40.4, 0); self.assertAlmostEqual(hi, 59.6, 0)
        self.assertEqual(wilson(100, 100)[1], 100.0)


class ServerMeasureTest(unittest.TestCase):
    def test_flow(self):
        import server
        from starlette.testclient import TestClient
        with tempfile.TemporaryDirectory() as tmp:
            S, app = server.build(server.parse_args(["--sim", "--data-dir", tmp]))
            with TestClient(app) as c:
                post = lambda path, **kw: c.post(path, json=kw)
                post("/api/sim/throw", label="T20")
                self.assertEqual(c.get("/api/measure").json()["pending"], [])
                post("/api/measure/setup", on=True, session="teszt1", condition="LED, nappal")
                post("/api/sim/throw", label="S5"); post("/api/sim/throw", label="D16")
                m = c.get("/api/measure").json()
                self.assertEqual((m["on"], len(m["pending"])), (True, 2))
                ids = [p["id"] for p in m["pending"]]
                self.assertTrue(post("/api/measure/label", id=ids[0], outcome="correct").json()["ok"])
                self.assertEqual(post("/api/measure/label", id=ids[1], outcome="wrong").status_code, 400)
                self.assertTrue(post("/api/measure/label", id=ids[1], outcome="wrong", true_label="d17").json()["ok"])
                self.assertTrue(post("/api/measure/missed", true_label="T1").json()["ok"])
                m = c.get("/api/measure", params={"session": "teszt1"}).json()
                self.assertEqual((m["summary"]["overall"]["n"], m["summary"]["overall"]["correct"]), (3, 1))
                self.assertEqual(m["summary"]["by_condition"]["LED, nappal"]["n"], 3)
                csv = c.get("/api/measure/export.csv").text
                self.assertTrue(csv.startswith("\ufeffsorszam;id;t;session"))
                self.assertIn("D17", csv)

                post("/api/measure/setup", session="masik", condition="este")
                post("/api/sim/throw", label="S1")
                pid = c.get("/api/measure").json()["pending"][0]["id"]
                post("/api/measure/label", id=pid, outcome="correct")
                one = c.get("/api/measure/export.csv", params={"session": "masik"})
                lines = one.text.strip().splitlines()
                self.assertEqual(len(lines), 2)
                self.assertTrue(lines[0].lstrip("\ufeff").startswith("sorszam;id;"))
                self.assertTrue(lines[1].startswith("1;"))
                self.assertIn("meresek_masik.csv", one.headers["content-disposition"])
                self.assertEqual(len(c.get("/api/measure/export.csv").text.strip().splitlines()), 5)
                self.assertEqual(post("/api/measure/delete", all_pending=True).json()["n"], 0)
            S.stats.close()


class MatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = StatsDB(Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_match_legs_saved_separately(self):
        g = X01Game(["Tomi", "Anna"], 40, True, False, legs=2)
        play(g, ["D20", "D20", "D20"])
        self.db.save_match(g)
        rows = self.db.db.execute("SELECT uid, status FROM games ORDER BY uid").fetchall()
        self.assertEqual([r["status"] for r in rows], ["finished"] * 3)
        self.assertTrue(all("-L" in r["uid"] for r in rows))
        self.assertEqual(self.db.player("Tomi")["wins"], 2)

    def test_undo_removes_extra_leg(self):
        g = X01Game(["Tomi", "Anna"], 40, True, False, legs=2)
        play(g, ["D20", "D20", "D20"])
        self.db.save_match(g)
        g.undo()
        self.db.save_match(g)
        n = self.db.db.execute("SELECT COUNT(*) c FROM games").fetchone()["c"]
        self.assertEqual(n, 2)


class ServerMatchTest(unittest.TestCase):
    def test_first_to_two_through_server(self):
        import server
        from starlette.testclient import TestClient
        with tempfile.TemporaryDirectory() as tmp:
            S, app = server.build(server.parse_args(["--sim", "--data-dir", tmp]))
            with TestClient(app) as c:
                post = lambda path, **kw: c.post(path, json=kw)
                post("/api/new", players=["Tomi", "Anna"], start=40, double_out=True, double_in=False, legs=2)
                post("/api/add", label="D20")
                g = c.get("/api/state").json()["game"]
                self.assertIsNone(g["winner"])
                self.assertEqual(g["leg_event"]["name"], "Tomi")
                self.assertEqual([p["legs_won"] for p in g["players"]], [1, 0])
                post("/api/add", label="D20")
                post("/api/add", label="D20")
                g = c.get("/api/state").json()["game"]
                self.assertEqual(g["winner_name"], "Tomi")
                wins = lambda: {p["name"]: p["wins"] for p in c.get("/api/stats/players").json()["players"]}
                self.assertEqual(wins(), {"Tomi": 2, "Anna": 1})
                post("/api/undo")
                self.assertEqual(wins(), {"Tomi": 1, "Anna": 1})
            S.stats.close()


if __name__ == "__main__":
    unittest.main()
