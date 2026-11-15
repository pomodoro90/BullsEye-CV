import unittest

import dartboard as db
from game import X01Game, checkout_hint, parse_label


def throw(g, *labels):
    for lab in labels:
        assert g.add_dart(lab)


class ParseTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(parse_label("T20"), (3, 20, 60))
        self.assertEqual(parse_label("d6"), (2, 6, 12))
        self.assertEqual(parse_label("S5"), (1, 5, 5))
        self.assertEqual(parse_label("25"), (1, 25, 25))
        self.assertEqual(parse_label("50"), (2, 25, 50))
        self.assertEqual(parse_label("MISS"), (0, 0, 0))

    def test_invalid(self):
        for bad in ("T21", "X5", "", "S0", "D", "TT", "75"):
            with self.assertRaises(ValueError):
                parse_label(bad)

    def test_label_to_xy_roundtrip(self):
        for lab in [f"{m}{n}" for m in "SDT" for n in range(1, 21)] + ["25", "50"]:
            xy = db.label_to_xy(lab)
            self.assertEqual(db.score_from_xy(*xy)[0], lab)
        self.assertIsNone(db.label_to_xy("MISS"))


class PlayTests(unittest.TestCase):
    def test_basic_turn_and_switch(self):
        g = X01Game(["A", "B"], 501)
        throw(g, "T20", "T20", "T20")
        s = g.state()
        self.assertEqual(s["players"][0]["remaining"], 321)
        self.assertEqual(s["current"], 1)
        self.assertTrue(s["turn_ended"])
        self.assertEqual(s["last_turn"]["total"], 180)
        throw(g, "S1")
        self.assertEqual(g.state()["players"][1]["remaining"], 500)
        self.assertFalse(g.state()["turn_ended"])
        self.assertEqual(g.state()["darts_left"], 2)

    def test_bust_resets_to_turn_start(self):
        g = X01Game(["A", "B"], 40)
        throw(g, "S20", "S19")
        s = g.state()
        self.assertEqual(s["players"][0]["remaining"], 40)
        self.assertEqual(s["current"], 1)
        self.assertTrue(s["last_turn"]["bust"])

    def test_bust_below_zero(self):
        g = X01Game(["A", "B"], 30)
        throw(g, "T20")
        self.assertEqual(g.state()["players"][0]["remaining"], 30)
        self.assertEqual(g.state()["current"], 1)

    def test_checkout_on_double(self):
        g = X01Game(["A", "B"], 40)
        throw(g, "D20")
        s = g.state()
        self.assertEqual(s["winner"], 0)
        self.assertEqual(s["winner_name"], "A")
        self.assertEqual(s["players"][0]["remaining"], 0)
        self.assertFalse(g.add_dart("S1"))

    def test_single_to_zero_is_bust_with_double_out(self):
        g = X01Game(["A", "B"], 20)
        throw(g, "S20")
        self.assertIsNone(g.state()["winner"])
        self.assertEqual(g.state()["players"][0]["remaining"], 20)

    def test_single_to_zero_wins_with_straight_out(self):
        g = X01Game(["A"], 20, double_out=False)
        throw(g, "S20")
        self.assertEqual(g.state()["winner"], 0)

    def test_bull_counts_as_double(self):
        g = X01Game(["A"], 50)
        throw(g, "50")
        self.assertEqual(g.state()["winner"], 0)
        g = X01Game(["A"], 25)
        throw(g, "25")
        self.assertIsNone(g.state()["winner"])

    def test_one_left_is_not_bust_with_straight_out(self):
        g = X01Game(["A", "B"], 21, double_out=False)
        throw(g, "S20")
        self.assertEqual(g.state()["players"][0]["remaining"], 1)
        self.assertEqual(g.state()["darts_left"], 2)

    def test_double_in(self):
        g = X01Game(["A"], 100, double_in=True)
        throw(g, "T20")
        self.assertEqual(g.state()["players"][0]["remaining"], 100)
        throw(g, "D10")
        self.assertEqual(g.state()["players"][0]["remaining"], 80)
        throw(g, "S20")
        self.assertEqual(g.state()["players"][0]["remaining"], 60)

    def test_end_turn_early_and_miss(self):
        g = X01Game(["A", "B"], 501)
        throw(g, "S20", "MISS")
        g.end_turn()
        s = g.state()
        self.assertEqual(s["current"], 1)
        self.assertEqual(s["players"][0]["remaining"], 481)
        self.assertEqual(s["players"][0]["darts"], 2)

    def test_undo(self):
        g = X01Game(["A", "B"], 501)
        throw(g, "T20", "T20", "T20", "S5")
        g.undo()
        self.assertEqual(g.state()["current"], 1)
        g.undo()
        s = g.state()
        self.assertEqual(s["current"], 0)
        self.assertEqual(s["players"][0]["remaining"], 381)
        self.assertEqual(s["darts_left"], 1)
        while g.undo():
            pass
        self.assertEqual(g.state()["players"][0]["remaining"], 501)
        self.assertFalse(g.state()["can_undo"])

    def test_undo_winning_dart(self):
        g = X01Game(["A"], 40)
        throw(g, "D20")
        g.undo()
        self.assertIsNone(g.state()["winner"])
        self.assertEqual(g.state()["players"][0]["remaining"], 40)

    def test_edit_dart_recomputes_everything(self):
        g = X01Game(["A", "B"], 501)
        throw(g, "T20", "T20", "T20", "S1")
        g.edit(1, "S20")
        s = g.state()
        self.assertEqual(s["last_turn"]["total"], 140)
        self.assertEqual(s["players"][0]["remaining"], 361)
        self.assertEqual(s["players"][1]["remaining"], 500)
        with self.assertRaises(ValueError):
            g.edit(99, "S1")
        g.end_turn()
        with self.assertRaises(ValueError):
            g.edit(len(g.history) - 1, "S1")

    def test_edit_clears_uncertain(self):
        g = X01Game(["A"], 501)
        g.add_dart("S20", uncertain=True, src="camera")
        self.assertTrue(g.state()["turn"][0]["uncertain"])
        g.edit(0, "T20")
        self.assertFalse(g.state()["turn"][0]["uncertain"])

    def test_edit_can_change_winner(self):
        g = X01Game(["A", "B"], 40)
        throw(g, "S20", "S10", "S10")
        throw(g, "D20")
        self.assertEqual(g.state()["winner"], 1)
        g.edit(0, "D20")
        self.assertEqual(g.state()["winner"], 0)

    def test_average(self):
        g = X01Game(["A", "B"], 501)
        throw(g, "T20", "T20", "T20")
        throw(g, "S1", "S1", "S1")
        self.assertEqual(g.state()["players"][0]["avg"], 180.0)
        self.assertEqual(g.state()["players"][1]["avg"], 3.0)

    def test_player_count_and_start_validation(self):
        with self.assertRaises(ValueError):
            X01Game([], 501)
        with self.assertRaises(ValueError):
            X01Game(["a", "b", "c", "d", "e"], 501)
        with self.assertRaises(ValueError):
            X01Game(["a"], 5000)

    def test_save_and_load(self):
        g = X01Game(["A", "B"], 301, double_in=True)
        throw(g, "D20", "T20", "S5", "S1")
        g.end_turn()
        h = X01Game.from_json(g.to_json())
        self.assertEqual(h.state(), g.state())


class CheckoutTests(unittest.TestCase):
    def test_known(self):
        self.assertEqual(checkout_hint(170), ["T20", "T20", "50"])
        self.assertEqual(checkout_hint(40), ["D20"])
        self.assertEqual(checkout_hint(50), ["50"])
        self.assertEqual(checkout_hint(2), ["D1"])

    def test_none_cases(self):
        self.assertIsNone(checkout_hint(171))
        self.assertIsNone(checkout_hint(169))
        self.assertIsNone(checkout_hint(41, 1))
        self.assertIsNone(checkout_hint(1))
        self.assertIsNone(checkout_hint(50, 0))

    def test_valid_for_all(self):
        for rem in range(2, 171):
            h = checkout_hint(rem, 3, True)
            if h is None:
                self.assertIn(rem, (159, 162, 163, 165, 166, 168, 169), rem)
                continue
            total, left = 0, rem
            for k, lab in enumerate(h):
                m, _, p = parse_label(lab)
                total += p
                left -= p
                if k < len(h) - 1:
                    self.assertGreaterEqual(left, 2)
                else:
                    self.assertEqual(m, 2)
            self.assertEqual(total, rem)

    def test_state_has_hint(self):
        g = X01Game(["A"], 100)
        self.assertEqual(g.state()["checkout"], ["T20", "D20"])
        throw(g, "S20", "S20")
        self.assertEqual(g.state()["checkout"], checkout_hint(60, 1))
        self.assertIsNone(g.state()["checkout"])


class CheckoutOptionsTest(unittest.TestCase):
    def test_two_options_distinct_finish(self):
        from game import checkout_options
        o = checkout_options(100, 3, True)
        self.assertEqual(o[0], ["T20", "D20"])
        self.assertEqual(len(o), 2)
        self.assertNotEqual(o[0][-1], o[1][-1])

    def test_single_and_none(self):
        from game import checkout_options, checkout_hint
        self.assertEqual(checkout_options(170, 3, True), [["T20", "T20", "50"]])
        self.assertEqual(checkout_options(171, 3, True), [])
        self.assertEqual(checkout_hint(40), ["D20"])

    def test_state_has_options(self):
        g = X01Game(["A"], 100, True, False)
        g.add_dart("S20")
        st = g.state()
        self.assertEqual(st["checkout"], st["checkouts"][0])


class TestLegs(unittest.TestCase):
    def test_first_to_two(self):
        g = X01Game(["A", "B"], 40, True, False, legs=2)
        g.add_dart("D20")
        s = g.state()
        self.assertIsNone(g.winner)
        self.assertEqual([p["legs_won"] for p in s["players"]], [1, 0])
        self.assertEqual(s["leg_event"]["name"], "A")
        self.assertEqual([p["remaining"] for p in s["players"]], [40, 40])
        self.assertEqual(g.cur, 1)
        g.add_dart("D20")
        self.assertIsNone(g.winner)
        self.assertEqual(g.cur, 0)
        g.add_dart("D20")
        self.assertEqual(g.winner, 0)
        self.assertEqual(g.leg_winners, [0, 1])
        self.assertEqual([t["leg"] for t in g.turns_log], [0, 1, 2])

    def test_undo_across_legs(self):
        g = X01Game(["A", "B"], 40, True, False, legs=3)
        g.add_dart("D20")
        g.add_dart("S20")
        g.undo(); g.undo()
        s = g.state()
        self.assertEqual(s["leg_number"], 1)
        self.assertEqual([p["legs_won"] for p in s["players"]], [0, 0])
        self.assertEqual(g.cur, 0)

    def test_save_load_keeps_legs(self):
        g = X01Game(["A"], 40, True, False, legs=4)
        g.add_dart("D20")
        g2 = X01Game.from_json(g.to_json())
        self.assertEqual(g2.legs, 4)
        self.assertEqual(g2.legs_won, [1])

    def test_default_single_game(self):
        g = X01Game(["A"], 40, True, False)
        g.add_dart("D20")
        self.assertEqual(g.winner, 0)
        self.assertIsNone(g.state()["leg_event"])

    def test_bad_legs(self):
        with self.assertRaises(ValueError):
            X01Game(["A"], 501, True, False, legs=0)


if __name__ == "__main__":
    unittest.main()
