import unittest

from tracker import DartTracker

T20 = (0.0, -103.0)
S5 = (-41.6, -127.9)


def throws(events):
    return [e for e in events if e["type"] == "throw"]


class DefaultBehaviour(unittest.TestCase):
    def test_needs_two_evaluations(self):
        t = DartTracker()
        self.assertEqual(throws(t.update([(T20, 3, 2.0, 0.9)])), [])
        ev = throws(t.update([(T20, 3, 2.0, 0.9)]))
        self.assertEqual([e["label"] for e in ev], ["T20"])

    def test_single_glitch_is_ignored(self):
        t = DartTracker()
        t.update([(T20, 3, 2.0, 0.9)])
        self.assertEqual(throws(t.update([])), [])
        self.assertEqual(t.snapshot(), [])


class FastConfirm(unittest.TestCase):
    def test_strong_detection_confirmed_at_once(self):
        t = DartTracker(fast_conf=0.75)
        ev = throws(t.update([(T20, 3, 2.0, 0.9)]))
        self.assertEqual([e["label"] for e in ev], ["T20"])
        self.assertEqual(len(t.snapshot()), 1)

    def test_not_counted_twice(self):
        t = DartTracker(fast_conf=0.75)
        t.update([(T20, 3, 2.0, 0.9)])
        self.assertEqual(throws(t.update([(T20, 3, 2.0, 0.9)])), [])
        self.assertEqual(len(t.snapshot()), 1)

    def test_weak_or_few_cameras_still_wait(self):
        for found in ([(T20, 3, 2.0, 0.5)],
                      [(T20, 2, 2.0, 0.95)],
                      [(T20, 3, 12.0, 0.95)]):
            t = DartTracker(fast_conf=0.75)
            self.assertEqual(throws(t.update(found)), [], found)
            self.assertEqual(len(throws(t.update(found))), 1, found)

    def test_second_dart_next_to_first(self):
        t = DartTracker(fast_conf=0.75)
        t.update([(T20, 3, 2.0, 0.9)])
        ev = throws(t.update([(T20, 3, 2.0, 0.9), (S5, 3, 3.0, 0.88)]))
        self.assertEqual([e["label"] for e in ev], ["S5"])

    def test_ghost_near_known_dart_still_blocked(self):
        t = DartTracker(fast_conf=0.4, near_conf=0.5)
        t.update([(T20, 3, 2.0, 0.9)])
        ghost = (T20[0] + 12.0, T20[1] + 5.0)
        ev = throws(t.update([(T20, 3, 2.0, 0.9), (ghost, 3, 2.0, 0.45)]))
        self.assertEqual(ev, [])

    def test_removal_still_needs_two_evaluations(self):
        t = DartTracker(fast_conf=0.75)
        t.update([(T20, 3, 2.0, 0.9)])
        self.assertEqual(t.update([]), [])
        ev = t.update([])
        self.assertEqual([e["type"] for e in ev], ["removed"])


if __name__ == "__main__":
    unittest.main()
