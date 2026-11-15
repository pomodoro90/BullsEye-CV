import time
import uuid
from functools import lru_cache

MAX_DARTS = 3


def parse_label(label):
    s = str(label).strip().upper()
    if s == "MISS":
        return 0, 0, 0
    if s == "25":
        return 1, 25, 25
    if s == "50":
        return 2, 25, 50
    if len(s) >= 2 and s[0] in "SDT" and s[1:].isdigit():
        n = int(s[1:])
        if 1 <= n <= 20:
            m = {"S": 1, "D": 2, "T": 3}[s[0]]
            return m, n, m * n
    raise ValueError(f"Érvénytelen címke: {label!r}")


def normalize_label(label):
    parse_label(label)
    return str(label).strip().upper()


ALL_DARTS = (
    [(f"S{n}", 1, n) for n in range(1, 21)] + [("25", 1, 25)] +
    [(f"D{n}", 2, 2 * n) for n in range(1, 21)] + [("50", 2, 50)] +
    [(f"T{n}", 3, 3 * n) for n in range(1, 21)]
)
_DOUBLE_PREF = [20, 16, 8, 10, 18, 12, 4, 6, 14, 2, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]


@lru_cache(maxsize=4096)
def checkout_options(remaining, darts_left=3, double_out=True, n=2):
    if remaining < 1 or remaining > 170 or darts_left < 1:
        return []
    min_left = 2 if double_out else 1

    def final_ok(mult):
        return (mult == 2) if double_out else True

    def key(seq):
        last = seq[-1]
        pref = 99
        if last[1] == 2:
            num = 25 if last[0] == "50" else int(last[0][1:])
            pref = _DOUBLE_PREF.index(num) if num in _DOUBLE_PREF else 50
        return (len(seq), pref, -sum(p for _, _, p in seq[:-1]))

    def all_with(k):
        found = []

        def rec(rem, left, seq):
            if left == 1:
                for lab, mult, pts in ALL_DARTS:
                    if pts == rem and final_ok(mult):
                        found.append(seq + [(lab, mult, pts)])
                return
            for lab, mult, pts in ALL_DARTS:
                if rem - pts >= min_left:
                    rec(rem - pts, left - 1, seq + [(lab, mult, pts)])
        rec(remaining, k, [])
        found.sort(key=key)
        return found

    out = []
    for k in range(1, darts_left + 1):
        cands = all_with(k)
        if not cands:
            continue
        if not out:
            out.append(cands[0])
            used_last = {cands[0][-1][0]}
        for c in cands:
            if len(out) >= n:
                break
            if c[-1][0] not in used_last:
                out.append(c)
                used_last.add(c[-1][0])
        if len(out) >= n:
            break
    return [[lab for lab, _, _ in seq] for seq in out]


def checkout_hint(remaining, darts_left=3, double_out=True):
    opts = checkout_options(remaining, darts_left, double_out, 1)
    return opts[0] if opts else None


class X01Game:
    def __init__(self, players, start=501, double_out=True, double_in=False, legs=1):
        players = [str(p).strip()[:20] for p in players if str(p).strip()]
        if not 1 <= len(players) <= 4:
            raise ValueError("1-4 játékos kell.")
        if not 2 <= int(start) <= 1001:
            raise ValueError("A kezdőérték 2 és 1001 között lehet.")
        if not 1 <= int(legs) <= 25:
            raise ValueError("A megnyerendő leg-ek száma 1 és 25 között lehet.")
        self.players = players
        self.legs = int(legs)
        self.start = int(start)
        self.double_out = bool(double_out)
        self.double_in = bool(double_in)
        self.history = []
        self.uid = uuid.uuid4().hex
        self.started = time.time()
        self._replay()

    def to_json(self):
        return {"players": self.players, "start": self.start, "double_out": self.double_out,
                "double_in": self.double_in, "history": self.history, "uid": self.uid, "started": self.started,
                "legs": self.legs}

    @classmethod
    def from_json(cls, d):
        g = cls(d["players"], d["start"], d.get("double_out", True), d.get("double_in", False), d.get("legs", 1))
        for h in d.get("history", []):
            if h.get("kind") == "dart":
                parse_label(h["label"])
        g.history = list(d.get("history", []))
        g.uid = d.get("uid") or g.uid
        g.started = d.get("started") or g.started
        g._replay()
        return g

    @property
    def over(self):
        return self.winner is not None

    def add_dart(self, label, uncertain=False, src="manual", x=None, y=None):
        if self.over:
            return False
        self.history.append({"kind": "dart", "label": normalize_label(label), "uncertain": bool(uncertain),
                             "src": src, "x": x, "y": y})
        self._replay()
        return True

    def end_turn(self):
        if self.over:
            return False
        self.history.append({"kind": "end"})
        self._replay()
        return True

    def undo(self):
        if not self.history:
            return False
        self.history.pop()
        self._replay()
        return True

    def edit(self, index, label):
        if not 0 <= index < len(self.history) or self.history[index]["kind"] != "dart":
            raise ValueError("Ez a bejegyzés nem javítható.")
        self.history[index]["label"] = normalize_label(label)
        self.history[index]["uncertain"] = False
        self.history[index]["src"] = "manual"
        self._replay()

    def _replay(self):
        n = len(self.players)
        self.rem = [self.start] * n
        self._opened = [not self.double_in] * n
        self.cur = 0
        self.turn = []
        self._turn_start = self.start
        self.winner = None
        self.last_turn = None
        self.darts_thrown = [0] * n
        self.points_scored = [0] * n
        self.turn_ended = False
        self.ignored = 0
        self.turns_log = []
        self.legs_won = [0] * n
        self.leg_index = 0
        self.leg_winners = []
        self.leg_event = None

        def finish_turn(bust, advance=True):
            p = self.cur
            self.turns_log.append({"player": p, "leg": self.leg_index, "darts": list(self.turn), "bust": bust,
                                   "total": 0 if bust else sum(d["effective"] for d in self.turn),
                                   "start": self._turn_start, "end": self.rem[p],
                                   "won": (not bust) and self.rem[p] == 0})
            self.last_turn = {"player": p, "darts": self.turn, "bust": bust,
                              "total": 0 if bust else sum(d["effective"] for d in self.turn)}
            self.darts_thrown[p] += len(self.turn)
            if not bust:
                self.points_scored[p] += sum(d["effective"] for d in self.turn)
            self.turn = []
            if advance:
                self.cur = (self.cur + 1) % n
            self._turn_start = self.rem[self.cur]
            self.turn_ended = True

        for i, h in enumerate(self.history):
            if self.winner is not None:
                self.ignored += 1
                continue
            self.turn_ended = False
            self.leg_event = None
            if h["kind"] == "end":
                finish_turn(False)
                continue
            mult, num, pts = parse_label(h["label"])
            p = self.cur
            effective = pts
            if not self._opened[p]:
                if mult == 2:
                    self._opened[p] = True
                else:
                    effective = 0
            new = self.rem[p] - effective
            dart = {"label": h["label"], "points": pts, "effective": effective,
                    "uncertain": bool(h.get("uncertain")), "index": i}
            self.turn.append(dart)
            bust = (new < 0 or (self.double_out and new == 1)
                    or (new == 0 and self.double_out and mult != 2))
            if bust:
                dart["bust"] = True
                self.rem[p] = self._turn_start
                finish_turn(True)
            elif new == 0:
                self.rem[p] = 0
                self.legs_won[p] += 1
                finish_turn(False, advance=False)
                if self.legs_won[p] >= self.legs:
                    self.winner = p
                    self.cur = p
                else:
                    self.leg_winners.append(p)
                    self.leg_event = {"winner": p, "leg": self.leg_index}
                    self.leg_index += 1
                    self.rem = [self.start] * n
                    self._opened = [not self.double_in] * n
                    self.cur = self.leg_index % n
                    self._turn_start = self.start
            else:
                self.rem[p] = new
                if len(self.turn) >= MAX_DARTS:
                    finish_turn(False)

    def partial_turn(self):
        if self.turn and not self.over:
            return {"player": self.cur, "leg": self.leg_index, "darts": list(self.turn), "bust": False,
                    "total": sum(d["effective"] for d in self.turn),
                    "start": self._turn_start, "end": self.rem[self.cur], "won": False}
        return None

    def state(self):
        n = len(self.players)
        players = []
        for p in range(n):
            thrown = self.darts_thrown[p]
            avg = round(3.0 * self.points_scored[p] / thrown, 1) if thrown else None
            players.append({"name": self.players[p], "remaining": self.rem[p], "avg": avg,
                            "darts": thrown, "current": p == self.cur and not self.over,
                            "legs_won": self.legs_won[p]})
        left = MAX_DARTS - len(self.turn)
        hint, options = None, []
        if not self.over and self._opened[self.cur]:
            options = checkout_options(self.rem[self.cur], left, self.double_out, 2)
            hint = options[0] if options else None
        last = None
        if self.last_turn:
            last = {"player": self.last_turn["player"], "name": self.players[self.last_turn["player"]],
                    "darts": self.last_turn["darts"], "bust": self.last_turn["bust"],
                    "total": self.last_turn["total"]}
        return {"type": "x01", "start": self.start, "double_out": self.double_out,
                "double_in": self.double_in, "players": players, "current": self.cur,
                "turn": self.turn, "darts_left": left, "turn_start": self._turn_start,
                "last_turn": last, "winner": self.winner,
                "winner_name": self.players[self.winner] if self.winner is not None else None,
                "turn_ended": self.turn_ended, "checkout": hint, "checkouts": options,
                "can_undo": bool(self.history), "legs": self.legs, "leg_number": self.leg_index + 1,
                "leg_event": ({"winner": self.leg_event["winner"], "name": self.players[self.leg_event["winner"]],
                               "leg": self.leg_event["leg"] + 1} if self.leg_event else None)}
