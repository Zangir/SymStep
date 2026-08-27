#!/usr/bin/env python3
"""The temporal fluent layer: narrative text -> interval-stamped state ->
value queries. The general mechanism behind "who is where / holding what /
since when" for ANY story, keyed entirely by sentence and question SHAPES
(rows in kb.py) — no task identifiers anywhere.

State = edges that are NEVER deleted. Each edge (subject, predicate,
object) carries polarity, a validity interval [start, end) in sentence
time, an optional coarse day-rank ("yesterday" < "this morning" < ...),
a `via` tag (a subject moved itself vs. was carried), and a disjunction
group ("either in A or B"). An exclusive relation update CLOSES the
previous open edge; the open edge is "current", closed edges are history.

Event rows (kb.py EVENT:*) fold sentences into edge operations; query rows
(kb.py QUERY:*) read the state back: current value, history walk ("where
was X before the Y"), counting, entailment via the constraint engine
(spatial claims, size claims), shortest path over compass edges, and
calibrated motive rows ("where will X go" — the state->destination mapping
is MINED from a labeled training source via calibrate_motives, never
hand-guessed).

Honesty: sentences no event row can read are reported; a query with no
support answers None (refusal), and an undetermined spatial claim abstains
unless the caller enables the closed-world convention explicitly."""
from __future__ import annotations
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import kb
from ..engine.ir import problem_from_ir
from ..engine.core import Engine

CLOSED_WORLD_SPATIAL = False    # answer "no" on scene-undetermined claims


def enable_closed_world_spatial():
    global CLOSED_WORLD_SPATIAL
    CLOSED_WORLD_SPATIAL = True


DIRS_OPP = {"north": "south", "south": "north",
            "east": "west", "west": "east"}

EVENT_ORDER = ["EVENT:MOVE", "EVENT:ACQUIRE", "EVENT:RELEASE",
               "EVENT:TRANSFER", "EVENT:NEG_LOC", "EVENT:EITHER_LOC",
               "EVENT:COLOR", "EVENT:ISA", "EVENT:STATE", "EVENT:LOCATED",
               "EVENT:AFRAID", "EVENT:COMPASS", "EVENT:SPATIAL",
               "EVENT:FITS", "EVENT:BIGGER"]

_TIME_PRE = re.compile(r"^(yesterday|this morning|this afternoon|"
                       r"this evening)\s+", re.I)
_TIME_SUF = re.compile(r"\s+(yesterday|this morning|this afternoon|"
                       r"this evening)\.?$", re.I)
_DISCOURSE = re.compile(r"^(after that|following that|afterwards|then)[, ]*",
                        re.I)


# ------------------------------------------------------------------ state

@dataclass
class FEdge:
    subj: str
    pred: str                  # location | carry | type | color | ...
    obj: str
    polarity: bool = True
    start: int = 0
    end: Optional[int] = None  # None = open = current
    rank: Optional[int] = None # coarse day rank when the sentence gave one
    via: str = "move"          # move | carried
    group: Optional[int] = None  # disjunction group id


class StoryState:
    def __init__(self):
        self.edges: List[FEdge] = []
        self.events: List[Tuple] = []      # ("give", giver, obj, recv, si)
        self.last_people: List[str] = []   # he/she/they antecedents
        self.first_rank: Dict[Tuple[str, str], int] = {}
        self._acq = 0
        self.color_order: List[Tuple[str, str]] = []
        self.compass: Dict[Tuple[str, str], str] = {}
        self.spatial: List[Tuple[str, str, str]] = []
        self.sizes: List[Tuple[str, str]] = []    # (small, big)

    # -- edge machinery (never delete; close on update) ------------------
    def open(self, e: FEdge) -> None:
        self.edges.append(e)

    def close(self, subj: str, pred: str, si: int, obj: str = None) -> None:
        for e in self.edges:
            if (e.subj == subj and e.pred == pred and e.end is None
                    and (obj is None or e.obj == obj)):
                e.end = si

    def open_edges(self, subj: str, pred: str) -> List[FEdge]:
        return [e for e in self.edges
                if e.subj == subj and e.pred == pred and e.end is None]

    # -- fluent operations ------------------------------------------------
    def loc(self, x: str) -> Optional[str]:
        cur = [e for e in self.open_edges(x, "location")
               if e.polarity and e.group is None and e.via == "move"]
        return cur[-1].obj if cur else None

    def obj_location(self, o: str) -> Optional[str]:
        hist = [e for e in self.edges
                if e.subj == o and e.pred == "location" and e.via == "carried"]
        return hist[-1].obj if hist else None

    def carriers_of(self, o: str) -> List[str]:
        return [e.subj for e in self.edges
                if e.pred == "carry" and e.obj == o and e.end is None]

    def carrying(self, p: str) -> List[str]:
        return [e.obj for e in self.open_edges(p, "carry")]

    def move(self, people: List[str], loc: str, si: int,
             rank: Optional[int]) -> None:
        for p in people:
            self.close(p, "location", si)      # definite, neg, disjunctive
            self.open(FEdge(p, "location", loc, start=si, rank=rank))
            for o in self.carrying(p):
                if self.obj_location(o) != loc:
                    self.open(FEdge(o, "location", loc, start=si,
                                    via="carried"))

    def acquire(self, p: str, o: str, si: int) -> None:
        self.open(FEdge(p, "carry", o, start=si))
        self._acq += 1
        self.first_rank.setdefault((p, o), self._acq)
        here = self.loc(p)
        if here and self.obj_location(o) != here:
            self.open(FEdge(o, "location", here, start=si, via="carried"))

    def release(self, p: str, o: str, si: int) -> None:
        self.close(p, "carry", si, obj=o)

    def statics(self, pred: str) -> Dict[str, str]:
        return {e.subj: e.obj for e in self.edges
                if e.pred == pred and e.end is None}


# ------------------------------------------------------------------ fold

def _subjects(raw: str, st: StoryState) -> List[str]:
    raw = raw.strip().lower()
    if raw in ("he", "she"):
        return st.last_people[-1:] or []
    if raw == "they":
        return list(st.last_people)
    names = [n.strip() for n in re.split(r"\s+and\s+", raw)]
    st.last_people = names
    return names


def fold(sent: str, st: StoryState, si: int) -> bool:
    """Apply one sentence to the state. True = an event row read it."""
    s = sent.strip()
    if not s:
        return True
    rank = None
    if m := _TIME_PRE.match(s):
        rank = kb.TIME_RANK[m.group(1).lower()]
        s = s[m.end():]
    elif m := _TIME_SUF.search(s):
        rank = kb.TIME_RANK[m.group(1).lower()]
        s = s[: m.start()] + "."
    s = _DISCOURSE.sub("", s).strip()

    hits = {row.symbol: m for row, m in kb.match_symbol(s, "EVENT:")}
    for sym in EVENT_ORDER:
        if sym not in hits:
            continue
        m = hits[sym]
        if sym == "EVENT:MOVE":
            ppl = _subjects(m.group(1), st)
            st.move(ppl, m.group(2), si, rank)
        elif sym == "EVENT:ACQUIRE":
            for p in _subjects(m.group(1), st):
                st.acquire(p, m.group(2), si)
        elif sym == "EVENT:RELEASE":
            for p in _subjects(m.group(1), st):
                st.release(p, m.group(2), si)
        elif sym == "EVENT:TRANSFER":
            giver = _subjects(m.group(1), st)[0]
            o, recv = m.group(2), m.group(3)
            st.release(giver, o, si)
            st.open(FEdge(recv, "carry", o, start=si))
            st.first_rank.setdefault((recv, o), st._acq + 1)
            st._acq += 1
            st.events.append(("give", giver, o, recv, si))
            st.last_people = [giver, recv]
        elif sym == "EVENT:NEG_LOC":
            p = m.group(1)
            st.close(p, "location", si)
            st.open(FEdge(p, "location", m.group(2), polarity=False,
                          start=si))
            st.last_people = [p]
        elif sym == "EVENT:EITHER_LOC":
            p = m.group(1)
            st.close(p, "location", si)
            for l in (m.group(2), m.group(3)):
                st.open(FEdge(p, "location", l, start=si, group=si))
            st.last_people = [p]
        elif sym == "EVENT:COLOR":
            st.open(FEdge(m.group(1), "color", m.group(2), start=si))
            st.color_order.append((m.group(1), m.group(2)))
        elif sym == "EVENT:ISA":
            st.open(FEdge(m.group(1), "type", m.group(2), start=si))
        elif sym == "EVENT:STATE":
            st.close(m.group(1), "state", si)
            st.open(FEdge(m.group(1), "state", m.group(2), start=si))
            st.last_people = [m.group(1)]
        elif sym == "EVENT:LOCATED":
            st.move([m.group(1)], m.group(2), si, rank)
            st.last_people = [m.group(1)]
        elif sym == "EVENT:AFRAID":
            st.open(FEdge(m.group(1), "afraid_of", m.group(2), start=si))
        elif sym == "EVENT:COMPASS":
            st.compass[(m.group(1), m.group(2))] = m.group(3)
        elif sym == "EVENT:SPATIAL":
            st.spatial.append((m.group(1), m.group(2), m.group(3)))
        elif sym == "EVENT:FITS":
            st.sizes.append((m.group(1), m.group(2)))
        elif sym == "EVENT:BIGGER":
            st.sizes.append((m.group(2), m.group(1)))
        return True
    return False


# ------------------------------------------------------------------ entail

def _entail(names, less_pairs, claim_pair) -> Optional[str]:
    dom = list(range(1, 2 * len(names) + 1))
    variables = {n: dom for n in names}
    base = [{"type": "less", "a": a, "b": b} for a, b in less_pairs]

    def sat(extra):
        r = Engine(problem_from_ir(
            {"variables": variables,
             "constraints": base + extra})).solve(max_solutions=1)
        return r.status in ("SOLVED", "AMBIGUOUS")

    a, b = claim_pair
    if not sat([{"type": "less", "a": a, "b": b}]):
        return "no"
    if not sat([{"type": "geq_offset", "a": a, "b": b, "k": 0}]):
        return "yes"
    return None


# ------------------------------------------------------------------ answer

def match_query(question: str):
    q = question.strip().rstrip("?").lower()
    hits = kb.match_symbol(q, "QUERY:")
    return (hits[0][0].symbol, hits[0][1]) if hits else None


def _plural(t: str) -> str:
    inv = {v: k for k, v in kb.IRREGULAR_SINGULAR.items()}
    return inv.get(t, t + "s")


def answer(question: str, st: StoryState) -> Optional[str]:
    hit = match_query(question)
    if hit is None:
        return None
    sym, m = hit

    if sym == "QUERY:WHERE_IS":
        x = m.group(1)
        if (l := st.loc(x)):
            return l
        holders = st.carriers_of(x)
        if holders:
            return st.loc(holders[-1]) or st.obj_location(x)
        return st.obj_location(x)

    if sym == "QUERY:WHERE_BEFORE":
        x, before = m.group(1), m.group(2)
        carried = [e.obj for e in st.edges
                   if e.subj == x and e.pred == "location"
                   and e.via == "carried"]
        ranked = sorted([e for e in st.edges
                         if e.subj == x and e.pred == "location"
                         and e.polarity and e.rank is not None],
                        key=lambda e: e.rank)
        moved = [e.obj for e in st.edges
                 if e.subj == x and e.pred == "location" and e.polarity
                 and e.group is None and e.via == "move"]
        hist = carried or [e.obj for e in ranked] or moved
        for i in range(len(hist) - 1, 0, -1):
            if hist[i] == before:
                return hist[i - 1]
        return None

    if sym == "QUERY:IS_IN":
        p, l = m.group(1), m.group(2)
        if (cur := st.loc(p)):
            return "yes" if cur == l else "no"
        disjunct = {e.obj for e in st.open_edges(p, "location")
                    if e.group is not None}
        if disjunct:
            return "maybe" if l in disjunct else "no"
        negated = {e.obj for e in st.open_edges(p, "location")
                   if not e.polarity}
        if l in negated:
            return "no"
        return None

    if sym == "QUERY:COUNT_CARRY":
        return kb.NUM_WORDS[len(st.carrying(m.group(1)))]

    if sym == "QUERY:WHAT_CARRY":
        p = m.group(1)
        os_ = sorted(st.carrying(p),
                     key=lambda o: st.first_rank.get((p, o), 0))
        return ",".join(os_) if os_ else "nothing"

    if sym in ("QUERY:GIVE_WHAT", "QUERY:GIVE_WHO", "QUERY:GIVE_RECV",
               "QUERY:RECEIVED", "QUERY:GAVE"):
        sel = {"QUERY:GIVE_WHAT":
               lambda g, o, r: o if (g == m.group(1) and r == m.group(2))
               else None,
               "QUERY:GIVE_WHO":
               lambda g, o, r: g if (o == m.group(1) and r == m.group(2))
               else None,
               "QUERY:GIVE_RECV":
               lambda g, o, r: r if (g == m.group(1) and o == m.group(2))
               else None,
               "QUERY:RECEIVED":
               lambda g, o, r: r if o == m.group(1) else None,
               "QUERY:GAVE":
               lambda g, o, r: g if o == m.group(1) else None}[sym]
        hits = [v for _, g, o, r, _si in st.events
                if (v := sel(g, o, r))]
        return hits[-1] if hits else None

    if sym == "QUERY:COMPASS_OF":
        room, d = m.group(1), m.group(2)
        if (room, d) in st.compass:
            return st.compass[(room, d)]
        for (a, dd), b in st.compass.items():
            if b == room and dd == DIRS_OPP[d]:
                return a
        return None

    if sym == "QUERY:OF_COMPASS":
        d, room = m.group(1), m.group(2)
        for (a, dd), b in st.compass.items():
            if dd == d and b == room:
                return a
        return st.compass.get((room, DIRS_OPP[d]))

    if sym == "QUERY:AFRAID_OF":
        types = st.statics("type")
        afraid = st.statics("afraid_of")
        t = types.get(m.group(1))
        prey = (afraid.get(_plural(t)) if t else None) \
            or (afraid.get(t) if t else None)
        return kb.IRREGULAR_SINGULAR.get(prey, prey)

    if sym == "QUERY:WHAT_COLOR":
        name = m.group(1)
        types = st.statics("type")
        t = types.get(name)
        hits = [c for n, c in st.color_order
                if n != name and types.get(n) == t]
        return hits[-1] if hits else None

    if sym == "QUERY:SPATIAL_CLAIM":
        a, rel, b = m.group(1), m.group(2), m.group(3)
        names = {a, b} | {x for s, _, o in st.spatial for x in (s, o)}
        dom = list(range(1, 2 * len(names) + 1))
        variables = {}
        for nm in names:
            variables[f"x_{nm}"] = dom
            variables[f"y_{nm}"] = dom

        def rel_specs(s, r, o):
            if "left" in r:
                return [{"type": "less", "a": f"x_{s}", "b": f"x_{o}"}]
            if "right" in r:
                return [{"type": "less", "a": f"x_{o}", "b": f"x_{s}"}]
            if r == "above":
                return [{"type": "less", "a": f"y_{o}", "b": f"y_{s}"}]
            return [{"type": "less", "a": f"y_{s}", "b": f"y_{o}"}]

        base = []
        for s, r, o in st.spatial:
            base.extend(rel_specs(s, r, o))
        claim = rel_specs(a, rel, b)

        def sat(extra):
            r2 = Engine(problem_from_ir(
                {"variables": variables,
                 "constraints": base + extra})).solve(max_solutions=1)
            return r2.status in ("SOLVED", "AMBIGUOUS")

        neg = [{"type": "not", "c": claim[0]}]
        if not sat(claim):
            return "no"
        if not sat(neg):
            return "yes"
        return "no" if CLOSED_WORLD_SPATIAL else None

    if sym == "QUERY:FITS_CLAIM":
        names = {x for pr in st.sizes for x in pr} | {m.group(1), m.group(2)}
        return _entail(names, st.sizes, (m.group(1), m.group(2)))

    if sym == "QUERY:BIGGER_CLAIM":
        names = {x for pr in st.sizes for x in pr} | {m.group(1), m.group(2)}
        return _entail(names, st.sizes, (m.group(2), m.group(1)))

    if sym == "QUERY:PATH":
        edges = defaultdict(dict)
        for (a, d), b in st.compass.items():   # a is d of b: b -d-> a
            edges[b][d] = a
            edges[a][DIRS_OPP[d]] = b
        start, goal = m.group(1), m.group(2)
        seen = {start: []}
        dq = deque([start])
        while dq:
            cur = dq.popleft()
            if cur == goal:
                return " ".join(seen[cur])
            for d, nxt in edges[cur].items():
                if nxt not in seen:
                    seen[nxt] = seen[cur] + [d]
                    dq.append(nxt)
        return None

    if sym == "QUERY:WHERE_WILL":
        state = st.statics("state").get(m.group(1))
        if state is None:
            return None
        for row in kb.KB:
            if row.symbol == "MOTIVE" and row.pattern == state:
                return row.payload
        return None

    if sym in ("QUERY:WHY_GO", "QUERY:WHY_GET"):
        return st.statics("state").get(m.group(1))

    return None


# --------------------------------------------------------- calibration

def calibrate_motives(source: str) -> None:
    """Mine the state -> destination mapping ("thirsty people head to the
    kitchen") from a LABELED training source given as "dataset@split".
    Adds MOTIVE rows and the EVENT:STATE row (over exactly the mined
    states) to the knowledge store, provenance-stamped. Test data is
    never touched."""
    from datasets import load_dataset
    name, split = source.split("@")
    ds = load_dataset(name, split=split)
    votes = defaultdict(lambda: defaultdict(int))
    is_state = re.compile(r"^(\w+) is (\w+)\.?$")
    for x in ds:
        qm = re.match(r"^where will (\w+) go\??$",
                      str(x.get("question", "")).strip().lower())
        if not qm:
            continue
        person, gold = qm.group(1), str(x.get("answer", "")).strip().lower()
        for line in str(x.get("passage", "")).split("\n"):
            sm = is_state.match(line.strip().lower())
            if sm and sm.group(1) == person:
                votes[sm.group(2)][gold] += 1
    states = []
    for state, rooms in votes.items():
        best = max(rooms, key=rooms.get)
        if rooms[best] >= 0.9 * sum(rooms.values()):
            kb.add(kb.Row(state, "MOTIVE", payload=best,
                          provenance=f"calibrated:{source}"))
            states.append(state)
    if states:
        kb.add(kb.Row(("re", rf"^(\w+) is ({'|'.join(sorted(states))})\.?$"),
                      "EVENT:STATE", provenance=f"calibrated:{source}"))
    print(f"  calibrated {len(states)} motive states from {source}")
