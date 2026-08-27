#!/usr/bin/env python3
"""bAbI (20 tasks) with GraphStep: a deterministic story-state engine
(temporal knowledge graph folded sentence by sentence) + the CSP engine for
the constraint-shaped tasks (17 positional, 18 size — answered by
ENTAILMENT: yes iff the negated claim is UNSAT) + graph search for path
finding (19). Zero LLM calls.

Usage:  python3 graphstep/bench_babi.py [--n 100]
"""
import sys, os, re, json, argparse
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from graphstep.engine.ir import problem_from_ir
from graphstep.engine.core import Engine

MOVE = re.compile(r"^(.+?) (?:went back|went|journeyed|travelled|moved|ran|"
                  r"walked|hurried) (?:back )?to the (\w+)\.?$")
GET = re.compile(r"^(.+?) (?:got|grabbed|took|picked up) the (\w+)"
                 r"(?: there)?\.?$")
DROP = re.compile(r"^(.+?) (?:dropped|discarded|put down|left) the (\w+)"
                  r"(?: there)?\.?$")
GIVE = re.compile(r"^(.+?) (?:gave|handed|passed) the (\w+) to (\w+)\.?$")
ISIN = re.compile(r"^(\w+) is in the (\w+)\.?$")
NOLONGER = re.compile(r"^(\w+) is no longer in the (\w+)\.?$")
EITHER = re.compile(r"^(\w+) is either in the (\w+) or the (\w+)\.?$")
AFRAID = re.compile(r"^(\w+) are afraid of (\w+)\.?$", re.I)
ISA = re.compile(r"^(\w+) is a (\w+)\.?$", re.I)
ISCOLOR = re.compile(r"^(\w+) is (white|gray|grey|green|yellow|red|blue)\.?$",
                     re.I)
COMPASS = re.compile(r"^The (\w+) is (north|south|east|west) of the (\w+)\.?$")
SPATIAL = re.compile(r"^The (.+?) is (to the left of|to the right of|above|"
                     r"below) the (.+?)\.?$")
FITS = re.compile(r"^The (.+?) fits in(?:side)? the (.+?)\.?$", re.I)
BIGGER = re.compile(r"^The (.+?) is bigger than the (.+?)\.?$", re.I)
STATEMENT = re.compile(r"^(\w+) is (hungry|thirsty|bored|tired)\.?$")
TIME = re.compile(r"^(Yesterday|This morning|This afternoon|This evening)\s+",
                  re.I)
TIME_SUF = re.compile(r"\s+(yesterday|this morning|this afternoon|"
                      r"this evening)\.?$", re.I)
NOTIN = re.compile(r"^(\w+) is not in the (\w+)\.?$")
TIME_RANK = {"yesterday": 0, "this morning": 1, "this afternoon": 2,
             "this evening": 3}
NUMW = ["none", "one", "two", "three", "four", "five", "six"]
SINGULAR = {"wolves": "wolf", "mice": "mouse", "cats": "cat",
            "sheep": "sheep", "lions": "lion", "swans": "swan",
            "frogs": "frog", "rhinos": "rhino"}
MOTIVE_ROOM = {"hungry": "kitchen", "thirsty": "kitchen", "bored": "garden",
               "tired": "bedroom"}
DIRS = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
CLOSED_WORLD_17 = False


class Story:
    def __init__(self):
        self.loc = {}                    # person -> location (lowercased keys)
        self.loc_hist = defaultdict(list)
        self.objs = defaultdict(list)    # person -> [objects]
        self.first_rank = {}             # (person, obj) -> acquisition rank
        self._rank = 0
        self.obj_hist = defaultdict(list)  # object -> [locations]
        self.neg = defaultdict(set)      # person -> {excluded locations}
        self.either = {}                 # person -> {loc1, loc2}
        self.gives = []                  # (giver, obj, receiver)
        self.last_people = []            # for he/she/they coreference
        self.afraid = {}                 # type -> prey type
        self.typeof = {}                 # name -> type
        self.type_order = []             # [(name, type)] in assertion order
        self.colors = {}                 # name -> color
        self.color_order = []            # [(name, color)] in assertion order
        self.compass = {}                # (room, dir) -> room  (A dir-of B)
        self.spatial = []                # (a, rel, b)
        self.sizes = []                  # (small, big) meaning small < big
        self.motive = {}                 # person -> state
        self.timed = defaultdict(list)   # person -> [(rank, loc)]

    # ---------------- state updates
    def move(self, people, loc):
        for p in people:
            self.loc[p] = loc
            self.loc_hist[p].append(loc)
            self.neg[p].clear()
            self.either.pop(p, None)
            for o in self.objs[p]:
                if not self.obj_hist[o] or self.obj_hist[o][-1] != loc:
                    self.obj_hist[o].append(loc)

    def obj_location(self, o):
        return self.obj_hist[o][-1] if self.obj_hist[o] else None


def subjects(raw: str, st: Story):
    raw = raw.strip()
    if raw.lower() in ("he", "she"):
        return st.last_people[-1:] or []
    if raw.lower() == "they":
        return list(st.last_people)
    names = [n.strip().lower() for n in re.split(r"\s+and\s+", raw)]
    st.last_people = names
    return names


def fold(sent: str, st: Story):
    sent = sent.strip()
    if not sent:
        return
    rank = None
    m = TIME.match(sent)
    if m:
        rank = TIME_RANK[m.group(1).lower()]
        sent = sent[m.end():].strip()
        sent = sent[0].upper() + sent[1:]
    else:
        m = TIME_SUF.search(sent)
        if m:
            rank = TIME_RANK[m.group(1).lower()]
            sent = sent[:m.start()].strip() + "."
    sent = re.sub(r"^(After that|Following that|Afterwards|Then)[, ]*", "",
                  sent, flags=re.I).strip()
    if sent and sent[0].islower():
        sent = sent[0].upper() + sent[1:]

    if m2 := MOVE.match(sent):
        ppl = subjects(m2.group(1), st)
        if rank is not None:
            for p in ppl:
                st.timed[p].append((rank, m2.group(2).lower()))
        st.move(ppl, m2.group(2).lower())
    elif m2 := GET.match(sent):
        for p in subjects(m2.group(1), st):
            o = m2.group(2).lower()
            st.objs[p].append(o)
            st._rank += 1
            st.first_rank.setdefault((p, o), st._rank)
            if p in st.loc and (not st.obj_hist[o]
                                or st.obj_hist[o][-1] != st.loc[p]):
                st.obj_hist[o].append(st.loc[p])
    elif m2 := DROP.match(sent):
        for p in subjects(m2.group(1), st):
            o = m2.group(2).lower()
            if o in st.objs[p]:
                st.objs[p].remove(o)
    elif m2 := GIVE.match(sent):
        giver = subjects(m2.group(1), st)[0]
        o, recv = m2.group(2).lower(), m2.group(3).lower()
        if o in st.objs[giver]:
            st.objs[giver].remove(o)
        st.objs[recv].append(o)
        st.gives.append((giver, o, recv))
        st.last_people = [giver, recv]
    elif m2 := (NOLONGER.match(sent) or NOTIN.match(sent)):
        p = m2.group(1).lower()
        st.neg[p].add(m2.group(2).lower())
        st.loc.pop(p, None)
        st.last_people = [p]
    elif m2 := EITHER.match(sent):
        p = m2.group(1).lower()
        st.either[p] = {m2.group(2).lower(), m2.group(3).lower()}
        st.loc.pop(p, None)
        st.last_people = [p]
    elif m2 := ISCOLOR.match(sent):
        st.colors[m2.group(1).lower()] = m2.group(2).lower()
        st.color_order.append((m2.group(1).lower(), m2.group(2).lower()))
    elif m2 := ISA.match(sent):
        st.typeof[m2.group(1).lower()] = m2.group(2).lower()
        st.type_order.append((m2.group(1).lower(), m2.group(2).lower()))
    elif m2 := STATEMENT.match(sent):
        st.motive[m2.group(1).lower()] = m2.group(2).lower()
        st.last_people = [m2.group(1).lower()]
    elif m2 := ISIN.match(sent):
        st.move([m2.group(1).lower()], m2.group(2).lower())
        st.last_people = [m2.group(1).lower()]
    elif m2 := AFRAID.match(sent):
        st.afraid[m2.group(1).lower()] = m2.group(2).lower()
    elif m2 := COMPASS.match(sent):
        st.compass[(m2.group(1).lower(), m2.group(2))] = m2.group(3).lower()
    elif m2 := SPATIAL.match(sent):
        st.spatial.append((m2.group(1).lower(), m2.group(2), m2.group(3).lower()))
    elif m2 := FITS.match(sent):
        st.sizes.append((m2.group(1).lower(), m2.group(2).lower()))
    elif m2 := BIGGER.match(sent):
        st.sizes.append((m2.group(2).lower(), m2.group(1).lower()))


# ---------------- CSP entailment helpers (tasks 17, 18)
def entail(names, less_pairs, claim_pair, axis_pairs=None):
    """yes iff claim entailed, no iff contradicted, None if undetermined.
    less_pairs: [(a, b)] meaning v(a) < v(b). claim_pair: (a, b) claim a < b."""
    dom = list(range(1, 2 * len(names) + 1))
    variables = {n: dom for n in names}
    base = [{"type": "less", "a": a, "b": b} for a, b in less_pairs]

    def sat(extra):
        r = Engine(problem_from_ir(
            {"variables": variables,
             "constraints": base + extra})).solve(max_solutions=1)
        return r.status in ("SOLVED", "AMBIGUOUS")

    a, b = claim_pair
    claim = {"type": "less", "a": a, "b": b}
    neg = {"type": "geq_offset", "a": a, "b": b, "k": 0}   # a >= b
    if not sat([claim]):
        return "no"
    if not sat([neg]):
        return "yes"
    return None


def answer(task, q, st: Story):
    ql = q.strip().rstrip("?").lower()

    if m := re.match(r"where is (?:the )?(\w+)$", ql):
        x = m.group(1)
        if x in st.loc:
            return st.loc[x]
        holder = next((p for p, os_ in st.objs.items() if x in os_), None)
        if holder:
            return st.loc.get(holder) or st.obj_location(x)
        return st.obj_location(x)

    if m := re.match(r"where was (?:the )?(\w+) before the (\w+)$", ql):
        x, before = m.group(1), m.group(2)
        hist = st.obj_hist.get(x) or [l for _, l in
                                      sorted(st.timed.get(x, []))] \
            or st.loc_hist.get(x, [])
        for i in range(len(hist) - 1, 0, -1):
            if hist[i] == before:
                return hist[i - 1]
        return None

    if m := re.match(r"is (\w+) in the (\w+)$", ql):
        p, l = m.group(1), m.group(2)
        if p in st.loc:
            return "yes" if st.loc[p] == l else "no"
        if p in st.either:
            return "maybe" if l in st.either[p] else "no"
        if l in st.neg.get(p, set()):
            return "no"
        return None

    if m := re.match(r"how many objects is (\w+) (?:carrying|holding)$", ql):
        return NUMW[len(st.objs.get(m.group(1), []))]

    if m := re.match(r"what is (\w+) (?:carrying|holding)$", ql):
        p = m.group(1)
        os_ = sorted(st.objs.get(p, []),
                     key=lambda o: st.first_rank.get((p, o), 0))
        return ",".join(os_) if os_ else "nothing"

    if m := re.match(r"what did (\w+) give to (\w+)$", ql):
        hits = [o for g, o, r in st.gives
                if g == m.group(1) and r == m.group(2)]
        return hits[-1] if hits else None
    if m := re.match(r"who gave the (\w+) to (\w+)$", ql):
        hits = [g for g, o, r in st.gives
                if o == m.group(1) and r == m.group(2)]
        return hits[-1] if hits else None
    if m := re.match(r"who did (\w+) give the (\w+) to$", ql):
        hits = [r for g, o, r in st.gives
                if g == m.group(1) and o == m.group(2)]
        return hits[-1] if hits else None
    if m := re.match(r"who received the (\w+)$", ql):
        hits = [r for g, o, r in st.gives if o == m.group(1)]
        return hits[-1] if hits else None
    if m := re.match(r"who gave the (\w+)$", ql):
        hits = [g for g, o, r in st.gives if o == m.group(1)]
        return hits[-1] if hits else None

    if m := re.match(r"what is (?:the )?(\w+) (north|south|east|west) of$", ql):
        room, d = m.group(1), m.group(2)
        if (room, d) in st.compass:
            return st.compass[(room, d)]
        for (a, dd), b in st.compass.items():          # inverse reading
            if b == room and dd == OPP[d]:
                return a
        return None
    if m := re.match(r"what is (north|south|east|west) of the (\w+)$", ql):
        d, room = m.group(1), m.group(2)
        for (a, dd), b in st.compass.items():
            if dd == d and b == room:
                return a
        if (room, OPP[d]) in st.compass:
            return st.compass[(room, OPP[d])]
        return None

    if m := re.match(r"what is (\w+) afraid of$", ql):
        t = st.typeof.get(m.group(1))
        prey = st.afraid.get(t + "s" if t else "", None) \
            or st.afraid.get({"mouse": "mice", "wolf": "wolves"}.get(t, ""),
                             None) or (st.afraid.get(t) if t else None)
        return SINGULAR.get(prey, prey)

    if m := re.match(r"what color is (\w+)$", ql):
        name = m.group(1)
        t = st.typeof.get(name)
        hits = [c for n, c in st.color_order
                if n != name and st.typeof.get(n) == t]
        return hits[-1] if hits else None

    if task == 17 and (m := re.match(
            r"is the (.+?) (to the left of|to the right of|above|below) "
            r"the (.+)$", ql)):
        a, rel, b = m.group(1), m.group(2), m.group(3)
        names = {a, b}
        for s, r, o in st.spatial:
            names |= {s, o}
        dom = list(range(1, 2 * len(names) + 1))
        variables = {}
        for nm in names:
            variables[f"x_{nm}"] = dom
            variables[f"y_{nm}"] = dom

        def rel_specs(s, r, o):
            # loose axis semantics: "left of" constrains x only, etc.
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

        neg = [{"type": "not",
                "c": claim[0] if len(claim) == 1 else
                {"type": "and", "clauses": claim}}]
        if not sat(claim):
            return "no"
        if not sat(neg):
            return "yes"
        return "no" if CLOSED_WORLD_17 else None   # undetermined: scene-dependent

    if task == 18:
        if m := re.match(r"does the (.+?) fit in(?:side)? the (.+)$", ql):
            names = {x for pr in st.sizes for x in pr} | \
                    {m.group(1), m.group(2)}
            return entail(names, st.sizes, (m.group(1), m.group(2)))
        if m := re.match(r"is the (.+?) bigger than the (.+)$", ql):
            names = {x for pr in st.sizes for x in pr} | \
                    {m.group(1), m.group(2)}
            return entail(names, st.sizes, (m.group(2), m.group(1)))

    if task == 19 and (m := re.match(
            r"how do you go from the (\w+) to the (\w+)$", ql)):
        pos = {}                                    # room -> (x, y) via BFS lay
        edges = defaultdict(dict)                   # room -> dir -> room
        for (a, d), b in st.compass.items():        # a is d of b: b -d-> a
            edges[b][d] = a
            edges[a][OPP[d]] = b
        start, goal = m.group(1), m.group(2)
        from collections import deque
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

    if task == 20:
        if m := re.match(r"where will (\w+) go$", ql):
            return MOTIVE_ROOM.get(st.motive.get(m.group(1), ""), None)
        if m := re.match(r"why did (\w+) go to the (\w+)$", ql):
            return st.motive.get(m.group(1))
        if m := re.match(r"why did (\w+) (?:get|grab|take|pick up) the (\w+)$",
                         ql):
            return st.motive.get(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="questions per task")
    ap.add_argument("--closed-world17", action="store_true",
                    help="answer 'no' on scene-undetermined task-17 items")
    args = ap.parse_args()
    global CLOSED_WORLD_17
    CLOSED_WORLD_17 = args.closed_world17
    from datasets import load_dataset
    ds = load_dataset("Muennighoff/babi", split="test")
    by_task = defaultdict(list)
    for x in ds:
        if len(by_task[x["task"]]) < args.n:
            by_task[x["task"]].append(x)

    results = {}
    for t in range(1, 21):
        correct = wrong = unanswered = 0
        for item in by_task[t]:
            st = Story()
            for sent in item["passage"].split("\n"):
                fold(sent, st)
            try:
                pred = answer(t, item["question"], st)
            except Exception:
                pred = None
            gold = item["answer"].strip().lower()
            if pred is None:
                unanswered += 1
            elif str(pred).strip().lower() == gold:
                correct += 1
            else:
                wrong += 1
        n = len(by_task[t])
        results[t] = {"n": n, "correct": correct, "wrong": wrong,
                      "unanswered": unanswered}
        print(f"  task {t:2d}: {correct:3d}/{n}  "
              f"(wrong {wrong}, unanswered {unanswered})")
    total_c = sum(r["correct"] for r in results.values())
    total_n = sum(r["n"] for r in results.values())
    print(f"\n  TOTAL: {total_c}/{total_n} = {100*total_c/total_n:.1f}%   "
          f"LLM calls: 0")
    out = os.path.join(ROOT, "graphstep", "results", "results_babi.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    main()
