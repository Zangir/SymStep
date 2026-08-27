#!/usr/bin/env python3
"""Unit tests for the GraphStep engine, constraint library, and IR."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphstep.engine.core import Problem, Engine
from graphstep.engine import constraints as C
from graphstep.engine.ir import problem_from_ir, build_constraint, IRError

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


# 1. Zebra-mini: 3 houses, classic value-position encoding, unique solution
ir = {
    "variables": {
        "Alice": [1, 2, 3], "Bob": [1, 2, 3], "Carol": [1, 2, 3],
        "Red": [1, 2, 3], "Blue": [1, 2, 3], "Green": [1, 2, 3],
        "Cat": [1, 2, 3], "Dog": [1, 2, 3], "Fish": [1, 2, 3],
    },
    "constraints": [
        {"type": "alldiff", "vars": ["Alice", "Bob", "Carol"]},
        {"type": "alldiff", "vars": ["Red", "Blue", "Green"]},
        {"type": "alldiff", "vars": ["Cat", "Dog", "Fish"]},
        {"type": "is", "var": "Alice", "value": 1},
        {"type": "same", "a": "Bob", "b": "Blue"},          # Bob in blue house
        {"type": "same", "a": "Red", "b": "Cat"},           # red house has cat
        {"type": "less", "a": "Alice", "b": "Bob"},         # Alice left of Bob
        {"type": "absdiff", "a": "Carol", "b": "Alice", "k": 1},  # neighbors
        {"type": "same", "a": "Carol", "b": "Green"},
        {"type": "same", "a": "Carol", "b": "Fish"},
        {"type": "is", "var": "Bob", "value": 3},
        {"type": "same", "a": "Alice", "b": "Red"},
    ],
}
res = Engine(problem_from_ir(ir)).solve()
check("zebra-mini solved", res.status == "SOLVED", res.status)
if res.solutions:
    s = res.solutions[0]
    check("zebra-mini assignment",
          s["Alice"] == 1 and s["Carol"] == 2 and s["Bob"] == 3
          and s["Cat"] == 1 and s["Fish"] == 2 and s["Blue"] == 3, s)

# 2. UNSAT with explanation
bad = {
    "variables": {"X": [1, 2], "Y": [1, 2]},
    "constraints": [
        {"type": "alldiff", "vars": ["X", "Y"]},
        {"type": "is", "var": "X", "value": 1},
        {"type": "is", "var": "Y", "value": 1},
    ],
}
res = Engine(problem_from_ir(bad)).solve()
check("unsat detected", res.status == "UNSAT", res.status)
check("unsat explained", "CONTRADICTION" in res.explanation, res.explanation)

# 3. Ambiguity detection
amb = {"variables": {"X": [1, 2], "Y": [1, 2]},
       "constraints": [{"type": "alldiff", "vars": ["X", "Y"]}]}
res = Engine(problem_from_ir(amb)).solve()
check("ambiguous detected", res.status == "AMBIGUOUS", res.status)

# 4. Unsat core: 2 conflicting + 1 innocent constraint
core_ir = {
    "variables": {"X": [1, 2, 3], "Y": [1, 2, 3]},
    "constraints": [
        {"type": "is", "var": "X", "value": 1, "origin": "clue A"},
        {"type": "is", "var": "X", "value": 2, "origin": "clue B"},
        {"type": "less", "a": "X", "b": "Y", "origin": "clue C (innocent)"},
    ],
}
eng = Engine(problem_from_ir(core_ir))
core = eng.unsat_core()
origins = {c.origin for c in core}
check("core minimal", origins == {"clue A", "clue B"}, origins)

# 5. Reification: Implies + Or + Not
reif = {
    "variables": {"A": [1, 2, 3], "B": [1, 2, 3]},
    "constraints": [
        {"type": "implies",
         "if": {"type": "is", "var": "A", "value": 1},
         "then": {"type": "is", "var": "B", "value": 3}},
        {"type": "is", "var": "A", "value": 1},
    ],
}
res = Engine(problem_from_ir(reif)).solve()
check("implies fires", res.status == "SOLVED" and res.solutions[0]["B"] == 3,
      res.solutions)

reif2 = {
    "variables": {"A": [1, 2], "B": [1, 2]},
    "constraints": [
        {"type": "or", "clauses": [
            {"type": "is", "var": "A", "value": 1},
            {"type": "is", "var": "B", "value": 2}]},
        {"type": "is_not", "var": "A", "value": 1},
        {"type": "is", "var": "B", "value": 1},
    ],
}
res = Engine(problem_from_ir(reif2)).solve()
check("or falsified -> unsat", res.status == "UNSAT", res.status)

# 6. Not / negate round trips
n = build_constraint({"type": "not", "c": {"type": "same", "a": "A", "b": "B"}},
                     {"A": [1, 2], "B": [1, 2]})
check("not(same) == diff", "!=" in n.describe(), n.describe())

# 7. Table universal fallback: XOR-like relation
tab = {
    "variables": {"P": [0, 1], "Q": [0, 1], "R": [0, 1]},
    "constraints": [
        {"type": "table", "vars": ["P", "Q", "R"],
         "allowed": [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]},  # R = P xor Q
        {"type": "is", "var": "P", "value": 1},
        {"type": "is", "var": "Q", "value": 1},
    ],
}
res = Engine(problem_from_ir(tab)).solve()
check("table xor", res.status == "SOLVED" and res.solutions[0]["R"] == 0,
      res.solutions)

# 8. Count constraint
cnt = {
    "variables": {"A": ["x", "y"], "B": ["x", "y"], "C": ["x", "y"]},
    "constraints": [
        {"type": "count", "vars": ["A", "B", "C"], "value": "x", "op": "==",
         "k": 1},
        {"type": "is", "var": "A", "value": "x"},
    ],
}
res = Engine(problem_from_ir(cnt)).solve()
check("count quota prunes",
      res.status in ("SOLVED", "AMBIGUOUS")
      and all(s["B"] == "y" and s["C"] == "y" for s in res.solutions),
      res.solutions)

# 9. IR validation errors are precise
try:
    build_constraint({"type": "is", "var": "Nope", "value": 1}, {"X": [1]})
    check("ir unknown var", False)
except IRError as e:
    check("ir unknown var", "unknown variable" in str(e), str(e))
try:
    build_constraint({"type": "is", "var": "X", "value": 9}, {"X": [1, 2]})
    check("ir bad value", False)
except IRError as e:
    check("ir bad value", "not in the domain" in str(e), str(e))

# 10. Offset / ordering interplay (scheduling-style)
sched = {
    "variables": {"Talk1": [1, 2, 3, 4], "Talk2": [1, 2, 3, 4],
                  "Talk3": [1, 2, 3, 4]},
    "constraints": [
        {"type": "alldiff", "vars": ["Talk1", "Talk2", "Talk3"]},
        {"type": "offset", "a": "Talk2", "b": "Talk1", "k": 1},  # T2 right after T1
        {"type": "less", "a": "Talk2", "b": "Talk3"},
        {"type": "is_not", "var": "Talk1", "value": 2},
    ],
}
res = Engine(problem_from_ir(sched)).solve()
check("scheduling solved", res.status in ("SOLVED", "AMBIGUOUS"), res.status)
for s in res.solutions:
    check("scheduling valid",
          s["Talk2"] == s["Talk1"] + 1 and s["Talk2"] < s["Talk3"]
          and s["Talk1"] != 2, s)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
