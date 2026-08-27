#!/usr/bin/env python3
"""Tests for the universal text->graph layer. Zero LLM calls throughout."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments"))

from graphstep.reading.universal import solve_universal, compile_universal

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {extra}")


def entity_view(res, rep):
    enc = {i: e for e, i in rep["encoding"]["entities"].items()}
    return {v: enc.get(p) for v, p in res.solutions[0].items()}


# 1. novel domain, novel verbs (musicians/instruments/cities)
text1 = """Mira, Jonas, and Petra each play a different instrument and each
performs in a different city. Mira does not play the violin.
Jonas performs in Vienna. The person in Berlin plays the cello.
Petra performs in Madrid. Petra plays the flute."""
res, rep = solve_universal(text1)
check("novel domain solved", res is not None and res.status == "SOLVED",
      rep if res is None else res.status)
if res and res.status == "SOLVED":
    view = entity_view(res, rep)
    check("novel domain correct",
          view.get("Cello") == "Mira" and view.get("Violin") == "Jonas"
          and view.get("Berlin") == "Mira", view)

# 2. conditional sentence through the universal path
text2 = """Dana, Eli, and Fay each drink a different beverage. Dana drinks tea.
If Dana drinks tea, then Eli drinks coffee. Fay does not drink coffee."""
res, rep = solve_universal(text2)
check("conditional solved", res is not None
      and res.status in ("SOLVED",), rep if res is None else res.status)
if res and res.solutions:
    view = entity_view(res, rep)
    check("conditional fires", view.get("Coffee") == "Eli", view)

# 3. arbitrary prose degrades honestly (no fake solve, coverage reported)
text3 = """The propagator handles bijective constraints with arc-consistency
propagation. Temporal ordering and counting are well-studied in constraint
programming. Adding SAT backends would extend the system to these domains."""
res, rep = solve_universal(text3)
check("prose: no fabricated answer",
      res is None or res.status in ("AMBIGUOUS", "UNKNOWN", "UNSAT"),
      rep.get("status"))
check("prose: coverage reported", "uncovered" in rep and "triples" in rep)

# 4. LGP-20 from RAW PROSE (no inventory handed over)
import symstep
from extended import NEW_PUZZLES, EXTRA_PUZZLES

lgp = symstep.PUZZLES + NEW_PUZZLES + EXTRA_PUZZLES
solved = correct = 0
for p in lgp:
    names = ", ".join(p.people[:-1]) + f", and {p.people[-1]}"
    intro = (f"{names} each have a different "
             + " and a different ".join(p.attributes.keys()) + ". ")
    intro += " ".join(
        f"The {attr}s are " + ", ".join(vs[:-1]) + f", and {vs[-1]}."
        for attr, vs in p.attributes.items())
    text = intro + " " + " ".join(p.clues)
    res, rep = solve_universal(text)
    if res is None or res.status != "SOLVED":
        continue
    solved += 1
    enc = {i: e for e, i in rep["encoding"]["entities"].items()}
    view = {}
    for val, pos in res.solutions[0].items():
        ent = enc.get(pos)
        if ent:
            view.setdefault(ent, set()).add(val.lower())
    ok = all(val.lower() in view.get(person, set())
             or any(val.lower() in v for v in view.get(person, set()))
             for person, attrs in p.solution.items()
             for val in attrs.values())
    correct += ok
print(f"\n  LGP-20 from raw prose: certified-solved {solved}/20, "
      f"correct {correct}/20 (0 LLM calls)")
check("raw-prose LGP majority correct", correct >= 12,
      f"{correct}/20")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
