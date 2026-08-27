#!/usr/bin/env python3
"""Tests for the induction pipeline: mine -> verify -> tier -> gated answers.
All shape-based and domain-agnostic; zero LLM."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphstep.reading.worldmodel import WorldModel

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {str(extra)[:110]}")


FAMILIES = ("Ann is the parent of Ben. Ben is the parent of Cal. "
            "Ann is the grandparent of Cal. "
            "Dee is the parent of Eli. Eli is the parent of Fay. "
            "Dee is the grandparent of Fay. "
            "Gus is the parent of Hal. Hal is the parent of Ida. "
            "Gus is the grandparent of Ida. ")

# 1. subsumption mined, direction disciplined by a counterexample
wm = WorldModel.from_text(
    "Momo is a lemur. Kiki is a lemur. Zaza is a lemur. Bobo is a monkey. "
    "Momo is a primate. Kiki is a primate. Zaza is a primate. "
    "Bobo is a primate. Bobo is not a lemur.", resolve_coref=False)
kept = wm.induce()
descs = [r.origin for r in kept]
check("S1 mined", any("lemur(x) => primate(x)" in d for d in descs), descs)
check("reverse blocked by observed counterexample",
      not any("primate(x) => lemur(x)" in d for d in descs), descs)

# 2. kinship: chain + inverse mined; unlabeled pair neutral under OWA
wm = WorldModel.from_text(
    FAMILIES + "Rex is the parent of Sam. Sam is the parent of Tia.",
    resolve_coref=False)
kept = wm.induce(assume_closed=False)
check("S4 chain mined despite unlabeled test pair (OWA-neutral)",
      any("grandparent" in r.origin and "S4" in r.origin for r in kept),
      [r.origin for r in kept])
wm_closed = WorldModel.from_text(
    FAMILIES + "Rex is the parent of Sam. Sam is the parent of Tia.",
    resolve_coref=False)
kept_closed = wm_closed.induce(assume_closed=True)
check("closed-world semantics blocks it (parameter works)",
      not any("grandparent" in r.origin and "S4" in r.origin
              for r in kept_closed))

# 3. tier gate: induced conclusions answer "Likely", never "True";
#    certified answers untouched
q = "Rex is the grandparent of Tia"
check("before closure: Unknown", wm.ask(q) == "Unknown")
wm.closure(include_induced=False)
check("stated-only closure does not fire induced rules",
      wm.ask(q) == "Unknown")
new = wm.closure(include_induced=True)
check("induced rule derives the new fact",
      ("rex", "grandparent_of", "tia") in wm.facts, [str(l) for l, _ in new])
check("gated verdict is Likely, not True", wm.ask(q) == "Likely", wm.ask(q))
check("stated facts still answer True",
      wm.ask("Rex is the parent of Sam") == "True")
check("proof discloses induction",
      "[induced:S4]" in wm.facts[("rex", "grandparent_of", "tia")][1])

# 4. tier propagation: stated rule fed by an induced premise stays 'induced'
wm4 = WorldModel.from_text(
    FAMILIES + "Rex is the parent of Sam. Sam is the parent of Tia. "
    "If A is a grandparent of B then A is an elder of B.",
    resolve_coref=False)
wm4.induce()
wm4.closure(include_induced=True)
check("induced taint propagates through stated rules",
      wm4.fact_tier.get(("rex", "elder_of", "tia")) == "induced",
      wm4.fact_tier.get(("rex", "elder_of", "tia")))
check("...and its verdict is Likely",
      wm4.ask("Rex is an elder of Tia") == "Likely")

# 5a. counterexample PRESENT at mining time: blocked at the mine stage
wm5 = WorldModel.from_text(
    FAMILIES + "Ned is the parent of Ott. Ott is the parent of Pia. "
    "Ned is not the grandparent of Pia.", resolve_coref=False)
check("mine-stage counterexample blocks the rule",
      not any("grandparent" in r.origin and "S4" in r.origin
              for r in wm5.induce()))

# 5b. counterexample arriving LATER: verify() retracts the living rule
from graphstep.reading.worldmodel import Literal
from graphstep.reading.induction import verify_rules
wm5b = WorldModel.from_text(FAMILIES, resolve_coref=False)
kept5b = wm5b.induce()
assert any("grandparent" in r.origin for r in kept5b)
wm5b.commit(Literal(False, "ann", "grandparent_of", "cal"),
            "New evidence: Ann is not the grandparent of Cal.",
            tier="agree")   # the world grew and now contradicts the rule
surviving, retracted = verify_rules(wm5b, wm5b.induced_impl)
check("late counterexample retracts via verify",
      retracted and not any("grandparent" in r.origin for r in surviving),
      retracted)

# 6. support threshold + stated-rule dedup
wm6 = WorldModel.from_text(
    "Al is a cook. Bo is a cook. Al is tall. Bo is tall.",
    resolve_coref=False)
check("below support: nothing mined", not wm6.induce())
wm7 = WorldModel.from_text(
    "Momo is a lemur. Kiki is a lemur. Zaza is a lemur. Momo is a primate. "
    "Kiki is a primate. Zaza is a primate. Lemurs are primates.",
    resolve_coref=False)
kept7 = wm7.induce()
check("stated rule not re-mined",
      not any("lemur(x) => primate(x)" in r.origin for r in kept7),
      [r.origin for r in kept7])

# 7. exclusion mining recorded
wm8 = WorldModel.from_text(
    "Rex is a mammal. Sam is a mammal. Tia is a mammal. "
    "Kip is a reptile. Lop is a reptile. Nub is a reptile.",
    resolve_coref=False)
wm8.induce()
check("exclusion induced", any("mutually exclusive" in r
                               for r in wm8.induced_rules),
      wm8.induced_rules)

# 8. optimized closure: join grounding handles a 200-entity chain fast
names = [f"P{i}" for i in range(200)]
chain = " ".join(f"{names[i]} is the parent of {names[i+1]}."
                 for i in range(199))
rules = (" If A is the parent of B then A is an ancestor of B. "
         "If A is an ancestor of B and B is an ancestor of C then "
         "A is an ancestor of C.")
wm9 = WorldModel.from_text(chain + rules, resolve_coref=False)
t0 = time.time()
new9 = wm9.closure()
dt = time.time() - t0
check(f"200-entity recursive closure ({len(new9)} facts, {dt:.2f}s)",
      ("p0", "ancestor_of", "p199") in wm9.facts and dt < 60, f"{dt:.2f}s")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
