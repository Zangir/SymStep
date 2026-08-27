#!/usr/bin/env python3
"""Cross-domain world-model tests: coreference, quantities, relational
recursion, contradiction detection/resolution, induced integrity rules —
each exercised on a different domain's text. Zero LLM."""
import sys, os
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


# 1. BIOLOGY + coreference: pronouns and demonstratives resolve across sentences
wm = WorldModel.from_text(
    "The aye-aye is a lemur. It is nocturnal. This solitary animal taps on "
    "trees. Lemurs are primates.")
wm.closure()
check("coref: pronoun subject", ("aye-aye", "nocturnal", None) in wm.facts,
      list(wm.facts))
check("coref: demonstrative NP", ("aye-aye", "tap_on", "trees") in wm.facts
      or ("aye-aye", "tap", "trees") in wm.facts
      or ("aye-aye", "tap", "on_trees") in wm.facts, list(wm.facts))
check("chain still derives", ("aye-aye", "primate", None) in wm.facts)

# 2. HISTORY + coreference on people (he/she)
wm = WorldModel.from_text(
    "Marie Curie was a physicist. She was also a chemist. "
    "Isaac Newton was a physicist. He was also a mathematician.")
check("she -> Marie", ("marie_curie", "chemist", None) in wm.facts,
      list(wm.facts))
check("he -> Newton", ("isaac_newton", "mathematician", None) in wm.facts,
      list(wm.facts))

# 3. GEOGRAPHY + quantities: measurement facts and comparisons via pint
wm = WorldModel.from_text(
    "The Nile is 6,682 km long. The Amazon is 6,400 km long. "
    "The Mississippi is 3,766 km long.")
check("quantity stored", ("nile", "length") in wm.quantities,
      wm.quantities)
check("comparison True", wm.ask("Is the Nile longer than the Amazon?")
      == "True")
check("comparison False", wm.ask("Is the Mississippi longer than the "
                                 "Amazon?") == "False")
check("unknown attr honest", wm.ask("Is the Nile older than the Amazon?")
      == "Unknown")

# 4. ASTRONOMY: unit normalization across different units
wm = WorldModel.from_text(
    "Mercury goes around the Sun once every 88 days. "
    "Halley's comet returns every 76 years.")
check("period units normalized",
      ("mercury", "period") in wm.quantities
      and wm.quantities[("mercury", "period")][1] == "second",
      wm.quantities)

# 5. GENEALOGY: relational facts + multi-variable RECURSIVE rules
wm = WorldModel.from_text(
    "Elizabeth is the mother of Charles. Charles is the father of William. "
    "William is the father of George. "
    "If A is the mother of B then A is a parent of B. "
    "If A is the father of B then A is a parent of B. "
    "If A is a parent of B and B is a parent of C then A is a grandparent "
    "of C. If A is a parent of B then A is an ancestor of B. "
    "If A is an ancestor of B and B is an ancestor of C then A is an "
    "ancestor of C.")
new = wm.closure()
check("recursion: 3-hop ancestor",
      ("elizabeth", "ancestor_of", "george") in wm.facts, len(new))
check("recursion: grandparent join",
      ("charles", "grandparent_of", "george") in wm.facts)
check("recursion: direction honest",
      wm.ask("George is an ancestor of Elizabeth") == "Unknown")

# 6. CONTRADICTIONS, four general classes
wm = WorldModel.from_text(
    "The dog is blue. The dog needs the bear. "
    "If someone needs the bear then they are not blue.")
wm.closure()
check("rule-vs-stated detected", len(wm.contradictions) == 1,
      wm.contradictions)

wm = WorldModel.from_text("The tower is tall. The tower is short.",
                          resolve_coref=False)
wm.induce_integrity()
check("antonym exclusion detected", any("antonym" in c[2]
                                        for c in wm.contradictions))

wm = WorldModel.from_text(
    "The Nile is 6,682 km long. The Nile is 5,000 km long.",
    resolve_coref=False)
check("quantity conflict detected", len(wm.contradictions) == 1)

wm = WorldModel.from_text(
    "Anne was born in Paris. Bob was born in Lyon. Carol was born in Nice. "
    "Dave was born in Toulouse. Dave was born in Marseille.",
    resolve_coref=False)
wm.induce_integrity()
check("induced functionality flags double value",
      any("single-valued" in c[1] for c in wm.contradictions),
      wm.contradictions)

# 7. SCALE cycle refuses (Z3 base check)
wm = WorldModel.from_text(
    "The box is bigger than the crate. The crate is bigger than the tub. "
    "The tub is bigger than the box.", resolve_coref=False)
check("scale cycle refused", wm.vector("box", "tub", metric=False) is None
      and wm.contradictions)

# 8. self-probe still clean on a mixed world
wm = WorldModel.from_text(
    "The aye-aye is a lemur. Lemurs are primates. Gertrude is afraid of "
    "wolves. The circuit has electricity.")
wm.closure()
check("self-probe clean", not wm.self_probe(), wm.self_probe())

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
