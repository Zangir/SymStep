#!/usr/bin/env python3
"""Retrieval-Augmented Deduction tests. WordNet is local/deterministic;
the Wikidata test tolerates an offline environment. Zero LLM."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphstep.reading.worldmodel import WorldModel
from graphstep.reading.retrieval import find_gaps, _CACHE

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {str(extra)[:110]}")


# 1. the core demo: one text sentence, KB closes the gap
wm = WorldModel.from_text("The aye-aye is a lemur.")
check("gap detector finds query pred + subject types",
      set(find_gaps(wm, "The aye-aye is a mammal")) >= {"mammal", "lemur"},
      find_gaps(wm, "The aye-aye is a mammal"))
check("before retrieval: Unknown",
      wm.ask("The aye-aye is a mammal") == "Unknown")
adm = wm.retrieve_for("The aye-aye is a mammal")
check("wordnet rules admitted", len(adm) >= 3, len(adm))
wm.closure()
exp = wm.ask_explained("The aye-aye is a mammal")
check("verdict True (agreed vocabulary)", exp["verdict"] == "True", exp)
check("tier is retrieved", exp["tier"] == "retrieved", exp["tier"])
check("provenance names wordnet", exp["sources"] == ["wordnet"], exp)
check("proof quotes the text premise",
      "The aye-aye is a lemur." in exp["proof"], exp["proof"][-90:])
check("proof quotes the KB", "[retrieved:wordnet]" in exp["proof"])

# 2. deeper hops through the retrieved chain
check("multi-hop: animal", wm.ask("The aye-aye is an animal") == "True")
check("multi-hop: organism", wm.ask("The aye-aye is an organism") == "True")

# 3. honest negatives survive retrieval
check("reptile stays Unknown", wm.ask("The aye-aye is a reptile") == "Unknown")
check("fish stays Unknown", wm.ask("The aye-aye is a fish") == "Unknown")

# 4a. text authority — DIRECT conflict rejected at admission
wm4 = WorldModel.from_text("Momo is a lemur. Momo is not a primate.",
                           resolve_coref=False)
adm4 = wm4.retrieve_for("Momo is a mammal")
check("direct conflict rejected at admission",
      not any("lemur is a primate" in r.origin for r in adm4)
      and any("REJECTED" in c[0] for c in wm4.contradictions),
      [r.origin for r in adm4])

# 4b. text authority — DERIVED conflict detected at closure, stated wins
wm4b = WorldModel.from_text("Rex is a lemur. Rex is not a mammal.",
                            resolve_coref=False)
wm4b.retrieve_for("Rex is a mammal")
wm4b.closure()
check("stated fact survives the retrieved chain",
      wm4b.ask("Rex is a mammal") == "False")
check("the clash is recorded",
      any("mammal(rex)" in c[0] for c in wm4b.contradictions),
      wm4b.contradictions[:1])

# 5. taint: a STATED text rule fed by retrieved facts yields tier retrieved
wm5 = WorldModel.from_text(
    "The aye-aye is a lemur. If someone is a mammal then they are warm.")
wm5.retrieve_for("The aye-aye is a mammal")
wm5.closure()
exp5 = wm5.ask_explained("The aye-aye is warm")
check("stated rule + retrieved premise -> True at tier retrieved",
      exp5["verdict"] == "True" and exp5["tier"] == "retrieved", exp5)

# 6. induced taint still dominates retrieved (weakest ingredient)
FAM = ("Ann is the parent of Ben. Ben is the parent of Cal. "
       "Ann is the grandparent of Cal. Dee is the parent of Eli. "
       "Eli is the parent of Fay. Dee is the grandparent of Fay. "
       "Gus is the parent of Hal. Hal is the parent of Ida. "
       "Gus is the grandparent of Ida. "
       "Rex is the parent of Sam. Sam is the parent of Tia.")
wm6 = WorldModel.from_text(FAM, resolve_coref=False)
wm6.induce()
wm6.closure(include_induced=True)
check("induced still gates to Likely alongside retrieval machinery",
      wm6.ask("Rex is the grandparent of Tia") == "Likely")

# 7. budget and dedup: repeat retrieval adds nothing; cache is warm
wm.retrieve_for("The aye-aye is a mammal")
origins = [r.origin for r in wm.retrieved_rules]
check("repeat retrieval never duplicates (new gaps may extend upward)",
      len(origins) == len(set(origins)), len(origins) - len(set(origins)))
check("source cache is warm", ("wordnet", "lemur") in _CACHE)

# 8. pure-text answers carry no foreign provenance
exp8 = wm.ask_explained("The aye-aye is a lemur")
check("text fact: True, no sources",
      exp8["verdict"] == "True" and exp8["sources"] == [], exp8)

# 9. Wikidata adapter (tolerates offline)
from graphstep.reading.retrieval import wikidata_knowledge
wd = wikidata_knowledge("lemur")
if wd:
    check("wikidata knowledge well-formed",
          all((getattr(r, "origin", "") or "").startswith(
              "[retrieved:wikidata") for r in wd),
          [getattr(r, "origin", "") for r in wd][:2])
else:
    check("wikidata offline -> graceful empty result (skipped)", True)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
