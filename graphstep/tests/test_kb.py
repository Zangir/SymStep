#!/usr/bin/env python3
"""Tests for the persistent knowledge store and the generalized Wikidata
adapter. Store tests are deterministic (temp DB); network tests tolerate
offline. Zero LLM."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import graphstep.reading.kbstore as kbstore
from graphstep.reading.kbstore import KBStore
from graphstep.reading.worldmodel import WorldModel, Rule, Literal
import graphstep.reading.retrieval as retrieval

PASS = FAIL = SKIP = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {str(extra)[:110]}")


# use an isolated temp store for ALL tests (never the shared cache file)
tmp = tempfile.mktemp(suffix=".sqlite")
kbstore.use(tmp)

# ---------------- 1. store round-trips -------------------------------------
r = Rule([Literal(True, "?", "lemur", None)],
         Literal(True, "?", "primate", None), "[retrieved:wordnet] test")
f = Literal(True, "marie", "occupation", "physicist")
kbstore.store(tmp).put("src", "lemur", [r, f])
back = kbstore.store(tmp).get("src", "lemur")
check("rule round-trips", any(isinstance(x, Rule)
                              and repr(x) == repr(r) for x in back), back)
check("fact round-trips", any(isinstance(x, Literal)
                              and x.key() == f.key() for x in back), back)
check("unknown term -> None (never fetched)",
      kbstore.store(tmp).get("src", "nope") is None)
kbstore.store(tmp).put("src", "empty_term", [])
check("negative caching: [] round-trips",
      kbstore.store(tmp).get("src", "empty_term") == [])

# ---------------- 2. persistence across instances (reopen file) ------------
kbstore._STORE = None
again = kbstore.store(tmp)
check("persists across store instances",
      any(isinstance(x, Rule) for x in again.get("src", "lemur")))
check("size counts non-empty rows", again.size() == 2, again.size())

# ---------------- 3. store serves when the source is DEAD ------------------
retrieval._CACHE.clear()
rules1 = retrieval.wordnet_rules("lemur")          # fills store via nltk
check("wordnet fetch produced rules", len(rules1) >= 3, len(rules1))
retrieval._CACHE.clear()                            # cold L1
real_ensure = retrieval._ensure_wordnet
retrieval._ensure_wordnet = lambda: (_ for _ in ()).throw(RuntimeError)
try:
    rules2 = retrieval.wordnet_rules("lemur")       # must come from SQLite
finally:
    retrieval._ensure_wordnet = real_ensure
check("served from persistent store with source dead",
      [x.origin for x in rules2] == [x.origin for x in rules1])

# ---------------- 4. generalized Wikidata adapter (offline-tolerant) -------
wd = retrieval.wikidata_knowledge("Marie Curie")
online = bool(wd)
if online:
    facts = [x for x in wd if isinstance(x, Literal)]
    rules = [x for x in wd if isinstance(x, Rule)]
    check("binary facts use property LABELS as predicates",
          any(x.pred == "occupation" and x.obj == "physicist"
              for x in facts), [(x.pred, x.obj) for x in facts][:6])
    check("classy property also yields unary type fact",
          any(x.pred == "physicist" and x.obj is None for x in facts))
    check("instance-of arrives as a rule",
          all("every" in r.origin for r in rules))
    check("provenance carries the QID",
          any("wikidata:Q7186" in getattr(x, "origin", "") for x in rules)
          or any("Q7186" in getattr(x, "origin", "") for x in rules),
          [r.origin for r in rules][:2])

    # ------------ 5. end-to-end person demo --------------------------------
    wm = WorldModel.from_text("Marie Curie discovered polonium.",
                              resolve_coref=False)
    check("before: Unknown", wm.ask("Marie Curie is a chemist") == "Unknown")
    wm.retrieve_for("Marie Curie is a chemist", sources=("wikidata",),
                    link_entities=True)
    wm.closure()
    exp = wm.ask_explained("Marie Curie is a chemist")
    check("person question answered True via generalized adapter",
          exp["verdict"] == "True", exp)
    check("tier retrieved + wikidata provenance",
          exp["tier"] == "retrieved" and "wikidata" in exp["sources"], exp)

    # ------------ 6. opt-in enforcement ------------------------------------
    wm6 = WorldModel.from_text("Marie Curie discovered polonium.",
                               resolve_coref=False)
    wm6.retrieve_for("Marie Curie is a chemist", sources=("wikidata",))
    wm6.closure()
    check("named entity NOT fetched by default (opt-in honored)",
          wm6.ask("Marie Curie is a chemist") == "Unknown",
          wm6.ask("Marie Curie is a chemist"))

    # ------------ 7. text authority over retrieved facts -------------------
    wm7 = WorldModel.from_text(
        "Marie Curie is not a chemist. Marie Curie discovered polonium.",
        resolve_coref=False)
    wm7.retrieve_for("Marie Curie is a chemist", sources=("wikidata",),
                     link_entities=True)
    wm7.closure()
    check("text beats the KB on conflict",
          wm7.ask("Marie Curie is a chemist") == "False")
    check("the clash is on record",
          any("chemist" in c[0] for c in wm7.contradictions),
          wm7.contradictions[:1])

    # ------------ 8. self-growing store: works with the network DEAD -------
    import requests
    retrieval._CACHE.clear()
    real_get = requests.get
    requests.get = lambda *a, **k: (_ for _ in ()).throw(RuntimeError)
    try:
        wm8 = WorldModel.from_text("Marie Curie discovered polonium.",
                                   resolve_coref=False)
        wm8.retrieve_for("Marie Curie is a chemist", sources=("wikidata",),
                         link_entities=True)
        wm8.closure()
        check("OFFLINE re-run answers from the local store",
              wm8.ask("Marie Curie is a chemist") == "True",
              wm8.ask("Marie Curie is a chemist"))
    finally:
        requests.get = real_get
    check("store grew (self-growing KB)", kbstore.store(tmp).size() > 10,
          kbstore.store(tmp).size())
else:
    SKIP += 6
    print("  skip Wikidata-dependent tests (offline)")

os.unlink(tmp)
print(f"\n{PASS} passed, {FAIL} failed" + (f", {SKIP} skipped" if SKIP else ""))
sys.exit(1 if FAIL else 0)
