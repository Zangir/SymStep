#!/usr/bin/env python3
"""Garbage-regression suite: extraction must be correct or ABSTAIN — never
confidently wrong. Specimens are real failures caught during development
(2026-08-20); every parser change must keep garbage at zero here.
Coverage may regress; precision may not."""
import sys, os, re, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphstep.reading.worldmodel import (WorldModel, parse_literal, parse_rules,
                                  Literal)

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}  {str(extra)[:120]}")


def is_garbage(key) -> bool:
    """A fact key that should never exist: globbed predicates/subjects."""
    subj, pred, obj = key
    for part in (subj, pred, obj or ""):
        if len(part) > 40 or "," in part:
            return True
        if re.search(r"(_|^)(is|are|was|were|then|which|that|who)(_|$)", part):
            return True
    if len(pred.split("_")) > 4:
        return True
    return False


def garbage_facts(wm):
    return [k for k in wm.facts if is_garbage(k)]


def garbage_rules(wm):
    bad = []
    for r in wm.rules:
        for lit in r.conds + [r.concl]:
            if is_garbage(lit.key()) or lit.pred in ("thi", "i", "most"):
                bad.append(repr(r))
    return bad


# ---- specimen 1: dense full-English Wikipedia lead (must abstain cleanly)
wiki_lead = """The aye-aye is a long-fingered lemur, a strepsirrhine primate
native to Madagascar with rodent-like teeth that perpetually grow and a
special thin middle finger. It is the world's largest nocturnal primate.
Lemurs are wet-nosed primates of the superfamily Lemuroidea. They are
endemic to the island of Madagascar. Most existing lemurs are small, with a
pointed snout, large eyes, and a long tail."""
wm = WorldModel.from_text(wiki_lead)
check("wiki lead: no garbage facts", not garbage_facts(wm), garbage_facts(wm))
check("wiki lead: no garbage rules", not garbage_rules(wm), garbage_rules(wm))
check("wiki lead: no 'they' subjects",
      not any(k[0].startswith("they") for k in wm.facts))

# ---- specimen 2: the 'is'-as-verb bug ("pred i") must stay dead
lit = parse_literal("The aye-aye is a long-fingered lemur, a strepsirrhine "
                    "primate native to Madagascar")
check("appositive glob abstains", lit is None, lit)

# ---- specimen 3: demonstrative subject must not become a plural rule
rules = parse_rules("This solitary animal is nocturnal.")
check("demonstrative not a rule", not rules, rules)

# ---- specimen 4: NatLang paraphrase — conditions are not facts
nl = ("Charlie is green, but often kind, even when he is blue and cold. "
      "Fred rather resembles the rainbow, as he is green, red and blue.")
wm = WorldModel.from_text(nl)
check("when-clause not a fact",
      ("charlie", "blue", None) not in wm.facts
      and ("charlie", "cold", None) not in wm.facts,
      dict(wm.facts))
check("as-clause IS a fact (asserting)",
      ("fred", "green", None) in wm.facts, dict(wm.facts))
check("natlang: no garbage", not garbage_facts(wm), garbage_facts(wm))

# ---- specimen 5: rule paraphrases stay unread, not facts about 'people'
wm = WorldModel.from_text(
    "Young people who are nice and look round are also going to be green.")
check("rule paraphrase abstains",
      not wm.facts and not wm.rules and len(wm.unread) == 1,
      (dict(wm.facts), wm.rules))

# ---- specimen 6: dual-reader disagreement is recorded, not committed
wm = WorldModel.from_text("The dog needs the bear.")
check("agreed fact commits", ("dog", "need", "bear") in wm.facts)
check("tier recorded", wm.fact_tier.get(("dog", "need", "bear")) == "agree",
      wm.fact_tier)

# ---- specimen 7: self-probe passes on a deduction world
wm = WorldModel.from_text(
    "The aye-aye is a lemur. Lemurs are primates. Primates are mammals. "
    "Gertrude is afraid of wolves. The circuit has electricity.")
wm.closure()
check("taxonomy derives", ("aye-aye", "mammal", None) in wm.facts)
probe_failures = wm.self_probe()
check("self-probe clean", not probe_failures, probe_failures)

# ---- property test: template sentences + noise -> correct or abstain
rng = random.Random(7)
NAMES = ["Anne", "Bob", "Charlie", "Dave", "Erin", "Fiona", "Gary", "Harry"]
ADJS = ["red", "blue", "green", "kind", "quiet", "rough", "furry", "nice"]
VERBS = ["chases", "needs", "likes", "sees", "visits"]
mutations = [
    lambda s: s,                                          # clean
    lambda s: s.replace(".", ", or so it seems."),        # hedge tail
    lambda s: "It is said that " + s[0].lower() + s[1:],  # report frame
    lambda s: s.replace(" is ", " is quite possibly "),   # modal insert
]
prop_fail = 0
for i in range(200):
    n1, n2 = rng.sample(NAMES, 2)
    adj = rng.choice(ADJS)
    verb = rng.choice(VERBS)
    form = rng.choice([f"{n1} is {adj}.", f"{n1} is not {adj}.",
                       f"{n1} {verb} the {n2.lower()}."])
    mut = rng.choice(mutations)
    sent = mut(form)
    wm = WorldModel.from_text(sent)
    clean = mut is mutations[0]
    for key, (pos, _) in wm.facts.items():
        subj, pred, obj = key
        # any committed fact must be a faithful reading of the CLEAN form
        ok = (subj == n1.lower()
              and (pred in (adj, verb.rstrip("s"),
                            verb[:-1] if verb.endswith("es") else verb))
              and not is_garbage(key))
        if clean and not ok:
            prop_fail += 1
            print("  PROP-FAIL", repr(sent), "->", key)
        if not clean and not ok:
            prop_fail += 1
            print("  PROP-FAIL (noisy commit)", repr(sent), "->", key)
check("property: 200 mutated sentences, zero unfaithful commits",
      prop_fail == 0, f"{prop_fail} failures")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
