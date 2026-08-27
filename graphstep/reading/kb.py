#!/usr/bin/env python3
"""The ONE knowledge store: every piece of domain content in the system is a
row of the same relation — no lexicon files per domain, no tables per task.

    Row(pattern, symbol, sig, payload, provenance, confidence)

Pattern forms (all served by the same match functions):
  - exact word/lemma            "remove"
  - regex over a statement      ("re", r"is directly left of")
  - key tuple                   ("REMOVE", "FIRST")     (assembly blocks)

Symbols are namespaced strings: OP:* (operations), SEL:* (selectors),
TYPE:* (role-noun types), THEME:*, DISCOURSE:* (imperative framing),
CLUE (statement->IR emitters), BLOCK (code fragments with typed holes).

Provenance is mandatory: 'hand' rows are authored; 'wordnet:<synset>' rows
are derived at grounding time and cached back into the store, so retrieved
knowledge and authored knowledge live side by side, distinguishable and
auditable. Nothing in this file knows any benchmark's name (enforced by
test_generality.py).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

NEG_RE = re.compile(r"\b(not|n't|no|never|neither)\b", re.IGNORECASE)

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8,
            "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6}
SMALL_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


@dataclass
class Row:
    pattern: object              # str | ("re", str) | tuple
    symbol: str                  # namespaced meaning symbol
    sig: dict = field(default_factory=dict)
    payload: object = None       # emit fn (CLUE) | code template (BLOCK)
    provenance: str = "hand"
    confidence: float = 1.0

    def __post_init__(self):
        if isinstance(self.pattern, tuple) and self.pattern[0] == "re":
            self._rx = re.compile(self.pattern[1])
        else:
            self._rx = None


KB: List[Row] = []


def add(*rows: Row) -> None:
    KB.extend(rows)


# ------------------------------------------------------------------ lookup

def match_word(word: str, namespace: str) -> Optional[Row]:
    """Exact word row in a namespace; on a miss, WordNet synonym sets are
    searched for a word the store knows — the derived row is cached back
    with the synset as provenance."""
    w = word.lower()
    for r in KB:
        if r.pattern == w and r.symbol.startswith(namespace):
            return r
    if namespace == "OP:":                       # widen verbs via WordNet
        try:
            from .retrieval import _ensure_wordnet
            wn = _ensure_wordnet()
        except Exception:
            return None
        for syn in wn.synsets(w, pos=wn.VERB):
            for other in syn.lemma_names():
                other = other.lower().replace("_", " ")
                for r in KB:
                    if (r.pattern == other and r.symbol.startswith("OP:")
                            and r.provenance == "hand"):
                        derived = Row(w, r.symbol, r.sig, r.payload,
                                      provenance=f"wordnet:{syn.name()}",
                                      confidence=0.9)
                        add(derived)
                        return derived
    return None


def match_statement(text: str) -> List[Row]:
    """All CLUE rows whose regex fires on the statement, in store order."""
    return [r for r in KB if r.symbol == "CLUE" and r._rx
            and r._rx.search(text.lower())]


def match_key(key: tuple) -> Optional[Row]:
    for r in KB:
        if r.pattern == key:
            return r
    return None


# ================================================================== rows
# -- operations (verbs). One canonical family so far; synonyms via rows
#    and WordNet widening.
add(
    Row("remove", "OP:REMOVE", sig={"roles": {"item": "of", "src": "from|in"}}),
    Row("delete", "OP:REMOVE", sig={"roles": {"item": "of", "src": "from|in"}}),
    Row("erase",  "OP:REMOVE", sig={"roles": {"item": "of", "src": "from|in"}}),
    Row("strip",  "OP:REMOVE", sig={"roles": {"item": "of", "src": "from|in"}}),
    Row("drop",   "OP:REMOVE", sig={"roles": {"item": "of", "src": "from|in"}}),
)

# -- selectors: which occurrences an operation applies to
add(
    Row("first", "SEL:FIRST"), Row("last", "SEL:LAST"),
    Row("all", "SEL:ALL"), Row("every", "SEL:ALL"),
)

# -- role nouns -> the argument type that can play them
add(
    Row("character", "TYPE:CHAR"), Row("char", "TYPE:CHAR"),
    Row("letter", "TYPE:CHAR"),
    Row("string", "TYPE:STRING"), Row("word", "TYPE:STRING"),
    Row("sentence", "TYPE:STRING"),
    Row("list", "TYPE:LIST"), Row("array", "TYPE:LIST"),
    Row("number", "TYPE:INT"), Row("integer", "TYPE:INT"),
)

# -- theme nouns that mean "positions where something appears"
add(
    Row("occurrence", "THEME:OCCURRENCE"), Row("instance", "THEME:OCCURRENCE"),
    Row("appearance", "THEME:OCCURRENCE"),
)

# -- imperative framing ("write a function to X"): discourse, not semantics
add(
    Row("write", "DISCOURSE:WRAPPER"), Row("create", "DISCOURSE:WRAPPER"),
    Row("define", "DISCOURSE:WRAPPER"), Row("implement", "DISCOURSE:WRAPPER"),
    Row("make", "DISCOURSE:WRAPPER"),
)

# -- assembly blocks: (operation, selector) -> code fragment. Fragments
#    operate on the accumulator `s`; `{item}` is the typed hole. The
#    renderer supplies `s = <src>` before and `return s` after.
add(
    Row(("REMOVE", "FIRST"), "BLOCK", sig={"needs": {"item": "ELEM"}},
        payload="i = s.find({item})\nif i != -1:\n    s = s[:i] + s[i+1:]"),
    Row(("REMOVE", "LAST"), "BLOCK", sig={"needs": {"item": "ELEM"}},
        payload="j = s.rfind({item})\nif j != -1:\n    s = s[:j] + s[j+1:]"),
    Row(("REMOVE", "ALL"), "BLOCK", sig={"needs": {"item": "ELEM"}},
        payload="s = s.replace({item}, '')"),
)


# ------------------------------------------------------- CLUE emitters
# ctx: {"values": [...], "entities": [...], "text": str, "negated": bool,
#       "origin": str, "value_attr": {value: group}}
# Each emitter returns a list of IR dicts, or None if its extra conditions
# fail (then the next matching row is tried).

def _two(ctx):
    v = ctx["values"]
    return (v[0], v[1]) if len(v) >= 2 else (None, None)


def _emit_directly_left(ctx, m):
    a, b = _two(ctx)
    if a is None: return None
    return [{"type": "offset", "a": b, "b": a, "k": 1, "origin": ctx["origin"]}]


def _emit_directly_right(ctx, m):
    a, b = _two(ctx)
    if a is None: return None
    return [{"type": "offset", "a": a, "b": b, "k": 1, "origin": ctx["origin"]}]


def _emit_somewhere_left(ctx, m):
    a, b = _two(ctx)
    if a is None: return None
    return [{"type": "less", "a": a, "b": b, "origin": ctx["origin"]}]


def _emit_somewhere_right(ctx, m):
    a, b = _two(ctx)
    if a is None: return None
    return [{"type": "less", "a": b, "b": a, "origin": ctx["origin"]}]


def _emit_next_to(ctx, m):
    a, b = _two(ctx)
    if a is None: return None
    return [{"type": "absdiff", "a": a, "b": b, "k": 1, "origin": ctx["origin"]}]


def _emit_between(ctx, m):
    a, b = _two(ctx)
    k = SMALL_NUM.get(m.group(1))
    if a is None or k is None: return None
    return [{"type": "absdiff", "a": a, "b": b, "k": k + 1,
             "origin": ctx["origin"]}]


def _emit_ordinal_pos(ctx, m):
    if not ctx["values"]: return None
    pos = ORDINALS.get(m.group(1))
    if pos is None:
        digits = re.sub(r"[^0-9]", "", m.group(1))
        pos = int(digits) if digits else None
    if pos is None: return None
    typ = "is_not" if "not in" in ctx["text"].lower() else "is"
    return [{"type": typ, "var": ctx["values"][0], "value": pos,
             "origin": ctx["origin"]}]


def _emit_colocate(ctx, m):
    """Anchored co-location: the LEFTMOST value mention (the grammatical
    subject in SVO prose) anchors; every other value is stated to share
    (or, under negation, not share) its holder. Guarded: an anchor/value
    pair from the SAME group never compiles here (the statement is then
    about something this row can't express)."""
    vals = ctx["values"]
    if ctx["entities"] or len(vals) < 2:
        return None
    anchor, rest = vals[0], vals[1:]
    if any(ctx["value_attr"].get(anchor) == ctx["value_attr"].get(v)
           for v in rest):
        return None
    typ = "diff" if ctx["negated"] else "same"
    return [{"type": typ, "a": anchor, "b": v, "origin": ctx["origin"]}
            for v in rest]


def _emit_entity_assign(ctx, m):
    """One named entity + value mention(s) -> is / is_not per value."""
    if len(ctx["entities"]) != 1 or not ctx["values"]:
        return None
    typ = "is_not" if ctx["negated"] else "is"
    return [{"type": typ, "var": v, "value": ctx["entities"][0],
             "origin": ctx["origin"]} for v in ctx["values"]]


# ---- truth-teller statements: boolean variables declared by the row ------
# Emitters may return a dict {"vars": {...}, "specs": [...], "query": spec}
# instead of a bare spec list; vars are new boolean variables the statement
# licenses (domain [0, 1] = lies / tells the truth).

def _emit_truth(ctx, m):
    x = m.group(1).capitalize()
    return {"vars": {x: [0, 1]},
            "specs": [{"type": "is", "var": x, "value": 1,
                       "origin": ctx["origin"]}]}


def _emit_lie(ctx, m):
    x = m.group(1).capitalize()
    return {"vars": {x: [0, 1]},
            "specs": [{"type": "is", "var": x, "value": 0,
                       "origin": ctx["origin"]}]}


def _emit_says(ctx, m):
    x, y = m.group(1).capitalize(), m.group(2).capitalize()
    y_true = 1 if "truth" in m.group(0) else 0
    return {"vars": {x: [0, 1], y: [0, 1]},
            "specs": [{"type": "iff",
                       "a": {"type": "is", "var": x, "value": 1},
                       "b": {"type": "is", "var": y, "value": y_true},
                       "origin": ctx["origin"]}]}


def _emit_truth_query(ctx, m):
    x = m.group(1).capitalize()
    return {"vars": {x: [0, 1]},
            "query": {"type": "is", "var": x, "value": 1,
                      "origin": ctx["origin"]}}


# order matters: positional readings first, identity/assignment last —
# the same precedence the deterministic compiler used.
add(
    Row(("re", r"is directly left of"), "CLUE", payload=_emit_directly_left),
    Row(("re", r"is directly right of"), "CLUE", payload=_emit_directly_right),
    Row(("re", r"somewhere to the left of"), "CLUE", payload=_emit_somewhere_left),
    Row(("re", r"somewhere to the right of"), "CLUE", payload=_emit_somewhere_right),
    Row(("re", r"are next to each other|is next to"), "CLUE", payload=_emit_next_to),
    Row(("re", r"there (?:is|are) (\w+) houses? between"), "CLUE", payload=_emit_between),
    Row(("re", r"\bin the (\w+) house\b"), "CLUE", payload=_emit_ordinal_pos),
    Row(("re", r"^(\w+) tells the truth\.?$"), "CLUE", payload=_emit_truth),
    Row(("re", r"^(\w+) lies\.?$"), "CLUE", payload=_emit_lie),
    Row(("re", r"^(\w+) says (\w+) tells the truth\.?$"), "CLUE",
        payload=_emit_says),
    Row(("re", r"^(\w+) says (\w+) lies\.?$"), "CLUE", payload=_emit_says),
    Row(("re", r"^does (\w+) tell the truth\?$"), "CLUE",
        payload=_emit_truth_query),
    Row(("re", r"."), "CLUE", payload=_emit_entity_assign),
    Row(("re", r"."), "CLUE", payload=_emit_colocate),
)


# ================================================================ narrative
# Event rows: verb families -> fluent edge operations on the interval-
# stamped story state (narrative.py implements the operations; the rows
# here are the knowledge). Row ORDER within a symbol family matters and is
# fixed by EVENT_ORDER in narrative.py.
add(
    Row(("re", r"^(.+?) (?:went back|went|journeyed|travelled|moved|ran|"
               r"walked|hurried) (?:back )?to the (\w+)\.?$"), "EVENT:MOVE"),
    Row(("re", r"^(.+?) (?:got|grabbed|took|picked up) the (\w+)"
               r"(?: there)?\.?$"), "EVENT:ACQUIRE"),
    Row(("re", r"^(.+?) (?:dropped|discarded|put down|left) the (\w+)"
               r"(?: there)?\.?$"), "EVENT:RELEASE"),
    Row(("re", r"^(.+?) (?:gave|handed|passed) the (\w+) to (\w+)\.?$"),
        "EVENT:TRANSFER"),
    Row(("re", r"^(\w+) is (?:no longer|not) in the (\w+)\.?$"),
        "EVENT:NEG_LOC"),
    Row(("re", r"^(\w+) is either in the (\w+) or the (\w+)\.?$"),
        "EVENT:EITHER_LOC"),
    Row(("re", r"^(\w+) is (white|gray|grey|green|yellow|red|blue)\.?$"),
        "EVENT:COLOR"),
    Row(("re", r"^(\w+) is a (\w+)\.?$"), "EVENT:ISA"),
    Row(("re", r"^(\w+) is in the (\w+)\.?$"), "EVENT:LOCATED"),
    Row(("re", r"^(\w+) are afraid of (\w+)\.?$"), "EVENT:AFRAID"),
    Row(("re", r"^the (\w+) is (north|south|east|west) of the (\w+)\.?$"),
        "EVENT:COMPASS"),
    Row(("re", r"^the (.+?) is (to the left of|to the right of|above|below) "
               r"the (.+?)\.?$"), "EVENT:SPATIAL"),
    Row(("re", r"^the (.+?) fits in(?:side)? the (.+?)\.?$"), "EVENT:FITS"),
    Row(("re", r"^the (.+?) is bigger than the (.+?)\.?$"), "EVENT:BIGGER"),
)

# Query rows: question shapes -> value/entailment queries over the state.
add(
    Row(("re", r"^where is (?:the )?(\w+)$"), "QUERY:WHERE_IS"),
    Row(("re", r"^where was (?:the )?(\w+) before the (\w+)$"),
        "QUERY:WHERE_BEFORE"),
    Row(("re", r"^is (\w+) in the (\w+)$"), "QUERY:IS_IN"),
    Row(("re", r"^how many objects is (\w+) (?:carrying|holding)$"),
        "QUERY:COUNT_CARRY"),
    Row(("re", r"^what is (\w+) (?:carrying|holding)$"), "QUERY:WHAT_CARRY"),
    Row(("re", r"^what did (\w+) give to (\w+)$"), "QUERY:GIVE_WHAT"),
    Row(("re", r"^who gave the (\w+) to (\w+)$"), "QUERY:GIVE_WHO"),
    Row(("re", r"^who did (\w+) give the (\w+) to$"), "QUERY:GIVE_RECV"),
    Row(("re", r"^who received the (\w+)$"), "QUERY:RECEIVED"),
    Row(("re", r"^who gave the (\w+)$"), "QUERY:GAVE"),
    Row(("re", r"^what is (?:the )?(\w+) (north|south|east|west) of$"),
        "QUERY:COMPASS_OF"),
    Row(("re", r"^what is (north|south|east|west) of the (\w+)$"),
        "QUERY:OF_COMPASS"),
    Row(("re", r"^what is (\w+) afraid of$"), "QUERY:AFRAID_OF"),
    Row(("re", r"^what color is (\w+)$"), "QUERY:WHAT_COLOR"),
    Row(("re", r"^is the (.+?) (to the left of|to the right of|above|below) "
               r"the (.+)$"), "QUERY:SPATIAL_CLAIM"),
    Row(("re", r"^does the (.+?) fit in(?:side)? the (.+)$"),
        "QUERY:FITS_CLAIM"),
    Row(("re", r"^is the (.+?) bigger than the (.+)$"), "QUERY:BIGGER_CLAIM"),
    Row(("re", r"^how do you go from the (\w+) to the (\w+)$"), "QUERY:PATH"),
    Row(("re", r"^where will (\w+) go$"), "QUERY:WHERE_WILL"),
    Row(("re", r"^why did (\w+) go to the (\w+)$"), "QUERY:WHY_GO"),
    Row(("re", r"^why did (\w+) (?:get|grab|take|pick up) the (\w+)$"),
        "QUERY:WHY_GET"),
)

# time expressions: coarse rank on a narrative day (generic English)
TIME_RANK = {"yesterday": 0, "this morning": 1, "this afternoon": 2,
             "this evening": 3}
# irregular English number words / plurals (generic morphology data)
NUM_WORDS = ["none", "one", "two", "three", "four", "five", "six"]
IRREGULAR_SINGULAR = {"wolves": "wolf", "mice": "mouse", "sheep": "sheep",
                      "cats": "cat", "lions": "lion", "swans": "swan",
                      "frogs": "frog", "rhinos": "rhino"}


def match_symbol(text: str, namespace: str):
    """All rows of a namespace whose regex fires, with their matches."""
    out = []
    for r in KB:
        if r.symbol.startswith(namespace) and r._rx:
            m = r._rx.match(text.lower().strip())
            if m:
                out.append((r, m))
    return out


def ground_statement(text: str, ctx: dict):
    """Run the statement through the CLUE rows in store order; the first
    emitter that reads it wins. Returns a spec list, a dict
    {"vars":..., "specs":..., "query":...}, or None (unread)."""
    ctx = dict(ctx)
    ctx["text"] = text
    ctx["origin"] = text.strip()
    ctx["negated"] = bool(NEG_RE.search(text))
    for row in match_statement(text):
        out = row.payload(ctx, row._rx.search(text.lower()))
        if out is not None and out != []:
            return out
    return None
