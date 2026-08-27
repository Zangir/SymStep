#!/usr/bin/env python3
"""General linear-ordering compiler: prose about N objects in a fixed order
-> position variables 1..N + Less/Is constraints.

Vocabulary-agnostic: any antonym scale (left/right, above/below, new/old,
cheap/expensive, first/last, ...) maps onto one abstract position axis.
Orientation per scale is arbitrary but consistent, which is all that
entailment queries need. Zero LLM; unparsed statements are returned so a
caller can escalate them.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

# antonym scales: (low-end stem, high-end stem). Position 1 == low end.
SCALES = [("left", "right"), ("above", "below"), ("first", "last"),
          ("new", "old"), ("cheap", "expensive"), ("young", "old"),
          ("light", "heavy"), ("fast", "slow"), ("early", "late"),
          ("top", "bottom"), ("best", "worst")]
LOW = {a for a, b in SCALES}
HIGH = {b for a, b in SCALES}

ORD_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
             "sixth": 6, "seventh": 7, "eighth": 8}

_LIST_RE = re.compile(
    r"\b(?:two|three|four|five|six|seven|eight|nine)\s+\w+s?:\s*([^.]+)\.")


def _stem_side(word: str) -> Optional[str]:
    """Map a comparative/superlative token to 'low' or 'high'."""
    w = word.lower().rstrip(".,;")
    for suffix in ("most", "est", "er"):
        if w.endswith(suffix):
            w2 = w[: -len(suffix)]
            for cand in (w2, w2 + "e", w2[:-1] if w2[-1:] == w2[-2:-1] else w2):
                if cand in LOW:
                    return "low"
                if cand in HIGH:
                    return "high"
    if w in LOW:
        return "low"
    if w in HIGH:
        return "high"
    return None


def extract_entities(text: str) -> List[str]:
    """'there are five birds: a quail, an owl, ...' -> ['quail', 'owl', ...]"""
    m = _LIST_RE.search(text)
    if not m:
        return []
    items = re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(1))
    out = []
    for it in items:
        it = re.sub(r"^(a|an|the)\s+", "", it.strip().rstrip("."), flags=re.I)
        if it:
            out.append(it.lower())
    return out


def _find_entities(sent: str, entities: List[str]) -> List[str]:
    hits = []
    low = sent.lower()
    for e in sorted(entities, key=len, reverse=True):
        m = re.search(rf"\b{re.escape(e)}\b", low)
        if m:
            hits.append((m.start(), e))
    hits.sort()
    return [e for _, e in hits]


def compile_order_statement(sent: str, entities: List[str],
                            n: int) -> Optional[List[dict]]:
    """One ordering statement -> IR (position vars named by entity)."""
    ents = _find_entities(sent, entities)
    o = sent.strip()
    sl = sent.lower()

    # binary comparative: "X is to the left of Y", "X finished above Y",
    # "X is newer than Y", "X is more/less expensive than Y"
    if len(ents) == 2:
        flip = bool(re.search(r"\b(less|fewer)\b", sl))
        for tok in re.findall(r"[a-z\-]+", sl):
            side = _stem_side(tok)
            if side is None:
                continue
            if flip:
                side = "low" if side == "high" else "high"
            a, b = ents
            if side == "high":
                a, b = b, a
            return [{"type": "less", "a": a, "b": b, "origin": o}]

    if len(ents) != 1:
        return None
    e = ents[0]

    # "second/third from the left|right"
    m = re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth)"
                  r"[\s\-]+from the ([a-z]+)", sl)
    if m:
        k = ORD_WORDS[m.group(1)]
        side = _stem_side(m.group(2))
        if side:
            pos = k if side == "low" else n + 1 - k
            return [{"type": "is", "var": e, "value": pos, "origin": o}]

    # "second-to-last", "third to first"
    m = re.search(r"\b(second|third|fourth|fifth|sixth|seventh)"
                  r"[\s\-]to[\s\-](last|first)\b", sl)
    if m:
        k = ORD_WORDS[m.group(1)]
        pos = (n + 1 - k) if m.group(2) == "last" else k
        return [{"type": "is", "var": e, "value": pos, "origin": o}]

    # "second-newest", "third-cheapest", "second most expensive"
    m = re.search(r"\b(second|third|fourth|fifth|sixth|seventh)"
                  r"[\s\-](?:most\s+|least\s+)?([a-z\-]+)", sl)
    if m:
        k = ORD_WORDS[m.group(1)]
        side = _stem_side(m.group(2))
        if "least" in m.group(0):
            side = {"low": "high", "high": "low", None: None}[side]
        if side:
            pos = k if side == "low" else n + 1 - k
            return [{"type": "is", "var": e, "value": pos, "origin": o}]

    # superlative: "the leftmost", "the newest", "the most expensive",
    # "finished first/last", "the least expensive"
    for m in re.finditer(r"\b(?:(most|least)\s+)?([a-z\-]+)\b", sl):
        side = _stem_side(m.group(2))
        if side is None:
            continue
        word = m.group(2).lower()
        superlative = (word.endswith(("most", "est"))
                       or m.group(1) is not None
                       or word in ("first", "last"))
        if not superlative:
            continue
        if m.group(1) == "least":
            side = "low" if side == "high" else "high"
        pos = 1 if side == "low" else n
        return [{"type": "is", "var": e, "value": pos, "origin": o}]

    # bare ordinal: "finished second", "is second"
    m = re.search(r"\b(?:finished|is|was|came)\s+(second|third|fourth|fifth"
                  r"|sixth|seventh)\b", sl)
    if m:
        return [{"type": "is", "var": e, "value": ORD_WORDS[m.group(1)],
                 "origin": o}]
    return None


def compile_ordering_problem(text: str) -> Tuple[Optional[dict], dict]:
    """Full LD-style paragraph -> IR + coverage report."""
    entities = extract_entities(text)
    report = {"entities": entities, "compiled": 0, "uncovered": []}
    if len(entities) < 2:
        return None, report
    n = len(entities)
    statements = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                  if s.strip() and _find_entities(s, entities)
                  and not _LIST_RE.search(s)]
    variables = {e: list(range(1, n + 1)) for e in entities}
    constraints = [{"type": "alldiff", "vars": entities,
                    "origin": "[frame] fixed order"}]
    for s in statements:
        got = compile_order_statement(s, entities, n)
        if got is None:
            report["uncovered"].append(s)
        else:
            constraints.extend(got)
            report["compiled"] += 1
    return ({"variables": variables, "constraints": constraints}, report)
