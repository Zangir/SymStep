#!/usr/bin/env python3
"""Text -> IR compilers (Tier 0: deterministic templates).

Any clue a template cannot compile is returned as `None` so the pipeline can
escalate that clue (and only that clue) to the LLM tier. The encoding for
assignment/position puzzles is the classic value-position model: every
attribute VALUE is a variable whose domain is the entity/house index set;
each attribute contributes one AllDifferent. Entities referenced by name are
constants (their index), which makes all common clue archetypes unary/binary.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

NEG_RE = re.compile(r"\b(not|n't|no|never|neither)\b", re.IGNORECASE)

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
            "6th": 6}
SMALL_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _surface_variants(v: str) -> set:
    """Lexical variants a clue may use for a canonical value: CamelCase
    respaced ('SamsungGalaxyS21' -> 'samsung galaxy s21'), plural, 3rd-person
    verb, -ing forms (with consonant doubling / silent-e), adjectival -ish
    ('brit' -> 'british', 'swede' -> 'swedish'), and spaced letter-digit
    boundaries ('Week1' -> 'week 1')."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", v).lower()
    bases = {v.lower(), spaced,
             re.sub(r"(?<=[a-z])(?=[0-9])", " ", v.lower()),
             re.sub(r"(?<=[a-z])(?=[0-9])", " ", spaced),
             re.sub(r"(?<=[a-z])(?=[0-9])", "-", v.lower()),
             re.sub(r"(?<=[a-z])(?=[0-9])", "-", spaced)}
    for s in list(bases):                           # hyphen <-> space forms
        if " " in s:
            bases.add(s.replace(" ", "-"))
        if "-" in s:
            bases.add(s.replace("-", " "))
    out = set(bases)
    for s in bases:
        if len(s) < 2:
            continue
        out.add(s + "s")
        out.add(s + "es")
        out.add(s + "ish")                          # brit -> british
        if s.endswith("y") and len(s) > 2:
            out.add(s[:-1] + "ies")                 # cherry -> cherries
        if s.endswith("s") and len(s) > 3:
            out.add(s[:-1])                         # roses -> rose
        if s.endswith("e"):
            out.add(s[:-1] + "ing")                 # dance -> dancing
            out.add(s[:-1] + "ish")                 # swede -> swedish
        else:
            out.add(s + "ing")                      # cook -> cooking
            out.add(s + s[-1] + "ing")              # swim -> swimming
        if s.endswith("ing") and len(s) > 4:
            stem = s[:-3]
            out |= {stem, stem + "s", stem + "e", stem + "es"}
            if len(stem) > 2 and stem[-1] == stem[-2]:
                out |= {stem[:-1], stem[:-1] + "s"} # swimming -> swim
    return out


class Inventory:
    """Entities (index-valued constants) + attribute values (variables)."""

    def __init__(self, entities: List[str], attributes: Dict[str, List[str]],
                 aliases: Optional[Dict[str, List[str]]] = None):
        self.entities = list(entities)
        self.attributes = {a: list(vs) for a, vs in attributes.items()}
        self.aliases = aliases or {}
        self.entity_index = {e.lower(): i + 1 for i, e in enumerate(entities)}
        self.value_attr: Dict[str, str] = {}
        for a, vs in attributes.items():
            for v in vs:
                self.value_attr[v] = a
        # mention table: surface form -> ("entity", idx) | ("value", name)
        # Single-char surfaces match case-sensitively (value 'A' vs article
        # 'a'); everything else matches case-insensitively via variants.
        surface_map: Dict[str, Tuple[str, object]] = {}
        clashes = set()

        def register(surface: str, ref: Tuple[str, object]):
            prev = surface_map.get(surface)
            if prev is not None and prev != ref:
                clashes.add(surface)
            else:
                surface_map[surface] = ref

        for e in entities:
            for s in _surface_variants(e):
                register(s, ("entity", self.entity_index[e.lower()]))
        for v in self.value_attr:
            if len(v) == 1:
                register(v, ("value", v))           # case-sensitive literal
            else:
                for s in _surface_variants(v):
                    register(s, ("value", v))
                for raw in self.aliases.get(v, []): # raw surface forms
                    for s in _surface_variants(raw):
                        register(s, ("value", v))
        for s in clashes:                           # ambiguous surfaces are unusable
            surface_map.pop(s, None)
        self.mentions = sorted(surface_map.items(), key=lambda m: -len(m[0]))

    def variables(self) -> Dict[str, list]:
        n = len(self.entities)
        return {v: list(range(1, n + 1)) for v in self.value_attr}

    def base_constraints(self) -> List[dict]:
        return [{"type": "alldiff", "vars": vs, "origin": f"[{a}] values are distinct"}
                for a, vs in self.attributes.items()]

    def find_mentions(self, text: str) -> List[Tuple[str, object]]:
        """Non-overlapping longest-first mention scan, left-to-right order.
        Single-char surfaces match case-sensitively; others case-insensitively.
        The same referent is reported once."""
        tl = text.lower()
        hits, spans, seen = [], [], set()
        for surface, ref in self.mentions:
            haystack = text if len(surface) == 1 else tl
            for m in re.finditer(rf"\b{re.escape(surface)}\b", haystack):
                span = (m.start(), m.end())
                if any(s0 < span[1] and span[0] < s1 for s0, s1 in spans):
                    continue
                spans.append(span)
                if ref in seen:
                    continue
                seen.add(ref)
                hits.append((m.start(), ref))
        # last-resort prefix pass: 'sept' -> 'september', 'feb' -> 'february'
        for surface, ref in self.mentions:
            if ref in seen or len(surface) < 3 or " " in surface:
                continue
            matches = list(re.finditer(rf"\b{re.escape(surface)}[a-z]+\b", tl))
            if len(matches) != 1:
                continue
            span = (matches[0].start(), matches[0].end())
            if any(s0 < span[1] and span[0] < s1 for s0, s1 in spans):
                continue
            word = matches[0].group(0)
            rivals = [s for s, r in self.mentions
                      if r != ref and word.startswith(s) and len(s) >= 3]
            if rivals:
                continue                            # ambiguous prefix: skip
            spans.append(span)
            seen.add(ref)
            hits.append((span[0], ref))
        hits.sort()
        return [ref for _, ref in hits]


# ---------------------------------------------------------------- assignment clues
def compile_assignment_clue(clue: str, inv: Inventory) -> Optional[List[dict]]:
    """LGP/SP-style clue -> IR constraints, or None if not template-parseable.

    Semantics by mention pattern:
      entity + value(s)           -> is / is_not per value
      two values, no entity       -> same / diff (same holder)
    """
    refs = inv.find_mentions(clue)
    entities = [r[1] for r in refs if r[0] == "entity"]
    values = [r[1] for r in refs if r[0] == "value"]
    negated = bool(NEG_RE.search(clue))
    o = clue.strip()

    if len(entities) == 1 and len(values) >= 1:
        typ = "is_not" if negated else "is"
        return [{"type": typ, "var": v, "value": entities[0], "origin": o}
                for v in values]
    if len(entities) == 0 and len(values) == 2:
        a, b = values
        if inv.value_attr[a] == inv.value_attr[b]:
            return None                    # same attribute: template unsure
        return [{"type": "diff" if negated else "same", "a": a, "b": b,
                 "origin": o}]
    return None


# ---------------------------------------------------------------- positional clues
def compile_position_clue(clue: str, inv: Inventory) -> Optional[List[dict]]:
    """ZebraLogic-style positional clue -> IR, or None.

    Handles: directly left/right, somewhere left/right, next to, N houses
    between, fixed/negated ordinal position, and same-house identity.
    A referenced entity name is itself a variable here (Name values are
    variables like any other attribute value).
    """
    cl = clue.lower().strip()
    o = clue.strip()
    refs = [r[1] for r in inv.find_mentions(clue) if r[0] == "value"]

    def two():
        return refs[0], refs[1] if len(refs) > 1 else None

    if re.search(r"is directly left of", cl) and len(refs) >= 2:
        a, b = two()
        return [{"type": "offset", "a": b, "b": a, "k": 1, "origin": o}]
    if re.search(r"is directly right of", cl) and len(refs) >= 2:
        a, b = two()
        return [{"type": "offset", "a": a, "b": b, "k": 1, "origin": o}]
    if re.search(r"somewhere to the left of", cl) and len(refs) >= 2:
        a, b = two()
        return [{"type": "less", "a": a, "b": b, "origin": o}]
    if re.search(r"somewhere to the right of", cl) and len(refs) >= 2:
        a, b = two()
        return [{"type": "less", "a": b, "b": a, "origin": o}]
    if re.search(r"are next to each other|is next to", cl) and len(refs) >= 2:
        a, b = two()
        return [{"type": "absdiff", "a": a, "b": b, "k": 1, "origin": o}]
    m = re.search(r"there (?:is|are) (\w+) houses? between", cl)
    if m and len(refs) >= 2:
        k = SMALL_NUM.get(m.group(1), None)
        if k is not None:
            a, b = two()
            return [{"type": "absdiff", "a": a, "b": b, "k": k + 1, "origin": o}]
    m = re.search(r"\bin the (\w+) house\b", cl)
    if m and refs:
        pos = ORDINALS.get(m.group(1))
        if pos is None:
            digits = re.sub(r"[^0-9]", "", m.group(1))
            pos = int(digits) if digits else None
        if pos is not None:
            typ = "is_not" if "not in" in cl else "is"
            return [{"type": typ, "var": refs[0], "value": pos, "origin": o}]
    if len(refs) >= 2 and not any(kw in cl for kw in
                                  ("left", "right", "next to", "between",
                                   "house")):
        neg = bool(NEG_RE.search(cl))
        return [{"type": "diff" if neg else "same", "a": refs[0], "b": refs[1],
                 "origin": o}]
    return None


def compile_clues(clues: List[str], inv: Inventory,
                  positional: bool) -> Tuple[List[dict], List[str]]:
    """Tier-0 pass. Returns (ir_constraints, unparsed_clues)."""
    out, unparsed = [], []
    for clue in clues:
        specs = (compile_position_clue(clue, inv) if positional
                 else compile_assignment_clue(clue, inv))
        if specs is None and positional:
            specs = compile_assignment_clue(clue, inv)   # identity fallback
        if specs is None:
            unparsed.append(clue)
        else:
            out.extend(specs)
    return out, unparsed
