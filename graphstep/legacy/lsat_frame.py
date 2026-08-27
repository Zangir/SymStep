#!/usr/bin/env python3
"""Deterministic AR-LSAT scheduling-frame compiler (minimal / zero LLM).

Recognizes the "N entities scheduled into day x (morning|afternoon) slots"
problem family, builds the variable frame deterministically, and compiles
condition sentences with the syntax tier (clause skeletons) + LSAT atom
patterns. Answer options in the rigid "Day. slot: Name; ..." format are also
compiled deterministically, so a full question can be answered with ZERO LLM
calls. Anything outside the recognized patterns is returned as unhandled so a
caller can escalate just that piece to the LLM tier.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

DAY_WORDS = ["monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday"]
DAY_ABBR = {"mon": "monday", "tue": "tuesday", "tues": "tuesday",
            "wed": "wednesday", "thu": "thursday", "thur": "thursday",
            "thurs": "thursday", "fri": "friday", "sat": "saturday",
            "sun": "sunday"}
NUM_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10}


class Frame:
    """entities -> slot-code variables. Slot code for (day i, part j):
    2*i + j + 1, with 0 = not scheduled. parts = [morning, afternoon]."""

    def __init__(self, entities: List[str], days: List[str],
                 parts: List[str], n_selected: int):
        self.entities = entities
        self.days = days
        self.parts = parts
        self.n_selected = n_selected
        self.n_slots = len(days) * len(parts)

    def slot(self, day_i: int, part_j: int) -> int:
        return day_i * len(self.parts) + part_j + 1

    def day_slots(self, day_i: int) -> List[int]:
        return [self.slot(day_i, j) for j in range(len(self.parts))]

    def part_slots(self, part_j: int) -> List[int]:
        return [self.slot(i, part_j) for i in range(len(self.days))]

    def variables(self) -> Dict[str, list]:
        dom = list(range(0, self.n_slots + 1))
        return {e: list(dom) for e in self.entities}

    def base_constraints(self) -> List[dict]:
        out = [{"type": "count", "vars": self.entities, "value": 0,
                "op": "==", "k": len(self.entities) - self.n_selected,
                "origin": "[frame] number scheduled"}]
        for s in range(1, self.n_slots + 1):
            out.append({"type": "count", "vars": self.entities, "value": s,
                        "op": "==", "k": 1, "origin": "[frame] one per slot"})
        return out

    def encoding_doc(self) -> str:
        legend = ", ".join(
            f"{self.slot(i, j)}={d.capitalize()} {p}"
            for i, d in enumerate(self.days) for j, p in enumerate(self.parts))
        return f"0=not scheduled, {legend}"


def detect_frame(setup: str) -> Optional[Frame]:
    """Deterministic frame detection for the day/part scheduling family."""
    m = re.search(r"(?:Of the|The) (\w+) \w+[—\-–]+([A-Z][\w, ]+(?:and [A-Z]\w+))[—\-–]+",
                  setup)
    if not m:
        return None
    names = re.split(r",\s*|\s+and\s+", m.group(2))
    names = [re.sub(r"^and\s+", "", n).strip() for n in names]
    entities = [n for n in names if n and n[0].isupper()]
    days = [d for d in DAY_WORDS if re.search(rf"\b{d}\b", setup, re.I)]
    parts = [p for p in ("morning", "afternoon", "evening")
             if re.search(rf"\b{p}\b", setup, re.I)]
    sel = re.search(r"exactly (\w+) (?:of them |students )?(?:will|must)? ?give",
                    setup, re.I)
    n_selected = NUM_WORDS.get(sel.group(1).lower(), None) if sel else None
    if not entities or not days or not parts or n_selected is None:
        return None
    if len(days) * len(parts) != n_selected:
        return None                                  # slots must equal reports
    return Frame(entities, days, parts, n_selected)


# ------------------------------------------------------------------ atoms
def _entity_in(text: str, fr: Frame) -> List[str]:
    return [e for e in fr.entities if re.search(rf"\b{e}\b", text)]


def _day_in(text: str, fr: Frame) -> List[int]:
    return [i for i, d in enumerate(fr.days)
            if re.search(rf"\b{d}\b", text, re.I)]


def _part_in(text: str, fr: Frame) -> List[int]:
    return [j for j, p in enumerate(fr.parts)
            if re.search(rf"\b{p}\b", text, re.I)]


def compile_lsat_atom(text: str, fr: Frame, origin: str) -> Optional[List[dict]]:
    """One atomic clause -> IR over the frame. None if unrecognized."""
    ents = _entity_in(text, fr)
    days = _day_in(text, fr)
    parts = _part_in(text, fr)
    neg = bool(re.search(r"\b(not|cannot|can't|neither|nor|no)\b", text, re.I))
    gives = bool(re.search(r"give|report|scheduled|speak", text, re.I))
    only = bool(re.search(r"\bonly\b", text, re.I))
    if len(ents) != 1:
        return None
    e = ents[0]

    # "<Day> is the only day on which X can ..." -> X in {0} + day slots
    if only and len(days) == 1 and gives:
        allowed = {0, *fr.day_slots(days[0])}
        return [{"type": "is_not", "var": e, "value": s, "origin": origin}
                for s in range(0, fr.n_slots + 1) if s not in allowed]
    # "X cannot give an <part> report" -> forbid that part's slots
    if neg and len(parts) == 1 and gives and not days:
        return [{"type": "is_not", "var": e, "value": s, "origin": origin}
                for s in fr.part_slots(parts[0])]
    # "X's report is given on <Day>" / "X gives a report on <Day>"
    if len(days) == 1 and gives and not only:
        clauses = [{"type": "is", "var": e, "value": s}
                   for s in fr.day_slots(days[0])]
        spec = {"type": "or", "clauses": clauses, "origin": origin}
        return [{"type": "not", "c": spec, "origin": origin}] if neg else [spec]
    # bare "X gives a report" -> scheduled (not 0)
    if gives and not days and not parts:
        spec = {"type": "is_not", "var": e, "value": 0, "origin": origin}
        return [{"type": "is", "var": e, "value": 0, "origin": origin}] \
            if neg else [spec]
    return None


def compile_lsat_sentence(sentence: str, fr: Frame) -> Optional[List[dict]]:
    """Sentence -> IR: syntax skeleton + coordination + LSAT atoms +
    relative-day ('the next day') expansion. None if any piece fails."""
    from ..reading.syntax_tier import clause_skeleton, expand_coordination
    origin = sentence.strip()
    skel = clause_skeleton(sentence)

    def compile_clause(text: str) -> Optional[List[dict]]:
        atoms, neg_all, comb = expand_coordination(text)
        out = []
        for a in atoms:
            a2 = f"{a} not" if neg_all and "not" not in a else a
            got = compile_lsat_atom(a2, fr, origin)
            if got is None:
                return None
            out.extend(got)
        if comb == "or" and len(out) > 1:
            return [{"type": "or", "clauses": out, "origin": origin}]
        return out

    # relative-day pattern: "if X gives a report, then on the next day ..."
    if "if" in skel and re.search(r"next day", skel["then"], re.I):
        cond_ents = _entity_in(skel["if"], fr)
        then_txt = re.sub(r"on the next day\s*", "", skel["then"], flags=re.I)
        then_ents = _entity_in(then_txt, fr)
        if len(cond_ents) == 1 and then_ents:
            x = cond_ents[0]
            unless_days = _day_in(skel.get("unless", ""), fr)
            out = []
            for i in range(len(fr.days) - 1):
                if i in unless_days:
                    continue
                ante = {"type": "or",
                        "clauses": [{"type": "is", "var": x, "value": s}
                                    for s in fr.day_slots(i)]}
                cons = [{"type": "or",
                         "clauses": [{"type": "is", "var": y, "value": s}
                                     for s in fr.day_slots(i + 1)]}
                        for y in then_ents]
                out.append({"type": "implies", "if": ante,
                            "then": (cons[0] if len(cons) == 1 else
                                     {"type": "and", "clauses": cons}),
                            "origin": origin})
            # unless-day for X: no requirement; nothing to add (vacuous)
            return out

    then_specs = compile_clause(skel["then"])
    if then_specs is None:
        return None
    if "if" not in skel and "unless" not in skel:
        return then_specs
    parts = []
    if "if" in skel:
        c = compile_clause(skel["if"])
        if c is None:
            return None
        parts.append(c[0] if len(c) == 1 else {"type": "and", "clauses": c})
    if "unless" in skel:
        c = compile_clause(skel["unless"])
        if c is None:
            return None
        parts.append({"type": "not",
                      "c": c[0] if len(c) == 1 else
                      {"type": "and", "clauses": c}})
    ante = parts[0] if len(parts) == 1 else {"type": "and", "clauses": parts}
    return [{"type": "implies", "if": ante,
             "then": (then_specs[0] if len(then_specs) == 1 else
                      {"type": "and", "clauses": then_specs}),
             "origin": origin}]


def compile_setup(setup: str, fr: Frame) -> Tuple[List[dict], List[str]]:
    """All condition sentences -> IR. Returns (specs, unhandled_sentences)."""
    text = re.sub(r".*according to the following conditions?:", "", setup,
                  flags=re.I | re.S) or setup
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|(?<=\.)(?=[A-Z])",
                                             text) if s.strip()]
    specs, unhandled = [], []
    for s in sentences:
        got = compile_lsat_sentence(s, fr)
        if got is None:
            unhandled.append(s)
        else:
            specs.extend(got)
    return specs, unhandled


# ------------------------------------------------------------------ choices
def compile_schedule_choice(choice: str, fr: Frame) -> Optional[List[dict]]:
    """'Mon. morning: Helen; Mon. afternoon: Robert; ...' -> is() constraints,
    plus 0 for unmentioned entities. Fully deterministic."""
    choice = re.sub(r"^\([A-E]\)", "", choice).strip()
    pairs = re.findall(
        r"([A-Z][a-z]+)\.?\s+(morning|afternoon|evening)\s*[:\-]\s*([A-Z]\w+)",
        choice)
    if not pairs:
        return None
    out, placed = [], set()
    for day_raw, part_raw, name in pairs:
        day_full = DAY_ABBR.get(day_raw.lower().rstrip("."), day_raw.lower())
        if day_full not in fr.days or name not in fr.entities:
            return None
        i = fr.days.index(day_full)
        j = fr.parts.index(part_raw.lower())
        out.append({"type": "is", "var": name, "value": fr.slot(i, j)})
        placed.add(name)
    if len(out) != fr.n_slots:
        return None
    for e in fr.entities:
        if e not in placed:
            out.append({"type": "is", "var": e, "value": 0})
    return out


# ------------------------------------------------------------------ questions
def split_stem(question: str) -> Tuple[List[str], str]:
    """'If X and Y do not ..., then <query>' -> (premise sentences, query)."""
    q = question.strip()
    m = re.match(r"If\s+(.+?),\s*(?:then\s+)?(which|the\b.*could|.*must)", q,
                 re.I | re.S)
    if not m:
        return [], q
    premise = m.group(1).strip()
    rest = q[m.end(1):].lstrip(", ").strip()
    return [premise], rest


def respectively_slots(stem: str, fr: Frame) -> Optional[List[int]]:
    """'the morning reports on Monday, Tuesday, and Wednesday, respectively'
    -> [slot(Mon,AM), slot(Tue,AM), slot(Wed,AM)]."""
    if "respectively" not in stem.lower():
        return None
    parts = _part_in(stem, fr)
    days = _day_in(stem, fr)
    if len(parts) == 1 and len(days) >= 2:
        return [fr.slot(d, parts[0]) for d in days]
    if len(parts) == 0 and len(days) >= 2:       # "reports on Mon and Tue"
        return None
    return None


def compile_name_list_choice(choice: str, slots: List[int],
                             fr: Frame) -> Optional[List[dict]]:
    """'(A) Helen, George, and Nina' + slot template -> is() constraints."""
    choice = re.sub(r"^\([A-E]\)", "", choice).strip().rstrip(".")
    names = [re.sub(r"^and\s+", "", n).strip()
             for n in re.split(r",\s*|\s+and\s+", choice)]
    names = [n for n in names if n]
    if len(names) != len(slots) or any(n not in fr.entities for n in names):
        return None
    return [{"type": "is", "var": n, "value": s}
            for n, s in zip(names, slots)]


def compile_premise(premise: str, fr: Frame) -> Optional[List[dict]]:
    """Question-stem premise -> IR (handles coordinated negation:
    'Kyle and Lenore do not give reports')."""
    m = re.match(r"([\w ,]+?)\s+(?:do not|don't|does not)\s+give", premise,
                 re.I)
    if m:
        names = [n.strip() for n in re.split(r",\s*|\s+and\s+", m.group(1))
                 if n.strip() in fr.entities]
        if names:
            return [{"type": "is", "var": n, "value": 0,
                     "origin": premise} for n in names]
    return compile_lsat_sentence(premise, fr)
