#!/usr/bin/env python3
"""The ONE clause reader: any text -> a semantic graph. Total capture,
two tiers of trust.

TOTAL CAPTURE — every clause of every sentence becomes a node with typed
roles (agent, object, attribute, place, time, ...), polarity, tense, and
modality. Nothing is dropped: a clause whose deeper meaning no row knows is
still in the graph, searchable and reportable as "the text asserts ...".

TWO TIERS — a captured clause is associative evidence ("the text says X").
Only when a knowledge-store row recognizes the construction (a rule shape,
a class statement, a comparative, a temporal update) is the clause promoted
to proof grade — usable inside derivations. Conclusions may cite proof-grade
edges only; the captured remainder is reported, never reasoned over as if
understood. Precision may not regress while capture grows.

MODALITY — reported/attitudinal frames (say, claim, expect, believe, deny,
announce, hope, ...) wrap their inner clause: "The CEO said the product is
safe" captures  claim(CEO) -> [product is safe]  and the inner clause is
NOT a fact about the world — it is a fact about who said what. Treating
attribution as truth is fabrication; the wrapper is what blocks it.

EDGES — (subject, predicate, object) with properties: polarity, tense,
modality/attribution, tier, provenance (the sentence), and a validity
interval [start, end) in sentence time. NOTHING IS EVER DELETED: for an
exclusive relation a newer value CLOSES the older edge (sets its end);
the open edge is "current", closed edges are history, and two OPEN
exclusive edges are a genuine contradiction rather than an update.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# attitudinal / reporting verbs: their complement clause is attributed, not
# asserted. A declarative row family (extendable via WordNet like OP: rows).
MODAL_VERBS = {"say", "claim", "expect", "believe", "deny", "announce",
               "report", "think", "hope", "suggest", "argue", "state",
               "predict", "estimate", "warn", "promise", "insist", "fear"}

# relations where a subject holds ONE value at a time: a newer edge closes
# the older one. Cumulative relations accumulate open edges instead.
EXCLUSIVE_RELATIONS = {"location", "position"}


@dataclass
class Clause:
    pred: str                          # verb lemma ("release", "be")
    roles: Dict[str, str] = field(default_factory=dict)
    polarity: bool = True
    tense: str = "present"             # past | present | future
    modality: Optional[str] = None     # e.g. "say" — attributed, not asserted
    attributed_to: Optional[str] = None
    condition: Optional[str] = None    # "if" / "unless" marker of an advcl
    tier: str = "captured"             # captured | grounded
    sent_index: int = 0
    text: str = ""

    def report(self) -> str:
        """Plain-English one-liner of what was captured."""
        core = self.pred
        agent = self.roles.get("agent", "")
        obj = self.roles.get("object") or self.roles.get("attribute") or ""
        extras = ", ".join(f"{k}: {v}" for k, v in self.roles.items()
                           if k not in ("agent", "object", "attribute"))
        neg = "" if self.polarity else " not"
        s = f"{agent}{neg} {core} {obj}".strip()
        if extras:
            s += f" ({extras})"
        s += f" [{self.tense}]"
        if self.modality:
            s = (f"{self.attributed_to or 'someone'} {self.modality}s: "
                 f"“{s}”")
        if self.condition:
            s = f"[{self.condition}] {s}"
        return s


@dataclass
class Edge:
    subj: str
    pred: str
    obj: Optional[str]
    polarity: bool = True
    tier: str = "captured"
    modality: Optional[str] = None
    attributed_to: Optional[str] = None
    start: int = 0                     # sentence index the edge opens at
    end: Optional[int] = None          # None = still current (open edge)
    provenance: str = ""


@dataclass
class SemGraph:
    clauses: List[Clause] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)  # name -> NE type
    unparsed: List[str] = field(default_factory=list)

    def assert_edge(self, e: Edge) -> None:
        """Add an edge; for an exclusive relation, CLOSE (never delete) any
        open edge of the same (subject, predicate)."""
        if e.pred in EXCLUSIVE_RELATIONS:
            for old in self.edges:
                if (old.subj == e.subj and old.pred == e.pred
                        and old.end is None and old.polarity):
                    old.end = e.start
        self.edges.append(e)

    def current(self, subj: str, pred: str) -> List[Edge]:
        return [e for e in self.edges
                if e.subj == subj and e.pred == pred and e.end is None]

    def coverage(self) -> Dict[str, int]:
        return {"clauses": len(self.clauses),
                "grounded": sum(c.tier == "grounded" for c in self.clauses),
                "attributed": sum(c.modality is not None
                                  for c in self.clauses),
                "unparsed": len(self.unparsed)}


_NLP = None
def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _tense(tok) -> str:
    m = tok.morph.get("Tense")
    if any(c.lemma_ in ("will", "shall") for c in tok.children
           if c.dep_ == "aux"):
        return "future"
    if m == ["Past"]:
        return "past"
    return "present"


def _phrase(tok) -> str:
    """Content phrase for a role filler: the token's compound/amod chain."""
    parts = [t.text for t in sorted(
        [tok] + [c for c in tok.children
                 if c.dep_ in ("compound", "amod", "nummod", "poss")],
        key=lambda t: t.i)]
    return " ".join(parts).lower()


def _clause_of(verb, sent_index: int, text: str) -> Clause:
    c = Clause(pred=verb.lemma_.lower(), sent_index=sent_index, text=text,
               tense=_tense(verb))
    for ch in verb.children:
        d = ch.dep_
        if d in ("nsubj", "nsubjpass"):
            c.roles["agent" if d == "nsubj" else "object"] = _phrase(ch)
        elif d in ("dobj", "obj"):
            c.roles["object"] = _phrase(ch)
        elif d in ("attr", "acomp"):
            c.roles["attribute"] = _phrase(ch)
        elif d == "neg":
            c.polarity = False
        elif d == "prep":
            pobj = [g for g in ch.children if g.dep_ == "pobj"]
            if pobj:
                c.roles[ch.lemma_.lower()] = _phrase(pobj[0])
        elif d == "npadvmod" or (d == "advmod"
                                 and ch.ent_type_ in ("DATE", "TIME")):
            c.roles["time"] = _phrase(ch)
        elif d == "mark" and ch.lemma_.lower() in ("if", "unless", "when"):
            c.condition = ch.lemma_.lower()
    return c


def read(text: str) -> SemGraph:
    """Any text -> semantic graph. Every clause captured; reporting verbs
    wrap their complement clause as ATTRIBUTED rather than asserted."""
    g = SemGraph()
    doc = _nlp()(text)
    for ent in doc.ents:
        g.entities[ent.text.lower()] = ent.label_
    for si, sent in enumerate(doc.sents):
        heads = [t for t in sent
                 if t.pos_ in ("VERB", "AUX") and t.dep_ in
                 ("ROOT", "conj", "advcl", "ccomp", "xcomp", "relcl")]
        if not heads:
            g.unparsed.append(sent.text.strip())
            continue
        by_tok = {}
        for v in heads:
            cl = _clause_of(v, si, sent.text.strip())
            by_tok[v.i] = cl
        # attribution: a complement under a reporting verb is attributed
        for v in heads:
            cl = by_tok[v.i]
            if cl.pred in MODAL_VERBS:
                for ch in v.children:
                    if ch.dep_ in ("ccomp", "xcomp") and ch.i in by_tok:
                        inner = by_tok[ch.i]
                        inner.modality = cl.pred
                        inner.attributed_to = cl.roles.get("agent")
        for cl in by_tok.values():
            g.clauses.append(cl)
            subj = cl.roles.get("agent")
            obj = (cl.roles.get("object") or cl.roles.get("attribute"))
            if subj:
                g.assert_edge(Edge(
                    subj=subj, pred=cl.pred, obj=obj,
                    polarity=cl.polarity, modality=cl.modality,
                    attributed_to=cl.attributed_to, start=si,
                    provenance=cl.text))
    return g
