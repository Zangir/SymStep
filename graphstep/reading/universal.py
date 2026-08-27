#!/usr/bin/env python3
"""Universal text -> relation graph -> constraint IR.

The general path: NO task-specific templates, NO given inventory. For any
input text this layer
  1. extracts a RELATION GRAPH: (subject, predicate, object) triples with
     negation scope and conditional skeletons, via the dependency parse —
     including anonymous nodes for descriptions like "the person who owns
     the dog";
  2. INDUCES the ontology from the graph itself: entity class (recurring
     subjects / listed names), attributes (predicates grouped by the value
     class they select), domains (the values seen with each attribute);
  3. compiles the graph into the same constraint IR the engine runs on,
     using the value-position encoding (each value is a variable over entity
     ids), with structural assumptions (each value used at most once ->
     alldiff) emitted EXPLICITLY as assumption records;
  4. reports coverage honestly: which sentences became constraints and which
     were not understood (callers may escalate just those to an LLM).

Deterministic end to end; zero LLM calls inside this module.
"""
from __future__ import annotations
import itertools
import re
from typing import Dict, List, Optional, Tuple

from .syntax_tier import _nlp, clause_skeleton

STOP_VALUES = {"person", "people", "one", "who", "that", "which", "each",
               "everyone", "day", "report", "thing", "student"}
QUANTIFIER_ADJ = {"different", "same", "unique", "various", "certain",
                  "other", "another", "distinct", "separate"}
GENERIC_NOUNS = {"person", "one", "student", "man", "woman", "people",
                 "individual", "resident", "owner"}


def _clean_value(v: str) -> Optional[str]:
    """Drop quantifier adjectives; reject quantifier-only / stop values."""
    words = [w for w in v.split() if w.lower() not in QUANTIFIER_ADJ]
    if not words:
        return None
    v2 = " ".join(words).strip(".,;:!?")
    if not v2 or v2.lower() in STOP_VALUES:
        return None
    return v2


# ===================================================================== triples
class Triple:
    """subject --predicate--> object, possibly negated.
    subject/object are either ('name', text) or ('anon', [(pred, value), ...])
    for descriptions ('the person who owns the dog')."""

    def __init__(self, subj, pred: str, obj, negated: bool, sent: str):
        self.subj, self.pred, self.obj = subj, pred, obj
        self.negated = negated
        self.sent = sent

    def __repr__(self):
        n = "NOT " if self.negated else ""
        return f"({self.subj} -{n}{self.pred}-> {self.obj})"


def _noun_phrase(tok) -> str:
    """Content head of a noun phrase (drop determiners/prep scaffolding)."""
    words = [t for t in tok.subtree
             if t.dep_ not in ("det", "punct", "prep", "pobj", "relcl",
                               "acl", "advcl")
             and t.head in (tok, tok.head) or t is tok]
    core = [t.text for t in sorted(set(words), key=lambda t: t.i)
            if t.pos_ in ("NOUN", "PROPN", "ADJ", "NUM")]
    return " ".join(core) if core else tok.text


def _describe_anon(tok, sent_text: str):
    """'the person who owns the dog'   -> ('anon', [('own', 'dog')])
       'the person in the red house'   -> ('anon', [('in', 'red house')])."""
    rel = next((t for t in tok.children if t.dep_ in ("relcl", "acl")), None)
    if rel is not None:
        obj = next((c for c in rel.children
                    if c.dep_ in ("dobj", "attr", "oprd", "pobj")), None)
        if obj is None:
            obj = next((c for c in rel.subtree
                        if c.dep_ == "pobj" and c.head.head == rel), None)
        if obj is not None:
            neg = any(c.dep_ == "neg" for c in rel.children)
            return ("anon", [(rel.lemma_, _noun_phrase(obj))], neg)
    # generic noun + prepositional description: "the person in the red house"
    if tok.lemma_.lower() in GENERIC_NOUNS:
        prep = next((c for c in tok.children if c.dep_ == "prep"), None)
        if prep is not None:
            po = next((g for g in prep.children if g.dep_ == "pobj"), None)
            if po is not None:
                return ("anon", [(prep.lower_, _noun_phrase(po))], False)
    return None


def _node_of(tok, sent_text: str):
    """Token -> graph node."""
    anon = _describe_anon(tok, sent_text)
    if anon is not None:
        return anon[:2], anon[2]
    return ("name", _noun_phrase(tok)), False


def extract_triples(sentence: str) -> List[Triple]:
    """Dependency-based open relation extraction for one clause, with a
    shallow fallback for short clauses the parser mangles."""
    sentence = sentence.strip()
    if sentence and sentence[-1] not in ".!?":
        sentence += "."                        # stabilizes short-clause parses
    doc = _nlp()(sentence)
    out: List[Triple] = []
    for tok in doc:
        if tok.pos_ not in ("VERB", "AUX"):
            continue
        subj = next((c for c in tok.children
                     if c.dep_ in ("nsubj", "nsubjpass")), None)
        if subj is None:
            continue
        neg = any(c.dep_ == "neg" for c in tok.children)
        # object candidates: direct object, attribute (copula), prep object
        objs = []
        for c in tok.children:
            if c.dep_ in ("dobj", "attr", "oprd", "acomp"):
                objs.append((tok.lemma_, c))
            elif c.dep_ == "prep":
                po = next((g for g in c.children if g.dep_ == "pobj"), None)
                if po is not None:
                    objs.append((f"{tok.lemma_}_{c.lower_}", po))
        for pred, objtok in objs:
            (snode, sneg) = _node_of(subj, sentence)
            (onode, oneg) = _node_of(objtok, sentence)
            out.append(Triple(snode, pred, onode,
                              neg or sneg or oneg, sentence))
    if not out:
        # shallow fallback for short "X <verb> Y" clauses the small parser
        # mangles (e.g. unusual proper names): first noun = subj,
        # last noun phrase = obj, the token between them = predicate
        content = [t for t in doc if t.pos_ in ("PROPN", "NOUN", "VERB",
                                                "AUX", "ADJ", "PART")]
        if 2 <= len(content) <= 7:
            nouns = [t for t in content if t.pos_ in ("PROPN", "NOUN")]
            if len(nouns) >= 2:
                subj_t, obj_t = nouns[0], nouns[-1]
                between = [t for t in doc
                           if subj_t.i < t.i < obj_t.i
                           and t.pos_ in ("VERB", "AUX", "NOUN")]
                neg = any(t.dep_ == "neg" or t.lower_ in ("not", "n't")
                          for t in doc)
                if between:
                    out.append(Triple(("name", subj_t.text),
                                      between[0].lemma_,
                                      ("name", _noun_phrase(obj_t)),
                                      neg, sentence))
    return out


# ===================================================================== ontology
class Ontology:
    def __init__(self):
        self.entities: List[str] = []           # entity class (ids 1..n)
        self.attributes: Dict[str, List[str]] = {}   # attr -> values
        self.value_attr: Dict[str, str] = {}
        self.assumptions: List[str] = []
        self.canon: Dict[str, str] = {}

    def entity_id(self, name: str) -> Optional[int]:
        low = [e.lower() for e in self.entities]
        return low.index(name.lower()) + 1 if name.lower() in low else None


_LIST_RE = re.compile(
    r"[:—\-–]\s*([A-Z][\w' ]*(?:,\s*[A-Z][\w' ]*)+,?\s*(?:and|or)\s+[A-Z][\w' ]*)")
_LEAD_LIST_RE = re.compile(
    r"\b([A-Z][a-z]\w*(?:,\s*[A-Z][a-z]\w*)+,?\s*(?:and|or)\s+[A-Z][a-z]\w*)\b")
_EACH_RE = re.compile(
    r"each (?:\w+ )?(?:has|owns|likes|drinks|plays|holds|gets|is assigned)"
    r"[^.]*?different", re.I)
_SCHEMA_RE = re.compile(
    r"\b(each|every)\b.{0,50}\b(different|unique|distinct|separate)\b", re.I)
_INV_RE = re.compile(
    r"(?:[Tt]he |unique |possible )(\w+?)s?(?:\s+are|\s+include|:)\s+"
    r"([\w' \-]+(?:,\s*[\w' \-]+)+,?\s*(?:and|or)\s+[\w' \-]+)")


def is_structural_sentence(s: str) -> bool:
    """Schema ('each … a different pet') or inventory ('The pets are …')
    sentences define the ontology; they are not facts about individuals."""
    return bool(_SCHEMA_RE.search(s) or _INV_RE.search(s))


def induce_ontology(text: str, triples: List[Triple]) -> Ontology:
    """Discover entities, attributes and domains from the text + graph."""
    onto = Ontology()

    # entities: explicit name list if present, else recurring subjects
    m = _LIST_RE.search(text) or _LEAD_LIST_RE.search(text)
    if m:
        names = [re.sub(r"^and\s+|^or\s+", "", n).strip()
                 for n in re.split(r",\s*|\s+and\s+|\s+or\s+", m.group(1))]
        onto.entities = [n for n in names if n and n[0].isupper()]
    if not onto.entities:
        counts: Dict[str, int] = {}
        for t in triples:
            if t.subj[0] == "name" and t.subj[1][:1].isupper():
                counts[t.subj[1]] = counts.get(t.subj[1], 0) + 1
        singles = {n for n in counts if " " not in n}
        onto.entities = sorted(
            [n for n in counts
             if " " not in n or not any(s in n for s in singles)],
            key=lambda n: -counts[n])[:12]

    ents_low = {e.lower() for e in onto.entities}

    def _usable(v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        v = _clean_value(v)
        if (not v or v.lower() in ents_low
                or any(w in ents_low for w in v.lower().split())
                or any(ch.isdigit() for ch in v)):
            return None
        return v.title()

    # inventory sentences state whole domains explicitly:
    # "The drinks are tea, coffee, and water." / "… unique jobs: A, B, C"
    pred_vals: Dict[str, set] = {}
    for m in _INV_RE.finditer(text):
        cat = m.group(1).lower()
        vals = [re.sub(r"^(and|or)\s+", "", v).strip()
                for v in re.split(r",\s*|\s+and\s+|\s+or\s+", m.group(2))]
        vals = [v for v in (_clean_value(v) for v in vals)
                if v and v.lower() not in {e.lower() for e in onto.entities}]
        if len(vals) >= 2:
            pred_vals.setdefault(f"inv_{cat}", set()).update(
                v.title() for v in vals)

    # attributes: group object values by predicate (excluding entity names).
    # Schema sentences ("each … a different pet") name CATEGORIES, not
    # values, so they are excluded from domain collection.
    for t in triples:
        if is_structural_sentence(t.sent):
            continue
        if t.obj[0] == "name":
            v = _usable(t.obj[1])
            if v:
                pred_vals.setdefault(t.pred, set()).add(v)
        # anonymous nodes contribute their description values too
        for node in (t.subj, t.obj):
            if node[0] == "anon":
                for p2, v2 in node[1]:
                    v = _usable(v2)
                    if v:
                        pred_vals.setdefault(p2, set()).add(v)

    # canonicalize onto inventory values: multiword containment
    # ("Blue House" -> "Blue") and morphological variants
    # ("Apples"/"Cherries" -> "Apple"/"Cherry"), using the same variant
    # machinery the mention scanner trusts.
    from .compile_text import _surface_variants
    inv_vals = {v for p, vs in pred_vals.items() if p.startswith("inv_")
                for v in vs}
    variant_of: Dict[str, str] = {}
    for w in inv_vals:
        for s in _surface_variants(w):
            variant_of.setdefault(s, w)
    canon: Dict[str, str] = {}
    for p, vs in list(pred_vals.items()):
        if p.startswith("inv_"):
            continue
        renamed = set()
        for v in vs:
            hit = None
            if v not in inv_vals:
                hit = variant_of.get(v.lower())
                if hit is None:
                    hit = next((w for w in inv_vals
                                if w.lower() in v.lower().split()), None)
            if hit and hit != v:
                canon[v] = hit
            renamed.add(canon.get(v, v))
        pred_vals[p] = renamed
    onto.canon = canon

    # merge predicates selecting the same value class (own/have/keep), and
    # bare prepositions with verb_prep forms (in ~ live_in)
    def _key(p: str) -> str:
        return p.split("_")[-1]

    merged: List[Tuple[set, set]] = []          # (preds, values)
    for p, vs in sorted(pred_vals.items()):
        hit = next((mv for mv in merged
                    if mv[1] & vs or any(_key(q) == _key(p) for q in mv[0])),
                   None)
        if hit:
            hit[0].add(p); hit[1].update(vs)
        else:
            merged.append(({p}, set(vs)))
    for preds, vs in merged:
        if len(vs) < 2:
            continue                             # a 1-value class is not an attribute
        attr = "/".join(sorted(preds))
        onto.attributes[attr] = sorted(vs)
        for v in vs:
            onto.value_attr[v] = attr

    n = len(onto.entities)
    for attr, vs in onto.attributes.items():
        if len(vs) == n or _EACH_RE.search(text):
            onto.assumptions.append(
                f"values of [{attr}] are used by distinct entities (alldiff)")
    return onto


# ===================================================================== compile
def compile_universal(text: str) -> Tuple[Optional[dict], dict]:
    """Any text -> (IR dict or None, report). Deterministic; coverage-honest."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                 if s.strip()]
    all_triples: List[Triple] = []
    sent_triples: Dict[str, List[Triple]] = {}
    for s in sentences:
        skel = clause_skeleton(s)
        parts = [skel[k] for k in ("if", "then", "unless") if k in skel]
        ts: List[Triple] = []
        for part in parts:
            ts.extend(extract_triples(part))
        sent_triples[s] = ts
        all_triples.extend(ts)

    onto = induce_ontology(text, all_triples)
    report = {"sentences": len(sentences), "triples": len(all_triples),
              "entities": onto.entities,
              "attributes": {a: vs for a, vs in onto.attributes.items()},
              "assumptions": onto.assumptions,
              "compiled": 0, "uncovered": []}
    if not onto.entities or not onto.attributes:
        report["uncovered"] = sentences
        return None, report

    n = len(onto.entities)
    variables = {v: list(range(1, n + 1)) for v in onto.value_attr}
    constraints: List[dict] = []

    # induced Inventory -> reuse the lexically-robust mention compiler as a
    # deterministic fallback for clauses the triple extractor cannot read.
    # An alias is only safe if its extra words are not surfaces of OTHER
    # values (else it would swallow their mentions in longest-first scans).
    from .compile_text import Inventory, compile_assignment_clue, \
        _surface_variants
    surface_owner: Dict[str, str] = {}
    for v in onto.value_attr:
        for sfc in _surface_variants(v):
            surface_owner.setdefault(sfc, v)
    alias_map: Dict[str, List[str]] = {}
    for long, short in onto.canon.items():
        extra = [w for w in long.lower().split()
                 if w not in short.lower().split()]
        if any(surface_owner.get(w, short) != short for w in extra):
            continue
        alias_map.setdefault(short, []).append(long)
    induced_inv = Inventory(onto.entities,
                            {a: vs for a, vs in onto.attributes.items()},
                            aliases=alias_map)
    for attr, vs in onto.attributes.items():
        if len(vs) == n:
            constraints.append({"type": "alldiff", "vars": vs,
                                "origin": f"[assumption] distinct {attr}"})

    def spec_of(triple: Triple) -> Optional[dict]:
        """Triple -> IR constraint under the value-position encoding."""
        def side_var(node):
            if node[0] == "anon":
                for _, v in node[1]:
                    t = onto.canon.get(v.title(), v.title())
                    if t in onto.value_attr:
                        return ("var", t)
                return None
            nm = node[1]
            eid = onto.entity_id(nm)
            if eid is not None:
                return ("ent", eid)
            t = onto.canon.get(nm.title(), nm.title())
            if t in onto.value_attr:
                return ("var", t)
            return None

        s, o = side_var(triple.subj), side_var(triple.obj)
        if s is None or o is None or (s[0] == "ent" and o[0] == "ent"):
            return None
        if s[0] == "ent":                        # Alice -owns-> cat
            typ = "is_not" if triple.negated else "is"
            return {"type": typ, "var": o[1], "value": s[1]}
        if o[0] == "ent":                        # the cat owner is Alice
            typ = "is_not" if triple.negated else "is"
            return {"type": typ, "var": s[1], "value": o[1]}
        if s[1] == o[1]:
            return None
        typ = "diff" if triple.negated else "same"
        return {"type": typ, "a": s[1], "b": o[1]}

    for s in sentences:
        if is_structural_sentence(s):
            report["compiled"] += 1          # ontology sentence, understood
            continue
        skel = clause_skeleton(s)
        specs_by_part: Dict[str, List[dict]] = {}
        ok = True
        for key in ("if", "then", "unless"):
            if key not in skel:
                continue
            part_specs = []
            for t in extract_triples(skel[key]):
                sp = spec_of(t)
                if sp is not None:
                    part_specs.append({**sp, "origin": s})
            if not part_specs:                     # ontology-aware fallback
                got = compile_assignment_clue(skel[key], induced_inv)
                if got:
                    part_specs = [{**g, "origin": s} for g in got]
            if not part_specs:
                ok = False
            specs_by_part[key] = part_specs
        if not ok or not specs_by_part.get("then"):
            report["uncovered"].append(s)
            continue

        def conj(specs):
            return specs[0] if len(specs) == 1 else \
                {"type": "and", "clauses": specs}

        if "if" in specs_by_part or "unless" in specs_by_part:
            ante = []
            if "if" in specs_by_part:
                ante.append(conj(specs_by_part["if"]))
            if "unless" in specs_by_part:
                ante.append({"type": "not",
                             "c": conj(specs_by_part["unless"])})
            constraints.append(
                {"type": "implies",
                 "if": ante[0] if len(ante) == 1 else
                 {"type": "and", "clauses": ante},
                 "then": conj(specs_by_part["then"]), "origin": s})
        else:
            constraints.extend(specs_by_part["then"])
        report["compiled"] += 1

    ir = {"variables": variables, "constraints": constraints,
          "encoding": {"entities": {e: i + 1
                                    for i, e in enumerate(onto.entities)}}}
    return ir, report


def solve_universal(text: str, max_solutions: int = 2,
                    allow_llm: bool = False, repair_rounds: int = 1):
    """End-to-end: any text -> induced graph -> certified solve + report.

    With allow_llm=True, the LLM is summoned ONLY for (a) sentences the
    deterministic layers left uncovered and (b) UNSAT-core repair — never
    for whole-problem translation. Every LLM output is validated against
    the induced ontology before acceptance."""
    from ..engine.ir import problem_from_ir
    from ..engine.core import Engine
    from ..legacy import llm as llm_mod

    calls_before = llm_mod.LLM_CALLS["n"]
    ir, report = compile_universal(text)
    if ir is None:
        report["llm_calls"] = 0
        return None, report

    entity_index = ir["encoding"]["entities"]

    # ---- LLM tier: only the uncovered sentences -------------------------
    if allow_llm and report["uncovered"]:
        for s in list(report["uncovered"]):
            got = llm_mod.llm_compile_clue(s, ir["variables"], entity_index)
            if got:
                ir["constraints"].extend(got)
                report["uncovered"].remove(s)
                report["compiled"] += 1

    def run(constraints):
        return Engine(problem_from_ir(
            {"variables": ir["variables"],
             "constraints": constraints})).solve(max_solutions=max_solutions)

    res = run(ir["constraints"])

    # ---- LLM repair: UNSAT core -> re-read exactly those sentences ------
    rounds = 0
    while (allow_llm and res.status == "UNSAT" and rounds < repair_rounds):
        rounds += 1
        eng = Engine(problem_from_ir({"variables": ir["variables"],
                                      "constraints": ir["constraints"]}))
        core = eng.unsat_core()
        core_sents = sorted({c.origin for c in core
                             if c.origin and not c.origin.startswith("[")})
        if not core_sents:
            break
        keep = [c for c in ir["constraints"]
                if c.get("origin") not in core_sents]
        repaired, ok = [], True
        for s in core_sents:
            got = llm_mod.llm_compile_clue(
                s, ir["variables"], entity_index,
                context=(f"\nNote: an earlier literal reading of these "
                         f"sentences was mutually inconsistent: {core_sents}. "
                         f"Read this one extra carefully.\n"))
            if got:
                repaired.extend(got)
            else:
                ok = False
        if not ok:
            break
        ir["constraints"] = keep + repaired
        report["repairs"] = rounds
        res = run(ir["constraints"])

    # ---- LLM repair: ambiguity-directed --------------------------------
    # AMBIGUOUS with nothing uncovered means some sentence compiled
    # INCOMPLETELY. Find the under-determined variables (they differ across
    # the solutions found), re-read exactly the sentences that mention them,
    # and ADD whatever constraints the re-read produces.
    if (allow_llm and res.status == "AMBIGUOUS" and len(res.solutions) >= 2):
        sols = res.solutions
        loose = {v for v in ir["variables"]
                 if len({s[v] for s in sols}) > 1}
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                 if s.strip() and not is_structural_sentence(s)
                 and any(re.search(rf"\b{re.escape(v.lower())}", s.lower())
                         for v in loose)]
        added = []
        for s in sents[:6]:
            got = llm_mod.llm_compile_clue(
                s, ir["variables"], entity_index,
                context=(f"\nNote: the current reading leaves "
                         f"{sorted(loose)} under-determined; this sentence "
                         f"mentions them. Translate it COMPLETELY.\n"))
            if got:
                added.extend(got)
        if added:
            trial = ir["constraints"] + added
            res2 = run(trial)
            if res2.status in ("SOLVED", "AMBIGUOUS") and res2.solutions:
                ir["constraints"] = trial          # only keep if consistent
                res = res2
                report["ambiguity_repair"] = len(added)

    report["llm_calls"] = llm_mod.LLM_CALLS["n"] - calls_before
    return res, {**report, "status": res.status,
                 "encoding": ir["encoding"]}
