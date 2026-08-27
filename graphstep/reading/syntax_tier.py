#!/usr/bin/env python3
"""Tier 0.5: syntax-guided compilation.

Uses a dependency parse for STRUCTURE only — clause skeletons (if/then/unless),
coordination expansion (and / or / neither-nor), and negation scope. The
atomic clauses it produces are handed to the deterministic template compilers;
combinators (implies/or/and/not) are assembled symbolically from the skeleton.
Confidence-gated: if any atom fails to compile, the whole sentence falls
through to the next tier (LLM). Zero LLM calls inside this module.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# ------------------------------------------------------------------ skeleton
def clause_skeleton(sentence: str) -> Dict[str, str]:
    """Split a sentence into {'if':…, 'then':…, 'unless':…} clause texts.
    Returns {'then': sentence} when no conditional structure is found.
    Pure surface + parse structure; no semantics."""
    doc = _nlp()(sentence)
    parts: Dict[str, str] = {}
    spans = []                                     # (kind, start_tok, end_tok)
    for tok in doc:
        if tok.dep_ == "mark" and tok.lower_ in ("if", "when", "whenever",
                                                 "unless"):
            head = tok.head                        # root of the subclause
            sub = sorted(t.i for t in head.subtree)
            kind = "unless" if tok.lower_ == "unless" else "if"
            spans.append((kind, sub[0], sub[-1]))
    if not spans:
        return {"then": sentence.strip().rstrip(".")}
    covered = set()
    for kind, a, b in spans:
        toks = [doc[i].text for i in range(a, b + 1)
                if doc[i].lower_ not in ("if", "when", "whenever", "unless")
                and doc[i].dep_ != "punct"]
        parts[kind] = " ".join(toks)
        covered.update(range(a, b + 1))
    main = [t.text for t in doc
            if t.i not in covered and t.dep_ != "punct"
            and t.lower_ not in ("then", ",")]
    parts["then"] = " ".join(main).strip()
    return parts


# ------------------------------------------------------------------ coordination
_NEITHER_RE = re.compile(r"\bneither\s+(.+?)\s+nor\s+(\S+)", re.IGNORECASE)


def expand_coordination(clause: str) -> Tuple[List[str], bool, str]:
    """Expand one clause into atomic clauses.
    Returns (atoms, negated_flag, combine) where combine is 'and' | 'or'.

    - "neither X nor Y <pred>"    -> ["X <pred>", "Y <pred>"], negated, and
    - "X and Y <pred>" (subjects) -> ["X <pred>", "Y <pred>"], -, and
    - "<subj> <vp1> and <vp2>"    -> ["<subj> <vp1>", "<subj> <vp2>"], -, and
    - "X or Y <pred>"             -> ["X <pred>", "Y <pred>"], -, or
    """
    m = _NEITHER_RE.search(clause)
    if m:
        rest = clause[m.end():].strip()
        return [f"{m.group(1)} {rest}", f"{m.group(2)} {rest}"], True, "and"

    doc = _nlp()(clause)
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return [clause], False, "and"

    # subject coordination: "Alice and Bob own …" (also when a nominal was
    # parsed as ROOT, e.g. ambiguous verbs like "like")
    subj = next((t for t in root.children if t.dep_ in ("nsubj", "nsubjpass")),
                None)
    if subj is None and root.pos_ in ("PROPN", "NOUN"):
        subj = root
    if subj is not None:
        conjs = [t for t in subj.children if t.dep_ == "conj"]
        if conjs:
            cc = next((t.lower_ for t in subj.children if t.dep_ == "cc"), "and")
            conj_idx = {x.i for c in conjs for x in c.subtree}
            pred = " ".join(t.text for t in doc
                            if t.i != subj.i and t.i not in conj_idx
                            and t.dep_ not in ("punct", "cc"))
            names = [subj.text] + [c.text for c in conjs]
            return ([f"{n} {pred}" for n in names], False,
                    "or" if cc == "or" else "and")

    # verb-phrase coordination: "… owns the cat and lives in the red house"
    vconjs = [t for t in root.children if t.dep_ == "conj" and t.pos_ == "VERB"]
    if vconjs:
        subj_txt = subj.text if subj else ""
        first = " ".join(t.text for t in doc
                         if all(t.i not in [x.i for x in c.subtree]
                                for c in vconjs)
                         and t.dep_ not in ("punct", "cc"))
        atoms = [first]
        for c in vconjs:
            vp = " ".join(t.text for t in c.subtree if t.dep_ != "punct")
            atoms.append(f"{subj_txt} {vp}")
        return atoms, False, "and"

    return [clause], False, "and"


# ------------------------------------------------------------------ compile
def compile_with_syntax(clue: str, inv, positional: bool) -> Optional[List[dict]]:
    """Syntax-guided compile: skeleton -> atoms -> templates -> combinators.
    Returns IR specs or None (fall through to LLM tier)."""
    from .compile_text import compile_assignment_clue, compile_position_clue

    def compile_atom(text: str, forced_neg: bool = False) -> Optional[List[dict]]:
        specs = (compile_position_clue(text, inv) if positional else None)
        if specs is None:
            specs = compile_assignment_clue(text, inv)
        if specs is None:
            return None
        if forced_neg:
            flipped = []
            for s in specs:
                t = s.get("type")
                flip = {"is": "is_not", "is_not": "is",
                        "same": "diff", "diff": "same"}.get(t)
                if flip is None:
                    return None
                flipped.append({**s, "type": flip})
            specs = flipped
        return specs

    skel = clause_skeleton(clue)
    origin = clue.strip()

    def compile_clause(text: str) -> Optional[List[dict]]:
        atoms, neg, combine = expand_coordination(text)
        compiled: List[dict] = []
        for a in atoms:
            got = compile_atom(a, forced_neg=neg)
            if got is None:
                return None
            compiled.extend(got)
        if combine == "or" and len(compiled) > 1:
            return [{"type": "or", "clauses": compiled, "origin": origin}]
        return compiled

    then_specs = compile_clause(skel["then"])
    if then_specs is None:
        return None
    for s in then_specs:
        s.setdefault("origin", origin)

    if "if" not in skel and "unless" not in skel:
        # plain sentence: only worth emitting if syntax actually decomposed it
        return then_specs

    def conj(specs: List[dict]) -> dict:
        return specs[0] if len(specs) == 1 else {"type": "and", "clauses": specs}

    cond_parts: List[dict] = []
    if "if" in skel:
        if_specs = compile_clause(skel["if"])
        if if_specs is None:
            return None
        cond_parts.append(conj(if_specs))
    if "unless" in skel:
        un_specs = compile_clause(skel["unless"])
        if un_specs is None:
            return None
        # "THEN, unless U"  ==  not(U) -> THEN   (with IF: (IF and not U) -> THEN)
        cond_parts.append({"type": "not", "c": conj(un_specs)})

    antecedent = (cond_parts[0] if len(cond_parts) == 1
                  else {"type": "and", "clauses": cond_parts})
    return [{"type": "implies", "if": antecedent, "then": conj(then_specs),
             "origin": origin}]
