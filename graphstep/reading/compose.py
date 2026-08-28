#!/usr/bin/env python3
"""The composition compiler: grammatical structure determines how grounded
meanings combine — "sum OF digits OF the number" IS sum(digits(n)), because
"of" means apply, in every domain.

Domain enters through exactly ONE plug: the leaf grounder each algebra
supplies (token -> atom row / variable / parameter). The composition
devices themselves (of-application, between-pairs, comparative-parameters,
coordination, conditionals) are DEVICE rows in the one knowledge store —
they are facts about English, not about code or puzzles or stories.

The output is an algebra-neutral tree:

    Node(kind="atom",  row=<Row>,   children=[...])   grounded meaning
    Node(kind="var",   hint="NUM")                    an unbound input
    Node(kind="param", value=7)                       a value from the text
    Node(kind="combine", device="and", children=[..]) coordination
    Node(kind="cond",  children=[test, body])         conditionals

Emitters per algebra turn the tree into an artifact; `emit_code` (programs)
ships here, other algebras supply their own. A tree is LICENSED BY
CONSTRUCTION: every atom sits exactly where the parse put its word, which
is the head-licensing guard generalized to all depths."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from . import kb


@dataclass
class Node:
    kind: str                      # atom | var | param | combine | cond
    row: object = None             # kb.Row for atom leaves/ops
    children: List["Node"] = field(default_factory=list)
    hint: Optional[str] = None     # type hint for var leaves
    value: object = None           # literal for param leaves
    word: Optional[str] = None     # the surface word (for traces)

    def atoms(self) -> List[str]:
        out = [self.word] if self.kind == "atom" else []
        for c in self.children:
            out += c.atoms()
        return out


_TYPE_HINT = {"number": "NUM", "integer": "NUM", "digit": "NUM",
              "list": "LIST", "array": "LIST", "sequence": "LIST",
              "string": "STRING", "word": "STRING", "sentence": "STRING",
              "tuple": "LIST", "value": "ANY", "element": "ANY",
              "item": "ANY"}


def default_leaf(tok) -> Optional[Node]:
    """The shared leaf grounder: numbers -> params; atom rows by lemma
    (REDUCE, FUNC, BINOP, PRED, PPRED); known type nouns -> variables."""
    if tok.like_num:
        txt = tok.text.replace(",", "")
        try:
            return Node("param", value=int(txt), word=tok.text)
        except ValueError:
            try:
                return Node("param", value=float(txt), word=tok.text)
            except ValueError:
                pass          # number-like words ("first") fall through
    lemma = tok.lemma_.lower()
    for ns in ("REDUCE", "FUNC", "BINOP", "PPRED", "PRED"):
        pass
    for r in kb.KB:
        if r.symbol == "GEOMHEAD" and r.pattern == lemma:
            return Node("atom", row=r, word=lemma)
    for ns in ("REDUCE", "FUNC", "BINOP", "PPRED", "PRED"):
        for r in kb.KB:
            if r.symbol == ns and r.pattern == lemma and r.payload:
                return Node("atom", row=r, word=lemma)
    if lemma in _TYPE_HINT:
        return Node("var", hint=_TYPE_HINT[lemma], word=lemma)
    return None


def _device(prep: str) -> Optional[str]:
    row = kb.match_key(("DEVICE", prep))
    return row.payload if row else None


def compile_tree(tok, leaf: Callable = default_leaf,
                 depth: int = 0) -> Optional[Node]:
    """Walk the parse from a head token; combine per DEVICE rows. Returns
    None when the head cannot ground — an honest gap, never a guess."""
    if depth > 6:
        return None
    node = leaf(tok)
    if node is None:
        return None

    for ch in tok.children:
        if ch.dep_ == "prep":
            dev = _device(ch.lemma_.lower())
            pobjs = [g for g in ch.children if g.dep_ == "pobj"]
            if not pobjs:
                continue
            if dev == "apply":
                sub = compile_tree(pobjs[0], leaf, depth + 1)
                if sub:
                    node.children.append(sub)
                else:   # keep the WORD: formula heads select by it
                    node.children.append(
                        Node("word", word=pobjs[0].lemma_.lower()))
            elif dev == "binop_pair":
                first = pobjs[0]
                pair = [first] + list(first.conjuncts)
                if len(pair) >= 2:                 # "between X and Y"
                    subs = [compile_tree(p, leaf, depth + 1)
                            for p in pair[:2]]
                else:   # coordination on the modifiers:
                        # "between the LARGEST and SMALLEST value"
                    subs = []
                    amods = [a for a in first.children if a.dep_ == "amod"]
                    if amods:
                        for adj in [amods[0]] + list(amods[0].conjuncts):
                            head = leaf(adj)
                            base = leaf(first)     # fresh var per branch
                            if head is not None and head.kind == "atom":
                                if base is not None:
                                    head.children.append(base)
                                subs.append(head)
                node.children.extend(s for s in subs if s)
            elif dev == "param":
                sub = compile_tree(pobjs[0], leaf, depth + 1)
                if sub:
                    node.children.append(sub)
        elif ch.dep_ == "amod":
            mod = leaf(ch)
            if mod and mod.kind == "atom":         # "LARGEST value"
                mod.children.append(node)
                node = mod
        elif ch.dep_ in ("nummod", "quantmod") and ch.like_num:
            num = leaf(ch)
            if num is not None and num.kind == "param":
                node.children.append(num)
        elif ch.dep_ == "conj":
            dev = _device((ch.head.head.lemma_ if False else "and"))
            sub = compile_tree(ch, leaf, depth + 1)
            if sub and node.kind == "atom":
                node.children.append(sub)
        elif ch.dep_ == "mark":
            dev = _device(ch.lemma_.lower())
            if dev in ("condition", "condition_neg"):
                node = Node("cond", children=[node],
                            word=ch.lemma_.lower())
    return node


# ------------------------------------------------------------- code emitter

def emit_code(node: Node, binding: dict) -> Optional[str]:
    """Tree -> a Python expression. `binding` maps var hints to argument
    names (chosen by the caller, enumerated under the coverage guard)."""
    if node.kind == "param":
        return repr(node.value)
    if node.kind == "var":
        return binding.get(id(node))
    if node.kind != "atom":
        return None
    r = node.row
    kids = [emit_code(c, binding) for c in node.children
            if c.kind not in ("param", "word")]
    params = [c.value for c in node.children if c.kind == "param"]
    if any(k is None for k in kids):
        return None
    if r.symbol in ("REDUCE", "FUNC"):
        if "{src}" not in r.payload:
            return r.payload                    # constant atom
        seq = binding.get(("map", id(node)))
        if seq and kids:      # (COERCE, APPLY, SEQ): map the inner chain
            co = kb.match_key(("COERCE", "APPLY", "SEQ"))
            if co is None:
                return None
            return r.payload.format(
                src=co.payload.format(f=kids[0], seq=seq))
        src = kids[0] if kids else binding.get(("free", id(node)))
        return r.payload.format(src=src) if src else None
    if r.symbol == "BINOP":
        if len(kids) == 2:
            return r.payload.format(a=kids[0], b=kids[1])
        if len(kids) == 1:
            b = binding.get(("bfree", id(node)))
            if b == "__FOLD__":
                # type coercion (COERCE, BINOP, SEQ): a binary operation
                # meeting a sequence iterates across it — any binop, any seq
                co = kb.match_key(("COERCE", "BINOP", "SEQ"))
                if co is None:
                    return None
                return co.payload.format(
                    op=r.payload.format(a="_a", b="_b"), seq=kids[0])
            return r.payload.format(a=kids[0], b=b) if b else None
        return None
    if r.symbol == "GEOMHEAD":
        shape = next((c.word for c in node.children if c.word), None)
        g = kb.match_key(("GEOM", node.word, shape)) if shape else None
        if g is None:
            return None
        holes = sorted(set("abc") & set(
            h for h in "abc" if "{%s}" % h in g.payload))
        vals = {h: binding.get(("geom", id(node), h)) for h in holes}
        if any(v is None for v in vals.values()):
            return None
        return g.payload.format(**vals)
    if r.symbol == "PPRED":
        x = kids[0] if kids else binding.get(("free", id(node)))
        k = repr(params[0]) if params else binding.get(("param", id(node)))
        return r.payload.format(x=x, k=k) if (x and k) else None
    if r.symbol == "PRED":
        x = kids[0] if kids else binding.get(("free", id(node)))
        return r.payload.format(x=x) if x else None
    return None


def free_slots(node: Node) -> List[tuple]:
    """Unbound holes the caller must map to arguments: var leaves, atom
    nodes with no child (their {src}/{x}), and PPRED missing {k}."""
    out = []
    if node.kind == "var":
        out.append((id(node), node.hint or "ANY"))
    if node.kind == "atom":
        kids = [c for c in node.children
                if c.kind not in ("param", "word")]
        params = [c for c in node.children if c.kind == "param"]
        if (not kids and node.row.symbol in ("REDUCE", "FUNC", "PRED",
                                             "PPRED")
                and ("{src}" in (node.row.payload or "")
                     or "{x}" in (node.row.payload or ""))):
            out.append((("free", id(node)),
                        node.row.sig.get("arg", "ANY")
                        if node.row.symbol in ("PRED", "PPRED") else "ANY"))
        if node.row.symbol == "PPRED" and not params:
            out.append((("param", id(node)), "NUM"))
        if node.row.symbol == "BINOP" and len(kids) == 1:
            out.append((("bfree", id(node)), "ANY"))
        if node.row.symbol == "GEOMHEAD":
            shape = next((c.word for c in node.children if c.word), None)
            g = kb.match_key(("GEOM", node.word, shape)) if shape else None
            if g is not None:
                for h in "abc":
                    if "{%s}" % h in g.payload:
                        out.append((("geom", id(node), h), "NUM"))
    for c in node.children:
        out += free_slots(c)
    return out
