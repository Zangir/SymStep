#!/usr/bin/env python3
"""Rule induction over world-model fact stores — the general pipeline:

    facts -> MINE -> VERIFY -> TIER -> (closure) -> gated answers

Four universal rule shapes (no domain content — pure logical forms):
  S1  unary subsumption   A(x) => B(x)
  S2  binary subsumption  R1(x,y) => R2(x,y)
  S3  inverse             R1(x,y) => R2(y,x)
  S4  chain               R1(x,y) & R2(y,z) => R3(x,z)
plus E1 exclusion mining (disjoint attribute alternatives).

Confidence semantics (the open-world tradeoff, parameterized):
  assume_closed=False (default): a pair counts AGAINST a rule only when the
    head is OBSERVED false — unobserved heads are neutral. Right for
    incomplete worlds; small worlds may license reversals, which is why
    induced conclusions are tier-gated to "Likely", never "True".
  assume_closed=True: unobserved heads count as counterexamples (subset
    semantics). Right for exhaustively described worlds.

Induced rules are living objects: verify() re-checks them against the
current facts and retracts any rule that has acquired a counterexample.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .worldmodel import Rule, Literal


def _partition_facts(facts):
    """facts dict -> (unary_pos, unary_neg, binary_pos, binary_neg)."""
    up, un = defaultdict(set), defaultdict(set)
    bp, bn = defaultdict(set), defaultdict(set)
    for (s, p, o), (pos, _) in facts.items():
        if o is None:
            (up if pos else un)[p].add(s)
        else:
            (bp if pos else bn)[p].add((s, o))
    return up, un, bp, bn


def _ok(head_pos, head_neg, pairs, assume_closed):
    """Confidence check for candidate pairs against observed head facts.
    Returns support if the rule stands, else None."""
    support = 0
    for pr in pairs:
        if pr in head_pos:
            support += 1
        elif pr in head_neg or assume_closed:
            return None                       # observed (or assumed) counterexample
    return support


def mine_rules(wm, min_support: int = 3,
               assume_closed: bool = False) -> List[Rule]:
    """All rules of shapes S1-S4 supported by the current facts."""
    up, un, bp, bn = _partition_facts(wm.facts)
    stated = {(tuple(sorted(repr(c) for c in r.conds)), repr(r.concl))
              for r in wm.rules}
    out: List[Rule] = []

    def emit(conds, concl, support, shape, desc):
        key = (tuple(sorted(repr(c) for c in conds)), repr(concl))
        if key in stated:
            return
        out.append(Rule(conds, concl,
                        f"[induced:{shape}] {desc} (support={support})"))

    for a, sa in up.items():                                      # S1
        if len(sa) < min_support:
            continue
        for b in up:
            if a == b:
                continue
            sup = _ok(up[b], un[b], sa, assume_closed)
            if sup is not None and sup >= min_support:
                emit([Literal(True, "?", a, None)],
                     Literal(True, "?", b, None), sup, "S1",
                     f"{a}(x) => {b}(x)")

    for r1, p1 in bp.items():                                     # S2 / S3
        if len(p1) < min_support:
            continue
        inv1 = {(o, s) for s, o in p1}
        for r2 in bp:
            if r1 != r2:
                sup = _ok(bp[r2], bn[r2], p1, assume_closed)
                if sup is not None and sup >= min_support:
                    emit([Literal(True, "?a", r1, "?b")],
                         Literal(True, "?a", r2, "?b"), sup, "S2",
                         f"{r1}(x,y) => {r2}(x,y)")
            sup = _ok(bp[r2], bn[r2], inv1, assume_closed)
            if sup is not None and sup >= min_support and \
                    not (r1 == r2 and p1 == inv1):
                emit([Literal(True, "?a", r1, "?b")],
                     Literal(True, "?b", r2, "?a"), sup, "S3",
                     f"{r1}(x,y) => {r2}(y,x)")

    for r1, p1 in bp.items():                                     # S4
        right = defaultdict(set)
        for s, o in p1:
            right[o].add(s)
        for r2, p2 in bp.items():
            joined = {(x, z) for (y, z) in p2 for x in right.get(y, ())}
            if len(joined) < min_support:
                continue
            for r3 in bp:
                sup = _ok(bp[r3], bn[r3], joined, assume_closed)
                if sup is not None and sup >= min_support:
                    emit([Literal(True, "?a", r1, "?b"),
                          Literal(True, "?b", r2, "?c")],
                         Literal(True, "?a", r3, "?c"), sup, "S4",
                         f"{r1}(x,y) & {r2}(y,z) => {r3}(x,z)")
    return out


def mine_exclusions(wm, min_support: int = 3) -> List[Tuple[str, str, int]]:
    """E1: attribute predicates with disjoint, well-supported extensions."""
    up, *_ = _partition_facts(wm.facts)
    preds = sorted(up)
    out = []
    for i, a in enumerate(preds):
        for b in preds[i + 1:]:
            if (len(up[a]) >= min_support and len(up[b]) >= min_support
                    and not (up[a] & up[b])):
                out.append((a, b, len(up[a] | up[b])))
    return out


def verify_rules(wm, rules: List[Rule]) -> Tuple[List[Rule], List[str]]:
    """Living-rule check: re-validate each induced rule against the CURRENT
    facts with strict (observed-counterexample) semantics; retract failures.
    Returns (surviving_rules, retraction_reports)."""
    up, un, bp, bn = _partition_facts(wm.facts)
    keep, retracted = [], []

    def holds_neg(lit: Literal, binding) -> bool:
        g = lit.subst(binding)
        hit = wm.facts.get(g.key())
        return hit is not None and hit[0] != g.pos

    for rule in rules:
        bad = None
        for binding in _bindings(rule, bp, up):
            if all(_matches(c, binding, up, bp) for c in rule.conds) \
                    and holds_neg(rule.concl, binding):
                bad = binding
                break
        if bad is None:
            keep.append(rule)
        else:
            retracted.append(f"RETRACTED {rule.origin}: counterexample "
                             f"{ {k: v for k, v in bad.items()} }")
    return keep, retracted


def _matches(lit: Literal, binding, up, bp) -> bool:
    g = lit.subst(binding)
    if g.obj is None:
        return g.subj in up.get(g.pred, ())
    return (g.subj, g.obj) in bp.get(g.pred, ())


def _bindings(rule: Rule, bp, up):
    """Join-based candidate bindings for a rule (drives verification)."""
    first = rule.conds[0]
    if first.obj is None:
        for s in up.get(first.pred, ()):
            yield {first.subj: s}
    else:
        for s, o in bp.get(first.pred, ()):
            base = {first.subj: s, first.obj: o}
            if len(rule.conds) == 1:
                yield base
                continue
            second = rule.conds[1]
            for s2, o2 in bp.get(second.pred, ()):
                b = dict(base)
                ok = True
                for var, val in ((second.subj, s2), (second.obj, o2)):
                    if var in b and b[var] != val:
                        ok = False
                        break
                    b[var] = val
                if ok:
                    yield b
