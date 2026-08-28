#!/usr/bin/env python3
"""The code reader: retrieved CODE passes through our own reader — exactly
as retrieved prose must pass through semgraph — and becomes typed rows,
never an opaque replay.

A gap names the missing meaning; a STRUCTURED query fetches candidate
functions for THAT word; this module decomposes them:

    def is_coprime(a, b): return gcd(a, b) == 1
        -> BPRED row 'coprime', payload "(math.gcd({a}, {b}) == 1)"
    def double(x): return 2 * x
        -> FUNC row, payload "(2 * ({src}))"
    acc = 0; for v in xs: if cond(v): acc += 1; return acc
        -> the pred inside a COUNT structure

Extracted pieces are candidate MEANINGS — admitted through the tribunal
with provenance "mined:<source>", composed by the same compiler under the
same licenses, and judged by the same oracles. Code structures the reader
does not recognize are left alone (the capture-tier discipline): nothing
unrecognized is ever guessed into a row.

Safety: an extracted expression may reference only the function's own
parameters and a whitelist of pure builtins/module calls; anything else
(I/O, attribute chains on unknowns, globals) disqualifies the extraction.
"""
from __future__ import annotations
import ast
from typing import List, Optional

from . import kb

_SAFE_CALLS = {"len", "str", "int", "float", "abs", "sum", "min", "max",
               "sorted", "list", "tuple", "set", "range", "bool", "round",
               "all", "any", "enumerate", "zip", "reversed"}
_SAFE_MODULES = {"math", "gcd", "sqrt", "floor", "ceil", "factorial",
                 "prod", "lcm"}


def _expr_names(node) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _calls_safe(node) -> bool:
    for c in ast.walk(node):
        if isinstance(c, ast.Call):
            f = c.func
            if isinstance(f, ast.Name):
                if f.id not in _SAFE_CALLS | _SAFE_MODULES:
                    return False
            elif isinstance(f, ast.Attribute):
                base = f.value
                if not (isinstance(base, ast.Name)
                        and base.id in _SAFE_MODULES):
                    return False
            else:
                return False
        if isinstance(c, (ast.Await, ast.Yield, ast.YieldFrom)):
            return False
    return True


def _rewrite(expr: str, imports: List[str]) -> str:
    """Qualify bare module references so the payload is self-contained."""
    for mod in imports:
        expr = expr.replace(f"{mod}.", f"__import__('{mod}').")
    # bare gcd/sqrt style calls from `from math import gcd`
    for name in ("gcd", "sqrt", "floor", "ceil", "factorial", "lcm"):
        expr = expr.replace(f" {name}(", f" __import__('math').{name}(")
        if expr.startswith(f"{name}("):
            expr = f"__import__('math').{name}(" + expr[len(name) + 1:]
    return expr


class _Rename(ast.NodeTransformer):
    def __init__(self, mapping):
        self.m = mapping

    def visit_Name(self, node):
        if node.id in self.m:
            return ast.copy_location(ast.Name(id=self.m[node.id],
                                              ctx=node.ctx), node)
        return node


def _holed(expr_node, mapping: dict, imports) -> str:
    """Substitute parameter NAMES (AST-level, no substring accidents) with
    hole placeholders, then render."""
    ph = {name: f"__H{i}__" for i, name in enumerate(mapping)}
    node = _Rename(ph).visit(ast.parse(ast.unparse(expr_node),
                                       mode="eval").body)
    text = _rewrite(ast.unparse(node), imports)
    for name, hole in mapping.items():
        text = text.replace(ph[name], hole)
    return text


def _is_boolish(node) -> bool:
    return isinstance(node, (ast.Compare, ast.BoolOp)) or (
        isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not))


def read_functions(code: str, word: str,
                   provenance: str) -> List[kb.Row]:
    """Decompose every function in `code` into candidate meaning rows for
    `word`. Recognizers are structural and closed; unrecognized shapes
    yield nothing."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    imports = [n.names[0].name for n in ast.walk(tree)
               if isinstance(n, ast.Import) and n.names]
    out: List[kb.Row] = []
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        params = [a.arg for a in fn.args.args]
        body = [st for st in fn.body
                if not isinstance(st, (ast.Expr,))]      # drop docstrings

        # -- shape 1: single `return <expr>` over the parameters ----------
        if len(body) == 1 and isinstance(body[0], ast.Return) \
                and body[0].value is not None:
            expr_node = body[0].value
            if not _calls_safe(expr_node):
                continue
            free = _expr_names(expr_node) - set(params) - _SAFE_CALLS \
                - _SAFE_MODULES
            if free:
                continue
            if len(params) == 1:
                boolish = _is_boolish(expr_node)
                hole = _holed(expr_node,
                              {params[0]: "{x}" if boolish else "{src}"},
                              imports)
                if boolish:
                    out.append(kb.Row(word, "PRED",
                                      sig={"name": f"MINED_{word.upper()}",
                                           "arg": "ANY"},
                                      payload=f"({hole})",
                                      provenance=provenance,
                                      confidence=0.6))
                else:
                    out.append(kb.Row(word, "FUNC", payload=f"({hole})",
                                      provenance=provenance,
                                      confidence=0.6))
            elif len(params) == 2:
                hole = _holed(expr_node, {params[0]: "{a}",
                                          params[1]: "{b}"}, imports)
                sym = "BPRED" if _is_boolish(expr_node) else "BINOP"
                out.append(kb.Row(word, sym, payload=f"({hole})",
                                  provenance=provenance, confidence=0.6))

        # -- shape 2: counting / summing loop with a condition -------------
        # acc = <0>; for v in seq: if cond: acc += (1 | v); return acc
        if (len(body) == 3 and isinstance(body[0], ast.Assign)
                and isinstance(body[1], ast.For)
                and isinstance(body[2], ast.Return)
                and isinstance(body[1].iter, ast.Name)
                and body[1].iter.id in params
                and len(body[1].body) == 1
                and isinstance(body[1].body[0], ast.If)
                and len(params) >= 1):
            iff = body[1].body[0]
            if (len(iff.body) == 1 and isinstance(iff.body[0], ast.AugAssign)
                    and isinstance(iff.body[0].op, ast.Add)
                    and _calls_safe(iff.test)
                    and isinstance(body[1].target, ast.Name)):
                v = body[1].target.id
                free = _expr_names(iff.test) - {v} - set(params) \
                    - _SAFE_CALLS - _SAFE_MODULES
                if not free:
                    seqp = body[1].iter.id
                    hole = _holed(iff.test, {v: "{x}", seqp: "{seq}"},
                                  imports)
                    sym = "CPRED" if "{seq}" in hole else "PRED"
                    sig = {"name": f"MINED_{word.upper()}"}
                    if sym == "PRED":
                        sig["arg"] = "ANY"
                    out.append(kb.Row(word, sym, sig=sig,
                                      payload=f"({hole})",
                                      provenance=provenance,
                                      confidence=0.6))
    return out
