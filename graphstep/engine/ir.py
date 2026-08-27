#!/usr/bin/env python3
"""GraphStep IR: a JSON logic representation any reasoning task compiles into.

A problem is {"variables": {name: [values]}, "constraints": [<constraint>]}.
Constraint schema (types map 1:1 onto graphstep.constraints):

  {"type":"is",      "var":V, "value":x}          V == x
  {"type":"is_not",  "var":V, "value":x}          V != x
  {"type":"same",    "a":V, "b":W}                V == W
  {"type":"diff",    "a":V, "b":W}                V != W
  {"type":"less",    "a":V, "b":W}                V <  W   (numeric)
  {"type":"geq",     "a":V, "b":W}                V >= W   (numeric)
  {"type":"offset",  "a":V, "b":W, "k":k}         V == W + k
  {"type":"absdiff", "a":V, "b":W, "k":k}         |V - W| == k
  {"type":"alldiff", "vars":[...]}
  {"type":"table",   "vars":[...], "allowed":[[...], ...]}
  {"type":"count",   "vars":[...], "value":x, "op":"=="|"<="|">=", "k":k}
  {"type":"not",     "c":<constraint>}
  {"type":"or",      "clauses":[<constraint>, ...]}
  {"type":"and",     "clauses":[<constraint>, ...]}
  {"type":"implies", "if":<constraint>, "then":<constraint>}
  {"type":"iff",     "a":<constraint>, "b":<constraint>}

`build_constraint` raises IRError with a message precise enough to hand back
to an LLM for self-repair.
"""
from __future__ import annotations
from typing import Dict, List, Any
from .core import Problem, Constraint
from . import constraints as C


class IRError(ValueError):
    pass


def _need(spec: dict, key: str, ctx: str):
    if key not in spec:
        raise IRError(f"constraint {ctx}: missing required field {key!r} "
                      f"in {spec}")
    return spec[key]


def _var(name: Any, variables: Dict[str, list], ctx: str) -> str:
    if name not in variables:
        close = [v for v in variables
                 if str(v).lower() == str(name).lower().replace(" ", "")]
        hint = f" (did you mean {close[0]!r}?)" if close else \
               f" (known variables: {sorted(map(str, variables))[:12]}…)"
        raise IRError(f"constraint {ctx}: unknown variable {name!r}{hint}")
    return name


def _val(value: Any, var: str, variables: Dict[str, list], ctx: str) -> Any:
    dom = variables[var]
    if value in dom:
        return value
    for v in dom:  # case-insensitive convenience
        if str(v).lower() == str(value).lower():
            return v
    raise IRError(f"constraint {ctx}: value {value!r} is not in the domain of "
                  f"{var!r} = {sorted(map(str, dom))}")


def build_constraint(spec: dict, variables: Dict[str, list],
                     origin: str = None) -> Constraint:
    if not isinstance(spec, dict):
        raise IRError(f"constraint must be a JSON object, got {spec!r}")
    t = _need(spec, "type", "?")
    ctx = t

    if t == "is":
        var = _var(_need(spec, "var", ctx), variables, ctx)
        return C.Is(var, _val(_need(spec, "value", ctx), var, variables, ctx),
                    origin)
    if t == "is_not":
        var = _var(_need(spec, "var", ctx), variables, ctx)
        return C.IsNot(var, _val(_need(spec, "value", ctx), var, variables, ctx),
                       origin)
    if t in ("same", "diff", "less", "geq"):
        a = _var(_need(spec, "a", ctx), variables, ctx)
        b = _var(_need(spec, "b", ctx), variables, ctx)
        cls = {"same": C.SameValue, "diff": C.DiffValue,
               "less": C.Less, "geq": C.GreaterEq}[t]
        return cls(a, b, origin)
    if t in ("offset", "absdiff", "leq_offset", "geq_offset"):
        a = _var(_need(spec, "a", ctx), variables, ctx)
        b = _var(_need(spec, "b", ctx), variables, ctx)
        k = int(_need(spec, "k", ctx))
        cls = {"offset": C.Offset, "absdiff": C.AbsDiff,
               "leq_offset": C.OffsetLeq, "geq_offset": C.OffsetGeq}[t]
        return cls(a, b, k, origin)
    if t == "alldiff":
        vs = [_var(v, variables, ctx) for v in _need(spec, "vars", ctx)]
        return C.AllDifferent(vs, origin)
    if t == "table":
        vs = [_var(v, variables, ctx) for v in _need(spec, "vars", ctx)]
        return C.Table(vs, [tuple(row) for row in _need(spec, "allowed", ctx)],
                       origin)
    if t == "count":
        vs = [_var(v, variables, ctx) for v in _need(spec, "vars", ctx)]
        op = _need(spec, "op", ctx)
        if op not in ("==", "<=", ">="):
            raise IRError(f"constraint count: op must be ==, <= or >=, got {op!r}")
        return C.Count(vs, _need(spec, "value", ctx), op,
                       int(_need(spec, "k", ctx)), origin)
    if t == "not":
        return C.Not(build_constraint(_need(spec, "c", ctx), variables, origin),
                     origin)
    if t in ("or", "and"):
        clauses = [build_constraint(c, variables, origin)
                   for c in _need(spec, "clauses", ctx)]
        if len(clauses) < 2:
            raise IRError(f"constraint {t}: needs >= 2 clauses")
        return (C.Or if t == "or" else C.AndAll)(clauses, origin)
    if t == "implies":
        return C.Implies(build_constraint(_need(spec, "if", ctx), variables, origin),
                         build_constraint(_need(spec, "then", ctx), variables, origin),
                         origin)
    if t == "iff":
        return C.Iff(build_constraint(_need(spec, "a", ctx), variables, origin),
                     build_constraint(_need(spec, "b", ctx), variables, origin),
                     origin)
    raise IRError(f"unknown constraint type {t!r}; valid types: is, is_not, "
                  f"same, diff, less, geq, offset, absdiff, alldiff, table, "
                  f"count, not, or, and, implies, iff")


def problem_from_ir(ir: dict) -> Problem:
    variables = ir["variables"]
    cons = [build_constraint(spec, variables,
                             origin=spec.get("origin"))
            for spec in ir.get("constraints", [])]
    return Problem(variables, cons)


IR_DOC = __doc__  # handed to the LLM as the target-language spec
