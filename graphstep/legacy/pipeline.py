#!/usr/bin/env python3
"""GraphStep pipeline: text -> logic -> propagate/search, LLM only on demand.

Escalation ladder:
  Tier 0  deterministic template compile (zero LLM calls)
  Tier 1  per-clue LLM compile for clues the templates cannot read
  Repair  on UNSAT: extract the unsat core, re-compile exactly the clues in
          the core with the LLM (told about the conflict), re-solve
The result is certified: SOLVED means a unique solution satisfying every
compiled constraint, with a ledger-backed explanation channel throughout.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from ..engine.core import Engine
from ..engine.ir import problem_from_ir, IRError
from ..reading.compile_text import Inventory, compile_clues
from . import llm as llm_mod


class PipelineResult:
    def __init__(self, status, solution, stats, explanation=""):
        self.status = status            # SOLVED | UNSAT | AMBIGUOUS | UNKNOWN | PARSE_FAIL
        self.solution = solution        # {value_var: index} or None
        self.stats = stats
        self.explanation = explanation


def solve_text_puzzle(clues: List[str], inv: Inventory, positional: bool,
                      allow_llm: bool = True, repair_rounds: int = 2,
                      verbose: bool = False) -> PipelineResult:
    stats = {"clues": len(clues), "template_parsed": 0, "syntax_parsed": 0,
             "llm_parsed": 0, "llm_calls_before": llm_mod.LLM_CALLS["n"],
             "repairs": 0, "unparsed": 0}
    variables = inv.variables()

    # ---- Tier 0: templates ------------------------------------------------
    specs, unparsed = compile_clues(clues, inv, positional)
    stats["template_parsed"] = len(clues) - len(unparsed)

    # ---- Tier 0.5: syntax-guided compile (no LLM) --------------------------
    if unparsed:
        from ..reading.syntax_tier import compile_with_syntax
        for clue in list(unparsed):
            try:
                got = compile_with_syntax(clue, inv, positional)
            except Exception:
                got = None
            if got:
                specs.extend(got)
                unparsed.remove(clue)
                stats["syntax_parsed"] += 1

    # ---- Tier 1: LLM for the stragglers -----------------------------------
    for clue in list(unparsed):
        if not allow_llm:
            continue
        got = llm_mod.llm_compile_clue(clue, variables, inv.entity_index)
        if got:
            specs.extend(got)
            unparsed.remove(clue)
            stats["llm_parsed"] += 1
    stats["unparsed"] = len(unparsed)
    if unparsed:
        stats["llm_calls"] = llm_mod.LLM_CALLS["n"] - stats.pop("llm_calls_before")
        return PipelineResult("PARSE_FAIL", None, stats,
                              f"unparsed clues: {unparsed}")

    ir = {"variables": variables,
          "constraints": inv.base_constraints() + specs}

    # ---- solve, with unsat-core repair loop --------------------------------
    for round_ in range(repair_rounds + 1):
        engine = Engine(problem_from_ir(ir))
        res = engine.solve()
        if res.status != "UNSAT" or not allow_llm or round_ == repair_rounds:
            break
        # repair: find the conflicting clue set, re-compile just those clues
        core = engine.unsat_core()
        core_clues = sorted({c.origin for c in core
                             if c.origin and not c.origin.startswith("[")})
        if not core_clues:
            break
        stats["repairs"] += 1
        if verbose:
            print(f"    [repair {round_+1}] core clues: {core_clues}")
        keep = [s for s in specs if s.get("origin") not in core_clues]
        repaired = []
        ok = True
        for clue in core_clues:
            got = llm_mod.llm_compile_clue(
                clue, variables, inv.entity_index,
                context=(f"\nNote: an earlier literal reading of these clues "
                         f"was mutually inconsistent: {core_clues}. Read this "
                         f"clue extra carefully.\n"))
            if got:
                repaired.extend(got)
            else:
                ok = False
        if not ok:
            break
        specs = keep + repaired
        ir = {"variables": variables,
              "constraints": inv.base_constraints() + specs}

    stats["llm_calls"] = llm_mod.LLM_CALLS["n"] - stats.pop("llm_calls_before")
    stats.update(res.stats)
    solution = res.solutions[0] if res.solutions else None
    return PipelineResult(res.status, solution, stats, res.explanation)


def solution_to_entity_view(solution: Dict[str, int],
                            inv: Inventory) -> Dict[str, Dict[str, str]]:
    """{value_var: index} -> {entity: {attr: value}} for benchmark scoring."""
    out = {e: {} for e in inv.entities}
    for val, idx in solution.items():
        attr = inv.value_attr.get(val)
        if attr is None:
            continue
        if 1 <= idx <= len(inv.entities):
            out[inv.entities[idx - 1]][attr] = val
    return out
