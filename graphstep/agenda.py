#!/usr/bin/env python3
"""The solver loop: a general agenda of goals worked one at a time until a
certified result or a named refusal — the knowledge escalation ladder run
per-part instead of per-task.

    AGENDA = [ the task as one root goal ]
    while open goals remain (within budget):
        pick the most constrained open goal
        GROUND     known rows / the one-shot pipeline answer it
        DECOMPOSE  split it by structure into subgoals
        RETRIEVE   ask an external source, admit typed rows
        DERIVE     bounded, spec-guided composition of ATOMS for this
                   subgoal only; every candidate faces the oracle
        VERIFY     the domain's oracle judges (tests / proofs / closure)
        REPAIR     a failure names the guilty subgoal; reopen exactly it
        LEARN      a verified composition becomes a new row (session scope;
                   persistence must be earned: verified + reusable + stamped)
        REFUSE     nothing applies -> record the named gap and close

Every iteration proves a goal, adds knowledge, or closes a goal honestly —
the loop always terminates. Nothing here knows any domain: GROUND is the
generic six-step pipeline, DERIVE composes whatever atom rows exist, and
the oracles come from the artifact type. The whole loop is bookkeeping
(Goal, Blackboard) around machinery that already exists.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import unified
from .reading import kb
from .engine.sandbox import run_tests


@dataclass
class Goal:
    desc: str
    status: str = "OPEN"          # OPEN | PROVEN | REFUSED
    artifact: object = None
    reason: Optional[str] = None


@dataclass
class Blackboard:
    sample: dict
    question: Optional[str]
    evidence: object = None
    sig: object = None
    frame: object = None
    trace: List[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.trace.append(msg)


# ------------------------------------------------------------------ DERIVE

def _derive_program(bb: Blackboard) -> Optional[dict]:
    """Spec-guided atom composition for the program algebra: the frame's
    grounded pieces license candidate compositions; the sandbox oracle
    judges every one. Returns {"code":..., "atoms":...} or None."""
    sig, frame = bb.sig, bb.frame
    if sig is None or frame is None:
        return None
    filt = kb.match_key(("ATOM", "FILTER"))
    if filt is None:
        return None
    params = [f"arg{i}" for i in range(sig.arity)]
    tried = 0
    # REMOVE over items with a grounded property -> filter OUT that property
    if frame.op == "REMOVE" and frame.item_mods:
        pred_rows = [r for mod in frame.item_mods for r in kb.KB
                     if r.symbol == "PRED" and r.pattern == mod and r.payload]
        src_cands = [i for i, t in enumerate(sig.arg_types)
                     if t in ("LIST", "STRING")]
        for prow in pred_rows:
            for si in src_cands:
                keep = f"not ({prow.payload.format(x='x')})"
                body = filt.payload.format(src=params[si], keep=keep)
                code = (f"def {sig.name}({', '.join(params)}):\n"
                        f"    return {body}")
                tried += 1
                res = run_tests(code, bb.evidence.assert_texts)
                bb.log(f"DERIVE: filter(not {prow.sig['name']}) over "
                       f"{params[si]} -> "
                       f"{'PASS' if res['ok'] else res['error']}")
                if res["ok"]:
                    return {"code": code, "atoms":
                            ["FILTER", prow.sig["name"]],
                            "tried": tried}
    return None


def _learn(bb: Blackboard, derived: dict) -> None:
    """A verified derivation becomes a session-scope row so the next task
    of the same shape takes the fast path. Persistence must be earned —
    this writes to the in-memory store only, provenance-stamped."""
    key = ("DERIVED", bb.frame.op, tuple(sorted(bb.frame.item_mods)),
           tuple(bb.sig.arg_types))
    if kb.match_key(key) is None:
        template = derived["code"].replace(f"def {bb.sig.name}(",
                                           "def {name}(")
        kb.add(kb.Row(key, "BLOCK", sig={"atoms": derived["atoms"]},
                      payload=template, provenance="derived:agenda",
                      confidence=0.9))
        bb.log(f"LEARN: cached derived block {key} (session scope)")


def _ground_learned(bb: Blackboard) -> Optional[dict]:
    """Fast path: a previously derived-and-verified block for this exact
    shape already exists as a row."""
    if bb.sig is None or bb.frame is None:
        return None
    key = ("DERIVED", bb.frame.op, tuple(sorted(bb.frame.item_mods)),
           tuple(bb.sig.arg_types))
    row = kb.match_key(key)
    if row is None:
        return None
    code = row.payload.replace("{name}", bb.sig.name)
    res = run_tests(code, bb.evidence.assert_texts)
    if res["ok"]:
        bb.log(f"GROUND: learned row {key} answers directly "
               f"(provenance {row.provenance})")
        return {"code": code, "atoms": row.sig.get("atoms", [])}
    return None


# ------------------------------------------------------------------ loop

def solve_loop(sample: dict, question: Optional[str] = None,
               budget: int = 12) -> dict:
    """The general solver loop. GROUND first (the one-shot pipeline — one
    iteration when knowledge suffices); on a program-shaped refusal,
    DECOMPOSE into read-spec / derive / verify subgoals and work them."""
    bb = Blackboard(sample=sample, question=question)
    agenda: List[Goal] = [Goal("solve the task")]
    rec: Dict = {}

    for step in range(budget):
        open_goals = [g for g in agenda if g.status == "OPEN"]
        if not open_goals:
            break
        goal = open_goals[0]

        if goal.desc == "solve the task":
            rec = unified.solve(sample, question=question)
            bb.log(f"GROUND: one-shot pipeline -> {rec['status']}")
            if rec["status"] in ("SOLVED", "CONCLUSIONS", "CAPTURED",
                                 "EXCLUDED", "AMBIGUOUS"):
                goal.status = "PROVEN" if rec["status"] == "SOLVED" \
                    else "REFUSED" if rec["status"] == "EXCLUDED" \
                    else "PROVEN"
                break
            # program-shaped? gather sig+frame evidence for decomposition
            bb.evidence = unified.recognize(sample)
            if question:
                bb.evidence.question = question
            if bb.evidence.assert_texts:
                try:
                    bb.sig = unified.parse_signature(bb.evidence.assert_texts)
                except (ValueError, SyntaxError) as e:
                    goal.status, goal.reason = "REFUSED", f"signature: {e}"
                    break
                for st, _ in bb.evidence.statements:
                    try:
                        bb.frame = unified.parse_frame(st)
                        break
                    except ValueError as e:
                        goal.reason = f"frame: {e}"
                if bb.frame is None:
                    goal.status = "REFUSED"
                    bb.log(f"REFUSE: {goal.reason}")
                    break
                bb.log(f"DECOMPOSE: spec read (op={bb.frame.op}, "
                       f"mods={bb.frame.item_mods}) -> "
                       f"subgoal: derive a program")
                goal.status = "PROVEN"          # the reading subpart
                agenda.append(Goal("derive a certified program"))
            else:
                goal.status = "REFUSED"
                goal.reason = rec.get("reasons", ["no route"])[0] \
                    if rec.get("reasons") else "no route"
                break

        elif goal.desc == "derive a certified program":
            derived = _ground_learned(bb) or _derive_program(bb)
            if derived:
                goal.status = "PROVEN"
                goal.artifact = derived
                _learn(bb, derived)
                rec.update(status="SOLVED", code=derived["code"],
                           reasons=[],
                           derivation={"atoms": derived["atoms"]})
            else:
                goal.status = "REFUSED"
                goal.reason = ("no atom composition passes the oracle "
                               f"(op={bb.frame.op}, mods={bb.frame.item_mods})")
                bb.log(f"REFUSE: {goal.reason}")
                rec.setdefault("reasons", []).append(goal.reason)

    rec["loop"] = {"iterations": step + 1, "trace": bb.trace,
                   "goals": [(g.desc, g.status) for g in agenda]}
    return rec
