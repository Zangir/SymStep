#!/usr/bin/env python3
"""Tier-1/2 LLM oracle: compile a clue to IR when templates fail, and
re-compile clues implicated by an UNSAT core (repair). Self-contained —
talks to the Claude Code CLI, no API key needed."""
from __future__ import annotations
import json, os, re, subprocess, time, glob
from typing import Dict, List, Optional

from ..engine.ir import build_constraint, IRError

LLM_CALLS = {"n": 0}          # global counter for benchmarking


def _find_claude_bin() -> str:
    if env := os.environ.get("CLAUDE_BIN"):
        return env
    candidates = [
        os.path.expanduser("~/.claude/local/claude"),
        os.path.expanduser("~/.local/bin/claude"),
        "/usr/local/bin/claude",
        *glob.glob(os.path.expanduser(
            "~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude")),
        *glob.glob(os.path.expanduser(
            "~/.cursor/extensions/anthropic.claude-code-*/resources/native-binary/claude")),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return "claude"


CLAUDE_BIN = _find_claude_bin()
MODEL = os.environ.get("GRAPHSTEP_MODEL", "haiku")

IR_CHEATSHEET = """\
Constraint JSON types (variables take integer positions as values):
  {"type":"is","var":V,"value":k}        V is at position k
  {"type":"is_not","var":V,"value":k}    V is not at position k
  {"type":"same","a":V,"b":W}            V and W are at the same position
  {"type":"diff","a":V,"b":W}            V and W are at different positions
  {"type":"less","a":V,"b":W}            pos(V) < pos(W)
  {"type":"offset","a":V,"b":W,"k":1}    pos(V) = pos(W) + 1  (V directly right of W)
  {"type":"absdiff","a":V,"b":W,"k":1}   |pos(V)-pos(W)| = 1  (neighbors)
  {"type":"or","clauses":[...]}  {"type":"and","clauses":[...]}
  {"type":"not","c":...}         {"type":"implies","if":...,"then":...}
  {"type":"count","vars":[...],"value":k,"op":"==","k":n}
  {"type":"table","vars":[...],"allowed":[[...],...]}   any finite relation"""


def call_llm(prompt: str, retries: int = 3, timeout: int = 120) -> str:
    for attempt in range(retries):
        try:
            LLM_CALLS["n"] += 1
            result = subprocess.run(
                [CLAUDE_BIN, "--print", "--model", MODEL,
                 "--no-session-persistence"],
                input=prompt, capture_output=True, text=True, timeout=timeout)
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    return ""


def _extract_json(text: str):
    """Pull the outermost JSON array/object out of an LLM reply: try every
    bracket start position in order, so the earliest (outermost) structure
    wins and an inner fragment is only a fallback."""
    text = re.sub(r"```(?:json)?", "", text)
    starts = [i for i, ch in enumerate(text) if ch in "[{"]
    for start in starts:
        opener = text[start]
        closer = "]" if opener == "[" else "}"
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def llm_compile_clue(clue: str, variables: Dict[str, list],
                     entity_index: Dict[str, int],
                     context: str = "", attempts: int = 2) -> Optional[List[dict]]:
    """Compile one clue to IR constraints via the LLM; validated before accept."""
    ent_doc = ", ".join(f"{e}={i}" for e, i in entity_index.items()) or "none"
    err_note = ""
    for _ in range(attempts):
        prompt = f"""Translate ONE puzzle clue into constraint JSON.

Variables (each takes one integer position): {sorted(variables)}
Positions run 1..{max(len(d) for d in variables.values())}. Entity name -> index: {ent_doc}
{IR_CHEATSHEET}
{context}{err_note}
Clue: "{clue}"

Reply with ONLY a JSON array of constraint objects (usually one)."""
        reply = call_llm(prompt)
        specs = _extract_json(reply)
        if specs is None:
            err_note = "\nYour previous reply was not valid JSON. JSON only."
            continue
        if isinstance(specs, dict):
            specs = [specs]
        try:
            for spec in specs:
                build_constraint(spec, variables)     # validate
            for spec in specs:
                spec["origin"] = clue.strip()
            return specs
        except IRError as e:
            err_note = f"\nYour previous attempt was invalid: {e}. Fix it."
    return None


def llm_compile_problem(text: str, attempts: int = 3) -> Optional[dict]:
    """Compile an ENTIRE reasoning problem (prose) into a full IR dict:
    {"variables": {...}, "constraints": [...]}. Used for domains with no
    template compiler (e.g. AR-LSAT setups). Validated before acceptance."""
    from ..engine.ir import problem_from_ir
    from ..engine.core import Engine
    err_note = ""
    for _ in range(attempts):
        prompt = f"""Model this reasoning problem as finite-domain constraints.

Output JSON: {{"encoding": "<one paragraph: what each variable means and what
every domain value stands for>", "variables": {{name: [values...]}},
"constraints": [...]}}
- Choose variables so every stated condition is expressible.
- Domains: use integers when order/position matters (Mon AM=1, Mon PM=2, ...).
  If an entity may be unselected/unscheduled, add 0 ("none") to its domain
  and express selection counts with "count" constraints.
{IR_CHEATSHEET}
- "less"/"offset"/"absdiff" need integer domains.
- Encode EVERY condition, including implicit ones (e.g. "exactly two per
  day", uniqueness of occupied slots — use count / alldiff / diff).
  For slot-sharing (two entities per day), prefer integer slot codes so
  uniqueness is one alldiff over selected entities.
- The "encoding" paragraph is essential: it is the only key later readers
  have to interpret the integer codes.
{err_note}
PROBLEM:
{text}

Reply with ONLY the JSON object."""
        reply = call_llm(prompt, timeout=180)
        ir = _extract_json(reply)
        if not isinstance(ir, dict) or "variables" not in ir:
            err_note = "\nPrevious reply was not a valid IR object. JSON only."
            continue
        try:
            prob = problem_from_ir(ir)
            res = Engine(prob).solve(max_solutions=1)
            if res.status == "UNSAT":
                err_note = ("\nPrevious model was over-constrained: it has NO "
                            "solution at all, so a condition must be encoded "
                            f"wrong. Engine says: {res.explanation[:400]}")
                continue
            return ir
        except (IRError, Exception) as e:
            err_note = f"\nPrevious attempt failed validation: {e}. Fix it."
    return None


def llm_compile_choice(choice: str, variables: Dict[str, list],
                       context: str, attempts: int = 2) -> Optional[List[dict]]:
    """Compile one MCQ answer option into IR constraints over an existing
    variable set (for entailment queries)."""
    err_note = ""
    for _ in range(attempts):
        prompt = f"""Given these variables (from a constraint model of a puzzle):
{json.dumps(variables, default=str)}

Context (how the model encodes the problem):
{context}
{IR_CHEATSHEET}
{err_note}
Translate this answer option into constraint JSON over those variables:
"{choice}"

Reply with ONLY a JSON array of constraint objects."""
        reply = call_llm(prompt, timeout=120)
        specs = _extract_json(reply)
        if specs is None:
            err_note = "\nPrevious reply was not valid JSON. JSON only."
            continue
        if isinstance(specs, dict):
            specs = [specs]
        try:
            for spec in specs:
                build_constraint(spec, variables)
            return specs
        except IRError as e:
            err_note = f"\nPrevious attempt was invalid: {e}. Fix it."
    return None
