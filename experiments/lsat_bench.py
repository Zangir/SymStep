#!/usr/bin/env python3
"""
lsat_bench.py -- LSAT Analytical Reasoning (AR-LSAT) benchmark for SymStep.

Source: AGIEval benchmark (Zhong et al., 2023), LSAT Analytical Reasoning section.
        230 questions involving scheduling, ordering, and grouping under constraints.
        Models: GPT-4 ~60%, Claude 3 ~55%, human ~90% — genuinely unsolved.

Each question provides a scenario with named entities and a set of hard constraints,
then asks which of five options (A-E) satisfies all constraints.  This is structurally
identical to the grid-puzzle domain SymStep was designed for: constraint extraction,
propagation, and contradiction detection drive the reasoning.

SymStep adaptation
------------------
We give the LLM a DEDUCE-based protocol for constraint satisfaction:
  DEDUCE: constraint_id = fact_or_elimination
Each step records a logical deduction (constraint noted, candidate eliminated, etc.)
An ArithmeticTracker-style LogicTracker confirms each step and detects contradictions
(asserting the same fact with a conflicting value).
CONCLUDE: answer = <letter>

Usage:
  python lsat_bench.py               # run all 230 questions
  python lsat_bench.py --n 50        # run 50 questions
  python lsat_bench.py --dry-run     # print sample, no LLM calls

Reference:
  Zhong et al., "AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models",
  arXiv:2304.06364, 2023.
"""

import os, sys, re, json, time, argparse, random
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from symstep import call_llm
import symstep as _sym
from ci_utils import wilson_ci

# ── LogicTracker: deduction register for constraint satisfaction ───────────────

class LogicTracker:
    """Records logical deductions; detects contradictions on re-assertion."""

    def __init__(self):
        self.facts: Dict[str, str] = {}
        self.steps: List[Tuple[str, str]] = []

    def apply(self, key: str, value: str) -> Tuple[bool, str]:
        key_n = key.strip().lower()
        value_n = value.strip().lower()
        if key_n in self.facts:
            prev = self.facts[key_n]
            if prev != value_n:
                return False, (
                    f"Contradiction: '{key}' was previously stated as '{prev}' "
                    f"but now stated as '{value}'"
                )
            return True, f"OK (already known: {key} = {prev})"
        self.facts[key_n] = value_n
        self.steps.append((key, value))
        return True, f"OK: {key} = {value}"

    def guidance(self) -> str:
        n = len(self.steps)
        if n == 0:
            return "[Hint] Start by extracting the constraints from the scenario."
        if n < 3:
            return "[Hint] Continue applying constraints to eliminate invalid options."
        return "[Hint] Apply remaining constraints and conclude with CONCLUDE: answer = <letter>."

# ── System prompt for LSAT-AR SymStep ─────────────────────────────────────────

LSAT_SYSTEM = """\
Solve this LSAT analytical reasoning question step by step.

For each logical deduction (constraint extracted, option eliminated, placement fixed),
output EXACTLY:
  DEDUCE: label = fact

Rules:
- Use short labels (e.g., george_day = tuesday, kyle_not_morning = true, option_A = eliminated).
- One DEDUCE per turn. Do not output multiple DEDUCE statements at once.
- After each DEDUCE you will receive confirmation or a contradiction report.
- When you are certain of the answer, output:   CONCLUDE: answer = <letter>   (A/B/C/D/E)
- Do NOT guess. Work through every constraint systematically."""

# ── Method implementations ─────────────────────────────────────────────────────

def _gold_letter(gold: List[int], choices: List[str]) -> str:
    """Convert gold index list to answer letter."""
    idx = gold[0] if gold else 0
    letters = "ABCDE"
    return letters[idx] if idx < len(letters) else "A"


def _format_choices(choices: List[str]) -> str:
    return "\n".join(choices)


def _extract_letter(text: str) -> Optional[str]:
    for pat in [
        r"(?:answer|choice|select|option)\s*(?:is|:)?\s*\**([A-E])\**\b",
        r"\b([A-E])\s*(?:is correct|is the answer|is the best)",
        r"(?:\*\*|```)([A-E])(?:\*\*|```)",
        r"CONCLUDE\s*:\s*answer\s*=\s*([A-E])",
        r"\b([A-E])\s*$",
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    letters = re.findall(r"\b([A-E])\b", text.upper())
    return letters[-1] if letters else None


def _run_direct_lsat(query: str, choices: List[str],
                      correct: str) -> Tuple[bool, int, float]:
    t0 = time.time()
    prompt = (query + f"\n\nOptions:\n{_format_choices(choices)}\n\n"
              "Reason through the constraints carefully and choose the correct option. "
              "State your final answer as a single letter (A, B, C, D, or E).")
    resp = call_llm(prompt)
    got = _extract_letter(resp)
    return got == correct, 1, time.time() - t0


def _run_cot_lsat(query: str, choices: List[str],
                   correct: str) -> Tuple[bool, int, float]:
    t0 = time.time()
    prompt = (query + f"\n\nOptions:\n{_format_choices(choices)}\n\n"
              "Let's think step by step. Extract every constraint, then systematically "
              "eliminate options until one remains. "
              "State your final answer as a single letter (A, B, C, D, or E).")
    resp = call_llm(prompt)
    got = _extract_letter(resp)
    return got == correct, 1, time.time() - t0


def _run_symstep_lsat(query: str, choices: List[str], correct: str,
                       with_guidance: bool = False,
                       max_steps: int = 20) -> Tuple[bool, int, float, int]:
    t0 = time.time()
    tracker = LogicTracker()
    calls = 0
    contradictions = 0
    intro = (query + f"\n\nOptions:\n{_format_choices(choices)}\n\n"
             + LSAT_SYSTEM + "\n\nMake your first DEDUCE now.")
    history = [intro]

    for _ in range(max_steps):
        resp = call_llm("\n\n".join(history))
        calls += 1
        history.append(f"[You]: {resp}")

        # Check CONCLUDE
        m_ans = re.search(r"CONCLUDE\s*:\s*answer\s*=\s*([A-E])", resp, re.IGNORECASE)
        if m_ans:
            got = m_ans.group(1).upper()
            return got == correct, calls, time.time() - t0, contradictions

        # Parse DEDUCE: label = value
        m = re.search(r"DEDUCE\s*:\s*([A-Za-z0-9_\-\s]+?)\s*=\s*(.+?)(?:\n|$)",
                      resp, re.IGNORECASE)
        if not m:
            history.append("[Verifier]: Could not parse. Use:  DEDUCE: label = fact")
            continue

        key = m.group(1).strip()
        val = m.group(2).strip().rstrip(".")
        ok, msg = tracker.apply(key, val)

        if ok:
            feedback = f"[Verifier]: ✓ {msg}"
            if with_guidance:
                feedback += "\n" + tracker.guidance()
        else:
            contradictions += 1
            feedback = f"[Verifier]: ✗ {msg}\nPlease re-examine this step."
        history.append(feedback)

    # Timed out — try to extract letter from last substantive response
    for msg in reversed(history):
        if msg.startswith("[You]:"):
            got = _extract_letter(msg)
            if got:
                return got == correct, calls, time.time() - t0, contradictions
    return False, calls, time.time() - t0, contradictions

# ── Data loading ──────────────────────────────────────────────────────────────

def load_lsat_ar(n: Optional[int] = None, seed: int = 42) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("dmayhem93/agieval-lsat-ar", split="test")
    items = list(ds)
    if n is not None and n < len(items):
        rng = random.Random(seed)
        rng.shuffle(items)
        items = items[:n]
    return items

# ── Experiment runner ─────────────────────────────────────────────────────────

def run_lsat_exp(n: Optional[int] = None,
                 out_path: str = "lsat_results.json") -> dict:
    items = load_lsat_ar(n=n)
    print(f"\n{'='*70}\nAR-LSAT (Analytical Reasoning)  n={len(items)}  model={_sym.MODEL}\n{'='*70}")

    methods = {
        "direct":    lambda it: _run_direct_lsat(
                         it["query"], it["choices"],
                         _gold_letter(it["gold"], it["choices"])) + (0,),
        "cot":       lambda it: _run_cot_lsat(
                         it["query"], it["choices"],
                         _gold_letter(it["gold"], it["choices"])) + (0,),
        "symstep":   lambda it: _run_symstep_lsat(
                         it["query"], it["choices"],
                         _gold_letter(it["gold"], it["choices"]),
                         with_guidance=False),
        "symstep_g": lambda it: _run_symstep_lsat(
                         it["query"], it["choices"],
                         _gold_letter(it["gold"], it["choices"]),
                         with_guidance=True),
    }
    results = {m: {"correct": 0, "total": 0, "calls": 0,
                   "contradictions": 0, "time": 0.0} for m in methods}
    per_problem = []

    for idx, item in enumerate(items):
        correct_letter = _gold_letter(item["gold"], item["choices"])
        print(f"  [{idx+1}/{len(items)}] {item['query'][:60]}…  (ans={correct_letter})")
        row = {"id": idx, "query": item["query"][:80], "correct": correct_letter}
        for mname, mfn in methods.items():
            r = mfn(item)
            correct, calls, elapsed, contra = r[0], r[1], r[2], r[3]
            results[mname]["correct"] += int(correct)
            results[mname]["total"] += 1
            results[mname]["calls"] += calls
            results[mname]["contradictions"] += contra
            results[mname]["time"] += elapsed
            print(f"    {mname:<14} {'✓' if correct else '✗'}  calls={calls}  "
                  f"contra={contra}  t={elapsed:.1f}s")
            row[mname] = {"correct": correct, "calls": calls,
                          "contradictions": contra, "time": round(elapsed, 1)}
        per_problem.append(row)

    _print_summary(results)
    out = {"experiment": "lsat_ar", "model": _sym.MODEL,
           "n": len(items), "summary": results, "per_problem": per_problem}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {out_path}")
    return out


def _print_summary(results: dict):
    print(f"\n{'='*70}\nSUMMARY — AR-LSAT\n{'='*70}")
    print(f"  {'Method':<14} {'Acc%':>6} {'95% CI':>16} {'Avg calls':>10} {'Avg contra':>12}")
    print(f"  {'-'*60}")
    for m, r in results.items():
        if r["total"] == 0:
            continue
        k, n = r["correct"], r["total"]
        lo, hi = wilson_ci(k, n)
        print(f"  {m:<14} {k/n*100:>5.0f}%  [{100*lo:.0f},{100*hi:.0f}]"
              f"  {r['calls']/n:>9.1f}  {r['contradictions']/n:>11.2f}")

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="Number of problems (default: all 230)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load dataset and print sample, no LLM calls")
    args = parser.parse_args()

    if args.dry_run:
        items = load_lsat_ar(n=args.n or 5)
        print(f"AR-LSAT: {len(items)} problems loaded")
        item = items[0]
        print("  Query:", item["query"][:200])
        print("  Choices:", item["choices"][:2])
        print("  Gold index:", item["gold"],
              "→ Answer:", _gold_letter(item["gold"], item["choices"]))
        print("Dry run done.")
    else:
        run_lsat_exp(n=args.n)
