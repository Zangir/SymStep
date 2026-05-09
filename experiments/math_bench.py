#!/usr/bin/env python3
"""
math_bench.py -- External math and quantitative reasoning benchmarks for SymStep.

Covers two published datasets:
  1. GSM8K  (Cobbe et al., NeurIPS 2021) -- grade-school arithmetic word problems.
             1,319 test problems; final answer is an integer after '####'.
  2. AQUA-RAT (Ling et al., ACL 2017) -- quantitative reasoning including financial
             math (profit/loss, interest, discount, investment, work-rate).
             254 test problems; 5-way multiple choice (A-E).

SymStep adaptation for arithmetic domains
------------------------------------------
The constraint propagator used for grid puzzles does not apply directly here.
Instead we use an ArithmeticTracker:
  * Records each `DEDUCE: var = value` the LLM asserts.
  * Optionally verifies: `DEDUCE: var = expr = value` evaluates the expression
    using already-known variables and checks consistency.
  * Provides guidance: lists the next unknown variable the problem asks for.
  * Catches contradictions: if the LLM re-asserts a variable with a conflicting
    numeric value.

Usage:
  python math_bench.py                      # both benchmarks, 100 problems each
  python math_bench.py --bench gsm8k        # GSM8K only
  python math_bench.py --bench aquarat      # AQUA-RAT only
  python math_bench.py --n 50               # first 50 problems per benchmark
  python math_bench.py --dry-run            # print problem texts, no LLM calls

GSM8K reference:
  Cobbe et al., "Training Verifiers to Solve Math Word Problems", NeurIPS 2021.
AQUA-RAT reference:
  Ling et al., "Program Induction by Rationale Generation: Learning to Solve and
  Explain Algebraic Word Problems", ACL 2017.
"""

import os, sys, re, json, time, argparse, random
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from symstep import call_llm
import symstep as _sym
from ci_utils import wilson_ci

# ── ArithmeticTracker ─────────────────────────────────────────────────────────

class ArithmeticTracker:
    """Light verifier for numeric step-by-step reasoning."""

    EPS = 1e-3

    def __init__(self):
        self.known: Dict[str, float] = {}
        self.steps: List[str] = []          # ordered list of var names

    def apply(self, var: str, expr: Optional[str], value: float) -> Tuple[bool, str]:
        """
        Record `var = value` (optionally checking `expr`).
        expr: arithmetic expression string (may reference prior variables).
        """
        # Contradiction: re-assertion with different value
        if var in self.known:
            prev = self.known[var]
            if abs(prev - value) > self.EPS:
                return False, (
                    f"Contradiction: {var} was previously stated as {prev:.4g} "
                    f"but now stated as {value:.4g}"
                )
            return True, f"OK (already known: {var} = {prev:.4g})"

        # Optional expression verification
        if expr and expr.strip():
            try:
                local = {k.replace("-", "_"): v for k, v in self.known.items()}
                expr_safe = re.sub(r"[A-Za-z_]\w*",
                    lambda m: str(local[m.group(0)]) if m.group(0) in local else m.group(0),
                    expr)
                computed = float(eval(expr_safe, {"__builtins__": {}},
                                      {"abs": abs, "round": round, "max": max, "min": min}))
                if abs(computed - value) > self.EPS:
                    return False, (
                        f"Arithmetic error: {expr} evaluates to {computed:.4g}, "
                        f"not {value:.4g}"
                    )
            except Exception:
                pass  # can't evaluate — just accept the stated value

        self.known[var] = value
        self.steps.append(var)
        return True, f"OK: {var} = {value:.4g}"

    def guidance(self, goal_var: Optional[str] = None) -> str:
        if goal_var and goal_var not in self.known:
            return f"[Hint] Still need to determine: {goal_var}"
        return "[Hint] Continue solving step by step."

# ── System prompt for arithmetic SymStep ─────────────────────────────────────

ARITH_SYSTEM = """\
Solve this problem one step at a time.

For each intermediate quantity, output EXACTLY:
  DEDUCE: variable_name = numeric_value
OR (preferred, for verification):
  DEDUCE: variable_name = arithmetic_expression = numeric_value

Rules:
- Use short snake_case variable names (e.g., total_eggs, price_per_dozen).
- Compute ONE quantity per turn. Do not output multiple DEDUCE statements at once.
- After each DEDUCE you will receive confirmation or a contradiction report.
- When you have the final answer, output:  CONCLUDE: answer = <value>
- For multiple-choice problems, output:    CONCLUDE: answer = <letter>  (e.g., A)
- Do NOT guess. Work through every arithmetic step."""

# ── GSM8K ─────────────────────────────────────────────────────────────────────

def _extract_gsm_answer(answer_text: str) -> Optional[float]:
    """Extract numeric answer after '####' in GSM8K format."""
    m = re.search(r"####\s*([\d,]+(?:\.\d+)?)", answer_text)
    if m:
        return float(m.group(1).replace(",", ""))
    nums = re.findall(r"[\d,]+(?:\.\d+)?", answer_text)
    return float(nums[-1].replace(",", "")) if nums else None


def _run_direct_gsm(question: str, _answer: float) -> Tuple[bool, int, float]:
    t0 = time.time()
    prompt = question + "\n\nSolve step by step. State the final numeric answer on the last line."
    resp = call_llm(prompt)
    nums = re.findall(r"[\d,]+(?:\.\d+)?", resp)
    got = float(nums[-1].replace(",", "")) if nums else None
    correct = got is not None and abs(got - _answer) < 1.0
    return correct, 1, time.time() - t0


def _run_cot_gsm(question: str, _answer: float) -> Tuple[bool, int, float]:
    t0 = time.time()
    prompt = (question + "\n\nLet's think step by step. "
              "Work through every arithmetic step carefully. "
              "State the final numeric answer at the end.")
    resp = call_llm(prompt)
    nums = re.findall(r"[\d,]+(?:\.\d+)?", resp)
    got = float(nums[-1].replace(",", "")) if nums else None
    correct = got is not None and abs(got - _answer) < 1.0
    return correct, 1, time.time() - t0


def _run_symstep_gsm(question: str, _answer: float,
                      with_guidance: bool = False,
                      max_steps: int = 15) -> Tuple[bool, int, float, int]:
    t0 = time.time()
    tracker = ArithmeticTracker()
    calls = 0
    contradictions = 0
    history = [question + "\n\n" + ARITH_SYSTEM + "\n\nMake your first DEDUCE now."]

    for _ in range(max_steps):
        resp = call_llm("\n\n".join(history))
        calls += 1
        history.append(f"[You]: {resp}")

        # Check for CONCLUDE
        m_ans = re.search(r"CONCLUDE\s*:\s*answer\s*=\s*([\d,.\-]+)", resp, re.IGNORECASE)
        if m_ans:
            raw = m_ans.group(1).replace(",", "")
            try:
                got = float(raw)
                correct = abs(got - _answer) < 1.0
            except ValueError:
                correct = False
            return correct, calls, time.time() - t0, contradictions

        # Parse DEDUCE: var = expr = value  OR  var = value
        m = re.search(
            r"DEDUCE\s*:\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*=\s*([+-]?[\d,]+(?:\.\d+)?)\s*$",
            resp, re.IGNORECASE | re.MULTILINE
        )
        if m:
            var, expr, val_str = m.group(1), m.group(2).strip(), m.group(3).replace(",", "")
        else:
            m = re.search(
                r"DEDUCE\s*:\s*([A-Za-z_]\w*)\s*=\s*([+-]?[\d,]+(?:\.\d+)?)",
                resp, re.IGNORECASE
            )
            if m:
                var, expr, val_str = m.group(1), None, m.group(2).replace(",", "")
            else:
                history.append("[Verifier]: Could not parse. Use: DEDUCE: var_name = value")
                continue

        try:
            value = float(val_str)
        except ValueError:
            history.append("[Verifier]: Value must be a number.")
            continue

        ok, msg = tracker.apply(var, expr, value)
        if ok:
            feedback = f"[Verifier]: ✓ {msg}"
            if with_guidance:
                feedback += "\n" + tracker.guidance()
        else:
            contradictions += 1
            feedback = f"[Verifier]: ✗ {msg}\nPlease correct this step."

        history.append(feedback)

    # Timed out — try to read the final answer from known vars
    if tracker.known:
        last_var = list(tracker.known)[-1]
        got = tracker.known[last_var]
        correct = abs(got - _answer) < 1.0
        return correct, calls, time.time() - t0, contradictions
    return False, calls, time.time() - t0, contradictions


def load_gsm8k(n: int = 100, seed: int = 42) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    rng = random.Random(seed)
    items = list(ds)
    rng.shuffle(items)
    return items[:n]


def run_gsm8k_exp(n: int = 100, out_path: str = "gsm8k_results.json") -> dict:
    items = load_gsm8k(n=n)
    print(f"\n{'='*70}\nGSM8K  n={len(items)}  model={_sym.MODEL}\n{'='*70}")

    methods = {
        "direct":    lambda q, a: _run_direct_gsm(q, a) + (0,),
        "cot":       lambda q, a: _run_cot_gsm(q, a) + (0,),
        "symstep":   lambda q, a: _run_symstep_gsm(q, a, with_guidance=False),
        "symstep_g": lambda q, a: _run_symstep_gsm(q, a, with_guidance=True),
    }
    results = {m: {"correct": 0, "total": 0, "calls": 0,
                   "contradictions": 0, "time": 0.0} for m in methods}
    per_problem = []

    for idx, item in enumerate(items):
        question = item["question"]
        gt = _extract_gsm_answer(item["answer"])
        if gt is None:
            continue
        print(f"  [{idx+1}/{len(items)}] {question[:60]}…")
        row = {"id": idx, "question": question[:80], "answer": gt}
        for mname, mfn in methods.items():
            r = mfn(question, gt)
            correct, calls, elapsed, contra = r[0], r[1], r[2], r[3]
            results[mname]["correct"] += int(correct)
            results[mname]["total"] += 1
            results[mname]["calls"] += calls
            results[mname]["contradictions"] += contra
            results[mname]["time"] += elapsed
            print(f"    {mname:<14} {'✓' if correct else '✗'}  calls={calls}  t={elapsed:.1f}s")
            row[mname] = {"correct": correct, "calls": calls, "time": round(elapsed, 1)}
        per_problem.append(row)

    _print_summary(results, "GSM8K")
    out = {"experiment": "gsm8k", "model": _sym.MODEL,
           "n": len(items), "summary": results, "per_problem": per_problem}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {out_path}")
    return out

# ── AQUA-RAT ──────────────────────────────────────────────────────────────────

def _run_direct_aqua(question: str, options: List[str],
                      correct: str) -> Tuple[bool, int, float]:
    t0 = time.time()
    opts_str = "\n".join(options)
    prompt = (question + f"\n\nOptions:\n{opts_str}\n\n"
              "Solve step by step and choose the correct answer. "
              "State your final answer as a single letter (A, B, C, D, or E).")
    resp = call_llm(prompt)
    got = _extract_letter(resp)
    return got == correct, 1, time.time() - t0


def _run_cot_aqua(question: str, options: List[str],
                   correct: str) -> Tuple[bool, int, float]:
    t0 = time.time()
    opts_str = "\n".join(options)
    prompt = (question + f"\n\nOptions:\n{opts_str}\n\n"
              "Let's think step by step. Work through the arithmetic carefully. "
              "State your final answer as a single letter (A, B, C, D, or E).")
    resp = call_llm(prompt)
    got = _extract_letter(resp)
    return got == correct, 1, time.time() - t0


def _run_symstep_aqua(question: str, options: List[str], correct: str,
                       with_guidance: bool = False,
                       max_steps: int = 15) -> Tuple[bool, int, float, int]:
    t0 = time.time()
    tracker = ArithmeticTracker()
    calls = 0
    contradictions = 0
    opts_str = "\n".join(options)
    intro = (question + f"\n\nOptions:\n{opts_str}\n\n" + ARITH_SYSTEM
             + "\n\nMake your first DEDUCE now.")
    history = [intro]

    for _ in range(max_steps):
        resp = call_llm("\n\n".join(history))
        calls += 1
        history.append(f"[You]: {resp}")

        # CONCLUDE with letter answer
        m_letter = re.search(r"CONCLUDE\s*:\s*answer\s*=\s*([A-E])", resp, re.IGNORECASE)
        if m_letter:
            got = m_letter.group(1).upper()
            return got == correct, calls, time.time() - t0, contradictions

        # CONCLUDE with numeric (match to closest option)
        m_num = re.search(r"CONCLUDE\s*:\s*answer\s*=\s*([\d,.\-]+)", resp, re.IGNORECASE)
        if m_num:
            val_str = m_num.group(1).replace(",", "")
            try:
                got = _match_option(float(val_str), options)
                return got == correct, calls, time.time() - t0, contradictions
            except Exception:
                pass

        # Parse DEDUCE
        m = re.search(
            r"DEDUCE\s*:\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*=\s*([+-]?[\d,]+(?:\.\d+)?)\s*$",
            resp, re.IGNORECASE | re.MULTILINE
        )
        if m:
            var, expr, val_str = m.group(1), m.group(2).strip(), m.group(3).replace(",", "")
        else:
            m = re.search(
                r"DEDUCE\s*:\s*([A-Za-z_]\w*)\s*=\s*([+-]?[\d,]+(?:\.\d+)?)",
                resp, re.IGNORECASE
            )
            if m:
                var, expr, val_str = m.group(1), None, m.group(2).replace(",", "")
            else:
                history.append("[Verifier]: Could not parse. Use: DEDUCE: var_name = value")
                continue

        try:
            value = float(val_str)
        except ValueError:
            history.append("[Verifier]: Value must be a number.")
            continue

        ok, msg = tracker.apply(var, expr, value)
        if ok:
            feedback = f"[Verifier]: ✓ {msg}"
            if with_guidance:
                feedback += "\n[Hint] Continue. When ready, output: CONCLUDE: answer = <letter>"
        else:
            contradictions += 1
            feedback = f"[Verifier]: ✗ {msg}\nPlease correct this step."
        history.append(feedback)

    # Timed out — try to extract letter from last response
    got = _extract_letter(history[-2] if len(history) >= 2 else "")
    return got == correct, calls, time.time() - t0, contradictions


def _extract_letter(text: str) -> Optional[str]:
    """Extract a final answer letter (A-E) from LLM output."""
    # "The answer is B" / "answer: C" / "= D" / standalone letter at end
    for pat in [
        r"(?:answer|choice|select|option)\s*(?:is|:)?\s*([A-E])\b",
        r"\b([A-E])\s*(?:is correct|is the answer)",
        r"(?:\*\*|```)([A-E])(?:\*\*|```)",
        r"\b([A-E])\s*$",
        r"^([A-E])\b",
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    # Last standalone letter in text
    letters = re.findall(r"\b([A-E])\b", text.upper())
    return letters[-1] if letters else None


def _match_option(value: float, options: List[str]) -> Optional[str]:
    """Match a computed numeric value to the closest option (A-E)."""
    for opt in options:
        letter = opt[0].upper()
        nums = re.findall(r"[\d,]+(?:\.\d+)?", opt)
        if nums:
            opt_val = float(nums[0].replace(",", ""))
            if abs(opt_val - value) / (abs(value) + 1e-9) < 0.05:  # 5% tolerance
                return letter
    return None


def load_aquarat(n: int = 100, seed: int = 42) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("aqua_rat", "raw", split="test")
    rng = random.Random(seed)
    items = list(ds)
    rng.shuffle(items)
    return items[:n]


def run_aquarat_exp(n: int = 100, out_path: str = "aquarat_results.json") -> dict:
    items = load_aquarat(n=n)
    print(f"\n{'='*70}\nAQUA-RAT (Quantitative Reasoning)  n={len(items)}  model={_sym.MODEL}\n{'='*70}")

    methods = {
        "direct":    lambda item: _run_direct_aqua(item["question"], item["options"], item["correct"]) + (0,),
        "cot":       lambda item: _run_cot_aqua(item["question"], item["options"], item["correct"]) + (0,),
        "symstep":   lambda item: _run_symstep_aqua(item["question"], item["options"],
                                                     item["correct"], with_guidance=False),
        "symstep_g": lambda item: _run_symstep_aqua(item["question"], item["options"],
                                                     item["correct"], with_guidance=True),
    }
    results = {m: {"correct": 0, "total": 0, "calls": 0,
                   "contradictions": 0, "time": 0.0} for m in methods}
    per_problem = []

    for idx, item in enumerate(items):
        print(f"  [{idx+1}/{len(items)}] {item['question'][:60]}…")
        row = {"id": idx, "question": item["question"][:80], "correct": item["correct"]}
        for mname, mfn in methods.items():
            r = mfn(item)
            correct, calls, elapsed, contra = r[0], r[1], r[2], r[3]
            results[mname]["correct"] += int(correct)
            results[mname]["total"] += 1
            results[mname]["calls"] += calls
            results[mname]["contradictions"] += contra
            results[mname]["time"] += elapsed
            print(f"    {mname:<14} {'✓' if correct else '✗'}  calls={calls}  t={elapsed:.1f}s")
            row[mname] = {"correct": correct, "calls": calls, "time": round(elapsed, 1)}
        per_problem.append(row)

    _print_summary(results, "AQUA-RAT")
    out = {"experiment": "aquarat", "model": _sym.MODEL,
           "n": len(items), "summary": results, "per_problem": per_problem}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {out_path}")
    return out

# ── Shared summary ─────────────────────────────────────────────────────────────

def _print_summary(results: dict, label: str):
    print(f"\n{'='*70}\nSUMMARY — {label}\n{'='*70}")
    print(f"  {'Method':<14} {'Acc%':>6} {'95% CI':>16} {'Avg calls':>10}")
    print(f"  {'-'*52}")
    for m, r in results.items():
        if r["total"] == 0:
            continue
        k, n = r["correct"], r["total"]
        lo, hi = wilson_ci(k, n)
        print(f"  {m:<14} {k/n*100:>5.0f}%  [{100*lo:.0f},{100*hi:.0f}]"
              f"  {r['calls']/n:>9.1f}")

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", choices=["gsm8k", "aquarat", "both"], default="both")
    parser.add_argument("--n", type=int, default=100, help="Problems per benchmark")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load datasets and print problem texts, no LLM calls")
    args = parser.parse_args()

    if args.dry_run:
        if args.bench in ("gsm8k", "both"):
            items = load_gsm8k(n=args.n)
            print(f"GSM8K: {len(items)} problems loaded")
            print("  Sample:", items[0]["question"][:100])
            print("  Answer:", _extract_gsm_answer(items[0]["answer"]))
        if args.bench in ("aquarat", "both"):
            items = load_aquarat(n=args.n)
            print(f"AQUA-RAT: {len(items)} problems loaded")
            print("  Sample:", items[0]["question"][:100])
            print("  Correct:", items[0]["correct"])
        print("Dry run done.")
    else:
        if args.bench in ("gsm8k", "both"):
            run_gsm8k_exp(n=args.n)
        if args.bench in ("aquarat", "both"):
            run_aquarat_exp(n=args.n)
