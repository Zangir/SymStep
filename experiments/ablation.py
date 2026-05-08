#!/usr/bin/env python3
"""
Ablation experiments for SymStep.
Tests: (1) guidance-only (no verification), (2) step-budget sensitivity.
Runs on the original LGP-6 puzzles.
"""
import subprocess, re, copy, time, json, os
from symstep import (PUZZLES, ConstraintPropagator, call_llm,
                     puzzle_text, answer_format, parse_solution,
                     check_solution, SYMSTEP_SYSTEM)

# ── Guidance-only: MRV hint after every turn, but NO contradiction detection ──

def run_guidance_only(p):
    """LLM gets MRV hints but violations are NOT reported — only accepted."""
    from symstep import ConstraintPropagator
    t0 = time.time()
    prop = ConstraintPropagator(p)
    calls = 0
    history = []

    intro = (
        puzzle_text(p)
        + "\n\n"
        + SYMSTEP_SYSTEM
        + "\n\nMake your first deduction now."
    )
    history.append(intro)
    contradictions = 0

    for _ in range(25):
        prompt = "\n\n".join(history)
        response = call_llm(prompt)
        calls += 1
        history.append(f"[You]: {response}")

        if re.search(r"CONCLUDE\s*:\s*done", response, re.IGNORECASE):
            break

        m = re.search(
            r"DEDUCE\s*:\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*,\s*(NOT\s+)?([A-Za-z]+)",
            response, re.IGNORECASE
        )
        if not m:
            history.append("[System]: Please use DEDUCE format.")
            continue

        person = m.group(1).strip()
        attr   = m.group(2).strip().lower()
        is_neg = m.group(3) is not None
        value  = m.group(4).strip()
        attr_norm = next((a for a in p.attributes if a.lower() == attr), attr)

        if is_neg:
            ok, msg = prop.apply_negative(person, attr_norm, value)
        else:
            ok, msg = prop.apply_positive(person, attr_norm, value)

        if not ok:
            contradictions += 1
            # Guidance-only: do NOT report contradiction, just give hint
            feedback = f"[System]: Step noted. {prop.guidance()}"
        else:
            feedback = f"[System]: ✓ Noted. {prop.guidance()}"
            if prop.is_solved():
                break

        history.append(feedback)

    sol = prop.get_solution() or {}
    correct = check_solution(sol, p.solution)
    return correct, calls, time.time() - t0, contradictions

# ── Budget sensitivity: SymStep+G with different max_steps ──────────────────

def run_symstep_g_budget(p, max_steps):
    """SymStep+G with a specific step budget."""
    from symstep import run_symstep
    # monkey-patch max_steps by calling with custom param via inline version
    t0 = time.time()
    prop = ConstraintPropagator(p)
    calls = 0
    history = []
    contradictions = 0

    intro = (
        puzzle_text(p)
        + "\n\n"
        + SYMSTEP_SYSTEM
        + "\n\nMake your first deduction now."
    )
    history.append(intro)

    for _ in range(max_steps):
        prompt = "\n\n".join(history)
        response = call_llm(prompt)
        calls += 1
        history.append(f"[You]: {response}")

        if re.search(r"CONCLUDE\s*:\s*done", response, re.IGNORECASE):
            break

        m = re.search(
            r"DEDUCE\s*:\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*,\s*(NOT\s+)?([A-Za-z]+)",
            response, re.IGNORECASE
        )
        if not m:
            history.append("[Verifier]: Use DEDUCE format.")
            continue

        person = m.group(1).strip()
        attr   = m.group(2).strip().lower()
        is_neg = m.group(3) is not None
        value  = m.group(4).strip()
        attr_norm = next((a for a in p.attributes if a.lower() == attr), attr)

        if is_neg:
            ok, msg = prop.apply_negative(person, attr_norm, value)
        else:
            ok, msg = prop.apply_positive(person, attr_norm, value)

        if ok:
            feedback = f"[Verifier]: ✓ {msg}\n{prop.guidance()}"
            if prop.is_solved():
                break
        else:
            contradictions += 1
            feedback = f"[Verifier]: ✗ CONTRADICTION: {msg}. Revise."

        history.append(feedback)

    sol = prop.get_solution() or {}
    correct = check_solution(sol, p.solution)
    return correct, calls, time.time() - t0, contradictions

# ── Runner ───────────────────────────────────────────────────────────────────

def run_ablation():
    results = {
        "guidance_only":  {"correct": 0, "total": 0, "calls": 0, "contradictions": 0, "time": 0.0},
        "symstep_g_b10":  {"correct": 0, "total": 0, "calls": 0, "contradictions": 0, "time": 0.0},
        "symstep_g_b50":  {"correct": 0, "total": 0, "calls": 0, "contradictions": 0, "time": 0.0},
    }
    per_puzzle = []

    print("=" * 65)
    print("Ablation Experiments (LGP-6)")
    print("=" * 65)

    for puzzle in PUZZLES:
        print(f"\n── {puzzle.name} ({puzzle.difficulty}) ──")
        row = {"puzzle": puzzle.name, "difficulty": puzzle.difficulty}

        c, calls, t, contra = run_guidance_only(puzzle)
        results["guidance_only"]["correct"] += int(c)
        results["guidance_only"]["total"]   += 1
        results["guidance_only"]["calls"]   += calls
        results["guidance_only"]["contradictions"] += contra
        results["guidance_only"]["time"]    += t
        print(f"  guidance_only   {'✓' if c else '✗'}  calls={calls}  contra={contra}  t={t:.1f}s")
        row["guidance_only"] = {"correct": c, "calls": calls, "contradictions": contra}

        c, calls, t, contra = run_symstep_g_budget(puzzle, max_steps=10)
        results["symstep_g_b10"]["correct"] += int(c)
        results["symstep_g_b10"]["total"]   += 1
        results["symstep_g_b10"]["calls"]   += calls
        results["symstep_g_b10"]["contradictions"] += contra
        results["symstep_g_b10"]["time"]    += t
        print(f"  symstep_g_b10   {'✓' if c else '✗'}  calls={calls}  contra={contra}  t={t:.1f}s")
        row["symstep_g_b10"] = {"correct": c, "calls": calls, "contradictions": contra}

        c, calls, t, contra = run_symstep_g_budget(puzzle, max_steps=50)
        results["symstep_g_b50"]["correct"] += int(c)
        results["symstep_g_b50"]["total"]   += 1
        results["symstep_g_b50"]["calls"]   += calls
        results["symstep_g_b50"]["contradictions"] += contra
        results["symstep_g_b50"]["time"]    += t
        print(f"  symstep_g_b50   {'✓' if c else '✗'}  calls={calls}  contra={contra}  t={t:.1f}s")
        row["symstep_g_b50"] = {"correct": c, "calls": calls, "contradictions": contra}

        per_puzzle.append(row)

    print("\n" + "=" * 65)
    print(f"{'Method':<18} {'Acc':>6} {'Avg calls':>10} {'Avg contra':>12}")
    print("-" * 65)
    for m, r in results.items():
        acc    = r["correct"] / r["total"] * 100
        acalls = r["calls"]   / r["total"]
        acont  = r["contradictions"] / r["total"]
        print(f"{m:<18} {acc:>5.0f}%  {acalls:>9.1f}  {acont:>11.1f}")
    print("=" * 65)

    out = {"summary": results, "per_puzzle": per_puzzle}
    with open("ablation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to ablation_results.json")

if __name__ == "__main__":
    run_ablation()
