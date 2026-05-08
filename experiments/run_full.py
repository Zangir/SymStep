#!/usr/bin/env python3
"""
run_full.py -- Comprehensive SymStep experiment runner.

Runs all experiments needed for the paper:
  1. LGP-14 main results (haiku, N=1)
  2. Multi-run ablation on LGP-6 (haiku, N=3) for statistical validity
  3. Sonnet model comparison on LGP-10 (N=1)

Usage:
  python run_full.py                     # all experiments
  python run_full.py --exp main          # LGP-14 main only
  python run_full.py --exp ablation      # multi-run ablation only
  python run_full.py --exp sonnet        # sonnet comparison only
  python run_full.py --exp verify        # verify all puzzles only
"""

import os, sys, re, json, time, copy, argparse, statistics
from typing import Dict, List, Tuple

# ── Import from existing modules ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import symstep as _sym
from symstep import (
    ConstraintPropagator, call_llm, puzzle_text, answer_format,
    parse_solution, check_solution, SYMSTEP_SYSTEM,
    run_direct, run_cot, run_self_refine, run_symstep,
    Puzzle, PUZZLES as LGP6_PUZZLES,
)
from extended import NEW_PUZZLES

# ── Full benchmark: LGP-14 ───────────────────────────────────────────────────

LGP14_PUZZLES = LGP6_PUZZLES + NEW_PUZZLES

# Subset: original LGP-10 (LGP6 + first 4 from NEW_PUZZLES: E3, M3, M4, H3)
LGP10_PUZZLES = LGP6_PUZZLES + NEW_PUZZLES[:4]

METHODS = {
    "direct":      run_direct,
    "cot":         run_cot,
    "self_refine": run_self_refine,
    "symstep":     lambda p: run_symstep(p, with_guidance=False),
    "symstep_g":   lambda p: run_symstep(p, with_guidance=True),
}

# ── Guidance-only (no contradiction reporting) ───────────────────────────────

def run_guidance_only(p: Puzzle, max_steps: int = 25) -> Tuple[bool, int, float, int]:
    """MRV hints provided; contradictions silently ignored (not reported to LLM)."""
    t0 = time.time()
    prop = ConstraintPropagator(p)
    calls = 0
    contradictions = 0
    history = [puzzle_text(p) + "\n\n" + SYMSTEP_SYSTEM + "\n\nMake your first deduction now."]

    for _ in range(max_steps):
        response = call_llm("\n\n".join(history))
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

        person    = m.group(1).strip()
        attr      = m.group(2).strip().lower()
        is_neg    = m.group(3) is not None
        value     = m.group(4).strip()
        attr_norm = next((a for a in p.attributes if a.lower() == attr), attr)

        if is_neg:
            ok, _ = prop.apply_negative(person, attr_norm, value)
        else:
            ok, _ = prop.apply_positive(person, attr_norm, value)

        if not ok:
            contradictions += 1
            # silently drop invalid deduction; just give hint
        feedback = f"[System]: ✓ Noted. {prop.guidance()}"
        if prop.is_solved():
            break
        history.append(feedback)

    sol = prop.get_solution() or {}
    return check_solution(sol, p.solution), calls, time.time() - t0, contradictions

# ── Utilities ─────────────────────────────────────────────────────────────────

def verify_all_puzzles(puzzles):
    print("Verifying puzzle uniqueness...")
    ok_all = True
    for p in puzzles:
        prop = ConstraintPropagator(p)
        ok = True
        for person, attrs in p.solution.items():
            for attr, val in attrs.items():
                success, msg = prop.apply_positive(person, attr, val)
                if not success:
                    print(f"  FAIL {p.name}: {msg}")
                    ok = False
        if ok and prop.is_solved():
            print(f"  ✓  {p.name}")
        elif ok:
            print(f"  PARTIAL {p.name} (propagator not fully solved)")
        else:
            ok_all = False
    return ok_all


def run_experiment(puzzles, methods, model_name, tag=""):
    """Run all methods on all puzzles once; return (results_dict, per_puzzle_list)."""
    results = {
        m: {"correct": 0, "total": 0, "calls": 0, "contradictions": 0, "time": 0.0}
        for m in methods
    }
    per_puzzle = []

    for puzzle in puzzles:
        print(f"  {puzzle.name} ({puzzle.difficulty})")
        row = {"puzzle": puzzle.name, "difficulty": puzzle.difficulty}
        for mname, mfn in methods.items():
            r = mfn(puzzle)
            correct, calls, elapsed = r[0], r[1], r[2]
            contra = r[3] if len(r) > 3 else 0
            results[mname]["correct"]        += int(correct)
            results[mname]["total"]          += 1
            results[mname]["calls"]          += calls
            results[mname]["contradictions"] += contra
            results[mname]["time"]           += elapsed
            print(f"    {mname:<14} {'✓' if correct else '✗'}  "
                  f"calls={calls}  contra={contra}  t={elapsed:.1f}s")
            row[mname] = {"correct": correct, "calls": calls,
                          "contradictions": contra, "time": round(elapsed, 1)}
        per_puzzle.append(row)

    return results, per_puzzle


def print_summary(results, per_puzzle=None, diff_breakdown=False):
    print(f"\n  {'Method':<14} {'Acc%':>6} {'Avg calls':>10} {'Avg contra':>12}")
    print(f"  {'-'*50}")
    for m, r in results.items():
        if r["total"] == 0:
            continue
        acc    = r["correct"]        / r["total"] * 100
        acalls = r["calls"]          / r["total"]
        acont  = r["contradictions"] / r["total"]
        print(f"  {m:<14} {acc:>5.0f}%  {acalls:>9.1f}  {acont:>11.2f}")

    if diff_breakdown and per_puzzle:
        print()
        for diff in ["easy", "medium", "hard"]:
            subset = [r for r in per_puzzle if r["difficulty"] == diff]
            if not subset:
                continue
            print(f"  {diff.upper()} ({len(subset)} puzzles):")
            for m in results:
                n = sum(1 for r in subset if r[m]["correct"])
                print(f"    {m:<14} {n}/{len(subset)}")

# ── Experiment 1: LGP-14 main results ─────────────────────────────────────────

def exp_main(out_dir="."):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 1: LGP-14 main results  (model={_sym.MODEL})")
    print("=" * 70)

    results, per_puzzle = run_experiment(LGP14_PUZZLES, METHODS, _sym.MODEL)
    print_summary(results, per_puzzle, diff_breakdown=True)

    out = {"experiment": "lgp14_main", "model": _sym.MODEL,
           "summary": results, "per_puzzle": per_puzzle}
    path = os.path.join(out_dir, "lgp14_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {path}")
    return out

# ── Experiment 2: Multi-run ablation on LGP-6 (N=3) ─────────────────────────

def exp_ablation(n_runs=3, out_dir="."):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: Multi-run ablation  N={n_runs}  model={_sym.MODEL}")
    print("=" * 70)

    abl_methods = {
        "cot":            lambda p: run_cot(p),
        "guidance_only":  run_guidance_only,
        "symstep":        lambda p: run_symstep(p, with_guidance=False),
        "symstep_g":      lambda p: run_symstep(p, with_guidance=True),
    }

    all_runs = []  # list of (run_idx, results_dict, per_puzzle)
    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx+1}/{n_runs} ---")
        results, per_puzzle = run_experiment(LGP6_PUZZLES, abl_methods, _sym.MODEL,
                                             tag=f"run{run_idx+1}")
        all_runs.append({"run": run_idx + 1, "results": results, "per_puzzle": per_puzzle})

    # Aggregate across runs
    agg = {m: [] for m in abl_methods}
    for run in all_runs:
        for m, r in run["results"].items():
            acc = r["correct"] / r["total"] * 100
            agg[m].append(acc)

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY (multi-run statistics)")
    print("=" * 70)
    print(f"  {'Method':<16} {'Mean Acc%':>10} {'Std%':>8} {'Min%':>8} {'Max%':>8}")
    print(f"  {'-'*56}")
    stats_out = {}
    for m, accs in agg.items():
        mean_acc = statistics.mean(accs)
        std_acc  = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"  {m:<16} {mean_acc:>9.1f}  {std_acc:>7.1f}  "
              f"{min(accs):>7.0f}  {max(accs):>7.0f}")
        stats_out[m] = {"mean": round(mean_acc, 1), "std": round(std_acc, 1),
                         "min": min(accs), "max": max(accs), "runs": accs}

    out = {"experiment": "ablation_multirun", "model": _sym.MODEL,
           "n_runs": n_runs, "statistics": stats_out, "all_runs": all_runs}
    path = os.path.join(out_dir, "ablation_multirun.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {path}")
    return out

# ── Experiment 3: Sonnet model comparison on LGP-10 ──────────────────────────

def exp_sonnet(out_dir="."):
    orig_model = _sym.MODEL
    _sym.MODEL = "sonnet"
    os.environ["SYMSTEP_MODEL"] = "sonnet"

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Sonnet model comparison  (LGP-10)")
    print("=" * 70)

    results, per_puzzle = run_experiment(LGP10_PUZZLES, METHODS, "sonnet")
    print_summary(results, per_puzzle, diff_breakdown=True)

    out = {"experiment": "sonnet_lgp10", "model": "sonnet",
           "summary": results, "per_puzzle": per_puzzle}
    path = os.path.join(out_dir, "sonnet_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {path}")

    _sym.MODEL = orig_model
    os.environ["SYMSTEP_MODEL"] = orig_model
    return out

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["main", "ablation", "sonnet", "verify", "all"],
                        default="all")
    parser.add_argument("--n_runs", type=int, default=3,
                        help="Number of runs for ablation experiment")
    parser.add_argument("--out_dir", default=".",
                        help="Directory to write result JSON files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.exp in ("verify", "all"):
        print("\n" + "=" * 70)
        print("PUZZLE VERIFICATION")
        print("=" * 70)
        if not verify_all_puzzles(LGP14_PUZZLES):
            print("Puzzle verification FAILED — aborting.")
            sys.exit(1)

    if args.exp in ("main", "all"):
        exp_main(out_dir=args.out_dir)

    if args.exp in ("ablation", "all"):
        exp_ablation(n_runs=args.n_runs, out_dir=args.out_dir)

    if args.exp in ("sonnet", "all"):
        exp_sonnet(out_dir=args.out_dir)

    print("\nAll done.")
