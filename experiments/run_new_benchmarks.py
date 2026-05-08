#!/usr/bin/env python3
"""
Run two new experiments:
  1. Logic-LM baseline on LGP-6 (first 6 LGP puzzles, same as ablation set)
  2. All methods on SP-6 scheduling benchmark

Usage:
  python run_new_benchmarks.py               # both experiments
  python run_new_benchmarks.py --logic-only  # only logic_lm on LGP-6
  python run_new_benchmarks.py --sched-only  # only SP-6
"""

import json, os, sys, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

from symstep import (
    PUZZLES, SCHEDULING_PUZZLES, METHODS, METHODS_NO_LLM,
    run_logic_lm, MODEL
)

BASE = os.path.dirname(__file__)


def run_experiment(puzzles, methods, label):
    print("=" * 70)
    print(f"EXPERIMENT: {label}  model={MODEL}  puzzles={len(puzzles)}")
    print("=" * 70)

    summary = {m: {"correct": 0, "total": 0, "calls": 0,
                   "contradictions": 0, "time": 0.0}
               for m in methods}
    per_puzzle = []

    for puzzle in puzzles:
        print(f"\n  {puzzle.name} ({puzzle.difficulty})")
        row = {"puzzle": puzzle.name, "difficulty": puzzle.difficulty}
        for mname, mfn in methods.items():
            result = mfn(puzzle)
            correct, calls, elapsed = result[0], result[1], result[2]
            contra = result[3] if len(result) > 3 else 0
            summary[mname]["correct"]        += int(correct)
            summary[mname]["total"]          += 1
            summary[mname]["calls"]          += calls
            summary[mname]["contradictions"] += contra
            summary[mname]["time"]           += elapsed
            mark = "✓" if correct else "✗"
            print(f"    {mname:<14} {mark}  calls={calls}  t={elapsed:.1f}s")
            row[mname] = {"correct": correct, "calls": calls,
                          "contradictions": contra, "time": round(elapsed, 1)}
        per_puzzle.append(row)

    print("\n" + "=" * 70)
    print(f"SUMMARY — {label}")
    print("=" * 70)
    print(f"  {'Method':<14} {'Acc%':>6}  {'Avg calls':>10}  {'Avg time':>10}")
    print(f"  {'-'*50}")
    for m, r in summary.items():
        acc   = r["correct"] / r["total"] * 100
        calls = r["calls"]   / r["total"]
        t     = r["time"]    / r["total"]
        print(f"  {m:<14} {acc:>5.0f}%  {calls:>9.1f}  {t:>8.1f}s")

    # Per-difficulty breakdown
    print()
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in per_puzzle if r["difficulty"] == diff]
        if not subset:
            continue
        print(f"  {diff.upper()} ({len(subset)}):")
        for m in methods:
            n = sum(1 for r in subset if r[m]["correct"])
            print(f"    {m:<14} {n}/{len(subset)}")

    return {"model": MODEL, "summary": summary, "per_puzzle": per_puzzle}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logic-only", action="store_true")
    parser.add_argument("--sched-only", action="store_true")
    args = parser.parse_args()

    do_logic = not args.sched_only
    do_sched = not args.logic_only

    # ── Experiment A: Logic-LM on LGP-6 ──────────────────────────────────────
    if do_logic:
        lgp6 = PUZZLES[:6]  # same 6 used in ablation
        logic_lm_results = run_experiment(
            lgp6,
            {"logic_lm": run_logic_lm},
            "Logic-LM on LGP-6"
        )
        out_a = os.path.join(BASE, "logic_lm_results.json")
        with open(out_a, "w") as f:
            json.dump(logic_lm_results, f, indent=2)
        print(f"\n  Saved → {out_a}")

    # ── Experiment B: All methods on SP-6 ─────────────────────────────────────
    if do_sched:
        sched_results = run_experiment(
            SCHEDULING_PUZZLES,
            METHODS,
            "All methods on SP-6 (Scheduling)"
        )
        out_b = os.path.join(BASE, "scheduling_results.json")
        with open(out_b, "w") as f:
            json.dump(sched_results, f, indent=2)
        print(f"\n  Saved → {out_b}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
