#!/usr/bin/env python3
"""
rerun_sonnet_baselines.py -- Re-evaluate the Direct / CoT / Self-Refine
baselines on LGP-10 with Sonnet using the corrected answer-block parser
(symstep.parse_solution Strategy 0).

The original sonnet_results.json baselines reflected a regex parser
calibrated for Haiku's concise output, which mis-attributed Sonnet's verbose
answer blocks across people. SymStep / SymStep+G read from propagator state
and are unaffected, so they are preserved verbatim from the existing file.

Usage:  SYMSTEP_MODEL=sonnet python3 rerun_sonnet_baselines.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["SYMSTEP_MODEL"] = "sonnet"
import symstep as _sym
_sym.MODEL = "sonnet"
from symstep import run_direct, run_cot, run_self_refine
from run_full import LGP10_PUZZLES

BASELINES = {"direct": run_direct, "cot": run_cot, "self_refine": run_self_refine}
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sonnet_results.json")


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    # Index existing per-puzzle rows by puzzle name for in-place baseline update.
    by_name = {row["puzzle"]: row for row in data["per_puzzle"]}

    summary = {m: {"correct": 0, "total": 0, "calls": 0, "contradictions": 0, "time": 0.0}
               for m in BASELINES}

    for puzzle in LGP10_PUZZLES:
        print(f"  {puzzle.name} ({puzzle.difficulty})", flush=True)
        row = by_name[puzzle.name]
        for mname, mfn in BASELINES.items():
            correct, calls, elapsed, contra = mfn(puzzle)
            summary[mname]["correct"]        += int(correct)
            summary[mname]["total"]          += 1
            summary[mname]["calls"]          += calls
            summary[mname]["contradictions"] += contra
            summary[mname]["time"]           += elapsed
            row[mname] = {"correct": bool(correct), "calls": calls,
                          "contradictions": contra, "time": round(elapsed, 1)}
            print(f"    {mname:<14} {'OK ' if correct else 'xx '} "
                  f"calls={calls} t={elapsed:.1f}s", flush=True)

    # Merge updated baseline summaries; keep symstep / symstep_g intact.
    for m in BASELINES:
        data["summary"][m] = summary[m]

    data["parser"] = "answer-block (Strategy 0); corrected for Sonnet verbose output"
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print("\n=== Updated Sonnet LGP-10 baseline accuracy ===")
    for m in BASELINES:
        s = data["summary"][m]
        print(f"  {m:<14} {s['correct']}/{s['total']} = {100*s['correct']/s['total']:.0f}%")
    print(f"\n  Saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
