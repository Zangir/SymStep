#!/usr/bin/env python3
"""Benchmark GraphStep on LGP-20 (grid puzzles) and SP-6 (scheduling).

Scores against the ground-truth solutions shipped with the SymStep repo.
Reports accuracy, template coverage, and LLM calls per puzzle.

Usage:  python3 graphstep/bench_lgp.py [--no-llm]
"""
import sys, os, json, time, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from graphstep.reading.compile_text import Inventory
from graphstep.legacy.pipeline import solve_text_puzzle, solution_to_entity_view
from graphstep.legacy import llm as llm_mod


def run(puzzles, label, allow_llm=True):
    correct = 0
    rows = []
    for p in puzzles:
        inv = Inventory(p.people, p.attributes)
        t0 = time.time()
        res = solve_text_puzzle(p.clues, inv, positional=False,
                                allow_llm=allow_llm)
        dt = time.time() - t0
        ok = False
        if res.solution:
            view = solution_to_entity_view(res.solution, inv)
            ok = all(view.get(person, {}).get(attr) == val
                     for person, attrs in p.solution.items()
                     for attr, val in attrs.items())
        correct += ok
        rows.append({"puzzle": p.name, "status": res.status, "correct": ok,
                     "template_parsed": res.stats["template_parsed"],
                     "clues": res.stats["clues"],
                     "llm_calls": res.stats.get("llm_calls", 0),
                     "nodes": res.stats.get("nodes", 0),
                     "time_s": round(dt, 3)})
        mark = "OK " if ok else ("?? " if res.status == "AMBIGUOUS" else "xx ")
        print(f"  {mark} {p.name:<14} {res.status:<10} "
              f"templates {res.stats['template_parsed']}/{res.stats['clues']}"
              f"  llm={res.stats.get('llm_calls', 0)}  t={dt:.2f}s"
              + ("" if ok else f"   {res.explanation[:90]}"))
    n = len(puzzles)
    tot_llm = sum(r["llm_calls"] for r in rows)
    print(f"\n  {label}: {correct}/{n} = {100*correct//n}%   "
          f"total LLM calls: {tot_llm} ({tot_llm/n:.2f}/puzzle)")
    return {"label": label, "correct": correct, "total": n,
            "llm_calls": tot_llm, "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true",
                    help="Tier 0 only (templates, zero LLM calls)")
    args = ap.parse_args()

    import symstep
    from extended import NEW_PUZZLES, EXTRA_PUZZLES

    lgp20 = symstep.PUZZLES + NEW_PUZZLES + EXTRA_PUZZLES
    print(f"=== LGP-20 (grid puzzles) — GraphStep, "
          f"{'templates only' if args.no_llm else 'full ladder'} ===")
    r1 = run(lgp20, "LGP-20", allow_llm=not args.no_llm)

    sched = getattr(symstep, "SCHEDULING_PUZZLES", [])
    r2 = None
    if sched:
        print(f"\n=== SP-6 (scheduling) ===")
        r2 = run(sched, "SP-6", allow_llm=not args.no_llm)

    out = {"lgp20": r1, "sp6": r2}
    path = os.path.join(ROOT, "graphstep", "results", "results_lgp.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")
