#!/usr/bin/env python3
"""Parity gate: the unified algorithm must reproduce the frozen per-task
results before it becomes the default. Gold answers live HERE (the harness),
never in the sample bag."""
import os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from graphstep.unified import solve


def check_puzzles(puzzles, label):
    correct = wrong = other = 0
    t0 = time.time()
    for p in puzzles:
        sample = {"people": p.people, "attributes": p.attributes,
                  "clues": p.clues}                     # gold withheld
        rec = solve(sample)
        if rec["status"] == "SOLVED":
            idx = {e: i + 1 for i, e in enumerate(p.people)}
            ans = rec["answer"]

            def holder(x):                 # person as variable or constant
                return ans[x] if x in ans else idx.get(x)

            ok = all(ans.get(val) is not None
                     and ans.get(val) == holder(person)
                     for person, attrs in p.solution.items()
                     for attr, val in attrs.items())
            correct += ok
            wrong += not ok
            mark = "OK " if ok else "XX "
        else:
            other += 1
            mark = "?? "
        if mark != "OK ":
            print(f"  {mark} {p.name:<16} {rec['status']:<11} "
                  f"grounded {rec['coverage']['grounded']}"
                  f"/{rec['coverage']['statements']} "
                  f"{(rec['reasons'] or [''])[0][:60]}")
    dt = time.time() - t0
    print(f"  {label}: {correct}/{len(puzzles)} correct, {wrong} wrong, "
          f"{other} unsolved  ({dt:.1f}s)")
    return correct, len(puzzles)


if __name__ == "__main__":
    import symstep
    from extended import NEW_PUZZLES, EXTRA_PUZZLES

    print("=== LGP-20 (grid) — unified path ===")
    c1, n1 = check_puzzles(symstep.PUZZLES + NEW_PUZZLES + EXTRA_PUZZLES,
                           "LGP-20")
    print("=== SP-6 (scheduling) — unified path ===")
    c2, n2 = check_puzzles(getattr(symstep, "SCHEDULING_PUZZLES", []), "SP-6")
    print(f"\nTOTAL: {c1+c2}/{n1+n2}  (frozen target: 26/26)")
