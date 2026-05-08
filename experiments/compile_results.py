#!/usr/bin/env python3
"""Merge all result files into lgp14_combined.json with per-puzzle and summary stats."""
import json, os, sys

BASE = os.path.dirname(__file__)

def load(path):
    with open(os.path.join(BASE, path)) as f:
        return json.load(f)

lgp6    = load("results.json")           # E1,E2,M1,M2,H1,H2  (haiku)
ext     = load("extended_results.json")  # E3(old),M3,M4,H3   (haiku)
new_puz = load("new_puzzles_results.json")  # E3(fixed),E4,M5,H4,H5 (haiku)
sonnet  = load("sonnet_results.json")    # LGP-10 (sonnet)

METHODS = ["direct","cot","self_refine","symstep","symstep_g"]

# Build per-puzzle list from lgp6 + extended(skip old E3) + new_puzzles
per_puzzle = []
for row in lgp6["per_puzzle"]:
    per_puzzle.append(row)
for row in ext["per_puzzle"]:
    if row["puzzle"] == "E3-Sport":
        continue  # skip old broken E3
    per_puzzle.append(row)
for row in new_puz["per_puzzle"]:
    per_puzzle.append(row)

# Compute summary
summary = {m: {"correct":0,"total":0,"calls":0,"contradictions":0} for m in METHODS}
for row in per_puzzle:
    for m in METHODS:
        r = row[m]
        summary[m]["correct"]        += int(r["correct"])
        summary[m]["total"]          += 1
        summary[m]["calls"]          += r["calls"]
        summary[m]["contradictions"] += r.get("contradictions",0)

print("=" * 70)
print("LGP-14 COMBINED RESULTS (haiku)")
print("=" * 70)
print(f"  Total puzzles: {len(per_puzzle)}")
print()
print(f"  {'Method':<14} {'Acc':>6} {'Calls':>8} {'Contra':>8}")
print(f"  {'-'*44}")
for m in METHODS:
    r = summary[m]
    acc   = r["correct"] / r["total"] * 100
    calls = r["calls"]   / r["total"]
    cont  = r["contradictions"] / r["total"]
    print(f"  {m:<14} {acc:>5.0f}%  {calls:>6.1f}  {cont:>7.2f}")

print()
print("Per difficulty:")
for diff in ["easy","medium","hard"]:
    subset = [r for r in per_puzzle if r["difficulty"] == diff]
    print(f"  {diff.upper()} ({len(subset)}):")
    for m in METHODS:
        n = sum(1 for r in subset if r[m]["correct"])
        print(f"    {m:<14} {n}/{len(subset)}")

print()
print("=" * 70)
print("PER-PUZZLE TABLE (haiku LGP-14)")
print("=" * 70)
print(f"{'Puzzle':<14} Dir CoT  SR   SS  SS+G")
for row in per_puzzle:
    vals = [row[m]["correct"] for m in METHODS]
    marks = ["✓" if v else "✗" for v in vals]
    print(f"  {row['puzzle']:<12} " + "   ".join(marks))

print()
print("=" * 70)
print("SONNET LGP-10 RESULTS")
print("=" * 70)
son_puz = sonnet["per_puzzle"]
son_sum = {m: {"correct":0,"total":0,"calls":0} for m in METHODS}
for row in son_puz:
    for m in METHODS:
        r = row[m]
        son_sum[m]["correct"] += int(r["correct"])
        son_sum[m]["total"]   += 1
        son_sum[m]["calls"]   += r["calls"]
for m in METHODS:
    r = son_sum[m]
    acc   = r["correct"] / r["total"] * 100
    calls = r["calls"]   / r["total"]
    print(f"  {m:<14} {acc:>5.0f}%  avg_calls={calls:.1f}")

# Save merged
out = {
    "lgp14_haiku": {"summary": summary, "per_puzzle": per_puzzle},
    "lgp10_sonnet": {"summary": son_sum, "per_puzzle": son_puz},
}
with open(os.path.join(BASE, "lgp14_combined.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved → lgp14_combined.json")
