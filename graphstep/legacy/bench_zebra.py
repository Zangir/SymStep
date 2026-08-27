#!/usr/bin/env python3
"""Benchmark GraphStep on ZebraLogicBench (allenai/ZebraLogicBench, grid_mode).

The dataset ships no ground-truth solutions (the solution field is blanked),
so we report the *certified* solve rate: a puzzle counts only when every clue
compiled to logic AND the engine proves the solution unique. Wrong parses
almost always surface as UNSAT (caught by the repair loop) or AMBIGUOUS,
never as a silent wrong answer with a uniqueness certificate.

Usage:
  python3 graphstep/bench_zebra.py --sizes 2x3,2x4,3x3,3x4 --n 10 --no-llm
  python3 graphstep/bench_zebra.py --sizes 4x4,5x5,6x6 --n 10
"""
import sys, os, re, json, time, argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from graphstep.reading.compile_text import Inventory
from graphstep.legacy.pipeline import solve_text_puzzle
from graphstep.legacy import llm as llm_mod

# attribute-name detection (adapted from experiments/zebralogic_bench.py)
_ATTR_PHRASES = [
    (r"unique name", "Name"), (r"nationalit", "Nationality"),
    (r"book genre", "BookGenre"), (r"music genre", "MusicGenre"),
    (r"favorite sport|favourite sport", "Sport"), (r"car model", "CarModel"),
    (r"phone model", "PhoneModel"), (r"house style", "HouseStyle"),
    (r"hair color|hair colour", "HairColor"),
    (r"favorite color|favourite color|\bcolor\b|\bcolour\b", "Color"),
    (r"\bcigar\b|\bsmok", "Cigar"), (r"\bdrink\b|\bbeverage\b", "Drink"),
    (r"\bsmoothie\b", "Smoothie"), (r"\bfood\b|\blunch\b|\beat\b", "Food"),
    (r"\banimal\b|\bpet\b|\bkeep\b", "Animal"), (r"\bflower\b", "Flower"),
    (r"\bvacation\b|\bholiday\b", "Vacation"), (r"\bhobby\b|\bhobbies\b", "Hobby"),
    (r"\boccupation\b|\bjob\b|\bwork\b", "Occupation"),
    (r"\beducation\b|\bdegree\b", "Education"),
    (r"\bheight\b|\bshort\b|\btall\b", "Height"),
    (r"\bbirthday\b|\bborn\b", "Birthday"), (r"\bmother\b|\bmom\b", "Mother"),
    (r"\bchild\b|\bson\b|\bdaughter\b", "Children"),
    (r"\bstyle\b", "Style"), (r"\bmodel\b", "Model"),
]


def _norm(v: str) -> str:
    return "".join(w.capitalize() for w in re.split(r"[\s\-_']+", v) if w)


def parse_zlb_item(item: dict):
    """ZLB puzzle text -> (Inventory, clues) or (None, reason)."""
    text = item["puzzle"]
    n_houses, n_attrs = map(int, item["size"].split("*"))
    attrs, aliases, used_names = {}, {}, set()
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith(("-", "*")):
            continue
        vals = re.findall(r"`([^`]+)`", line)
        if len(vals) != n_houses:
            continue
        name = None
        ll = line.lower()
        for pat, cand in _ATTR_PHRASES:
            if re.search(pat, ll):
                name = cand
                break
        if name is None:
            name = f"Attr{len(attrs) + 1}"
        base, i = name, 2
        while name in used_names:
            name = f"{base}{i}"; i += 1
        used_names.add(name)
        attrs[name] = [_norm(v) for v in vals]
        for raw, nv in zip(vals, attrs[name]):      # keep raw surface forms
            aliases.setdefault(nv, []).append(raw)
    if len(attrs) != n_attrs:
        return None, "attribute blocks unparsed"
    flat = [v for vs in attrs.values() for v in vs]
    if len(flat) != len(set(flat)):
        return None, "duplicate value names across attributes"

    clues = []
    if "## Clues:" not in text:
        return None, "no clues section"
    for line in text.split("## Clues:")[1].strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            clues.append(". ".join(line.split(". ")[1:]).strip())
    if not clues:
        return None, "no clues parsed"
    entities = [f"H{i}" for i in range(1, n_houses + 1)]
    return (Inventory(entities, attrs, aliases=aliases), clues), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2x3,2x4,3x3,3x4")
    ap.add_argument("--n", type=int, default=10, help="puzzles per size class")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    want = {s.replace("x", "*") for s in args.sizes.split(",")}

    from datasets import load_dataset
    ds = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")

    per_size = defaultdict(list)
    for item in ds:
        if item["size"] in want and len(per_size[item["size"]]) < args.n:
            per_size[item["size"]].append(item)

    summary = {}
    for size in sorted(want):
        items = per_size[size]
        solved = excluded = 0
        cov_num = cov_den = llm_total = 0
        statuses = defaultdict(int)
        t0 = time.time()
        for item in items:
            parsed, why = parse_zlb_item(item)
            if parsed is None:
                excluded += 1
                statuses["EXCLUDED:" + why] += 1
                continue
            inv, clues = parsed
            res = solve_text_puzzle(clues, inv, positional=True,
                                    allow_llm=not args.no_llm)
            statuses[res.status] += 1
            cov_num += res.stats["template_parsed"]
            cov_den += res.stats["clues"]
            llm_total += res.stats.get("llm_calls", 0)
            if res.status == "SOLVED":
                solved += 1
        dt = time.time() - t0
        n_eval = len(items) - excluded
        summary[size] = {
            "n": len(items), "excluded": excluded, "certified_solved": solved,
            "template_coverage": round(cov_num / max(cov_den, 1), 3),
            "llm_calls": llm_total, "statuses": dict(statuses),
            "time_s": round(dt, 1),
        }
        rate = f"{solved}/{n_eval}" if n_eval else "-"
        print(f"  {size}: certified {rate}   coverage "
              f"{100 * cov_num / max(cov_den, 1):.0f}%   llm={llm_total}   "
              f"({dt:.1f}s)   {dict(statuses)}")

    out = args.out or os.path.join(ROOT, "graphstep", "results", "results_zebra.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    tot_s = sum(v["certified_solved"] for v in summary.values())
    tot_n = sum(v["n"] - v["excluded"] for v in summary.values())
    tot_l = sum(v["llm_calls"] for v in summary.values())
    print(f"\n  TOTAL certified: {tot_s}/{tot_n}   LLM calls: {tot_l}")
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    main()
