#!/usr/bin/env python3
"""
zebralogic_bench.py -- ZebraLogicBench cross-benchmark evaluation for SymStep.

Loads the allenai/ZebraLogicBench (grid_mode) dataset, solves each puzzle with
a CSP solver to derive ground-truth solutions, converts to the Puzzle format
used by symstep.py, and runs Direct / CoT / SymStep / SymStep+G.

Usage:
  python zebralogic_bench.py                       # default 100 puzzles
  python zebralogic_bench.py --n 200               # more puzzles
  python zebralogic_bench.py --sizes 2x3,2x4,3x3  # specific sizes only
  python zebralogic_bench.py --dry-run             # parse + solve, no LLM calls

ZebraLogicBench reference:
  Lin et al., "ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning",
  arXiv:2502.01100 / ICML 2025.
"""

import os, sys, re, json, argparse, itertools
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from symstep import (
    Puzzle, run_direct, run_cot, run_symstep,
    check_solution, ConstraintPropagator,
)
from ci_utils import wilson_ci

# ── Attribute-name normalisation ─────────────────────────────────────────────

_ATTR_PHRASE_MAP = [
    (r"unique name",                     "Name"),
    (r"nationalit",                      "Nationality"),
    (r"book genre",                      "BookGenre"),
    (r"music genre",                     "MusicGenre"),
    (r"favorite sport|favourite sport",  "FavoriteSport"),
    (r"car model",                       "CarModel"),
    (r"phone model",                     "PhoneModel"),
    (r"house style",                     "HouseStyle"),
    (r"hair color|hair colour",          "HairColor"),
    (r"favorite color|favourite color",  "Color"),
    (r"\bcolor\b|\bcolour\b",            "Color"),
    (r"\bcigar\b|\bsmok",                "Cigar"),
    (r"\bdrink\b|\bbeverage\b",          "Drink"),
    (r"\bsmoothie\b",                    "Smoothie"),
    (r"\bfood\b|\blunch\b|\beat\b",      "Food"),
    (r"\banimal\b|\bpet\b|\bkeep\b",     "Animal"),
    (r"\bflower\b",                      "Flower"),
    (r"\bvacation\b|\bholiday\b",        "Vacation"),
    (r"\bhobby\b|\bhobbies\b",           "Hobby"),
    (r"\boccupation\b|\bjob\b|\bwork\b", "Occupation"),
    (r"\beducation\b|\bdegree\b",        "Education"),
    (r"\bheight\b|\bshort\b|\btall\b",   "Height"),
    (r"\bbirthday\b|\bborn\b",           "Birthday"),
    (r"\bmother\b|\bmom\b",              "Mother"),
    (r"\bchild\b|\bson\b|\bdaughter\b",  "Children"),
]

def _detect_attr_name(line: str) -> Optional[str]:
    ll = line.lower()
    for pattern, name in _ATTR_PHRASE_MAP:
        if re.search(pattern, ll):
            return name
    # Last resort: first capitalised word-sequence after "unique"
    m = re.search(r"unique\s+([a-zA-Z ]+?):", line, re.IGNORECASE)
    if m:
        return m.group(1).strip().title().replace(" ", "")
    return None

# ── Value normalisation ───────────────────────────────────────────────────────

def _norm(v: str) -> str:
    """CamelCase identifier safe for DEDUCE regex."""
    return "".join(w.capitalize() for w in re.split(r"[\s\-_']+", v) if w)

# ── Puzzle-text parser ─────────────────────────────────────────────────────────

def _parse_zbench_item(item: dict) -> Optional[Puzzle]:
    """
    Parse a ZebraLogicBench grid_mode item.
    Returns None if the puzzle cannot be fully parsed or uniquely solved.
    """
    puzzle_text = item["puzzle"]
    size        = item["size"]
    pid         = item["id"]

    n_houses, n_attrs = map(int, size.split("*"))

    # ── Extract attribute header + values ────────────────────────────────────
    raw_attrs: Dict[str, List[str]] = {}          # attr_name → [raw_val, ...]
    norm_attrs: Dict[str, List[str]] = {}          # attr_name → [normed_val, ...]
    raw_to_norm: Dict[str, str] = {}               # raw_val → normed_val
    val_to_attr: Dict[str, str] = {}               # normed_val → attr_name

    for line in puzzle_text.split("\n"):
        line = line.strip()
        if not line.startswith("- ") and not line.startswith("* "):
            continue
        vals = re.findall(r"`([^`]+)`", line)
        if not vals or len(vals) != n_houses:
            continue
        attr = _detect_attr_name(line)
        if attr is None:
            attr = f"Attr{len(raw_attrs)+1}"

        # Guarantee unique attr names
        base = attr
        idx = 2
        while attr in raw_attrs:
            attr = f"{base}{idx}"
            idx += 1

        raw_attrs[attr]  = vals
        norm_attrs[attr] = [_norm(v) for v in vals]
        for rv, nv in zip(vals, norm_attrs[attr]):
            raw_to_norm[rv.lower()] = nv
            val_to_attr[nv] = attr

    if len(norm_attrs) != n_attrs:
        return None  # could not parse all attribute blocks

    # Skip puzzles where two attributes share the same normalised value
    all_nv = [nv for nvals in norm_attrs.values() for nv in nvals]
    if len(all_nv) != len(set(all_nv)):
        return None

    # ── Extract clue text ────────────────────────────────────────────────────
    clue_raw: List[str] = []
    if "## Clues:" in puzzle_text:
        section = puzzle_text.split("## Clues:")[1].strip()
        for line in section.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                clue_raw.append(". ".join(line.split(". ")[1:]))

    if not clue_raw:
        return None

    # ── Build entity → var mapping for CSP ──────────────────────────────────
    # Each variable: entity_key → house position 1..N
    # Entity keys: normed attr values and Person names (Name attr values)
    all_normed: List[Tuple[str, str]] = []   # (attr, normed_val)
    for attr, nvals in norm_attrs.items():
        for nv in nvals:
            all_normed.append((attr, nv))

    # ── Greedy entity finder in clue text ────────────────────────────────────
    # Sorted longest-first so "Tesla Model 3" matches before "Tesla"
    all_raw_vals = sorted(
        [(rv, raw_to_norm[rv]) for rv in raw_to_norm],
        key=lambda x: -len(x[0])
    )

    def find_entities(text: str) -> List[str]:
        """Return normed entity values found in text, in left-to-right order."""
        tl = text.lower()
        hits: List[Tuple[int, str]] = []  # (start_pos, normed_val)
        matched_spans: List[Tuple[int, int]] = []
        for rv, nv in all_raw_vals:
            for m in re.finditer(re.escape(rv), tl):
                span = (m.start(), m.end())
                if not any(s[0] <= span[0] < s[1] or span[0] <= s[0] < span[1]
                           for s in matched_spans):
                    hits.append((m.start(), nv))
                    matched_spans.append(span)
        hits.sort(key=lambda x: x[0])
        return [nv for _, nv in hits]

    def ordinal_to_int(word: str) -> Optional[int]:
        mapping = {
            "first": 1, "second": 2, "third": 3, "fourth": 4,
            "fifth": 5, "sixth": 6, "1st": 1, "2nd": 2, "3rd": 3,
            "4th": 4, "5th": 5, "6th": 6,
        }
        return mapping.get(word.lower())

    # ── CSP constraint list ──────────────────────────────────────────────────
    # Represented as lambda(assignment) → bool, where assignment: normed_val → int
    constraints: List = []

    def _c(fn):
        """Wrap constraint fn: return True if any entity unassigned."""
        def check(asgn, _fn=fn):
            try:
                return _fn(asgn)
            except (KeyError, TypeError):
                return True
        constraints.append(check)

    def add_pos_direct_left(a, b):
        _c(lambda asgn, _a=a, _b=b: (
            asgn.get(_a) is None or asgn.get(_b) is None or
            asgn[_a] + 1 == asgn[_b]))

    def add_pos_direct_right(a, b):
        _c(lambda asgn, _a=a, _b=b: (
            asgn.get(_a) is None or asgn.get(_b) is None or
            asgn[_a] == asgn[_b] + 1))

    def add_pos_left_of(a, b):
        _c(lambda asgn, _a=a, _b=b: (
            asgn.get(_a) is None or asgn.get(_b) is None or
            asgn[_a] < asgn[_b]))

    def add_pos_right_of(a, b):
        _c(lambda asgn, _a=a, _b=b: (
            asgn.get(_a) is None or asgn.get(_b) is None or
            asgn[_a] > asgn[_b]))

    def add_next_to(a, b):
        _c(lambda asgn, _a=a, _b=b: (
            asgn.get(_a) is None or asgn.get(_b) is None or
            abs(asgn[_a] - asgn[_b]) == 1))

    def add_k_between(a, b, k):
        _c(lambda asgn, _a=a, _b=b, _k=k: (
            asgn.get(_a) is None or asgn.get(_b) is None or
            abs(asgn[_a] - asgn[_b]) == _k + 1))

    def add_same_house(a, b):
        _c(lambda asgn, _a=a, _b=b: (
            asgn.get(_a) is None or asgn.get(_b) is None or
            asgn[_a] == asgn[_b]))

    def add_fixed_pos(a, pos):
        _c(lambda asgn, _a=a, _p=pos: (
            asgn.get(_a) is None or asgn[_a] == _p))

    def add_not_pos(a, pos):
        _c(lambda asgn, _a=a, _p=pos: (
            asgn.get(_a) is None or asgn[_a] != _p))

    clues_parsed = 0

    for clue in clue_raw:
        cl = clue.lower().strip()

        # Ordinal "in the Nth house"
        ordinal_re = r"\bin the (first|second|third|fourth|fifth|sixth|\d+(?:st|nd|rd|th)) house\b"

        # Pattern: "X is directly left of Y"
        m = re.search(r"(.+?) is directly left of (.+)", cl)
        if m:
            ents = find_entities(clue)
            if len(ents) >= 2:
                add_pos_direct_left(ents[0], ents[1])
                clues_parsed += 1
                continue

        # Pattern: "X is directly right of Y"
        m = re.search(r"(.+?) is directly right of (.+)", cl)
        if m:
            ents = find_entities(clue)
            if len(ents) >= 2:
                add_pos_direct_right(ents[0], ents[1])
                clues_parsed += 1
                continue

        # Pattern: "X is somewhere to the left of Y"
        m = re.search(r"(.+?) is somewhere to the left of (.+)", cl)
        if m:
            ents = find_entities(clue)
            if len(ents) >= 2:
                add_pos_left_of(ents[0], ents[1])
                clues_parsed += 1
                continue

        # Pattern: "X is somewhere to the right of Y"
        m = re.search(r"(.+?) is somewhere to the right of (.+)", cl)
        if m:
            ents = find_entities(clue)
            if len(ents) >= 2:
                add_pos_right_of(ents[0], ents[1])
                clues_parsed += 1
                continue

        # Pattern: "X and Y are next to each other"
        m = re.search(r"(.+?) and (.+?) are next to each other", cl)
        if m:
            ents = find_entities(clue)
            if len(ents) >= 2:
                add_next_to(ents[0], ents[1])
                clues_parsed += 1
                continue

        # Pattern: "there is/are K house(s) between X and Y"
        m = re.search(r"there (?:is|are) (\w+) house[s]? between (.+?) and (.+)", cl)
        if m:
            k_word = m.group(1)
            k_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "1": 1, "2": 2}
            k = k_map.get(k_word, 1)
            ents = find_entities(clue)
            if len(ents) >= 2:
                add_k_between(ents[0], ents[1], k)
                clues_parsed += 1
                continue

        # Pattern: "X is in the Nth house" / "X is not in the Nth house"
        m = re.search(ordinal_re, cl)
        if m:
            ord_str = m.group(1)
            pos_num = ordinal_to_int(ord_str)
            if pos_num is None:
                try:
                    pos_num = int(re.sub(r"[^0-9]", "", ord_str))
                except ValueError:
                    pos_num = None
            if pos_num is not None:
                ents = find_entities(clue)
                if ents:
                    if "not in" in cl:
                        add_not_pos(ents[0], pos_num)
                    else:
                        add_fixed_pos(ents[0], pos_num)
                    clues_parsed += 1
                    continue

        # Pattern: "X is Y" / "Y is X" (identity — same house)
        # Try to find exactly 2 entities with no positional keywords
        ents = find_entities(clue)
        if len(ents) >= 2 and not any(kw in cl for kw in
                ("left", "right", "next to", "between", "house")):
            add_same_house(ents[0], ents[1])
            clues_parsed += 1

    # Need at least half the clues parsed to trust the solution
    if clues_parsed < max(1, len(clue_raw) // 2):
        return None

    # ── CSP solver (backtracking) ─────────────────────────────────────────────
    # Variables: one per (attr, value); domain 1..N
    # AllDifferent per attribute
    # Plus parsed constraints

    attr_vars: Dict[str, List[str]] = norm_attrs  # attr → [nv1, nv2, ...]

    def _solve(assignment: Dict[str, int], remaining: List[Tuple[str, str]]) -> Optional[Dict[str, int]]:
        if not remaining:
            # Check all constraints
            for c in constraints:
                if not c(assignment):
                    return None
            return dict(assignment)

        attr, nv = remaining[0]
        rest = remaining[1:]

        for pos in range(1, n_houses + 1):
            # AllDifferent within attribute
            if any(assignment.get(other_nv) == pos
                   for other_nv in attr_vars[attr] if other_nv != nv):
                continue

            assignment[nv] = pos

            # Early pruning: check constraints that are fully determined
            ok = True
            for c in constraints:
                involved = {v for v in assignment if assignment.get(v) is not None}
                # Only check if all entities in the constraint are assigned
                try:
                    result = c(assignment)
                    if result is False:
                        ok = False
                        break
                except Exception:
                    pass

            if ok:
                result = _solve(assignment, rest)
                if result is not None:
                    return result

            del assignment[nv]
        return None

    # Order: Name attribute first (most constrained in clues), then others
    order = []
    if "Name" in attr_vars:
        order.extend((("Name", nv) for nv in attr_vars["Name"]))
    for attr, nvals in attr_vars.items():
        if attr != "Name":
            for nv in nvals:
                order.append((attr, nv))

    sol1 = _solve({}, order)
    if sol1 is None:
        return None

    # Verify uniqueness: try to find a second solution
    def _solve_second(assignment, remaining, exclude):
        if not remaining:
            for c in constraints:
                if not c(assignment):
                    return None
            # Must differ from exclude in at least one variable
            if assignment == exclude:
                return None
            return dict(assignment)

        attr, nv = remaining[0]
        rest = remaining[1:]
        results = []
        for pos in range(1, n_houses + 1):
            if any(assignment.get(other_nv) == pos
                   for other_nv in attr_vars[attr] if other_nv != nv):
                continue
            assignment[nv] = pos
            ok = True
            for c in constraints:
                try:
                    if c(assignment) is False:
                        ok = False
                        break
                except Exception:
                    pass
            if ok:
                r = _solve_second(assignment, rest, exclude)
                if r is not None:
                    results.append(r)
                    if len(results) >= 1:
                        break
            del assignment[nv]
        return results[0] if results else None

    sol2 = _solve_second({}, order, sol1)
    # If a second solution exists, puzzle is ambiguous — skip
    if sol2 is not None:
        return None

    # ── Build position → attribute → value mapping ───────────────────────────
    pos_to_attrs: Dict[int, Dict[str, str]] = {h: {} for h in range(1, n_houses + 1)}
    for attr, nvals in attr_vars.items():
        for nv in nvals:
            pos = sol1[nv]
            pos_to_attrs[pos][attr] = nv

    # Verify complete solution
    for h in range(1, n_houses + 1):
        if len(pos_to_attrs[h]) != n_attrs:
            return None

    # ── Build Puzzle object ──────────────────────────────────────────────────
    people   = [f"House{h}" for h in range(1, n_houses + 1)]
    solution = {f"House{h}": pos_to_attrs[h] for h in range(1, n_houses + 1)}

    difficulty = "easy" if n_houses <= 2 else ("medium" if n_houses <= 3 else "hard")

    return Puzzle(
        name=pid,
        difficulty=difficulty,
        people=people,
        attributes={attr: list(nvals) for attr, nvals in attr_vars.items()},
        clues=clue_raw,
        solution=solution,
    )

# ── Dataset loading ───────────────────────────────────────────────────────────

def load_zebralogic(
    sizes: List[str] = None,
    n: int = 100,
    seed: int = 42,
) -> List[Puzzle]:
    """
    Load ZebraLogicBench puzzles, solve them with the CSP solver, and return
    up to `n` successfully parsed+solved Puzzle objects.

    sizes: e.g. ["2*3","2*4","3*3","3*4"]. None = use default small sizes.
    """
    from datasets import load_dataset
    import random

    if sizes is None:
        sizes = ["2*3", "2*4", "3*3", "3*4"]

    ds = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")

    rng = random.Random(seed)
    by_size = defaultdict(list)
    for item in ds:
        if item["size"] in sizes:
            by_size[item["size"]].append(item)

    # Shuffle each bucket
    for sz in by_size:
        rng.shuffle(by_size[sz])

    # Round-robin across sizes to keep balance
    interleaved = list(itertools.chain.from_iterable(
        itertools.zip_longest(*[by_size[sz] for sz in sorted(by_size)])
    ))
    interleaved = [x for x in interleaved if x is not None]

    puzzles = []
    parse_ok = 0
    parse_fail = 0
    for item in interleaved:
        if len(puzzles) >= n:
            break
        p = _parse_zbench_item(item)
        if p is not None:
            puzzles.append(p)
            parse_ok += 1
        else:
            parse_fail += 1

    print(f"  ZebraLogicBench: parsed {parse_ok} / (skipped {parse_fail}); "
          f"returning {len(puzzles)} puzzles")
    return puzzles

# ── Experiment runner ─────────────────────────────────────────────────────────

ZL_METHODS = {
    "direct":    run_direct,
    "cot":       run_cot,
    "symstep":   lambda p: run_symstep(p, with_guidance=False),
    "symstep_g": lambda p: run_symstep(p, with_guidance=True),
}

def run_zebralogic_exp(
    puzzles: List[Puzzle],
    out_path: str = "zebralogic_results.json",
    dry_run: bool = False,
) -> dict:
    import symstep as _sym, time

    results = {m: {"correct": 0, "total": 0, "calls": 0,
                    "contradictions": 0, "time": 0.0}
               for m in ZL_METHODS}
    per_puzzle = []

    for p in puzzles:
        row = {"puzzle": p.name, "difficulty": p.difficulty,
               "size": p.name.split("-")[2] if "-" in p.name else "?"}
        print(f"  {p.name}  ({p.difficulty}, {len(p.people)} houses ×"
              f" {len(p.attributes)} attrs)")

        if dry_run:
            # Just check the solver works; skip LLM
            prop = ConstraintPropagator(p)
            for person, attrs in p.solution.items():
                for attr, val in attrs.items():
                    ok, msg = prop.apply_positive(person, attr, val)
                    if not ok:
                        print(f"    VERIFY FAIL: {msg}")
            print(f"    solved={prop.is_solved()}")
            per_puzzle.append(row)
            continue

        for mname, mfn in ZL_METHODS.items():
            r = mfn(p)
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

    if not dry_run:
        # Summary
        print(f"\n  {'Method':<14} {'Acc%':>6} {'95% CI':>16} {'Avg calls':>10}")
        print(f"  {'-'*52}")
        for m, r in results.items():
            if r["total"] == 0:
                continue
            k, n = r["correct"], r["total"]
            lo, hi = wilson_ci(k, n)
            print(f"  {m:<14} {k/n*100:>5.0f}%  [{100*lo:.0f},{100*hi:.0f}]"
                  f"  {r['calls']/n:>9.1f}")

        out = {
            "experiment":  "zebralogic_bench",
            "model":       _sym.MODEL,
            "n_puzzles":   len(puzzles),
            "summary":     results,
            "per_puzzle":  per_puzzle,
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved → {out_path}")
        return out
    return {}

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=100, help="Max puzzles")
    parser.add_argument("--sizes",  default="2*3,2*4,3*3,3*4",
                        help="Comma-separated sizes, e.g. 2*3,2*4")
    parser.add_argument("--out",    default="zebralogic_results.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse + verify only, no LLM calls")
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    print(f"Loading ZebraLogicBench  sizes={sizes}  n={args.n}")
    puzzles = load_zebralogic(sizes=sizes, n=args.n)
    print(f"Running experiment on {len(puzzles)} puzzles …\n")
    run_zebralogic_exp(puzzles, out_path=args.out, dry_run=args.dry_run)
