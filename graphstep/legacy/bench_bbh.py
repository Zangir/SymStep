#!/usr/bin/env python3
"""BIG-Bench Hard benchmarks, previously untouched task shapes:

  logical_deduction_{three,five,seven}_objects — linear-ordering MCQ.
    Deterministic ordering compiler -> enumerate solutions -> pick the
    option entailed (true in every consistent arrangement). LLM only for
    statements the compiler cannot read (counted).

  web_of_lies — truth-teller/liar chains. Pure template compile to boolean
    variables + reified Iff constraints; the engine propagates the chain.
    Zero LLM by construction.

Usage:  python3 graphstep/bench_bbh.py [--n 250] [--no-llm]
"""
import sys, os, re, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from graphstep.reading.ordering import (compile_ordering_problem,
                                compile_order_statement, extract_entities)
from graphstep.engine.ir import problem_from_ir
from graphstep.engine.core import Engine
from graphstep.legacy import llm as llm_mod


# ------------------------------------------------------------ logical deduction
def run_logical_deduction(cfg: str, n: int, allow_llm: bool):
    from datasets import load_dataset
    ds = load_dataset("lukaemon/bbh", cfg, split="test")
    correct = attempted = skipped = 0
    llm_before = llm_mod.LLM_CALLS["n"]
    for item in list(ds)[:n]:
        text = item["input"].split("Options:")[0]
        options = re.findall(r"\(([A-G])\)\s*([^\n]+)", item["input"])
        ir, rep = compile_ordering_problem(text)
        if ir is None:
            skipped += 1
            continue
        ents = rep["entities"]
        # LLM tier for unreadable statements
        ok = True
        for s in rep["uncovered"]:
            if not allow_llm:
                ok = False
                break
            got = llm_mod.llm_compile_clue(
                s, ir["variables"],
                {e: i + 1 for i, e in enumerate(ents)},
                context="\nVariables are positions 1..N on a line "
                        "(1 = leftmost/newest/cheapest end).\n")
            if got:
                ir["constraints"].extend(got)
            else:
                ok = False
        if not ok:
            skipped += 1
            continue
        res = Engine(problem_from_ir(ir)).solve(max_solutions=400)
        if not res.solutions:
            skipped += 1
            continue
        verdicts = []
        for _, opt_text in options:
            spec = compile_order_statement(opt_text, ents, len(ents))
            if spec is None:
                verdicts.append(None)
                continue
            cons = [problem_from_ir({"variables": ir["variables"],
                                     "constraints": [sp]}).constraints[0]
                    for sp in spec]
            verdicts.append(all(all(c.check(sol) for c in cons)
                                for sol in res.solutions))
        picks = [i for i, v in enumerate(verdicts) if v]
        if len(picks) != 1:
            skipped += 1
            continue
        attempted += 1
        correct += options[picks[0]][0] == item["target"].strip("()")
    calls = llm_mod.LLM_CALLS["n"] - llm_before
    return {"cfg": cfg, "n": len(list(ds)[:n]), "attempted": attempted,
            "correct": correct, "skipped": skipped, "llm_calls": calls}


# ------------------------------------------------------------ web of lies
_WOL = [
    (re.compile(r"^(\w+) tells the truth\.?$"), "truth"),
    (re.compile(r"^(\w+) lies\.?$"), "lie"),
    (re.compile(r"^(\w+) says (\w+) tells the truth\.?$"), "says_truth"),
    (re.compile(r"^(\w+) says (\w+) lies\.?$"), "says_lie"),
]


def run_web_of_lies(n: int):
    from datasets import load_dataset
    ds = load_dataset("lukaemon/bbh", "web_of_lies", split="test")
    correct = attempted = skipped = 0
    for item in list(ds)[:n]:
        text = item["input"].replace("Question:", "").strip()
        qm = re.search(r"Does (\w+) tell the truth\?", text)
        if not qm:
            skipped += 1
            continue
        target_person = qm.group(1)
        sents = [s.strip() for s in
                 re.split(r"(?<=[.?])\s+", text[:qm.start()]) if s.strip()]
        variables, constraints, ok = {}, [], True
        for s in sents:
            matched = False
            for rx, kind in _WOL:
                m = rx.match(s)
                if not m:
                    continue
                matched = True
                for g in m.groups():
                    variables.setdefault(g, [0, 1])
                if kind == "truth":
                    constraints.append({"type": "is", "var": m.group(1),
                                        "value": 1, "origin": s})
                elif kind == "lie":
                    constraints.append({"type": "is", "var": m.group(1),
                                        "value": 0, "origin": s})
                else:
                    y_true = 1 if kind == "says_truth" else 0
                    constraints.append(
                        {"type": "iff",
                         "a": {"type": "is", "var": m.group(1), "value": 1},
                         "b": {"type": "is", "var": m.group(2),
                               "value": y_true},
                         "origin": s})
                break
            if not matched:
                ok = False
        if not ok or target_person not in variables:
            skipped += 1
            continue
        res = Engine(problem_from_ir(
            {"variables": variables,
             "constraints": constraints})).solve(max_solutions=2)
        vals = {sol[target_person] for sol in res.solutions}
        if len(vals) != 1:
            skipped += 1
            continue
        attempted += 1
        answer = "Yes" if vals.pop() == 1 else "No"
        correct += answer == item["target"].strip()
    return {"cfg": "web_of_lies", "n": n, "attempted": attempted,
            "correct": correct, "skipped": skipped, "llm_calls": 0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    results = []
    for cfg in ("logical_deduction_three_objects",
                "logical_deduction_five_objects",
                "logical_deduction_seven_objects"):
        r = run_logical_deduction(cfg, args.n, allow_llm=not args.no_llm)
        results.append(r)
        print(f"  {cfg}: {r['correct']}/{r['attempted']} attempted "
              f"({r['skipped']} skipped)  llm={r['llm_calls']}")
    r = run_web_of_lies(args.n)
    results.append(r)
    print(f"  web_of_lies: {r['correct']}/{r['attempted']} attempted "
          f"({r['skipped']} skipped)  llm=0")

    out = os.path.join(ROOT, "graphstep", "results", "results_bbh.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"  Saved -> {out}")
