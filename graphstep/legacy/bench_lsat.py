#!/usr/bin/env python3
"""AR-LSAT demo: whole-problem LLM compilation + entailment queries.

The LLM is used purely as a COMPILER (setup -> IR once, each option -> IR
once); all reasoning is done by the symbolic engine as SAT/entailment tests:

  "could be true / could be the schedule"  -> option is SAT with the setup
  "must be true"                            -> setup AND NOT(option) is UNSAT
  "cannot be true"                          -> setup AND option is UNSAT

Questions outside these types are reported as UNSUPPORTED (honest coverage).

Usage:  python3 graphstep/bench_lsat.py --n 5
"""
import sys, os, re, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from graphstep.engine.core import Engine, Problem
from graphstep.engine.ir import problem_from_ir, build_constraint
from graphstep.engine import constraints as C
from graphstep.legacy import llm as llm_mod


def sat(ir, extra_specs):
    variables = ir["variables"]
    cons = [build_constraint(s, variables) for s in ir.get("constraints", [])]
    cons += [build_constraint(s, variables) for s in extra_specs]
    res = Engine(Problem(variables, cons)).solve(max_solutions=1)
    return res.status in ("SOLVED", "AMBIGUOUS")


def negate_specs(specs, variables):
    """NOT(AND(specs)) = OR(negations). Returns spec or None if unnegatable."""
    try:
        built = [build_constraint(s, variables) for s in specs]
        for b in built:
            b.negate()                       # probe negatability
    except Exception:
        return None
    if len(specs) == 1:
        return [{"type": "not", "c": specs[0]}]
    return [{"type": "or",
             "clauses": [{"type": "not", "c": s} for s in specs]}]


def question_mode(qtext: str):
    q = qtext.lower()
    if "cannot" in q or "could not" in q:
        return "CANNOT"
    if "must be true" in q or "must be" in q:
        return "MUST"
    if "could be" in q or "could each be" in q or "acceptable" in q:
        return "COULD"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("dmayhem93/agieval-lsat-ar", split="test")

    correct = attempted = unsupported = 0
    for i, item in enumerate(list(ds)[:args.n]):
        query = item["query"]
        setup, question = (query.split("Q:", 1) + [""])[:2]
        mode = question_mode(question)
        gold = item["gold"][0]
        print(f"\n[{i}] mode={mode or 'UNSUPPORTED'} gold={'ABCDE'[gold]}")
        if mode is None:
            unsupported += 1
            continue

        ir = llm_mod.llm_compile_problem(setup.strip())
        if ir is None:
            print("    setup compile failed")
            unsupported += 1
            continue
        context = (f"ENCODING: {ir.get('encoding', '(none)')}\n"
                   f"SETUP CONSTRAINTS: "
                   f"{json.dumps(ir.get('constraints', []), default=str)[:1200]}")

        verdicts = []
        for ci, choice in enumerate(item["choices"]):
            specs = llm_mod.llm_compile_choice(
                re.sub(r"^\([A-E]\)", "", choice), ir["variables"], context)
            if specs is None:
                verdicts.append(None)
                continue
            try:
                if mode == "COULD":
                    verdicts.append(sat(ir, specs))
                elif mode == "CANNOT":
                    verdicts.append(not sat(ir, specs))
                else:                                    # MUST
                    neg = negate_specs(specs, ir["variables"])
                    verdicts.append(None if neg is None
                                    else not sat(ir, neg))
            except Exception as e:
                verdicts.append(None)
        picks = [i for i, v in enumerate(verdicts) if v]
        print(f"    verdicts={verdicts}  -> pick {['ABCDE'[i] for i in picks]}")
        attempted += 1
        if len(picks) == 1 and picks[0] == gold:
            correct += 1
            print("    CORRECT")

    print(f"\nAR-LSAT demo: {correct}/{attempted} correct "
          f"({unsupported} unsupported question types) — "
          f"LLM calls: {llm_mod.LLM_CALLS['n']}")


if __name__ == "__main__":
    main()
