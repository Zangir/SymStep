#!/usr/bin/env python3
"""AR-LSAT with the deterministic scheduling-frame compiler — ZERO LLM calls.

Coverage-honest evaluation over all 230 items:
  - frame detected?  setup fully compiled?  question mode supported?
  - choices compiled (schedule-grid or sentence form)?
  - answer via SAT/entailment; self-check requires a unique pick.
Anything not covered is counted, never guessed.
"""
import sys, os, re, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from graphstep.legacy.lsat_frame import (detect_frame, compile_setup,
                                  compile_schedule_choice,
                                  compile_lsat_sentence, split_stem,
                                  respectively_slots, compile_premise,
                                  compile_name_list_choice)
from graphstep.engine.ir import problem_from_ir
from graphstep.engine.core import Engine


def q_mode(q: str):
    ql = q.lower()
    if "cannot" in ql or "could not" in ql or "not be true" in ql:
        return "CANNOT"
    if "must be" in ql:
        return "MUST"
    if "could be" in ql or "acceptable" in ql:
        return "COULD"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=230)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("dmayhem93/agieval-lsat-ar", split="test")

    stats = {"items": 0, "no_frame": 0, "setup_partial": 0,
             "mode_unsupported": 0, "choice_fail": 0, "no_unique_pick": 0,
             "attempted": 0, "correct": 0}

    for item in list(ds)[:args.n]:
        stats["items"] += 1
        query = item["query"]
        setup, question = (query.split("Q:", 1) + [""])[:2]
        fr = detect_frame(setup)
        if fr is None:
            stats["no_frame"] += 1
            continue
        specs, unhandled = compile_setup(setup, fr)
        if unhandled:
            stats["setup_partial"] += 1
            continue
        # question-stem premises ("If Kyle and Lenore do not give reports, …")
        premises, query = split_stem(question)
        prem_specs, prem_ok = [], True
        for pr in premises:
            got = compile_premise(pr, fr)
            if got is None:
                prem_ok = False
            else:
                prem_specs.extend(got)
        if not prem_ok:
            stats["mode_unsupported"] += 1
            continue
        mode = q_mode(query) or ("COULD" if "could be" in query.lower()
                                 else None)
        if mode is None:
            stats["mode_unsupported"] += 1
            continue
        specs = specs + prem_specs
        slots_template = respectively_slots(query, fr)

        def is_sat(extra):
            ir = {"variables": fr.variables(),
                  "constraints": fr.base_constraints() + specs + extra}
            r = Engine(problem_from_ir(ir)).solve(max_solutions=1)
            return r.status in ("SOLVED", "AMBIGUOUS")

        verdicts = []
        for ch in item["choices"]:
            cs = compile_schedule_choice(ch, fr)
            if cs is None and slots_template:
                cs = compile_name_list_choice(ch, slots_template, fr)
            if cs is None:
                cs = compile_lsat_sentence(re.sub(r"^\([A-E]\)", "", ch), fr)
            if cs is None:
                verdicts.append(None)
                continue
            try:
                if mode == "COULD":
                    verdicts.append(is_sat(cs))
                elif mode == "CANNOT":
                    verdicts.append(not is_sat(cs))
                else:                                          # MUST
                    neg = [{"type": "not",
                            "c": cs[0] if len(cs) == 1 else
                            {"type": "and", "clauses": cs}}]
                    verdicts.append(not is_sat(neg))
            except Exception:
                verdicts.append(None)
        if any(v is None for v in verdicts):
            stats["choice_fail"] += 1
            continue
        picks = [i for i, v in enumerate(verdicts) if v]
        if len(picks) != 1:
            stats["no_unique_pick"] += 1               # self-check failed
            continue
        stats["attempted"] += 1
        gold = item["gold"][0]
        ok = picks[0] == gold
        stats["correct"] += ok
        print(f"  [{stats['items']-1:3d}] {mode:<6} pick={'ABCDE'[picks[0]]} "
              f"gold={'ABCDE'[gold]} {'CORRECT' if ok else 'WRONG'}")

    print("\n=== AR-LSAT deterministic-frame results (0 LLM calls) ===")
    for k, v in stats.items():
        print(f"  {k:<18} {v}")
    if stats["attempted"]:
        print(f"  accuracy on attempted: "
              f"{100*stats['correct']/stats['attempted']:.0f}%")
    out = os.path.join(ROOT, "graphstep", "results", "results_lsat_frame.json")
    json.dump(stats, open(out, "w"), indent=2)
    print(f"  Saved -> {out}")


if __name__ == "__main__":
    main()
