#!/usr/bin/env python3
"""World-model benchmarks on broad general-reasoning domains. Zero LLM.

  ProofWriter (tasksource/proofwriter): NL facts + rules, open-world
    True/False/Unknown questions at proof depths 0-5, plus paraphrased
    (NatLang) and zero-shot-domain (birds-electricity) splits.
    -> WorldModel.closure() forward-chains and answers by entailment.

  StepGame (tasksource/stepgame): multi-hop spatial composition, k=1..10
    with distractors; 8 directions + overlap.
    -> spatial lexicon edges + metric Offset propagation, answer =
       componentwise sign of the forced displacement.

Usage:
  python3 graphstep/bench_worldmodel.py --bench proofwriter --n 200
  python3 graphstep/bench_worldmodel.py --bench stepgame --n 100
"""
import sys, os, re, json, argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from graphstep.reading.worldmodel import WorldModel


# ------------------------------------------------------------- ProofWriter
def run_proofwriter(n_per_config: int):
    from datasets import load_dataset
    ds = load_dataset("tasksource/proofwriter", split="test")
    buckets = defaultdict(list)
    for x in ds:
        if len(buckets[x["config"]]) < n_per_config:
            buckets[x["config"]].append(x)

    theory_cache = {}
    results = {}
    for cfg in sorted(buckets):
        correct = wrong = abstain = 0
        for item in buckets[cfg]:
            th = item["theory"]
            if th not in theory_cache:
                wm = WorldModel.from_text(th)
                coverage_ok = not wm.unread
                wm.closure()
                theory_cache[th] = (wm, coverage_ok)
            wm, coverage_ok = theory_cache[th]
            if not coverage_ok:
                abstain += 1          # precision first: read all or abstain
                continue
            pred = wm.ask(item["question"])
            gold = str(item["answer"])
            if pred == gold:
                correct += 1
            else:
                wrong += 1
        n = len(buckets[cfg])
        results[cfg] = {"n": n, "correct": correct, "wrong": wrong,
                        "abstained": abstain}
        print(f"  {cfg:22s}: {correct:4d}/{n}  (wrong {wrong}, "
              f"theory-unread {abstain})")
    tot_c = sum(r["correct"] for r in results.values())
    tot_n = sum(r["n"] for r in results.values())
    print(f"  TOTAL: {tot_c}/{tot_n} = {100*tot_c/tot_n:.1f}%   LLM calls: 0")
    return results


# ------------------------------------------------------------- StepGame
LABEL_OF = {(-1, 0): "left", (1, 0): "right", (0, 1): "above",
            (0, -1): "below", (-1, 1): "upper-left", (1, 1): "upper-right",
            (-1, -1): "lower-left", (1, -1): "lower-right",
            (0, 0): "overlap"}


VEC_OF = {v: k for k, v in LABEL_OF.items()}


def calibrate_stepgame_templates():
    from graphstep.reading.worldmodel import calibrate_spatial_templates
    calibrate_spatial_templates("tasksource/stepgame@validation")


def run_stepgame(n_per_config: int):
    from datasets import load_dataset
    calibrate_stepgame_templates()
    ds = load_dataset("tasksource/stepgame", split="test")
    buckets = defaultdict(list)
    for x in ds:
        if len(buckets[x["config"]]) < n_per_config:
            buckets[x["config"]].append(x)

    results = {}
    for cfg in sorted(buckets, key=lambda c: int(c[2:])):
        correct = wrong = abstain = 0
        for item in buckets[cfg]:
            wm = WorldModel.from_text(item["story"], resolve_coref=False)
            qm = re.search(r"relation of the agent (\w+) to the agent (\w+)",
                           item["question"])
            if qm is None or wm.unread:
                abstain += 1
                continue
            vec = wm.vector(qm.group(1), qm.group(2), metric=True)
            pred = LABEL_OF.get(vec) if vec else None
            if pred is None:
                abstain += 1
            elif pred == item["label"]:
                correct += 1
            else:
                wrong += 1
        n = len(buckets[cfg])
        results[cfg] = {"n": n, "correct": correct, "wrong": wrong,
                        "abstained": abstain}
        print(f"  {cfg:5s}: {correct:4d}/{n}  (wrong {wrong}, "
              f"abstained {abstain})")
    tot_c = sum(r["correct"] for r in results.values())
    tot_n = sum(r["n"] for r in results.values())
    print(f"  TOTAL: {tot_c}/{tot_n} = {100*tot_c/tot_n:.1f}%   LLM calls: 0")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["proofwriter", "stepgame", "both"],
                    default="both")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    out = {}
    if args.bench in ("proofwriter", "both"):
        print("=== ProofWriter (open-world rule deduction) ===")
        out["proofwriter"] = run_proofwriter(args.n)
    if args.bench in ("stepgame", "both"):
        print("=== StepGame (multi-hop spatial composition) ===")
        out["stepgame"] = run_stepgame(args.n)
    path = os.path.join(ROOT, "graphstep", "results", "results_worldmodel.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"Saved -> {path}")
