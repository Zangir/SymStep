#!/usr/bin/env python3
"""The ONE entry point: run the unified algorithm over any dataset.

No manifests, no field mappings, no benchmark names: every row is handed to
unified.solve() as-is — the shape recognizers discover what the row contains.

The only caller-supplied hygiene is --drop: fields the CALLER declares to be
reference answers, withheld from the bag so the solver cannot see them (the
recognizer's quarantine of answer-shaped strings is the second line of
defense). Scoring against gold, when gold exists, is the harness's job, not
the solver's.

Usage:
  python3 -m graphstep.run --hf <org/dataset> [--config <name>] [--split test] [--limit N] [--drop <gold fields>]
  python3 -m graphstep.run --json samples.json
"""
from __future__ import annotations
import argparse, json, os, time
from collections import Counter, defaultdict

from .unified import solve

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", help="HuggingFace dataset name")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--json", help="path to a JSON list of samples")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--drop", nargs="*", default=[],
                    help="answer fields withheld from the sample bag")
    ap.add_argument("--question", default=None,
                    help="field holding the query, when it is a separate "
                         "declarative-shaped field")
    ap.add_argument("--gold", default=None,
                    help="field holding the reference answer (scoring only; "
                         "implies --drop of it)")
    ap.add_argument("--per", default=None, metavar="FIELD=N",
                    help="take the first N rows per value of FIELD")
    ap.add_argument("--calibrate", nargs="*", default=[],
                    metavar="MODULE:FUNC[:ARG]",
                    help="knowledge-acquisition hooks run once before "
                         "solving (e.g. template calibration from a "
                         "caller-named training source)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import importlib
    for spec in args.calibrate:
        parts = spec.split(":", 2)
        fn = getattr(importlib.import_module(parts[0]), parts[1])
        fn(parts[2]) if len(parts) > 2 else fn()

    if args.hf:
        from datasets import load_dataset
        ds = load_dataset(args.hf, args.config, split=args.split) \
            if args.config else load_dataset(args.hf, split=args.split)
        rows = [dict(r) for r in ds]
        label = args.hf.split("/")[-1]
    else:
        rows = json.load(open(args.json))
        label = os.path.basename(args.json).rsplit(".", 1)[0]
    if args.per:
        fld, n = args.per.split("=")
        buckets = defaultdict(list)
        for r in rows:
            if len(buckets[r.get(fld)]) < int(n):
                buckets[r.get(fld)].append(r)
        rows = [r for b in sorted(buckets, key=str) for r in buckets[b]]
    if args.limit:
        rows = rows[: args.limit]

    hidden = set(args.drop) | ({args.gold} if args.gold else set()) \
        | ({args.question} if args.question else set())
    results, statuses = [], Counter()
    scored = correct = 0
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        bag = {k: v for k, v in row.items() if k not in hidden}
        rec = solve(bag, question=row.get(args.question)
                    if args.question else None)
        rec["_row"] = i - 1
        if args.gold and rec["status"] == "SOLVED":
            gold = str(row.get(args.gold, "")).strip().strip("()").lower()
            got = str(rec.get("answer", "")).strip().strip("()").lower()
            rec["_correct"] = (got == gold)
            scored += 1
            correct += rec["_correct"]
        results.append(rec)
        statuses[rec["status"]] += 1
        if i % 50 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] {dict(statuses)}"
                  + (f"  correct {correct}/{scored}" if args.gold else ""),
                  flush=True)

    dt = time.time() - t0
    out = args.out or os.path.join(HERE, "results",
                                   f"results_unified_{label}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\n=== unified over {label}: {len(rows)} samples in {dt:.0f}s ===")
    for s, n in statuses.most_common():
        print(f"  {s}: {n}")
    if args.gold:
        print(f"  scored vs gold: {correct}/{scored} correct among SOLVED")
    reasons = defaultdict(int)
    for r in results:
        if r["status"] not in ("SOLVED",) and r["reasons"]:
            reasons[r["reasons"][0][:70]] += 1
    if reasons:
        print("  top refusal reasons:")
        for why, n in sorted(reasons.items(), key=lambda x: -x[1])[:8]:
            print(f"    {n:4d}  {why}")
    print(f"  results -> {out}")


if __name__ == "__main__":
    main()
