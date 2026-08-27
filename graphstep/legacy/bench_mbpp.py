#!/usr/bin/env python3
"""MBPP benchmark: NL spec -> code, the assert suite is the certificate.

GraphStep ladder, adapted to code generation:
  Tier 1   LLM writes the function from the NL spec + the asserts
           (asserts are part of the task statement in MBPP: they pin the
           function name and signature).
  Verify   run the asserts in a sandboxed subprocess (timeout-guarded).
           PASS means every assert holds — the external oracle, never
           an LLM judgment.
  Repair   on failure, the LLM is shown its own code, the EXACT failing
           assert and the error (the code-generation analogue of the
           unsat-core repair loop), and re-writes. Up to --repairs rounds.

Nothing is averaged away: every sample's full history (code, failing
assert, error, tier that solved it) is written to results_mbpp.json for
qualitative analysis.

Usage:
  python3 -m graphstep.bench_mbpp --limit 20          # pilot
  python3 -m graphstep.bench_mbpp                     # full 500 test split
  python3 -m graphstep.bench_mbpp --workers 8 --repairs 2
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from .llm import call_llm
from ..engine.sandbox import run_tests, RUN_TIMEOUT

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results_mbpp.json")
RUN_TIMEOUT = 15          # seconds per sandbox execution


# ---------------------------------------------------------------- sandbox

def run_tests(code: str, tests: List[str], setup: str = "") -> Dict:
    """Execute code + asserts in a fresh subprocess. Returns
    {"ok": bool, "failed_test": str|None, "error": str|None}.
    Asserts run one by one so the FIRST failing assert is named."""
    lines = [code, "", setup or ""]
    for i, t in enumerate(tests):
        lines += [f"# __test_{i}__", t,
                  f"print('__PASS_{i}__')"]
    script = "\n".join(lines)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True,
                              text=True, timeout=RUN_TIMEOUT)
        passed = {i for i in range(len(tests))
                  if f"__PASS_{i}__" in proc.stdout}
        if len(passed) == len(tests) and proc.returncode == 0:
            return {"ok": True, "failed_test": None, "error": None}
        first_fail = min(set(range(len(tests))) - passed, default=None)
        err = (proc.stderr.strip().splitlines() or ["nonzero exit"])[-1] \
            if proc.returncode != 0 else "assert returned but did not pass"
        return {"ok": False,
                "failed_test": tests[first_fail] if first_fail is not None else None,
                "error": err[:500]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "failed_test": None,
                "error": f"TIMEOUT after {RUN_TIMEOUT}s (infinite loop?)"}
    finally:
        os.unlink(path)


# ---------------------------------------------------------------- prompts

def _extract_code(reply: str) -> Optional[str]:
    """Pull python source out of an LLM reply (fenced block preferred)."""
    if "```" in reply:
        parts = reply.split("```")
        for part in parts[1::2]:                      # inside fences
            body = part.split("\n", 1)
            src = body[1] if body[0].strip() in ("python", "py", "") else part
            if "def " in src:
                return src.strip()
    return reply.strip() if "def " in reply else None


def gen_prompt(text: str, tests: List[str]) -> str:
    t = "\n".join(tests)
    return f"""Write a Python function for this task.

Task: {text}

Your code must pass these tests (they fix the function name and signature):
{t}

Rules: include any imports you need; define helper functions if useful;
do NOT include the tests themselves. Reply with ONLY a python code block."""


def repair_prompt(text: str, tests: List[str], code: str,
                  failed: Optional[str], error: str) -> str:
    t = "\n".join(tests)
    where = f"Failing test: {failed}\n" if failed else ""
    return f"""Your Python solution to this task is WRONG. Fix it.

Task: {text}

Required tests:
{t}

Your code:
```python
{code}
```

{where}Error: {error}

Re-read the task and the failing test carefully — the test is the ground
truth for any ambiguity in the wording. Reply with ONLY the corrected
python code block."""


# ---------------------------------------------------------------- ladder

def solve_sample(ex: Dict, repairs: int = 2) -> Dict:
    text, tests = ex["text"], list(ex["test_list"])
    setup = ex.get("test_setup_code") or ""
    rec = {"task_id": ex["task_id"], "text": text, "tests": tests,
           "status": "FAILED", "tier": None, "llm_calls": 0, "history": []}

    reply = call_llm(gen_prompt(text, tests))
    rec["llm_calls"] += 1
    code = _extract_code(reply)
    if code is None:
        rec["history"].append({"tier": "gen", "error": "no code in reply",
                               "reply": reply[:300]})
    else:
        res = run_tests(code, tests, setup)
        rec["history"].append({"tier": "gen", "code": code, **res})
        if res["ok"]:
            rec.update(status="PASS", tier="gen", code=code)
            return rec

    for r in range(1, repairs + 1):
        last = rec["history"][-1]
        code_prev = last.get("code", "")
        reply = call_llm(repair_prompt(text, tests, code_prev,
                                       last.get("failed_test"),
                                       last.get("error") or ""))
        rec["llm_calls"] += 1
        code = _extract_code(reply)
        if code is None:
            rec["history"].append({"tier": f"repair{r}",
                                   "error": "no code in reply",
                                   "reply": reply[:300]})
            continue
        res = run_tests(code, tests, setup)
        rec["history"].append({"tier": f"repair{r}", "code": code, **res})
        if res["ok"]:
            rec.update(status="PASS", tier=f"repair{r}", code=code)
            return rec

    rec["code"] = rec["history"][-1].get("code")
    return rec


# ---------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N test tasks")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--repairs", type=int, default=2)
    ap.add_argument("--out", default=RESULTS_PATH)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", split="test")
    samples = list(ds)[: args.limit] if args.limit else list(ds)
    print(f"MBPP test split: {len(samples)} tasks, "
          f"{args.repairs} repair rounds, {args.workers} workers",
          flush=True)

    results, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(solve_sample, ex, args.repairs): ex["task_id"]
                for ex in samples}
        for n, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            results.append(rec)
            mark = "ok " if rec["status"] == "PASS" else "FAIL"
            print(f"[{n}/{len(samples)}] task {rec['task_id']}: {mark} "
                  f"(tier={rec['tier']}, calls={rec['llm_calls']})",
                  flush=True)
            if n % 10 == 0 or n == len(samples):
                results.sort(key=lambda r: r["task_id"])
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=1)

    results.sort(key=lambda r: r["task_id"])
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)

    n = len(results)
    solved = [r for r in results if r["status"] == "PASS"]
    by_tier: Dict[str, int] = {}
    for r in solved:
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
    calls = sum(r["llm_calls"] for r in results)
    print(f"\n=== MBPP: {len(solved)}/{n} solved "
          f"({100*len(solved)/n:.1f}%) in {time.time()-t0:.0f}s ===")
    print(f"first-shot: {by_tier.get('gen', 0)}  " +
          "  ".join(f"{t}: {c}" for t, c in sorted(by_tier.items())
                    if t != "gen"))
    print(f"LLM calls: {calls} total, {calls/n:.2f}/task")
    print(f"failed: {[r['task_id'] for r in results if r['status'] != 'PASS']}")
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
