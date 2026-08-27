#!/usr/bin/env python3
"""Generality CI: the unified path must contain NO task differentiation.

Rule 1 (name ban): benchmark identifiers may not appear anywhere in the
unified modules — a benchmark is somewhere a sample comes FROM, never
something the algorithm knows about.

Rule 2 (branch ban): no function in the unified modules may take a
parameter that names a task/dataset/benchmark — the only allowed dispatch
is on evidence shape and artifact type.

Like the garbage CI: this may never regress. New unified-path modules must
be added to UNIFIED_MODULES."""
import ast, os, re, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UNIFIED_MODULES = ["unified.py", "run.py",
                   "engine/core.py", "engine/constraints.py",
                   "engine/ir.py", "engine/sandbox.py",
                   "reading/kb.py", "reading/semgraph.py",
                   "reading/narrative.py"]

BANNED_NAMES = re.compile(
    r"(?i)\b(mbpp|zebra|zebralogic|lsat|babi|lgp\b|proofwriter|stepgame|"
    r"bbh|gsm8k|aqua|bigcodebench|swe[-_]?bench)\b")

BANNED_PARAMS = {"task", "dataset", "benchmark", "task_type", "domain"}


def main() -> int:
    failures = []
    for mod in UNIFIED_MODULES:
        path = os.path.join(HERE, mod)
        src = open(path).read()
        for i, line in enumerate(src.splitlines(), 1):
            if BANNED_NAMES.search(line):
                failures.append(f"{mod}:{i}: benchmark name in unified "
                                f"module: {line.strip()[:70]}")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args + node.args.kwonlyargs]
                bad = BANNED_PARAMS & set(args)
                if bad:
                    failures.append(
                        f"{mod}:{node.lineno}: function {node.name}() "
                        f"dispatches on {sorted(bad)}")
    if failures:
        print("GENERALITY VIOLATIONS:")
        for f in failures:
            print(" ", f)
        return 1
    print(f"generality CI: {len(UNIFIED_MODULES)} modules clean "
          f"(no benchmark names, no task dispatch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
