#!/usr/bin/env python3
"""The execution oracle: run candidate code against assertion tests in a
fresh, timeout-guarded subprocess. This is the external judge for the
program algebra — code is never trusted, only executed and observed."""
from __future__ import annotations
import os, subprocess, sys, tempfile
from typing import Dict, List

RUN_TIMEOUT = 15          # seconds per sandbox execution


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
