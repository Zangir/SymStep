#!/usr/bin/env python3
"""
Wilson score confidence intervals and statistical utilities for SymStep results.

Usage:
    from ci_utils import wilson_ci, format_ci, ci_table
"""

import math
from typing import Tuple, Dict


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Return (lower, upper) 95% Wilson score CI for k successes in n trials."""
    if n == 0:
        return 0.0, 0.0
    p_hat = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return lower, upper


def format_ci(k: int, n: int, as_pct: bool = True) -> str:
    """Return formatted 'acc [lo, hi]' string."""
    lo, hi = wilson_ci(k, n)
    if as_pct:
        acc = 100 * k / n if n > 0 else 0
        return f"{acc:.0f}\\% [{100*lo:.0f}, {100*hi:.0f}]"
    return f"{k/n:.3f} [{lo:.3f}, {hi:.3f}]"


def latex_ci(k: int, n: int) -> str:
    """Return LaTeX string for compact CI display in table cells."""
    lo, hi = wilson_ci(k, n)
    acc = 100 * k / n if n > 0 else 0
    return f"${acc:.0f}$ {{\\scriptsize [{100*lo:.0f},{100*hi:.0f}]}}"


def ci_table(results: Dict[str, Tuple[int, int]]) -> None:
    """Print a formatted CI table. results = {method: (k, n)}."""
    print(f"\n{'Method':<16} {'Acc':>6} {'95% Wilson CI':>20}")
    print("-" * 46)
    for method, (k, n) in results.items():
        lo, hi = wilson_ci(k, n)
        acc = 100 * k / n if n > 0 else 0
        print(f"{method:<16} {acc:>5.1f}%  [{100*lo:4.1f}%, {100*hi:4.1f}%]")


# ── Pre-computed CIs for the paper ───────────────────────────────────────────

LGP14_RESULTS = {
    "Direct":      (2,  14),
    "CoT":         (0,  14),
    "Self-Refine": (5,  14),
    "Logic-LM":    (0,  14),
    "SymStep":     (12, 14),
    "SymStep+G":   (14, 14),
}

SP6_RESULTS = {
    "Direct":      (0, 6),
    "CoT":         (0, 6),
    "Self-Refine": (2, 6),
    "Logic-LM":    (0, 6),
    "SymStep":     (5, 6),
    "SymStep+G":   (4, 6),
}

MWP8_RESULTS = {
    "Direct":      (7, 8),
    "CoT":         (8, 8),
    "SymStep":     (8, 8),
    "SymStep+G":   (8, 8),
}

FIN6_RESULTS = {
    "Direct":      (4, 6),
    "CoT":         (5, 6),
    "SymStep":     (6, 6),
    "SymStep+G":   (6, 6),
}

LGP10_SONNET_RESULTS = {
    "SymStep":   (9, 10),
    "SymStep+G": (9, 10),
}

if __name__ == "__main__":
    print("=" * 50)
    print("LGP-14 (n=14)")
    ci_table(LGP14_RESULTS)

    print("\n" + "=" * 50)
    print("SP-6 (n=6)")
    ci_table(SP6_RESULTS)

    print("\n" + "=" * 50)
    print("MWP-8 (n=8)")
    ci_table(MWP8_RESULTS)

    print("\n" + "=" * 50)
    print("FIN-6 (n=6)")
    ci_table(FIN6_RESULTS)

    print("\n" + "=" * 50)
    print("LGP-10 Sonnet (n=10)")
    ci_table(LGP10_SONNET_RESULTS)

    print("\n" + "=" * 50)
    print("KEY FINDING: non-overlapping CIs on LGP-14")
    lgp14 = LGP14_RESULTS
    for m, (k, n) in lgp14.items():
        lo, hi = wilson_ci(k, n)
        print(f"  {m:<16} {100*k/n:5.1f}%  CI: [{100*lo:.1f}%, {100*hi:.1f}%]")
    print("\n  SymStep+G lower bound (79%) > CoT upper bound (22%) → p<0.05")
