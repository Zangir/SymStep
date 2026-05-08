#!/usr/bin/env python3
"""
Math Word Problem benchmark (MWP-8) for SymStep evaluation.

Tasks require multi-step numeric reasoning: deriving variable values
step-by-step from arithmetic relationships expressed in natural language.

The symbolic component is a NumericTracker that:
 - records established variable values
 - verifies each new deduction is arithmetically consistent
 - detects contradictions (previously established value conflicts)
 - cascades implied values automatically after each assertion
 - provides guidance toward the next seed variable to assert

Design: each problem has "seed" variables (read directly from problem text)
and "derived" variables (computed by cascade from seeds). The LLM's job is
to assert seeds; the propagator cascades derivations automatically.

DEDUCE format:  DEDUCE: VarName = numeric_value
"""

import re, time, json, subprocess, os, sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from symstep import call_llm, MODEL

# Claude binary is resolved by symstep.py's _find_claude_bin(); CLAUDE_BIN kept for
# compatibility with any direct call_llm override in this module.
CLAUDE_BIN = None  # unused: call_llm imported from symstep uses its own CLAUDE_BIN


# ── MathProblem dataclass ─────────────────────────────────────────────────────

@dataclass
class MathProblem:
    name:        str
    difficulty:  str                      # easy | medium | hard
    text:        str                      # full natural-language problem (includes var names)
    variables:   List[str]                # ordered list: seeds first, then derived
    constraints: Dict[str, Callable]      # varname → fn(known_dict) → value (derived only)
    solution:    Dict[str, float]         # ground truth


# ── Numeric propagation ────────────────────────────────────────────────────────

class NumericTracker:
    """Tracks known numeric variable values; verifies and cascades."""

    EPS = 1e-6

    def __init__(self, problem: MathProblem):
        self.problem  = problem
        self.known: Dict[str, float] = {}

    def apply(self, var: str, value: float) -> Tuple[bool, str]:
        if var not in self.problem.variables:
            return False, f"Unknown variable '{var}'. Valid names: {', '.join(self.problem.variables)}"
        if var in self.known:
            if abs(self.known[var] - value) > self.EPS:
                return False, (
                    f"Contradiction: {var} was already established as "
                    f"{self.known[var]:.4g}, but you deduced {value:.4g}"
                )
            return True, "OK (already known)"
        self.known[var] = value
        self._cascade()
        return True, "OK"

    def _cascade(self):
        """Derive any variable whose constraint can now be evaluated."""
        changed = True
        while changed:
            changed = False
            for var, fn in self.problem.constraints.items():
                if var in self.known:
                    continue
                try:
                    val = fn(self.known)
                    if val is not None:
                        self.known[var] = float(val)
                        changed = True
                except (KeyError, ZeroDivisionError, TypeError):
                    pass

    def is_solved(self) -> bool:
        return all(v in self.known for v in self.problem.variables)

    def seeds_remaining(self) -> List[str]:
        """Variables not in constraints (i.e., must come from LLM) that are not yet known."""
        return [v for v in self.problem.variables
                if v not in self.problem.constraints and v not in self.known]

    def guidance(self) -> str:
        seeds = self.seeds_remaining()
        if seeds:
            return f"[Hint] Next: assert seed variable '{seeds[0]}' from the problem text."
        unknown = [v for v in self.problem.variables if v not in self.known]
        if not unknown:
            return "[All variables determined]"
        return f"[Hint] Still unknown after cascade: {', '.join(unknown)}"


# ── MWP-8 benchmark problems ──────────────────────────────────────────────────
#
# Design convention:
#   - Seed variables appear explicitly in the problem text (the LLM reads them off).
#   - Derived variables are computed by cascade; problem text explains the formula.
#   - Variable names are stated in a "Variables to compute" section at the end.
#
MWP_PROBLEMS: List[MathProblem] = [

    # ── EASY ─────────────────────────────────────────────────────────────────
    MathProblem(
        name="MWP-E1-Salaries", difficulty="easy",
        text=(
            "Alex, Beth, and Carol work at a tech company.\n"
            "Alex earns $2000 per month.\n"
            "Beth earns $500 more than Alex per month.\n"
            "Carol earns twice as much as Alex per month.\n"
            "What is Carol's monthly salary in dollars?\n\n"
            "Variables to compute (use these exact names in DEDUCE statements):\n"
            "  Alex   — Alex's monthly salary in dollars\n"
            "  Beth   — Beth's monthly salary in dollars\n"
            "  Carol  — Carol's monthly salary in dollars"
        ),
        variables=["Alex", "Beth", "Carol"],
        constraints={
            "Beth":  lambda k: k["Alex"] + 500,
            "Carol": lambda k: k["Alex"] * 2,
        },
        solution={"Alex": 2000.0, "Beth": 2500.0, "Carol": 4000.0},
    ),

    MathProblem(
        name="MWP-E2-Ages", difficulty="easy",
        text=(
            "Three friends compare their ages.\n"
            "Sara is 8 years old.\n"
            "Mike is 3 times Sara's age.\n"
            "Tom is 5 years older than Mike.\n"
            "How old is Tom?\n\n"
            "Variables to compute (use these exact names in DEDUCE statements):\n"
            "  Sara — Sara's age in years\n"
            "  Mike — Mike's age in years\n"
            "  Tom  — Tom's age in years"
        ),
        variables=["Sara", "Mike", "Tom"],
        constraints={
            "Mike": lambda k: k["Sara"] * 3,
            "Tom":  lambda k: k["Mike"] + 5,
        },
        solution={"Sara": 8.0, "Mike": 24.0, "Tom": 29.0},
    ),

    MathProblem(
        name="MWP-E3-Distances", difficulty="easy",
        text=(
            "Three cities lie along a highway.\n"
            "The distance from City-A to City-B is 120 km.\n"
            "City-B to City-C is half the distance from A to B.\n"
            "What is the total distance from A to C passing through B?\n\n"
            "Variables to compute (use these exact names in DEDUCE statements):\n"
            "  dist_AB — distance from City-A to City-B in km\n"
            "  dist_BC — distance from City-B to City-C in km\n"
            "  dist_AC — total distance from City-A to City-C in km"
        ),
        variables=["dist_AB", "dist_BC", "dist_AC"],
        constraints={
            "dist_BC": lambda k: k["dist_AB"] / 2,
            "dist_AC": lambda k: k["dist_AB"] + k["dist_BC"],
        },
        solution={"dist_AB": 120.0, "dist_BC": 60.0, "dist_AC": 180.0},
    ),

    # ── MEDIUM ────────────────────────────────────────────────────────────────
    MathProblem(
        name="MWP-M1-Store", difficulty="medium",
        text=(
            "A shop sells notebooks and pens.\n"
            "Each notebook costs notebook_price = $3.00.\n"
            "Each pen costs pen_price = $1.50.\n"
            "Jake buys num_notebooks = 4 notebooks and pays total_payment = $18.00 for everything.\n"
            "How many pens (num_pens) did Jake buy?\n\n"
            "Variables to compute (use these exact names in DEDUCE statements):\n"
            "  notebook_price  — price per notebook in dollars\n"
            "  pen_price       — price per pen in dollars\n"
            "  num_notebooks   — number of notebooks bought\n"
            "  total_payment   — total amount paid in dollars\n"
            "  notebooks_cost  — total cost of all notebooks (notebook_price × num_notebooks)\n"
            "  pens_cost       — total cost of all pens (total_payment − notebooks_cost)\n"
            "  num_pens        — number of pens bought (pens_cost ÷ pen_price)"
        ),
        variables=["notebook_price", "pen_price", "num_notebooks", "total_payment",
                   "notebooks_cost", "pens_cost", "num_pens"],
        constraints={
            "notebooks_cost": lambda k: k["notebook_price"] * k["num_notebooks"],
            "pens_cost":      lambda k: k["total_payment"] - k["notebooks_cost"],
            "num_pens":       lambda k: k["pens_cost"] / k["pen_price"],
        },
        solution={
            "notebook_price": 3.0, "pen_price": 1.5,
            "num_notebooks": 4.0,  "total_payment": 18.0,
            "notebooks_cost": 12.0, "pens_cost": 6.0, "num_pens": 4.0,
        },
    ),

    MathProblem(
        name="MWP-M2-Speed", difficulty="medium",
        text=(
            "A train travels from Station-A to Station-B.\n"
            "Total distance: total_dist = 360 km.\n"
            "First-half speed: speed1 = 90 km/h.\n"
            "Second-half speed: speed2 = 60 km/h.\n"
            "What is total_time, the total travel time in hours?\n\n"
            "Variables to compute (use these exact names in DEDUCE statements):\n"
            "  total_dist — total distance in km\n"
            "  speed1     — speed for first half in km/h\n"
            "  speed2     — speed for second half in km/h\n"
            "  half_dist  — distance of each half in km (total_dist ÷ 2)\n"
            "  time1      — time for first half in hours (half_dist ÷ speed1)\n"
            "  time2      — time for second half in hours (half_dist ÷ speed2)\n"
            "  total_time — total travel time in hours (time1 + time2)"
        ),
        variables=["total_dist", "speed1", "speed2",
                   "half_dist", "time1", "time2", "total_time"],
        constraints={
            "half_dist":  lambda k: k["total_dist"] / 2,
            "time1":      lambda k: k["half_dist"] / k["speed1"],
            "time2":      lambda k: k["half_dist"] / k["speed2"],
            "total_time": lambda k: k["time1"] + k["time2"],
        },
        solution={
            "total_dist": 360.0, "speed1": 90.0, "speed2": 60.0,
            "half_dist": 180.0,  "time1": 2.0,   "time2": 3.0, "total_time": 5.0,
        },
    ),

    MathProblem(
        name="MWP-M3-Investment", difficulty="medium",
        text=(
            "Lisa invests money in two accounts.\n"
            "Account X: principal_X = $1000 at annual_rate_X = 0.05 (5% per year).\n"
            "Account Y: principal_Y = $500 at annual_rate_Y = 0.08 (8% per year).\n"
            "What is total_interest, the total interest earned after 1 year?\n\n"
            "Variables to compute (use these exact names in DEDUCE statements):\n"
            "  principal_X   — amount invested in Account X in dollars\n"
            "  annual_rate_X — annual interest rate for Account X (as a decimal)\n"
            "  principal_Y   — amount invested in Account Y in dollars\n"
            "  annual_rate_Y — annual interest rate for Account Y (as a decimal)\n"
            "  interest_X    — interest earned from Account X (principal_X × annual_rate_X)\n"
            "  interest_Y    — interest earned from Account Y (principal_Y × annual_rate_Y)\n"
            "  total_interest — total interest from both accounts (interest_X + interest_Y)"
        ),
        variables=["principal_X", "annual_rate_X", "principal_Y", "annual_rate_Y",
                   "interest_X", "interest_Y", "total_interest"],
        constraints={
            "interest_X":     lambda k: k["principal_X"] * k["annual_rate_X"],
            "interest_Y":     lambda k: k["principal_Y"] * k["annual_rate_Y"],
            "total_interest": lambda k: k["interest_X"] + k["interest_Y"],
        },
        solution={
            "principal_X": 1000.0, "annual_rate_X": 0.05,
            "principal_Y": 500.0,  "annual_rate_Y": 0.08,
            "interest_X": 50.0,    "interest_Y": 40.0, "total_interest": 90.0,
        },
    ),

    # ── HARD ──────────────────────────────────────────────────────────────────
    MathProblem(
        name="MWP-H1-Workers", difficulty="hard",
        text=(
            "Three workers are paid at different rates.\n"
            "Alice works 40 hours; Alice_rate = $15.00 per hour.\n"
            "Bob works 35 hours; Bob_rate = Alice_rate × 1.20 (20% more than Alice).\n"
            "Carol works 30 hours; Carol_rate = Bob_rate × 0.90 (10% less than Bob).\n"
            "What is total_payroll, the total cost for all three workers?\n\n"
            "Variables to compute (use these exact names in DEDUCE statements):\n"
            "  Alice_rate    — Alice's hourly rate in $/hour\n"
            "  Alice_pay     — Alice's total pay (Alice_rate × 40)\n"
            "  Bob_rate      — Bob's hourly rate (Alice_rate × 1.20)\n"
            "  Bob_pay       — Bob's total pay (Bob_rate × 35)\n"
            "  Carol_rate    — Carol's hourly rate (Bob_rate × 0.90)\n"
            "  Carol_pay     — Carol's total pay (Carol_rate × 30)\n"
            "  total_payroll — sum of all three workers' pay"
        ),
        variables=["Alice_rate", "Alice_pay",
                   "Bob_rate",   "Bob_pay",
                   "Carol_rate", "Carol_pay",
                   "total_payroll"],
        constraints={
            "Alice_pay":     lambda k: k["Alice_rate"] * 40,
            "Bob_rate":      lambda k: k["Alice_rate"] * 1.20,
            "Bob_pay":       lambda k: k["Bob_rate"] * 35,
            "Carol_rate":    lambda k: k["Bob_rate"] * 0.90,
            "Carol_pay":     lambda k: k["Carol_rate"] * 30,
            "total_payroll": lambda k: k["Alice_pay"] + k["Bob_pay"] + k["Carol_pay"],
        },
        solution={
            "Alice_rate": 15.0,   "Alice_pay": 600.0,
            "Bob_rate":   18.0,   "Bob_pay":   630.0,
            "Carol_rate": 16.2,   "Carol_pay": 486.0,
            "total_payroll": 1716.0,
        },
    ),

    MathProblem(
        name="MWP-H2-Mixture", difficulty="hard",
        text=(
            "A chemist mixes two solutions.\n"
            "Solution A: volume_A = 200 mL at concentration_A = 0.30 (30% acid).\n"
            "Solution B: volume_B = 100 mL at concentration_B = 0.60 (60% acid).\n"
            "What is acid_pct, the acid percentage in the resulting mixture?\n\n"
            "Variables to compute (use these exact names in DEDUCE statements):\n"
            "  volume_A       — volume of Solution A in mL\n"
            "  concentration_A — acid fraction in Solution A (as a decimal)\n"
            "  volume_B       — volume of Solution B in mL\n"
            "  concentration_B — acid fraction in Solution B (as a decimal)\n"
            "  acid_A         — mL of acid from Solution A (volume_A × concentration_A)\n"
            "  acid_B         — mL of acid from Solution B (volume_B × concentration_B)\n"
            "  total_acid     — total mL of acid in mixture (acid_A + acid_B)\n"
            "  mixture_vol    — total volume of mixture (volume_A + volume_B)\n"
            "  acid_pct       — acid percentage in mixture (total_acid ÷ mixture_vol × 100)"
        ),
        variables=["volume_A", "concentration_A", "volume_B", "concentration_B",
                   "acid_A", "acid_B", "total_acid", "mixture_vol", "acid_pct"],
        constraints={
            "acid_A":      lambda k: k["volume_A"] * k["concentration_A"],
            "acid_B":      lambda k: k["volume_B"] * k["concentration_B"],
            "total_acid":  lambda k: k["acid_A"] + k["acid_B"],
            "mixture_vol": lambda k: k["volume_A"] + k["volume_B"],
            "acid_pct":    lambda k: (k["total_acid"] / k["mixture_vol"]) * 100,
        },
        solution={
            "volume_A": 200.0, "concentration_A": 0.30,
            "volume_B": 100.0, "concentration_B": 0.60,
            "acid_A": 60.0,    "acid_B": 60.0,
            "total_acid": 120.0, "mixture_vol": 300.0, "acid_pct": 40.0,
        },
    ),
]


# ── SymStep runners for math ──────────────────────────────────────────────────

MATH_SYMSTEP_SYSTEM = """\
You are solving a math word problem using a step-by-step symbolic verifier.

Each step must assert ONE variable using this EXACT format:
  DEDUCE: VariableName = numeric_value

Rules:
- Use ONLY the variable names listed in the "Variables to compute" section.
- Start by asserting seed variables (values stated directly in the problem text).
- After each DEDUCE the verifier will automatically cascade and compute derived variables.
- When the verifier reports all variables are determined, output: CONCLUDE: done
- Do NOT output multiple DEDUCE statements at once. One per turn.
- Do NOT compute derived variables yourself — the verifier handles cascading."""


def run_math_direct(prob: MathProblem) -> Tuple[bool, int, float, int]:
    t0 = time.time()
    prompt = (
        prob.text + "\n\nSolve step by step. "
        "At the end, provide the final answer as a number."
    )
    resp = call_llm(prompt)
    last_var = prob.variables[-1]
    expected = prob.solution[last_var]
    nums = re.findall(r'[-+]?\d+(?:\.\d+)?', resp)
    correct = any(abs(float(n) - expected) < 0.1 for n in nums[-5:]) if nums else False
    return correct, 1, time.time() - t0, 0


def run_math_cot(prob: MathProblem) -> Tuple[bool, int, float, int]:
    t0 = time.time()
    prompt = (
        prob.text + "\n\nThink step by step through the problem. "
        "Show all intermediate computations. "
        "State the final answer as a number at the end."
    )
    resp = call_llm(prompt)
    last_var = prob.variables[-1]
    expected = prob.solution[last_var]
    nums = re.findall(r'[-+]?\d+(?:\.\d+)?', resp)
    correct = any(abs(float(n) - expected) < 0.1 for n in nums[-5:]) if nums else False
    return correct, 1, time.time() - t0, 0


def run_math_symstep(prob: MathProblem, with_guidance: bool = False,
                     max_steps: int = 20) -> Tuple[bool, int, float, int]:
    t0 = time.time()
    tracker  = NumericTracker(prob)
    calls    = 0
    contradictions = 0
    history: List[str] = []

    seeds = tracker.seeds_remaining()
    seed_hint = (f"\nStart by asserting the seed variable '{seeds[0]}' "
                 f"(read its value directly from the problem text)."
                 if seeds else "")

    intro = (
        prob.text + "\n\n" + MATH_SYMSTEP_SYSTEM
        + "\n\nKnown variables so far: (none)" + seed_hint
        + "\nMake your first DEDUCE now."
    )
    history.append(intro)

    for _ in range(max_steps):
        prompt = "\n\n".join(history)
        response = call_llm(prompt)
        calls += 1
        history.append(f"[You]: {response}")

        if re.search(r"CONCLUDE\s*:\s*done", response, re.IGNORECASE):
            break

        # Parse DEDUCE: VarName = value
        m = re.search(
            r"DEDUCE\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([+-]?\d+(?:\.\d+)?)",
            response, re.IGNORECASE
        )
        if not m:
            history.append(
                "[Verifier]: Could not parse your deduction. "
                "Use exactly: DEDUCE: VariableName = numeric_value\n"
                f"Valid variable names: {', '.join(prob.variables)}"
            )
            continue

        var   = m.group(1)
        value = float(m.group(2))

        ok, msg = tracker.apply(var, value)

        if ok:
            known_str = ", ".join(f"{k}={v:.4g}" for k, v in tracker.known.items())
            feedback = f"[Verifier]: ✓ Accepted. {msg}\nKnown: {known_str}"
            if tracker.is_solved():
                feedback += "\n[Verifier]: All variables determined! Output: CONCLUDE: done"
            elif with_guidance:
                feedback += "\n" + tracker.guidance()
        else:
            contradictions += 1
            feedback = (
                f"[Verifier]: ✗ REJECTED — {msg}\n"
                "Please reconsider and make a corrected DEDUCE."
            )

        history.append(feedback)
        if tracker.is_solved():
            break

    last_var = prob.variables[-1]
    expected = prob.solution[last_var]
    known_val = tracker.known.get(last_var)
    correct = (known_val is not None and abs(known_val - expected) < 0.1)
    return correct, calls, time.time() - t0, contradictions


MATH_METHODS = {
    "direct":    run_math_direct,
    "cot":       run_math_cot,
    "symstep":   lambda p: run_math_symstep(p, with_guidance=False),
    "symstep_g": lambda p: run_math_symstep(p, with_guidance=True),
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_math_all(out_file: str = "math_results.json"):
    print("=" * 70)
    print(f"MWP-8 Math Word Problems  model={MODEL}")
    print("=" * 70)

    summary = {m: {"correct": 0, "total": 0, "calls": 0,
                   "contradictions": 0, "time": 0.0}
               for m in MATH_METHODS}
    per_problem = []

    for prob in MWP_PROBLEMS:
        print(f"\n  {prob.name} ({prob.difficulty})")
        row = {"puzzle": prob.name, "difficulty": prob.difficulty}
        for mname, mfn in MATH_METHODS.items():
            result = mfn(prob)
            correct, calls, elapsed, contra = result[0], result[1], result[2], result[3]
            summary[mname]["correct"]        += int(correct)
            summary[mname]["total"]          += 1
            summary[mname]["calls"]          += calls
            summary[mname]["contradictions"] += contra
            summary[mname]["time"]           += elapsed
            print(f"    {mname:<14} {'✓' if correct else '✗'}  calls={calls}  t={elapsed:.1f}s")
            row[mname] = {"correct": correct, "calls": calls,
                          "contradictions": contra, "time": round(elapsed, 1)}
        per_problem.append(row)

    print("\n" + "=" * 70)
    print(f"{'Method':<14} {'Acc%':>6}  {'Avg calls':>10}")
    print("-" * 40)
    for m, r in summary.items():
        acc   = r["correct"] / r["total"] * 100
        calls = r["calls"]   / r["total"]
        print(f"{m:<14} {acc:>5.0f}%  {calls:>9.1f}")

    out = {"model": MODEL, "summary": summary, "per_problem": per_problem}
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_file}")


if __name__ == "__main__":
    run_math_all()
