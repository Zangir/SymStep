#!/usr/bin/env python3
"""
Financial Reasoning benchmark (FIN-6) for SymStep evaluation.

Six multi-step financial problems covering compound interest, profit/margin
analysis, break-even analysis, portfolio returns, tax calculation, and ROI.

Same seed+cascade design as MWP-8: the LLM reads explicit seed values from
the problem text; the NumericTracker propagates all derived quantities.

DEDUCE format:  DEDUCE: VarName = numeric_value
"""

import re, time, json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from math_reasoning import MathProblem, NumericTracker, MATH_METHODS, MATH_SYMSTEP_SYSTEM
from symstep import call_llm, MODEL


FIN_PROBLEMS = [

    # ── EASY ─────────────────────────────────────────────────────────────────
    MathProblem(
        name="FIN-E1-SimpleInterest", difficulty="easy",
        text=(
            "A bank offers a simple interest savings account.\n"
            "You deposit principal = $8000.\n"
            "The annual interest rate is annual_rate = 0.05 (5% per year).\n"
            "You keep the money for years = 4 years.\n"
            "What is your total_amount at the end?\n\n"
            "Variables to compute (use these exact names in DEDUCE statements):\n"
            "  principal    — initial deposit in dollars\n"
            "  annual_rate  — annual interest rate as a decimal\n"
            "  years        — number of years\n"
            "  interest     — total interest earned (principal × annual_rate × years)\n"
            "  total_amount — final balance (principal + interest)"
        ),
        variables=["principal", "annual_rate", "years", "interest", "total_amount"],
        constraints={
            "interest":     lambda k: k["principal"] * k["annual_rate"] * k["years"],
            "total_amount": lambda k: k["principal"] + k["interest"],
        },
        solution={"principal": 8000.0, "annual_rate": 0.05, "years": 4.0,
                  "interest": 1600.0, "total_amount": 9600.0},
    ),

    MathProblem(
        name="FIN-E2-GrossMargin", difficulty="easy",
        text=(
            "A retailer analyses its quarterly performance.\n"
            "Total revenue = revenue = $15000.\n"
            "Cost of goods sold = cogs = $9000.\n"
            "What is the gross_margin_pct (gross profit as a percentage of revenue)?\n\n"
            "Variables to compute (use these exact names in DEDUCE statements):\n"
            "  revenue         — total revenue in dollars\n"
            "  cogs            — cost of goods sold in dollars\n"
            "  gross_profit    — revenue minus cogs\n"
            "  gross_margin_pct — gross profit as a percentage of revenue "
                                 "(gross_profit ÷ revenue × 100)"
        ),
        variables=["revenue", "cogs", "gross_profit", "gross_margin_pct"],
        constraints={
            "gross_profit":     lambda k: k["revenue"] - k["cogs"],
            "gross_margin_pct": lambda k: k["gross_profit"] / k["revenue"] * 100,
        },
        solution={"revenue": 15000.0, "cogs": 9000.0,
                  "gross_profit": 6000.0, "gross_margin_pct": 40.0},
    ),

    # ── MEDIUM ────────────────────────────────────────────────────────────────
    MathProblem(
        name="FIN-M1-CompoundInterest", difficulty="medium",
        text=(
            "An investor deposits money in a compound interest account.\n"
            "Initial deposit: principal = $5000.\n"
            "Annual interest rate: annual_rate = 0.10 (10%).\n"
            "Interest compounds annually for 2 years.\n"
            "What is total_interest earned after 2 years?\n\n"
            "Variables to compute (use these exact names in DEDUCE statements):\n"
            "  principal      — initial deposit in dollars\n"
            "  annual_rate    — annual interest rate as a decimal\n"
            "  balance_year1  — balance after year 1 (principal × (1 + annual_rate))\n"
            "  balance_year2  — balance after year 2 (balance_year1 × (1 + annual_rate))\n"
            "  total_interest — interest earned over 2 years (balance_year2 − principal)"
        ),
        variables=["principal", "annual_rate",
                   "balance_year1", "balance_year2", "total_interest"],
        constraints={
            "balance_year1":  lambda k: k["principal"] * (1 + k["annual_rate"]),
            "balance_year2":  lambda k: k["balance_year1"] * (1 + k["annual_rate"]),
            "total_interest": lambda k: k["balance_year2"] - k["principal"],
        },
        solution={"principal": 5000.0, "annual_rate": 0.10,
                  "balance_year1": 5500.0, "balance_year2": 6050.0,
                  "total_interest": 1050.0},
    ),

    MathProblem(
        name="FIN-M2-BreakEven", difficulty="medium",
        text=(
            "A startup analyses its break-even point.\n"
            "Monthly fixed costs: fixed_costs = $40000.\n"
            "Variable cost per unit: variable_cost = $25.\n"
            "Selling price per unit: selling_price = $65.\n"
            "How many units must be sold to break even (break_even_units)?\n\n"
            "Variables to compute (use these exact names in DEDUCE statements):\n"
            "  fixed_costs          — total fixed monthly costs in dollars\n"
            "  variable_cost        — variable cost per unit in dollars\n"
            "  selling_price        — selling price per unit in dollars\n"
            "  contribution_margin  — profit per unit (selling_price − variable_cost)\n"
            "  break_even_units     — units needed to break even "
                                     "(fixed_costs ÷ contribution_margin)\n"
            "  break_even_revenue   — revenue at break-even "
                                     "(break_even_units × selling_price)"
        ),
        variables=["fixed_costs", "variable_cost", "selling_price",
                   "contribution_margin", "break_even_units", "break_even_revenue"],
        constraints={
            "contribution_margin": lambda k: k["selling_price"] - k["variable_cost"],
            "break_even_units":    lambda k: k["fixed_costs"] / k["contribution_margin"],
            "break_even_revenue":  lambda k: k["break_even_units"] * k["selling_price"],
        },
        solution={"fixed_costs": 40000.0, "variable_cost": 25.0, "selling_price": 65.0,
                  "contribution_margin": 40.0, "break_even_units": 1000.0,
                  "break_even_revenue": 65000.0},
    ),

    # ── HARD ──────────────────────────────────────────────────────────────────
    MathProblem(
        name="FIN-H1-Portfolio", difficulty="hard",
        text=(
            "An investor holds a three-asset portfolio.\n"
            "Asset A: value_A = $12000, with annual return rate_A = 0.08 (8%).\n"
            "Asset B: value_B = $5000, with annual return rate_B = 0.15 (15%).\n"
            "Asset C: value_C = $3000, with annual return rate_C = 0.04 (4%).\n"
            "What is portfolio_return_pct, the weighted annual return in percent?\n\n"
            "Variables to compute (use these exact names in DEDUCE statements):\n"
            "  value_A             — value of Asset A in dollars\n"
            "  rate_A              — annual return rate of Asset A (decimal)\n"
            "  value_B             — value of Asset B in dollars\n"
            "  rate_B              — annual return rate of Asset B (decimal)\n"
            "  value_C             — value of Asset C in dollars\n"
            "  rate_C              — annual return rate of Asset C (decimal)\n"
            "  total_portfolio     — sum of all asset values\n"
            "  weight_A            — portfolio weight of A (value_A ÷ total_portfolio)\n"
            "  weight_B            — portfolio weight of B\n"
            "  weight_C            — portfolio weight of C\n"
            "  weighted_return     — weighted return "
                                    "(weight_A×rate_A + weight_B×rate_B + weight_C×rate_C)\n"
            "  portfolio_return_pct — weighted_return × 100 (in percent)"
        ),
        variables=["value_A", "rate_A", "value_B", "rate_B", "value_C", "rate_C",
                   "total_portfolio", "weight_A", "weight_B", "weight_C",
                   "weighted_return", "portfolio_return_pct"],
        constraints={
            "total_portfolio":      lambda k: k["value_A"] + k["value_B"] + k["value_C"],
            "weight_A":             lambda k: k["value_A"] / k["total_portfolio"],
            "weight_B":             lambda k: k["value_B"] / k["total_portfolio"],
            "weight_C":             lambda k: k["value_C"] / k["total_portfolio"],
            "weighted_return":      lambda k: (k["weight_A"] * k["rate_A"]
                                               + k["weight_B"] * k["rate_B"]
                                               + k["weight_C"] * k["rate_C"]),
            "portfolio_return_pct": lambda k: k["weighted_return"] * 100,
        },
        solution={
            "value_A": 12000.0, "rate_A": 0.08,
            "value_B": 5000.0,  "rate_B": 0.15,
            "value_C": 3000.0,  "rate_C": 0.04,
            "total_portfolio": 20000.0,
            "weight_A": 0.60,   "weight_B": 0.25,   "weight_C": 0.15,
            "weighted_return": 0.0925, "portfolio_return_pct": 9.25,
        },
    ),

    MathProblem(
        name="FIN-H2-TaxBracket", difficulty="hard",
        text=(
            "A taxpayer calculates their income tax under a two-bracket system.\n"
            "Annual income: income = $80000.\n"
            "First bracket covers up to bracket1_limit = $50000 at bracket1_rate = 0.20 (20%).\n"
            "Income above $50000 is taxed at bracket2_rate = 0.30 (30%).\n"
            "What is effective_tax_rate, the effective tax rate as a percentage?\n\n"
            "Variables to compute (use these exact names in DEDUCE statements):\n"
            "  income               — total annual income in dollars\n"
            "  bracket1_limit       — upper limit of first bracket in dollars\n"
            "  bracket1_rate        — tax rate for first bracket (decimal)\n"
            "  bracket2_rate        — tax rate for second bracket (decimal)\n"
            "  tax_bracket1         — tax paid on first bracket (bracket1_limit × bracket1_rate)\n"
            "  income_above_bracket1 — income above the first bracket (income − bracket1_limit)\n"
            "  tax_bracket2         — tax paid on second bracket "
                                     "(income_above_bracket1 × bracket2_rate)\n"
            "  total_tax            — total tax paid (tax_bracket1 + tax_bracket2)\n"
            "  effective_tax_rate   — effective rate as percentage (total_tax ÷ income × 100)"
        ),
        variables=["income", "bracket1_limit", "bracket1_rate", "bracket2_rate",
                   "tax_bracket1", "income_above_bracket1", "tax_bracket2",
                   "total_tax", "effective_tax_rate"],
        constraints={
            "tax_bracket1":         lambda k: k["bracket1_limit"] * k["bracket1_rate"],
            "income_above_bracket1": lambda k: k["income"] - k["bracket1_limit"],
            "tax_bracket2":         lambda k: k["income_above_bracket1"] * k["bracket2_rate"],
            "total_tax":            lambda k: k["tax_bracket1"] + k["tax_bracket2"],
            "effective_tax_rate":   lambda k: k["total_tax"] / k["income"] * 100,
        },
        solution={
            "income": 80000.0, "bracket1_limit": 50000.0,
            "bracket1_rate": 0.20, "bracket2_rate": 0.30,
            "tax_bracket1": 10000.0, "income_above_bracket1": 30000.0,
            "tax_bracket2": 9000.0,  "total_tax": 19000.0,
            "effective_tax_rate": 23.75,
        },
    ),
]


def run_fin_all(out_file: str = "fin_results.json"):
    print("=" * 70)
    print(f"FIN-6 Financial Reasoning  model={MODEL}")
    print("=" * 70)

    summary = {m: {"correct": 0, "total": 0, "calls": 0,
                   "contradictions": 0, "time": 0.0}
               for m in MATH_METHODS}
    per_problem = []

    for prob in FIN_PROBLEMS:
        print(f"\n  {prob.name} ({prob.difficulty})")
        row = {"problem": prob.name, "difficulty": prob.difficulty}
        for mname, mfn in MATH_METHODS.items():
            result = mfn(prob)
            correct, calls, elapsed, contra = result
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
    run_fin_all()
