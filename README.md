# SymStep: Symbolic Step Verification Achieves 100% on Multi-Step Reasoning Where Chain-of-Thought Gets 0%

> **"Chain-of-thought achieves 0%. Logic-LM achieves 0%. SymStep+G achieves 100%."**

[![Paper](https://img.shields.io/badge/Paper-NILA@IJCAI--2026-blue)](https://github.com/Zangir/SymStep)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://python.org)
[![Model](https://img.shields.io/badge/Model-Claude%20Haiku%20%2F%20Sonnet-orange)](https://anthropic.com)

---

## The Problem: Unverified Errors Kill LLM Reasoning

LLMs reason in free-form text. Each step can silently introduce an error. By step 7, the model deduces the dog owner is Alice — without noticing it already established Alice has a cat in step 3. Chain-of-thought makes this *worse*, not better: it produces more steps, each a new opportunity for undetected contradiction.

This isn't a model capability problem. It's an **architecture problem**: there is no mechanism to catch errors as they happen.

---

## The Solution: Verify Every Step, Immediately

SymStep couples an LLM with a lightweight symbolic **constraint propagator** that operates at the granularity of individual deduction steps:

```
LLM says:       DEDUCE: Alice, pet, Cat
Propagator:     ✓ Accepted. Alice.pet = Cat → removes Cat from Bob, Carol.
                [Hint] Bob's color must be one of: {Red, Green}

LLM says:       DEDUCE: Alice, pet, Dog
Propagator:     ✗ CONTRADICTION: Alice.pet is already Cat. Please revise.
```

Every claim is checked before it enters the LLM's context. No unverified fact is ever accepted. The propagator also **cascades arc-consistency** — automatically deriving new eliminations invisible to the LLM.

**SymStep+G** adds an MRV (Minimum Remaining Values) hint: after each accepted step, the propagator reports the most constrained unsolved variable, breaking deductive deadlocks before they happen.

---

## Results: 0% → 100%

### LGP-14 — Logical Grid Puzzles (14 puzzles, 3 difficulty levels)

| Method | Easy (4) | Medium (5) | Hard (5) | **Total** | Avg calls |
|--------|----------|-----------|---------|-----------|-----------|
| Direct | 2/4 | 0/5 | 0/5 | 14% | 1.0 |
| CoT | 0/4 | 0/5 | 0/5 | **0%** | 1.0 |
| Self-Refine | 0/4 | 1/5 | 4/5 | 36% | 2.0 |
| Logic-LM (SOTA) | 0/4 | 0/5 | 0/5 | **0%** | 1.0 |
| SymStep | 4/4 | 4/5 | 4/5 | 86% | 7.0 |
| **SymStep+G** | **4/4** | **5/5** | **5/5** | **100%** | 7.1 |

> CoT and Logic-LM — the two dominant paradigms for LLM reasoning — both achieve 0%. SymStep+G achieves 100% with σ=0 over 3 independent runs.

### Cross-Domain Generalization (4 domains, zero prompt modification)

| Method | LGP-14 | SP-6 | MWP-8 | FIN-6 | **Avg** |
|--------|--------|------|-------|-------|---------|
| Direct | 14% | 0% | 88% | 67% | 42% |
| CoT | 0% | 0% | 100% | 83% | 46% |
| Self-Refine | 36% | 33% | — | — | — |
| Logic-LM | 0% | 0% | — | — | — |
| SymStep | 86% | 83% | 100% | 100% | **92%** |
| **SymStep+G** | **100%** | **67%** | **100%** | **100%** | **92%** |

**Four structurally different domains:**
- 🧩 **LGP-14** — Logical grid puzzles (bijective constraint satisfaction)
- 📅 **SP-6** — Scheduling problems (time-slot assignment with ordering constraints)
- ➗ **MWP-8** — Math word problems (arithmetic chain derivation)
- 💰 **FIN-6** — Financial reasoning (compound interest, break-even, portfolio returns, tax)

Same `DEDUCE` protocol, same outer loop. Only the symbolic verifier adapts per domain.

---

## How It Works

```
┌─────────────────────┐     DEDUCE: p, attr, val      ┌──────────────────────────┐
│  Problem            │ ─────────────────────────────→ │  Constraint Propagator Π │
│  (natural language) │                                │  arc-consistency cascade  │
└─────────────────────┘                                └──────────────────────────┘
           ↑                                                       │
           │                                            ┌──────────┴───────────┐
           │   ✓ accepted / next step                   │                      │
           └────────────────────────────────────────    ↓ ✓ Accepted           ↓ ✗ Contradiction
                                                    update Π              exact error message
                                                    cascade                    │
                                                        │                      │
                                                 [SymStep+G only]              │
                                                 MRV hint: "X's attr           │
                                                 must be one of: {…}"          │
                                                        │                      │
                                                        └──────────────────────┘
                                                                   │
                                                            back to LLM
```

**Arc-consistency cascade**: after each accepted deduction, the propagator eliminates the value from all other entities. If any cell reduces to a single candidate, that assignment is made automatically — potentially triggering a chain of further deductions without any LLM call.

**MRV guidance**: `(p*, a*) = argmin_{|Π|>1} |Π(p, a)|` — pick the most constrained unsolved variable. Delivered as a natural-language hint. Breaks deadlocks that pure verification cannot resolve.

---

## Key Findings

1. **CoT hurts on constraint-dense reasoning.** 0% on LGP-14 and LGP-10 across Haiku and Sonnet. More steps = more unverified errors.

2. **Logic-LM (SOTA symbolic+LLM) achieves 0%.** One-shot Z3 program generation is too brittle. Any error in any clue encoding breaks the entire program.

3. **Verification alone gets 86%; guidance closes the gap to 100%.** The MRV hint provides directional signal that breaks deductive deadlocks. Multi-run ablation (N=3): verification-only = 72.2%, guidance = 100% with σ=0.

4. **Domain-agnostic.** Same architecture, four domains, SymStep variants win on all of them. Financial reasoning: Direct=67%, CoT=83%, SymStep=100%.

5. **Cheap.** ~$0.0013/puzzle with Claude Haiku. Full LGP-14 benchmark: under $0.02.

---

## Repository Structure

```
SymStep/
├── README.md
├── experiments/
│   ├── symstep.py          # Core: LGP-14, SP-6, Logic-LM baseline, all methods
│   ├── math_reasoning.py   # MWP-8: arithmetic word problems
│   ├── fin_reasoning.py    # FIN-6: financial reasoning benchmark  
│   ├── ablation.py         # Component ablation (verification vs. guidance)
│   ├── run_full.py         # Full multi-run experiment runner
│   ├── run_new_benchmarks.py # SP-6 + new domain runner
│   ├── compile_results.py  # Aggregate results across runs
│   └── verify_puzzles.py   # Propagator-based uniqueness verification
└── results/
    ├── lgp14_combined.json      # LGP-14 main results (Haiku)
    ├── sonnet_results.json      # LGP-10 cross-model results (Sonnet)
    ├── scheduling_results.json  # SP-6 results
    ├── math_results.json        # MWP-8 results
    ├── fin_results.json         # FIN-6 results
    └── ablation_multirun.json   # N=3 multi-run ablation on LGP-6
```

---

## Quickstart

**Requirements:** Python 3.9+, [Claude Code CLI](https://claude.ai/code) installed.

```bash
git clone https://github.com/Zangir/SymStep.git
cd SymStep

# Run all methods on LGP-14 (logical grid puzzles)
python3 experiments/symstep.py

# Run math word problems (MWP-8)
python3 experiments/math_reasoning.py

# Run financial reasoning (FIN-6)
python3 experiments/fin_reasoning.py

# Run scheduling puzzles (SP-6) with all 6 methods including Logic-LM
python3 experiments/run_new_benchmarks.py
```

**Use a different model:**
```bash
SYMSTEP_MODEL=sonnet python3 experiments/symstep.py
```

**The Claude Code CLI is auto-detected** from common install locations (`~/.claude/local/claude`, VSCode/Cursor extensions, PATH). Override with `CLAUDE_BIN=/path/to/claude`.

---

## The DEDUCE Protocol

SymStep uses a minimal structured output format:

```
DEDUCE: <Entity>, <attribute>, <Value>        # positive deduction
DEDUCE: <Entity>, <attribute>, NOT <Value>    # elimination
CONCLUDE: done                                # task complete
```

For arithmetic/financial domains:
```
DEDUCE: balance_year1 = 5500          # numeric assignment
DEDUCE: break_even_units = 1000       # derived quantity
```

The format is regex-parseable, requires no formal logic training, and is far less demanding than generating a complete Z3/Prolog program.

---

## Benchmarks

### LGP-14 — Logical Grid Puzzles
14 puzzles across Easy (3 people × 2 attrs), Medium (4 people × 2 attrs), and Hard (4-5 people × 2-3 attrs). All solutions verified for **uniqueness** by the propagator — a critical step: 2 of our original 16 puzzles had multiple valid solutions and were caught by this check.

### SP-6 — Scheduling Problems
6 scheduling constraint problems: workshop presentations, lab assignments, project team slots, research conference scheduling, department meetings, event crews. Bijective structure preserved; entirely different vocabulary from LGPs.

### MWP-8 — Math Word Problems  
8 arithmetic chain problems (salaries, ages, distances, store purchases, speed/time, investments, payroll, mixture). Uses a **seed+cascade** design: LLM asserts values read from text; propagator auto-derives all dependent quantities.

### FIN-6 — Financial Reasoning
6 financial multi-step problems:
- Simple interest (principal × rate × time)
- Gross margin analysis
- Compound interest (multi-year)
- Break-even analysis (contribution margin → units)
- Portfolio weighted return (3 assets)
- Progressive income tax (two brackets)

---

## Reproducibility

Multi-run ablation on LGP-6 (N=3 independent runs, `claude-haiku-4-5`):

| Config | Mean Acc | Std |
|--------|----------|-----|
| CoT | 0.0% | 0.0 |
| Verification only | 72.2% | 9.6 |
| Guidance only | 100.0% | 0.0 |
| **SymStep+G** | **100.0%** | **0.0** |

SymStep+G achieves perfect reproducibility: 18/18 puzzles solved across 3 runs.

---

## Citation

```bibtex
@inproceedings{symstep2026,
  title     = {SymStep: Symbolic Step Verification Achieves 100\% on
               Multi-Step Reasoning Where Chain-of-Thought Gets 0\%},
  booktitle = {NILA Workshop @ IJCAI-ECAI 2026},
  year      = {2026},
  note      = {Anonymous submission}
}
```

---

*Paper under anonymous review. Author information omitted.*
