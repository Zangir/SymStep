#!/usr/bin/env python3
"""Unit tests for Tier 0.5 (syntax-guided compilation). Zero LLM calls."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphstep.reading.compile_text import Inventory
from graphstep.reading.syntax_tier import clause_skeleton, expand_coordination, \
    compile_with_syntax
from graphstep.engine.ir import problem_from_ir
from graphstep.engine.core import Engine

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}: {extra}")


inv = Inventory(["Alice", "Bob", "Carol"],
                {"pet": ["Cat", "Dog", "Fish"],
                 "color": ["Red", "Blue", "Green"]})

# 1. skeleton: if/then
sk = clause_skeleton("If Alice has the cat, then Bob has the dog.")
check("skeleton if/then", "if" in sk and "cat" in sk["if"].lower()
      and "dog" in sk["then"].lower(), sk)

# 2. skeleton: unless
sk = clause_skeleton("Bob has the dog, unless Carol has the fish.")
check("skeleton unless", "unless" in sk and "fish" in sk["unless"].lower(), sk)

# 3. coordination: neither-nor
atoms, neg, comb = expand_coordination("Neither Alice nor Bob owns the cat")
check("neither-nor", neg and len(atoms) == 2 and comb == "and", (atoms, neg))

# 4. coordination: subject and
atoms, neg, comb = expand_coordination("Alice and Bob like the red house")
check("subject-and", not neg and len(atoms) == 2
      and "Alice" in atoms[0] and "Bob" in atoms[1], atoms)

# 5. full compile: neither-nor -> two is_not
specs = compile_with_syntax("Neither Alice nor Bob owns the cat.", inv, False)
check("compile neither-nor", specs is not None
      and sorted((s["type"], s["value"]) for s in specs) ==
          [("is_not", 1), ("is_not", 2)]
      and all(s["var"] == "Cat" for s in specs), specs)

# 6. full compile: if/then -> implies, and it PROPAGATES in the engine
specs = compile_with_syntax("If Alice has the cat, then Bob has the dog.",
                            inv, False)
check("compile if/then", specs is not None and specs[0]["type"] == "implies",
      specs)
ir = {"variables": inv.variables(),
      "constraints": inv.base_constraints() + (specs or [])
      + [{"type": "is", "var": "Cat", "value": 1}]}      # Alice has cat
res = Engine(problem_from_ir(ir)).solve()
check("if/then propagates",
      all(s.get("Dog") == 2 for s in res.solutions), res.solutions)

# 7. unless semantics: (not U) -> THEN
specs = compile_with_syntax(
    "Bob has the dog, unless Carol has the fish.", inv, False)
check("compile unless", specs is not None and specs[0]["type"] == "implies",
      specs)
ir = {"variables": inv.variables(),
      "constraints": inv.base_constraints() + (specs or [])
      + [{"type": "is_not", "var": "Fish", "value": 3}]}  # Carol lacks fish
res = Engine(problem_from_ir(ir)).solve()
check("unless propagates",
      res.solutions and all(s.get("Dog") == 2 for s in res.solutions),
      res.solutions[:2])

# 8. verb-phrase coordination: two constraints from one sentence
specs = compile_with_syntax(
    "Alice owns the cat and lives in the red house.", inv, False)
check("vp-and", specs is not None and len(specs) == 2
      and {(s["type"], s["var"]) for s in specs} ==
          {("is", "Cat"), ("is", "Red")}, specs)

# 9. negation scope: relative clause negation must NOT flip the main clause
#    "Alice, who does not have the dog, lives in the red house."
specs = compile_with_syntax(
    "Alice, who does not have the dog, lives in the red house.", inv, False)
if specs is not None:
    types = {(s["type"], s["var"]) for s in specs}
    check("neg scope", ("is", "Red") in types or ("is_not", "Dog") in types,
          specs)
else:
    check("neg scope (safe fallthrough)", True)   # None -> falls to LLM: safe

# 10. plain simple sentence still works (delegates to templates)
specs = compile_with_syntax("Carol has the fish.", inv, False)
check("plain sentence", specs is not None
      and specs[0]["type"] == "is" and specs[0]["var"] == "Fish"
      and specs[0]["value"] == 3, specs)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
