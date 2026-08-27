#!/usr/bin/env python3
"""GraphStep core: a general finite-domain constraint engine with explanations.

The reasoning substrate is a typed factor graph:
  - variable nodes: finite domains over hashable values,
  - constraint nodes: typed relations implementing filter/status/check/negate.

Everything a task needs is expressed as (variables, constraints); the engine
provides worklist propagation (generalized arc consistency), MRV + dom/wdeg
backtracking search, uniqueness certification, human-readable contradiction
explanations (via a removal ledger), and deletion-based UNSAT-core extraction
so a repair loop can re-translate exactly the clues that conflict.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Iterable, Any
import itertools

Value = Any  # hashable


class Fail(Exception):
    """Raised by a constraint when the current domains are inconsistent."""

    def __init__(self, constraint: "Constraint", note: str):
        self.constraint = constraint
        self.note = note
        super().__init__(note)


class Constraint:
    """Base class. Subclasses define semantics over `self.vars`."""

    _next_id = itertools.count(1)

    def __init__(self, vars: List[str], origin: Optional[str] = None):
        self.vars = list(vars)
        self.origin = origin          # e.g. the clue text this came from
        self.cid = next(Constraint._next_id)
        self.weight = 1               # dom/wdeg failure weight

    # -- required -------------------------------------------------------
    def filter(self, doms: Dict[str, set]) -> List[Tuple[str, Value, str]]:
        """Return [(var, value_to_remove, reason)]. Raise Fail on wipeout."""
        raise NotImplementedError

    def check(self, asgn: Dict[str, Value]) -> bool:
        """Truth under a full assignment (used for certification/search leaves)."""
        raise NotImplementedError

    # -- optional (needed under reification) ------------------------------
    def status(self, doms: Dict[str, set]) -> int:
        """+1 entailed / -1 falsified / 0 unknown, judged from domains alone.
        Default: decidable only when all vars are singletons."""
        if all(len(doms[v]) == 1 for v in self.vars):
            return 1 if self.check({v: next(iter(doms[v])) for v in self.vars}) else -1
        return 0

    def negate(self) -> "Constraint":
        raise NotImplementedError(f"{type(self).__name__} has no negation")

    def describe(self) -> str:
        return f"{type(self).__name__}({', '.join(map(str, self.vars))})"

    def __repr__(self):
        return self.describe()


class Problem:
    def __init__(self, variables: Dict[str, Iterable[Value]],
                 constraints: List[Constraint]):
        self.variables = {v: frozenset(d) for v, d in variables.items()}
        self.constraints = list(constraints)
        for c in self.constraints:
            for v in c.vars:
                if v not in self.variables:
                    raise ValueError(
                        f"constraint {c.describe()} references unknown variable "
                        f"{v!r} (origin: {c.origin})")


class Ledger:
    """Records why each value was removed -> explanations & shallow proofs."""

    def __init__(self):
        self.removals: Dict[Tuple[str, Value], Tuple[Constraint, str]] = {}

    def record(self, var: str, val: Value, constraint: Constraint, note: str):
        self.removals.setdefault((var, val), (constraint, note))

    def why(self, var: str, val: Value) -> Optional[str]:
        hit = self.removals.get((var, val))
        if hit is None:
            return None
        c, note = hit
        src = f" [from clue: {c.origin}]" if c.origin else ""
        return f"{note}{src}"


class Result:
    def __init__(self, status: str, solutions: List[Dict[str, Value]],
                 explanation: str = "", stats: Optional[dict] = None):
        self.status = status              # SOLVED | UNSAT | AMBIGUOUS | UNKNOWN
        self.solutions = solutions
        self.explanation = explanation
        self.stats = stats or {}

    def __repr__(self):
        return f"Result({self.status}, n_solutions={len(self.solutions)})"


class Engine:
    """Worklist GAC propagation + MRV/dom-wdeg backtracking search."""

    def __init__(self, problem: Problem):
        self.problem = problem
        self.watches: Dict[str, List[Constraint]] = {v: [] for v in problem.variables}
        for c in problem.constraints:
            for v in c.vars:
                self.watches[v].append(c)

    # ---------------------------------------------------------------- propagation
    def propagate(self, doms: Dict[str, set], ledger: Ledger,
                  queue: Optional[List[Constraint]] = None) -> None:
        """Run constraints to fixpoint. Mutates doms. Raises Fail on conflict."""
        pending = list(self.problem.constraints) if queue is None else list(queue)
        in_queue = {c.cid for c in pending}
        while pending:
            c = pending.pop()
            in_queue.discard(c.cid)
            try:
                removals = c.filter(doms)
            except Fail:
                c.weight += 1
                raise
            for var, val, note in removals:
                if val not in doms[var]:
                    continue
                doms[var].discard(val)
                ledger.record(var, val, c, note)
                if not doms[var]:
                    c.weight += 1
                    raise Fail(c, f"{var} has no possible value left ({note})")
                for dep in self.watches[var]:
                    if dep.cid != c.cid and dep.cid not in in_queue:
                        pending.append(dep)
                        in_queue.add(dep.cid)

    # ---------------------------------------------------------------- search
    def solve(self, max_solutions: int = 2, node_limit: int = 500_000) -> Result:
        doms = {v: set(d) for v, d in self.problem.variables.items()}
        ledger = Ledger()
        stats = {"nodes": 0, "failures": 0}
        try:
            self.propagate(doms, ledger)
        except Fail as f:
            return Result("UNSAT", [], self._explain(f, doms, ledger), stats)

        solutions: List[Dict[str, Value]] = []

        def branch(doms: Dict[str, set]) -> bool:
            """Return True to stop (enough solutions found)."""
            stats["nodes"] += 1
            if stats["nodes"] > node_limit:
                raise TimeoutError("node limit exceeded")
            undecided = [v for v in doms if len(doms[v]) > 1]
            if not undecided:
                asgn = {v: next(iter(doms[v])) for v in doms}
                if all(c.check(asgn) for c in self.problem.constraints):
                    solutions.append(asgn)
                    return len(solutions) >= max_solutions
                return False
            # MRV with dom/wdeg tie-break
            var = min(undecided,
                      key=lambda v: (len(doms[v]),
                                     -sum(c.weight for c in self.watches[v])))
            for val in sorted(doms[var], key=repr):
                child = {w: set(s) for w, s in doms.items()}
                child[var] = {val}
                try:
                    self.propagate(child, ledger, queue=self.watches[var])
                    if branch(child):
                        return True
                except Fail:
                    stats["failures"] += 1
            return False

        try:
            branch(doms)
        except TimeoutError:
            return Result("UNKNOWN", solutions, "search node limit exceeded", stats)

        if not solutions:
            return Result("UNSAT", [], "exhaustive search found no solution", stats)
        if len(solutions) == 1:
            return Result("SOLVED", solutions, "unique solution certified", stats)
        return Result("AMBIGUOUS", solutions,
                      "constraints admit more than one solution", stats)

    # ---------------------------------------------------------------- explanations
    def _explain(self, f: Fail, doms: Dict[str, set], ledger: Ledger) -> str:
        lines = [f"CONTRADICTION at {f.constraint.describe()}: {f.note}"]
        if f.constraint.origin:
            lines.append(f"  source clue: {f.constraint.origin}")
        for v in f.constraint.vars:
            gone = [val for val in self.problem.variables[v] if val not in doms[v]]
            for val in gone[:4]:
                why = ledger.why(v, val)
                if why:
                    lines.append(f"  {v} != {val!r} because {why}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- unsat core
    def unsat_core(self, node_limit: int = 200_000) -> List[Constraint]:
        """Deletion-based minimal-ish core. Only meaningful if problem is UNSAT."""
        core = list(self.problem.constraints)
        i = 0
        while i < len(core):
            trial = core[:i] + core[i + 1:]
            sub = Engine(Problem(self.problem.variables, trial))
            if sub.solve(max_solutions=1, node_limit=node_limit).status == "UNSAT":
                core.pop(i)          # removable: still UNSAT without it
            else:
                i += 1               # needed for the conflict
        return core
