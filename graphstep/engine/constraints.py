#!/usr/bin/env python3
"""GraphStep constraint library.

Primitive relations plus propositional combinators (Not/Or/Implies/Iff) give a
general finite-domain logic: any finite relation is expressible via Table, and
any propositional structure over relations via the combinators. Each type
implements filter (pruning + reason strings), check, status, and (where
meaningful) negate — which is what the combinators need for reification.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any
from .core import Constraint, Fail

Value = Any


# ---------------------------------------------------------------- unary
class Is(Constraint):
    """var == value"""

    def __init__(self, var: str, value: Value, origin=None):
        super().__init__([var], origin)
        self.value = value

    def filter(self, doms):
        var = self.vars[0]
        if self.value not in doms[var]:
            raise Fail(self, f"{var} must be {self.value!r} but that value "
                             f"is already excluded")
        return [(var, v, f"{var} is fixed to {self.value!r}")
                for v in list(doms[var]) if v != self.value]

    def check(self, asgn):
        return asgn[self.vars[0]] == self.value

    def status(self, doms):
        var = self.vars[0]
        if self.value not in doms[var]:
            return -1
        return 1 if doms[var] == {self.value} else 0

    def negate(self):
        return IsNot(self.vars[0], self.value, self.origin)

    def describe(self):
        return f"{self.vars[0]} = {self.value!r}"


class IsNot(Constraint):
    """var != value"""

    def __init__(self, var: str, value: Value, origin=None):
        super().__init__([var], origin)
        self.value = value

    def filter(self, doms):
        var = self.vars[0]
        if doms[var] == {self.value}:
            raise Fail(self, f"{var} is already forced to {self.value!r}, "
                             f"which this forbids")
        if self.value in doms[var]:
            return [(var, self.value, f"{var} may not be {self.value!r}")]
        return []

    def check(self, asgn):
        return asgn[self.vars[0]] != self.value

    def status(self, doms):
        var = self.vars[0]
        if self.value not in doms[var]:
            return 1
        return -1 if doms[var] == {self.value} else 0

    def negate(self):
        return Is(self.vars[0], self.value, self.origin)

    def describe(self):
        return f"{self.vars[0]} != {self.value!r}"


# ---------------------------------------------------------------- binary
class SameValue(Constraint):
    """x == y  (equality of two variables)"""

    def __init__(self, x: str, y: str, origin=None):
        super().__init__([x, y], origin)

    def filter(self, doms):
        x, y = self.vars
        inter = doms[x] & doms[y]
        if not inter:
            raise Fail(self, f"{x} and {y} must be equal but share no "
                             f"possible value")
        out = []
        for var, other in ((x, y), (y, x)):
            for v in list(doms[var]):
                if v not in inter:
                    out.append((var, v,
                                f"{var}={v!r} impossible: {other} can never "
                                f"take {v!r} and {x} must equal {y}"))
        return out

    def check(self, asgn):
        return asgn[self.vars[0]] == asgn[self.vars[1]]

    def status(self, doms):
        x, y = self.vars
        if not (doms[x] & doms[y]):
            return -1
        if len(doms[x]) == 1 and doms[x] == doms[y]:
            return 1
        return 0

    def negate(self):
        return DiffValue(*self.vars, origin=self.origin)

    def describe(self):
        return f"{self.vars[0]} = {self.vars[1]}"


class DiffValue(Constraint):
    """x != y"""

    def __init__(self, x: str, y: str, origin=None):
        super().__init__([x, y], origin)

    def filter(self, doms):
        x, y = self.vars
        out = []
        for var, other in ((x, y), (y, x)):
            if len(doms[other]) == 1:
                v = next(iter(doms[other]))
                if doms[var] == {v}:
                    raise Fail(self, f"{x} and {y} must differ but both are "
                                     f"forced to {v!r}")
                if v in doms[var]:
                    out.append((var, v, f"{other} is {v!r} and {var} must differ"))
        return out

    def check(self, asgn):
        return asgn[self.vars[0]] != asgn[self.vars[1]]

    def status(self, doms):
        x, y = self.vars
        if not (doms[x] & doms[y]):
            return 1
        if len(doms[x]) == 1 and doms[x] == doms[y]:
            return -1
        return 0

    def negate(self):
        return SameValue(*self.vars, origin=self.origin)

    def describe(self):
        return f"{self.vars[0]} != {self.vars[1]}"


class Less(Constraint):
    """x < y (numeric domains) — bounds consistency."""

    def __init__(self, x: str, y: str, origin=None):
        super().__init__([x, y], origin)

    def filter(self, doms):
        x, y = self.vars
        if min(doms[x]) >= max(doms[y]):
            raise Fail(self, f"{x} < {y} impossible: min({x})={min(doms[x])} "
                             f">= max({y})={max(doms[y])}")
        out = []
        ymax, xmin = max(doms[y]), min(doms[x])
        for v in list(doms[x]):
            if v >= ymax:
                out.append((x, v, f"{x}={v} leaves no room for {y} > {v}"))
        for v in list(doms[y]):
            if v <= xmin:
                out.append((y, v, f"{y}={v} leaves no room for {x} < {v}"))
        return out

    def check(self, asgn):
        return asgn[self.vars[0]] < asgn[self.vars[1]]

    def status(self, doms):
        x, y = self.vars
        if max(doms[x]) < min(doms[y]):
            return 1
        if min(doms[x]) >= max(doms[y]):
            return -1
        return 0

    def negate(self):                       # not(x < y)  ==  y <= x  ==  y < x+? — use GreaterEq
        return GreaterEq(self.vars[0], self.vars[1], self.origin)

    def describe(self):
        return f"{self.vars[0]} < {self.vars[1]}"


class GreaterEq(Constraint):
    """x >= y (numeric)."""

    def __init__(self, x: str, y: str, origin=None):
        super().__init__([x, y], origin)

    def filter(self, doms):
        x, y = self.vars
        if max(doms[x]) < min(doms[y]):
            raise Fail(self, f"{x} >= {y} impossible: max({x}) < min({y})")
        out = []
        ymin, xmax = min(doms[y]), max(doms[x])
        for v in list(doms[x]):
            if v < ymin:
                out.append((x, v, f"{x}={v} would be below every value of {y}"))
        for v in list(doms[y]):
            if v > xmax:
                out.append((y, v, f"{y}={v} would exceed every value of {x}"))
        return out

    def check(self, asgn):
        return asgn[self.vars[0]] >= asgn[self.vars[1]]

    def negate(self):
        return Less(self.vars[0], self.vars[1], self.origin)

    def describe(self):
        return f"{self.vars[0]} >= {self.vars[1]}"


class Offset(Constraint):
    """x == y + k (numeric)."""

    def __init__(self, x: str, y: str, k: int, origin=None):
        super().__init__([x, y], origin)
        self.k = k

    def filter(self, doms):
        x, y = self.vars
        xs = {v for v in doms[x] if v - self.k in doms[y]}
        ys = {v for v in doms[y] if v + self.k in doms[x]}
        if not xs or not ys:
            raise Fail(self, f"{x} = {y} + {self.k} has no support in current "
                             f"domains")
        out = [(x, v, f"{v} - {self.k} not available for {y}")
               for v in list(doms[x]) if v not in xs]
        out += [(y, v, f"{v} + {self.k} not available for {x}")
                for v in list(doms[y]) if v not in ys]
        return out

    def check(self, asgn):
        return asgn[self.vars[0]] == asgn[self.vars[1]] + self.k

    def status(self, doms):
        x, y = self.vars
        if not any(v - self.k in doms[y] for v in doms[x]):
            return -1
        if len(doms[x]) == 1 and len(doms[y]) == 1:
            return 1 if next(iter(doms[x])) == next(iter(doms[y])) + self.k else -1
        return 0

    def negate(self):
        return NotOffset(self.vars[0], self.vars[1], self.k, self.origin)

    def describe(self):
        return f"{self.vars[0]} = {self.vars[1]} + {self.k}"


class NotOffset(Constraint):
    """x != y + k."""

    def __init__(self, x: str, y: str, k: int, origin=None):
        super().__init__([x, y], origin)
        self.k = k

    def filter(self, doms):
        x, y = self.vars
        out = []
        if len(doms[y]) == 1:
            v = next(iter(doms[y])) + self.k
            if doms[x] == {v}:
                raise Fail(self, f"{x} = {y} + {self.k} is forced but forbidden")
            if v in doms[x]:
                out.append((x, v, f"{x}={v} would equal {y} + {self.k}"))
        if len(doms[x]) == 1:
            v = next(iter(doms[x])) - self.k
            if doms[y] == {v}:
                raise Fail(self, f"{x} = {y} + {self.k} is forced but forbidden")
            if v in doms[y]:
                out.append((y, v, f"{y}={v} would make {x} = {y} + {self.k}"))
        return out

    def check(self, asgn):
        return asgn[self.vars[0]] != asgn[self.vars[1]] + self.k

    def negate(self):
        return Offset(self.vars[0], self.vars[1], self.k, self.origin)

    def describe(self):
        return f"{self.vars[0]} != {self.vars[1]} + {self.k}"


class AbsDiff(Constraint):
    """|x - y| == k (numeric); k=1 is adjacency."""

    def __init__(self, x: str, y: str, k: int, origin=None):
        super().__init__([x, y], origin)
        self.k = k

    def filter(self, doms):
        x, y = self.vars
        xs = {v for v in doms[x] if (v - self.k in doms[y]) or (v + self.k in doms[y])}
        ys = {v for v in doms[y] if (v - self.k in doms[x]) or (v + self.k in doms[x])}
        if not xs or not ys:
            raise Fail(self, f"|{x} - {y}| = {self.k} has no support in "
                             f"current domains")
        out = [(x, v, f"no value of {y} at distance {self.k} from {v}")
               for v in list(doms[x]) if v not in xs]
        out += [(y, v, f"no value of {x} at distance {self.k} from {v}")
                for v in list(doms[y]) if v not in ys]
        return out

    def check(self, asgn):
        return abs(asgn[self.vars[0]] - asgn[self.vars[1]]) == self.k

    def negate(self):
        return NotAbsDiff(self.vars[0], self.vars[1], self.k, self.origin)

    def describe(self):
        return f"|{self.vars[0]} - {self.vars[1]}| = {self.k}"


class NotAbsDiff(Constraint):
    """|x - y| != k."""

    def __init__(self, x: str, y: str, k: int, origin=None):
        super().__init__([x, y], origin)
        self.k = k

    def filter(self, doms):
        x, y = self.vars
        out = []
        for a, b in ((x, y), (y, x)):
            if len(doms[b]) == 1:
                w = next(iter(doms[b]))
                forbidden = {w - self.k, w + self.k}
                if doms[a] <= forbidden:
                    raise Fail(self, f"every value of {a} sits at distance "
                                     f"{self.k} from {b}={w}")
                for v in forbidden:
                    if v in doms[a]:
                        out.append((a, v, f"|{a} - {b}| = {self.k} is forbidden"))
        return out

    def check(self, asgn):
        return abs(asgn[self.vars[0]] - asgn[self.vars[1]]) != self.k

    def negate(self):
        return AbsDiff(self.vars[0], self.vars[1], self.k, self.origin)

    def describe(self):
        return f"|{self.vars[0]} - {self.vars[1]}| != {self.k}"


# ---------------------------------------------------------------- global
class AllDifferent(Constraint):
    """All vars pairwise different. Singleton pruning + pigeonhole check."""

    def filter(self, doms):
        out = []
        union = set()
        for v in self.vars:
            union |= doms[v]
        if len(union) < len(self.vars):
            raise Fail(self, f"{len(self.vars)} variables share only "
                             f"{len(union)} possible values")
        for v in self.vars:
            if len(doms[v]) == 1:
                val = next(iter(doms[v]))
                for w in self.vars:
                    if w != v and val in doms[w]:
                        out.append((w, val, f"{v} already takes {val!r} and "
                                            f"all of {self._short()} differ"))
        return out

    def check(self, asgn):
        vals = [asgn[v] for v in self.vars]
        return len(vals) == len(set(vals))

    def _short(self):
        return "{" + ", ".join(self.vars[:3]) + (", …}" if len(self.vars) > 3 else "}")

    def describe(self):
        return f"AllDifferent{self._short()}"


class Table(Constraint):
    """Extensional relation: allowed tuples over vars. Universal fallback —
    any finite relation compiles to this. GAC via support scan."""

    def __init__(self, vars: List[str], allowed: List[Tuple[Value, ...]],
                 origin=None):
        super().__init__(vars, origin)
        self.allowed = [tuple(t) for t in allowed]

    def _live(self, doms):
        return [t for t in self.allowed
                if all(t[i] in doms[v] for i, v in enumerate(self.vars))]

    def filter(self, doms):
        live = self._live(doms)
        if not live:
            raise Fail(self, "no allowed tuple is compatible with current "
                             "domains")
        out = []
        for i, v in enumerate(self.vars):
            support = {t[i] for t in live}
            for val in list(doms[v]):
                if val not in support:
                    out.append((v, val, f"{v}={val!r} appears in no allowed "
                                        f"combination"))
        return out

    def check(self, asgn):
        return tuple(asgn[v] for v in self.vars) in set(self.allowed)

    def status(self, doms):
        live = self._live(doms)
        if not live:
            return -1
        total = 1
        for v in self.vars:
            total *= len(doms[v])
        return 1 if len(live) == total else 0

    def describe(self):
        return f"Table({', '.join(self.vars)}; {len(self.allowed)} tuples)"


class Count(Constraint):
    """#(vars taking `value`) op k, op in {'==','<=','>='}."""

    def __init__(self, vars: List[str], value: Value, op: str, k: int, origin=None):
        super().__init__(vars, origin)
        self.value, self.op, self.k = value, op, k

    def _bounds(self, doms):
        lo = sum(1 for v in self.vars if doms[v] == {self.value})
        hi = sum(1 for v in self.vars if self.value in doms[v])
        return lo, hi

    def filter(self, doms):
        lo, hi = self._bounds(doms)
        out = []
        if self.op in ("==", ">=") and hi < self.k:
            raise Fail(self, f"at most {hi} of {len(self.vars)} vars can be "
                             f"{self.value!r}, need {self.op} {self.k}")
        if self.op in ("==", "<=") and lo > self.k:
            raise Fail(self, f"already {lo} vars forced to {self.value!r}, "
                             f"allowed {self.op} {self.k}")
        if self.op in ("==", ">=") and hi == self.k:
            for v in self.vars:                      # every candidate is needed
                if self.value in doms[v] and len(doms[v]) > 1:
                    for val in list(doms[v]):
                        if val != self.value:
                            out.append((v, val, f"{v} is needed as "
                                                f"{self.value!r} to reach the "
                                                f"count of {self.k}"))
        if self.op in ("==", "<=") and lo == self.k:
            for v in self.vars:                      # quota full
                if self.value in doms[v] and doms[v] != {self.value}:
                    out.append((v, self.value, f"the quota of {self.k} "
                                               f"{self.value!r} is already met"))
        return out

    def check(self, asgn):
        n = sum(1 for v in self.vars if asgn[v] == self.value)
        return {"==": n == self.k, "<=": n <= self.k, ">=": n >= self.k}[self.op]

    def describe(self):
        return f"Count({self.value!r} over {len(self.vars)} vars) {self.op} {self.k}"


# ---------------------------------------------------------------- combinators
class Not(Constraint):
    def __init__(self, c: Constraint, origin=None):
        self.child = c.negate()
        super().__init__(self.child.vars, origin or c.origin)

    def filter(self, doms):
        return self.child.filter(doms)

    def check(self, asgn):
        return self.child.check(asgn)

    def status(self, doms):
        return self.child.status(doms)

    def negate(self):
        return self.child.negate()

    def describe(self):
        return self.child.describe()


class Or(Constraint):
    """At least one clause holds. Propagates the last live clause; fails when
    all clauses are falsified by the current domains."""

    def __init__(self, clauses: List[Constraint], origin=None):
        vars: List[str] = []
        for c in clauses:
            for v in c.vars:
                if v not in vars:
                    vars.append(v)
        super().__init__(vars, origin)
        self.clauses = clauses

    def filter(self, doms):
        alive = [c for c in self.clauses if c.status(doms) >= 0]
        if not alive:
            raise Fail(self, "every alternative is already impossible: " +
                             " | ".join(c.describe() for c in self.clauses))
        if any(c.status(doms) == 1 for c in alive):
            return []
        if len(alive) == 1:
            return alive[0].filter(doms)
        return []

    def check(self, asgn):
        return any(c.check(asgn) for c in self.clauses)

    def status(self, doms):
        st = [c.status(doms) for c in self.clauses]
        if any(s == 1 for s in st):
            return 1
        if all(s == -1 for s in st):
            return -1
        return 0

    def negate(self):
        return AndAll([c.negate() for c in self.clauses], self.origin)

    def describe(self):
        return " OR ".join(c.describe() for c in self.clauses)


class AndAll(Constraint):
    """Every clause holds (useful mainly as the negation of Or)."""

    def __init__(self, clauses: List[Constraint], origin=None):
        vars: List[str] = []
        for c in clauses:
            for v in c.vars:
                if v not in vars:
                    vars.append(v)
        super().__init__(vars, origin)
        self.clauses = clauses

    def filter(self, doms):
        out = []
        for c in self.clauses:
            out.extend(c.filter(doms))
        return out

    def check(self, asgn):
        return all(c.check(asgn) for c in self.clauses)

    def status(self, doms):
        st = [c.status(doms) for c in self.clauses]
        if any(s == -1 for s in st):
            return -1
        if all(s == 1 for s in st):
            return 1
        return 0

    def negate(self):
        return Or([c.negate() for c in self.clauses], self.origin)

    def describe(self):
        return " AND ".join(c.describe() for c in self.clauses)


class Implies(Constraint):
    """if A then B  ==  (not A) or B."""

    def __init__(self, a: Constraint, b: Constraint, origin=None):
        self.a, self.b = a, b
        self._or = Or([a.negate(), b], origin)
        super().__init__(self._or.vars, origin)

    def filter(self, doms):
        return self._or.filter(doms)

    def check(self, asgn):
        return (not self.a.check(asgn)) or self.b.check(asgn)

    def status(self, doms):
        return self._or.status(doms)

    def negate(self):
        return AndAll([self.a, self.b.negate()], self.origin)

    def describe(self):
        return f"({self.a.describe()}) -> ({self.b.describe()})"


class Iff(Constraint):
    def __init__(self, a: Constraint, b: Constraint, origin=None):
        self.a, self.b = a, b
        self._and = AndAll([Implies(a, b), Implies(b, a)], origin)
        super().__init__(self._and.vars, origin)

    def filter(self, doms):
        return self._and.filter(doms)

    def check(self, asgn):
        return self.a.check(asgn) == self.b.check(asgn)

    def status(self, doms):
        return self._and.status(doms)

    def negate(self):
        return Iff(self.a, self.b.negate(), self.origin)

    def describe(self):
        return f"({self.a.describe()}) <-> ({self.b.describe()})"


class OffsetLeq(Constraint):
    """x <= y + k (numeric domains) — bounds consistency.
    Expresses gaps: 'at least g positions between x and y' is
    OffsetLeq(x, y, -g-?) composed in an Or; 'x at least 2 before y' is
    OffsetLeq(x, y, -2) i.e. x <= y - 2."""

    def __init__(self, x: str, y: str, k: int, origin=None):
        super().__init__([x, y], origin)
        self.k = k

    def filter(self, doms):
        x, y = self.vars
        ymax = max(doms[y]) + self.k
        xmin = min(doms[x]) - self.k
        if min(doms[x]) > ymax:
            raise Fail(self, f"{x} <= {y} + {self.k} impossible: "
                             f"min({x})={min(doms[x])} > max({y})+{self.k}={ymax}")
        out = [(x, v, f"{x}={v} exceeds max({y})+{self.k}={ymax}")
               for v in list(doms[x]) if v > ymax]
        out += [(y, v, f"{y}={v} is below min({x})-{self.k}={xmin}")
                for v in list(doms[y]) if v < xmin]
        return out

    def check(self, asgn):
        return asgn[self.vars[0]] <= asgn[self.vars[1]] + self.k

    def status(self, doms):
        x, y = self.vars
        if max(doms[x]) <= min(doms[y]) + self.k:
            return 1
        if min(doms[x]) > max(doms[y]) + self.k:
            return -1
        return 0

    def negate(self):                    # not(x <= y+k)  ==  x >= y+k+1
        return OffsetGeq(self.vars[0], self.vars[1], self.k + 1, self.origin)

    def describe(self):
        return f"{self.vars[0]} <= {self.vars[1]} + {self.k}"


class OffsetGeq(Constraint):
    """x >= y + k (numeric domains)."""

    def __init__(self, x: str, y: str, k: int, origin=None):
        super().__init__([x, y], origin)
        self.k = k

    def filter(self, doms):
        x, y = self.vars
        xmax = max(doms[x]) - self.k
        ymin = min(doms[y]) + self.k
        if max(doms[x]) < min(doms[y]) + self.k:
            raise Fail(self, f"{x} >= {y} + {self.k} impossible")
        out = [(x, v, f"{x}={v} is below min({y})+{self.k}={ymin}")
               for v in list(doms[x]) if v < ymin]
        out += [(y, v, f"{y}={v} exceeds max({x})-{self.k}={xmax}")
                for v in list(doms[y]) if v > xmax]
        return out

    def check(self, asgn):
        return asgn[self.vars[0]] >= asgn[self.vars[1]] + self.k

    def status(self, doms):
        x, y = self.vars
        if min(doms[x]) >= max(doms[y]) + self.k:
            return 1
        if max(doms[x]) < min(doms[y]) + self.k:
            return -1
        return 0

    def negate(self):
        return OffsetLeq(self.vars[0], self.vars[1], self.k - 1, self.origin)

    def describe(self):
        return f"{self.vars[0]} >= {self.vars[1]} + {self.k}"
