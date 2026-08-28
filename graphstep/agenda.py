#!/usr/bin/env python3
"""The solver loop: a general agenda of goals worked one at a time until a
certified result or a named refusal — the knowledge escalation ladder run
per-part instead of per-task.

    AGENDA = [ the task as one root goal ]
    while open goals remain (within budget):
        pick the most constrained open goal
        GROUND     known rows / the one-shot pipeline answer it
        DECOMPOSE  split it by structure into subgoals
        RETRIEVE   ask an external source, admit typed rows
        DERIVE     bounded, spec-guided composition of ATOMS for this
                   subgoal only; every candidate faces the oracle
        VERIFY     the domain's oracle judges (tests / proofs / closure)
        REPAIR     a failure names the guilty subgoal; reopen exactly it
        LEARN      a verified composition becomes a new row (session scope;
                   persistence must be earned: verified + reusable + stamped)
        REFUSE     nothing applies -> record the named gap and close

Every iteration proves a goal, adds knowledge, or closes a goal honestly —
the loop always terminates. Nothing here knows any domain: GROUND is the
generic six-step pipeline, DERIVE composes whatever atom rows exist, and
the oracles come from the artifact type. The whole loop is bookkeeping
(Goal, Blackboard) around machinery that already exists.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import unified
from .reading import kb
from .engine.sandbox import run_tests


@dataclass
class Goal:
    desc: str
    status: str = "OPEN"          # OPEN | PROVEN | REFUSED
    artifact: object = None
    reason: Optional[str] = None


@dataclass
class Blackboard:
    sample: dict
    question: Optional[str]
    evidence: object = None
    sig: object = None
    frame: object = None
    trace: List[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.trace.append(msg)


# ------------------------------------------------------------------ DERIVE

def _pred_fits(pred_arg: str, arg_type: str) -> bool:
    return (pred_arg == "ANY"
            or (pred_arg == "SIZED" and arg_type in ("LIST", "STRING"))
            or (pred_arg == "NUM" and arg_type in ("INT", "FLOAT")))


def _derive_program(bb: Blackboard) -> Optional[dict]:
    """Spec-licensed atom composition: only atoms whose WORD appears in the
    spec may enter a candidate; the operation family picks the composition
    schema; the sandbox oracle judges every candidate. Wrong extraction can
    only ever produce a refusal, never an unverified answer."""
    import re
    sig, frame = bb.sig, bb.frame
    if sig is None or frame is None:
        return None
    words = set(re.findall(
        r"[a-z]+", " ".join(t for t, _ in bb.evidence.statements).lower()))
    if frame.theme_token is not None:       # lemmas too: rows are lemma-keyed
        words |= {t.lemma_.lower() for t in frame.theme_token.doc}
    # HEAD-LICENSING: an atom may only fill the frame's head position —
    # the theme noun or its modifiers. "find the SMALLEST number" licenses
    # min; "find the frequency of the smallest value" does NOT (the head
    # is 'frequency'). Prevents passing weak tests by coincidence.
    head_words = set(frame.item_mods) | ({frame.item} if frame.item else set())
    preds = [r for r in kb.KB if r.symbol == "PRED" and r.payload
             and isinstance(r.pattern, str)
             and (r.pattern in head_words
                  or (frame.op == "CHECK" and r.pattern in words))]
    reduces = [r for r in kb.KB if r.symbol == "REDUCE"
               and isinstance(r.pattern, str) and r.pattern in head_words]

    # ARGUMENT COVERAGE: a candidate must reference every argument whose
    # value varies across the examples — a program that ignores a varying
    # input cannot have read the task, however the tests fall.
    varying = [i for i in range(sig.arity)
               if len({repr(ex[0][i]) for ex in sig.examples}) > 1]

    def _covers(code: str) -> bool:
        return all(f"arg{i}" in code.split(":", 1)[1] for i in varying)
    params = [f"arg{i}" for i in range(sig.arity)]
    seq_args = [i for i, t in enumerate(sig.arg_types)
                if t in ("LIST", "STRING")]

    def mk(body: str) -> str:
        return f"def {sig.name}({', '.join(params)}):\n    return {body}"

    atom = {k: kb.match_key(("ATOM", k)) for k in ("FILTER", "COUNT", "SORT")}
    cands: List[tuple] = []
    op = frame.op

    # COMPOSITION-COMPILED candidates first: the parse tree dictates the
    # atom tree ("sum of digits of the number" -> sum(digits(arg))), so the
    # license holds at every depth by construction.
    if op in ("FIND", "CHECK") and frame.theme_token is not None:
        from .reading.compose import (compile_tree, emit_code, free_slots,
                                      default_leaf, Node)
        import itertools

        def _leaf_arity_pref(tok):
            # always ALSO try the 2-ary reading of the head word: with two
            # scalar args it becomes gcd(a, b); with one sequence arg the
            # (BINOP, SEQ) coercion folds it across the sequence
            lemma = tok.lemma_.lower()
            for r2 in kb.KB:
                if (r2.symbol == "BINOP" and r2.pattern == lemma
                        and r2.payload):
                    return Node("atom", row=r2, word=lemma)
            return default_leaf(tok)

        trees = []
        for lf in (_leaf_arity_pref, default_leaf):
            t = compile_tree(frame.theme_token, leaf=lf)
            if t is not None and t.kind == "atom" and \
                    not any(t.atoms() == u.atoms() and t.row is u.row
                            for u in trees):
                trees.append(t)
        for tree in trees:
            slots = free_slots(tree)
            compat_of = {"ANY": ("INT", "FLOAT", "LIST", "STRING", "CHAR",
                                 "BOOL"),
                         "NUM": ("INT", "FLOAT"), "LIST": ("LIST",),
                         "STRING": ("STRING", "CHAR"),
                         "SIZED": ("LIST", "STRING")}
            options = []
            for key, hint in slots:
                opts = [params[i] for i, t in enumerate(sig.arg_types)
                        if t in compat_of.get(hint, ())]
                if isinstance(key, tuple) and key[0] == "bfree":
                    opts = opts + ["__FOLD__"]
                options.append((key, opts))
            combos = itertools.product(*[c for _, c in options]) \
                if all(c for _, c in options) else []
            label_atoms = "*".join(tree.atoms())
            for combo in itertools.islice(combos, 12):
                binding = {k: v for (k, _), v in zip(options, combo)}
                expr = emit_code(tree, binding)
                if expr:
                    body = f"bool({expr})" if op == "CHECK" else expr
                    cands.append((f"compose[{label_atoms}]", mk(body)))
            if not slots:
                expr = emit_code(tree, {})
                if expr:
                    body = f"bool({expr})" if op == "CHECK" else expr
                    cands.append((f"compose[{label_atoms}]", mk(body)))
            # MAP variant: the whole inner chain evaluated per element of a
            # sequence argument ((COERCE, APPLY, SEQ) — "maximum sum of the
            # sublists" -> max([sum(_e) for _e in arg]))
            if slots and tree.row.symbol == "REDUCE" and tree.children:
                for si in seq_args:
                    mbind = {key: "_e" for key, _ in slots}
                    mbind[("map", id(tree))] = params[si]
                    expr = emit_code(tree, mbind)
                    if expr:
                        body = f"bool({expr})" if op == "CHECK" else expr
                        cands.append((f"compose-map[{label_atoms}]@arg{si}",
                                      mk(body)))
    if op == "REMOVE" and atom["FILTER"]:
        for p in preds:
            for si in seq_args:
                keep = f"not ({p.payload.format(x='x')})"
                cands.append((f"filter(not {p.sig['name']})@arg{si}",
                              mk(atom["FILTER"].payload.format(
                                  src=params[si], keep=keep))))
    if op == "COUNT" and atom["COUNT"]:
        for p in preds:
            for si in seq_args:
                cands.append((f"count({p.sig['name']})@arg{si}",
                              mk(atom["COUNT"].payload.format(
                                  src=params[si],
                                  keep=p.payload.format(x="x")))))
    if op in ("CHECK", "FIND"):
        for p in preds:
            for ai, t in enumerate(sig.arg_types):
                if _pred_fits(p.sig.get("arg", "ANY"), t):
                    cands.append((f"pred {p.sig['name']}@arg{ai}",
                                  mk(f"bool({p.payload.format(x=params[ai])})")))
    if op == "FIND":
        for r in reduces:
            for si in seq_args:
                cands.append((f"reduce {r.pattern}@arg{si}",
                              mk(r.payload.format(src=params[si]))))
    if op == "SORT" and atom["SORT"]:
        plain = [(f"sorted@arg{si}",
                  mk(atom["SORT"].payload.format(src=params[si])))
                 for si in seq_args]
        if frame.theme_token is not None:
            from .reading.compose import (compile_tree as _ct,
                                          emit_code as _ec,
                                          free_slots as _fs)
            for t in frame.theme_token.doc:
                if t.dep_ != "prep":
                    continue
                drow = kb.match_key(("DEVICE", t.lemma_.lower()))
                if not (drow and drow.payload == "keyparam"):
                    continue
                pobjs = [g for g in t.subtree
                         if g.dep_ == "pobj" and g.head.lemma_.lower()
                         in (t.lemma_.lower(), "to")]
                for pobj in pobjs:
                    ktree = _ct(pobj)
                    if ktree is None or ktree.kind != "atom":
                        continue
                    kbind = {k: "_x" for k, _ in _fs(ktree)}
                    kexpr = _ec(ktree, kbind)
                    if kexpr:
                        for si in seq_args:
                            cands.append((
                                f"sorted-by[{'*'.join(ktree.atoms())}]@arg{si}",
                                mk(f"sorted({params[si]}, "
                                   f"key=lambda _x: {kexpr})")))
        cands.extend(plain)
    if op == "XFORM" and frame.verb:
        vrow = kb.match_word(frame.verb, "OP:", widen=False)
        if vrow and vrow.payload:
            for si in seq_args:
                cands.append((f"xform {frame.verb}@arg{si}",
                              mk(vrow.payload.format(src=params[si]))))
    if op == "CHECK" and sig.arity >= 2:
        bpreds = [r for r in kb.KB if r.symbol == "BPRED"
                  and isinstance(r.pattern, str) and r.pattern in words]
        for b in bpreds:
            for i in range(sig.arity):
                for j in range(sig.arity):
                    if i != j:
                        cands.append((
                            f"bpred {b.pattern}(arg{i},arg{j})",
                            mk(f"bool({b.payload.format(a=params[i], b=params[j])})")))

    seen, passes, tried = set(), [], 0
    for label, code in cands:
        if code in seen:
            continue
        seen.add(code)
        if not _covers(code):
            bb.log(f"DERIVE: {label} rejected (ignores a varying argument)")
            continue
        if tried >= 32:
            break
        tried += 1
        res = run_tests(code, bb.evidence.assert_texts)
        bb.log(f"DERIVE: {label} -> "
               f"{'PASS' if res['ok'] else (res['error'] or 'fail')[:60]}")
        if res["ok"]:
            passes.append((label, code))
    if passes:
        label, code = passes[0]
        import re as _re
        if label.startswith("compose") and "[" in label:
            word = label.split("[", 1)[1].split("]")[0].split("*")[0]
        else:
            word = None
        m = _re.search(r"\((?:not )?IS_\w+\)|reduce (\w+)|pred (\w+)"
                       r"|bpred (\w+)", label)
        word = word or next((g for g in (m.groups() if m else ()) if g), None)
        # map pred NAME back to its row word when needed
        if word is None and "IS_" in label:
            nm = label.split("IS_")[1].split(")")[0]
            word = next((r.pattern for r in kb.KB if r.symbol == "PRED"
                         and r.sig.get("name") == "IS_" + nm), None)
        return {"code": code, "atoms": [label], "tried": tried,
                "variants": len(passes), "atom_word": word}
    return None


def _learn(bb: Blackboard, derived: dict) -> None:
    if bb.frame is None or str(derived.get("grade", "")).startswith(
            "verified"):
        return          # procedures are task-shaped: never stored
    """A verified derivation becomes a session-scope row so the next task
    of the same shape takes the fast path. Persistence must be earned —
    this writes to the in-memory store only, provenance-stamped."""
    key = ("DERIVED", bb.frame.op, derived["atoms"][0],
           tuple(bb.sig.arg_types))
    if kb.match_key(key) is None:
        template = derived["code"].replace(f"def {bb.sig.name}(",
                                           "def {name}(")
        kb.add(kb.Row(key, "BLOCK",
                      sig={"atoms": derived["atoms"],
                           "atom_word": derived.get("atom_word")},
                      payload=template, provenance="derived:agenda",
                      confidence=0.9))
        bb.log(f"LEARN: cached derived block {key} (session scope)")


def _ground_learned(bb: Blackboard) -> Optional[dict]:
    """Fast path: a previously derived-and-verified block for this exact
    shape already exists as a row."""
    if bb.sig is None or bb.frame is None:
        return None
    argt = tuple(bb.sig.arg_types)
    import re as _re
    head_words = set(bb.frame.item_mods) | \
        ({bb.frame.item} if bb.frame.item else set())
    spec_words = set(_re.findall(
        r"[a-z]+", " ".join(t for t, _ in bb.evidence.statements).lower()))
    for row in kb.KB:
        if (isinstance(row.pattern, tuple) and len(row.pattern) == 4
                and row.pattern[0] == "DERIVED"
                and row.pattern[1] == bb.frame.op and row.pattern[3] == argt):
            w = row.sig.get("atom_word")
            if w is not None and w not in head_words and not (
                    bb.frame.op == "CHECK" and w in spec_words):
                continue                     # not licensed for THIS spec
            code = row.payload.replace("{name}", bb.sig.name)
            res = run_tests(code, bb.evidence.assert_texts)
            if res["ok"]:
                bb.log(f"GROUND: learned row {row.pattern} answers directly "
                       f"(provenance {row.provenance})")
                return {"code": code, "atoms": row.sig.get("atoms", [])}
    return None


def _employ_procedure(bb: Blackboard) -> Optional[dict]:
    """Retrieve worked examples for the whole spec; a candidate is employed
    only if it passes the task's oracle after aliasing its entry point to
    the required name. Grade: verified (not certified — the program is
    example-sourced, not spec-derived)."""
    import ast as _ast
    from .reading.sources import Gap, retrieve
    spec = " ".join(t for t, _ in bb.evidence.statements)
    cands, _rej = retrieve(Gap(word="", kind="procedure", context=spec))
    for row in cands:
        code = row.payload
        try:
            names = [n.name for n in _ast.parse(code).body
                     if isinstance(n, _ast.FunctionDef)]
        except SyntaxError:
            continue
        variants = ([code] if bb.sig.name in names else []) + \
            [code + f"\n{bb.sig.name} = {nm}" for nm in names
             if nm != bb.sig.name]
        for v in variants:
            res = run_tests(v, bb.evidence.assert_texts)
            if res["ok"]:
                bb.log(f"PROCEDURE: {row.provenance} "
                       f"(match {row.sig['match']}) -> PASS")
                return {"code": v,
                        "atoms": [f"procedure({row.provenance})"],
                        "grade": "verified (example-sourced, "
                                 "oracle-passed)"}
        bb.log(f"PROCEDURE: {row.provenance} "
               f"(match {row.sig['match']}) -> fails oracle")
    return None


# ------------------------------------------------------------------ loop

def solve_loop(sample: dict, question: Optional[str] = None,
               budget: int = 12) -> dict:
    """The general solver loop. GROUND first (the one-shot pipeline — one
    iteration when knowledge suffices); on a program-shaped refusal,
    DECOMPOSE into read-spec / derive / verify subgoals and work them."""
    bb = Blackboard(sample=sample, question=question)
    agenda: List[Goal] = [Goal("solve the task")]
    rec: Dict = {}

    for step in range(budget):
        open_goals = [g for g in agenda if g.status == "OPEN"]
        if not open_goals:
            break
        goal = open_goals[0]

        if goal.desc == "solve the task":
            rec = unified.solve(sample, question=question)
            bb.log(f"GROUND: one-shot pipeline -> {rec['status']}")
            if rec["status"] in ("SOLVED", "CONCLUSIONS", "CAPTURED",
                                 "EXCLUDED", "AMBIGUOUS"):
                goal.status = "PROVEN" if rec["status"] == "SOLVED" \
                    else "REFUSED" if rec["status"] == "EXCLUDED" \
                    else "PROVEN"
                break
            # program-shaped? gather sig+frame evidence for decomposition
            bb.evidence = unified.recognize(sample)
            if question:
                bb.evidence.question = question
            if bb.evidence.assert_texts:
                try:
                    bb.sig = unified.parse_signature(bb.evidence.assert_texts)
                except (ValueError, SyntaxError) as e:
                    goal.status, goal.reason = "REFUSED", f"signature: {e}"
                    break
                for st, _ in bb.evidence.statements:
                    try:
                        bb.frame = unified.parse_frame(st)
                        break
                    except ValueError as e:
                        goal.reason = f"frame: {e}"
                if bb.frame is None:
                    # the frame is unread, but the PROCEDURE route needs
                    # only the signature and the spec text — derivation
                    # still gets its goal (fixes: retrieval was wrongly
                    # gated behind frame success)
                    bb.log(f"DECOMPOSE: frame unread ({goal.reason}); "
                           f"procedure route remains available")
                else:
                    bb.log(f"DECOMPOSE: spec read (op={bb.frame.op}, "
                           f"mods={bb.frame.item_mods}) -> "
                           f"subgoal: derive a program")
                goal.status = "PROVEN"          # the reading subpart
                agenda.append(Goal("derive a certified program"))
            else:
                goal.status = "REFUSED"
                goal.reason = rec.get("reasons", ["no route"])[0] \
                    if rec.get("reasons") else "no route"
                break

        elif goal.desc == "derive a certified program":
            derived = _ground_learned(bb) or _derive_program(bb)
            if derived is None and bb.frame is not None:
                # RETRIEVE: the refusal names the gap — ask the sources,
                # admit through the tribunal, and retry the derivation
                from .reading.sources import Gap, retrieve
                from .reading.compose import default_leaf
                theme_word = None
                ungrounded = []
                if bb.frame.theme_token is not None:
                    theme_word = bb.frame.theme_token.lemma_.lower()
                    # every ungrounded content word of the spec is a gap
                    # candidate (parse-derived, domain-free)
                    ungrounded = [t.lemma_.lower()
                                  for t in bb.frame.theme_token.doc
                                  if t.pos_ in ("NOUN", "ADJ", "VERB")
                                  and default_leaf(t) is None][:6]
                for word in dict.fromkeys(
                        [theme_word, bb.frame.item, bb.frame.verb]
                        + list(bb.frame.item_mods) + ungrounded):
                    if not word:
                        continue
                    admitted, rejected = retrieve(Gap(
                        word=word, kind="callable",
                        arity=bb.sig.arity if bb.sig else None))
                    for r in rejected:
                        bb.log(f"RETRIEVE: rejected {r}")
                    if admitted:
                        bb.log(f"RETRIEVE: admitted "
                               f"{[r.provenance for r in admitted]}")
                if any("RETRIEVE: admitted" in l for l in bb.trace):
                    derived = _derive_program(bb)
                    if derived:
                        derived["retrieved"] = [
                            l for l in bb.trace if "admitted" in l]
            if derived is None and bb.sig is not None:
                # PROCEDURE kind: worked examples from any registered
                # corpus, employed strictly through the task's oracle,
                # graded 'verified', never stored (task-shaped). Needs
                # only the signature and spec text — NOT the frame.
                derived = _employ_procedure(bb)
            if derived:
                goal.status = "PROVEN"
                goal.artifact = derived
                _learn(bb, derived)
                rec.update(status="SOLVED", code=derived["code"],
                           reasons=[],
                           grade=derived.get("grade", "certified"),
                           derivation={"atoms": derived["atoms"]})
            else:
                goal.status = "REFUSED"
                goal.reason = (
                    "no atom composition passes the oracle "
                    + (f"(op={bb.frame.op}, mods={bb.frame.item_mods})"
                       if bb.frame is not None else "(frame unread)"))
                bb.log(f"REFUSE: {goal.reason}")
                rec.setdefault("reasons", []).append(goal.reason)

    rec["loop"] = {"iterations": step + 1, "trace": bb.trace,
                   "goals": [(g.desc, g.status) for g in agenda]}
    return rec
