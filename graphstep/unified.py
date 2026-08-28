#!/usr/bin/env python3
"""ONE algorithm for any sample from any source. No benchmark names, no
per-task branches, no manifests — a sample is a bag of text and structures,
and every sample runs the identical six steps:

  1 RECOGNIZE  discover shapes in the bag: assertion statements -> typed I/O
               examples; bulleted value grids / keyed value lists -> value
               groups; numbered lines / sentence lists -> statements; short
               name lists -> a roster; program-shaped strings -> QUARANTINED
               (a bag may carry a reference answer; it is never consumed).
  2 PARSE      statements -> syntactic structure (regex rows first, full
               dependency parse where a semantic frame is needed).
  3 GROUND     every element -> rows of the ONE knowledge store (kb.py);
               WordNet widens on a miss, with provenance. Ungrounded
               elements are reported, never guessed.
  4 EMIT       everything grounded becomes ONE constraint problem. Value
               groups license position variables (+ alldiff); grounded
               statements license constraint nodes; typed I/O examples +
               a grounded frame license PROGRAM-SLOT variables whose
               domains are the licensed symbols, plus an EvalCheck
               constraint that runs the examples — program synthesis IS
               constraint solving, on the same engine.
  5 SOLVE      core.Engine: propagation + search + uniqueness certification.
  6 ANSWER     SOLVED with certificate and fired-row provenance, or a named
               refusal (EXCLUDED / UNGROUNDED / UNSAT / AMBIGUOUS / UNKNOWN).

The only dispatch anywhere is on evidence shape and artifact type — never on
where the sample came from.
"""
from __future__ import annotations
import ast, re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .engine.core import Constraint, Engine, Problem
from .engine.ir import build_constraint
from .reading.compile_text import Inventory
from .reading import kb


# ================================================================ evidence

@dataclass
class Evidence:
    statements: List[Tuple[str, str]] = field(default_factory=list)
    # (text, kind): kind "declared" = an enumerated premise (numbered line,
    # list item) that MUST ground; kind "prose" = a free-running sentence —
    # it must ground if it mentions anything the model knows, and is merely
    # inert (recorded, non-blocking) if it mentions nothing.
    groups: Dict[str, List[str]] = field(default_factory=dict)
    roster: List[str] = field(default_factory=list)
    options: List[Tuple[str, str]] = field(default_factory=list)  # (label, text)
    question: Optional[str] = None
    assert_texts: List[str] = field(default_factory=list)
    setup: List[str] = field(default_factory=list)
    heldout: List[str] = field(default_factory=list)   # quarantined answers
    context: List[str] = field(default_factory=list)


def _is_assert(s: str) -> bool:
    try:
        mod = ast.parse(s.strip())
        return len(mod.body) == 1 and isinstance(mod.body[0], ast.Assert)
    except SyntaxError:
        return False


def _is_program(s: str) -> bool:
    try:
        mod = ast.parse(s)
        return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   for n in ast.walk(mod))
    except SyntaxError:
        return False


_NAME_RE = re.compile(r"^[A-Z][A-Za-z0-9_'\-]{0,19}$")


def _is_name(s: str) -> bool:
    return bool(_NAME_RE.match(s.strip()))


def _is_sentence(s: str) -> bool:
    toks = s.split()
    has_word = any(re.search(r"[A-Za-z]{2}", t) for t in toks)
    # ≥3 tokens, or a short clause with terminal punctuation ("Jana reads.")
    return has_word and (len(toks) >= 3
                         or (len(toks) >= 2 and s.rstrip()[-1:] in ".!?"))


_GROUP_LINE = re.compile(r"`([^`]+)`")
_OPTION_LINE = re.compile(r"^\(([A-Z])\)\s*(.+)$")
_LABEL_PREFIX = re.compile(r"^\s*[A-Za-z]+:\s+")


def _add_sentences(text: str, ev: Evidence, kind: str) -> None:
    """Split running prose into sentences; '?'-sentences become the query."""
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        sent = sent.strip()
        if not sent:
            continue
        if sent.endswith("?") and _is_sentence(sent):
            ev.question = sent
        elif _is_sentence(sent):
            ev.statements.append((sent, kind))
        else:
            ev.context.append(sent)


def _classify_text(s: str, ev: Evidence, kind: str = "prose") -> None:
    if _is_assert(s):
        ev.assert_texts.append(s.strip()); return
    if _is_program(s):
        ev.heldout.append(s); return
    s = _LABEL_PREFIX.sub("", s, count=1)   # "Question: ..." -> "..."
    if "\n" in s:
        gk = 0
        for line in s.split("\n"):
            line = line.strip()
            if not line:
                continue
            vals = _GROUP_LINE.findall(line)
            opt = _OPTION_LINE.match(line)
            if line.startswith(("-", "*")) and len(vals) >= 2:
                gk += 1
                ev.groups[f"G{gk}"] = vals
            elif opt:
                ev.options.append((opt.group(1), opt.group(2).strip()))
            elif line[0].isdigit() and ". " in line:
                ev.statements.append(
                    (". ".join(line.split(". ")[1:]).strip(), "declared"))
            elif line.endswith(":") or len(line.split()) < 2:
                ev.context.append(line)          # header / marker line
            else:
                _add_sentences(line, ev, "prose")
        return
    opt = _OPTION_LINE.match(s.strip())
    if opt:
        ev.options.append((opt.group(1), opt.group(2).strip()))
    elif _is_sentence(s) and kind == "declared":
        ev.statements.append((s.strip(), "declared"))
    elif _is_sentence(s) or s.strip().endswith("?"):
        _add_sentences(s, ev, kind)
    elif s.strip():
        ev.context.append(s.strip())


def recognize(sample: dict) -> Evidence:
    """Walk an arbitrary sample bag; classify every piece by SHAPE."""
    ev = Evidence()

    def _is_table(d: dict) -> bool:
        """A dict carrying a list-of-lists-of-strings is a table — an
        ASSIGNMENT shape (someone's answer, filled or blanked), never
        premises. Quarantined like program-shaped strings."""
        return any(isinstance(v, list) and v
                   and all(isinstance(x, list)
                           and all(isinstance(y, str) for y in x)
                           for x in v)
                   for v in d.values())

    def walk(val, keyed: Optional[str] = None):
        if isinstance(val, str):
            _classify_text(val, ev)
        elif isinstance(val, dict):
            if _is_table(val):
                ev.heldout.append(f"table({', '.join(map(str, val))})")
                return
            for k, v in val.items():
                if (isinstance(v, list) and v
                        and all(isinstance(x, str) for x in v)
                        and not any(_is_sentence(x) or _is_assert(x)
                                    or _OPTION_LINE.match(x.strip())
                                    for x in v)):
                    ev.groups[str(k)] = list(v)      # keyed value list
                else:
                    walk(v, keyed=str(k))
        elif isinstance(val, list):
            if val and all(isinstance(x, str) for x in val):
                if all(_is_assert(x) for x in val):
                    ev.assert_texts.extend(x.strip() for x in val)
                elif all(_is_name(x) for x in val) and not ev.roster:
                    ev.roster = [x.strip() for x in val]
                else:   # mixed list: each item is an enumerated premise
                    for x in val:
                        _classify_text(x, ev, kind="declared")
            else:
                for x in val:
                    walk(x)
        # numbers / None / bools carry no shape

    walk(sample)
    return ev


# ================================================================ signature

@dataclass
class Signature:
    name: str
    arity: int
    examples: List[Tuple[list, object]]
    arg_types: List[str]


def parse_signature(tests: List[str]) -> Signature:
    name, examples = None, []
    for t in tests:
        stmt = ast.parse(t.strip()).body[0]
        test = stmt.test
        if isinstance(test, ast.Compare) and isinstance(test.left, ast.Call):
            call, expected = test.left, ast.literal_eval(test.comparators[0])
        elif isinstance(test, ast.Call):
            call, expected = test, True
        else:
            raise ValueError(f"unreadable assert shape: {t}")
        if not isinstance(call.func, ast.Name):
            raise ValueError(f"non-simple callee: {t}")
        if name and call.func.id != name:
            raise ValueError("asserts call different functions")
        name = call.func.id
        examples.append(([ast.literal_eval(a) for a in call.args], expected))
    arity = len(examples[0][0])
    types = []
    for pos in range(arity):
        vals = [ex[0][pos] for ex in examples]
        if all(isinstance(v, str) and len(v) == 1 for v in vals):
            types.append("CHAR")
        elif all(isinstance(v, str) for v in vals):
            types.append("STRING")
        elif all(isinstance(v, list) for v in vals):
            types.append("LIST")
        elif all(isinstance(v, bool) for v in vals):
            types.append("BOOL")
        elif all(isinstance(v, int) for v in vals):
            types.append("INT")
        elif all(isinstance(v, (int, float)) for v in vals):
            types.append("FLOAT")
        else:
            types.append("?")
    return Signature(name, arity, examples, types)


# ================================================================ frame

@dataclass
class Frame:
    op: str
    op_provenance: str
    theme_kind: str
    selectors: List[str]
    item: Optional[str]
    source: Optional[str]
    item_mods: List[str] = field(default_factory=list)  # "EMPTY lists"


_NLP = None
def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def parse_frame(text: str) -> Frame:
    doc = _nlp()(text)
    sent = next(doc.sents)
    verb = sent.root
    if kb.match_word(verb.lemma_, "DISCOURSE:"):
        inner = [c for c in verb.children if c.pos_ == "VERB"
                 and c.dep_ in ("advcl", "xcomp", "ccomp")]
        if not inner:
            raise ValueError("imperative framing with no inner action verb")
        verb = inner[0]

    op_row = kb.match_word(verb.lemma_, "OP:")
    if op_row is None:
        raise ValueError(f"ungrounded verb: '{verb.lemma_}'")

    themes = [c for c in verb.children if c.dep_ in ("dobj", "obj")]
    if not themes:
        raise ValueError(f"no direct object under '{verb.text}'")
    theme = themes[0]

    selectors: List[str] = []
    item_mods: List[str] = []
    for m in theme.children:
        if m.dep_ == "amod":
            for tok in sorted([m] + list(m.conjuncts), key=lambda t: t.i):
                row = kb.match_word(tok.lemma_, "SEL:")
                if row and row.symbol[4:] not in selectors:
                    selectors.append(row.symbol[4:])
                elif not row:
                    item_mods.append(tok.lemma_.lower())

    def pobj_of(head, preps) -> Optional[str]:
        for p in head.children:
            if p.dep_ == "prep" and p.lemma_.lower() in preps:
                objs = [c for c in p.children if c.dep_ == "pobj"]
                if objs:
                    return objs[0].lemma_.lower()
        return None

    item = pobj_of(theme, {"of"})
    source = None
    for head in (theme, verb) + tuple(theme.subtree):
        source = pobj_of(head, {"from", "in"})
        if source:
            break

    kind = ("OCCURRENCE" if kb.match_word(theme.lemma_, "THEME:")
            else "ITEM")
    if kind == "ITEM" and item is None:
        item = theme.lemma_.lower()
    return Frame(op=op_row.symbol[3:], op_provenance=op_row.provenance,
                 theme_kind=kind, selectors=selectors, item=item,
                 source=source, item_mods=item_mods)


# ================================================================ program CSP

_TYPE_OK = {("STRING", "CHAR"), ("SEQ", "STRING"), ("SEQ", "CHAR"),
            ("SEQ", "LIST"), ("ELEM", "CHAR")}


def _compatible(want: str, have: str) -> bool:
    return want == have or (want, have) in _TYPE_OK


def render_program(sig: Signature, asgn: Dict[str, object]) -> str:
    """Assemble Python source from BLOCK rows under a slot assignment."""
    params = [f"arg{i}" for i in range(sig.arity)]
    src = params[asgn["prog:role:src"]]
    item = params[asgn["prog:role:item"]]
    op = asgn["prog:op"]
    lines = [f"def {sig.name}({', '.join(params)}):", f"    s = {src}"]
    for sel in ("FIRST", "LAST", "ALL"):
        if asgn.get(f"prog:sel:{sel}") == 1:
            row = kb.match_key((op, sel))
            if row is None:
                raise ValueError(f"no BLOCK row for {(op, sel)}")
            frag = row.payload.format(item=item)
            lines += ["    " + ln for ln in frag.split("\n")]
    lines.append("    return s")
    return "\n".join(lines)


class EvalCheck(Constraint):
    """The examples as a constraint node: satisfied iff the program rendered
    from the slot assignment passes every assertion (run sandboxed). No
    propagation — a pure search-leaf oracle, the semantic analogue of Table."""

    def __init__(self, sig: Signature, tests: List[str], setup: str,
                 slots: List[str]):
        super().__init__(slots, origin="typed I/O examples")
        self.sig, self.tests, self.setup = sig, tests, setup

    def filter(self, doms):
        return []

    def check(self, asgn) -> bool:
        from .engine.sandbox import run_tests
        try:
            code = render_program(self.sig, asgn)
        except ValueError:
            return False
        return run_tests(code, self.tests, self.setup)["ok"]

    def describe(self):
        return f"EvalCheck({len(self.tests)} assertions)"


# ================================================================ pipeline

def _norm_value(v: str) -> str:
    return "".join(w.capitalize() for w in re.split(r"[\s\-_']+", v) if w)


def solve(sample: dict, question: Optional[str] = None) -> dict:
    """The one algorithm. Returns a full record: status, answer, coverage,
    fired rows, refusal reasons. `question` marks the query explicitly for
    presentations where it is a separate field of declarative shape
    (indistinguishable from a premise by shape alone)."""
    rec: Dict = {"status": "FAILED", "reasons": [], "fired": [],
                 "coverage": None, "answer": None}
    ev = recognize(sample)
    if question is not None:
        ev.question = question.strip()
    rec["evidence"] = {
        "statements": len(ev.statements), "groups": len(ev.groups),
        "roster": len(ev.roster), "assertions": len(ev.assert_texts),
        "quarantined": len(ev.heldout)}

    variables: Dict[str, list] = {}
    specs: List[dict] = []
    extra_cons: List[Constraint] = []
    fired: set = set()

    # ---- value groups license position variables --------------------------
    inv = None
    if ev.groups:
        widths = {len(vs) for vs in ev.groups.values()}
        if len(widths) != 1:
            rec.update(status="EXCLUDED",
                       reasons=[f"ragged value groups (widths {sorted(widths)})"])
            return rec
        n = widths.pop()
        attrs = {g: [_norm_value(v) for v in vs] for g, vs in ev.groups.items()}
        flat = [v for vs in attrs.values() for v in vs]
        if len(flat) != len(set(flat)):
            rec.update(status="EXCLUDED",
                       reasons=["duplicate value names across groups"])
            return rec
        aliases: Dict[str, List[str]] = {}
        raw_name: Dict[str, str] = {}       # canonical -> the text's own word
        for g, vs in ev.groups.items():
            for raw, nv in zip(vs, attrs[g]):
                aliases.setdefault(nv, []).append(raw)
                raw_name.setdefault(nv, raw)
        entities = ev.roster if len(ev.roster) == n else \
            [f"H{i}" for i in range(1, n + 1)]
        inv = Inventory(entities, attrs, aliases=aliases)
        variables.update(inv.variables())
        specs.extend(inv.base_constraints())

    # ---- statements ground route by route (all evidence-licensed) ----------
    unread: List[str] = []
    inert: List[str] = []
    query: Optional[dict] = None

    def _absorb(out) -> bool:
        nonlocal query
        if out is None:
            return False
        if isinstance(out, dict):        # row declared its own variables
            for v, dom in out.get("vars", {}).items():
                variables.setdefault(v, list(dom))
            specs.extend(out.get("specs", []))
            if out.get("query"):
                query = out["query"]
            return True
        specs.extend(out)
        return True

    # a linear-order frame is licensed by an inventory sentence in the prose
    # ("...five books: a red book, a green book, ...") when no value grid
    # already licensed variables and no examples are present
    from .reading.ordering import (compile_order_statement, extract_entities,
                           _find_entities, _LIST_RE)
    order_ents: List[str] = []
    if inv is None and not ev.assert_texts and ev.statements:
        order_ents = extract_entities(" ".join(t for t, _ in ev.statements))
        if len(order_ents) >= 2:
            variables.update({e: list(range(1, len(order_ents) + 1))
                              for e in order_ents})
            specs.append({"type": "alldiff", "vars": list(order_ents),
                          "origin": "[frame] a fixed linear order"})
        else:
            order_ents = []

    def _mention_ctx(text: str) -> dict:
        if inv is None:
            return {"values": [], "entities": [], "value_attr": {}}
        refs = inv.find_mentions(text)
        return {"values": [r[1] for r in refs if r[0] == "value"],
                "entities": [r[1] for r in refs if r[0] == "entity"],
                "value_attr": inv.value_attr}

    for st, kind in ev.statements:
        if inv is not None:
            ctx = _mention_ctx(st)
            if _absorb(kb.ground_statement(st, ctx)):
                fired.add(st); continue
            mentions = bool(ctx["values"] or ctx["entities"])
        elif order_ents:
            if _LIST_RE.search(st):          # the inventory sentence itself
                fired.add(st); continue
            if _absorb(compile_order_statement(st, order_ents,
                                               len(order_ents))):
                fired.add(st); continue
            mentions = bool(_find_entities(st, order_ents))
        else:
            if _absorb(kb.ground_statement(st, _mention_ctx(st))):
                fired.add(st); continue
            mentions = False
        if kind == "prose" and not mentions:
            inert.append(st)     # mentions nothing the model knows: inert
        else:
            unread.append(st)

    # the question grounds through the same rows (query shape)
    if ev.question and variables:
        _absorb(kb.ground_statement(ev.question, _mention_ctx(ev.question)))

    # ---- typed I/O examples + a grounded frame license program slots -------
    sig = frame = None
    if ev.assert_texts:
        try:
            sig = parse_signature(ev.assert_texts)
        except (ValueError, SyntaxError) as e:
            rec["reasons"].append(f"signature: {e}")
        if sig:
            frame_errs = []
            for st in list(unread) + list(inert):
                try:
                    frame = parse_frame(st)
                    (unread if st in unread else inert).remove(st)
                    fired.add(st)
                    break
                except ValueError as e:
                    frame_errs.append(str(e))
            if frame is None:
                rec["reasons"].extend(f"frame: {e}" for e in frame_errs)
    if sig and frame:
        ok = True
        slot_vars: Dict[str, list] = {"prog:op": [frame.op]}
        for sel in ("FIRST", "LAST", "ALL"):
            slot_vars[f"prog:sel:{sel}"] = [1 if sel in frame.selectors else 0]
        if not frame.selectors and frame.theme_kind == "OCCURRENCE":
            rec["reasons"].append("frame: no selector grounded"); ok = False
        for role, noun in (("src", frame.source), ("item", frame.item)):
            if noun is None:
                rec["reasons"].append(f"role '{role}': no noun in frame")
                ok = False; continue
            trow = kb.match_word(noun, "TYPE:")
            if trow is None:
                rec["reasons"].append(f"role noun '{noun}' has no TYPE row")
                ok = False; continue
            want = trow.symbol[5:]
            cands = [i for i, t in enumerate(sig.arg_types)
                     if _compatible(want, t)]
            if not cands:
                rec["reasons"].append(
                    f"role '{role}' ({noun}:{want}) fits no argument")
                ok = False; continue
            slot_vars[f"prog:role:{role}"] = cands
        missing = [kb.match_key((frame.op, s)) is None
                   for s in frame.selectors]
        if any(missing):
            rec["reasons"].append(
                f"no BLOCK rows for op {frame.op} × {frame.selectors}")
            ok = False
        if ok:
            variables.update(slot_vars)
            extra_cons.append(EvalCheck(
                sig, ev.assert_texts, "\n".join(ev.setup),
                slots=list(slot_vars)))
            rec["frame"] = vars(frame)

    # ---- one problem, one engine -------------------------------------------
    total = len(ev.statements)
    rec["coverage"] = {"grounded": len(fired), "statements": total,
                       "unread": unread, "inert": len(inert)}

    # claim algebra: a question over prose that licensed nothing
    # finite-domain -> quantified-rule reading + forward-chaining closure
    # (the worldmodel — general machinery, same honesty gate: read
    # everything or abstain)
    if ev.question and not variables and ev.statements:
        story = _story_route(rec, ev)
        if story is not None:
            return story
        return _claim_route(rec, ev)

    # conclusions mode: prose, no question, nothing finite-domain — read the
    # text and return what logically FOLLOWS from it, each conclusion with
    # its proof chain; unread sentences are reported, never guessed over
    if (not ev.question and not variables and not ev.assert_texts
            and ev.statements):
        return _conclude_route(rec, ev)

    if not variables:
        rec["status"] = "UNGROUNDED" if (unread or ev.statements) else "EMPTY"
        if not rec["reasons"]:
            rec["reasons"].append("nothing licensed any variables")
        return rec
    if unread:
        rec["status"] = "UNGROUNDED"
        rec["reasons"].append(f"{len(unread)} statement(s) unread")
        return rec
    if not fired and not extra_cons:
        # variables with no grounded statement constrain nothing: answering
        # would certify a vacuous model
        rec["status"] = "UNGROUNDED"
        rec["reasons"].append("no statement grounded — vacuous model")
        return rec

    # ---- symmetry breaking: when NO constraint references an absolute
    # position (all types are permutation-invariant), positions are pure
    # labels — pinning ONE group to the identity permutation is a sound
    # canonicalization (every co-location class keeps exactly one
    # representative), not an assumption about the world.
    _PERM_INVARIANT = {"alldiff", "same", "diff"}
    if inv is not None and specs and \
            all(s["type"] in _PERM_INVARIANT for s in specs):
        g0 = next(iter(ev.groups))
        pin_vals = [_norm_value(v) for v in ev.groups[g0]]
        specs.extend({"type": "is", "var": v, "value": i + 1,
                      "origin": f"[symmetry] '{g0}' pinned as canonical axis"}
                     for i, v in enumerate(pin_vals))
        rec["assumptions"] = [f"positions are labels; group '{g0}' pinned"]

    # entailment mode: a query or options ask what holds in EVERY model,
    # so the engine enumerates models instead of certifying uniqueness
    entail = bool(ev.options or query is not None)
    cons = [build_constraint(s, variables) for s in specs] + extra_cons
    res = Engine(Problem(variables, cons)).solve(
        max_solutions=400 if entail else 2)
    rec["status"] = res.status
    rec["engine"] = {"explanation": res.explanation, **res.stats}

    if entail and res.solutions:
        if query is not None:
            qc = build_constraint(query, variables)
            sat = [qc.check(sol) for sol in res.solutions]
            if all(sat):
                rec.update(status="SOLVED", answer="Yes")
            elif not any(sat):
                rec.update(status="SOLVED", answer="No")
            else:
                rec.update(status="UNDETERMINED")
                rec["reasons"].append("query truth varies across models")
        if ev.options:
            verdicts = []
            for label, text in ev.options:
                sp = (compile_order_statement(text, order_ents,
                                              len(order_ents))
                      if order_ents else None)
                if sp is None:
                    verdicts.append((label, None))
                    continue
                ocons = [build_constraint(s, variables) for s in sp]
                verdicts.append((label, all(
                    all(c.check(sol) for c in ocons)
                    for sol in res.solutions)))
            rec["entailment"] = verdicts
            picks = [l for l, v in verdicts if v]
            if len(picks) == 1:
                rec.update(status="SOLVED", answer=picks[0])
            else:
                rec.update(status="UNDETERMINED")
                rec["reasons"].append(
                    f"{len(picks)} option(s) entailed, need exactly 1")
    elif res.solutions:
        sol = res.solutions[0]
        if inv is not None:                 # answers in the text's own words
            rec["answer"] = {raw_name.get(v, v): sol[v]
                             for v in inv.value_attr}
        if sig and any(k.startswith("prog:") for k in sol):
            rec["code"] = render_program(sig, sol)
    return rec


# ------------------------------------------------------- narrative algebra

def _story_route(rec: Dict, ev: Evidence) -> Optional[Dict]:
    """Value queries over a story: the question must match a narrative
    query shape; sentences fold into the interval-stamped fluent state.
    Returns None (not story-shaped) to fall through to the claim route."""
    from .reading import narrative
    if narrative.match_query(ev.question) is None:
        return None
    st = narrative.StoryState()
    unread = []
    for i, (t, _) in enumerate(ev.statements):
        if not narrative.fold(t, st, i):
            unread.append(t)
    rec["algebra"] = "narrative"
    rec["coverage"] = {"grounded": len(ev.statements) - len(unread),
                       "statements": len(ev.statements), "unread": unread}
    ans = narrative.answer(ev.question, st)
    if ans is None:
        rec["status"] = "UNDETERMINED"
        rec["reasons"].append("story state does not determine the answer")
    else:
        rec.update(status="SOLVED", answer=str(ans))
    return rec


# ---------------------------------------------------------- claim algebra

_WM_CACHE: Dict[str, tuple] = {}


def _conclude_route(rec: Dict, ev: Evidence) -> Dict:
    """Open text -> what follows from it. Reads the prose into facts and
    quantified rules, forward-chains to fixpoint, and returns every NEW fact
    with the proof that derives it. Partial reading is allowed here (unlike
    question answering): conclusions come only from what WAS read, and the
    unread remainder is listed."""
    from .reading.worldmodel import WorldModel
    text = " ".join(t for t, _ in ev.statements)
    has_pronoun = bool(re.search(
        r"\b(he|she|it|they|him|her|them|his|hers|its|their)\b", text, re.I))
    wm = WorldModel.from_text(text, resolve_coref=has_pronoun)
    derived = wm.closure()
    rec["algebra"] = "closure"
    rec["read"] = {"facts": len(wm.facts), "rules": len(wm.rules),
                   "unread": list(wm.unread)}
    rec["conclusions"] = [
        {"conclusion": wm.render(lit.key(), lit.pos), "proof": str(proof)}
        for lit, proof in derived]
    if wm.contradictions:
        rec["contradictions"] = [str(c) for c in wm.contradictions]

    # total capture (the associative tier): EVERY clause of the text enters
    # the semantic graph — reported as "the text asserts ...", attributed
    # claims wrapped, never reasoned over as if proof-grade
    from .reading.semgraph import read as sg_read
    g = sg_read(text)
    rec["capture"] = g.coverage()
    rec["captured"] = [c.report() for c in g.clauses]
    rec["attributions"] = [c.report() for c in g.clauses
                           if c.modality is not None]

    if wm.facts or wm.rules:
        rec["status"] = "CONCLUSIONS"
    elif g.clauses:
        rec["status"] = "CAPTURED"      # represented, nothing proof-grade
    else:
        rec["status"] = "UNGROUNDED"
    if wm.unread:
        rec["reasons"].append(
            f"{len(wm.unread)} sentence(s) not proof-grade — conclusions "
            f"cover only what was read; the rest is captured associatively")
    return rec


def _claim_route(rec: Dict, ev: Evidence) -> Dict:
    """Prose theory + question -> worldmodel reading, forward-chaining
    closure, open-world entailment. Same honesty gate as everywhere:
    every sentence must be read, or the sample is refused."""
    from .reading.worldmodel import WorldModel
    text = " ".join(t for t, _ in ev.statements)
    rec["algebra"] = "closure"
    hit = _WM_CACHE.get(text)
    if hit is None:
        # coreference resolution exists to resolve pronouns; running it on
        # pronoun-free text can only corrupt mentions, never help
        has_pronoun = bool(re.search(
            r"\b(he|she|it|they|him|her|them|his|hers|its|their)\b",
            text, re.I))
        wm = WorldModel.from_text(text, resolve_coref=has_pronoun)
        ok = not wm.unread
        if ok:
            wm.closure()
        hit = (wm, ok)
        if len(_WM_CACHE) < 4096:
            _WM_CACHE[text] = hit
    wm, ok = hit
    if not ok:
        rec["status"] = "UNGROUNDED"
        rec["reasons"].append(
            f"claim route: {len(wm.unread)} sentence(s) unread")
        return rec

    # directional-relation query shape: "...relation of (the) X to (the) Y?"
    m = re.search(r"relation of (?:the )?(?:\w+ )??(\w+) to "
                  r"(?:the )?(?:\w+ )??(\w+)\s*\?", ev.question)
    if m:
        from .reading.worldmodel import LABEL_OF
        vec = wm.vector(m.group(1), m.group(2), metric=True)
        label = LABEL_OF.get(vec) if vec is not None else None
        if label is None:
            rec["status"] = "UNDETERMINED"
            rec["reasons"].append("relative direction not forced")
        else:
            rec.update(status="SOLVED", answer=label)
        return rec

    # ask() judges a CLAIM (True/False/Unknown). A WH-question asks for a
    # VALUE and an option set asks for a CHOICE — different query kinds;
    # answering either with a truth value would be nonsense.
    if ev.options:
        rec["status"] = "UNDETERMINED"
        rec["reasons"].append("options over a prose theory: choice query "
                              "kind not supported by the claim route")
        return rec
    if re.match(r"(?i)\s*(where|what|who|whom|which|how|why|when)\b",
                ev.question):
        rec["status"] = "UNDETERMINED"
        rec["reasons"].append("WH-question: value query kind not supported "
                              "by the claim route")
        return rec
    rec["status"] = "SOLVED"
    rec["answer"] = str(wm.ask(ev.question))
    return rec


# ================================================================ CLI

if __name__ == "__main__":
    import argparse, json, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("sample", help="path to a JSON sample, or '-' for stdin")
    args = ap.parse_args()
    data = json.load(sys.stdin if args.sample == "-" else open(args.sample))
    print(json.dumps(solve(data), indent=1, default=str))
