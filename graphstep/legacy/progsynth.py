#!/usr/bin/env python3
"""Zero-LLM program synthesis: NL spec + asserts -> certified Python code.

The GraphStep recipe applied to code generation — the engine, not an LLM,
produces the program:

  SIGNATURE  the asserts are part of the task statement: ast-parse them into
             the function name, arity, and typed I/O examples (deterministic).
  FRAME      dependency parse of the spec -> a semantic frame:
             action verb -> OPERATION, grounded through the lexicon and,
             on a miss, WordNet synonym lookup (external knowledge admitted
             as symbols with provenance — the retrieval-tier discipline);
             theme noun (+ selectors like "first and last" via coordination),
             item role ("of a given character"), source role ("from the
             string").
  BIND       frame roles bind to function parameters by TYPED EVIDENCE from
             the examples (a "character" is the argument that is a length-1
             string in every example; a "string" is a str; a "list" a list).
  COMPILE    the frame compiles to Python over a small typed DSL of
             primitives. No search, no guessing: an unsupported frame is an
             honest failure with the reason named.
  CERTIFY    the asserts run in a sandboxed subprocess (bench_mbpp.run_tests)
             — code is only emitted if EVERY assert passes. With only a few
             examples this certificate is weaker than a uniqueness proof;
             the defense against overfitting is that the program comes from
             the SPEC (the frame), and the tests only confirm it.

Coverage grows frame-by-frame (the AR-LSAT model): each supported
(operation, theme, roles) shape is one "frame family", and the report always
says which layer produced the answer and which specs did not compile.

Usage:
  python3 -m graphstep.progsynth --task 11        # one MBPP task by id
  python3 -m graphstep.progsynth --limit 20       # first N of the test split
"""
from __future__ import annotations
import argparse, ast, json, textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..engine.sandbox import run_tests
from ..reading.retrieval import _ensure_wordnet


# ------------------------------------------------------------- signature

@dataclass
class Signature:
    name: str
    arity: int
    examples: List[Tuple[list, object]]      # (args, expected)
    arg_types: List[str] = field(default_factory=list)  # CHAR/STRING/LIST/INT/FLOAT/?


def _literal(node):
    return ast.literal_eval(node)            # raises on non-literals


def parse_signature(tests: List[str]) -> Signature:
    """asserts -> (name, arity, typed examples). Raises ValueError with the
    reason when an assert is not a readable `assert f(lits) == lit` shape."""
    name, examples = None, []
    for t in tests:
        stmt = ast.parse(t.strip()).body[0]
        if not isinstance(stmt, ast.Assert):
            raise ValueError(f"not an assert: {t}")
        test = stmt.test
        if isinstance(test, ast.Compare) and isinstance(test.left, ast.Call):
            call, expected = test.left, _literal(test.comparators[0])
        elif isinstance(test, ast.Call):
            call, expected = test, True
        else:
            raise ValueError(f"unreadable assert shape: {t}")
        if not isinstance(call.func, ast.Name):
            raise ValueError(f"non-simple callee: {t}")
        if name and call.func.id != name:
            raise ValueError("asserts call different functions")
        name = call.func.id
        examples.append(([_literal(a) for a in call.args], expected))
    arity = len(examples[0][0])
    sig = Signature(name=name, arity=arity, examples=examples)
    for pos in range(arity):
        vals = [ex[0][pos] for ex in examples]
        if all(isinstance(v, str) and len(v) == 1 for v in vals):
            sig.arg_types.append("CHAR")
        elif all(isinstance(v, str) for v in vals):
            sig.arg_types.append("STRING")
        elif all(isinstance(v, list) for v in vals):
            sig.arg_types.append("LIST")
        elif all(isinstance(v, bool) for v in vals):
            sig.arg_types.append("BOOL")
        elif all(isinstance(v, int) for v in vals):
            sig.arg_types.append("INT")
        elif all(isinstance(v, (int, float)) for v in vals):
            sig.arg_types.append("FLOAT")
        else:
            sig.arg_types.append("?")
    return sig


# ------------------------------------------------------------- lexicon

# operation lexicon: verb lemma -> abstract operation. WordNet synonym
# closure widens this at grounding time (with provenance), so only the
# canonical member of each family needs listing.
OPERATION_LEX = {
    "remove": "REMOVE", "delete": "REMOVE", "erase": "REMOVE",
    "strip": "REMOVE", "drop": "REMOVE",
}

SELECTOR_LEX = {"first": "FIRST", "last": "LAST",
                "all": "ALL", "every": "ALL"}

# noun -> the argument type that can play this role (typed role binding)
ROLE_TYPE_LEX = {
    "character": "CHAR", "char": "CHAR", "letter": "CHAR",
    "string": "STRING", "word": "STRING", "sentence": "STRING",
    "list": "LIST", "array": "LIST",
    "number": "INT", "integer": "INT",
}

THEME_OCCURRENCE = {"occurrence", "instance", "appearance"}


def ground_verb(lemma: str) -> Tuple[Optional[str], Optional[str]]:
    """verb lemma -> (OPERATION, provenance). Lexicon first; on a miss the
    WordNet synonym sets are searched for a lemma the lexicon knows —
    knowledge admitted as a symbol, with the synset named as the source."""
    if lemma in OPERATION_LEX:
        return OPERATION_LEX[lemma], "lexicon"
    try:
        wn = _ensure_wordnet()
    except Exception:
        return None, None
    for syn in wn.synsets(lemma, pos=wn.VERB):
        for other in syn.lemma_names():
            other = other.lower().replace("_", " ")
            if other in OPERATION_LEX:
                return OPERATION_LEX[other], f"wordnet:{syn.name()}"
    return None, None


# ------------------------------------------------------------- frame

@dataclass
class Frame:
    op: str                       # abstract operation (REMOVE, ...)
    op_provenance: str
    theme: str                    # theme noun ("occurrence")
    theme_kind: str               # OCCURRENCE | ITEM
    selectors: List[str]          # FIRST/LAST/ALL (order as written)
    item: Optional[str]           # what is being located ("character")
    source: Optional[str]         # where ("string")


_NLP = None
def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


WRAPPER_VERBS = {"write", "create", "define", "implement", "make"}


def parse_frame(text: str) -> Frame:
    """Spec sentence -> semantic frame, or ValueError naming what failed."""
    doc = _nlp()(text)
    sent = next(doc.sents)

    # peel the "write a (python) function to X" wrapper: the real action is
    # the advcl/xcomp verb under the wrapper root.
    verb = sent.root
    if verb.lemma_.lower() in WRAPPER_VERBS:
        inner = [c for c in verb.children if c.pos_ == "VERB"
                 and c.dep_ in ("advcl", "xcomp", "ccomp")]
        if not inner:
            raise ValueError("wrapper verb with no inner action verb")
        verb = inner[0]

    op, prov = ground_verb(verb.lemma_.lower())
    if op is None:
        raise ValueError(f"ungrounded verb: '{verb.lemma_}' "
                         f"(not in lexicon, no WordNet route)")

    themes = [c for c in verb.children if c.dep_ in ("dobj", "obj")]
    if not themes:
        raise ValueError(f"no direct object under '{verb.text}'")
    theme = themes[0]

    # selectors: adjectival modifiers of the theme + their coordination chain
    selectors: List[str] = []
    for m in theme.children:
        if m.dep_ == "amod":
            chain = [m] + list(m.conjuncts)
            for tok in sorted(chain, key=lambda t: t.i):
                sel = SELECTOR_LEX.get(tok.lemma_.lower())
                if sel and sel not in selectors:
                    selectors.append(sel)

    # roles: "of X" under the theme -> item; "from/in Y" anywhere -> source
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

    kind = "OCCURRENCE" if theme.lemma_.lower() in THEME_OCCURRENCE else "ITEM"
    if kind == "ITEM" and item is None:
        item = theme.lemma_.lower()

    return Frame(op=op, op_provenance=prov, theme=theme.lemma_.lower(),
                 theme_kind=kind, selectors=selectors, item=item,
                 source=source)


# ------------------------------------------------------------- binding

def bind_roles(frame: Frame, sig: Signature) -> Dict[str, int]:
    """Map frame roles -> parameter positions using typed evidence from the
    examples. Raises ValueError when a role has no unique typed home."""
    binding: Dict[str, int] = {}
    taken = set()
    for role_name, noun in (("source", frame.source), ("item", frame.item)):
        if noun is None:
            continue
        want = ROLE_TYPE_LEX.get(noun)
        if want is None:
            raise ValueError(f"role noun '{noun}' has no known type")
        # CHAR evidence is stronger than STRING: try exact first, then allow
        # a STRING role to take a CHAR-typed argument (a char IS a string).
        cands = [i for i, t in enumerate(sig.arg_types)
                 if i not in taken and t == want]
        if not cands and want == "STRING":
            cands = [i for i, t in enumerate(sig.arg_types)
                     if i not in taken and t == "CHAR"]
        if len(cands) != 1:
            raise ValueError(f"role '{role_name}' ({noun}:{want}) has "
                             f"{len(cands)} candidate parameters, need 1")
        binding[role_name] = cands[0]
        taken.add(cands[0])
    return binding


# ------------------------------------------------------------- compile

def compile_frame(frame: Frame, sig: Signature,
                  binding: Dict[str, int]) -> str:
    """Frame -> Python source. Only supported frame families compile;
    anything else raises ValueError naming the unsupported shape."""
    params = [f"arg{i}" for i in range(sig.arity)]
    header = f"def {sig.name}({', '.join(params)}):"

    if (frame.op == "REMOVE" and frame.theme_kind == "OCCURRENCE"
            and "source" in binding and "item" in binding
            and frame.selectors
            and set(frame.selectors) <= {"FIRST", "LAST", "ALL"}):
        s, ch = params[binding["source"]], params[binding["item"]]
        body = [f"    s = {s}"]
        if "ALL" in frame.selectors:
            body.append(f"    s = s.replace({ch}, '')")
        else:
            if "FIRST" in frame.selectors:
                body += [f"    i = s.find({ch})",
                         "    if i != -1:",
                         "        s = s[:i] + s[i+1:]"]
            if "LAST" in frame.selectors:
                body += [f"    j = s.rfind({ch})",
                         "    if j != -1:",
                         "        s = s[:j] + s[j+1:]"]
        body.append("    return s")
        return "\n".join([header] + body)

    raise ValueError(
        f"unsupported frame family: op={frame.op} theme={frame.theme_kind} "
        f"selectors={frame.selectors} roles={sorted(binding)}")


# ------------------------------------------------------------- pipeline

def synthesize(text: str, tests: List[str]) -> Dict:
    """Full zero-LLM ladder for one task. Returns a record with the trace;
    status SOLVED only when the certificate (all asserts) passes."""
    rec: Dict = {"text": text, "tests": tests, "status": "FAILED",
                 "stage": None, "reason": None}
    try:
        sig = parse_signature(tests)
        rec["signature"] = {"name": sig.name, "arity": sig.arity,
                            "arg_types": sig.arg_types}
    except (ValueError, SyntaxError) as e:
        rec.update(stage="signature", reason=str(e)); return rec
    try:
        frame = parse_frame(text)
        rec["frame"] = vars(frame)
    except ValueError as e:
        rec.update(stage="frame", reason=str(e)); return rec
    try:
        binding = bind_roles(frame, sig)
        rec["binding"] = binding
    except ValueError as e:
        rec.update(stage="binding", reason=str(e)); return rec
    try:
        code = compile_frame(frame, sig, binding)
        rec["code"] = code
    except ValueError as e:
        rec.update(stage="compile", reason=str(e)); return rec

    res = run_tests(code, tests)
    rec["certificate"] = res
    if res["ok"]:
        rec.update(status="SOLVED", stage="certified")
    else:
        rec.update(stage="certificate",
                   reason=f"tests failed: {res['failed_test']} "
                          f"({res['error']})")
    return rec


# ------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=None, help="MBPP task_id")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N tasks of the test split")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", split="test")
    if args.task is not None:
        samples = [ex for ex in ds if ex["task_id"] == args.task]
    else:
        samples = list(ds)[: args.limit or 1]

    solved = 0
    for ex in samples:
        rec = synthesize(ex["text"], list(ex["test_list"]))
        rec["task_id"] = ex["task_id"]
        solved += rec["status"] == "SOLVED"
        print(json.dumps(rec, indent=1, default=str))
    print(f"\n=== zero-LLM synthesis: {solved}/{len(samples)} certified ===")


if __name__ == "__main__":
    main()
