#!/usr/bin/env python3
"""Gap-driven retrieval: the refusal IS the query.

When reasoning stops because a word has no meaning in the store, the
refusal names it. That name becomes a Gap; SOURCE ADAPTERS answer it with
candidate rows in the ONE Row schema, provenance-stamped; the ADMISSION
TRIBUNAL decides what enters the working store; EMPLOYMENT stays
oracle-gated (a retrieved meaning is used only if the composition it
enters survives the tests / proofs). Nothing is ever absorbed as text.

Trust ranks (a higher-trust row is never overridden):
    hand = calibrated (3)  >  derived = introspected (2)  >  kb (1)  > web (0)
'introspected' means facts obtained BY EXECUTION — the Python runtime is
interrogated for real callables and their signatures, which makes the
entire standard library an on-demand atom space for the program algebra:
deep, reliable, and reproducible offline.

Adapters are the ONLY plug — like leaf grounders in the compiler and
calibrate hooks in the runner. The registry is data; nothing here knows
any domain or benchmark."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from . import kb

TRUST = {"hand": 3, "calibrated": 3, "derived": 2, "introspected": 2,
         "wordnet": 1, "wikidata": 1, "corpus": 1, "web": 0}

# The knowledge-kind ONTOLOGY: a closed set. Every kind declares its
# representation, the oracle that admits/verifies it, and the grade its
# use confers on an answer. No oracle, no kind.
KINDS = {
    "meaning":   {"grade": "certified", "oracle": "composition+task-oracle",
                  "stored": True},
    "callable":  {"grade": "certified", "oracle": "signature+execution",
                  "stored": True},
    "formula":   {"grade": "certified", "oracle": "reproduces-known-examples",
                  "stored": True},
    "fact":      {"grade": "likely",    "oracle": "text-conflict+closure",
                  "stored": True},
    "procedure": {"grade": "verified",  "oracle": "sandbox-vs-task-oracle",
                  "stored": False},   # task-shaped: employed, never stored
    "example":   {"grade": "certified", "oracle": "consistency-with-given",
                  "stored": False},
    "policy":    {"grade": "calibrated", "oracle": "labeled-data-agreement",
                  "stored": True},
}


def rank(provenance: str) -> int:
    return TRUST.get(str(provenance).split(":")[0], 0)


@dataclass
class Gap:
    word: str                    # the missing meaning, by its word
    kind: str = "callable"       # callable | fact
    arity: Optional[int] = None  # arity hint from the task signature
    context: str = ""            # the sentence that raised the gap


# ------------------------------------------------------------ introspection

_STDLIB = ["builtins", "math", "statistics", "itertools", "functools",
           "operator", "string", "collections"]
_WRAP_LIST = {"itertools"}       # iterator factories -> list(...) for tests


def _name_matches(name: str, word: str) -> bool:
    n, w = name.lower(), word.lower()
    if len(w) < 3:
        return False                    # too short to identify anything
    return (n == w or (len(w) >= 4 and (n == w + "s" or n + "s" == w))
            or (len(n) >= 4 and w.startswith(n))
            or (len(w) >= 4 and n.startswith(w)))


def introspection_adapter(gap: Gap) -> List[kb.Row]:
    """Interrogate the Python runtime: find real callables matching the gap
    word, harvest their true arity from their signatures, and propose them
    as FUNC (1-ary) / BINOP (2-ary) atom rows. Facts by execution."""
    if gap.kind != "callable":
        return []
    import importlib, inspect
    out: List[kb.Row] = []
    for modname in _STDLIB:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        for name in dir(mod):
            if name.startswith("_") or not _name_matches(name, gap.word):
                continue
            obj = getattr(mod, name)
            if not callable(obj):
                continue
            try:
                sig = inspect.signature(obj)
                pars = list(sig.parameters.values())
                n = len([p for p in pars
                         if p.kind in (p.POSITIONAL_ONLY,
                                       p.POSITIONAL_OR_KEYWORD)
                         and p.default is p.empty])
                variadic = any(p.kind == p.VAR_POSITIONAL for p in pars)
            except (ValueError, TypeError):
                n, variadic = 0, True     # C builtins without signatures
            call = name if modname == "builtins" \
                else f"__import__('{modname}').{name}"
            prov = f"introspected:{modname}.{name}"
            arities = {n} if not variadic else {n, 1, 2}
            if 1 in arities:
                body = f"{call}({{src}})"
                if modname in _WRAP_LIST:
                    body = f"list({body})"
                out.append(kb.Row(gap.word.lower(), "FUNC", payload=body,
                                  provenance=prov, confidence=0.8))
            if 2 in arities:
                out.append(kb.Row(gap.word.lower(), "BINOP",
                                  payload=f"({call}({{a}}, {{b}}))",
                                  provenance=prov, confidence=0.8))
    return out


# ------------------------------------------------------- procedure corpora

# Worked-example corpora: any caller-named source of (spec text, solution)
# pairs. Candidates are matched by lexical overlap with the gap's context,
# returned as PROCEDURE rows — task-shaped knowledge, employed only through
# the task's own oracle and NEVER stored (KINDS["procedure"]["stored"]).
PROCEDURE_CORPORA: List[tuple] = []


def _extract_code(text: str) -> str:
    """Corpus payloads are often prose-wrapped ("Here's a function: ```python
    ...```"). Knowledge must be code, not text ABOUT code: extract fenced
    blocks when present; otherwise return the text unchanged."""
    import re, textwrap
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    keep = [b for b in blocks if "def " in b]
    return textwrap.dedent("\n\n".join(keep)) if keep else text

_STOP = {"write", "a", "an", "the", "to", "of", "in", "for", "and", "or",
         "python", "function", "given", "find", "that", "with", "is"}


def _norm_text(text: str) -> str:
    """Hyphenated compounds collapse ('co-prime' -> 'coprime') so the same
    concept is the same token on BOTH sides of every lookup."""
    import re
    return re.sub(r"(?<=[a-z])-(?=[a-z])", "", text.lower())


def _norm_word(w: str) -> str:
    """Trailing plural/3rd-person 's' strips ('checks' -> 'check')."""
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def _variants(w: str) -> set:
    """ALL forms of a word — surface AND normalized — so variants meet
    without ever losing the original: {co-prime, coprime}, {checks, check}.
    Matching = any variant in common; nothing is replaced or lost."""
    w = w.lower()
    out = {w, w.replace("-", "")}
    for f in list(out):
        if f.endswith("s") and len(f) > 3:
            out.add(f[:-1])
    return out


def _tokens(text: str):
    import re
    return re.findall(r"[a-z]+(?:-[a-z]+)*", text.lower())


def _words(text: str) -> frozenset:
    out = set()
    for t in _tokens(text):
        out |= _variants(t)
    return frozenset(out) - _STOP


def register_procedure_corpus(source: str) -> None:
    """source = "dataset[#config]@split1[+split2]:textfield:codefield"
    (caller names it, like calibrate hooks). Corpora of any size: an
    inverted word index makes lookup independent of corpus length."""
    import re
    from collections import defaultdict
    from datasets import load_dataset
    name_split, tf, cf = source.split(":")
    name, splits = name_split.split("@")
    config = None
    if "#" in name:
        name, config = name.split("#")
    pairs = []
    for split in splits.split("+"):
        ds = load_dataset(name, config, split=split) if config             else load_dataset(name, split=split)
        for x in ds:
            spec, code = str(x.get(tf, "")), str(x.get(cf, ""))
            code = _extract_code(code)
            if spec and code and "def " in code and len(code) < 4000:
                words = _words(spec)
                if words:
                    pairs.append((words, spec, code))
    index = defaultdict(list)
    for i, (words, _, _) in enumerate(pairs):
        for w in words:
            index[w].append(i)
    index = {w: ids for w, ids in index.items() if len(ids) <= 20000}
    PROCEDURE_CORPORA.append((source, pairs, index))
    print(f"  registered procedure corpus {source}: {len(pairs)} examples")


def procedure_adapter(gap: Gap) -> List[kb.Row]:
    """Top matching worked examples for a procedure gap: candidates come
    from the inverted index (pairs sharing >=1 informative word), scored by
    lexical overlap."""
    if gap.kind != "procedure" or not gap.context or not PROCEDURE_CORPORA:
        return []
    q = _words(gap.context)
    # FOCUSED REQUEST: the query is the conjunction of the task's
    # DISCRIMINATIVE words (generic words carry no identity). A candidate
    # containing ALL of them is fetched no matter how the rest of the
    # wording differs; lexical overlap only RANKS within the fetched set.
    from .compose import _TYPE_HINT
    generic = set(_TYPE_HINT) | _STOP |         {r.pattern for r in kb.KB if isinstance(r.pattern, str)
         and r.symbol.startswith(("OP:", "DISCOURSE:"))} |         {"give", "value", "item", "element", "name", "use", "take",
         "whether", "one", "two", "three", "number", "code", "program",
         "not", "no", "if", "they", "are", "it", "this", "when", "from",
         "into", "out", "up", "down", "all", "any", "each", "which"}
    gen_vars = set()
    for g in generic:
        gen_vars |= _variants(g)
    # each discriminative task word is a VARIANT GROUP; a candidate covers
    # it if ANY variant is present (surface or normalized)
    disc_groups = [frozenset(_variants(t)) for t in set(_tokens(gap.context))
                   if not (_variants(t) & (gen_vars | _STOP))]
    scored = []
    for src, pairs, index in PROCEDURE_CORPORA:
        cand_ids = set()
        for w in q:
            cand_ids.update(index.get(w, ()))
        for i in cand_ids:
            words, spec, code = pairs[i]
            j = len(q & words) / max(1, len(q | words))
            if disc_groups and all(g & words for g in disc_groups):
                scored.append((1.0 + j, spec, code, src))   # conjunctive hit
            elif j >= 0.25:
                scored.append((j, spec, code, src))
    scored.sort(key=lambda t: -t[0])
    return [kb.Row(("PROCEDURE", spec[:64]), "PROCEDURE", payload=code,
                   sig={"match": round(j, 3), "spec": spec},
                   provenance=f"corpus:{src}", confidence=j)
            for j, spec, code, src in scored[:8]]


def mining_adapter(gap: Gap) -> List[kb.Row]:
    """The STRUCTURED query route: for a missing WORD, search the corpora
    for functions whose descriptions feature that word (a specific request,
    not sentence similarity), and DECOMPOSE them through the code reader
    into candidate meaning rows. The tribunal and the oracles judge what
    survives — mined knowledge is composable, not a replayed black box."""
    if gap.kind != "callable" or not PROCEDURE_CORPORA:
        return []
    from .codereader import read_functions
    from .compose import _TYPE_HINT
    w = _norm_word(_norm_text(gap.word))
    # a mined meaning may only be keyed by a DISCRIMINATIVE word: type
    # nouns, operation verbs, and function words appear in almost every
    # spec, so rows keyed on them are licensed everywhere — the imposter
    # recipe. (Caught empirically: an oddness test mined under 'number'
    # passed a Woodall task's three asserts.)
    generic = set(_TYPE_HINT) | _STOP |         {r.pattern for r in kb.KB if isinstance(r.pattern, str)
         and r.symbol.startswith("OP:")} |         {"give", "value", "item", "element", "name", "use", "take",
         "verify", "validity", "input", "output", "result", "program"}
    if w in generic:
        return []
    out: List[kb.Row] = []
    for src, pairs, index in PROCEDURE_CORPORA:
        ids = []
        for v in _variants(w):
            ids += list(index.get(v, ()))
        for i in ids[:40]:                   # the word itself IS the query
            words, spec, code = pairs[i]
            out.extend(read_functions(
                code, w, provenance=f"mined:{src.split('@')[0]}"))
            if len(out) >= 12:
                break
    # dedupe identical payloads
    seen, uniq = set(), []
    for r in out:
        if (r.symbol, r.payload) not in seen:
            seen.add((r.symbol, r.payload))
            uniq.append(r)
    return uniq[:8]


# ------------------------------------------------------------ the registry

ADAPTERS: List[Callable[[Gap], List[kb.Row]]] = [introspection_adapter,
                                                 mining_adapter,
                                                 procedure_adapter]


# ------------------------------------------------------------ the tribunal

def tribunal(gap: Gap,
             candidates: List[kb.Row]) -> Tuple[List[kb.Row], List[str]]:
    """Admission: a candidate is rejected when a higher-trust row already
    defines the same word in the same role (hand knowledge wins), and
    deduplicated against identical payloads. Admission is SESSION-scope;
    employment stays oracle-gated downstream."""
    admitted, rejected = [], []
    for c in candidates:
        if c.symbol == "PROCEDURE":     # task-shaped: employed, not stored
            admitted.append(c)
            continue
        clash = [r for r in kb.KB
                 if r.pattern == c.pattern and r.symbol == c.symbol]
        if any(rank(r.provenance) > rank(c.provenance) for r in clash):
            rejected.append(f"{c.provenance}: a higher-trust row already "
                            f"defines '{c.pattern}'")
            continue
        if any(r.payload == c.payload for r in clash):
            continue
        kb.add(c)
        admitted.append(c)
    return admitted, rejected


def retrieve(gap: Gap) -> Tuple[List[kb.Row], List[str]]:
    """Run every adapter, pool the candidates, admit through the tribunal.
    Adapter failures are contained — a dead source never kills the loop."""
    cands: List[kb.Row] = []
    for adapter in ADAPTERS:
        try:
            cands.extend(adapter(gap))
        except Exception as e:                      # noqa: BLE001
            pass
    return tribunal(gap, cands)
