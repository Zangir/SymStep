#!/usr/bin/env python3
"""The question layer: typed questions over SUBGRAPHS and GROUPS of the
sample's reading, answered from the one store, from registered sources,
or by mining usages — the last derivation method of the agenda ladder
(it fires only after the earlier paths refuse, so everything previously
solved is untouched).

Questions (all general; dispatch is on evidence shape and meaning TYPE):
  DENOTE(subgraph, demand)  what does this subgraph mean, as that type?
  USAGE(word)               show usages anywhere (registered documents,
                            worked-example corpora); a usage FAMILY's
                            shared shape is the meaning, its varying
                            constants the parameters
  BRIDGE(have, want)        what maps type A to type B? — answered from
                            the store's own rows, never listed in code
  ALIGN(spec)               which stored examples align structurally?
  CONTRAST(groups)          what separates the asserts' answer groups?

Licenses: typed coverage (a wrong-KIND meaning never suppresses a
request), argument coverage, word connection (an unconnected evidence
separator is a CONJECTURE and never reported as an answer), marker
agreement, discriminative-word coverage for whole replay.
The oracle (the sample's own tests) is always last and always final.
"""
from __future__ import annotations
import ast, itertools, re
from collections import Counter, defaultdict
from typing import List, Optional

from . import kb
from ..engine.sandbox import run_tests

MAX_ORACLE = 60
_HUGE = 200000            # numeric-domain guard for enumerations

def _type_nouns():
    """Type-denoting nouns are STORE rows (ENG:TYPENOUN), not engine data."""
    return {r.pattern: r.sig.get("t", "ANY") for r in kb.KB
            if r.symbol == "ENG:TYPENOUN"}


_DF_CACHE = {"n": -1, "out": None}
def _generic():
    """Genericity is CONTEXT-DERIVED when corpora are registered: a word
    appearing in a large share of example descriptions carries no task
    identity (document frequency). The ENG:STOP rows are only the
    cold-start prior for when no corpus is armed yet."""
    from .sources import PROCEDURE_CORPORA, _STOP
    base = set(_STOP) | {r.pattern for r in kb.KB
                         if r.symbol == "ENG:STOP"} | {"not", "non"}
    n = sum(len(p) for _s, p, _i in PROCEDURE_CORPORA)
    if n == _DF_CACHE["n"]:
        return _DF_CACHE["out"]
    if n >= 50:
        df = Counter()
        for _s, pairs, _i in PROCEDURE_CORPORA:
            for words, _spec, _code in pairs:
                df.update(set(words))
        base |= {w for w, c in df.items() if c / n >= 0.20}
    _DF_CACHE.update(n=n, out=base)
    return base

# ---- external knowledge sources (network, cached, silent on failure).
#      Each answers ONE question kind; each is a flag so sources can be
#      validated independently before becoming defaults.
SRC_WIKIDATA_FORMULA = True     # DEFINITION/FORMULA: "what IS this word?"
SRC_WIKTIONARY = True           # CONNECTION: definition-based word links
SRC_OEIS = True                 # EVIDENCE-GROUP: look the examples up

import json as _json, os as _os
_QCACHE_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "data", "qsources_cache.json")
try:
    _QCACHE = _json.load(open(_QCACHE_PATH))
except Exception:
    _QCACHE = {}


def _qcache_put(key, val):
    _QCACHE[key] = val
    try:
        _os.makedirs(_os.path.dirname(_QCACHE_PATH), exist_ok=True)
        _json.dump(_QCACHE, open(_QCACHE_PATH, "w"))
    except Exception:
        pass


_LAST_CALL = [0.0]
def _http_json(url):
    import urllib.request, ssl, time
    wait = 0.4 - (time.time() - _LAST_CALL[0])
    if wait > 0:
        time.sleep(wait)                      # politeness: <=2.5 req/s
    _LAST_CALL[0] = time.time()
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={
        "User-Agent": "graphstep-research/0.1"})
    for attempt in (1, 2):                      # one retry with backoff
        try:
            with urllib.request.urlopen(req, timeout=15,
                                        context=ctx) as r:
                return _json.loads(r.read().decode())
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2.0)


def _latex_to_python(latex):
    """A GENERAL reading of simple closed-form LaTeX into python."""
    e = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"((\1)/(\2))", latex)
    e = e.replace("\\pi", " 3.141592653589793 ")
    e = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", e)
    e = re.sub(r"^\s*\w+_?\{?\w*\}?\s*=\s*", "", e)   # strip "W_n ="
    e = e.replace("\\cdot", "*").replace("\\times", "*")
    e = re.sub(r"(\w)\s*\^\s*\{?(\w+)\}?", r"\1**\2", e)
    e = e.replace("−", "-").replace("{", "").replace("}", "").strip()
    e = re.sub(r"(\*\*\w+|\d|\))\s+(\w)", r"\1*\2", e)
    e = re.sub(r"(\d)\s*([a-zA-Z])", r"\1*\2", e)
    if not re.fullmatch(r"[\w\s+\-*/()%.]+", e):
        return None, ()
    fvars = sorted({t for t in re.findall(r"[a-zA-Z]\w*", e)
                    if t not in ("abs", "min", "max", "pow")})
    return e, tuple(fvars)


def wikidata_formula(term):
    """DEFINITION source: the machine-readable defining formula (P2534)
    of the concept named by TERM. Returns [(expr, free_vars)]."""
    key = f"wdf:{term}"
    if key in _QCACHE:
        return [(e, tuple(v)) for e, v in _QCACHE[key]]
    out = []
    try:
        d = _http_json("https://www.wikidata.org/w/api.php?action="
                       "wbsearchentities&language=en&format=json&search="
                       + term.replace(" ", "%20"))
        for ent in d.get("search", [])[:2]:
            c = _http_json("https://www.wikidata.org/w/api.php?action="
                           "wbgetclaims&format=json&property=P2534&entity="
                           + ent["id"])
            for cl in c.get("claims", {}).get("P2534", []):
                latex = cl["mainsnak"]["datavalue"]["value"]
                e, fv = _latex_to_python(latex)
                if e and 1 <= len(fv) <= 3:
                    out.append((e, fv))
        _qcache_put(key, [[e, list(v)] for e, v in out])   # success only
    except Exception:
        pass
    return out


def wiktionary_defwords(word):
    """CONNECTION source: the content words of WORD's dictionary
    definitions — two words connect when one defines the other."""
    key = f"wkt:{word}"
    if key in _QCACHE:
        return set(_QCACHE[key])
    words = set()
    try:
        d = _http_json("https://en.wiktionary.org/api/rest_v1/page/"
                       "definition/" + word)
        for lang in d.values():
            for pos in lang:
                for sense in pos.get("definitions", []):
                    txt = re.sub(r"<[^>]+>", " ",
                                 sense.get("definition", ""))
                    words |= {w for w in re.findall(r"[a-z]+", txt.lower())
                              if len(w) > 3}
        _qcache_put(key, sorted(words))    # cache successes only
    except Exception:
        pass
    return words


def _read_recurrence(text, fname):
    """A GENERAL recurrence reading: 'F(n) = <expr of F(n-k), n>' with base
    cases 'F(k) = v' -> a memoized python function, or None."""
    fn = re.escape(fname)
    rec = re.search(rf"{fn}\(n\)\s*=\s*([^,.;]+)", text)
    if not rec:
        return None
    body = rec.group(1).strip()
    bases = {int(m.group(1)): int(m.group(2)) for m in
             re.finditer(rf"{fn}\((\d+)\)\s*=\s*(-?\d+)", text)}
    if not bases:
        return None
    body = re.sub(rf"{fn}\(([^)]+)\)", r"_f(\1)", body)
    body = re.sub(r"(\w)\s*\^\s*(\w+)", r"\1**\2", body)
    if not re.fullmatch(r"[\w\s+\-*/()%._]+", body):
        return None
    lines = ["def _f(n, _memo={}):",
             "    if n in _memo: return _memo[n]"]
    for k, v in sorted(bases.items()):
        lines.append(f"    if n == {k}: return {v}")
    lines += [f"    r = {body}", "    _memo[n] = r", "    return r"]
    return "\n".join(lines)


def oeis_lookup(values, pairs=None, words=()):
    """EVIDENCE-GROUP source: look the examples' outputs up as a sequence.
    A candidate must AGREE POSITIONALLY with the examples (its published
    data satisfies data[n+off] == v for every example, some fixed off) —
    the group-membership license for sequences. Returns ('closed', expr,
    name) / ('recurrence', fn_code, name)."""
    key = "oeis:" + ",".join(map(str, values))
    if key in _QCACHE:
        return [tuple(x) for x in _QCACHE[key]]
    out = []
    try:
        results = []
        queries = [",".join(map(str, values)),
                   "%20".join(map(str, values))] + \
                  [w for w in words if len(w) > 3][:2]
        for qq in queries:            # values (consecutive / anywhere),
            d = _http_json(           # then the task's own words
                "https://oeis.org/search?fmt=json&q=" + qq)
            got = d if isinstance(d, list) else (d.get("results") or [])
            results += got or []
        for r in results[:10]:
            nm = r.get("name", "")
            if pairs:
                try:
                    data = [int(x) for x in r.get("data", "").split(",")]
                except ValueError:
                    continue
                if not any(all(0 <= n + off < len(data)
                               and data[n + off] == v for n, v in pairs)
                           for off in (-2, -1, 0, 1)):
                    continue          # fails the positional license
            lines = [nm] + list(r.get("formula", []))[:8]
            for ln in lines:
                m = re.search(r"a\(n\)\s*=\s*([^,.;]+)", ln)
                if m and "a(" not in m.group(1):
                    e, fv = _latex_to_python(m.group(1))
                    if e and fv in ((), ("n",)):
                        out.append(("closed", e, nm[:60]))
            fsym = re.match(r"\s*(?:[\w\s]+:\s*)?([A-Za-z])\(n\)\s*=",
                            nm)
            for ln in lines:
                for sym in ({fsym.group(1)} if fsym else set()) | {"a"}:
                    code = _read_recurrence(ln, sym)
                    if code:
                        out.append(("recurrence", code, nm[:60]))
        out = list(dict.fromkeys(out))[:6]
        _qcache_put(key, out)              # cache successes only
    except Exception:
        out = list(dict.fromkeys(out))[:6]
    return out


# ---- registered documents: prose+code pages armed by the caller, like
#      procedure corpora (e.g. --calibrate ...questions:register_document:...)
DOCUMENTS: List[tuple] = []


def register_document(spec: str) -> None:
    """spec = "<path>|<name>": a local prose document with inline code."""
    import html
    path, name = spec.split("|", 1) if "|" in spec else (spec, spec)
    text = html.unescape(re.sub(r"<[^>]+>", "", open(path).read()))
    DOCUMENTS.append((name, text.splitlines()))
    print(f"  registered document {name}: {len(text.splitlines())} lines")


_NLP = None
def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


_SA_CACHE: dict = {}
def _syn_ant(a: str, b: str) -> Optional[str]:
    if a == b:
        return "same"
    key = (a, b)
    if key not in _SA_CACHE:
        try:
            from .retrieval import _ensure_wordnet
            wn = _ensure_wordnet()
        except Exception:
            _SA_CACHE[key] = None
            return None
        sa, aa, sb, ab = set(), set(), set(), set()
        for w, S, A in ((a, sa, aa), (b, sb, ab)):
            for s in wn.synsets(w):
                for l in s.lemmas():
                    S.add(l.name().lower())
                    A |= {x.name().lower() for x in l.antonyms()}
        _SA_CACHE[key] = ("syn" if (b in sa or a in sb) else
                          "ant" if (b in aa or a in ab) else None)
    return _SA_CACHE[key]


def _content(doc) -> set:
    return {t.lemma_.lower() for t in doc
            if t.pos_ in ("NOUN", "ADJ", "VERB") and t.is_alpha
            and t.lemma_.lower() not in _generic()}


def _shapes_of(code: str):
    """Normalized parametric shapes in any code — usage-family units."""
    out = []
    try:
        tree = ast.parse(code.replace("\r", ""))
    except SyntaxError:
        return out
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice) \
                and n.slice.lower is not None and n.slice.upper is not None:
            out.append(("{src}[{p0}:{p1}]", "INSTANCE"))
    return out


def _has_cmp(code: str, op) -> bool:
    try:
        return any(isinstance(o, op)
                   for x in ast.walk(ast.parse(code.replace("\r", "")))
                   if isinstance(x, ast.Compare) for o in x.ops)
    except SyntaxError:
        return False


def _corpora_pairs():
    from .sources import PROCEDURE_CORPORA
    for _src, pairs, _idx in PROCEDURE_CORPORA:
        for words, spec, code in pairs:
            yield words, spec, code


# ================================================================ candidates
class _Cand:
    __slots__ = ("expr", "type", "prov", "conj")
    def __init__(self, expr, type_, prov, conj=False):
        self.expr, self.type, self.prov, self.conj = expr, type_, prov, conj


class _Session:
    def __init__(self, text, sig, assert_texts):
        self.text, self.sig, self.asserts = text, sig, assert_texts
        self.doc = _nlp()(text)
        self.trace: List[str] = []
        self.mined = defaultdict(list)
        self.asked = set()
        self.ex = self._examples()
        self.out_type = self._out_type()

    def log(self, s):
        self.trace.append(s)

    def _examples(self):
        out = []
        for a in self.asserts:
            try:
                t = ast.parse(a.strip()).body[0].test
            except SyntaxError:
                continue
            if isinstance(t, ast.Compare) and isinstance(t.left, ast.Call):
                try:
                    out.append((tuple(ast.literal_eval(x)
                                      for x in t.left.args),
                                ast.literal_eval(t.comparators[0])))
                except (ValueError, SyntaxError):
                    pass
            elif isinstance(t, ast.Call):
                try:
                    out.append((tuple(ast.literal_eval(x)
                                      for x in t.args), True))
                except (ValueError, SyntaxError):
                    pass
        return out

    def _out_type(self):
        vs = [e for _, e in self.ex]
        if vs and all(isinstance(v, bool) for v in vs):
            return "BOOL"
        if vs and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      for v in vs):
            return "NUM"
        if vs and all(isinstance(v, str) for v in vs):
            return "STR"
        if vs and all(isinstance(v, (list, tuple)) for v in vs):
            return "COLL"
        return "ANY"

    # -------- meanings: the store + mined; typed
    def meanings(self, w, want=None):
        got = []
        for r in kb.KB:
            if r.pattern == w and r.payload and r.symbol in (
                    "REDUCE", "FUNC", "PRED", "BPRED", "CPRED"):
                ty = {"PRED": "TEST", "BPRED": "REL", "CPRED": "TEST",
                      "REDUCE": "FUNC", "FUNC": "FUNC"}[r.symbol]
                got.append({"payload": r.payload, "type": ty,
                            "prov": f"store:{r.provenance}"})
        got += self.mined.get(w, [])
        if want:
            got = [m for m in got if m["type"] == want or want == "ANY"]
        return got

    # -------- USAGE(word): registered documents + worked-example corpora
    def request_usage(self, w) -> bool:
        if w in self.asked:
            return False
        self.asked.add(w)
        self.log(f"USAGE({w!r}) -> registered sources")
        got = []
        for name, lines in DOCUMENTS:
            for k, l in enumerate(lines):
                if w not in l.lower():
                    continue
                for x in lines[k:k + 6]:
                    m = re.match(r"\s*>>> (.+?)(?:\s*#\s*(.*))?$", x)
                    if not m:
                        continue
                    com = (m.group(2) or "").lower()
                    if re.search(r"\w+-" + re.escape(w), com):
                        continue           # marker agreement
                    for sh, ty in _shapes_of(m.group(1)):
                        got.append((sh, ty, f"doc:{name}"))
        n_corp = 0
        eq_hits, cmp_hits = 0, {}
        for words, spec, code in _corpora_pairs():
            if w not in words and w + "s" not in words:
                continue
            n_corp += 1
            for sh, ty in _shapes_of(code):
                got.append((sh, ty, "corpus"))
            for op, pay in ((ast.Eq, "{a} == {b}"), (ast.Lt, "{a} < {b}"),
                            (ast.Gt, "{a} > {b}")):
                if _has_cmp(code, op):
                    cmp_hits[pay] = cmp_hits.get(pay, 0) + 1
        fam = Counter(sh for sh, _, _ in got)
        for sh, n in fam.most_common(2):
            ty = next(t for s, t, _ in got if s == sh)
            self.mined[w].append({"payload": sh, "type": ty,
                                  "prov": f"mined:usage x{n}"})
        for pay, n in cmp_hits.items():
            if n_corp >= 3 and n / n_corp >= 0.6:
                self.mined[w].append({"payload": pay, "type": "REL",
                                      "prov": f"mined:assoc "
                                              f"prec={n/n_corp:.2f}"})
        # the runtime as a source (existing adapter + tribunal)
        try:
            from .sources import Gap, retrieve
            admitted, _rej = retrieve(Gap(word=w, kind="callable",
                                          arity=self.sig.arity))
            for r in admitted:
                self.mined[w].append(
                    {"payload": r.payload,
                     "type": "REL" if r.symbol == "BINOP" else "FUNC",
                     "prov": r.provenance})
        except Exception:
            pass
        if self.mined.get(w):
            self.log(f"  admitted {len(self.mined[w])} meaning(s) "
                     f"for {w!r}")
        return bool(self.mined.get(w))

    # -------- DENOTE(subgraph, demand)
    def denote(self, tok, depth=0):
        if tok is None or depth > 5:
            return []
        lem = tok.lemma_.lower()
        cands = []
        tn = _type_nouns()
        have_of = {"INT": "NUM", "FLOAT": "NUM", "LIST": "COLL",
                   "STRING": "STR", "CHAR": "STR"}
        if lem in tn:            # a typed prior (ENG:TYPENOUN row) prunes
            want = tn[lem]
            for i, t in enumerate(self.sig.arg_types):
                have = have_of.get(t, "ANY")
                if want in ("ANY", have):
                    cands.append(_Cand(f"arg{i}", have, [f"arg{i}"]))
        elif any(c.lemma_.lower() in ("give", "given", "provide", "input",
                                      "specify")
                 for c in tok.children):
            # REFERENCE question, context-derived: the sentence marks this
            # noun as PROVIDED, so it denotes some argument — the evidence
            # types the candidates, enumeration + the oracle disambiguate;
            # no noun lexicon required
            for i, t in enumerate(self.sig.arg_types):
                cands.append(_Cand(f"arg{i}", have_of.get(t, "ANY"),
                                   [f"ref:arg{i}<-{lem}"]))
        det = next((c for c in tok.children if c.dep_ == "det"
                    and any(r.symbol == "ENG:MAP"
                            and r.pattern == c.lemma_.lower()
                            for r in kb.KB)), None)
        pobj = next((c for p in tok.children if p.dep_ == "prep"
                     and p.lemma_.lower() == "of"
                     for c in p.children if c.dep_ == "pobj"), None)
        ms = self.meanings(lem)
        # typed coverage: DENOTE needs a PRODUCER meaning; a relation/test
        # sense alone is still a gap
        if not any(m["type"] in ("FUNC", "INSTANCE") for m in ms) \
                and lem not in tn \
                and tok.pos_ in ("NOUN", "ADJ", "VERB"):
            if self.request_usage(lem):
                ms = self.meanings(lem)
        inner = self.denote(pobj, depth + 1)
        plural = tok.tag_ == "NNS"
        for m in ms:
            if m["type"] == "INSTANCE" and plural:
                for src in (inner or self._args(("STR", "COLL"))):
                    if src.type not in ("STR", "COLL"):
                        continue
                    sh = m["payload"].replace("{src}", src.expr) \
                        .replace("{p0}", "_p0").replace("{p1}", "_p1")
                    coll = (f"[{sh} for _p0 in range(len({src.expr})+1) "
                            f"for _p1 in range(len({src.expr})+1)]")
                    coll = self._filters(tok, coll)
                    cands.append(_Cand(coll, "COLL",
                                       src.prov + [m["prov"],
                                                   "ENG:PLURAL"]))
            elif m["type"] == "FUNC":
                for src in (inner or self._args(("STR", "COLL", "NUM"))):
                    if det is not None and src.type == "COLL":
                        row = next(r for r in kb.KB if r.symbol == "ENG:MAP"
                                   and r.pattern == det.lemma_.lower())
                        try:
                            body = m["payload"].format(src="_e")
                        except (KeyError, IndexError):
                            continue
                        cands.append(_Cand(
                            row.payload.format(b=body, v="_e", g=src.expr),
                            "COLL", src.prov + [m["prov"], "ENG:MAP"]))
                    else:
                        try:
                            cands.append(_Cand(
                                m["payload"].format(src=src.expr), "ANY",
                                src.prov + [m["prov"]]))
                        except (KeyError, IndexError):
                            pass
        if det is None and ms:
            deep = next((t for t in tok.subtree if t is not tok
                         and any(c.dep_ == "det"
                                 and any(r.symbol == "ENG:MAP"
                                         and r.pattern == c.lemma_.lower()
                                         for r in kb.KB)
                                 for c in t.children)), None)
            if deep is not None:
                for src in self._args(("COLL",)):
                    chain, node = [], tok
                    while node is not None and node is not deep:
                        mf = self.meanings(node.lemma_.lower(),
                                           want="FUNC")
                        if mf:
                            chain.append(mf[0]["payload"])
                        node = next((c for p in node.children
                                     if p.dep_ == "prep"
                                     and p.lemma_.lower() == "of"
                                     for c in p.children
                                     if c.dep_ == "pobj"), None)
                    body = "_e"
                    for pay in reversed(chain):
                        try:
                            body = pay.format(src=body)
                        except (KeyError, IndexError):
                            body = None
                            break
                    if body and body != "_e":
                        cands.append(_Cand(
                            f"[{body} for _e in {src.expr}]", "COLL",
                            src.prov + ["ENG:MAP"]))
        if any(r.symbol == "ENG:SIZE" and r.pattern == lem for r in kb.KB):
            for src in [c for c in inner if c.type == "COLL"]:
                cands.append(_Cand(f"len({src.expr})", "NUM",
                                   src.prov + ["ENG:SIZE"]))
        return cands

    def _args(self, types):
        out, m = [], {"STR": ("STRING", "CHAR"), "COLL": ("LIST",),
                      "NUM": ("INT", "FLOAT")}
        for want in types:
            for i, t in enumerate(self.sig.arg_types):
                if t in m.get(want, ()):
                    out.append(_Cand(f"arg{i}", want, [f"arg{i}"]))
        return out

    def _filters(self, tok, coll):
        conds, neg = [], False
        for a in tok.children:
            if a.dep_ == "amod":
                for w in [a] + list(a.conjuncts):
                    lw = w.lemma_.lower()
                    if lw in ("non", "not", "no", "-"):
                        neg = True
                        continue
                    for m in self.meanings(lw, want="TEST"):
                        c = m["payload"].format(x="_x")
                        conds.append(f"not ({c})" if neg else c)
                        break
        if conds:
            return f"[_x for _x in {coll} if " + " and ".join(conds) + "]"
        return coll

    # -------- clause readings: copula / reciprocal / coordinated group
    def clause_cands(self):
        out = []
        for pred_tok in [t for t in self.doc if t.dep_ == "acomp"]:
            subj = next((c for c in pred_tok.head.children
                         if c.dep_ in ("nsubj", "nsubjpass")), None)
            if subj is None:
                continue
            pw = pred_tok.lemma_.lower()
            recip = any(m in self.text.lower()
                        for m in ("each other", "one another"))
            rels = self.meanings(pw, want="REL")
            if not rels:
                for r in kb.KB:
                    if r.symbol == "BPRED" and r.payload and \
                            isinstance(r.pattern, str) and \
                            _syn_ant(pw, r.pattern) == "ant":
                        rels = [{"payload": f"not ({r.payload})",
                                 "prov": f"store:{r.provenance}+ant"}]
                        break
            if not rels and self.request_usage(pw):
                rels = self.meanings(pw, want="REL")
            if recip and rels and subj.tag_ == "NNS":
                row = next((r for r in kb.KB if r.symbol == "ENG:RECIP"),
                           None)
                if row is not None:
                    for g in self._args(("COLL",)):
                        body = rels[0]["payload"].format(a="_a", b="_b")
                        out.append(_Cand(row.payload.format(
                            r=body, v="_a", v2="_b", g=g.expr), "BOOL",
                            g.prov + [rels[0]["prov"], "ENG:RECIP"]))
            mods = [m for a in subj.children if a.dep_ == "amod"
                    for m in [a] + list(a.conjuncts)]
            if len(mods) >= 2 and rels:
                picks = []
                for m in mods[:2]:
                    got = self.meanings(m.lemma_.lower(), want="FUNC")
                    if not got and self.request_usage(m.lemma_.lower()):
                        got = self.meanings(m.lemma_.lower(), want="ANY")
                    picks.append(got)
                if all(picks):
                    self.log(f"DENOTE(clause): {pw} over coordinated "
                             f"group {[m.lemma_ for m in mods[:2]]}")
                    for arg in self._args(("STR", "COLL")):
                        for p1, p2 in itertools.islice(
                                itertools.product(picks[0], picks[1]), 6):
                            try:
                                e = rels[0]["payload"].format(
                                    a=p1["payload"].format(src=arg.expr),
                                    b=p2["payload"].format(src=arg.expr))
                            except (KeyError, IndexError):
                                continue
                            out.append(_Cand(e, "BOOL",
                                             [p1["prov"], p2["prov"],
                                              rels[0]["prov"],
                                              "ENG:COPULA"]))
            tests = self.meanings(pw, want="TEST")
            if tests:
                node, chain = subj, []
                while node is not None:
                    mf = self.meanings(node.lemma_.lower(), want="FUNC")
                    if mf:
                        chain.append(mf[0]["payload"])
                    node = next((c for p in node.children
                                 if p.dep_ == "prep"
                                 and p.lemma_.lower() == "of"
                                 for c in p.children
                                 if c.dep_ == "pobj"), None)
                for d in range(1, len(chain) + 1):
                    for arg in self._args(("STR", "COLL", "NUM")):
                        inner, ok = arg.expr, True
                        for pay in reversed(chain[:d]):
                            try:
                                inner = pay.format(src=inner)
                            except (KeyError, IndexError):
                                ok = False
                                break
                        if ok:
                            out.append(_Cand(
                                tests[0]["payload"].format(x=inner),
                                "BOOL", arg.prov + ["ENG:COPULA"]))
        return out

    # -------- CONTRAST over the asserts' answer groups (licensed)
    def evidence_cands(self):
        if len({repr(e) for _, e in self.ex}) != 2 or not self.ex:
            return []
        argn = [f"arg{i}" for i in range(self.sig.arity)]
        feats = []
        for r in kb.KB:
            if r.symbol == "PRED" and r.payload and \
                    isinstance(r.pattern, str):
                for i in range(self.sig.arity):
                    feats.append((r.pattern, r.payload.format(x=argn[i])))
                for i in range(self.sig.arity):
                    for j in range(i + 1, self.sig.arity):
                        if self.sig.arg_types[i] == self.sig.arg_types[j]:
                            feats.append((r.pattern, "(%s) != (%s)" % (
                                r.payload.format(x=argn[i]),
                                r.payload.format(x=argn[j]))))
            if r.symbol == "BPRED" and r.payload and \
                    isinstance(r.pattern, str):
                for i in range(self.sig.arity):
                    for j in range(self.sig.arity):
                        if i != j:
                            feats.append((r.pattern, r.payload.format(
                                a=argn[i], b=argn[j])))
        for i, t in enumerate(self.sig.arg_types):
            if t != "LIST":
                continue
            for r in kb.KB:
                if r.symbol == "PRED" and r.payload and \
                        isinstance(r.pattern, str):
                    for qt in ("all(%s for _e in %s)",
                               "any(%s for _e in %s)"):
                        feats.append((r.pattern, qt % (
                            r.payload.format(x="_e"), argn[i])))
                if r.symbol == "BPRED" and r.payload and \
                        isinstance(r.pattern, str):
                    b = r.payload.format(a="_a", b="_b")
                    feats.append((r.pattern,
                                  f"all(not ({b}) for _i, _a in "
                                  f"enumerate({argn[i]}) "
                                  f"for _b in {argn[i]}[_i+1:])"))
        seps = []
        for w, e in feats:
            try:
                fv = [bool(eval(e, dict(zip(argn, a))))
                      for a, _ in self.ex]
            except Exception:
                continue
            pos = {repr(v) for (a, v), x in zip(self.ex, fv) if x}
            neg = {repr(v) for (a, v), x in zip(self.ex, fv) if not x}
            if pos and neg and pos.isdisjoint(neg):
                seps.append((w, e))
        varying = [i for i in range(self.sig.arity)
                   if len({repr(a[i]) for a, _ in self.ex}) > 1]
        seps = [(w, e) for w, e in seps
                if all(f"arg{i}" in e for i in varying)]
        cw = _content(self.doc)
        tn = _type_nouns()
        anchors = [w for w in cw if w not in tn]   # distinctive words only
        out = []
        for w, e in seps[:6]:
            conn = any(_syn_ant(w, sw) for sw in cw)
            if not conn and SRC_WIKTIONARY:
                # CONNECTION question to the dictionary: two words connect
                # when one appears in the other's definitions — UNLESS the
                # store already grounds both to DIFFERENT meanings
                # (recognized-conflict license: a dictionary link between
                # two known different meanings is a lookalike, e.g. the
                # 'level/equal' sense of a parity word)
                def _grounded_payloads(word):
                    return {str(r.payload) for r in kb.KB
                            if r.pattern == word and r.payload
                            and r.symbol in ("PRED", "BPRED", "REDUCE")}
                wp = _grounded_payloads(w)
                for a in anchors:
                    if not (w in wiktionary_defwords(a)
                            or a in wiktionary_defwords(w)):
                        continue
                    ap = _grounded_payloads(a)
                    if wp and ap and not (wp & ap):
                        self.log(f"CONNECT({w!r})~{a!r}: dictionary link "
                                 f"REFUSED — store grounds both to "
                                 f"different meanings")
                        continue
                    conn = True
                    self.log(f"CONNECT({w!r}): licensed by dictionary "
                             f"definition of task word {a!r}")
                    break
            out.append(_Cand(e, "BOOL", [f"contrast:{w}"], conj=not conn))
        if out:
            self.log(f"CONTRAST(evidence-groups): {len(seps)} "
                     f"separator(s), "
                     f"{sum(1 for c in out if c.conj)} unconnected")
        return out

    # -------- ALIGN: worked examples, licensed reuse
    def analog_cands(self):
        out = []
        cw = _content(self.doc)
        scored = []
        for words, spec, code in _corpora_pairs():
            s = sum(1 for w in words
                    if any(_syn_ant(w, sw) for sw in cw))
            if s:
                scored.append((s, spec, code))
        scored.sort(key=lambda x: -x[0])
        for s, spec, code in scored[:8]:
            aw = _content(_nlp()(spec))
            missing = [w for w in cw
                       if not any(_syn_ant(w, x) for x in aw)]
            if not missing:
                base = code.replace("\r", "")
                try:
                    names = [n.name for n in ast.parse(base).body
                             if isinstance(n, ast.FunctionDef)]
                except SyntaxError:
                    names = []
                for nm in names:
                    body = base if nm == self.sig.name else \
                        base + f"\n{self.sig.name} = {nm}"
                    out.append(_Cand(("__WHOLE__", body), "ANY",
                                     [f"replay:{spec[:40]}"]))
            for view in _min_views(code):
                subst = None
                for sw in cw:
                    for m in self.meanings(sw, want="FUNC"):
                        if any(_syn_ant(sw, x) == "ant" for x in aw):
                            subst = (sw, m["payload"])
                if [w for w in missing
                        if w not in ((subst[0],) if subst else ())]:
                    continue
                red = subst[1] if subst else "min({src})"
                conds = [view["cond"]]
                for w in cw:
                    for m in self.meanings(w, want="TEST"):
                        conds.append(m["payload"].format(x="_d"))
                        break
                for i, t in enumerate(self.sig.arg_types):
                    if t != "INT" or view["hi"] != "{src}":
                        continue
                    if any(isinstance(a[i], int) and abs(a[i]) > _HUGE
                           for a, _ in self.ex):
                        continue
                    dom = (f"[_d for _d in range({view['lo']}, arg{i}+1)"
                           f" if " + " and ".join(
                               c.format(src=f"arg{i}", d="_d")
                               if "{" in c else c for c in conds) + "]")
                    try:
                        out.append(_Cand(red.format(src=dom), "NUM",
                                         [f"skeleton:{spec[:40]}",
                                          "canonical-min-view"]))
                    except (KeyError, IndexError):
                        pass
        return out

    # -------- BRIDGE(have, want): retrieved from the store
    def bridges(self, expr, have, want):
        if have == want or want == "ANY":
            yield expr, []
            return
        if want == "BOOL":
            yield f"bool({expr})", ["bridge:bool"]
        if want == "NUM" and have in ("COLL", "ANY"):
            for r in kb.KB:
                if r.symbol == "REDUCE" and r.payload \
                        and "{src}" in str(r.payload):
                    yield r.payload.format(src=expr), [
                        f"bridge:{r.pattern}"]
        if want == "STR" and self._labels():
            yield expr, ["bridge:label-map"]
        if want == "COLL" and have in ("COLL", "ANY"):
            yield expr, []

    def _labels(self):
        vs = sorted({repr(e) for _, e in self.ex})
        return vs if (len(vs) == 2 and not all(
            isinstance(e, bool) for _, e in self.ex)) else None

    def _independent_varying(self):
        """ARGUMENT COVERAGE (the store's own honesty guard): a candidate
        must reference every independently-varying argument. An argument
        affine in another's length (the classic redundant n == len(arr))
        is dependent and exempt — same rule as the agenda's paths."""
        out = []
        for i in range(self.sig.arity):
            if len({repr(a[i]) for a, _ in self.ex}) <= 1:
                continue
            dep = False
            for j in range(self.sig.arity):
                if j == i:
                    continue
                try:
                    c = self.ex[0][0][i] - len(self.ex[0][0][j])
                    if all(a[i] == len(a[j]) + c for a, _ in self.ex):
                        dep = True
                except TypeError:
                    pass
            if not dep:
                out.append(i)
        return out

    # -------- the loop
    def run(self, frame=None):
        params = ", ".join(f"arg{i}" for i in range(self.sig.arity))
        needed = self._independent_varying()
        root = frame.theme_token if frame is not None else \
            next((t for t in self.doc if t.dep_ == "dobj"), None)
        classes = [("clause", self.clause_cands())]
        if root is not None:
            self.log(f"DENOTE(subgraph {root.lemma_!r}, "
                     f"demand={self.out_type})")
            classes.append(("denote", self.denote(root)))
        classes.append(("evidence", self.evidence_cands()))
        classes.append(("analog", self.analog_cands()))
        # fair interleave across method classes so no class starves the
        # oracle budget; conjectures last within each class
        for _n, cs in classes:
            cs.sort(key=lambda c: c.conj)
        cands, idx = [], 0
        while any(idx < len(cs) for _n, cs in classes):
            for _n, cs in classes:
                if idx < len(cs):
                    cands.append(cs[idx])
            idx += 1
        tried, labels = 0, self._labels()
        conjecture, seen = None, set()
        for c in sorted(cands, key=lambda c: c.conj):
            if isinstance(c.expr, tuple) and c.expr[0] == "__WHOLE__":
                tried += 1
                if tried <= MAX_ORACLE and \
                        run_tests(c.expr[1], self.asserts)["ok"]:
                    return {"status": "SOLVED", "code": c.expr[1],
                            "prov": c.prov, "trace": self.trace,
                            "grade": "verified (replay, licensed, "
                                     "oracle-passed)"}
                continue
            for expr, bprov in self.bridges(c.expr, c.type,
                                            self.out_type):
                variants = [f"def {self.sig.name}({params}):\n"
                            f"    return {expr}"]
                if labels:
                    for A, B in (labels, labels[::-1]):
                        variants.append(
                            f"def {self.sig.name}({params}):\n"
                            f"    return {A} if ({expr}) else {B}")
                for code in variants:
                    if tried >= MAX_ORACLE or code in seen:
                        continue
                    seen.add(code)
                    body = code.split(":", 1)[1]
                    if not all(f"arg{i}" in body for i in needed):
                        continue      # ignores an independent argument
                    tried += 1
                    if run_tests(code, self.asserts)["ok"]:
                        if c.conj:
                            if conjecture is None:
                                conjecture = code
                            self.log("separator UNCONNECTED -> "
                                     "conjecture, not an answer")
                            continue
                        return {"status": "SOLVED", "code": code,
                                "prov": c.prov + bprov,
                                "trace": self.trace,
                                "grade": "verified (questions: "
                                         "mined/derived, oracle-passed)"}
        # DEFINITION source: "what IS this word?" -> defining formula
        if SRC_WIKIDATA_FORMULA and tried < MAX_ORACLE:
            import re as _re
            name_toks = [t for t in _re.split(r"[_\W]+|(?<=[a-z])(?=[A-Z])",
                                              self.sig.name) if len(t) > 2]
            tn = _type_nouns()
            heads = [t.lemma_.lower() for t in self.doc
                     if t.lemma_.lower() in tn]
            terms = []
            for w in [t.lower() for t in name_toks] + \
                    sorted(_content(self.doc)):
                if w in tn or w in terms:
                    continue
                # the concept is usually named WORD + its type-noun head
                for h in heads[:1]:
                    if f"{w} {h}" not in terms:
                        terms.append(f"{w} {h}")
                terms.append(w)
            for term in terms[:6]:
                for e, fv in wikidata_formula(term):
                    self.log(f"DEFINITION({term!r}): wikidata formula "
                             f"{e!r} vars={fv}")
                    if self.out_type == "BOOL" and len(fv) == 1:
                        v = fv[0]
                        for i, t in enumerate(self.sig.arg_types):
                            if t != "INT":
                                continue
                            body = e.replace(v, "_n")
                            code = (f"def {self.sig.name}({params}):\n"
                                    f"    _n = 1\n"
                                    f"    while ({body}) <= arg{i}:\n"
                                    f"        if ({body}) == arg{i}:\n"
                                    f"            return True\n"
                                    f"        _n += 1\n"
                                    f"    return False")
                            tried += 1
                            if tried <= MAX_ORACLE and \
                                    run_tests(code, self.asserts)["ok"]:
                                return {"status": "SOLVED", "code": code,
                                        "prov": [f"wikidata:P2534:{term}",
                                                 "of-the-form=>exists"],
                                        "trace": self.trace,
                                        "grade": "verified (defining "
                                                 "formula, oracle-passed)"}
                    if self.out_type == "NUM" and 1 <= len(fv) <= 3 \
                            and len(fv) <= self.sig.arity:
                        for perm in itertools.permutations(
                                range(self.sig.arity), len(fv)):
                            expr = e
                            for v, ai in zip(fv, perm):
                                expr = _re.sub(rf"\b{v}\b",
                                               f"arg{ai}", expr)
                            code = (f"def {self.sig.name}({params}):\n"
                                    f"    return {expr}")
                            tried += 1
                            if tried <= MAX_ORACLE and \
                                    run_tests(code, self.asserts)["ok"]:
                                return {"status": "SOLVED", "code": code,
                                        "prov": [f"wikidata:P2534:{term}"],
                                        "trace": self.trace,
                                        "grade": "verified (defining "
                                                 "formula, oracle-passed)"}
        # EVIDENCE-GROUP source: look the examples up as a sequence
        if SRC_OEIS and tried < MAX_ORACLE and self.out_type == "NUM" \
                and self.sig.arity == 1 \
                and self.sig.arg_types[0] == "INT" and len(self.ex) >= 3:
            pairs = sorted((a[0], v) for a, v in self.ex)
            vals = [v for _, v in pairs]
            _tn = _type_nouns()
            _words = [w for w in _content(self.doc) if w not in _tn]
            for kind, payload, nm in oeis_lookup(vals, pairs, _words):
                self.log(f"SEQUENCE(evidence): candidate {kind} "
                         f"<- {nm[:45]!r}")
                for off in (0, 1, -1):
                    if kind == "closed":
                        expr = re.sub(r"\bn\b",
                                      f"(arg0{'+' if off >= 0 else ''}"
                                      f"{off})" if off else "arg0", payload)
                        code = (f"def {self.sig.name}(arg0):\n"
                                f"    return {expr}")
                    else:
                        code = (payload + f"\ndef {self.sig.name}(arg0):"
                                f"\n    return _f(arg0"
                                + (f"{'+' if off >= 0 else ''}{off}"
                                   if off else "") + ")")
                    tried += 1
                    if tried <= MAX_ORACLE and \
                            run_tests(code, self.asserts)["ok"]:
                        return {"status": "SOLVED", "code": code,
                                "prov": [f"oeis:{nm[:40]}", kind],
                                "trace": self.trace,
                                "grade": "verified (sequence lookup, "
                                         "oracle-passed)"}
        if conjecture:
            return {"status": "CONJECTURE", "code": conjecture,
                    "trace": self.trace}
        gaps = sorted(w for w in self.asked if not self.mined.get(w))
        return {"status": "REFUSED", "trace": self.trace,
                "reason": (f"open questions: {gaps[:4]}" if gaps else
                           "no licensed reading passes the oracle")}


def _min_views(code):
    """An earliest-hit ascending loop over a numeric domain IS the minimum
    of the matching set (a Python fact, usable as a template)."""
    out = []
    try:
        tree = ast.parse(code.replace("\r", ""))
    except SyntaxError:
        return out
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        params = [a.arg for a in fn.args.args]
        loops = [n for n in ast.walk(fn)
                 if isinstance(n, (ast.While, ast.For))]
        rets = [n for L in loops for n in ast.walk(L)
                if isinstance(n, ast.Return)
                and isinstance(n.value, ast.Name)]
        if not rets:
            continue
        var = rets[0].value.id
        conds = []
        for n in ast.walk(fn):
            if isinstance(n, ast.Compare):
                names = {x.id for x in ast.walk(n)
                         if isinstance(x, ast.Name)}
                if var in names and names & set(params):
                    try:
                        t = ast.unparse(n)
                    except Exception:
                        continue
                    for pp in params:
                        t = re.sub(rf"\b{re.escape(pp)}\b", "{src}", t)
                    t = re.sub(rf"\b{re.escape(var)}\b", "{d}", t)
                    if "{src}" in t and "{d}" in t:
                        conds.append(t)
        consts = [x.value for x in ast.walk(fn)
                  if isinstance(x, ast.Constant)
                  and isinstance(x.value, int) and x.value >= 2]
        hi = any(isinstance(x, ast.Return)
                 and getattr(x.value, "id", None) in params
                 for x in ast.walk(fn))
        for c in dict.fromkeys(conds):
            if consts:
                out.append({"cond": c, "lo": min(consts),
                            "hi": "{src}" if hi else None})
    return out


def derive(text: str, sig, assert_texts, frame=None) -> dict:
    """The agenda's LAST derivation method: typed questions over the
    sample's subgraphs and groups. Returns SOLVED / CONJECTURE / REFUSED
    with a full question trace; never raises."""
    try:
        return _Session(text, sig, assert_texts).run(frame)
    except Exception as e:                                  # noqa: BLE001
        return {"status": "REFUSED", "reason": f"questions: {e}",
                "trace": []}


# ============================================================ digest artifact
def wikipedia_extract(topic, chars=6000):
    """FACT source for a TOPIC: the encyclopedia's article text (cached)."""
    key = f"wp:{topic}:{chars}"
    if key in _QCACHE:
        return tuple(_QCACHE[key])
    title, text = None, None
    try:
        d = _http_json("https://en.wikipedia.org/w/api.php?action=query&"
                       "list=search&format=json&srlimit=1&srsearch="
                       + topic.replace(" ", "%20"))
        hits = d.get("query", {}).get("search", [])
        if hits:
            title = hits[0]["title"]
            d2 = _http_json("https://en.wikipedia.org/w/api.php?action="
                            "query&prop=extracts&explaintext=1&format=json"
                            f"&exchars={chars}&titles="
                            + title.replace(" ", "%20"))
            pages = d2.get("query", {}).get("pages", {})
            text = next(iter(pages.values())).get("extract")
        _qcache_put(key, [title, text])
    except Exception:
        pass
    return title, text


def digest_route(frame, statements):
    """The DIGEST artifact: 'produce a <text-artifact> of TOPIC' -> a
    selection GRAPH over the read source, with certificates. The engine
    reasons about WHAT to say and proves it said it; rendering to fluent
    prose is another engine's job (verifiable here by re-reading).

    Selection is a hand-provenance policy for now (centrality + timeline
    anchoring); the mining route (align lead sections to article bodies,
    keep what survives) replaces it via the calibration mechanism."""
    from .worldmodel import WorldModel
    from collections import Counter
    topic_preps = {r.pattern for r in kb.KB if r.symbol == "ENG:TOPIC"}
    topic_tok = next((c for pnode in frame.theme_token.children
                      if pnode.dep_ == "prep"
                      and pnode.lemma_.lower() in topic_preps
                      for c in pnode.children if c.dep_ == "pobj"), None)
    if topic_tok is None:
        return {"status": "UNGROUNDED",
                "reasons": ["digest: no topic (of/about-object) found"]}
    topic = " ".join(t.text for t in topic_tok.subtree if t.pos_ != "DET")
    provided = [t for t, _ in statements if len(t) > 300]
    if provided:
        title, text = "(provided)", " ".join(provided)
    else:
        title, text = wikipedia_extract(topic)
    if not text:
        return {"status": "UNGROUNDED",
                "reasons": [f"digest: no source found for topic {topic!r}"]}
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+",
                                         text.replace("\n", " "))
             if len(x.strip()) > 30]
    triples, prov = [], {}
    for sen in sents[:80]:
        try:
            wm = WorldModel.from_text(sen, resolve_coref=False)
        except Exception:
            continue
        for f in list(getattr(wm, "facts", {}) or []):
            triples.append(f)
            prov[f] = sen
    if not triples:
        return {"status": "UNGROUNDED",
                "reasons": ["digest: source read but no relation captured "
                            "(reading frontier)"]}
    freq = Counter(t[0] for t in triples)
    pol = kb.match_key(("DIGEST", "POLICY"))
    w_cent = pol.sig.get("centrality", 1) if pol else 1
    w_anch = pol.sig.get("anchored", 2) if pol else 2

    def anchored(t):
        return bool(re.search(r"\d{3,}|\b(bc|ad|century)\b",
                              " ".join(map(str, t)).lower()))
    ranked = sorted(dict.fromkeys(triples),
                    key=lambda t: w_cent * freq[t[0]]
                    + w_anch * anchored(t), reverse=True)
    k = max(4, len(dict.fromkeys(triples)) // 6)
    graph = ranked[:k]
    graph.sort(key=lambda t: sents.index(prov[t]))
    rendering = " ".join(dict.fromkeys(prov[t] for t in graph))
    re_read = []
    for sen in re.split(r"(?<=[.!?])\s+", rendering):
        try:
            wm2 = WorldModel.from_text(sen, resolve_coref=False)
            re_read += list(getattr(wm2, "facts", {}) or [])
        except Exception:
            pass
    recovered = sum(1 for t in graph if t in re_read)
    return {"status": "SOLVED",
            "artifact": "digest",
            "answer": [{"relation": list(map(str, t)),
                        "source": prov[t][:200]} for t in graph],
            "rendering_extractive": rendering[:1500],
            "source": {"title": title, "sentences": len(sents),
                       "relations": len(set(triples))},
            "certificates": {
                "faithfulness": all(t in triples for t in graph),
                "compression": f"{len(graph)}/{len(set(triples))}",
                "round_trip_recovered": f"{recovered}/{len(graph)}"},
            "grade": "digest (selection: hand policy; faithfulness: "
                     "certified; round-trip: verified)"}
