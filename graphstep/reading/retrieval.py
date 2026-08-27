#!/usr/bin/env python3
"""Retrieval-Augmented Deduction: fill PRECISE knowledge gaps from external
knowledge bases, admitted as tiered symbols — never absorbed as text.

    ask(q) = Unknown
      -> GAP DETECTOR   backward-chain: which premises/terms are missing
      -> SOURCES        WordNet (hypernym chains -> subsumption rules),
                        Wikidata (instance-of / subclass-of), pluggable
      -> ADMISSION      dedup; reject on conflict with higher-tier facts;
                        generic terms only (named individuals opt-in)
      -> TIER           'retrieved:<source>' — text authority always wins
      -> closure        answers stay True/False; provenance is attached
                        and every proof quotes its sources

Budget: per-call term cap, depth cap, cross-call cache. Zero LLM.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

from .worldmodel import Rule, Literal, norm_ent

_CACHE: Dict[tuple, list] = {}


# ------------------------------------------------------------------ gaps
def find_gaps(wm, query: str, depth: int = 2,
              link_entities: bool = False) -> List[str]:
    """Terms whose background knowledge could close the query.
    Backward idea: the query's predicate, plus the types/attributes the
    query subject is known to have (each is a hook a KB rule could attach
    to), expanded along existing rule conclusions up to `depth`."""
    from .worldmodel import parse_literal
    lit = parse_literal(query)
    if lit is None:
        return []
    terms = []
    if link_entities and lit.subj and not lit.subj.startswith("?"):
        terms.append(lit.subj)                # named entity: OPT-IN only
    if lit.pred and not lit.pred.startswith("?"):
        terms.append(lit.pred)
    for (s, p, o), (pos, _) in wm.facts.items():
        if pos and s == lit.subj and o is None:
            terms.append(p)                     # known types of the subject
    seen, frontier = set(terms), list(terms)
    for _ in range(depth - 1):                  # follow stated-rule heads
        nxt = []
        for r in wm.rules:
            if any(c.pred in seen for c in r.conds) \
                    and r.concl.pred not in seen:
                seen.add(r.concl.pred)
                nxt.append(r.concl.pred)
        frontier = nxt
    return list(dict.fromkeys(terms + sorted(seen - set(terms))))


# ------------------------------------------------------------------ sources
def _ensure_wordnet():
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("dog")
        return wn
    except LookupError:
        import ssl
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except Exception:
            pass
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn
        return wn


def wordnet_rules(term: str, max_hops: int = 8) -> List[Rule]:
    """Hypernym chain -> subsumption rules: lemur=>primate=>mammal=>..."""
    key = ("wordnet", term)
    if key in _CACHE:
        return _CACHE[key]
    from .kbstore import store
    hit = store().get("wordnet", term)
    if hit is not None:
        _CACHE[key] = hit
        return hit
    out: List[Rule] = []
    try:
        wn = _ensure_wordnet()
        synsets = wn.synsets(term.replace("_", " "), pos=wn.NOUN)
        if synsets:
            s = synsets[0]
            cur = norm_ent(term)
            for _ in range(max_hops):
                hypers = s.hypernyms()
                if not hypers:
                    break
                s = hypers[0]
                nxt = norm_ent(s.lemmas()[0].name())
                out.append(Rule([Literal(True, "?", cur, None)],
                                Literal(True, "?", nxt, None),
                                f"[retrieved:wordnet] every {cur} "
                                f"is a {nxt}"))
                cur = nxt
    except Exception:
        pass
    _CACHE[key] = out
    store().put("wordnet", term, out)
    return out


def wikidata_knowledge(term: str, max_values: int = 3,
                       max_props: int = 12) -> List:
    """GENERALIZED Wikidata adapter: fetch ALL truthy item-valued claims and
    use each property's own LABEL as the predicate — no hardcoded property
    ids. Class-like properties additionally yield unary type facts / rules:

      instance of / subclass of  -> subsumption RULES  (term(x) => class(x))
      CLASSY properties (occupation, position held) -> also value(term)
      everything else -> binary FACT  property_label(term, value_label)
    """
    key = ("wikidata", term)
    if key in _CACHE:
        return _CACHE[key]
    from .kbstore import store
    hit = store().get("wikidata", term)
    if hit is not None:
        _CACHE[key] = hit
        return hit

    CLASS_PROPS = {"instance of", "subclass of"}
    CLASSY_PROPS = {"occupation", "position held"}
    out: List = []
    try:
        import requests, urllib.parse

        def api(params):
            return requests.get(
                "https://www.wikidata.org/w/api.php?format=json&" + params,
                headers={"User-Agent": "graphstep"}, timeout=8).json()

        s = api("action=wbsearchentities&language=en&limit=1&search="
                + urllib.parse.quote(term.replace("_", " ")))
        if s.get("search"):
            qid = s["search"][0]["id"]
            claims = api(f"action=wbgetentities&props=claims&ids={qid}"
                         )["entities"][qid]["claims"]
            triples = []                      # (prop_id, value_qid)
            for pid, cs in list(claims.items())[:60]:
                for c in cs[:max_values]:
                    snak = c.get("mainsnak", {})
                    if snak.get("datatype") == "wikibase-item":
                        try:
                            triples.append(
                                (pid, snak["datavalue"]["value"]["id"]))
                        except Exception:
                            pass
            ids = list(dict.fromkeys(
                [p for p, _ in triples] + [v for _, v in triples]))
            labels = {}
            for i in range(0, len(ids), 50):
                got = api("action=wbgetentities&props=labels&languages=en"
                          "&ids=" + "|".join(ids[i:i + 50]))
                for k, v in got.get("entities", {}).items():
                    labels[k] = v.get("labels", {}).get("en", {}).get("value")
            n_props = 0
            seen_props = set()
            for pid, vid in triples:
                plab, vlab = labels.get(pid), labels.get(vid)
                if not plab or not vlab or len(vlab.split()) > 4:
                    continue
                if pid not in seen_props:
                    seen_props.add(pid)
                    n_props += 1
                    if n_props > max_props:
                        break
                pred, val = norm_ent(plab), norm_ent(vlab)
                subj = norm_ent(term)
                src_tag = f"[retrieved:wikidata:{qid}]"
                if plab in CLASS_PROPS:
                    out.append(Rule([Literal(True, "?", subj, None)],
                                    Literal(True, "?", val, None),
                                    f"{src_tag} every {subj} is a {val}"))
                else:
                    out.append(Literal(True, subj, pred, val,
                                       f"{src_tag} {pred}({subj}, {val})"))
                    if plab in CLASSY_PROPS:
                        out.append(Literal(True, subj, val, None,
                                           f"{src_tag} {subj} is a {val}"))
    except Exception:
        pass
    _CACHE[key] = out
    store().put("wikidata", term, out)
    return out


SOURCES = {"wordnet": wordnet_rules,
           "wikidata": wikidata_knowledge}


# ------------------------------------------------------------------ admission
def admit(wm, candidates: List[Rule]) -> Tuple[List[Rule], List[str]]:
    """Dedup against every known rule; reject candidates whose conclusion
    contradicts a higher-tier stated fact for any subject that satisfies
    the premise (text authority wins). Returns (admitted, rejections)."""
    existing = {(tuple(sorted(repr(c) for c in r.conds)), repr(r.concl))
                for r in wm.rules + getattr(wm, "retrieved_rules", [])
                + getattr(wm, "induced_impl", [])}
    admitted, rejected = [], []
    for r in candidates:
        key = (tuple(sorted(repr(c) for c in r.conds)), repr(r.concl))
        if key in existing:
            continue
        conflict = None
        for (s, p, o), (pos, _) in wm.facts.items():
            if o is None and pos and any(
                    c.pred == p and c.obj is None for c in r.conds):
                hit = wm.facts.get((s, r.concl.pred, None))
                if hit is not None and hit[0] != r.concl.pos:
                    conflict = (s, hit[1])
                    break
        if conflict:
            rejected.append(f"REJECTED {r.origin}: the text says otherwise "
                            f"for '{conflict[0]}' ({conflict[1]})")
        else:
            existing.add(key)
            admitted.append(r)
    return admitted, rejected
