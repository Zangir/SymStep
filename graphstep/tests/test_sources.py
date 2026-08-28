#!/usr/bin/env python3
"""Gap-driven retrieval CI: sources answer gaps with typed rows; the
tribunal enforces trust ranks; employment stays oracle-gated; unknown
words refuse honestly. May never regress."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from graphstep.reading.sources import Gap, retrieve, tribunal, rank
from graphstep.reading import sources
from graphstep.reading import kb
from graphstep.agenda import solve_loop
import graphstep.unified as unified


def test_introspection_finds_real_callables():
    adm, _ = retrieve(Gap("gcd", arity=2))
    provs = [r.provenance for r in adm]
    assert any("introspected:math.gcd" in p for p in provs), provs
    assert all(p.startswith("introspected:") for p in provs)
    print("ok introspection_finds_real_callables")


def test_tribunal_hand_rows_win():
    cand = kb.Row("sum", "REDUCE", payload="WRONG({src})",
                  provenance="web:somewhere", confidence=0.1)
    adm, rej = tribunal(Gap("sum"), [cand])
    assert not adm and rej, (adm, rej)
    assert "higher-trust" in rej[0]
    assert rank("hand") > rank("introspected:math.gcd") > rank("web:x")
    print("ok tribunal_hand_rows_win")


def test_gap_solves_via_retrieval_and_oracle():
    rec = solve_loop({
        "text": "Write a python function to find the gcd of two numbers.",
        "test_list": ["assert g(12, 8) == 4", "assert g(7, 5) == 1",
                      "assert g(20, 5) == 5"]})
    assert rec["status"] == "SOLVED"
    # either retrieved in this call, or already admitted earlier in the
    # session (knowledge persists per session by design)
    retrieved_now = any("RETRIEVE: admitted" in l
                        for l in rec["loop"]["trace"])
    already = any(r.provenance.startswith("introspected:math.gcd")
                  for r in kb.KB if r.pattern == "gcd")
    assert retrieved_now or already
    assert "math').gcd" in rec["code"]
    print("ok gap_solves_via_retrieval_and_oracle")


def test_unknown_word_refuses_honestly():
    rec = solve_loop({
        "text": "Write a function to find the frobnication of a number.",
        "test_list": ["assert f(1) == 99", "assert f(2) == 98"]})
    assert rec["status"] != "SOLVED"
    print("ok unknown_word_refuses_honestly")


def test_knowledge_retrieval_graded_and_optin():
    rec = unified.solve({"text": "Rex is a wolf."},
                        question="Rex is a canine.")
    assert rec["answer"] == "Unknown", "must stay Unknown while opt-out"
    unified.enable_knowledge_retrieval()
    try:
        rec = unified.solve({"text": "Rex is a wolf."},
                            question="Rex is a canine.")
        assert rec["answer"] == "True"
        assert "likely" in rec.get("grade", ""), rec.get("grade")
        assert "wordnet" in rec.get("grade", "")
        assert "Rex is a wolf." in rec.get("proof", "")
    finally:
        unified.KNOWLEDGE_RETRIEVAL = False
    print("ok knowledge_retrieval_graded_and_optin")


def test_runtime_plugged_new_domain_adapter():
    """THE agnosticism gate: a knowledge source for a domain the system has
    never seen, plugged at runtime with zero core edits, must flow through
    gap -> tribunal -> derivation -> oracle like any built-in source."""
    from graphstep.reading import sources

    def weather_adapter(gap):
        facts = {"freeze": kb.Row(
            "freeze", "PRED", sig={"name": "IS_FREEZING", "arg": "NUM"},
            payload="({x} <= 0)", provenance="web:weather-glossary")}
        return [facts[gap.word]] if gap.word in facts else []

    sources.ADAPTERS.append(weather_adapter)
    try:
        rec = solve_loop({
            "text": "Write a function to check whether the temperature "
                    "is freezing.",
            "test_list": ["assert f(-5) == True", "assert f(3) == False",
                          "assert f(0) == True"]})
        assert rec["status"] == "SOLVED", rec.get("reasons")
        assert "<= 0" in rec["code"]
    finally:
        sources.ADAPTERS.remove(weather_adapter)
    print("ok runtime_plugged_new_domain_adapter")


def test_kind_ontology_contract():
    """Every kind declares its oracle and grade — no oracle, no kind."""
    for kind, spec in sources.KINDS.items():
        assert spec.get("oracle"), kind
        assert spec.get("grade"), kind
        assert "stored" in spec, kind
    assert sources.KINDS["procedure"]["stored"] is False
    print("ok kind_ontology_contract")


def test_procedure_kind_oracle_gated():
    """A worked example is employed ONLY through the task oracle: a wrong
    candidate with a better lexical match must lose to a correct one."""
    from graphstep.reading import sources as S
    pairs = [
        (frozenset({"triple", "number"}),
         "find the triple of a number",
         "def triple(x):\n    return 3 * x + 1"),        # WRONG solution
        (frozenset({"triple", "value", "number"}),
         "find the triple value of a number",
         "def make_triple(v):\n    return 3 * v"),       # correct one
    ]
    index = {}
    for i, (ws, _, _) in enumerate(pairs):
        for w in ws:
            index.setdefault(w, []).append(i)
    S.PROCEDURE_CORPORA.append(("test-corpus", pairs, index))
    try:
        rec = solve_loop({
            "text": "Write a function to find the triple of a number.",
            "test_list": ["assert trip(2) == 6", "assert trip(0) == 0",
                          "assert trip(5) == 15"]})
        assert rec["status"] == "SOLVED"
        assert "3 * v" in rec["code"]
        assert rec.get("grade", "").startswith("verified")
        assert not any(r.symbol == "PROCEDURE" for r in kb.KB), \
            "procedures are task-shaped and must never be stored"
    finally:
        S.PROCEDURE_CORPORA.pop()
    print("ok procedure_kind_oracle_gated")


if __name__ == "__main__":
    test_introspection_finds_real_callables()
    test_runtime_plugged_new_domain_adapter()
    test_kind_ontology_contract()
    test_procedure_kind_oracle_gated()
    test_tribunal_hand_rows_win()
    test_gap_solves_via_retrieval_and_oracle()
    test_unknown_word_refuses_honestly()
    test_knowledge_retrieval_graded_and_optin()
    print("sources CI: all green")
