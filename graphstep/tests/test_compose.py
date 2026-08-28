#!/usr/bin/env python3
"""Composition-compiler CI: the same compiler must build correct trees for
DIFFERENT algebras — domain enters only through the leaf grounder. Plus the
honesty property: an ungroundable head yields None, never a guess."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from graphstep.reading.compose import (compile_tree, emit_code, free_slots,
                                       default_leaf, Node)
from graphstep.reading import kb


def _head(text, word):
    import spacy
    global _NLP
    try:
        _NLP
    except NameError:
        _NLP = spacy.load("en_core_web_sm")
    doc = _NLP(text)
    return next(t for t in doc if t.lemma_.lower() == word), doc


def test_code_algebra_depth2():
    tok, _ = _head("the sum of the digits of the number", "sum")
    tree = compile_tree(tok)
    assert tree.atoms() == ["sum", "digit"]   # 'number' is a VAR leaf
    slots = free_slots(tree)
    assert len(slots) == 1 and slots[0][1] == "NUM"
    code = emit_code(tree, {slots[0][0]: "arg0"})
    assert eval(code.replace("arg0", "123")) == 6
    print("ok code_algebra_depth2 ->", code)


def test_code_algebra_binop():
    tok, _ = _head("the difference between the largest and smallest value",
                   "difference")
    tree = compile_tree(tok)
    slots = free_slots(tree)
    binding = {k: "arg0" for k, _ in slots}
    code = emit_code(tree, binding)
    assert eval(code.replace("arg0", "[3, 9, 1]")) == 8
    print("ok code_algebra_binop ->", code)


def test_other_algebra_same_compiler():
    """A TOY relational algebra: the identical compiler + a different leaf
    grounder builds the right tree for 'the owner of the dog' — proof that
    nothing in the compiler is about code."""
    rel = kb.Row("owner", "REDUCE", payload="owner_of({src})",
                 provenance="test")
    ent = kb.Row("dog", "REDUCE", payload="DOG", provenance="test")

    def toy_leaf(tok):
        if tok.lemma_.lower() == "owner":
            return Node("atom", row=rel, word="owner")
        if tok.lemma_.lower() == "dog":
            return Node("atom", row=ent, word="dog")
        return None

    tok, _ = _head("the owner of the dog", "owner")
    tree = compile_tree(tok, leaf=toy_leaf)
    assert tree.atoms() == ["owner", "dog"]
    assert emit_code(tree, {}) == "owner_of(DOG)"
    print("ok other_algebra_same_compiler -> owner_of(DOG)")


def test_honest_none_on_unknown_head():
    tok, _ = _head("the frobnication of the list", "frobnication")
    assert compile_tree(tok) is None, "unknown head must refuse, not guess"
    print("ok honest_none_on_unknown_head")


def test_fold_coercion_is_open_class():
    """(BINOP, SEQ) coercion must fire for ANY binary atom — including one
    plugged at runtime it has never seen — or it's a patch, not a rule."""
    from graphstep.agenda import solve_loop
    from graphstep.reading import sources
    glue = kb.Row("glue", "BINOP", payload="(({a}) * 10 + ({b}))",
                  provenance="web:test")

    def toy(gap):
        return [glue] if gap.word == "glue" else []
    sources.ADAPTERS.append(toy)
    try:
        rec = solve_loop({
            "text": "Write a function to find the glue of the given "
                    "array elements.",
            "test_list": ["assert g([1, 2, 3]) == 123",
                          "assert g([4, 5]) == 45"]})
        assert rec["status"] == "SOLVED", rec.get("reasons")
        assert "reduce" in rec["code"]
    finally:
        sources.ADAPTERS.remove(toy)
    print("ok fold_coercion_is_open_class")


def test_key_device_composes():
    from graphstep.agenda import solve_loop
    rec = solve_loop({
        "text": "Write a function to sort the given matrix according to "
                "the sum of its rows.",
        "test_list": [
            "assert s([[2, 4, 5], [1, 2, 3], [1, 1, 1]]) == "
            "[[1, 1, 1], [1, 2, 3], [2, 4, 5]]"]})
    assert rec["status"] == "SOLVED"
    assert "key=lambda" in rec["code"] and "sum" in rec["code"]
    print("ok key_device_composes")


def test_collection_relative_predicate():
    from graphstep.agenda import solve_loop
    rec = solve_loop({
        "text": "Write a python function to find the sum of repeated "
                "elements in a given array.",
        "test_list": ["assert f([1,2,3,1,1,4,5,6]) == 3",
                      "assert f([1,2,2]) == 4"]})
    assert rec["status"] == "SOLVED" and rec.get("grade") == "certified"
    assert ".count(_e)" in rec["code"]
    print("ok collection_relative_predicate")


if __name__ == "__main__":
    test_code_algebra_depth2()
    test_code_algebra_binop()
    test_other_algebra_same_compiler()
    test_honest_none_on_unknown_head()
    test_fold_coercion_is_open_class()
    test_key_device_composes()
    test_collection_relative_predicate()
    print("compose CI: all green")
