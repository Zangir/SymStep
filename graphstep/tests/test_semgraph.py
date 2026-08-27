#!/usr/bin/env python3
"""Semantic-graph CI: total capture, attribution honesty, edge history.
Like the garbage CI — these may never regress."""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphstep.reading.semgraph import read, SemGraph, Edge
from graphstep.unified import solve


def test_total_capture():
    g = read("The company released a cheaper phone on Tuesday. "
             "Analysts expect strong sales.")
    assert len(g.clauses) >= 2, "every sentence must yield clauses"
    preds = {c.pred for c in g.clauses}
    assert "release" in preds and "expect" in preds
    rel = [c for c in g.clauses if c.pred == "release"][0]
    assert rel.tense == "past" and rel.roles.get("on") == "tuesday"
    print("ok total_capture")


def test_attribution_never_asserted():
    rec = solve({"text": "The spokesman said the water is safe. "
                         "Independent tests found lead in the water."})
    claims = " ".join(c["conclusion"] for c in rec.get("conclusions", []))
    assert "safe" not in claims, \
        "an attributed claim must never surface as a proved conclusion"
    assert any("spokesman" in a for a in rec.get("attributions", []))
    print("ok attribution_never_asserted")


def test_proof_tier_beats_capture():
    rec = solve({"text": "Anne is kind. If someone is kind then they help "
                         "the team. The manager claimed Anne is lazy."})
    proved = [c["conclusion"] for c in rec.get("conclusions", [])]
    assert any("helps the team" in p for p in proved)
    assert not any("lazy" in p for p in proved)
    print("ok proof_tier_beats_capture")


def test_edge_history_never_deleted():
    g = SemGraph()
    g.assert_edge(Edge("john", "location", "hallway", start=0))
    g.assert_edge(Edge("john", "location", "kitchen", start=2))
    assert len(g.edges) == 2, "history is kept, never deleted"
    cur = g.current("john", "location")
    assert [e.obj for e in cur] == ["kitchen"], "newest edge is current"
    closed = [e for e in g.edges if e.end is not None]
    assert closed and closed[0].obj == "hallway" and closed[0].end == 2, \
        "older exclusive edge is closed at the update time"
    # cumulative relations accumulate instead
    g.assert_edge(Edge("john", "carry", "ball", start=1))
    g.assert_edge(Edge("john", "carry", "key", start=3))
    assert len(g.current("john", "carry")) == 2
    print("ok edge_history_never_deleted")


if __name__ == "__main__":
    test_total_capture()
    test_attribution_never_asserted()
    test_proof_tier_beats_capture()
    test_edge_history_never_deleted()
    print("semgraph CI: all green")
