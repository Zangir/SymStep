#!/usr/bin/env python3
"""The self-growing knowledge store: every admitted retrieval symbol is
persisted to SQLite, so knowledge fetched once is a local lookup forever.
Serves as L2 behind the adapters' in-memory L1 cache; survives processes.

Rows keep full provenance. Empty results are cached too (negative caching)
so unknown terms don't re-hit the network."""
from __future__ import annotations
import json, os, sqlite3, time
from typing import List, Optional, Union

from .worldmodel import Rule, Literal

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "kb_cache.sqlite")


def _ser(item: Union[Rule, Literal]) -> str:
    if isinstance(item, Rule):
        return json.dumps({
            "kind": "rule",
            "conds": [[c.pos, c.subj, c.pred, c.obj] for c in item.conds],
            "concl": [item.concl.pos, item.concl.subj, item.concl.pred,
                      item.concl.obj],
            "origin": item.origin})
    return json.dumps({"kind": "fact",
                       "lit": [item.pos, item.subj, item.pred, item.obj,
                               item.origin]})


def _de(payload: str, origin: str) -> Union[Rule, Literal]:
    d = json.loads(payload)
    if d["kind"] == "rule":
        return Rule([Literal(*c) for c in d["conds"]],
                    Literal(*d["concl"]), d["origin"])
    return Literal(*d["lit"])


class KBStore:
    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS admissions ("
            " source TEXT, term TEXT, payload TEXT, origin TEXT,"
            " created REAL)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_st ON admissions(source, term)")
        self.db.commit()

    def get(self, source: str, term: str) -> Optional[list]:
        """None = never fetched; [] = fetched and known-empty."""
        rows = self.db.execute(
            "SELECT payload, origin FROM admissions WHERE source=? AND "
            "term=?", (source, term)).fetchall()
        if not rows:
            return None
        return [_de(p, o) for p, o in rows if p != "__EMPTY__"]

    def put(self, source: str, term: str, items: list):
        self.db.execute("DELETE FROM admissions WHERE source=? AND term=?",
                        (source, term))
        now = time.time()
        if not items:
            self.db.execute("INSERT INTO admissions VALUES (?,?,?,?,?)",
                            (source, term, "__EMPTY__", "", now))
        for it in items:
            origin = it.origin if isinstance(it, Rule) else ""
            self.db.execute("INSERT INTO admissions VALUES (?,?,?,?,?)",
                            (source, term, _ser(it), origin, now))
        self.db.commit()

    def size(self) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM admissions WHERE payload != '__EMPTY__'"
        ).fetchone()[0]


_STORE: Optional[KBStore] = None
_DEFAULT = DEFAULT_PATH


def use(path: str):
    """Redirect the default store (tests use a temp file)."""
    global _DEFAULT, _STORE
    _DEFAULT = path
    _STORE = None


def store(path: Optional[str] = None) -> KBStore:
    global _STORE
    path = path or _DEFAULT
    if _STORE is None or _STORE.path != path:
        _STORE = KBStore(path)
    return _STORE
