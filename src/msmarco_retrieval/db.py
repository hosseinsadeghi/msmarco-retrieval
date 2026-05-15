"""Unified SQLite schema used for every dataset.

  passages  : (id, external_id, text, embedding BLOB)
  queries   : (id, external_id, text, answers JSON or NULL)
  qrels     : (query_pk, passage_pk, relevance)  — graded relevance, 0 = miss

`external_id` is whatever id the source dataset uses (e.g. BEIR's "_id" strings,
or a synthetic "p123" for MS MARCO). Internal joins all use the integer pks.
`answers` is populated for MS MARCO (a JSON list of human-written answers) and
NULL for BEIR datasets.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .config import EMBED_DIM


SCHEMA = """
CREATE TABLE IF NOT EXISTS passages (
    id          INTEGER PRIMARY KEY,
    external_id TEXT NOT NULL UNIQUE,
    text        TEXT NOT NULL,
    embedding   BLOB
);
CREATE TABLE IF NOT EXISTS queries (
    id          INTEGER PRIMARY KEY,
    external_id TEXT NOT NULL UNIQUE,
    text        TEXT NOT NULL,
    answers     TEXT
);
CREATE TABLE IF NOT EXISTS qrels (
    query_pk   INTEGER NOT NULL,
    passage_pk INTEGER NOT NULL,
    relevance  INTEGER NOT NULL,
    PRIMARY KEY (query_pk, passage_pk)
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def vec_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).reshape(EMBED_DIM)


def load_all_passages(conn: sqlite3.Connection) -> tuple[list[int], list[str], np.ndarray]:
    """Returns parallel lists of (pk, text) plus a dense embeddings matrix."""
    rows = conn.execute(
        "SELECT id, text, embedding FROM passages WHERE embedding IS NOT NULL ORDER BY id"
    ).fetchall()
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    mat = (
        np.stack([blob_to_vec(r[2]) for r in rows])
        if rows else np.empty((0, EMBED_DIM), dtype=np.float32)
    )
    return ids, texts, mat
