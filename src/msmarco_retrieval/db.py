"""SQLite helpers. Embeddings are stored as float32 BLOBs alongside the text.

Schema:
  passages(id INTEGER PK, text TEXT UNIQUE, embedding BLOB)
  queries(id INTEGER PK, query_id TEXT, query TEXT, answers TEXT)
  query_passages(query_pk INT, passage_pk INT, is_selected INT)
"""
import sqlite3
from pathlib import Path

import numpy as np

from .config import DB_PATH, EMBED_DIM


SCHEMA = """
CREATE TABLE IF NOT EXISTS passages (
    id        INTEGER PRIMARY KEY,
    text      TEXT NOT NULL UNIQUE,
    embedding BLOB
);
CREATE TABLE IF NOT EXISTS queries (
    id       INTEGER PRIMARY KEY,
    query_id TEXT,
    query    TEXT NOT NULL,
    answers  TEXT
);
CREATE TABLE IF NOT EXISTS query_passages (
    query_pk    INTEGER NOT NULL,
    passage_pk  INTEGER NOT NULL,
    is_selected INTEGER NOT NULL,
    PRIMARY KEY (query_pk, passage_pk)
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def vec_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).reshape(EMBED_DIM)


def load_all_passages(conn: sqlite3.Connection) -> tuple[list[int], list[str], np.ndarray]:
    """Returns parallel lists of (pk, text, embeddings-matrix)."""
    rows = conn.execute(
        "SELECT id, text, embedding FROM passages WHERE embedding IS NOT NULL ORDER BY id"
    ).fetchall()
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    mat = np.stack([blob_to_vec(r[2]) for r in rows]) if rows else np.empty((0, EMBED_DIM), dtype=np.float32)
    return ids, texts, mat
