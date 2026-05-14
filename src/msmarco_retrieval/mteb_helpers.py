"""Helpers for MTEB / MTEB-v2 retrieval datasets in BEIR format.

A BEIR-format dataset on the HuggingFace `mteb/*` org has three subsets:
    - corpus  : {_id, title, text}
    - queries : {_id, text}
    - default : qrels with {query-id, corpus-id, score}, split by test/dev/train

Each dataset gets its own SQLite file and BM25 pickle so they don't collide.
External string ids from the source dataset (e.g. "MED-10") are preserved
alongside our integer primary keys; qrels are stored using the integer pks.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import ROOT

# name -> HuggingFace dataset id.
#
# MTEB and MTEB v2 (MMTEB) retrieval datasets live under the `mteb/` org. Most
# share the standard BEIR layout (subsets `corpus` / `queries` / `default`),
# which is what mteb_download.py expects, so the same scripts work for either
# benchmark version — the v2 release mostly adds new datasets without changing
# the data format.
#
# A few MTEB v2 variants (e.g. the *HardNegatives* qrels-only repos) reference
# the corpus + queries from the base task and ship only refined qrels; those
# need extra plumbing and aren't included here.
PRESETS: dict[str, str] = {
    "NFCorpus":    "mteb/nfcorpus",
    "SciFact":     "mteb/scifact",
    "ArguAna":     "mteb/arguana",
    "FiQA2018":    "mteb/fiqa",
    "SCIDOCS":     "mteb/scidocs",
    "TRECCOVID":   "mteb/trec-covid",
    "Touche2020":  "mteb/touche2020",
    "Quora":       "mteb/quora",
}


def resolve_hf_id(dataset: str, override: str | None = None) -> str:
    if override:
        return override
    if dataset in PRESETS:
        return PRESETS[dataset]
    raise SystemExit(
        f"Unknown dataset {dataset!r}. Known names: {sorted(PRESETS)}.\n"
        f"Pass --hf-id to point at any other mteb/* repo."
    )


def dataset_paths(name: str) -> dict[str, Path]:
    return {
        "data_dir":     ROOT / "data" / "mteb" / name,
        "db_path":      ROOT / "db" / f"mteb_{name}.sqlite",
        "bm25_path":    ROOT / "indexes" / f"mteb_{name}_bm25.pkl",
        "results_path": ROOT / "results" / f"mteb_{name}.md",
    }


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
    text        TEXT NOT NULL
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
