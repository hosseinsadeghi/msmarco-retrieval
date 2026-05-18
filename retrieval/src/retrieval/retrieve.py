"""The three retrievers in one place so search.py and eval.py share them.

Each function returns a ranked list of passage primary keys (best first).
None of these touch SQLite — they take pre-loaded state — which keeps the eval
loop tight (no per-query DB round-trips for the corpus).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .text import tokenize


def bm25_search(bm25, bm25_ids: list[int], query: str, depth: int) -> list[int]:
    scores = bm25.get_scores(tokenize(query))
    top = np.argsort(-scores)[:depth]
    return [bm25_ids[i] for i in top]


def dense_search(mat: np.ndarray, ids: list[int], q_vec: np.ndarray, depth: int) -> list[int]:
    """`mat` and `q_vec` must already be L2-normalized so dot = cosine."""
    scores = mat @ q_vec
    top = np.argsort(-scores)[:depth]
    return [ids[i] for i in top]


def rrf_fuse(rank_lists: list[list[int]], rrf_k: int, depth: int) -> list[int]:
    score: dict[int, float] = defaultdict(float)
    for ranked in rank_lists:
        for rank, pk in enumerate(ranked):
            score[pk] += 1.0 / (rrf_k + rank + 1)
    return [pk for pk, _ in sorted(score.items(), key=lambda kv: -kv[1])[:depth]]
