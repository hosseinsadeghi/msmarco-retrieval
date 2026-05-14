"""Hybrid BM25 + dense search via Reciprocal Rank Fusion (RRF).

RRF is the simplest competitive fusion method: each retriever contributes a
ranked list, and the final score for a doc is sum(1 / (rrf_k + rank_in_list)).
It has one hyperparameter (rrf_k, conventionally 60) and ignores raw score
scales, which matters here because BM25 and cosine sims live on different scales.

Pipeline:
    1. Load the query.
    2. Run BM25 to get a top-N ranked list.
    3. Run dense to get a top-N ranked list.
    4. Fuse with RRF and print the top-k.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from msmarco_retrieval.config import BM25_PATH  # noqa: E402
from msmarco_retrieval.db import connect, load_all_passages  # noqa: E402
from msmarco_retrieval.embeddings import encode  # noqa: E402
from msmarco_retrieval.text import tokenize  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--question", "-q", help="question text (omit to pick a random one from the DB)")
    p.add_argument("--k", type=int, default=5, help="top-k passages to print")
    p.add_argument("--candidates", type=int, default=50, help="candidates from each retriever")
    p.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant")
    args = p.parse_args()

    conn = connect()

    # 1. Load the query.
    if args.question:
        question = args.question
        answers = None
    else:
        row = conn.execute(
            "SELECT query, answers FROM queries ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        question, answers = row

    # 2. BM25 top-N.
    with BM25_PATH.open("rb") as f:
        bundle = pickle.load(f)
    bm25_ids: list[int] = bundle["ids"]
    bm25 = bundle["bm25"]
    bm25_scores = bm25.get_scores(tokenize(question))
    bm25_top = np.argsort(-bm25_scores)[: args.candidates]
    bm25_ranked = [bm25_ids[i] for i in bm25_top]  # list of passage pks, best first

    # 3. Dense top-N.
    ids, texts, mat = load_all_passages(conn)
    q_vec = encode([question])[0]
    dense_scores = mat @ q_vec
    dense_top = np.argsort(-dense_scores)[: args.candidates]
    dense_ranked = [ids[i] for i in dense_top]

    # 4. RRF fusion.
    rrf: dict[int, float] = defaultdict(float)
    for rank, pk in enumerate(bm25_ranked):
        rrf[pk] += 1.0 / (args.rrf_k + rank + 1)
    for rank, pk in enumerate(dense_ranked):
        rrf[pk] += 1.0 / (args.rrf_k + rank + 1)

    fused = sorted(rrf.items(), key=lambda kv: -kv[1])[: args.k]

    # Fetch text for the winners.
    winner_ids = [pk for pk, _ in fused]
    placeholders = ",".join("?" * len(winner_ids))
    text_by_pk = dict(
        conn.execute(
            f"SELECT id, text FROM passages WHERE id IN ({placeholders})", winner_ids
        )
    )

    # Also build a quick rank lookup so we can show where each winner came from.
    bm25_rank_of = {pk: r + 1 for r, pk in enumerate(bm25_ranked)}
    dense_rank_of = {pk: r + 1 for r, pk in enumerate(dense_ranked)}

    print(f"Q: {question}")
    if answers:
        print(f"A (gold): {answers}")
    print("=" * 80)
    for rank, (pk, rrf_score) in enumerate(fused, 1):
        b = bm25_rank_of.get(pk, "-")
        d = dense_rank_of.get(pk, "-")
        print(f"[{rank}] rrf={rrf_score:.4f}  bm25_rank={b}  dense_rank={d}  passage_id={pk}")
        print(f"     {text_by_pk[pk][:240].strip()}")
        print()


if __name__ == "__main__":
    main()
