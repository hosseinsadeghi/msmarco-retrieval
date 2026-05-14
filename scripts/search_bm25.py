"""BM25-only search.

Pipeline (top-to-bottom):
    1. Load the query (from --question or a random row in the DB).
    2. Tokenize it into a bag-of-words list.
    3. Score every passage with the pickled BM25 index.
    4. Print the top-k passages with their scores.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from msmarco_retrieval.config import BM25_PATH  # noqa: E402
from msmarco_retrieval.db import connect  # noqa: E402
from msmarco_retrieval.text import tokenize  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--question", "-q", help="question text (omit to pick a random one from the DB)")
    p.add_argument("--k", type=int, default=5, help="top-k passages to print")
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

    # 2. Tokenize.
    query_tokens = tokenize(question)

    # 3. Load BM25 index and score every passage.
    with BM25_PATH.open("rb") as f:
        bundle = pickle.load(f)
    ids: list[int] = bundle["ids"]
    bm25 = bundle["bm25"]
    scores = bm25.get_scores(query_tokens)

    top_idx = np.argsort(-scores)[: args.k]
    top_ids = [ids[i] for i in top_idx]

    # 4. Fetch the passage text for the winners and print.
    placeholders = ",".join("?" * len(top_ids))
    rows = dict(
        conn.execute(f"SELECT id, text FROM passages WHERE id IN ({placeholders})", top_ids)
    )

    print(f"Q: {question}")
    if answers:
        print(f"A (gold): {answers}")
    print("=" * 80)
    for rank, idx in enumerate(top_idx, 1):
        pk = ids[idx]
        print(f"[{rank}] score={scores[idx]:.3f}  passage_id={pk}")
        print(f"     {rows[pk][:240].strip()}")
        print()


if __name__ == "__main__":
    main()
