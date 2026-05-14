"""Dense-embedding-only search.

Pipeline (top-to-bottom):
    1. Load the query (from --question or a random row in the DB).
    2. Encode it with the same model used at indexing time.
    3. Cosine-score it against every stored passage embedding (one matmul).
    4. Print the top-k passages.

Because passage and query vectors are L2-normalized at encode time, cosine
similarity is just a dot product.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from msmarco_retrieval.db import connect, load_all_passages  # noqa: E402
from msmarco_retrieval.embeddings import encode  # noqa: E402


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

    # 2. Encode the query (returns shape (1, d), already L2-normalized).
    q_vec = encode([question])[0]

    # 3. Load the passage matrix from SQLite and score with a single dot product.
    ids, texts, mat = load_all_passages(conn)
    scores = mat @ q_vec  # cosine sim because both sides are unit-norm

    top_idx = np.argsort(-scores)[: args.k]

    # 4. Print results.
    print(f"Q: {question}")
    if answers:
        print(f"A (gold): {answers}")
    print("=" * 80)
    for rank, i in enumerate(top_idx, 1):
        print(f"[{rank}] cosine={scores[i]:.3f}  passage_id={ids[i]}")
        print(f"     {texts[i][:240].strip()}")
        print()


if __name__ == "__main__":
    main()
