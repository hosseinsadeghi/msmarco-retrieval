"""Single-query search against any built dataset.

Pipeline (top-to-bottom):
    1. Load the query (from --question, or a random one with qrels from the DB).
    2. Prepare (tokenize for BM25, encode for dense, both for hybrid).
    3. Score with the chosen --retriever.
    4. Print top-k passages, marking qrel-positive ones with ✅.

Examples:
    python scripts/search.py --dataset NFCorpus -q "vegan diet and heart disease"
    python scripts/search.py --dataset NFCorpus --retriever bm25 -q "..."
    python scripts/search.py --dataset MSMARCO --retriever dense    # random query
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msmarco_retrieval.datasets import dataset_paths, get  # noqa: E402
from msmarco_retrieval.db import connect, load_all_passages  # noqa: E402
from msmarco_retrieval.embeddings import encode  # noqa: E402
from msmarco_retrieval.retrieve import bm25_search, dense_search, rrf_fuse  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="see scripts/list_datasets.py")
    p.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="hybrid")
    p.add_argument("-q", "--question", help="question text (omit -> pick a random one with qrels)")
    p.add_argument("--k", type=int, default=5, help="top-k passages to print")
    p.add_argument("--candidates", type=int, default=50, help="hybrid: depth from each retriever")
    p.add_argument("--rrf-k", type=int, default=60, help="hybrid: RRF smoothing constant")
    args = p.parse_args()

    spec = get(args.dataset)
    paths = dataset_paths(spec)
    conn = connect(paths["db_path"])

    # 1. Load the query.
    if args.question:
        question, answers, gold = args.question, None, set()
    else:
        row = conn.execute(
            """
            SELECT q.id, q.text, q.answers
            FROM queries q
            WHERE q.id IN (SELECT query_pk FROM qrels WHERE relevance > 0)
            ORDER BY RANDOM() LIMIT 1
            """
        ).fetchone()
        if row is None:
            raise SystemExit("No queries with qrels. Did scripts/build.py run?")
        q_pk, question, answers = row
        gold = {
            r[0]
            for r in conn.execute(
                "SELECT passage_pk FROM qrels WHERE query_pk = ? AND relevance > 0", (q_pk,)
            )
        }

    # 2. Prepare and 3. Score.
    if args.retriever in ("bm25", "hybrid"):
        with paths["bm25_path"].open("rb") as f:
            bundle = pickle.load(f)
        bm25_ids, bm25 = bundle["ids"], bundle["bm25"]
    if args.retriever in ("dense", "hybrid"):
        p_ids, p_texts, mat = load_all_passages(conn)
        q_vec = encode([question])[0]

    if args.retriever == "bm25":
        ranked = bm25_search(bm25, bm25_ids, question, depth=args.k)
    elif args.retriever == "dense":
        ranked = dense_search(mat, p_ids, q_vec, depth=args.k)
    else:  # hybrid
        bm25_top = bm25_search(bm25, bm25_ids, question, depth=args.candidates)
        dense_top = dense_search(mat, p_ids, q_vec, depth=args.candidates)
        ranked = rrf_fuse([bm25_top, dense_top], rrf_k=args.rrf_k, depth=args.k)

    # 4. Print.
    placeholders = ",".join("?" * len(ranked))
    text_by_pk = dict(
        conn.execute(f"SELECT id, text FROM passages WHERE id IN ({placeholders})", ranked)
    )

    print(f"[{args.dataset}/{args.retriever}]  Q: {question}")
    if answers:
        print(f"A (gold): {answers}")
    print("=" * 80)
    for rank, pk in enumerate(ranked, 1):
        mark = " ✅" if pk in gold else ""
        snippet = text_by_pk[pk][:240].replace("\n", " ").strip()
        print(f"[{rank}] passage_id={pk}{mark}")
        print(f"     {snippet}")
        print()


if __name__ == "__main__":
    main()
