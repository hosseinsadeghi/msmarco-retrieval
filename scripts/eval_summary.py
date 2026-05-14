"""Run all three retrievers over many queries and write a Markdown summary.

For each query that has at least one human-selected "gold" passage in the DB:
    1. Get gold passage ids from the query_passages table.
    2. Run BM25, dense, and hybrid (RRF) retrieval over the full passage corpus.
    3. Record Recall@k (does ANY gold land in top-k) and MRR@k (1 / rank_of_first_gold).
    4. Save aggregate metrics + sample qualitative rows to results/summary.md.

Notes on the metric choice:
    - The MS MARCO QA task only flags `is_selected=1` on the candidates *for that
      query*, so a passage that is gold for one query is not gold for another.
    - With ~10k passages in the index, surfacing a gold passage in the top few
      is non-trivial — BM25 has to beat 10k distractors, not just rank within 10.
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
from tqdm import tqdm  # noqa: E402

from msmarco_retrieval.config import BM25_PATH  # noqa: E402
from msmarco_retrieval.db import connect, load_all_passages  # noqa: E402
from msmarco_retrieval.embeddings import encode  # noqa: E402
from msmarco_retrieval.text import tokenize  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def bm25_rank(bm25, bm25_ids, query: str, depth: int) -> list[int]:
    scores = bm25.get_scores(tokenize(query))
    top = np.argsort(-scores)[:depth]
    return [bm25_ids[i] for i in top]


def dense_rank(mat, ids, q_vec, depth: int) -> list[int]:
    scores = mat @ q_vec
    top = np.argsort(-scores)[:depth]
    return [ids[i] for i in top]


def rrf_fuse(rankings: list[list[int]], rrf_k: int, depth: int) -> list[int]:
    score: dict[int, float] = defaultdict(float)
    for rank_list in rankings:
        for rank, pk in enumerate(rank_list):
            score[pk] += 1.0 / (rrf_k + rank + 1)
    return [pk for pk, _ in sorted(score.items(), key=lambda kv: -kv[1])[:depth]]


def metrics(ranked: list[int], gold: set[int], ks: tuple[int, ...]) -> dict[str, float]:
    """Recall@k (binary: any gold in top-k) and MRR@k (reciprocal rank of first gold)."""
    out: dict[str, float] = {}
    first_hit_rank = next((r + 1 for r, pk in enumerate(ranked) if pk in gold), None)
    for k in ks:
        top_k = ranked[:k]
        out[f"recall@{k}"] = 1.0 if any(pk in gold for pk in top_k) else 0.0
        out[f"mrr@{k}"] = 1.0 / first_hit_rank if (first_hit_rank and first_hit_rank <= k) else 0.0
    return out


def fmt_table(rows: list[tuple[str, dict[str, float]]], ks: tuple[int, ...]) -> str:
    headers = ["Retriever"] + [f"Recall@{k}" for k in ks] + [f"MRR@{k}" for k in ks]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for name, m in rows:
        row = [name] + [f"{m[f'recall@{k}']:.3f}" for k in ks] + [f"{m[f'mrr@{k}']:.3f}" for k in ks]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=200, help="number of queries to evaluate")
    p.add_argument("--depth", type=int, default=50, help="retrieval depth per retriever (used for RRF too)")
    p.add_argument("--samples", type=int, default=5, help="qualitative sample queries to embed in the report")
    p.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "summary.md")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    KS = (1, 5, 10)

    conn = connect()

    # 1. Pick queries that have at least one is_selected=1 gold passage.
    eligible = conn.execute(
        """
        SELECT q.id, q.query, q.answers
        FROM queries q
        WHERE q.id IN (SELECT query_pk FROM query_passages WHERE is_selected = 1)
        ORDER BY q.id
        """
    ).fetchall()
    if not eligible:
        raise SystemExit("No queries with gold passages — did 02_build_db.py run?")

    pick = rng.choice(len(eligible), size=min(args.num, len(eligible)), replace=False)
    queries = [eligible[i] for i in pick]
    print(f"Evaluating on {len(queries)} queries (eligible pool: {len(eligible)})")

    # 2. Load shared state once.
    print("Loading BM25 index...")
    with BM25_PATH.open("rb") as f:
        bundle = pickle.load(f)
    bm25_ids, bm25 = bundle["ids"], bundle["bm25"]

    print("Loading passage matrix from SQLite...")
    p_ids, p_texts, mat = load_all_passages(conn)
    text_by_pk = dict(zip(p_ids, p_texts))

    # 3. Encode all queries in one batch (much faster than one-by-one).
    print("Encoding queries...")
    q_texts = [q for _, q, _ in queries]
    q_vecs = encode(q_texts, batch_size=64, show_progress=True)

    # 4. Run all three retrievers per query and accumulate metrics.
    agg: dict[str, list[dict[str, float]]] = {"BM25": [], "Dense": [], "Hybrid (RRF)": []}
    sample_rows = []  # qualitative

    for i, (q_pk, q_text, q_answers) in enumerate(tqdm(queries)):
        gold_rows = conn.execute(
            "SELECT passage_pk FROM query_passages WHERE query_pk = ? AND is_selected = 1",
            (q_pk,),
        ).fetchall()
        gold = {r[0] for r in gold_rows}

        bm25_top = bm25_rank(bm25, bm25_ids, q_text, args.depth)
        dense_top = dense_rank(mat, p_ids, q_vecs[i], args.depth)
        hybrid_top = rrf_fuse([bm25_top, dense_top], args.rrf_k, args.depth)

        agg["BM25"].append(metrics(bm25_top, gold, KS))
        agg["Dense"].append(metrics(dense_top, gold, KS))
        agg["Hybrid (RRF)"].append(metrics(hybrid_top, gold, KS))

        if i < args.samples:
            sample_rows.append({
                "query": q_text,
                "answers": q_answers,
                "gold": gold,
                "bm25": bm25_top[:3],
                "dense": dense_top[:3],
                "hybrid": hybrid_top[:3],
            })

    # 5. Aggregate and write Markdown.
    summary_table_rows = []
    for name in ("BM25", "Dense", "Hybrid (RRF)"):
        avg = {k: float(np.mean([m[k] for m in agg[name]])) for k in agg[name][0].keys()}
        summary_table_rows.append((name, avg))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write("# MS MARCO retrieval summary\n\n")
        f.write(f"Evaluated **{len(queries)}** queries (random sample, seed={args.seed}) ")
        f.write(f"from MS MARCO v2.1 validation, against a corpus of **{len(p_ids):,}** unique passages.\n\n")
        f.write("Gold = passages flagged `is_selected=1` by a human for that query.\n")
        f.write(f"Retrieval depth per retriever: {args.depth}.\n\n")

        f.write("## Aggregate metrics\n\n")
        f.write(fmt_table(summary_table_rows, KS) + "\n\n")
        f.write("- **Recall@k**: fraction of queries where at least one gold passage appears in the top-k.\n")
        f.write("- **MRR@k**: mean reciprocal rank of the first gold hit (0 if no gold in top-k).\n\n")

        f.write("## Sample queries\n\n")
        for s in sample_rows:
            f.write(f"### {s['query']}\n\n")
            f.write(f"- **Gold answer(s):** `{s['answers']}`\n")
            f.write(f"- **Gold passage_id(s):** {sorted(s['gold'])}\n\n")
            for name, ranked in (("BM25", s["bm25"]), ("Dense", s["dense"]), ("Hybrid", s["hybrid"])):
                f.write(f"**{name} top-3:**\n\n")
                for rank, pk in enumerate(ranked, 1):
                    hit = " ✅" if pk in s["gold"] else ""
                    snippet = text_by_pk[pk][:200].replace("\n", " ").strip()
                    f.write(f"{rank}. `pk={pk}`{hit} — {snippet}…\n")
                f.write("\n")
            f.write("---\n\n")

    print(f"\nWrote {args.out}")
    # Also echo the table to stdout.
    print("\n" + fmt_table(summary_table_rows, KS))


if __name__ == "__main__":
    main()
