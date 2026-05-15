"""Benchmark eval (Recall@k, MRR@k, nDCG@k) for any built dataset.

Picks queries that have at least one positive qrel, runs all three retrievers,
aggregates metrics, and writes results/<name>.md.

Examples:
    python scripts/eval.py --dataset NFCorpus
    python scripts/eval.py --dataset MSMARCO --num 200
    python scripts/eval.py --dataset SciFact --depth 100 --samples 5
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from msmarco_retrieval.datasets import dataset_paths, get  # noqa: E402
from msmarco_retrieval.db import connect, load_all_passages  # noqa: E402
from msmarco_retrieval.embeddings import encode  # noqa: E402
from msmarco_retrieval.retrieve import bm25_search, dense_search, rrf_fuse  # noqa: E402

KS = (1, 5, 10)


def metrics_for_query(
    ranked: list[int], gold: dict[int, int], ks: tuple[int, ...]
) -> dict[str, float]:
    out: dict[str, float] = {}
    relevant = {pk for pk, rel in gold.items() if rel > 0}
    first_hit = next((r + 1 for r, pk in enumerate(ranked) if pk in relevant), None)
    ideal_rels = sorted(gold.values(), reverse=True)

    for k in ks:
        top_k = ranked[:k]
        out[f"recall@{k}"] = 1.0 if any(pk in relevant for pk in top_k) else 0.0
        out[f"mrr@{k}"] = 1.0 / first_hit if (first_hit and first_hit <= k) else 0.0
        dcg = sum(
            (2 ** gold.get(pk, 0) - 1) / math.log2(rank + 2) for rank, pk in enumerate(top_k)
        )
        ideal = sum(
            (2 ** rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(ideal_rels[:k])
        )
        out[f"ndcg@{k}"] = dcg / ideal if ideal > 0 else 0.0
    return out


def fmt_table(rows: list[tuple[str, dict[str, float]]], ks: tuple[int, ...]) -> str:
    headers = (
        ["Retriever"]
        + [f"Recall@{k}" for k in ks]
        + [f"MRR@{k}" for k in ks]
        + [f"nDCG@{k}" for k in ks]
    )
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for name, m in rows:
        cells = (
            [name]
            + [f"{m[f'recall@{k}']:.3f}" for k in ks]
            + [f"{m[f'mrr@{k}']:.3f}" for k in ks]
            + [f"{m[f'ndcg@{k}']:.3f}" for k in ks]
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="see scripts/list_datasets.py")
    p.add_argument("--num", type=int, default=None, help="cap on queries to evaluate (default: all eligible)")
    p.add_argument("--depth", type=int, default=100, help="retrieval depth per retriever")
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--samples", type=int, default=3, help="qualitative samples in the report")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    spec = get(args.dataset)
    paths = dataset_paths(spec)
    rng = np.random.default_rng(args.seed)

    conn = connect(paths["db_path"])

    # ---- load shared state once ----
    with paths["bm25_path"].open("rb") as f:
        bundle = pickle.load(f)
    bm25_ids, bm25 = bundle["ids"], bundle["bm25"]

    p_ids, p_texts, mat = load_all_passages(conn)
    text_by_pk = dict(zip(p_ids, p_texts))

    eligible = conn.execute(
        """
        SELECT q.id, q.text
        FROM queries q
        WHERE q.id IN (SELECT query_pk FROM qrels WHERE relevance > 0)
        ORDER BY q.id
        """
    ).fetchall()
    if not eligible:
        raise SystemExit("No queries with qrels — was build.py run?")

    if args.num and len(eligible) > args.num:
        pick = rng.choice(len(eligible), size=args.num, replace=False)
        queries = [eligible[i] for i in pick]
    else:
        queries = eligible
    print(f"Evaluating {len(queries):,} queries (eligible pool: {len(eligible):,})")

    print("Encoding queries...")
    q_vecs = encode([q[1] for q in queries], batch_size=128, show_progress=True)

    # ---- run all three retrievers per query ----
    agg: dict[str, list[dict[str, float]]] = {"BM25": [], "Dense": [], "Hybrid (RRF)": []}
    sample_rows: list[dict] = []

    for i, (q_pk, q_text) in enumerate(tqdm(queries)):
        gold = {
            pk: rel
            for pk, rel in conn.execute(
                "SELECT passage_pk, relevance FROM qrels WHERE query_pk = ?", (q_pk,)
            )
        }

        bm25_top = bm25_search(bm25, bm25_ids, q_text, depth=args.depth)
        dense_top = dense_search(mat, p_ids, q_vecs[i], depth=args.depth)
        hybrid_top = rrf_fuse([bm25_top, dense_top], rrf_k=args.rrf_k, depth=args.depth)

        agg["BM25"].append(metrics_for_query(bm25_top, gold, KS))
        agg["Dense"].append(metrics_for_query(dense_top, gold, KS))
        agg["Hybrid (RRF)"].append(metrics_for_query(hybrid_top, gold, KS))

        if i < args.samples:
            sample_rows.append({
                "query": q_text,
                "gold": {pk for pk, rel in gold.items() if rel > 0},
                "bm25": bm25_top[:3],
                "dense": dense_top[:3],
                "hybrid": hybrid_top[:3],
            })

    table_rows = []
    for name in ("BM25", "Dense", "Hybrid (RRF)"):
        avg = {k: float(np.mean([m[k] for m in agg[name]])) for k in agg[name][0]}
        table_rows.append((name, avg))

    paths["results_path"].parent.mkdir(parents=True, exist_ok=True)
    with paths["results_path"].open("w") as f:
        f.write(f"# {spec.name} — retrieval summary\n\n")
        f.write(f"Source: `{spec.hf_id}`   |   Benchmarks: {', '.join(spec.benchmarks)}\n\n")
        f.write(f"Corpus: **{len(p_ids):,}** passages.   Queries evaluated: **{len(queries):,}**.\n")
        f.write(f"Retrieval depth: {args.depth}.  RRF k = {args.rrf_k}.\n\n")
        f.write("## Aggregate metrics\n\n")
        f.write(fmt_table(table_rows, KS) + "\n\n")
        f.write(
            "- **Recall@k**: any qrel-positive passage in top-k.\n"
            "- **MRR@k**: reciprocal rank of first relevant hit (0 if not in top-k).\n"
            "- **nDCG@k**: discounted gain weighted by qrel grade, normalized by ideal DCG.\n\n"
        )
        f.write("## Sample queries\n\n")
        for s in sample_rows:
            f.write(f"### {s['query']}\n\n")
            for name, ranked in (("BM25", s["bm25"]), ("Dense", s["dense"]), ("Hybrid", s["hybrid"])):
                f.write(f"**{name} top-3:**\n\n")
                for rank, pk in enumerate(ranked, 1):
                    mark = " ✅" if pk in s["gold"] else ""
                    snippet = text_by_pk[pk][:200].replace("\n", " ").strip()
                    f.write(f"{rank}. `pk={pk}`{mark} — {snippet}…\n")
                f.write("\n")
            f.write("---\n\n")

    print(f"\nWrote {paths['results_path']}")
    print("\n" + fmt_table(table_rows, KS))


if __name__ == "__main__":
    main()
