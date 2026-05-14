"""End-to-end BM25 / dense / hybrid evaluation for one MTEB retrieval dataset.

The script does four things in order:
    1. Load corpus + queries + qrels JSONL written by scripts/mteb_download.py.
    2. Build the SQLite DB and encode every passage (or restore a previous build).
    3. Build (or reload) a BM25 index over the same passages.
    4. Score every test query with BM25, dense, and hybrid (RRF), then compute
       Recall@k, MRR@k, and nDCG@k. Write results/mteb_<name>.md.

Reusing artifacts: if --rebuild is not passed, an existing .sqlite / .pkl is
reused. That makes "tweak the report and rerun" cheap.

Usage:
    python scripts/mteb_run.py --dataset NFCorpus
    python scripts/mteb_run.py --dataset SciFact --rebuild
    python scripts/mteb_run.py --dataset FiQA2018 --depth 100
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from tqdm import tqdm  # noqa: E402

from msmarco_retrieval.db import blob_to_vec, vec_to_blob  # noqa: E402
from msmarco_retrieval.embeddings import encode  # noqa: E402
from msmarco_retrieval.mteb_helpers import connect, dataset_paths  # noqa: E402
from msmarco_retrieval.text import tokenize  # noqa: E402


# ----------------------------- small utilities ------------------------------
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def metrics_for_query(
    ranked_pks: list[int], gold: dict[int, int], ks: tuple[int, ...]
) -> dict[str, float]:
    """gold maps passage_pk -> relevance grade (>=1 means relevant)."""
    out: dict[str, float] = {}
    relevant = {pk for pk, rel in gold.items() if rel > 0}
    first_hit = next((r + 1 for r, pk in enumerate(ranked_pks) if pk in relevant), None)

    # Pre-compute the ideal DCG once.
    ideal_rels = sorted(gold.values(), reverse=True)

    for k in ks:
        top_k = ranked_pks[:k]
        out[f"recall@{k}"] = 1.0 if any(pk in relevant for pk in top_k) else 0.0
        out[f"mrr@{k}"] = 1.0 / first_hit if (first_hit and first_hit <= k) else 0.0

        dcg = sum(
            (2 ** gold.get(pk, 0) - 1) / math.log2(rank + 2)
            for rank, pk in enumerate(top_k)
        )
        ideal = sum(
            (2 ** rel - 1) / math.log2(rank + 2)
            for rank, rel in enumerate(ideal_rels[:k])
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


# ------------------------- pipeline steps ----------------------------------
def step_load_jsonl(paths: dict[str, Path]) -> tuple[list[dict], list[dict], list[dict]]:
    data_dir = paths["data_dir"]
    corpus = load_jsonl(data_dir / "corpus.jsonl")
    queries = load_jsonl(data_dir / "queries.jsonl")
    # Pick whichever qrels split exists. Test is conventional.
    qrels_path = next(data_dir.glob("qrels-*.jsonl"), None)
    if qrels_path is None:
        raise SystemExit(f"No qrels-*.jsonl in {data_dir}. Run scripts/mteb_download.py first.")
    qrels = load_jsonl(qrels_path)
    print(f"Loaded  corpus={len(corpus):,}  queries={len(queries):,}  qrels={len(qrels):,}")
    return corpus, queries, qrels


def step_build_db(
    paths: dict[str, Path],
    corpus: list[dict],
    queries: list[dict],
    qrels: list[dict],
    rebuild: bool,
) -> None:
    db_path = paths["db_path"]
    if db_path.exists() and not rebuild:
        print(f"Reusing existing DB at {db_path} (pass --rebuild to recompute)")
        return
    if db_path.exists():
        db_path.unlink()

    conn = connect(db_path)
    print("Encoding corpus...")
    texts = [c["text"] for c in corpus]
    embeddings = encode(texts, batch_size=128, show_progress=True)

    with conn:
        conn.executemany(
            "INSERT INTO passages (id, external_id, text, embedding) VALUES (?, ?, ?, ?)",
            (
                (i + 1, c["_id"], c["text"], vec_to_blob(embeddings[i]))
                for i, c in enumerate(corpus)
            ),
        )
        conn.executemany(
            "INSERT INTO queries (id, external_id, text) VALUES (?, ?, ?)",
            ((i + 1, q["_id"], q["text"]) for i, q in enumerate(queries)),
        )

    # Map external ids -> our integer pks so qrels can reference them.
    p_pk = {ex_id: pk for pk, ex_id in conn.execute("SELECT id, external_id FROM passages")}
    q_pk = {ex_id: pk for pk, ex_id in conn.execute("SELECT id, external_id FROM queries")}
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO qrels (query_pk, passage_pk, relevance) VALUES (?, ?, ?)",
            (
                (q_pk[r["query-id"]], p_pk[r["corpus-id"]], r["score"])
                for r in qrels
                if r["query-id"] in q_pk and r["corpus-id"] in p_pk
            ),
        )

    conn.close()
    print(f"DB written -> {db_path}")


def step_build_bm25(paths: dict[str, Path], rebuild: bool) -> None:
    bm25_path = paths["bm25_path"]
    if bm25_path.exists() and not rebuild:
        print(f"Reusing existing BM25 index at {bm25_path}")
        return

    conn = connect(paths["db_path"])
    rows = conn.execute("SELECT id, text FROM passages ORDER BY id").fetchall()
    print(f"Tokenizing {len(rows):,} passages for BM25...")
    ids = [r[0] for r in rows]
    corpus_tokens = [tokenize(r[1]) for r in tqdm(rows)]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with bm25_path.open("wb") as f:
        pickle.dump({"ids": ids, "bm25": bm25}, f)
    print(f"BM25 -> {bm25_path}")


def step_eval(paths: dict[str, Path], samples: int, depth: int, rrf_k: int) -> None:
    KS = (1, 5, 10)
    conn = connect(paths["db_path"])

    # Load BM25.
    with paths["bm25_path"].open("rb") as f:
        bundle = pickle.load(f)
    bm25_ids: list[int] = bundle["ids"]
    bm25 = bundle["bm25"]

    # Load passage matrix.
    print("Loading passage embeddings...")
    rows = conn.execute(
        "SELECT id, text, embedding FROM passages WHERE embedding IS NOT NULL ORDER BY id"
    ).fetchall()
    p_ids = [r[0] for r in rows]
    p_texts = [r[1] for r in rows]
    mat = np.stack([blob_to_vec(r[2]) for r in rows])
    text_by_pk = dict(zip(p_ids, p_texts))

    # Load queries that actually have at least one positive qrel.
    queries = conn.execute(
        """
        SELECT q.id, q.text
        FROM queries q
        WHERE q.id IN (SELECT query_pk FROM qrels WHERE relevance > 0)
        ORDER BY q.id
        """
    ).fetchall()
    print(f"Evaluating on {len(queries):,} queries...")

    print("Encoding queries...")
    q_vecs = encode([q[1] for q in queries], batch_size=128, show_progress=True)

    agg: dict[str, list[dict[str, float]]] = {"BM25": [], "Dense": [], "Hybrid (RRF)": []}
    sample_rows = []

    for i, (q_pk, q_text) in enumerate(tqdm(queries)):
        gold_rows = conn.execute(
            "SELECT passage_pk, relevance FROM qrels WHERE query_pk = ?", (q_pk,)
        ).fetchall()
        gold = {pk: rel for pk, rel in gold_rows}

        bm25_scores = bm25.get_scores(tokenize(q_text))
        bm25_top = [bm25_ids[j] for j in np.argsort(-bm25_scores)[:depth]]

        dense_scores = mat @ q_vecs[i]
        dense_top = [p_ids[j] for j in np.argsort(-dense_scores)[:depth]]

        rrf: dict[int, float] = defaultdict(float)
        for rank, pk in enumerate(bm25_top):
            rrf[pk] += 1.0 / (rrf_k + rank + 1)
        for rank, pk in enumerate(dense_top):
            rrf[pk] += 1.0 / (rrf_k + rank + 1)
        hybrid_top = [pk for pk, _ in sorted(rrf.items(), key=lambda kv: -kv[1])[:depth]]

        agg["BM25"].append(metrics_for_query(bm25_top, gold, KS))
        agg["Dense"].append(metrics_for_query(dense_top, gold, KS))
        agg["Hybrid (RRF)"].append(metrics_for_query(hybrid_top, gold, KS))

        if i < samples:
            sample_rows.append(
                {
                    "query": q_text,
                    "gold": {pk for pk, rel in gold.items() if rel > 0},
                    "bm25": bm25_top[:3],
                    "dense": dense_top[:3],
                    "hybrid": hybrid_top[:3],
                }
            )

    table_rows = []
    for name in ("BM25", "Dense", "Hybrid (RRF)"):
        avg = {k: float(np.mean([m[k] for m in agg[name]])) for k in agg[name][0].keys()}
        table_rows.append((name, avg))

    paths["results_path"].parent.mkdir(parents=True, exist_ok=True)
    with paths["results_path"].open("w") as f:
        f.write(f"# {paths['results_path'].stem.replace('mteb_', '')} — retrieval summary\n\n")
        f.write(f"Corpus: **{len(p_ids):,}** passages.   Queries with qrels: **{len(queries):,}**.\n")
        f.write(f"Retrieval depth: {depth}.  RRF k = {rrf_k}.\n\n")
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
                    hit = " ✅" if pk in s["gold"] else ""
                    snippet = text_by_pk[pk][:200].replace("\n", " ").strip()
                    f.write(f"{rank}. `pk={pk}`{hit} — {snippet}…\n")
                f.write("\n")
            f.write("---\n\n")

    print(f"\nWrote {paths['results_path']}")
    print("\n" + fmt_table(table_rows, KS))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="MTEB dataset name (matches mteb_download.py)")
    p.add_argument("--rebuild", action="store_true", help="rebuild DB + BM25 even if cached")
    p.add_argument("--depth", type=int, default=100, help="retrieval depth per retriever")
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--samples", type=int, default=3, help="qualitative samples in the report")
    args = p.parse_args()

    paths = dataset_paths(args.dataset)
    print(f"\n=== MTEB run: {args.dataset} ===")
    corpus, queries, qrels = step_load_jsonl(paths)
    step_build_db(paths, corpus, queries, qrels, args.rebuild)
    step_build_bm25(paths, args.rebuild)
    step_eval(paths, samples=args.samples, depth=args.depth, rrf_k=args.rrf_k)


if __name__ == "__main__":
    main()
