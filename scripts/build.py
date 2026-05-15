"""Build the SQLite DB + embeddings + BM25 index for any registered dataset.

Reads whatever scripts/download.py wrote to data/<name>/ and produces:
  - db/<name>.sqlite      (passages + embeddings + queries + qrels)
  - indexes/<name>_bm25.pkl

Both flavors land in the same schema, so search.py and eval.py are
kind-agnostic from this point on.

Examples:
    python scripts/build.py --dataset MSMARCO
    python scripts/build.py --dataset NFCorpus --rebuild
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rank_bm25 import BM25Okapi  # noqa: E402
from tqdm import tqdm  # noqa: E402

from msmarco_retrieval.datasets import dataset_paths, get  # noqa: E402
from msmarco_retrieval.db import connect, vec_to_blob  # noqa: E402
from msmarco_retrieval.embeddings import encode  # noqa: E402
from msmarco_retrieval.text import tokenize  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


# ----------------------- per-kind build functions ------------------------- #
def build_msmarco(paths: dict[str, Path], batch_size: int) -> None:
    raw_path = paths["data_dir"] / "raw.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} not found. Run scripts/download.py first.")
    raw = load_jsonl(raw_path)

    # 1) Dedupe passages by text, assign integer pks, build qrels from is_selected.
    passage_to_pk: dict[str, int] = {}
    rows_queries: list[tuple[int, str, str, str]] = []  # (pk, external_id, text, answers_json)
    rows_qrels: list[tuple[int, int, int]] = []

    for i, ex in enumerate(raw):
        q_pk = i + 1
        rows_queries.append((
            q_pk,
            str(ex.get("query_id") or f"q{q_pk}"),
            ex["query"],
            json.dumps(ex.get("answers", [])),
        ))
        for txt, sel in zip(ex["passages"]["passage_text"], ex["passages"]["is_selected"]):
            if txt not in passage_to_pk:
                passage_to_pk[txt] = len(passage_to_pk) + 1
            rows_qrels.append((q_pk, passage_to_pk[txt], int(sel)))

    print(f"Unique passages: {len(passage_to_pk):,}    queries: {len(rows_queries):,}")

    # 2) Encode passages in one batched run.
    passages_in_order = sorted(passage_to_pk.items(), key=lambda kv: kv[1])
    texts = [t for t, _ in passages_in_order]
    print("Encoding passages...")
    embeddings = encode(texts, batch_size=batch_size, show_progress=True)

    # 3) Insert into unified schema.
    conn = connect(paths["db_path"])
    with conn:
        conn.executemany(
            "INSERT INTO passages (id, external_id, text, embedding) VALUES (?, ?, ?, ?)",
            (
                (pk, f"p{pk}", txt, vec_to_blob(embeddings[i]))
                for i, (txt, pk) in enumerate(passages_in_order)
            ),
        )
        conn.executemany(
            "INSERT INTO queries (id, external_id, text, answers) VALUES (?, ?, ?, ?)",
            rows_queries,
        )
        conn.executemany(
            "INSERT OR IGNORE INTO qrels (query_pk, passage_pk, relevance) VALUES (?, ?, ?)",
            rows_qrels,
        )
    conn.close()


def build_beir(paths: dict[str, Path], batch_size: int) -> None:
    data_dir = paths["data_dir"]
    corpus = load_jsonl(data_dir / "corpus.jsonl")
    queries = load_jsonl(data_dir / "queries.jsonl")
    qrels_path = next(data_dir.glob("qrels-*.jsonl"), None)
    if qrels_path is None:
        raise SystemExit(f"No qrels-*.jsonl in {data_dir}. Run scripts/download.py first.")
    qrels = load_jsonl(qrels_path)
    print(f"Loaded  corpus={len(corpus):,}  queries={len(queries):,}  qrels={len(qrels):,}")

    print("Encoding corpus...")
    texts = [c["text"] for c in corpus]
    embeddings = encode(texts, batch_size=batch_size, show_progress=True)

    conn = connect(paths["db_path"])
    with conn:
        conn.executemany(
            "INSERT INTO passages (id, external_id, text, embedding) VALUES (?, ?, ?, ?)",
            (
                (i + 1, c["_id"], c["text"], vec_to_blob(embeddings[i]))
                for i, c in enumerate(corpus)
            ),
        )
        conn.executemany(
            "INSERT INTO queries (id, external_id, text, answers) VALUES (?, ?, ?, NULL)",
            ((i + 1, q["_id"], q["text"]) for i, q in enumerate(queries)),
        )

    # Map external string ids -> our integer pks so we can store qrels.
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


def build_bm25(paths: dict[str, Path]) -> None:
    conn = connect(paths["db_path"])
    rows = conn.execute("SELECT id, text FROM passages ORDER BY id").fetchall()
    print(f"Tokenizing {len(rows):,} passages for BM25...")
    ids = [r[0] for r in rows]
    tokens = [tokenize(r[1]) for r in tqdm(rows)]
    bm25 = BM25Okapi(tokens)
    paths["bm25_path"].parent.mkdir(parents=True, exist_ok=True)
    with paths["bm25_path"].open("wb") as f:
        pickle.dump({"ids": ids, "bm25": bm25}, f)
    print(f"BM25 -> {paths['bm25_path']}")


# --------------------------------- main ----------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="see scripts/list_datasets.py")
    p.add_argument("--batch", type=int, default=128, help="embedding batch size")
    p.add_argument("--rebuild", action="store_true", help="recompute even if artifacts exist")
    args = p.parse_args()

    spec = get(args.dataset)
    paths = dataset_paths(spec)

    if paths["db_path"].exists() and not args.rebuild:
        print(f"Reusing existing DB at {paths['db_path']} (pass --rebuild to recompute)")
    else:
        if paths["db_path"].exists():
            paths["db_path"].unlink()
        if spec.kind == "msmarco":
            build_msmarco(paths, args.batch)
        elif spec.kind == "beir":
            build_beir(paths, args.batch)
        else:
            raise SystemExit(f"Unknown kind {spec.kind!r}")
        print(f"DB written -> {paths['db_path']}")

    if paths["bm25_path"].exists() and not args.rebuild:
        print(f"Reusing existing BM25 index at {paths['bm25_path']}")
    else:
        build_bm25(paths)


if __name__ == "__main__":
    main()
