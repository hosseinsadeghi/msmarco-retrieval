"""Load the JSONL produced by 01_download.py into SQLite + compute embeddings.

Tables filled:
  passages        — unique passage text + float32 embedding BLOB (cosine-normalized)
  queries         — query text and human-written answers (JSON-encoded list)
  query_passages  — link from each query to its candidate passages, with the
                    is_selected flag (useful for evaluating retrievers later)

Embeddings live in the passages table itself: the "database has indexed
embeddings" requirement is satisfied by load_all_passages(), which returns
a dense matrix ready for cosine search.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # noqa: E402

from msmarco_retrieval.config import DATA_DIR, DB_PATH  # noqa: E402
from msmarco_retrieval.db import connect, vec_to_blob  # noqa: E402
from msmarco_retrieval.embeddings import encode  # noqa: E402


def find_default_jsonl() -> Path:
    candidates = sorted(DATA_DIR.glob("msmarco_*.jsonl"))
    if not candidates:
        raise SystemExit(f"No msmarco_*.jsonl in {DATA_DIR}. Run scripts/01_download.py first.")
    return candidates[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=Path, default=None, help="path to msmarco_*.jsonl")
    p.add_argument("--batch", type=int, default=128, help="embedding batch size")
    args = p.parse_args()

    jsonl_path = args.jsonl or find_default_jsonl()
    print(f"Reading {jsonl_path}")

    # Pass 1: collect unique passages, queries, and the (query, passage, is_selected) links.
    passage_to_pk: dict[str, int] = {}
    rows_queries: list[tuple[str, str, str]] = []
    rows_links: list[tuple[int, str, int]] = []  # (query_idx, passage_text, is_selected)

    with jsonl_path.open() as f:
        for query_idx, line in enumerate(f):
            ex = json.loads(line)
            rows_queries.append(
                (str(ex.get("query_id", "")), ex["query"], json.dumps(ex.get("answers", [])))
            )
            for txt, sel in zip(ex["passages"]["passage_text"], ex["passages"]["is_selected"]):
                if txt not in passage_to_pk:
                    passage_to_pk[txt] = len(passage_to_pk) + 1  # 1-indexed
                rows_links.append((query_idx, txt, int(sel)))

    print(f"Unique passages: {len(passage_to_pk)}    queries: {len(rows_queries)}")

    # Pass 2: embed all passages in one batched run.
    passages_in_order = sorted(passage_to_pk.items(), key=lambda kv: kv[1])
    texts = [t for t, _ in passages_in_order]
    print("Encoding passages...")
    embeddings = encode(texts, batch_size=args.batch, show_progress=True)

    # Write to SQLite.
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = connect(DB_PATH)

    print("Inserting passages...")
    with conn:
        conn.executemany(
            "INSERT INTO passages (id, text, embedding) VALUES (?, ?, ?)",
            (
                (pk, txt, vec_to_blob(embeddings[i]))
                for i, (txt, pk) in enumerate(passages_in_order)
            ),
        )

    print("Inserting queries...")
    with conn:
        conn.executemany(
            "INSERT INTO queries (id, query_id, query, answers) VALUES (?, ?, ?, ?)",
            ((i + 1, qid, q, ans) for i, (qid, q, ans) in enumerate(rows_queries)),
        )

    print("Inserting query<->passage links...")
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO query_passages (query_pk, passage_pk, is_selected) VALUES (?, ?, ?)",
            (
                (q_idx + 1, passage_to_pk[txt], sel)
                for q_idx, txt, sel in tqdm(rows_links)
            ),
        )

    conn.close()
    print(f"\nDone. DB at: {DB_PATH}")


if __name__ == "__main__":
    main()
