"""Build a BM25 index over every passage in the SQLite DB.

We tokenize each passage, fit BM25Okapi, and pickle (passage_ids, bm25) together
so the search script can map BM25's array offsets back to DB ids.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rank_bm25 import BM25Okapi  # noqa: E402
from tqdm import tqdm  # noqa: E402

from msmarco_retrieval.config import BM25_PATH  # noqa: E402
from msmarco_retrieval.db import connect  # noqa: E402
from msmarco_retrieval.text import tokenize  # noqa: E402


def main() -> None:
    conn = connect()
    rows = conn.execute("SELECT id, text FROM passages ORDER BY id").fetchall()
    if not rows:
        raise SystemExit("No passages in DB. Run scripts/02_build_db.py first.")

    ids = [r[0] for r in rows]
    print(f"Tokenizing {len(rows)} passages...")
    corpus_tokens = [tokenize(r[1]) for r in tqdm(rows)]

    print("Fitting BM25Okapi...")
    bm25 = BM25Okapi(corpus_tokens)

    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BM25_PATH.open("wb") as f:
        pickle.dump({"ids": ids, "bm25": bm25}, f)
    print(f"Saved BM25 index -> {BM25_PATH}")


if __name__ == "__main__":
    main()
