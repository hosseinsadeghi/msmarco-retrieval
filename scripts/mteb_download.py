"""Download an MTEB / MTEB-v2 retrieval dataset (BEIR format) to data/mteb/<name>/.

Three JSONL files are written:
    corpus.jsonl         - {_id, text}        (title is folded into text)
    queries.jsonl        - {_id, text}
    qrels-<split>.jsonl  - {query-id, corpus-id, score}

For datasets not in the PRESETS map in mteb_helpers.py, pass --hf-id to point
at any other `mteb/*` repo. The corpus/queries/qrels subset names are nearly
universal across the mteb HF org.

Usage:
    python scripts/mteb_download.py --dataset NFCorpus
    python scripts/mteb_download.py --dataset SciFact
    python scripts/mteb_download.py --dataset MyCustom --hf-id mteb/my-custom
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset  # noqa: E402

from msmarco_retrieval.mteb_helpers import dataset_paths, resolve_hf_id  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="MTEB dataset name (e.g. NFCorpus, SciFact)")
    p.add_argument("--hf-id", help="override HuggingFace repo id (default: from PRESETS)")
    p.add_argument("--split", default="test", help="qrels split (test/dev)")
    args = p.parse_args()

    hf_id = resolve_hf_id(args.dataset, args.hf_id)
    out_dir = dataset_paths(args.dataset)["data_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {hf_id}  (->  {out_dir})")

    # corpus
    corpus = load_dataset(hf_id, "corpus", split="corpus")
    with (out_dir / "corpus.jsonl").open("w") as f:
        for ex in corpus:
            title = (ex.get("title") or "").strip()
            text = (ex.get("text") or "").strip()
            joined = f"{title}. {text}" if title else text
            f.write(json.dumps({"_id": str(ex["_id"]), "text": joined}) + "\n")

    # queries
    queries = load_dataset(hf_id, "queries", split="queries")
    with (out_dir / "queries.jsonl").open("w") as f:
        for ex in queries:
            f.write(json.dumps({"_id": str(ex["_id"]), "text": ex["text"]}) + "\n")

    # qrels
    qrels = load_dataset(hf_id, "default", split=args.split)
    with (out_dir / f"qrels-{args.split}.jsonl").open("w") as f:
        for ex in qrels:
            f.write(
                json.dumps(
                    {
                        "query-id": str(ex["query-id"]),
                        "corpus-id": str(ex["corpus-id"]),
                        "score": int(ex["score"]),
                    }
                )
                + "\n"
            )

    print(
        f"Wrote  corpus={len(corpus):,}  queries={len(queries):,}  "
        f"qrels={len(qrels):,}  ({args.split} split)"
    )


if __name__ == "__main__":
    main()
