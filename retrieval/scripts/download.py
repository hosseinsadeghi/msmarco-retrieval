"""Download the raw files for any registered dataset.

Dispatches on DatasetSpec.kind:
  - "msmarco": pulls N validation queries from microsoft/ms_marco -> data/<name>/raw.jsonl
  - "beir":    pulls corpus + queries + qrels from mteb/* -> three JSONL files

Examples:
    python scripts/download.py --dataset MSMARCO --num 1000
    python scripts/download.py --dataset NFCorpus
    python scripts/download.py --dataset SciFact --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset  # noqa: E402

from retrieval.datasets import dataset_paths, get  # noqa: E402


# ----------------------- per-kind download functions ----------------------- #
def download_msmarco(spec, out_dir: Path, num: int, config: str, split: str) -> None:
    print(f"Loading {spec.hf_id}  config={config}  split={split}")
    ds = load_dataset(spec.hf_id, config, split=split)
    n = min(num, len(ds))
    ds = ds.select(range(n))

    out_path = out_dir / "raw.jsonl"
    with out_path.open("w") as f:
        for ex in ds:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {n} examples -> {out_path}")

    # QA sanity check.
    sample = ds[0]
    print("\n--- QA sanity check ---")
    print(f"query_type : {sample.get('query_type')}")
    print(f"query      : {sample['query']}")
    print(f"answers    : {sample['answers']}")
    n_pass = len(sample["passages"]["passage_text"])
    n_sel = sum(sample["passages"]["is_selected"])
    print(f"passages   : {n_pass} candidates ({n_sel} flagged is_selected=1)")


def download_beir(spec, out_dir: Path, split: str) -> None:
    print(f"Loading {spec.hf_id}  (BEIR layout, qrels split={split})")
    corpus = load_dataset(spec.hf_id, "corpus", split="corpus")
    queries = load_dataset(spec.hf_id, "queries", split="queries")
    qrels = load_dataset(spec.hf_id, "default", split=split)

    with (out_dir / "corpus.jsonl").open("w") as f:
        for ex in corpus:
            title = (ex.get("title") or "").strip()
            text = (ex.get("text") or "").strip()
            joined = f"{title}. {text}" if title else text
            f.write(json.dumps({"_id": str(ex["_id"]), "text": joined}) + "\n")

    with (out_dir / "queries.jsonl").open("w") as f:
        for ex in queries:
            f.write(json.dumps({"_id": str(ex["_id"]), "text": ex["text"]}) + "\n")

    with (out_dir / f"qrels-{split}.jsonl").open("w") as f:
        for ex in qrels:
            f.write(
                json.dumps({
                    "query-id": str(ex["query-id"]),
                    "corpus-id": str(ex["corpus-id"]),
                    "score": int(ex["score"]),
                }) + "\n"
            )

    print(
        f"Wrote  corpus={len(corpus):,}  queries={len(queries):,}  "
        f"qrels={len(qrels):,}  ({split} split)"
    )


# --------------------------------- main ----------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="see scripts/list_datasets.py")
    p.add_argument("--num", type=int, default=1000, help="(MS MARCO only) number of queries")
    p.add_argument("--config", default="v2.1", help="(MS MARCO only) dataset config")
    p.add_argument(
        "--split",
        default=None,
        help="MS MARCO: train/validation/test (default validation). BEIR: test/dev (default test).",
    )
    args = p.parse_args()

    spec = get(args.dataset)
    out_dir = dataset_paths(spec)["data_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if spec.kind == "msmarco":
        split = args.split or "validation"
        download_msmarco(spec, out_dir, num=args.num, config=args.config, split=split)
    elif spec.kind == "beir":
        split = args.split or "test"
        download_beir(spec, out_dir, split=split)
    else:
        raise SystemExit(f"Unknown kind {spec.kind!r}")


if __name__ == "__main__":
    main()
