"""Download MS MARCO and confirm it's a QA dataset.

We use the HuggingFace `microsoft/ms_marco` dataset (v2.1 by default). Each
example has:
  - query           : the natural-language question
  - answers         : list of human-written answer strings
  - passages        : dict of parallel lists (passage_text, is_selected, url)
  - query_type      : description / numeric / entity / location / person

`is_selected == 1` marks passages a human picked as containing the answer.

The script writes a JSONL file to data/msmarco_<split>_<n>.jsonl and prints a
sample so you can eyeball that questions/answers actually look like QA data.

Usage:
  python scripts/01_download.py                   # default: 1000 validation queries
  python scripts/01_download.py --num 5000
  python scripts/01_download.py --split train --num 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset  # noqa: E402

from msmarco_retrieval.config import (  # noqa: E402
    DATA_DIR,
    DEFAULT_CONFIG,
    DEFAULT_NUM_QUERIES,
    DEFAULT_SPLIT,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG, help="MS MARCO config (v1.1 or v2.1)")
    p.add_argument("--split", default=DEFAULT_SPLIT, help="train / validation / test")
    p.add_argument("--num", type=int, default=DEFAULT_NUM_QUERIES, help="number of queries to keep")
    args = p.parse_args()

    print(f"Loading microsoft/ms_marco config={args.config} split={args.split} ...")
    ds = load_dataset("microsoft/ms_marco", args.config, split=args.split, streaming=False)

    n = min(args.num, len(ds))
    ds = ds.select(range(n))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"msmarco_{args.split}_{n}.jsonl"
    with out_path.open("w") as f:
        for ex in ds:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {n} examples -> {out_path}")

    # --- QA sanity check ---------------------------------------------------
    print("\n--- QA sanity check ---")
    sample = ds[0]
    print(f"query_type : {sample.get('query_type')}")
    print(f"query      : {sample['query']}")
    print(f"answers    : {sample['answers']}")
    n_passages = len(sample["passages"]["passage_text"])
    n_selected = sum(sample["passages"]["is_selected"])
    print(f"passages   : {n_passages} candidates, {n_selected} marked is_selected=1")
    print("\nFirst selected passage:")
    for txt, sel in zip(sample["passages"]["passage_text"], sample["passages"]["is_selected"]):
        if sel == 1:
            print(f"  {txt[:300]}...")
            break
    print(
        "\n=> Each example has a question, a human-written answer, and passages a human "
        "marked as containing the answer. This is a QA dataset."
    )


if __name__ == "__main__":
    main()
