"""End-to-end RAG eval: retrieval/ + a small generator, scored with Ragas.

Run:
    uv run python end_to_end/demo.py --dataset NFCorpus --retriever hybrid --num 20

Top-to-bottom shape:
    1. Load (queries, qrels, passages) from the retrieval/ subproject's SQLite DB.
    2. Run the chosen retriever to get top-k contexts per query.
    3. Generate an answer with a small local LLM (default: flan-t5-base).
    4. Score the (q, contexts, answer, gt) rows with Ragas.
    5. Write a Markdown report next to this file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add the retrieval/ subproject to sys.path so we can reuse its code.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "retrieval" / "src"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="Dataset name from retrieval/src/retrieval/datasets.py "
                        "(e.g. NFCorpus, SciFact, MSMARCO).")
    p.add_argument("--retriever", choices=["bm25", "dense", "hybrid"],
                   default="hybrid")
    p.add_argument("--generator", default="google/flan-t5-base")
    p.add_argument("--judge", default="openai/gpt-4o-mini",
                   help="Ragas judge LLM (LiteLLM id).")
    p.add_argument("--num", type=int, default=20,
                   help="Number of queries to evaluate.")
    p.add_argument("--depth", type=int, default=5,
                   help="Top-k contexts to pass to the generator.")
    args = p.parse_args()

    # --- 1. Load passages, queries, qrels from retrieval/'s SQLite DB. ---
    # (Imports deferred so --help works without the retrieval extras installed.)
    from retrieval.config import db_path  # type: ignore
    from retrieval.io import load_eval_data  # type: ignore

    db = db_path(args.dataset)
    if not db.exists():
        raise SystemExit(
            f"DB not found: {db}\n"
            f"Build it first:  python retrieval/scripts/build.py --dataset {args.dataset}"
        )
    eval_data = load_eval_data(args.dataset, num=args.num)
    print(f"Loaded {len(eval_data.queries)} queries from {db}")

    # --- 2. Retrieve top-k contexts per query. ---
    from retrieval.retrieve import bm25_search, dense_search, rrf_fuse  # type: ignore

    def retrieve(q_text: str, q_vec) -> list[str]:
        if args.retriever == "bm25":
            ranked = bm25_search(eval_data.bm25, eval_data.bm25_ids, q_text, args.depth)
        elif args.retriever == "dense":
            ranked = dense_search(eval_data.dense_mat, eval_data.dense_ids, q_vec, args.depth)
        else:
            b = bm25_search(eval_data.bm25, eval_data.bm25_ids, q_text, args.depth * 4)
            d = dense_search(eval_data.dense_mat, eval_data.dense_ids, q_vec, args.depth * 4)
            ranked = rrf_fuse([b, d], rrf_k=60, depth=args.depth)
        return [eval_data.passage_text[pid] for pid, _ in ranked]

    # --- 3. Generate answers with a small local LLM. ---
    from transformers import pipeline
    gen = pipeline("text2text-generation", model=args.generator,
                   max_new_tokens=128, do_sample=False)

    rows = []
    for q in eval_data.queries:
        ctx = retrieve(q.text, q.vec)
        prompt = ("Answer the question using only the provided context.\n\n"
                  f"Context:\n{chr(10).join(ctx)}\n\nQuestion: {q.text}\nAnswer:")
        ans = gen(prompt)[0]["generated_text"].strip()
        rows.append({
            "question": q.text,
            "contexts": ctx,
            "answer": ans,
            "ground_truth": q.ground_truth or "",
        })

    # --- 4. Ragas scoring. ---
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (answer_relevancy, context_precision,
                                context_recall, faithfulness)
    from langchain_litellm import ChatLiteLLM

    judge = LangchainLLMWrapper(ChatLiteLLM(model=args.judge, temperature=0))
    result = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
    )

    # --- 5. Write a Markdown report. ---
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.dataset}_{args.retriever}.md"
    df = result.to_pandas()
    out.write_text(
        f"# {args.dataset} — {args.retriever} retriever + {args.generator}\n\n"
        f"Judge: `{args.judge}`. n = {len(df)}.\n\n"
        f"## Aggregate\n\n{result}\n\n"
        f"## Per-row\n\n{df.to_markdown(index=False)}\n"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
