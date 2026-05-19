"""Ragas demo — score a small inline RAG sample with the four core metrics.

Run:
    uv run python ragas/demo.py
    uv run python ragas/demo.py --model openai/gpt-4o-mini --num 5

Top-to-bottom shape:
    1. Build a tiny eval dataset (question / contexts / answer / ground_truth).
    2. Wrap a judge LLM via LiteLLM so we can swap OpenAI for a local server.
    3. Call ragas.evaluate(...) with the four core metrics.
    4. Pretty-print the per-row + aggregate scores.
"""
from __future__ import annotations

import argparse
import os

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


# ---------------------------------------------------------------------------
# 1. Tiny inline sample. Swap this for a slice of e.g. `mteb/nfcorpus` or your
#    own production-trace dump for a real eval. Keys are the column names
#    Ragas expects.
# ---------------------------------------------------------------------------
SAMPLE = [
    {
        "question": "Who wrote the novel '1984'?",
        "contexts": [
            "Nineteen Eighty-Four, often published as 1984, is a dystopian "
            "novel by the English author George Orwell, first published in 1949.",
        ],
        "answer": "George Orwell wrote 1984; it was published in 1949.",
        "ground_truth": "George Orwell.",
    },
    {
        "question": "What does HHEM stand for?",
        "contexts": [
            "Vectara HHEM (Hallucination Evaluation Model) is a small "
            "DeBERTa-v3 cross-encoder that scores factual consistency.",
        ],
        "answer": "HHEM stands for Hallucination Evaluation Model.",
        "ground_truth": "Hallucination Evaluation Model.",
    },
    {
        # Faithfulness should drop here — the answer invents a year.
        "question": "When was the Eiffel Tower completed?",
        "contexts": [
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ "
            "de Mars in Paris, France, named after the engineer Gustave Eiffel.",
        ],
        "answer": "The Eiffel Tower was completed in 1850 in Paris.",
        "ground_truth": "1889.",
    },
]


def build_dataset(num: int) -> Dataset:
    rows = SAMPLE[:num] if num else SAMPLE
    return Dataset.from_list(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/gpt-4o-mini",
                   help="LiteLLM-style model id (e.g. openai/gpt-4o-mini, "
                        "anthropic/claude-haiku-4-5-20251001).")
    p.add_argument("--num", type=int, default=len(SAMPLE),
                   help="Number of rows to score (default: all inline samples).")
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY") and args.model.startswith("openai/"):
        raise SystemExit("Set OPENAI_API_KEY (or point OPENAI_API_BASE at a "
                         "local OpenAI-compatible endpoint).")

    # LangchainLLMWrapper is Ragas's adapter around any LangChain ChatModel;
    # we use the LiteLLM chat wrapper so the same code points at OpenAI,
    # Anthropic, Ollama, vLLM, etc.
    from langchain_litellm import ChatLiteLLM
    judge = LangchainLLMWrapper(ChatLiteLLM(model=args.model, temperature=0))

    ds = build_dataset(args.num)
    print(f"Scoring {len(ds)} rows with judge={args.model}\n")

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy,
                 context_precision, context_recall, answer_correctness],
        llm=judge,
    )

    # `result` is a ragas EvaluationResult; .to_pandas() gives per-row scores,
    # the object's own repr shows aggregates.
    print(result)
    print("\nPer-row:")
    print(result.to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
