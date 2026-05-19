"""TruLens demo — wrap a toy RAG app and score it with the RAG triad.

Run:
    uv run python trulens/demo.py

Top-to-bottom shape:
    1. Build a toy RAG app (in-memory passages + cosine retrieval + canned
       generator). The point isn't the app, it's that TruLens treats *any*
       Python callable as the thing under test.
    2. Define three feedback functions for the RAG triad.
    3. Wrap the app with TruApp, run a handful of queries, print scores.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from trulens.apps.basic import TruBasicApp
from trulens.core import Feedback, TruSession
from trulens.providers.litellm import LiteLLM


PASSAGES = [
    "George Orwell wrote the novel Nineteen Eighty-Four (1984), published in 1949.",
    "The Eiffel Tower was completed in 1889 for the Paris World's Fair.",
    "Vectara HHEM is a DeBERTa-v3 cross-encoder for hallucination detection.",
    "Mount Everest is 8,848 meters tall and sits on the Nepal-China border.",
]


# Toy retriever: bag-of-words cosine. Replace with the retrieval/ subproject's
# index for a real run.
def _retrieve(query: str, k: int = 2) -> list[str]:
    vocab = sorted({w.lower() for p in PASSAGES + [query] for w in p.split()})
    idx = {w: i for i, w in enumerate(vocab)}

    def vec(text: str) -> np.ndarray:
        v = np.zeros(len(vocab))
        for w in text.lower().split():
            if w in idx:
                v[idx[w]] += 1
        return v / (np.linalg.norm(v) + 1e-9)

    qv = vec(query)
    scored = sorted(((float(qv @ vec(p)), p) for p in PASSAGES), reverse=True)
    return [p for _, p in scored[:k]]


def rag_app(query: str) -> str:
    """The thing under test: retrieve then 'generate'.

    We use a canned template instead of a real LLM so the demo runs without
    a generator API key. The interesting LLM calls are the *judges* below.
    """
    ctx = _retrieve(query, k=2)
    return f"Based on the retrieved context: {ctx[0]}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/gpt-4o-mini",
                   help="LiteLLM-style model id for the feedback judge.")
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY") and args.model.startswith("openai/"):
        raise SystemExit("Set OPENAI_API_KEY (or point OPENAI_API_BASE at a "
                         "local OpenAI-compatible endpoint).")

    session = TruSession()
    session.reset_database()  # clean slate for the demo

    provider = LiteLLM(model_engine=args.model)

    # The classic RAG triad. Each is a Feedback object that runs the judge
    # prompt over recorded (input, output) pairs.
    f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons,
                              name="Groundedness").on_input_output()
    f_answer_rel = Feedback(provider.relevance_with_cot_reasons,
                            name="Answer Relevance").on_input_output()
    f_context_rel = Feedback(provider.context_relevance_with_cot_reasons,
                             name="Context Relevance").on_input_output()

    tru_app = TruBasicApp(
        rag_app,
        app_name="rag-eval-demo",
        feedbacks=[f_groundedness, f_answer_rel, f_context_rel],
    )

    queries = [
        "Who wrote 1984?",
        "When was the Eiffel Tower built?",
        "How tall is Mount Everest?",
    ]
    with tru_app as recording:
        for q in queries:
            print(f"Q: {q}")
            print(f"A: {tru_app.app(q)}\n")

    # Pull aggregate leaderboard.
    leaderboard = session.get_leaderboard(app_ids=[tru_app.app_id])
    print(leaderboard.to_string())


if __name__ == "__main__":
    main()
