"""ARES demo — scaffold walking through synthesize -> train -> score.

This is a **scaffold**. The full pipeline needs:
  - an LLM API key for the synthetic-data step,
  - a GPU for the classifier training step.

Run (after editing the paths below to point at a real corpus):
    uv run python ares/demo.py --step synth
    uv run python ares/demo.py --step train
    uv run python ares/demo.py --step score
"""
from __future__ import annotations

import argparse
from pathlib import Path


HERE = Path(__file__).parent
DOCS = HERE / "docs.tsv"           # tab-separated: doc_id\ttext
QUERIES = HERE / "queries.tsv"     # tab-separated: query_id\ttext\tdoc_id (positive)
SYNTH_OUT = HERE / "synth.tsv"
CLASSIFIER_DIR = HERE / "classifier"


def step_synth(model: str) -> None:
    """Synthesize (query, positive, negative) training triples with an LLM."""
    from ares import ARES  # ares-ai

    ares = ARES(
        synthetic_query_generator_model_identifier=model,
        document_filepaths=[str(DOCS)],
    )
    ares.generate_synthetic_queries(
        synthetic_queries_filename=str(SYNTH_OUT),
        number_of_queries=200,
    )
    print(f"Wrote {SYNTH_OUT}")


def step_train() -> None:
    """Fine-tune a small classifier on the synthetic triples."""
    from ares import ARES

    ares = ARES(
        classifier_model_identifier="microsoft/deberta-v3-base",
        training_dataset=[str(SYNTH_OUT)],
        validation_set=[str(SYNTH_OUT)],   # toy: same file; use a holdout in practice
        label_column=["Context_Relevance_Label",
                      "Answer_Faithfulness_Label",
                      "Answer_Relevance_Label"],
        model_choice=str(CLASSIFIER_DIR),
        num_epochs=3,
    )
    ares.train_classifier()
    print(f"Trained classifier at {CLASSIFIER_DIR}")


def step_score() -> None:
    """Score a held-out (query, context, answer) set with the trained classifier."""
    from ares import ARES

    ares = ARES(
        classifier_model=str(CLASSIFIER_DIR),
        evaluation_datasets=[str(SYNTH_OUT)],   # toy: reuse synth as eval
        labels=["Context_Relevance_Label",
                "Answer_Faithfulness_Label",
                "Answer_Relevance_Label"],
    )
    results = ares.evaluate_RAG()
    print(results)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--step", choices=["synth", "train", "score"], required=True)
    p.add_argument("--model", default="gpt-4o-mini",
                   help="LLM for synthetic-query generation (--step synth only).")
    args = p.parse_args()

    if args.step == "synth":
        step_synth(args.model)
    elif args.step == "train":
        step_train()
    elif args.step == "score":
        step_score()


if __name__ == "__main__":
    main()
