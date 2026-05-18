"""Vectara HHEM-2.1-Open hallucination-detection demo.

The model is a small DeBERTa-v3-base cross-encoder. Given a (premise, hypothesis)
pair it returns a score in [0, 1] where higher = more factually consistent.

We do three things:

  1. Run HHEM on three hand-crafted (premise, hypothesis) categories — faithful,
     subtle hallucination, outright fabrication — and print the scores side by
     side so you can see how the model separates them.

  2. Apply a threshold to convert scores into binary "supported / hallucinated"
     decisions, mimicking how you'd wire HHEM into a RAG pipeline.

  3. (Optional, --no-benchmark to skip) run the same model on a few examples
     from HaluEval-Summarization on HuggingFace, to show the demo code is the
     same code you'd use for real evaluation.
"""
from __future__ import annotations

import argparse

from transformers import AutoModelForSequenceClassification

MODEL_ID = "vectara/hallucination_evaluation_model"


# Three categories of test pairs. Each item is (premise, hypothesis, expected_label).
# "supported" is what HHEM should say is faithful (~high score),
# "hallucinated" is what HHEM should flag (~low score).
INLINE_EXAMPLES = [
    # --- faithful: hypothesis is a clean rephrase of the premise --------------
    (
        "The Eiffel Tower, completed in 1889, stands 330 metres tall on the "
        "Champ de Mars in Paris.",
        "Built in 1889, the Eiffel Tower in Paris is 330 metres tall.",
        "supported",
    ),
    (
        "NVIDIA reported $26.0 billion in revenue for Q2 FY2024, a 101% "
        "increase year-over-year, driven by data center demand.",
        "NVIDIA's Q2 FY2024 revenue was $26 billion, roughly double the prior "
        "year, with data center as the main driver.",
        "supported",
    ),
    # --- subtle: numbers / dates / locations flipped while staying fluent ----
    (
        "The Eiffel Tower, completed in 1889, stands 330 metres tall on the "
        "Champ de Mars in Paris.",
        "Built in 1899, the Eiffel Tower in Lyon is 430 metres tall.",
        "hallucinated",
    ),
    (
        "NVIDIA reported $26.0 billion in revenue for Q2 FY2024, a 101% "
        "increase year-over-year, driven by data center demand.",
        "NVIDIA's Q2 FY2024 revenue was $36 billion, a 50% YoY increase, "
        "driven by gaming demand.",
        "hallucinated",
    ),
    # --- outright fabrication: claims with no anchor in the premise ----------
    (
        "The Eiffel Tower, completed in 1889, stands 330 metres tall on the "
        "Champ de Mars in Paris.",
        "Gustave Eiffel originally designed the tower as a temporary radio "
        "antenna for use during World War I.",
        "hallucinated",
    ),
]


def load_hhem() -> AutoModelForSequenceClassification:
    """HHEM ships custom code in the repo (a `predict` method that handles
    tokenization, batching, and pooling internally). We must pass
    `trust_remote_code=True` for that to load — it's the documented usage on
    the model card."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    model.eval()
    return model


def score_pairs(model, pairs: list[tuple[str, str]]) -> list[float]:
    """HHEM exposes a convenience `predict` method on the model object that
    takes a list of (premise, hypothesis) tuples and returns one score per pair.
    This is the API documented on the model card; we use it directly so the
    demo stays a few lines."""
    scores = model.predict(pairs)
    return [float(s) for s in scores]


def print_table(rows: list[dict], threshold: float) -> None:
    """Render the example results as a fixed-width table."""
    print(f"\n{'PREMISE'[:60]:<60}  {'HYPOTHESIS'[:60]:<60}  SCORE   DECISION   EXPECTED")
    print("-" * 160)
    for r in rows:
        decision = "SUPPORTED" if r["score"] >= threshold else "HALLUC"
        mark = "✓" if (decision == "SUPPORTED") == (r["expected"] == "supported") else "✗"
        print(
            f"{r['premise'][:60]:<60}  {r['hypothesis'][:60]:<60}  "
            f"{r['score']:.3f}   {decision:<9}  {r['expected']} {mark}"
        )


def run_benchmark_slice(model) -> None:
    """Pull a few examples from HaluEval-Summarization and run HHEM on them.

    HaluEval is a standard hallucination-detection benchmark: each example has
    a document (premise), a "right" summary (faithful), and a "hallucinated"
    summary. A useful model should score the right summary higher than the
    hallucinated one for the same document — that's the eyeball test below.
    """
    print("\n" + "=" * 80)
    print("Mini-benchmark: HaluEval-Summarization (5 examples)")
    print("=" * 80)
    from datasets import load_dataset

    ds = load_dataset("pminervini/HaluEval", "summarization", split="data")
    examples = ds.select(range(5))

    pairs = []
    rows = []
    for ex in examples:
        document = ex["document"]
        pairs.append((document, ex["right_summary"]))
        pairs.append((document, ex["hallucinated_summary"]))

    scores = score_pairs(model, pairs)
    print(f"\n{'#':<2}  RIGHT_SCORE   HALLUC_SCORE   GAP   ORDERED?")
    print("-" * 50)
    correct = 0
    for i in range(0, len(scores), 2):
        right, hallu = scores[i], scores[i + 1]
        ordered = right > hallu
        correct += ordered
        print(f"{i//2 + 1:<2}  {right:.3f}        {hallu:.3f}        "
              f"{right-hallu:+.3f}   {'✓' if ordered else '✗'}")
    print(f"\nCorrectly ordered: {correct}/5  (higher score for the faithful summary)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.5,
                   help="score >= threshold → flagged as supported")
    p.add_argument("--skip-benchmark", action="store_true",
                   help="skip the HaluEval slice (use if offline)")
    args = p.parse_args()

    print(f"Loading {MODEL_ID} ... (downloads ~400 MB on first run)")
    model = load_hhem()

    # 1. Score the inline examples.
    pairs = [(p, h) for p, h, _ in INLINE_EXAMPLES]
    scores = score_pairs(model, pairs)
    rows = [
        {"premise": p, "hypothesis": h, "expected": exp, "score": s}
        for (p, h, exp), s in zip(INLINE_EXAMPLES, scores)
    ]

    # 2. Threshold them.
    print("\nResults on inline examples")
    print_table(rows, threshold=args.threshold)
    n_correct = sum(
        (r["score"] >= args.threshold) == (r["expected"] == "supported")
        for r in rows
    )
    print(f"\nAccuracy at threshold {args.threshold}: {n_correct}/{len(rows)}")

    # 3. Optional benchmark slice.
    if not args.skip_benchmark:
        try:
            run_benchmark_slice(model)
        except Exception as e:  # network / dataset hiccups shouldn't crash the demo
            print(f"\n[Skipping benchmark slice: {e}]")


if __name__ == "__main__":
    main()
