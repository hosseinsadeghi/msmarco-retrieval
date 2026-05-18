"""Prometheus 2 demo: absolute scoring + relative (pairwise) scoring.

Same scenario for both modes so the outputs are directly comparable:

    Question:  a 3rd-grade science prompt
    Response A: factually correct, well-explained
    Response B: confidently states a subtle but wrong number

A useful judge should rate A higher in absolute mode and pick A in
relative mode, citing the defect in B.

We use the official `prometheus-eval` Python package, which wraps
Prometheus 2 behind `AbsoluteGrader` / `RelativeGrader`. By default it
loads with vLLM (needs a GPU). `--backend transformers` falls back to
plain HuggingFace generation — slow, but works everywhere.
"""
from __future__ import annotations

import argparse


# Scenario shared by both eval modes.
INSTRUCTION = (
    "Explain to a 3rd-grader why the sky looks blue during the day, in 2-3 "
    "sentences. Be scientifically accurate."
)

RESPONSE_GOOD = (
    "Sunlight is actually made of all the rainbow colors mixed together. When "
    "it travels through the air, the tiny gas molecules scatter the blue color "
    "more than the others, so when you look up, you see lots of blue light "
    "coming from every direction."
)

RESPONSE_FLAWED = (
    "Sunlight is made of all the rainbow colors. The air around us is made "
    "mostly of oxygen, which is naturally blue, and that's why the sky looks "
    "blue. At sunset, the oxygen turns red, which is why the sky is red then."
)

REFERENCE_ANSWER = (
    "Sunlight contains all visible colors. As it passes through Earth's "
    "atmosphere, shorter (blue) wavelengths are scattered by gas molecules "
    "more than longer wavelengths — this is Rayleigh scattering — so light "
    "from every direction in the sky appears predominantly blue."
)

# Rubric: 5 score levels, each anchored to an observable behavior.
# Prometheus is much more reliable when the rubric is concrete like this
# vs. vibes like "Score 5 = excellent."
RUBRIC = """
[Does the response correctly explain why the sky is blue at a 3rd-grade level?]
Score 1: Wrong mechanism (e.g. attributes color to a property the air doesn't have) and misleading.
Score 2: Mentions a real concept (sunlight, atmosphere) but the explanation is mostly wrong.
Score 3: Roughly correct gist (something about sunlight + air) but uses incorrect specifics like wrong gas or wrong color behavior.
Score 4: Correct mechanism (scattering of shorter wavelengths) but explained in language a 3rd-grader may not follow, or omits an analogy.
Score 5: Correct mechanism, age-appropriate, and the explanation is causally complete (light has many colors → scattered by atmosphere → blue dominates).
""".strip()


def run_absolute(grader, response: str) -> tuple[str, int]:
    """Single-response grading: returns (feedback, score 1..5)."""
    feedback, score = grader.single_absolute_grade(
        instruction=INSTRUCTION,
        response=response,
        rubric=RUBRIC,
        reference_answer=REFERENCE_ANSWER,
    )
    return feedback, int(score)


def run_relative(grader, response_a: str, response_b: str) -> tuple[str, str]:
    """Pairwise grading: returns (feedback, 'A'|'B')."""
    feedback, score = grader.single_relative_grade(
        instruction=INSTRUCTION,
        response_A=response_a,
        response_B=response_b,
        rubric=RUBRIC,
        reference_answer=REFERENCE_ANSWER,
    )
    return feedback, str(score)


def load_graders(model_name: str, backend: str):
    """Load both AbsoluteGrader and RelativeGrader with the chosen backend.

    The `prometheus-eval` package's high-level graders accept a `model`
    parameter that can be either a vLLM `LLM` instance or a HuggingFace
    transformers pipeline. We pick one based on `--backend`.
    """
    from prometheus_eval import AbsoluteGrader, RelativeGrader
    from prometheus_eval.prompts import ABSOLUTE_PROMPT, RELATIVE_PROMPT

    if backend == "vllm":
        from prometheus_eval.vllm import VLLM
        # vLLM gives ~10x throughput vs HF generate on the same GPU, which is
        # why prometheus-eval recommends it. Needs a GPU and CUDA.
        model = VLLM(model=model_name)
    elif backend == "transformers":
        from prometheus_eval.litellm import LiteLLM  # noqa: F401  (compat shim)
        # The 'transformers' path uses the LiteLLM-style wrapper around a
        # local HF pipeline. Slower but no extra runtime deps.
        import torch
        from transformers import pipeline
        pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        # Wrap into a minimal callable that .generate() expects. The package
        # supports custom callables — see its docs for the full interface.
        model = pipe
    else:
        raise SystemExit(f"Unknown backend {backend!r}")

    abs_grader = AbsoluteGrader(model=model, absolute_grade_template=ABSOLUTE_PROMPT)
    rel_grader = RelativeGrader(model=model, relative_grade_template=RELATIVE_PROMPT)
    return abs_grader, rel_grader


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="kaist-ai/prometheus-7b-v2.0",
        help="HuggingFace model id. v2 family: prometheus-7b-v2.0 (small) or "
             "prometheus-8x7b-v2.0 (paper SOTA, much bigger).",
    )
    p.add_argument(
        "--backend", choices=("vllm", "transformers"), default="vllm",
        help="vllm = fast, GPU required. transformers = slow fallback.",
    )
    args = p.parse_args()

    print(f"Loading {args.model} via {args.backend} ...")
    print("(first run downloads ~15 GB for the 7B model)")
    abs_grader, rel_grader = load_graders(args.model, args.backend)

    # -------- Absolute grading on each response separately ----------------
    print("\n" + "=" * 80)
    print("ABSOLUTE GRADING — each response judged on its own against the rubric")
    print("=" * 80)
    for label, response in [("GOOD", RESPONSE_GOOD), ("FLAWED", RESPONSE_FLAWED)]:
        print(f"\n--- Response: {label} ---")
        feedback, score = run_absolute(abs_grader, response)
        print(f"Score: {score}/5")
        print(f"Feedback:\n{feedback}")

    # -------- Relative grading (A vs B) -----------------------------------
    print("\n" + "=" * 80)
    print("RELATIVE GRADING — direct A vs B comparison against the rubric")
    print("=" * 80)
    print("A = GOOD, B = FLAWED")
    feedback, winner = run_relative(rel_grader, RESPONSE_GOOD, RESPONSE_FLAWED)
    print(f"\nWinner: {winner}  (a useful judge should pick A)")
    print(f"Feedback:\n{feedback}")


if __name__ == "__main__":
    main()
