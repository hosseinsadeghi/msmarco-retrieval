"""RouteLLM demo: predict which prompts need the strong model.

We load a pre-trained router and run it over a curated set of prompts that
span a clear complexity range. For each prompt the router returns a
`strong_win_rate ∈ [0, 1]` — the probability that the strong model would
beat the weak model on this prompt (calibrated against Chatbot Arena
preferences).

Then we sweep across thresholds and report how many prompts each threshold
would route to the strong model. This is the knob you'd actually tune in
production: pick a threshold that hits your target spend / quality mix.

The demo does *not* call any LLM. The point is to see what the router
decides, which is the interesting and reusable bit.
"""
from __future__ import annotations

import argparse
import os

# RouteLLM's `similarity_weighted` router module instantiates an `OpenAI()`
# client at import time (for embedding-based similarity), and that requires
# a key — even though we only use the `mf` / `bert` routers here, which do
# NOT call OpenAI. Set a placeholder so the import succeeds. If you actually
# want the sw_ranking router, swap this for a real key.
os.environ.setdefault("OPENAI_API_KEY", "unused-by-mf-and-bert-routers")

# Prompts span a clear complexity gradient. The router *should* assign these
# monotonically higher `strong_win_rate` as you go down the list. Real prompt
# mixes are messier, but this is enough to sanity-check the router's behavior.
PROMPTS = [
    # --- trivial: any weak model handles these ----------------------------
    "What is the capital of France?",
    "Convert 100 fahrenheit to celsius.",
    "Translate 'thank you' to Spanish.",
    # --- medium: factual but multi-step or specialized -------------------
    "Write a Python one-liner that returns the n-th Fibonacci number.",
    "Summarize the plot of Pride and Prejudice in three sentences.",
    "Why does my React component re-render when its parent's state changes?",
    # --- hard: needs strong reasoning, domain depth, or long output ------
    "Derive the Black-Scholes equation from first principles, explaining each "
    "assumption you make about the underlying asset's dynamics.",
    "Design a distributed consensus protocol that tolerates Byzantine failures "
    "in a partially synchronous network, and prove its safety.",
    "Read the following 1000-word financial filing excerpt and identify any "
    "subtle inconsistencies between the cash-flow statement and the income "
    "statement that an auditor should flag.",
]


# RouteLLM exposes routers via a class registry keyed by short name. The
# `mf` router is matrix-factorization — small, fast, no GPU.
ROUTER_CHECKPOINTS = {
    "mf":   "routellm/mf_gpt4_augmented",
    "bert": "routellm/bert_gpt4_augmented",
}


def load_router(name: str):
    """Load a pre-trained RouteLLM router by short name.

    RouteLLM's `Controller` class is designed to also call the underlying LLMs;
    we don't need that here — only the router's score function. We instantiate
    the router class directly with the HF checkpoint.
    """
    from routellm.routers.routers import ROUTER_CLS

    if name not in ROUTER_CLS:
        raise SystemExit(f"Unknown router {name!r}. Available: {sorted(ROUTER_CLS)}")
    if name not in ROUTER_CHECKPOINTS:
        raise SystemExit(
            f"This demo only ships checkpoints for: {sorted(ROUTER_CHECKPOINTS)}.\n"
            f"Other routers exist in the library but need their own training."
        )

    router_cls = ROUTER_CLS[name]
    # All routers in the library accept a checkpoint_path kwarg pointing at
    # the HF repo with the pre-trained weights.
    return router_cls(checkpoint_path=ROUTER_CHECKPOINTS[name])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--router", choices=sorted(ROUTER_CHECKPOINTS), default="bert",
        help="which pre-trained router to load (default: bert — fully local). "
             "The 'mf' router internally calls the OpenAI embeddings API at "
             "inference time, so it needs OPENAI_API_KEY set to a real key.",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="single-threshold cutoff (separate from the sweep table)",
    )
    args = p.parse_args()

    print(f"Loading router {args.router!r} from {ROUTER_CHECKPOINTS[args.router]}")
    print("(first run downloads the checkpoint to the HF cache)\n")
    router = load_router(args.router)

    # --- 1. Score each prompt and print -----------------------------------
    print(f"{'WIN_RATE':>9}  {'DECISION@'+str(args.threshold):>14}  PROMPT")
    print("-" * 110)
    win_rates: list[float] = []
    for prompt in PROMPTS:
        # `calculate_strong_win_rate` is the routers' single-prompt scoring API.
        wr = float(router.calculate_strong_win_rate(prompt))
        win_rates.append(wr)
        decision = "STRONG" if wr >= args.threshold else "weak  "
        # Truncate prompts for the table; show the routing decision.
        snippet = prompt if len(prompt) <= 80 else prompt[:77] + "..."
        print(f"{wr:>9.3f}  {decision:>14}  {snippet}")

    # --- 2. Threshold sweep -----------------------------------------------
    # This is the calibration table you'd use in production: pick the
    # threshold that hits your "% of traffic to strong" budget.
    print("\nThreshold sweep (fraction of these prompts routed to STRONG):\n")
    print(f"{'THRESHOLD':>10}  {'% TO STRONG':>12}  ROUTED PROMPTS (indices)")
    print("-" * 60)
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        routed = [i + 1 for i, wr in enumerate(win_rates) if wr >= tau]
        pct = 100 * len(routed) / len(win_rates)
        print(f"{tau:>10.2f}  {pct:>11.1f}%  {routed}")

    print(
        "\nTip: in production, run this sweep over a representative sample of"
        " your *own* prompt mix\nto pick the threshold that hits your target"
        " quality-vs-cost tradeoff."
    )


if __name__ == "__main__":
    main()
