# PandaLM — pairwise LLM judge (with rationale)

## What it is

[PandaLM](https://arxiv.org/abs/2306.05087) (Peking University &
collaborators, 2023) is one of the **earliest open-source LLM-as-judge
models**. It's a fine-tune of LLaMA-7B (and later LLaMA-2) on ~300k
preference pairs, trained to do one specific thing:

> Given an instruction and **two candidate responses**, output:
> - a winner: **`1`** (response 1 wins), **`2`** (response 2 wins), or **`Tie`**
> - a short **rationale**
> - optionally a **reference answer**

That's it — PandaLM is a **pairwise-only** judge. Unlike Prometheus, it
doesn't grade a single response against a rubric. Its job is to settle
the question "of these two, which is better?"

Checkpoints on HuggingFace:

| Model                              | Base       | Notes                                |
| ---                                | ---        | ---                                  |
| `WeOpenML/PandaLM-7B-v1`           | LLaMA-7B   | Original release, most-cited         |
| `WeOpenML/PandaLM-13B-v1`          | LLaMA-13B  | Bigger version, slightly stronger     |
| `WeOpenML/PandaLM-Alpaca-7B-v1`    | Alpaca-7B  | Alternate base                       |

## Why this exists

When PandaLM came out (mid-2023), the only credible LLM judge was GPT-4 via
API. The motivation was the same as Prometheus's a year later — let people
evaluate model outputs reproducibly, cheaply, locally — but the design was
narrower: just nail the pairwise case, since that's what RLHF/DPO data
labeling and head-to-head model comparisons actually need.

PandaLM's contribution was showing that a 7B model fine-tuned on
preference data could **agree with human raters ~70% of the time on
held-out pairwise judgments**, vs ~80% for GPT-4 — much closer to the
ceiling than naive 0-shot prompting of base LLaMA-7B (which was barely
above chance).

## When to reach for it

- **Pairwise preference data generation** for DPO / RLHF: you have two
  candidate responses per prompt and need a label. PandaLM is small,
  cheap, and matches human raters most of the time.
- **Head-to-head model bake-offs** ("does our fine-tune beat the
  base model on these 1000 prompts?") — PandaLM produces win rates.
- **As a free check against GPT-4-as-judge** to catch cases where the
  expensive judge is being inconsistent.

## When **not** to reach for it

- **Anything single-response** (scoring one output on its own). PandaLM
  wasn't trained for this; use Prometheus.
- **Highly specialized domains** (medical, legal) where the base
  LLaMA-7B's knowledge is thin. PandaLM inherits the base model's
  blind spots.
- **Multilingual eval** — trained mostly on English instruction data.

Today, **Prometheus 2 and Auto-J generally outperform PandaLM** on the
same pairwise task; PandaLM remains useful as a small, well-understood,
historically important baseline.

## What the demo does

`demo.py` runs PandaLM-7B on three pairwise scenarios that exercise
different failure modes:

1. **Clear quality gap**: a complete answer vs a one-word answer to the
   same question. The judge should pick the complete one.
2. **Subtle factual error**: two equally fluent answers, one of which
   gets a single number wrong. Tests whether the judge catches a
   non-stylistic defect.
3. **Tie case**: two equivalent-quality paraphrases of the same correct
   answer. A calibrated judge should call this a Tie or have low
   confidence either way.

For each case it prints the winner, the rationale, and (if PandaLM
generated one) the reference answer it would have written instead.

Implementation note: PandaLM is **not on PyPI as an installable judge
library**. Their official repo ships training/eval scripts; for
inference you load the model with plain HF `transformers` and apply
PandaLM's prompt template yourself. The demo does exactly that, which is
also clearer to read than going through a wrapper.

## Running

```bash
uv sync --extra pandalm

# Default: PandaLM-7B in fp16 on whatever device is available
uv run python pandalm/demo.py

# Force a specific checkpoint
uv run python pandalm/demo.py --model WeOpenML/PandaLM-13B-v1

# CPU-only (slow but works)
uv run python pandalm/demo.py --device cpu
```

First run downloads ~14 GB for the 7B model (~26 GB for 13B).

## Reading the output

PandaLM's expected output format is:

```
1            ← the winner: "1", "2", or "Tie"
<rationale>  ← one or two sentences explaining the choice
<reference>  ← optionally, what PandaLM thinks the ideal answer would be
```

In aggregate, what matters is:

- **Win rate** across many pairs (Response-A-wins fraction).
- **Tie rate** — if it's very high, your judge isn't discriminating well
  enough; if it's very low, your judge may be overconfident.
- **Bias check**: when you flip the order of (A, B), does the winner
  flip too? Most LLM judges, PandaLM included, have measurable
  position bias. The standard mitigation is to evaluate each pair twice
  (A,B) and (B,A) and only count agreement.
