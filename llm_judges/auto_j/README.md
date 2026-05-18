# Auto-J — critique-then-rate LLM judge

## What it is

[Auto-J](https://arxiv.org/abs/2310.05470) (GAIR Lab, SJTU, 2023) is a
**generative LLM judge** fine-tuned from LLaMA-2-13B-Chat. Like Prometheus
and PandaLM, it scores LLM outputs — but Auto-J is distinctive for two
things:

1. **Always produces a free-form critique first**, then a rating or
   winner. This is the "critique-then-rate" pattern: the model is forced
   to externalize its reasoning before committing to a score, which
   improves calibration on hard cases.
2. **Trained jointly on ~58 evaluation scenarios** spanning helpfulness,
   harmlessness, reasoning, factuality, code, math, role-play, etc. —
   so it generalizes across the kinds of tasks an LLM product actually
   sees, not just the narrow rubric distribution of a single judge.

It supports both **pairwise** ("which response is better?") and **single
response** ("score this on a 1–10 scale with critique") modes.

Checkpoints on HuggingFace:

| Model                       | Size  | Notes                                        |
| ---                         | ---   | ---                                          |
| `GAIR/autoj-13b`            | 13B   | English judge, primary release               |
| `GAIR/autoj-bilingual-6b`   | 6B    | English + Chinese, lighter weight             |

## Why this exists

Compared to its contemporaries:

- **PandaLM** is pairwise-only and trained on a narrower distribution.
- **GPT-4-as-judge** is expensive and not reproducible.
- **Prometheus 1** (the contemporary version) needed a custom rubric.

Auto-J's pitch was: "ship one model that handles 58 evaluation scenarios
out of the box, requires no per-task rubric, and always gives you a
critique to audit." That made it the practical default for general
"evaluate this chatbot output" use cases in late 2023, and it's still a
solid pick today as a one-stop judge.

## When to reach for it

- **General-purpose LLM eval pipelines** where you have heterogeneous
  prompts (some helpful-question, some safety-edge, some code) and don't
  want to maintain 58 rubrics.
- **You want a critique for every score**. Auto-J always produces one,
  not as an extra prompt, which makes downstream debugging much easier.
- **Chinese + English mixed traffic** — the bilingual 6B is one of the
  few open judges that ships explicit Chinese support.

## When **not** to reach for it

- **Highly custom rubrics** ("score 5 if response cites GAAP correctly").
  Prometheus is much better at following a bespoke rubric you wrote.
- **CPU-only or low-RAM** — 13B in fp16 needs ~26 GB. The 6B bilingual
  is more tractable. The 4-bit quantization path documented below brings
  the 13B down to ~7 GB but requires a CUDA GPU.

## What about 4-bit?

`bitsandbytes` lets you load Auto-J in 4-bit precision (NF4 quantization,
double quant), which:

- Cuts memory from ~26 GB → ~7 GB for the 13B model.
- Lets it fit on a single 24 GB consumer GPU (RTX 3090/4090) with room
  for the KV cache.
- Costs a small quality drop on the judge benchmarks (a few % accuracy)
  vs full fp16.

The catch is that **`bitsandbytes` only has CUDA kernels** — it won't
work on CPU, Apple Silicon, or AMD GPUs without ROCm. The demo treats
`--quantization 4bit` as a hard requirement-of-CUDA and falls back with
an informative error if CUDA isn't available.

## What the demo does

`demo.py` runs Auto-J in both of its modes on the same two responses, so
you can see how the same model handles "rate one" vs "compare two":

1. **Single-response evaluation** (1–10 + critique) of one chatbot
   answer to a moderately tricky factual question.
2. **Pairwise evaluation** (winner + critique) of that same answer
   against a worse competing answer.

Both use Auto-J's official prompt templates from its repo — they bake in
the expected output format and shouldn't be modified.

Quantization is exposed via `--quantization {none, 4bit, 8bit}`:

- `none`: load in fp16 (default; needs ~26 GB VRAM for the 13B).
- `4bit`: load via `bitsandbytes` NF4 (needs ~7 GB VRAM, CUDA only).
- `8bit`: same path, NF8 quantization (~14 GB, CUDA only).

## Running

```bash
uv sync --extra auto_j

# Default: fp16 13B model — needs a beefy GPU
uv run python auto_j/demo.py

# 4-bit for consumer GPU (24 GB)
uv run python auto_j/demo.py --quantization 4bit

# Smaller bilingual model (6B), fits on smaller GPUs
uv run python auto_j/demo.py --model GAIR/autoj-bilingual-6b
```

First run downloads the model from HuggingFace (~26 GB for 13B fp16; the
weights themselves are not quantized in the cache — quantization happens
at load time).

## Reading the output

Auto-J's outputs follow consistent shapes:

- **Single mode**: a long critique covering helpfulness, accuracy,
  clarity, etc., then a final line `Rating: [[N]]` where N ∈ 1..10.
  Anything ≥ 8 means "ship-quality" by the model's training distribution;
  4–7 means "useful but flawed"; ≤ 3 means "actively wrong / unsafe."

- **Pairwise mode**: critiques of both responses, then a verdict like
  `So Response 1 is better.` or `So both responses are equally good.`
  Same prompt twice with the responses swapped is the canonical
  position-bias check.

A common pattern in eval pipelines is to **use Auto-J in single mode for
absolute scoring on a 1–10 scale**, then aggregate to a mean as the
overall quality number, and **use it in pairwise mode for head-to-head
model bake-offs** where the relative ranking is what matters.
