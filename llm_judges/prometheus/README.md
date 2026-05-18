# Prometheus 2 — fine-grained LLM-as-judge

## What it is

[Prometheus 2](https://arxiv.org/abs/2405.01535) (KAIST, 2024) is an
**open-source LLM-as-judge model**. You give it (1) an instruction, (2) the
response you want to evaluate, (3) an optional reference answer, and (4) a
**custom rubric** with five score-level descriptions. It returns a verbal
critique and an integer score from 1 to 5 calibrated to the rubric.

The official lineage on HuggingFace:

| Model                                 | Base                 | Size | Best for                          |
| ---                                   | ---                  | ---  | ---                               |
| `kaist-ai/prometheus-7b-v1.0`         | LLaMA-2-Chat-7B      | 7B   | Legacy (v1) absolute scoring only |
| `kaist-ai/prometheus-13b-v1.0`        | LLaMA-2-Chat-13B     | 13B  | Legacy (v1)                       |
| `kaist-ai/prometheus-7b-v2.0`         | Mistral-7B-Instruct  | 7B   | **Current single-model default**  |
| `kaist-ai/prometheus-8x7b-v2.0`       | Mixtral-8x7B-Instruct | 47B (MoE) | **Current best — paper SOTA** |
| `kaist-ai/prometheus-bgb-8x7b-v2.0`   | Mixtral-8x7B          | 47B (MoE) | Bigger context, retraining |

Prometheus 2 supports **two evaluation modes**:

- **Absolute grading** (single response → 1–5 score on a rubric): the
  classic use case, replaces "ask GPT-4 to score this."
- **Relative grading** (two responses + rubric → "A" or "B" winner +
  critique): for preference data labelling.

There is also **Prometheus-Vision** (`kaist-ai/prometheus-vision-13b-v1.0`)
for multimodal/visual-instruction eval, and **Prometheus 2.5 / 3** variants
appearing in the wild — but the v2 series is what most papers cite and what
the official `prometheus-eval` Python package targets.

> **Are there other "Prometheus" models?** Yes — but they're unrelated:
> NVIDIA's Prometheus is a monitoring system; "Prometheus" on HuggingFace
> filtered to text models is mostly the KAIST line. For LLM evaluation,
> Prometheus 2 (above) is *the* one to know.

## Why this exists

By mid-2023 the field had collapsed around "ask GPT-4 to score it" as the
default eval pattern. That worked, but:

- **Cost**: paying OpenAI to evaluate your own model gets expensive at scale.
- **Reproducibility**: GPT-4's scoring drifts across snapshots and refuses
  some prompts.
- **Lock-in**: your eval pipeline depends on a proprietary API.

Prometheus is trained on the **Feedback Collection** (1k rubrics × 20k
instructions × responses scored 1–5 by GPT-4), then DPO-aligned in v2 with
the Preference Collection. The result is a 7B / 47B model that **agrees with
human raters about as well as GPT-4 does** on standard judge benchmarks,
runs locally, is fully reproducible, and is Apache-2 licensed.

## When to reach for it

- **Offline eval of LLM apps**: replace the GPT-4-as-judge line in your CI.
  Same dashboard, no API spend, deterministic numbers.
- **Building preference data for DPO/RLHF**: relative grading produces
  (chosen, rejected) pairs at high throughput without paying per pair.
- **Custom-rubric evaluation**: the killer feature. Your domain needs a
  weird rubric ("score 5 if the response correctly cites the relevant section
  of GAAP"), Prometheus follows it. Generic LLMs trained as judges don't
  cope as well with novel rubrics.

## When **not** to reach for it

- **Pairwise eval where you only care about win rate**, not a critique —
  PandaLM or Auto-J are also fine here and similar size.
- **Hallucination-only evals against a source document** — HHEM is 60× smaller
  and good enough for that narrow task.
- **CPU-only inference for many prompts** — you'll be waiting minutes per
  eval. Use HHEM or RouteLLM-as-classifier instead if quality scoring is
  cheap enough.

## What the demo does

`demo.py` runs both Prometheus 2 evaluation modes on the same scenario so
you can see them side by side. The scenario is intentionally tractable: a
3rd-grade-level question and two responses — one good, one with a subtle
factual error. The rubric awards points for "scientific accuracy across
grade level." A good judge model should:

1. **Absolute grading**: rate the bad response in the lower half of the
   rubric (1–3) and the good one in the upper half (4–5), and explain
   *why* in terms of the rubric.
2. **Relative grading**: prefer the good response and articulate the
   defect in the bad one.

Both calls use the [`prometheus-eval`](https://github.com/prometheus-eval/prometheus-eval)
Python package, which wraps the model behind a clean `AbsoluteGrader` /
`RelativeGrader` interface. By default the package loads via **vLLM** for
fast inference (GPU required). The demo also exposes a `--backend transformers`
mode that falls back to raw HuggingFace generation — slower, no vLLM, runs
on CPU in principle but realistically you want a GPU for any meaningful
throughput.

## Running

```bash
uv sync --extra prometheus

# Default: 7B model on GPU via vLLM (recommended)
uv run python prometheus/demo.py --model kaist-ai/prometheus-7b-v2.0

# Larger, paper-best model — needs ~96 GB VRAM in fp16
uv run python prometheus/demo.py --model kaist-ai/prometheus-8x7b-v2.0

# Pure-HF fallback (slow, no vLLM dependency)
uv run python prometheus/demo.py --backend transformers
```

First run downloads the model (~15 GB for the 7B, ~90 GB for the 8x7B).

## Reading the output

You get back two things per evaluation:

1. **A textual critique** — the model's reasoning, written against the
   rubric you supplied. This is what makes Prometheus more useful than a
   plain "score 4/5" classifier; you can audit *why* it judged what it did.
2. **A score** — 1–5 for absolute mode, "A"/"B" for relative mode.

For production eval, average the score across many examples (or use win
rate for relative mode). Anything below ~3 in absolute mode means the
response materially fails the rubric you defined.

## Rubric design tips

Prometheus is more sensitive to rubric quality than to model choice. A few
things that matter in practice:

- **Anchor each score level to an observable behavior**, not a vibe.
  "Score 3 = the response is partially correct but omits one key step" is
  better than "score 3 = adequate."
- **Keep the rubric to one dimension at a time** (correctness, OR
  conciseness, OR safety — not all three in one rubric). The model conflates
  axes otherwise.
- **Reference answers help a lot** for factual rubrics. They give the judge
  a concrete ground truth to compare against rather than relying on its own
  world knowledge.
