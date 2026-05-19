# Ragas — LLM-as-judge metrics for RAG

## What it is

[Ragas](https://github.com/explodinggradients/ragas) is the de-facto default
library for offline RAG evaluation. It packages the **RAG triad** (and a few
extras) as drop-in metrics that score `(question, contexts, answer,
ground_truth?)` tuples:

| Metric | Needs ground truth? | What it measures |
| --- | --- | --- |
| `faithfulness` | No | Every claim in `answer` decomposable & supported by `contexts`. |
| `answer_relevancy` | No | `answer` actually addresses `question` (uses synthetic-question round-trip). |
| `context_precision` | Yes | Of retrieved contexts, how many are relevant to the question. |
| `context_recall` | Yes | Of the info needed to produce `ground_truth`, how much is in `contexts`. |
| `answer_correctness` | Yes | Semantic + factual overlap between `answer` and `ground_truth`. |

Internally each metric is an LLM prompt — Ragas just gives you a clean
interface, batching, and a `datasets`-friendly output schema.

## Why this exists

Before Ragas, every RAG team rolled their own "ask GPT-4 if the answer is
faithful" script. Ragas standardizes that prompt, decouples the backend
(LiteLLM under the hood, so you can swap OpenAI for local vLLM without
changing the metric code), and converges with the published literature on
how each prompt should be phrased.

## When to reach for it

- **Offline batch eval** of a RAG system or release candidate.
- **Comparing two retrievers** on the same generator (or vice versa) — Ragas
  decomposes the signal so you can tell *which half* regressed.
- **CI gating**: pin a small (50–200 query) golden set and fail the build if
  `faithfulness` drops by > X.

## When *not* to reach for it

- **Production inline scoring** — every metric is one or more LLM calls; too
  slow & expensive for every user request. Use HHEM (in `llm_judges/`) or
  TruLens with sampling instead.
- **Datasets without a clear `contexts` field** — Ragas assumes a retrieve-then-
  generate flow. For closed-book QA, the context-side metrics don't apply.

## Backend

`demo.py` defaults to **`gpt-4o-mini`** via LiteLLM (set `OPENAI_API_KEY`).
To use a local vLLM endpoint:

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=dummy
uv run python ragas/demo.py --model openai/Qwen2.5-7B-Instruct
```

Same code path; Ragas / LiteLLM just talks OpenAI-compatible JSON to your
local server.
