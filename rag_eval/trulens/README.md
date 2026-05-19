# TruLens — instrumentation-first RAG evaluation

## What it is

[TruLens](https://github.com/truera/trulens) (from TruEra, now Snowflake)
takes a different angle than Ragas: instead of giving you a `evaluate(ds,
metrics=[...])` function over a static dataset, it **wraps your live RAG
chain** and records every call. You then attach **feedback functions** that
score those recorded traces — in batch, or sampled in production.

The metrics themselves cover the same RAG triad as Ragas:

- `groundedness`     — answer claims supported by retrieved contexts
- `context_relevance` — retrieved contexts relevant to the query
- `answer_relevance`  — answer addresses the query

Plus a long tail of out-of-the-box feedback functions (toxicity, sentiment,
PII, custom prompts).

## Why this exists

Production RAG quality work needs:

1. **Traces of real user queries**, not just offline benchmarks.
2. **Per-component scoring** — which step of the chain went wrong?
3. **Sampling**, because you can't afford to LLM-judge every request.

Ragas covers (1)-style use cases as an afterthought (feed it a dataframe);
TruLens makes them the main thing. If you already have a LangChain /
LlamaIndex / custom RAG chain in code, TruLens is closer to "drop in" than
Ragas.

## When to reach for it

- You already have a RAG chain implemented and want feedback **without
  rewriting it** into an eval-friendly shape.
- You want to look at **per-call traces** in a dashboard, not just aggregate
  numbers (TruLens ships a Streamlit dashboard).
- You want to **sample 1-in-N** production requests for online quality
  monitoring.

## When *not* to reach for it

- One-shot offline ranking of "model A vs model B on a static benchmark" —
  Ragas's dataframe-in/dataframe-out API has less ceremony.
- You don't want a SQLite DB of traces sitting around. TruLens persists
  everything by default; cleaner for production, heavier for one-off scripts.

## Backend

Same story as Ragas: feedback functions call an LLM via LiteLLM. Default in
`demo.py` is `openai/gpt-4o-mini`; point `OPENAI_API_BASE` at a local server
to use Ollama / vLLM / etc.
