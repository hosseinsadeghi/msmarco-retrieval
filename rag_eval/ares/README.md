# ARES — trained classifier for RAG evaluation

## What it is

[ARES](https://github.com/stanford-futuredata/ARES) (Saad-Falcon et al.,
NAACL 2024) is the third style of RAG eval, and the most unusual: instead of
using an LLM judge at scoring time, it **trains a small classifier** on
LLM-synthesized labeled data, then uses *that classifier* to score new
(query, context, answer) triples.

Workflow:

1. **Synthesize** — sample a small in-domain corpus, ask an LLM to write
   plausible (query, positive_doc, negative_doc) triples.
2. **Train** — fine-tune a DeBERTa / RoBERTa-scale model on the triples to
   predict context_relevance / answer_faithfulness / answer_relevance.
3. **Score** — at eval time, the classifier is deterministic, sub-second on
   CPU, and free.
4. **PPI (Prediction-Powered Inference)** — combine the cheap classifier
   scores with a small human-labeled sample to get **statistically valid
   confidence intervals**, not just point estimates.

## Why this exists

Ragas / TruLens both pay an LLM API call (or local GPU minute) for every
score. That's fine for offline batches but breaks down at three places:

- **CI gates** that need to run on every PR.
- **Per-release sweeps** over thousands of queries × multiple system variants.
- **No-network environments** (regulated industries, air-gapped clusters).

ARES front-loads the LLM cost into the synthetic-data step (you pay once),
then scoring becomes a forward pass through a 100M-parameter model. The PPI
trick is the part papers cite — it lets you trade a small amount of human
labeling effort for tight confidence intervals on system comparisons.

## When to reach for it

- You need an eval metric that runs **inside CI** without flaky API calls.
- You're comparing many system variants and the LLM-judge bill matters.
- You care about **confidence intervals**, not just point scores (PPI).

## When *not* to reach for it

- You only have ~50 queries — the trained classifier needs enough synthetic
  data to be useful; LLM-judge libraries don't.
- Your domain shifts often. The classifier is fine-tuned on a snapshot; if
  your query distribution drifts you need to re-train.
- You don't have a GPU for the (one-time) training step. ARES technically
  supports CPU training but it's slow enough to be annoying.

## Status of this folder

The demo is a **scaffold**: it walks through the ARES API (synthesize →
train → score) on a tiny inline corpus, but the synthesize and train steps
need an LLM API key and a GPU respectively. Treat it as a worked example,
not a one-command end-to-end run.
