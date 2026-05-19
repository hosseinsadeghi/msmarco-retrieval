# RAG evaluation — joint retriever + generator quality

The companion folders in this repo each cover one side of the RAG equation:

- [`retrieval/`](../retrieval/) — *did we fetch the right passages?* (Recall@k,
  nDCG@k against qrels)
- [`llm_judges/`](../llm_judges/) — *is this single LLM answer good?* (judge
  models, hallucination classifiers)

This folder is what sits *between* them: given a **question**, a set of
**retrieved passages**, and a **generated answer**, decide whether the joint
system is doing the right thing. That's the metric a user actually feels.

## The "RAG triad"

Most modern RAG-eval frameworks converge on the same three questions
(sometimes called the *RAG triad*):

1. **Context relevance** — were the retrieved passages relevant to the query?
   *(This is what `retrieval/` already measures, but here without qrels.)*
2. **Groundedness / faithfulness** — is every claim in the answer supported by
   those retrieved passages? *(Closely related to what HHEM measures.)*
3. **Answer relevance** — does the answer actually address the question?

A failure on (1) is a retrieval problem. A failure on (2) is a hallucination.
A failure on (3) is the generator going off-topic.

## What's in each subfolder

| Folder | Library | Method | LLM needed? |
| --- | --- | --- | --- |
| [`ragas/`](ragas/) | [Ragas](https://github.com/explodinggradients/ragas) | LLM-as-judge metrics (faithfulness, answer_relevancy, context_precision, context_recall). | Yes — OpenAI / Anthropic / local. |
| [`trulens/`](trulens/) | [TruLens](https://github.com/truera/trulens) | Instrumentation + feedback functions; same triad framing. | Yes (judge backend). |
| [`ares/`](ares/) | [ARES](https://github.com/stanford-futuredata/ARES) | *Trained* PEFT classifier on synthetic queries — no LLM judge at scoring time. | At train time only. |
| [`end_to_end/`](end_to_end/) | — | Wires `retrieval/` (BM25 / dense / hybrid) into a small generator LLM, then scores the resulting (q, contexts, answer) triples with the metrics above. | Yes for generation. |

## Why three libraries

They cover the same triad but make different operational trade-offs:

- **Ragas** is the de-facto default — Pythonic, integrates with `datasets`,
  works with any LiteLLM-compatible backend. Best starting point.
- **TruLens** is *instrumentation-first*: wrap your existing chain and it
  records every call, so you can run feedback on production traces, not just
  offline batches. Right tool when you already have a chain in code.
- **ARES** is the odd one out: it trains a small classifier on
  LLM-synthesized (query, positive, negative) triples, so once trained,
  scoring is cheap and deterministic. Right tool when you want a CI metric
  that doesn't depend on a paid API.

## Install

Each metric library has its own extra; only pull what you need:

```bash
cd rag_eval

uv sync --extra ragas      # Ragas + datasets + LiteLLM
uv sync --extra trulens    # TruLens core
uv sync --extra ares       # ARES + its torch / PEFT deps
uv sync --extra all        # everything
```

## Run a demo

```bash
uv run python ragas/demo.py
uv run python trulens/demo.py
uv run python ares/demo.py
uv run python end_to_end/demo.py --dataset NFCorpus --retriever hybrid
```

The `end_to_end/` demo is the interesting one: it picks one dataset from the
`retrieval/` registry, runs the chosen retriever, asks a small local LLM to
answer using the top-k passages, then prints the four Ragas metrics side by
side with the retrieval metrics. This is the only place in the repo where
the two halves meet.

## A note on cost and determinism

Every LLM-judge metric is non-deterministic and costs API calls. Realistic
batch sizes for an offline eval are 50–500 queries, not the full 1k+ test
sets in `retrieval/`. The demos default to a 20-row slice; bump
`--num` for a real run, and pin a `temperature=0` judge model in
`config.py` to keep scores comparable across runs.
