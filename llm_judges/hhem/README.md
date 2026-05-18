# Vectara HHEM-2.1-Open — hallucination detection

## What it is

**HHEM = Hallucination Evaluation Model**. A small classifier (≈110M
parameters, DeBERTa-v3-base) released by [Vectara](https://www.vectara.com/)
under Apache 2.0. The current open checkpoint is
[`vectara/hallucination_evaluation_model`](https://huggingface.co/vectara/hallucination_evaluation_model)
(version 2.1).

It takes a **(premise, hypothesis)** pair and returns a single score in
**[0, 1]**:

- **close to 1**  → the hypothesis is **factually consistent** with the premise
- **close to 0**  → the hypothesis **contradicts or invents content** not
                    supported by the premise (i.e. it's hallucinating)

The model is framed as **natural-language inference for factual consistency**.
Under the hood it's a cross-encoder fine-tuned on consistency-labeled
summarization data (XSum, CNN/DailyMail with human annotations, AggreFact,
SummEdits, plus synthetic pairs).

## Why this exists

LLMs, especially in RAG pipelines, regularly produce fluent text that isn't
actually supported by the retrieved source. The dominant evaluation pattern
used to be "ask GPT-4 if the summary is faithful," which is slow, expensive,
non-deterministic, and ironically can itself hallucinate.

HHEM trades the LLM judge for a small, deterministic, MIT/Apache-licensed
classifier that is:

- **cheap** — sub-second on CPU per pair
- **stable** — same input always gives the same score
- **leaderboarded** — Vectara publishes the [HHEM leaderboard](https://huggingface.co/spaces/vectara/leaderboard)
  showing per-model hallucination rates across summarization-style tasks

It's now the default cheap hallucination guard for many production RAG
pipelines and a standard reference number in eval papers.

## When to reach for it

- **Production RAG**: score every (retrieved_passage, generated_answer) pair;
  threshold to flag/redact suspicious responses.
- **Offline eval of a RAG release**: replace the "GPT-4 fact-check" line item
  in your dashboard. Much cheaper, runs in CI.
- **Comparing two summarizers**: HHEM doesn't tell you fluency or relevance,
  but it gives you a quantitative consistency number that ranks models in
  roughly the same order as human raters on summarization.

## When **not** to reach for it

- **Open-ended QA without a source document**. HHEM needs a premise — there's
  no way to tell if "Paris is the capital of France" is hallucinated without
  giving the model a reference.
- **Long-form, multi-document reasoning** where the "source" is implicit or
  spans many docs. HHEM was trained on short premise/short hypothesis pairs;
  scores on multi-page premises are noisier.
- **Stylistic or factual nuance** (e.g. "is this analysis insightful?"). It's
  a consistency classifier, not a quality judge.

## What the demo does

`demo.py` loads HHEM and runs it on three categories of (premise, hypothesis)
pairs hand-crafted to make the scoring behavior visible:

1. **Faithful**: the hypothesis is a direct rephrasing of the premise → score
   should be near 1.
2. **Subtle hallucination**: the hypothesis flips a number, location, or date
   while keeping surface fluency → score should drop noticeably (often the
   regime that fools LLM judges but HHEM catches).
3. **Outright fabrication**: the hypothesis introduces entities or claims not
   in the premise at all → score near 0.

It prints a side-by-side table so the relative magnitudes are easy to read,
then shows how you'd apply a threshold (e.g. 0.5) to convert scores into
flags. Finally it runs the same model on the first few examples from
HaluEval-Summarization on HuggingFace to demonstrate the same code on a
real benchmark slice.

## Running

```bash
uv sync --extra hhem
uv run python hhem/demo.py                          # inline examples + tiny benchmark slice
uv run python hhem/demo.py --skip-benchmark         # offline / no internet
uv run python hhem/demo.py --threshold 0.6          # change the flag cutoff
```

The model is downloaded to the standard HuggingFace cache on first run
(~400 MB total). Subsequent runs hit the cache and start in ~1 s.

## Interpreting the score

There is no universal "good" threshold — it depends on your tolerance for
false-positive flags vs missed hallucinations. Two reference points from
the literature and Vectara's own leaderboard:

- **0.5** is the natural midpoint; a reasonable starting threshold for
  "flag for human review" in a production RAG pipeline.
- **0.85+** as a hard cutoff if you want to ship answers automatically without
  any flag — this is closer to "I'm confident this is supported."

Always calibrate on a small labeled sample of *your* domain before trusting
either number.
