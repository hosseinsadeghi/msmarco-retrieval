# end_to_end — wire `retrieval/` into a generator and score with Ragas

This is where the two halves of the repo meet. The script:

1. **Loads a dataset from the `retrieval/` registry** (e.g. NFCorpus,
   SciFact, MSMARCO) — same DB and BM25 index that `retrieval/scripts/eval.py`
   builds.
2. **Runs the chosen retriever** (BM25 / dense / hybrid) for each query to
   get the top-k contexts. Reuses `retrieval/src/retrieval/retrieve.py`.
3. **Generates an answer** with a small local LLM (default
   `google/flan-t5-base`, ~250M params, CPU-friendly).
4. **Scores the (q, contexts, answer, ground_truth) rows** with Ragas's four
   core metrics (`faithfulness`, `answer_relevancy`, `context_precision`,
   `context_recall`).
5. **Prints a comparison table**: classical IR metrics (nDCG@10, Recall@k)
   side-by-side with the Ragas metrics so you can see which side of the
   pipeline a regression came from.

## Why this is interesting

`retrieval/scripts/eval.py` already tells you *did we fetch the right
passages?* Ragas alone tells you *was the answer faithful to the passages
we fetched?* Neither tells you the thing you actually care about, which is
*did the user get a correct, well-grounded answer?* — which is the
**intersection** of the two.

This script lets you ablate that intersection:

- Swap BM25 → dense → hybrid retrievers, keep the generator fixed.
- Swap generators (flan-t5-base → flan-t5-large → a 7B), keep the retriever
  fixed.
- See which knob actually moves user-facing quality.

## Run

```bash
# Assumes you've already run, in the retrieval/ subproject:
#   python scripts/download.py --dataset NFCorpus
#   python scripts/build.py    --dataset NFCorpus

uv sync --extra ragas --extra end_to_end
uv run python end_to_end/demo.py \
    --dataset NFCorpus \
    --retriever hybrid \
    --generator google/flan-t5-base \
    --num 20
```

Output is a Markdown report at `end_to_end/results/<dataset>_<retriever>.md`
with per-query rows and aggregate metrics.

## Cost / determinism

The Ragas judge is non-deterministic and costs API calls (default
`gpt-4o-mini`). Defaults to `--num 20` rows for the demo; real ablations want
100–500. Pin `temperature=0` on the judge (already set) for run-to-run
comparability.
