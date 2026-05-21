# Summarization — small-model benchmark on LongBench subsets

End-to-end summarization quality comparison: two small instruction-tuned
LLMs, three LongBench subsets, two metrics, one HTML report.

| Component | Choice | Why |
| --- | --- | --- |
| **Models** | `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct` | Same family, two sizes — isolates the effect of capacity. Both fit on an 8 GB GPU together. |
| **Dataset** | [LongBench v1](https://huggingface.co/datasets/THUDM/LongBench) summarization subsets: `multi_news`, `gov_report`, `qmsum` | Three different summarization shapes: multi-doc news, single-doc government report, query-based meeting summary. |
| **Metrics** | **ROUGE-L** (lexical overlap, F1) and **BERTScore-F1** (contextual semantic match using `roberta-large` by default) | Lexical and semantic; they often disagree, which is the interesting signal. |
| **Output** | `results/results.json` (machine-readable) + `results/report.html` (human-readable) | JSON for re-analysis, HTML for skimming. |

> **A note on model naming.** The original task asked for "qwen3.5-0.6b
> and 0.8b". No model under that exact name exists. The closest small
> instruction-tuned Qwen models are **Qwen2.5-0.5B-Instruct** and
> **Qwen2.5-1.5B-Instruct**, which is what's used here. To swap in
> Qwen3-0.6B / Qwen3-1.7B (which need `enable_thinking=False` in the
> chat template to skip their `<think>` traces), edit the `MODELS` list
> in `scripts/run.py`.

## Why LongBench (v1) and not v2?

LongBench v2 (Dec 2024) deliberately moved *away from summarization* —
its task categories are QA, in-context learning, long dialogue, code
understanding, and structured-data understanding, all graded by exact
match or multiple choice. None of them are open-ended summarization, so
ROUGE-L / BERTScore wouldn't apply meaningfully.

LongBench v1 keeps the summarization subsets that this script uses. The
three English ones:

- **`multi_news`** — multi-document news summarization (~2k token input).
- **`gov_report`** — long-form government report summarization (~8k token
  input).
- **`qmsum`** — query-based meeting summarization (~10k token input; the
  `input` field is the user query, the `context` field is the transcript).

Inputs are truncated to ~12k characters in the prompt to keep generation
times reasonable on the 8 GB GPU. The 0.5B and 1.5B Qwen2.5 models both
support 32k tokens of context natively, so this truncation is for *time*,
not capacity.

## Install

```bash
cd summarization
uv sync
```

This pulls torch + transformers + the metric libs. First run also downloads
~3 GB of model weights (Qwen2.5-0.5B, Qwen2.5-1.5B) and ~1.3 GB of BERTScore
weights (`roberta-large`) into the HuggingFace cache.

## Run

```bash
# Generate predictions + scores. Default: 10 examples per subset per model.
uv run python scripts/run.py --num 10

# Render the HTML report from the JSON.
uv run python scripts/report.py
```

Outputs land in `results/`:

```
results/results.json    # per-example predictions + scores + aggregates
results/report.html     # styled summary report (open in a browser)
```

Both `run.py` and `report.py` are top-to-bottom readable; no helper
hierarchy.

## What the report shows

`results/report.html` has three sections:

1. **Aggregate scores** — a model × subset table of mean ROUGE-L,
   BERTScore-F1, and average generation time.
2. **Per-metric bar charts** — pure CSS (no JS), one bar per
   (model, subset) cell, so you can eyeball which combination won.
3. **Example outputs** — a handful of (context preview / reference /
   prediction-from-each-model) cards so you can sanity-check what the
   models actually produced. Click a card to expand the full context.

## When to reach for this folder

- Sanity-checking that a fine-tune of a small model didn't regress on
  open-ended summarization.
- Comparing two model checkpoints with the same prompt & inputs.
- Teaching example for "what's the difference between ROUGE-L and
  BERTScore?" — `qmsum` and `gov_report` rows are usually where they
  diverge most.

## When *not* to reach for this folder

- For a **frontier-scale** summarization eval — small models leave a lot
  of headroom on these benchmarks; a 7B+ model is a different conversation.
- For **factuality / hallucination** scoring on summaries — neither
  ROUGE-L nor BERTScore-F1 catches that. Use [`llm_judges/hhem/`](../llm_judges/hhem/)
  instead; HHEM was designed for exactly this.
- For **RAG-style** "did the answer cite the right source?" — see
  [`rag_eval/`](../rag_eval/).
