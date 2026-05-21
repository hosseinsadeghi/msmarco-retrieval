# Summarization — small-model benchmark on LongBench subsets

End-to-end summarization quality comparison: two small instruction-tuned
LLMs, three LongBench subsets, two metrics, one HTML report.

| Component | Choice | Why |
| --- | --- | --- |
| **Models** | `Qwen2.5-0.5B-Instruct`, `Qwen2.5-1.5B-Instruct`, `Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3.5-0.8B` | Small checkpoints across three Qwen generations — lets you read the *capacity* effect (within a generation) and the *generation* effect (across them) off the same chart. Qwen3.5-0.8B is a multimodal model (text + vision); we feed it text only, which it handles fine. Models are loaded one at a time and unloaded before the next; an 8 GB GPU is enough. |
| **Dataset** | [LongBench v1](https://huggingface.co/datasets/THUDM/LongBench) summarization subsets: `multi_news`, `gov_report`, `qmsum` | Three different summarization shapes: multi-doc news, single-doc government report, query-based meeting summary. |
| **Metrics** | **ROUGE-L** (lexical overlap, F1) and **BERTScore-F1** (contextual semantic match using `roberta-large` by default) | Lexical and semantic; they often disagree, which is the interesting signal. |
| **Output** | `results/results.json` (machine-readable) + `results/report.html` (human-readable) | JSON for re-analysis, HTML for skimming. |

> **A note on model naming.** The original task asked for "qwen3.5-0.6b
> and 0.8b". Qwen3.5-0.8B exists (released 2026-02-28); Qwen3.5-0.6B
> does not — 0.8B is the smallest checkpoint in that family. The
> benchmark covers five models: the two Qwen2.5 small instruct variants
> (0.5B, 1.5B), the two Qwen3 small dense checkpoints (0.6B, 1.7B), and
> the Qwen3.5-0.8B multimodal model (used in text-only mode).
>
> Qwen3, Qwen3.5 are multi-mode reasoning models. We pass
> `enable_thinking=False` to the chat template so they emit direct
> summaries instead of `<think>` traces, and as a guard the script strips
> any `<think>...</think>` block from generated text before scoring.

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

This pulls torch + transformers (5.x — required for the `qwen3_5`
architecture) + torchvision (the Qwen3.5 stack uses it) + the metric libs.
First run also downloads ~6 GB of model weights (the five Qwens) and
~1.3 GB of BERTScore weights (`roberta-large`) into the HuggingFace cache.

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

`results/report.html` has four sections:

1. **Aggregate scores** — a model × subset table of mean ROUGE-L,
   BERTScore-F1, and average generation time. Best score per
   (metric × subset) highlighted.
2. **Reference numbers** — a small table with literature values per subset
   (frontier-LLM ROUGE-L from the LongBench paper, supervised-SoTA
   ROUGE-L and BERTScore-F1 from the standalone benchmarks, and a
   BERTScore floor for unrelated text). This is the context you need to
   read the aggregate table sensibly.
3. **Per-metric bar charts** — pure CSS (no JS), one bar per
   (model, subset) cell, **with the two reference values from §2 drawn as
   vertical markers** on each bar so you can see at a glance how far each
   small model is from frontier / SoTA.
4. **Example outputs** — two (context preview / reference / prediction-from-
   each-model) cards per subset so you can sanity-check what the models
   actually produced. Click *Show full context preview* to expand.

### What counts as a reference number

- **Frontier LLM (ROUGE-L)** — `GPT-3.5-Turbo-16k` numbers from Table 4 of
  the LongBench paper (Bai et al., [arXiv:2308.14508](https://arxiv.org/abs/2308.14508)).
  Approximate: Multi-News 0.264, GovReport 0.295, QMSum 0.234.
- **Supervised SoTA (ROUGE-L)** — strong fine-tuned abstractive models on
  the *standalone* benchmark (not LongBench's truncated slice): PRIMERA
  (Multi-News, ~0.249), LED / PEGASUS-X (GovReport, ~0.35), DialogLED /
  Locator+Summarizer (QMSum, ~0.32).
- **Supervised SoTA (BERTScore-F1)** — same models, with
  `rescale_with_baseline=False` (matches our eval). Typically ~0.86–0.88.
- **BERTScore-F1 floor** — between *unrelated* English text, ~0.83. Any
  system anywhere near this is producing noise.

These are approximate, not single-paper authoritative — they exist to put
the small-model numbers in *roughly the right ballpark*, not to claim a
precise gap.

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
