# MS MARCO retrieval playground

Compare three retrievers — sparse (BM25), dense (sentence-transformer
embeddings), and hybrid (Reciprocal Rank Fusion of the two) — over passages
from the MS MARCO QA dataset.

## Why MS MARCO?

`microsoft/ms_marco` (v2.1) is a QA dataset built from real Bing queries. Each
example carries:

- `query` — a natural-language question
- `answers` — one or more human-written answer strings
- `passages` — a handful of candidate web passages, with `is_selected=1` on the
  ones a human marked as containing the answer
- `query_type` — `description` / `numeric` / `entity` / `location` / `person`

`scripts/01_download.py` prints a sample after the download so you can verify
this for yourself.

## Layout

```
msmarco/
├── data/                       # raw downloaded JSONL
├── db/msmarco.sqlite           # passages + queries + embeddings (BLOB)
├── indexes/bm25.pkl            # pickled BM25Okapi + passage-id mapping
├── src/msmarco_retrieval/      # small shared helpers
│   ├── config.py               # paths and defaults
│   ├── text.py                 # BM25 tokenizer
│   ├── embeddings.py           # sentence-transformer wrapper
│   └── db.py                   # SQLite schema + helpers
└── scripts/
    ├── 01_download.py          # download + QA sanity check
    ├── 02_build_db.py          # SQLite + embedding index
    ├── 03_build_bm25.py        # BM25 index
    ├── search_bm25.py          # BM25-only query
    ├── search_vector.py        # dense-only query
    └── search_hybrid.py        # BM25 + dense fused via RRF
```

## Install

```bash
cd msmarco
uv sync                  # or: pip install -e .
```

## Pipeline

Run the three build steps once. Each is independent of the others except for
ordering:

```bash
uv run python scripts/01_download.py               # ~1000 validation queries
uv run python scripts/02_build_db.py               # fills SQLite + embeddings
uv run python scripts/03_build_bm25.py             # builds BM25 index
```

Then query with any of the three search scripts. They all accept either a
hand-written question via `-q` or pull a random query from the loaded MS MARCO
slice when called with no argument:

```bash
uv run python scripts/search_bm25.py   -q "what is the capital of france?"
uv run python scripts/search_vector.py -q "what is the capital of france?"
uv run python scripts/search_hybrid.py -q "what is the capital of france?"

uv run python scripts/search_hybrid.py            # random question from the dataset
```

## How each search script is organized

All three follow the same four-step shape so you can diff them mentally:

1. **Load Q** — from CLI or random row in `queries` table
2. **Prepare** — tokenize (BM25) and/or encode (dense)
3. **Score** — `bm25.get_scores(...)` and/or `mat @ q_vec`
4. **Print top-k**

The hybrid script adds one extra step: fuse the two ranked lists with RRF
(`score = sum(1 / (rrf_k + rank))`). RRF needs no score normalization, which
matters because BM25 scores and cosine similarities live on different scales.

## MTEB / MTEB-v2 retrieval datasets

The same three retrievers can run against any MTEB retrieval dataset (BEIR
format on the `mteb/*` HuggingFace org). Two scripts handle the lifecycle:

```bash
# 1. Download (corpus + queries + qrels JSONL -> data/mteb/<name>/)
uv run python scripts/mteb_download.py --dataset NFCorpus
uv run python scripts/mteb_download.py --dataset SciFact
uv run python scripts/mteb_download.py --dataset ArguAna

# 2. End-to-end: build DB + embeddings + BM25, run all 3 retrievers,
#    compute Recall@k / MRR@k / nDCG@k, write results/mteb_<name>.md
uv run python scripts/mteb_run.py --dataset NFCorpus
uv run python scripts/mteb_run.py --dataset SciFact
uv run python scripts/mteb_run.py --dataset ArguAna
```

Each dataset gets its own `db/mteb_<name>.sqlite` and `indexes/mteb_<name>_bm25.pkl`,
so they don't collide. Pass `--rebuild` to recompute the DB/index for a
dataset; otherwise cached artifacts are reused.

A small map of named presets lives in
`src/msmarco_retrieval/mteb_helpers.py` (`NFCorpus`, `SciFact`, `FiQA2018`,
`ArguAna`, `SCIDOCS`, `TRECCOVID`, `Touche2020`, `Quora`). For anything else
on the `mteb/*` org, pass `--hf-id mteb/<other>` directly.

**MTEB vs MTEB-v2.** MTEB v2 (MMTEB) mostly *adds* new retrieval datasets in
the same BEIR layout — the same two scripts work for either version. A few
v2-specific variants (e.g. the qrels-only `*HardNegatives` repos) need extra
plumbing that isn't included here.

Sample headline metrics (full reports in `results/mteb_*.md`):

| Dataset  | Retriever    | nDCG@10 |
| ---      | ---          | ---     |
| NFCorpus | BM25         | 0.313   |
| NFCorpus | Dense        | 0.319   |
| NFCorpus | Hybrid (RRF) | **0.334** |
| SciFact  | BM25         | 0.667   |
| SciFact  | Dense        | 0.648   |
| SciFact  | Hybrid (RRF) | **0.689** |
| ArguAna  | BM25         | 0.354   |
| ArguAna  | Dense        | 0.368   |
| ArguAna  | Hybrid (RRF) | **0.389** |

(Hybrid wins consistently on BEIR-style data — the opposite of what happened
on the MS MARCO Q-validation eval above, because MS MARCO queries are short
and semantic, while BEIR has more specialized vocabulary that BM25 helps on.)

## Evaluation (MS MARCO)

`scripts/eval_summary.py` runs all three retrievers over a random sample of
queries and scores them against the human `is_selected` flag for each query
(treating those passages as gold):

```bash
uv run python scripts/eval_summary.py --num 200
```

It reports **Recall@k** (any gold in top-k) and **MRR@k** (reciprocal rank of
the first gold hit), and writes a full Markdown report — table + qualitative
samples with ✅ markers — to `results/summary.md`. A pre-computed example
(200 queries, seed=42, 9947-passage corpus) is checked in at
[`results/summary.md`](results/summary.md):

| Retriever    | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
| ---          | ---      | ---      | ---       | ---    |
| BM25         | 0.220    | 0.655    | 0.850     | 0.398  |
| Dense        | 0.385    | 0.845    | 0.975     | 0.579  |
| Hybrid (RRF) | 0.330    | 0.795    | 0.940     | 0.512  |

(Dense beats RRF here because the BM25 ranks it averages in are noticeably
weaker. Hybrid wins when you weight the two retrievers, train a reranker on
top, or run on a domain where exact-term matching matters more.)

## Tweakable knobs

- `src/msmarco_retrieval/config.py` — embedding model, default split, default
  query count, paths
- `src/msmarco_retrieval/text.py` — BM25 stopwords and tokenizer regex
- `--candidates` and `--rrf-k` on `search_hybrid.py` — fusion depth and smoothing
- `--num`, `--depth`, `--samples`, `--seed` on `eval_summary.py`

## License

MIT — see [LICENSE](LICENSE).
