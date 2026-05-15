# Retrieval playground: BM25 / dense / hybrid over MTEB-style QA datasets

Compare three retrievers — sparse (BM25), dense (sentence-transformer
embeddings), and hybrid (Reciprocal Rank Fusion) — over MS MARCO and BEIR /
MTEB / MTEB-v2 retrieval datasets, using the same five-script CLI for
everything.

## Five-script CLI (same for every dataset)

```bash
scripts/list_datasets.py    # show available datasets and their benchmarks
scripts/download.py         # --dataset X     pull raw files
scripts/build.py            # --dataset X     SQLite + embeddings + BM25
scripts/search.py           # --dataset X --retriever {bm25,dense,hybrid} -q "..."
scripts/eval.py             # --dataset X     full benchmark eval (Recall@k / MRR@k / nDCG@k)
```

`scripts/list_datasets.py` is the entry point — it tells you which datasets
exist and which benchmarks each one is part of:

```
$ python scripts/list_datasets.py
NAME        KIND     BENCHMARKS               DESCRIPTION
----------------------------------------------------------------------
ArguAna     beir     BEIR, MTEB, MTEB-v2      Counter-argument retrieval (~8.7k docs).
FiQA2018    beir     BEIR, MTEB, MTEB-v2      Financial opinion QA (~57k docs).
MSMARCO     msmarco  MS-MARCO, MTEB, MTEB-v2  MS MARCO v2.1 QA passages from Bing queries.
NFCorpus    beir     BEIR, MTEB, MTEB-v2      Medical scientific literature retrieval.
Quora       beir     BEIR, MTEB, MTEB-v2      Duplicate-question retrieval.
SCIDOCS     beir     BEIR, MTEB, MTEB-v2      Citation prediction over scientific papers.
SciFact     beir     BEIR, MTEB, MTEB-v2      Scientific claim verification.
TRECCOVID   beir     BEIR, MTEB, MTEB-v2      COVID-19 scientific literature retrieval.
Touche2020  beir     BEIR, MTEB, MTEB-v2      Controversial-topic argument retrieval.

$ python scripts/list_datasets.py --benchmark MTEB-v2     # filter
```

The registry lives in `src/msmarco_retrieval/datasets.py`. Each `DatasetSpec`
has a `kind` (`"msmarco"` or `"beir"`) that dispatches the download/build
behavior, and a list of `benchmarks` it belongs to. Most BEIR datasets are
also part of MTEB and MTEB-v2 — v2 mostly *adds* new datasets in the same
format rather than changing the format.

## End-to-end usage

```bash
uv sync                                                    # one-time

# Any registered dataset goes through the same four steps:
python scripts/download.py --dataset NFCorpus
python scripts/build.py    --dataset NFCorpus              # DB + embeddings + BM25
python scripts/search.py   --dataset NFCorpus --retriever hybrid -q "vegan heart disease"
python scripts/eval.py     --dataset NFCorpus              # writes results/NFCorpus.md

# Same flow for MS MARCO:
python scripts/download.py --dataset MSMARCO --num 1000    # MSMARCO-only flag
python scripts/build.py    --dataset MSMARCO
python scripts/eval.py     --dataset MSMARCO --num 200
```

Per-dataset files live at:

```
data/<name>/...           # raw downloaded files
db/<name>.sqlite          # passages + embeddings + queries + qrels
indexes/<name>_bm25.pkl   # pickled BM25Okapi + id mapping
results/<name>.md         # eval report
```

`build.py` and `eval.py` reuse cached artifacts if present; pass `--rebuild`
to force recomputation.

## How each script is organized

Every script follows a tight "load → prepare → score → output" shape so they
read top-to-bottom with no surprises:

- **`search.py`** — load Q (CLI or random) → tokenize and/or encode → score
  with the chosen retriever → print top-k with ✅ on qrel-positive hits.
- **`eval.py`** — pick all queries with at least one positive qrel → encode in
  one batch → run all three retrievers per query → write Markdown report with
  aggregate metrics + sample top-3 rows.

The three retrievers themselves live in `src/msmarco_retrieval/retrieve.py`:

```python
bm25_search(bm25, ids, query, depth)        # tokenize + BM25Okapi.get_scores
dense_search(mat, ids, q_vec, depth)        # single matmul (cosine, both sides unit-norm)
rrf_fuse([bm25_top, dense_top], rrf_k, depth)
```

## Sample results

| Dataset  | Retriever    | nDCG@10 | Notes                                |
| ---      | ---          | ---     | ---                                  |
| MSMARCO  | BM25         | 0.535   | 200 random queries, 4,973-passage corpus |
| MSMARCO  | Dense        | **0.681** | short semantic queries — dense wins |
| MSMARCO  | Hybrid (RRF) | 0.633   |                                      |
| NFCorpus | BM25         | 0.313   | 323 queries, 3,633-passage corpus    |
| NFCorpus | Dense        | 0.319   |                                      |
| NFCorpus | Hybrid (RRF) | **0.334** | medical jargon → exact match helps  |

Full reports with sample qualitative rows live in `results/<name>.md`.

## Unified SQLite schema

Both dataset kinds land in the same three tables, so everything downstream of
`build.py` is kind-agnostic:

```
passages(id, external_id, text, embedding BLOB)
queries (id, external_id, text, answers JSON or NULL)
qrels   (query_pk, passage_pk, relevance)
```

- `external_id` is the source dataset's id (BEIR `_id` strings, or `pN` for
  MS MARCO's deduplicated passages).
- `answers` is populated for MS MARCO and NULL for BEIR.
- `qrels.relevance > 0` means "relevant"; nDCG honors the graded value.

## Tweakable knobs

- `src/msmarco_retrieval/config.py` — embedding model, paths.
- `src/msmarco_retrieval/text.py` — BM25 tokenizer regex + stopwords.
- `src/msmarco_retrieval/datasets.py` — add new entries to `REGISTRY` to
  support more datasets. For anything else on the `mteb/*` HF org that follows
  the BEIR layout (corpus / queries / default), one new `DatasetSpec` entry is
  the whole change.
- CLI flags on `build.py` / `search.py` / `eval.py`: `--rebuild`, `--depth`,
  `--rrf-k`, `--candidates`, `--num`, `--samples`, `--seed`.

## License

MIT — see [LICENSE](LICENSE).
