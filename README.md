# ml-playground

Self-contained subprojects exploring different ML / NLP / LLM topics. Each
subfolder is an independent mini-repo — its own `pyproject.toml`, its own
README, its own runnable scripts. No shared state, no central package; pick a
folder and start there.

## Topics

| Folder                            | What's inside                                                                                              |
| ---                               | ---                                                                                                        |
| [`retrieval/`](retrieval/)        | BM25 / dense / hybrid retrieval over MS MARCO + BEIR / MTEB / MTEB-v2 datasets. One unified CLI: `download → build → search → eval`. |
| [`llm_judges/`](llm_judges/)      | Hands-on demos of 5 LLM-quality / LLM-routing models: Prometheus 2, PandaLM, Auto-J (+ 4-bit), Vectara HHEM, RouteLLM. |
| [`rag_eval/`](rag_eval/)          | Joint retriever+generator quality: Ragas, TruLens, ARES, plus an end-to-end pipeline that wires `retrieval/` into a generator. |
| [`agent_eval/`](agent_eval/)      | Tool-using, multi-turn agent benchmarks: τ-bench (retail / airline), SWE-bench Verified, GAIA. |

A ranked roadmap for what else to add (code eval, long-context, safety,
calibration, …) lives in [`EVAL_ROADMAP.md`](EVAL_ROADMAP.md).

## Repo philosophy

- **Subprojects don't share dependencies.** Each folder owns its `pyproject.toml`
  so the retrieval stack (sentence-transformers, rank-bm25) doesn't pull in the
  judges' 13B-LLM stack and vice versa.
- **Top-to-bottom readable scripts.** Where possible, the demo / pipeline scripts
  are 50–200 lines you can scan in one sitting. The interesting work happens
  in those scripts, not in deep helper hierarchies.
- **Data-scientist-style READMEs.** Each subproject's README explains *what*
  the model / method is, *why* it exists, *when* you'd reach for it, and *when
  not to* — not just install/run commands.
- **Verified end-to-end where feasible.** Anything that fits on a CPU has been
  run and the results checked in (`retrieval/results/*.md`). 7B+ LLM demos
  document GPU requirements rather than burning hours on CPU inference.

## Adding a new topic

1. `mkdir new_topic/` at the repo root.
2. Drop in a `pyproject.toml`, a `README.md`, and your scripts.
3. Link it from the table above.

That's it — no central registry, no plugin system, no glue code.

## License

MIT — see [LICENSE](LICENSE). Applies to the whole repo.
