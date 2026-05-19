# AI-assistant evaluation: what's here, what's missing, what to build next

Snapshot of the repo's eval coverage and a ranked list of folders that would
round it out into a complete **AI-assistant evaluation** toolkit. Same house
style as everything else: each suggested folder would be self-contained,
DS-style README, runnable demos.

## What's already here

- [`retrieval/`](retrieval/) — recall side of RAG (BM25 / dense / hybrid over
  MS MARCO + BEIR / MTEB / MTEB-v2).
- [`llm_judges/`](llm_judges/) — scoring a single LLM output: Prometheus 2,
  PandaLM, Auto-J, Vectara HHEM, RouteLLM.

Gap: we can eval the retriever *or* the generator in isolation, but nothing
about joint RAG quality, agentic / tool-using behavior, code execution,
safety, or long-context.

## Tier 1 — closes obvious gaps next to what's already here

| Folder | What's inside | Why it complements existing folders |
| --- | --- | --- |
| **`rag_eval/`** | RAGAS, TruLens, ARES — faithfulness / answer-relevancy / context-precision metrics, plus a "retrieval+generation" pipeline that wires `retrieval/` into a small LLM. | Bridges `retrieval/` ↔ `llm_judges/`. Right now you can eval the retriever *or* the generator; this evals the joint system, which is what users actually see. |
| **`agent_eval/`** | τ-bench (tau-bench), SWE-bench Verified (a tiny slice), GAIA, WebArena-Lite. Multi-turn, tool-using rollouts with deterministic graders. | LLM-as-judge ≠ agent eval. Agent eval needs *environments* and *trajectory-level scoring*, which is a different beast from PandaLM/Prometheus. |
| **`code_eval/`** | HumanEval, MBPP, BigCodeBench, LiveCodeBench — execution-based pass@k with a sandboxed runner (`subprocess` + tmpdir + timeout). | Coding is the single biggest assistant use case; LLM-as-judge is the wrong tool here — you want `pytest`-style ground truth, not a 7B judge. |
| **`long_context/`** | Needle-in-a-Haystack, RULER, LongBench, ZeroSCROLLS. Sweeps context length × depth and plots the heatmap. | Distinct failure mode from RAG: tests *how much* the model can attend to, not *what* it retrieves. |

## Tier 2 — round out the safety / reliability axis

| Folder | What's inside |
| --- | --- |
| **`safety_redteam/`** | HarmBench, JailbreakBench, AdvBench, garak / PyRIT runners — refusal rate, ASR (attack success rate), and a small jailbreak corpus. |
| **`prompt_injection/`** | Indirect-injection corpora (TensorTrust, InjecAgent), tool-augmented attacks against the `agent_eval/` envs. Different failure mode than jailbreaks (data → instructions, not user → instructions). |
| **`structured_output/`** | JSON-schema adherence, function-call argument accuracy (BFCL — Berkeley Function-Calling Leaderboard, ToolBench). Cheap to grade, very actionable. |
| **`calibration/`** | Confidence calibration & selective answering — ECE, Brier, selective-risk curves on TriviaQA / MMLU. Pairs well with HHEM for "should the assistant abstain?". |

## Tier 3 — nice to have, smaller scope

- **`reward_models/`** — RewardBench harness. Sibling to `llm_judges/`: judges *score outputs*, RMs *rank pairs*; both feed RLHF/DPO but the artifacts differ.
- **`traces/`** — OpenTelemetry GenAI / OpenInference trace replay; load a trace dump (JSONL), re-score offline with judges from `llm_judges/`. Glue layer for "eval on real traffic, not just benchmarks."
- **`multimodal_eval/`** — MMBench, MMMU, ChartQA. Only worth it if you care about VLMs.
- **`bias_fairness/`** — BBQ, BOLD, StereoSet.

## If you had to pick three to build first

1. **`rag_eval/`** — the most natural extension of what's already here; very little new infra, big new signal.
2. **`code_eval/`** — fastest to make useful (HumanEval + a sandbox runner is ~150 lines), and execution-graded results are unambiguous.
3. **`agent_eval/`** — highest leverage for "AI assistant" specifically. τ-bench's airline/retail envs are the cleanest entry point; SWE-bench is heavier.

## Status

- [x] `retrieval/`
- [x] `llm_judges/`
- [x] `rag_eval/` — scaffolded
- [x] `agent_eval/` — scaffolded
- [ ] `code_eval/`
- [ ] `long_context/`
- [ ] `safety_redteam/`
- [ ] `prompt_injection/`
- [ ] `structured_output/`
- [ ] `calibration/`
- [ ] `reward_models/`
- [ ] `traces/`
