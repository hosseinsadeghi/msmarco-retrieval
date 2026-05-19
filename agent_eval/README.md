# Agent evaluation — tool-using, multi-turn, trajectory-graded

LLM-as-judge models (in [`llm_judges/`](../llm_judges/)) score a **single
output**. Agent evaluation is a different problem: the model takes
**multiple steps**, calls **tools**, and the thing you grade is the
**trajectory and final environment state**, not a single response. You need:

- An **environment** the agent can act in (a database, a code repo, the web,
  a fake customer-service tool API).
- A **reward / grader** that's deterministic where possible — checking the
  final DB state, running a unit-test suite, comparing against a reference
  answer — and only falls back to LLM-as-judge when nothing else works.

## What's in each subfolder

| Folder | Benchmark | Domain | Grader |
| --- | --- | --- | --- |
| [`tau_bench/`](tau_bench/) | [τ-bench](https://github.com/sierra-research/tau-bench) (Sierra) | Customer-service: airline + retail, with a mock DB and tool API. | Final DB-state diff against reference. |
| [`swe_bench/`](swe_bench/) | [SWE-bench Verified](https://www.swebench.com/) | Real GitHub issues from Django, sympy, sklearn, etc. | Apply patch, run repo's own test suite, check it now passes. |
| [`gaia/`](gaia/) | [GAIA](https://huggingface.co/gaia-benchmark) | General-purpose questions that need browsing / file reading / multi-step reasoning. | Exact-match string compare against reference answer. |

## Why these three

They span the axes that matter for "is this AI assistant any good?":

1. **τ-bench** is the cleanest **tool-use** benchmark. The agent has a
   ~10-tool API (book a flight, refund an order, lookup a customer), needs
   to take multiple turns with a simulated user, and grading is just
   *did the DB end up in the right state?* No LLM judges, no ambiguity.
2. **SWE-bench Verified** is the **hard real-world** benchmark. Real
   bug-fix issues from popular Python repos, with the original maintainer-
   written test as the grader. There's no clever way to "game" this — either
   your patch makes the test pass or it doesn't. *(Note: full eval needs
   Docker + ~150 GB of disk for the repo images.)*
3. **GAIA** is the **broad assistant** benchmark — the questions look like
   what a user might actually ask an assistant, requiring varying mixes of
   browsing, calculation, file reading, and chaining results.

## Install

```bash
cd agent_eval

uv sync --extra tau_bench    # tau-bench env + grader
uv sync --extra swe_bench    # swebench package (Docker required to actually run)
uv sync --extra gaia         # GAIA dataset loader + a simple browsing-tool stub
uv sync --extra all
```

## Run a demo

```bash
uv run python tau_bench/demo.py --domain retail --num 3
uv run python swe_bench/demo.py --num 1            # picks the smallest task
uv run python gaia/demo.py --level 1 --num 3
```

Every `demo.py` follows the same `load tasks → roll out agent → grade →
print summary` shape. The agents themselves are minimal (single-turn
function-calling against an OpenAI-compatible endpoint) so the *evaluation
plumbing* is the focus, not the agent architecture.

## Hardware / cost notes

- **τ-bench** is mostly LLM API calls (one rollout = N user-agent turns).
  CPU-only host is fine; budget ~$0.10 per rollout with gpt-4o-mini.
- **SWE-bench** needs Docker, ~150 GB disk for the repo images, and
  patch-time can be minutes per task. **Don't** try to run the full 500-task
  Verified set casually.
- **GAIA** is light on the harness side; the cost is browsing tools + an
  LLM with vision-capable browsing.

## A note on "agent eval" vs "judge eval"

If you're tempted to evaluate an agent with an LLM judge ("did this
trajectory look reasonable?"), please don't — that's how the
overpromising-agent-paper genre happened. Use the deterministic grader the
benchmark ships with. Reserve LLM judges for the messy in-between cases the
deterministic grader can't cover (e.g. natural-language responses to the
simulated user in τ-bench).
