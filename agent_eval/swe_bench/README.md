# SWE-bench Verified — patch real GitHub issues, run the repo's tests

## What it is

[SWE-bench](https://www.swebench.com/) (Jimenez et al., ICLR 2024) is a
benchmark of **real GitHub issues** from popular Python repos (Django,
sympy, sklearn, matplotlib, …). Each task gives you:

- The repo at a specific commit
- An issue description (what's broken)
- A **gold test** that was added to fix the issue

Your agent must produce a **unified diff patch** that, when applied to the
repo, makes the gold test pass *without breaking any of the previously-
passing tests*. The grader is `pytest`.

**SWE-bench Verified** (released 2024-08) is the high-quality 500-task
subset, manually curated to filter out under-specified issues and broken
gold tests. **Use Verified** — the original 2k-task set has known noise.
**SWE-bench Lite** is a smaller (~300 tasks) easier subset for quick iteration.

## Why it's the gold standard for coding agents

- The grader is **the actual test suite of the actual repo**. No LLM judge,
  no fuzzy match, no patch syntax tolerance — either `pytest` is green or
  it isn't.
- The tasks are **real**, not synthesized. Models can't game them by
  memorizing patterns; they need to read the codebase, understand the bug,
  and write a correct patch.
- It's **the** benchmark frontier labs publish on for coding-agent claims,
  so your number is directly comparable.

## When *not* to reach for it

- You want a **fast iteration loop**. SWE-bench evaluation is slow:
  building the Docker image for each repo, applying the patch, running the
  test suite. Budget minutes per task, hours for a meaningful slice.
- You don't have Docker + ~150 GB of free disk for the repo images.
- You care about **non-Python** or **non-bug-fix** coding tasks. SWE-bench
  is exclusively Python bug-fix issues.

## Status of this folder

This is a **scaffold**. A complete pipeline is:

1. **Load tasks** from `princeton-nlp/SWE-bench_Verified` on HF.
2. **For each task**: clone the repo at the right commit, give the agent
   the issue + a tool to read/write files in the repo, let it produce a
   diff.
3. **Build the per-repo Docker image** (one-time per repo).
4. **Run `swebench.harness.run_evaluation`** with the predicted patches.

The demo in this folder does steps 1-2 against a tiny slice and writes a
predictions JSONL in the format the official grader expects. Steps 3-4
require Docker and are documented but not invoked from the script.

## Run

```bash
uv sync --extra swe_bench
uv run python swe_bench/demo.py --num 1    # writes predictions.jsonl

# Then (assuming Docker is running):
python -m swebench.harness.run_evaluation \
    --predictions_path predictions.jsonl \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --max_workers 1 \
    --run_id demo
```
