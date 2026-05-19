# GAIA — General AI Assistant benchmark

## What it is

[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) (Mialon et al.,
Meta + HuggingFace, 2023) is a benchmark of **real-world questions a user
might ask a personal assistant** that require chaining together multiple
capabilities: browsing, file reading, calculation, knowledge lookup, common
sense, and so on.

Questions are grouped by difficulty:

- **Level 1** — single tool, a few reasoning steps.
- **Level 2** — multiple tools, longer reasoning chains.
- **Level 3** — long-horizon planning over many tools and intermediate
  artifacts.

The grader is **exact-match** against a reference answer (after light
normalization — case, punctuation, number formatting). This is intentional:
GAIA's premise is that an assistant should produce one correct answer, not
a paragraph of waffle that "kind of" gets there.

The dataset is **gated** on HuggingFace — you need to accept the terms at
`huggingface.co/datasets/gaia-benchmark/GAIA` and `huggingface-cli login`
before you can pull it.

## Why this rounds out the agent-eval coverage

- **τ-bench** tests tool use in a *closed* environment (mock APIs).
- **SWE-bench** tests tool use in a *closed* environment (a specific repo).
- **GAIA** tests tool use in the **open environment** — the actual web,
  arbitrary file types, the messy reality. Most "assistant" use cases live
  here.

It also tests **abstention** and **calibration** indirectly: many GAIA
questions have a single short correct answer, and confidently-wrong
answers look much worse than "I couldn't find this." Frontier models score
~50% on Level 1 and noticeably less on Level 3.

## When *not* to reach for it

- You want a **fast iteration loop**. GAIA rollouts can take minutes per
  question because they involve real web requests.
- You want **deterministic, reproducible** scores. The web is non-stationary
  — a 2023 question about a Wikipedia article may have a different "correct"
  answer in 2026.
- You want a **public test set**. GAIA reserves a private leaderboard set;
  the demo here uses the validation split.

## Status of this folder

The demo wires GAIA's validation split to a minimal browsing-capable agent
(`requests` + `beautifulsoup4`, no JavaScript) and reports exact-match. It's
intentionally a weak baseline — replace the agent loop with something
real (SerpAPI search, headless Chromium, file-type-aware reading) for a
serious run.
