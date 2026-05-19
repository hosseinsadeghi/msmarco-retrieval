# τ-bench — multi-turn tool-using customer-service agents

## What it is

[τ-bench](https://github.com/sierra-research/tau-bench) (Yao et al., Sierra,
2024) is a benchmark for **agents that take actions through tools across
multiple turns of conversation with a (simulated) user**.

Two domains ship with it:

- **retail** — a fake e-commerce backend with ~14 tools (lookup orders,
  initiate refund, exchange item, modify shipping address, …) over a small
  user/order/product DB.
- **airline** — a fake airline backend (search flights, book, cancel,
  upgrade cabin, …) over a flight/reservation DB.

A task gives you:

- A **user persona + intent** ("I want to refund my order because it arrived
  damaged; I also have a question about the warranty").
- A **simulated user** (another LLM) that responds to the agent in
  character, revealing information only when asked.
- A **starting DB state**.

The agent rolls out a multi-turn conversation, calling tools as needed. At
the end, the grader **diffs the final DB state against the reference final
state**. No LLM judging — it either matches or it doesn't.

## Why this is the right starting point for agent eval

- The grader is **deterministic**. Most "agent benchmarks" devolve into
  LLM-as-judge with all the noise that implies; τ-bench grades on
  side-effects.
- The tools are **non-trivial** (10+ tools, with nuanced preconditions like
  "can't refund a delivered order more than 30 days old"), so it actually
  exercises planning, not just one-shot tool calls.
- It's **cheap** to run — full eval is ~1k tasks × a few turns each;
  budget a few hours and a few dollars on gpt-4o-mini.
- It's **published** with a real leaderboard, so your number is comparable
  to frontier models.

## When *not* to reach for it

- You care about **coding agents** — use SWE-bench instead.
- You care about **browsing / open-domain** — use GAIA instead.
- You want a **single-turn function-calling** benchmark — use BFCL
  (Berkeley Function-Calling Leaderboard); τ-bench's multi-turn signal
  swamps the single-tool-call signal.

## Status of this folder

The demo runs an end-to-end rollout on a handful of tasks via the official
`tau-bench` package. Defaults to **`gpt-4o-mini`** for both the agent and
the simulated user; swap with `--model`.
