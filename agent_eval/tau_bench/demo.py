"""τ-bench demo — roll out an agent on a handful of tasks, print pass rate.

Run:
    uv run python tau_bench/demo.py --domain retail --num 3
    uv run python tau_bench/demo.py --domain airline --num 3 --model openai/gpt-4o

Top-to-bottom shape:
    1. Load `num` tasks from the chosen domain.
    2. For each task: spin up the env, run a tool-calling agent against the
       simulated user, capture the final DB-state diff.
    3. Print per-task pass/fail and an aggregate pass rate.
"""
from __future__ import annotations

import argparse
import os


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=["retail", "airline"], default="retail")
    p.add_argument("--num", type=int, default=3, help="Number of tasks to run.")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="Model id for both agent and simulated user.")
    p.add_argument("--max-turns", type=int, default=30)
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY (or point OPENAI_API_BASE at a "
                         "local OpenAI-compatible endpoint).")

    # tau-bench's public entry points. The exact import path here tracks the
    # upstream repo; if they reshape it, update accordingly.
    from tau_bench.envs import get_env
    from tau_bench.agents.tool_calling_agent import ToolCallingAgent
    from tau_bench.run import run

    env = get_env(args.domain, user_strategy="llm", user_model=args.model)
    agent = ToolCallingAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=args.model,
        provider="openai",
    )

    # `run` iterates tasks, executes the rollout, and returns a list of
    # per-task result dicts (with `reward` in {0, 1} and the trajectory).
    results = run(
        env=env,
        agent=agent,
        task_ids=list(range(args.num)),
        max_concurrency=1,
        max_steps=args.max_turns,
    )

    passed = sum(int(r["reward"] > 0.5) for r in results)
    print(f"\n{args.domain}: {passed}/{len(results)} passed "
          f"({passed/len(results):.0%})\n")
    for r in results:
        print(f"  task {r['task_id']:>3}  reward={r['reward']:.2f}  "
              f"turns={len(r['trajectory'])}")


if __name__ == "__main__":
    main()
