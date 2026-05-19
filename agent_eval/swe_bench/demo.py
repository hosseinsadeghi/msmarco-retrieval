"""SWE-bench Verified demo — produce predictions.jsonl for the official grader.

This is a **scaffold**. The agent here is intentionally minimal: it just asks
the LLM to read the issue text and emit a unified diff. Real SWE-bench
agents (Agentless, SWE-Agent, OpenHands) use multi-step retrieval into the
repo, file-editing tools, and test-execution feedback loops.

Run:
    uv run python swe_bench/demo.py --num 1
    # Then run the official harness over predictions.jsonl (needs Docker).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

OUT = Path(__file__).parent / "predictions.jsonl"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=1,
                   help="Number of tasks to attempt.")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="LiteLLM-style model id for patch generation.")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY.")

    from datasets import load_dataset
    from litellm import completion

    ds = load_dataset(args.dataset, split="test")
    print(f"Loaded {len(ds)} tasks from {args.dataset}")

    # Sort by problem statement length so --num 1 picks the smallest task.
    indices = sorted(range(len(ds)), key=lambda i: len(ds[i]["problem_statement"]))[:args.num]

    with OUT.open("w") as f:
        for i in indices:
            task = ds[i]
            prompt = (
                "You are fixing a bug in a Python repo. Below is the issue. "
                "Reply with ONLY a unified diff patch (no markdown fences, "
                "no commentary) that fixes the bug.\n\n"
                f"Repo: {task['repo']}\n"
                f"Base commit: {task['base_commit']}\n\n"
                f"Issue:\n{task['problem_statement']}\n"
            )
            resp = completion(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            patch = resp["choices"][0]["message"]["content"].strip()

            # Schema expected by swebench.harness.run_evaluation.
            row = {
                "instance_id": task["instance_id"],
                "model_name_or_path": args.model,
                "model_patch": patch,
            }
            f.write(json.dumps(row) + "\n")
            print(f"  {task['instance_id']}: {len(patch)} chars of patch")

    print(f"\nWrote {OUT}")
    print("Next step (needs Docker):")
    print("  python -m swebench.harness.run_evaluation \\")
    print(f"    --predictions_path {OUT} \\")
    print(f"    --dataset_name {args.dataset} \\")
    print("    --max_workers 1 --run_id demo")


if __name__ == "__main__":
    main()
