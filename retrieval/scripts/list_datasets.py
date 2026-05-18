"""Show the dataset registry, with an optional --benchmark filter.

This is the canonical place to learn which `--dataset` names work and which
benchmarks each one belongs to (MTEB, MTEB-v2, BEIR, MS-MARCO).

Examples:
    python scripts/list_datasets.py
    python scripts/list_datasets.py --benchmark MTEB-v2
    python scripts/list_datasets.py --benchmark BEIR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval.datasets import REGISTRY  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", help='filter, e.g. "MTEB", "MTEB-v2", "BEIR", "MS-MARCO"')
    args = p.parse_args()

    rows = []
    for name, spec in sorted(REGISTRY.items()):
        if args.benchmark and args.benchmark not in spec.benchmarks:
            continue
        rows.append((name, spec.kind, ", ".join(spec.benchmarks), spec.description))

    if not rows:
        print(f"No datasets match benchmark={args.benchmark!r}.")
        return

    name_w = max(len(r[0]) for r in rows + [("NAME", "", "", "")])
    kind_w = max(len(r[1]) for r in rows + [("", "KIND", "", "")])
    bm_w = max(len(r[2]) for r in rows + [("", "", "BENCHMARKS", "")])

    print(f"{'NAME':<{name_w}}  {'KIND':<{kind_w}}  {'BENCHMARKS':<{bm_w}}  DESCRIPTION")
    print("-" * (name_w + kind_w + bm_w + 30))
    for name, kind, bmarks, desc in rows:
        print(f"{name:<{name_w}}  {kind:<{kind_w}}  {bmarks:<{bm_w}}  {desc}")


if __name__ == "__main__":
    main()
