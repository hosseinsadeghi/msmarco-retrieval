"""Central registry of supported retrieval datasets.

Every dataset is a `DatasetSpec` with:
    name        — the key used everywhere on the CLI (`--dataset NFCorpus`)
    kind        — "msmarco" (single rich JSONL) or "beir" (corpus/queries/qrels)
    hf_id       — HuggingFace repo id
    benchmarks  — which suites this dataset is part of, e.g.
                  ("BEIR", "MTEB", "MTEB-v2"). Used by list_datasets.py to
                  let users filter ("show me MTEB-v2 only").
    description — one-line blurb

`kind` determines how download.py and build.py treat the dataset; everything
downstream of build.py (search.py, eval.py) is kind-agnostic because both
flavors land in the same SQLite schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import ROOT

DatasetKind = Literal["msmarco", "beir"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: DatasetKind
    hf_id: str
    benchmarks: tuple[str, ...]
    description: str = ""


REGISTRY: dict[str, DatasetSpec] = {
    "MSMARCO": DatasetSpec(
        name="MSMARCO",
        kind="msmarco",
        hf_id="microsoft/ms_marco",
        benchmarks=("MS-MARCO", "MTEB", "MTEB-v2"),
        description="MS MARCO v2.1 QA passages from Bing queries (default: 1000 validation queries).",
    ),
    # ----- BEIR / MTEB / MTEB-v2 retrieval (all in standard BEIR layout) -----
    "NFCorpus": DatasetSpec(
        name="NFCorpus", kind="beir", hf_id="mteb/nfcorpus",
        benchmarks=("BEIR", "MTEB", "MTEB-v2"),
        description="Medical scientific literature retrieval (~3.6k docs).",
    ),
    "SciFact": DatasetSpec(
        name="SciFact", kind="beir", hf_id="mteb/scifact",
        benchmarks=("BEIR", "MTEB", "MTEB-v2"),
        description="Scientific claim verification (~5.2k docs).",
    ),
    "ArguAna": DatasetSpec(
        name="ArguAna", kind="beir", hf_id="mteb/arguana",
        benchmarks=("BEIR", "MTEB", "MTEB-v2"),
        description="Counter-argument retrieval (~8.7k docs).",
    ),
    "FiQA2018": DatasetSpec(
        name="FiQA2018", kind="beir", hf_id="mteb/fiqa",
        benchmarks=("BEIR", "MTEB", "MTEB-v2"),
        description="Financial opinion QA (~57k docs).",
    ),
    "SCIDOCS": DatasetSpec(
        name="SCIDOCS", kind="beir", hf_id="mteb/scidocs",
        benchmarks=("BEIR", "MTEB", "MTEB-v2"),
        description="Citation prediction over scientific papers (~25k docs).",
    ),
    "TRECCOVID": DatasetSpec(
        name="TRECCOVID", kind="beir", hf_id="mteb/trec-covid",
        benchmarks=("BEIR", "MTEB", "MTEB-v2"),
        description="COVID-19 scientific literature retrieval (~171k docs).",
    ),
    "Touche2020": DatasetSpec(
        name="Touche2020", kind="beir", hf_id="mteb/touche2020",
        benchmarks=("BEIR", "MTEB", "MTEB-v2"),
        description="Controversial-topic argument retrieval.",
    ),
    "Quora": DatasetSpec(
        name="Quora", kind="beir", hf_id="mteb/quora",
        benchmarks=("BEIR", "MTEB", "MTEB-v2"),
        description="Duplicate-question retrieval.",
    ),
}


def get(name: str) -> DatasetSpec:
    if name not in REGISTRY:
        raise SystemExit(
            f"Unknown dataset {name!r}. Known: {sorted(REGISTRY)}.\n"
            f"Run scripts/list_datasets.py to see them with descriptions."
        )
    return REGISTRY[name]


def dataset_paths(spec: DatasetSpec) -> dict[str, Path]:
    return {
        "data_dir":     ROOT / "data" / spec.name,
        "db_path":      ROOT / "db" / f"{spec.name}.sqlite",
        "bm25_path":    ROOT / "indexes" / f"{spec.name}_bm25.pkl",
        "results_path": ROOT / "results" / f"{spec.name}.md",
    }
