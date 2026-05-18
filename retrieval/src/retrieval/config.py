"""Central paths and model choices. Per-dataset paths live in datasets.py."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
