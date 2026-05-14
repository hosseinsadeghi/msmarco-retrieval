"""Central paths and knobs. Edit here to change defaults across all scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "indexes"
DB_PATH = ROOT / "db" / "msmarco.sqlite"
BM25_PATH = INDEX_DIR / "bm25.pkl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# MS MARCO is large. Limit how many queries we pull from the split.
# Each query carries ~10 candidate passages, so 1000 queries ≈ 8-10k unique passages.
DEFAULT_NUM_QUERIES = 1000
DEFAULT_SPLIT = "validation"
DEFAULT_CONFIG = "v2.1"
