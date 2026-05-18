"""Embedding model wrapper.

We normalize embeddings to unit length so that a dot-product equals cosine
similarity. That makes the vector search step a single matrix multiply.
"""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def encode(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    model = get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)
