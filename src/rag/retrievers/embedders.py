"""Text embedders. `SentenceTransformerEmbedder` is the real one (optional `rag` dependency group);
`HashEmbedder` is a deterministic bag-of-words stand-in for tests (no model, no torch)."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Maps text to L2-normalized vectors. `name` identifies the model in cache keys."""

    name: str

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFD", text.lower())
    return re.findall(r"\w+", "".join(c for c in text if unicodedata.category(c) != "Mn"))


class HashEmbedder:
    """Hashed bag-of-words (term counts, cosine-normalized). Same input -> same vector, across runs and machines."""

    def __init__(self, dim: int = 512, name: str = "hash-embedder"):
        self.dim, self.name = dim, name

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in _words(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class SentenceTransformerEmbedder:
    """Dense embedder on CPU. Imports sentence-transformers lazily so the core install does not need torch."""

    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 16):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("sentence-transformers is not installed; run: UV_TORCH_BACKEND=cpu uv sync --group rag") from e
        self.name, self.batch_size = model_name, batch_size
        self._model = SentenceTransformer(model_name, device="cpu")
        # e5-style models need role prefixes; bge-m3 does not.
        self._query_prefix, self._doc_prefix = ("query: ", "passage: ") if "e5" in model_name else ("", "")

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._model.encode([self._doc_prefix + t for t in texts], batch_size=self.batch_size,
                                  normalize_embeddings=True)

    def encode_query(self, text: str) -> np.ndarray:
        return self._model.encode([self._query_prefix + text], normalize_embeddings=True)[0]
