"""On-disk embedding cache keyed by SHA-256(chunk text) + model name, so editing one chunk re-embeds only it.

Indexing the 530 chunks with bge-m3 on CPU takes ~1000 s, hence the cache. One .npz file per model.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from src.rag.retrievers.embedders import Embedder

LEGACY_CHECK_MIN_COSINE = 0.99   # a legacy row must match a fresh embedding of the same text this closely


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Maps text -> vector for one model. `embed` fills the gaps with the embedder and persists the result."""

    def __init__(self, directory: str | Path, model_name: str):
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
        self.path = Path(directory) / f"emb_{safe}.npz"
        self._vectors: dict[str, np.ndarray] = {}
        if self.path.exists():
            data = np.load(self.path, allow_pickle=False)
            self._vectors = dict(zip(data["keys"].tolist(), data["vectors"], strict=True))

    def __len__(self) -> int:
        return len(self._vectors)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._vectors)
        np.savez(self.path, keys=np.array(keys), vectors=np.stack([self._vectors[k] for k in keys]))

    def import_legacy(self, npy_path: str | Path, texts: list[str], embedder: Embedder) -> bool:
        """Adopt a positional cache (scripts/eval_retrieval.py: emb_<model>_<n>.npy) for `texts`, in order.

        Trusted only if the row count matches AND the first and last rows equal a fresh embedding of the same
        texts (guards against a stale file). Returns True if imported.
        """
        path = Path(npy_path)
        if not path.exists():
            return False
        matrix = np.load(path)
        if matrix.shape[0] != len(texts) or not texts:
            return False
        probes = [0, len(texts) - 1]
        fresh = embedder.encode_documents([texts[i] for i in probes])
        if any(float(matrix[i] @ f) < LEGACY_CHECK_MIN_COSINE for i, f in zip(probes, fresh, strict=True)):
            return False
        for text, vec in zip(texts, matrix, strict=True):
            self._vectors.setdefault(text_key(text), vec.astype(np.float32))
        self._save()
        return True

    def embed(self, texts: list[str], embedder: Embedder) -> np.ndarray:
        """Vectors for `texts` (same order); computes and stores only the missing ones."""
        keys = [text_key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._vectors]
        if missing:
            computed = embedder.encode_documents([texts[i] for i in missing])
            for i, vec in zip(missing, computed, strict=True):
                self._vectors[keys[i]] = vec.astype(np.float32)
            self._save()
        return np.stack([self._vectors[k] for k in keys])
