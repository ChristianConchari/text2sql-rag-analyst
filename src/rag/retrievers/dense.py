"""Dense retriever over the chunked corpus: cosine similarity, results grouped by article, similarity threshold."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.rag.retrievers.base import Passage
from src.rag.retrievers.corpus import Chunk, load_chunks
from src.rag.retrievers.embedders import Embedder
from src.rag.retrievers.embedding_cache import EmbeddingCache

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "rag" / "retrieval.yaml"
DEFAULT_K = 5
DEFAULT_MIN_SCORE = 0.57   # fallback when the YAML omits it; see configs/rag/retrieval.yaml


@dataclass(frozen=True)
class RetrievalConfig:
    chunks_path: Path
    cache_dir: Path
    model: str = "BAAI/bge-m3"
    k: int = DEFAULT_K
    min_score: float = DEFAULT_MIN_SCORE


def load_retrieval_config(path: str | Path = DEFAULT_CONFIG) -> RetrievalConfig:
    """Read configs/rag/retrieval.yaml; relative paths resolve against the repository root."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = Path(path).resolve().parents[2]
    return RetrievalConfig(chunks_path=root / cfg["chunks_path"], cache_dir=root / cfg["cache_dir"],
                           model=cfg.get("model", "BAAI/bge-m3"), k=int(cfg.get("k", DEFAULT_K)),
                           min_score=float(cfg.get("min_score", DEFAULT_MIN_SCORE)))


class DenseRetriever:
    """Embeds the corpus once (cached), then answers queries by cosine similarity.

    Extension point: a re-ranker would take `search` results (or a larger candidate list) and reorder them; it is
    deliberately not implemented (measured gain: 1 question of 25).
    """

    def __init__(self, chunks: list[Chunk], embedder: Embedder, cache: EmbeddingCache,
                 k: int = DEFAULT_K, min_score: float = DEFAULT_MIN_SCORE):
        self.chunks, self.embedder, self.k, self.min_score = chunks, embedder, k, min_score
        self._matrix = cache.embed([c.text for c in chunks], embedder)

    def search(self, query: str, k: int | None = None) -> list[Passage]:
        """Up to `k` articles whose best chunk scores >= min_score, best first. Empty list = no evidence."""
        scores = self._matrix @ self.embedder.encode_query(query)
        by_article: dict[str, list[int]] = defaultdict(list)
        for i in np.argsort(-scores):
            if scores[i] >= self.min_score:
                by_article[self.chunks[i].art_id].append(int(i))
        passages = []
        for indices in by_article.values():      # dict order = order of each article's best chunk
            first = self.chunks[indices[0]]
            text = "\n".join(self.chunks[i].text for i in sorted(indices))
            passages.append(Passage(first.art_id, first.norma, first.label, first.article_number, text,
                                    float(scores[indices[0]]), first.version))
        return passages[: (k or self.k)]


def build_dense_retriever(config: RetrievalConfig, embedder: Embedder) -> DenseRetriever:
    """Wire corpus + cache for `embedder`; adopts the legacy positional cache from eval_retrieval.py if valid."""
    chunks = load_chunks(config.chunks_path)
    cache = EmbeddingCache(config.cache_dir, embedder.name)
    if len(cache) == 0:
        short = embedder.name.split("/")[-1]
        cache.import_legacy(config.cache_dir / f"emb_{short}_{len(chunks)}.npy", [c.text for c in chunks], embedder)
    return DenseRetriever(chunks, embedder, cache, config.k, config.min_score)
