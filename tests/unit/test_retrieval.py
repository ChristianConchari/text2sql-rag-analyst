from pathlib import Path

import numpy as np
import pytest

from src.rag.retrievers.corpus import load_chunks
from src.rag.retrievers.dense import DenseRetriever, RetrievalConfig, build_dense_retriever, load_retrieval_config
from src.rag.retrievers.embedders import HashEmbedder
from src.rag.retrievers.embedding_cache import EmbeddingCache, text_key

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rag_corpus.jsonl"
ROOT = Path(__file__).resolve().parents[2]


class CountingEmbedder(HashEmbedder):
    """Counts how many documents were embedded, to prove the cache is used."""

    def __init__(self):
        super().__init__()
        self.embedded = 0

    def encode_documents(self, texts):
        self.embedded += len(texts)
        return super().encode_documents(texts)


@pytest.fixture()
def chunks():
    return load_chunks(FIXTURE)


def make(chunks, tmp_path, min_score=0.2, k=5):
    return DenseRetriever(chunks, HashEmbedder(), EmbeddingCache(tmp_path, "hash-embedder"), k, min_score)


def test_fixture_is_synthetic_and_has_a_multi_chunk_article(chunks):
    assert {c.norma for c in chunks} == {"Norma Ficticia 100"}
    assert sum(c.art_id.endswith("Artículo 4") for c in chunks) == 2


def test_finds_the_right_article_first(chunks, tmp_path):
    top = make(chunks, tmp_path).search("¿Dentro de qué plazo se puede reclamar ante el Tribunal de Cuentas Imaginario?")[0]
    assert top.art_id == "Norma Ficticia 100|Artículo 2" and top.article_number == "2" and top.score > 0.2


def test_results_are_grouped_by_article_and_limited_by_k(chunks, tmp_path):
    passages = make(chunks, tmp_path, min_score=0.0).search("contratos de mantención de jardines renovación", k=3)
    assert len(passages) == 3 and len({p.art_id for p in passages}) == 3
    assert passages[0].art_id.endswith("Artículo 4") and passages[0].text.count("Norma Ficticia 100, Artículo 4") <= 2
    assert [p.score for p in passages] == sorted((p.score for p in passages), reverse=True)


def test_below_threshold_means_no_evidence(chunks, tmp_path):
    r = make(chunks, tmp_path, min_score=0.5)
    assert r.search("¿Cuál es la capital de Francia y quién ganó el mundial?") == []


def test_threshold_is_respected(chunks, tmp_path):
    passages = make(chunks, tmp_path, min_score=0.3).search("globos y banderines para ferias comunales")
    assert passages and all(p.score >= 0.3 for p in passages)


def test_cache_embeds_each_text_once(chunks, tmp_path):
    first = CountingEmbedder()
    DenseRetriever(chunks, first, EmbeddingCache(tmp_path, first.name))
    assert first.embedded == len(chunks)
    second = CountingEmbedder()
    DenseRetriever(chunks, second, EmbeddingCache(tmp_path, second.name))
    assert second.embedded == 0


def test_cache_recomputes_only_changed_text(chunks, tmp_path):
    e = CountingEmbedder()
    cache = EmbeddingCache(tmp_path, e.name)
    texts = [c.text for c in chunks]
    cache.embed(texts, e)
    e.embedded = 0
    EmbeddingCache(tmp_path, e.name).embed(texts[:-1] + ["texto nuevo"], e)
    assert e.embedded == 1


def test_cache_is_keyed_by_model_name(chunks, tmp_path):
    a, b = EmbeddingCache(tmp_path, "model/a"), EmbeddingCache(tmp_path, "model/b")
    assert a.path != b.path and text_key("x") == text_key("x") != text_key("y")


def test_legacy_positional_cache_is_adopted_only_when_valid(chunks, tmp_path):
    e = HashEmbedder()
    texts = [c.text for c in chunks]
    np.save(tmp_path / "emb_legacy_6.npy", e.encode_documents(texts))
    cache = EmbeddingCache(tmp_path, e.name)
    assert cache.import_legacy(tmp_path / "emb_legacy_6.npy", texts, e) and len(cache) == len(set(texts))
    stale = EmbeddingCache(tmp_path / "other", e.name)
    np.save(tmp_path / "emb_stale_6.npy", np.roll(e.encode_documents(texts), 1, axis=0))
    assert not stale.import_legacy(tmp_path / "emb_stale_6.npy", texts, e)
    assert not stale.import_legacy(tmp_path / "missing.npy", texts, e)


def test_build_dense_retriever_uses_config(tmp_path):
    cfg = RetrievalConfig(FIXTURE, tmp_path, "hash-embedder", k=2, min_score=0.0)
    r = build_dense_retriever(cfg, HashEmbedder())
    assert len(r.search("jardines")) == 2


def test_project_config_loads():
    cfg = load_retrieval_config()
    assert cfg.chunks_path == ROOT / "data/docs/chunks.jsonl" and 0 < cfg.min_score < 1 and cfg.k >= 1
