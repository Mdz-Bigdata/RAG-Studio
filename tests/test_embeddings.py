"""向量模型：默认 BAAI/bge-m3（sentence-transformers）；测试里默认用哈希向量。

`RAG_TEST_EMBEDDER=1 pytest tests/test_embeddings.py -k live` 用本机缓存的真实 BGE-M3。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from prism.config import EmbeddingConfig
from prism.index.embeddings import HashEmbedder, SentenceTransformerEmbedder, build_embedder


def test_default_embedding_is_bge_m3():
    cfg = EmbeddingConfig()
    assert cfg.provider == "sentence_transformers" and cfg.model == "BAAI/bge-m3"


def test_kill_switch_makes_callers_fall_back(monkeypatch):
    monkeypatch.setenv("RAG_DISABLE_LOCAL_EMBEDDER", "1")
    with pytest.raises(RuntimeError, match="RAG_DISABLE_LOCAL_EMBEDDER"):
        SentenceTransformerEmbedder("BAAI/bge-m3")
    assert isinstance(build_embedder(EmbeddingConfig(provider="hash", dim=64)), HashEmbedder)


@pytest.mark.skipif(os.environ.get("RAG_TEST_EMBEDDER") != "1", reason="设置 RAG_TEST_EMBEDDER=1 才加载真实的 BGE-M3")
def test_live_bge_m3_matches_paraphrases():
    emb = SentenceTransformerEmbedder("BAAI/bge-m3")
    assert emb.dim == 1024
    docs = ["申请售后：在订单详情页点击申请售后，选择仅退款或退货退款", "年假可以跨年使用，最晚次年 3 月 31 日前休完", "K8s Ingress 返回 502 时先检查 upstream 健康检查"]
    d = emb.embed_documents(docs)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-3)
    q = emb.embed_queries(["这个东西咋退货", "年假能留到明年吗", "网关 502 怎么查"])
    sims = q @ d.T
    # 没有共同关键词的口语问法也要命中对应文档（这正是哈希向量做不到的）
    assert list(sims.argmax(axis=1)) == [0, 1, 2]
    assert all(sims[i, i] - np.delete(sims[i], i).max() > 0.05 for i in range(3))
