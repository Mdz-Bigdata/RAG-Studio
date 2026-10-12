"""摘要索引 / 多粒度分层索引 / 权限过滤 / 向量检索调优。全部离线（哈希向量 + BM25），不需要模型与网络。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prism.config import Settings
from prism.index import Index
from prism.index.vector_store import NumpyVectorStore
from prism.ingest import IngestPipeline
from prism.retrieval import HybridRetriever
from prism.retrieval.hierarchy import classify_query
from prism.retrieval.rerank import NoopReranker

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def _build(tmp_path: Path, **overrides) -> Index:
    s = Settings.lite(index_dir=tmp_path / "index").with_overrides(overrides)
    pipe = IngestPipeline(s, index=Index(s, directory=tmp_path / "index"))
    report = pipe.ingest_paths([RAW / "员工手册_2026版.md", RAW / "售后政策.md"], save=True)
    assert report.documents == 2 and not report.failed
    return pipe.index


def _retriever(index: Index) -> HybridRetriever:
    return HybridRetriever(index.settings, index, llm=None, reranker=NoopReranker())


def test_summary_index_route(tmp_path):
    index = _build(tmp_path, enrich={"summary_index": True})
    parents = [c for c in index.store.chunks.values() if c.kind == "parent"]
    assert parents and all(p.summary for p in parents)          # 无 LLM → 抽取式摘要
    assert len(index.aux["summary"]) == len(parents)
    res = _retriever(index).retrieve("年假可以跨年使用吗", top_k=3)
    assert "summary:q0" in res.trace.routes and res.trace.routes["summary:q0"]
    assert any("次年 3 月 31 日" in c.chunk.text for c in res.contexts)
    # 持久化后能读回
    loaded = Index.load(index.settings, tmp_path / "index", embedder=index.embedder)
    assert len(loaded.aux["summary"]) == len(parents)


def test_multi_granularity_routing(tmp_path):
    index = _build(tmp_path, chunking={"multi_granularity": True})
    kinds = {c.kind for c in index.store.chunks.values()}
    assert {"parent", "child", "section", "sentence"} <= kinds
    assert len(index.aux["section"]) and len(index.aux["sentence"])
    assert classify_query("年假有几天") == "detail"
    assert classify_query("介绍一下差旅管理制度") == "concept"
    assert classify_query("报销流程") == "general"

    r = _retriever(index)
    detail = r.retrieve("出差住宿每晚多少钱", top_k=3)
    assert any(k.startswith("sentence:") for k in detail.trace.routes)
    assert not any(k.startswith("section:") for k in detail.trace.routes)
    assert any("600 元" in c.chunk.text for c in detail.contexts)
    assert all(c.chunk.kind in ("parent", "section") for c in detail.contexts)   # 句子命中 → 返回所属段落

    sections = [c for c in index.store.chunks.values() if c.kind == "section"]
    parents = [c for c in index.store.chunks.values() if c.kind == "parent"]
    assert len(sections) < len(parents)                     # 按一级标题归并：一章包含多个段落
    assert any(len(s.child_ids) > 1 for s in sections)

    concept = r.retrieve("介绍一下差旅管理制度", top_k=3)
    assert any(k.startswith("section:") for k in concept.trace.routes)
    assert any("分层路由" in n for n in concept.trace.notes)
    ids = [c.chunk.chunk_id for c in concept.contexts]
    assert len(ids) == len(set(ids))
    covered = {pid for c in concept.contexts if c.chunk.kind == "section" for pid in c.chunk.child_ids}
    assert not any(c.chunk.chunk_id in covered for c in concept.contexts)   # 被章节覆盖的段落不重复出现

    # 删除文档时辅助索引同步清理
    doc_id = next(d for d, md in index.store.docs.items() if "售后" in (md.title or ""))
    before = len(index.aux["sentence"])
    index.remove_document(doc_id)
    assert len(index.aux["sentence"]) < before
    assert not any(c.doc_id == doc_id for c in index.store.chunks.values())


def test_access_filter(tmp_path):
    index = _build(tmp_path)
    after_sale = next(d for d, md in index.store.docs.items() if "售后" in (md.title or ""))
    index.store.docs[after_sale].access_level = 2
    index.store.docs[after_sale].allowed_departments = ["客服部"]
    r = _retriever(index)
    q = "订单 SO-8817 的状态"

    public = r.retrieve(q, top_k=3, filters={"user": {"level": 0}})
    assert not any(c.chunk.doc_id == after_sale for c in public.contexts)
    assert any("权限过滤" in n for n in public.trace.notes)

    wrong_dept = r.retrieve(q, top_k=3, filters={"user": {"level": 2, "department": "财务部"}})
    assert not any(c.chunk.doc_id == after_sale for c in wrong_dept.contexts)

    ok = r.retrieve(q, top_k=3, filters={"user": {"level": 2, "department": "客服部"}})
    assert any("SO-8817" in c.chunk.text for c in ok.contexts)
    admin = r.retrieve(q, top_k=3, filters={"user": {"level": 3}})
    assert any("SO-8817" in c.chunk.text for c in admin.contexts)
    # 不传身份 = 不做权限过滤（CLI / 评测）
    assert any("SO-8817" in c.chunk.text for c in r.retrieve(q, top_k=3).contexts)


@pytest.mark.parametrize("pca_dim,quantize", [(64, "none"), (None, "int8"), (64, "int8")])
def test_vector_tuning_recall(pca_dim, quantize):
    rng = np.random.default_rng(0)
    # 有结构的数据（低秩 + 噪声），更接近真实 embedding
    basis = rng.normal(size=(32, 256))
    x = rng.normal(size=(2000, 32)) @ basis + 0.1 * rng.normal(size=(2000, 256))
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    store = NumpyVectorStore(dim=256)
    store.add([f"c{i}" for i in range(len(x))], x)
    queries = x[rng.choice(len(x), 30, replace=False)] + 0.05 * rng.normal(size=(30, 256)).astype(np.float32)

    exact = [[cid for cid, _ in store.search_exact(q, 10)] for q in queries]
    store.configure(pca_dim=pca_dim, quantize=quantize, rescore=True, oversample=4)
    tuned = [[cid for cid, _ in store.search(q, 10)] for q in queries]
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(exact, tuned)])
    assert recall >= 0.95
    info = store.coarse_info()
    assert info["bytes"] < info["full_bytes"]

    store.configure(pca_dim=None, quantize="none")
    assert not store.tuned
