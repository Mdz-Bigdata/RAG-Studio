"""Milvus 向量后端（嵌入式 Milvus Lite）：与 numpy 后端结果一致、文档级过滤下推、整篇删除、重开持久化、整库入库检索。

默认跳过，`RAG_TEST_MILVUS=1 pytest tests/test_milvus.py` 单独跑：milvus_lite 的原生库一旦载入进程，同进程里再跑 onnxruntime（rapidocr OCR）
会段错误（tests/test_feishu.py 的 OCR 用例实测复现），不能和全量测试放在一个进程里。生产环境请连 Milvus 服务端，Lite 只用于开发。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RAG_TEST_MILVUS") != "1", reason="RAG_TEST_MILVUS=1 时单独运行（milvus_lite 与 onnxruntime 同进程会段错误）")
if os.environ.get("RAG_TEST_MILVUS") == "1":
    pytest.importorskip("pymilvus")
    pytest.importorskip("milvus_lite")

from prism.config import Settings  # noqa: E402
from prism.index import ChunkScope, Index, NumpyVectorStore  # noqa: E402
from prism.index.milvus_store import MilvusVectorStore  # noqa: E402
from prism.ingest import IngestPipeline  # noqa: E402
from prism.retrieval import HybridRetriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _unit(n: int, dim: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_store_matches_numpy_and_pushes_filters_down(tmp_path: Path):
    v = _unit(600, 32, seed=1)
    ids = [f"d{i // 20:03d}:p{i % 20 // 5:04d}:c{i % 5:03d}" for i in range(600)]
    m = MilvusVectorStore(tmp_path / "milvus.db", 32)
    m.add(ids, v, doc_ids=[i.split(":")[0] for i in ids], fresh=True)
    n = NumpyVectorStore(dim=32)
    n.add(ids, v)
    assert len(m) == 600 and m.dim == 32
    for q in v[:5]:
        assert [cid for cid, _ in m.search(q, 5)] == [cid for cid, _ in n.search(q, 5)]
        assert abs(m.search(q, 1)[0][1] - n.search(q, 1)[0][1]) < 1e-4            # 同一尺度：余弦相似度
    # 文档级范围：允许列表 / 排除列表都下推
    inc = ChunkScope(include=frozenset(["d003", "d007"]))
    assert {cid.split(":")[0] for cid, _ in m.search(v[0], 10, inc)} == {"d003", "d007"}
    exc = ChunkScope.of_docs([f"d{i:03d}" for i in range(30) if i != 0], [f"d{i:03d}" for i in range(30)])
    assert exc.exclude == {"d000"} and all(not cid.startswith("d000:") for cid, _ in m.search(v[0], 50, exc))
    assert m.search(v[0], 5, ChunkScope(include=frozenset())) == []
    # 段落 id 集合（老路径）
    assert {cid for cid, _ in m.search(v[0], 10, set(ids[:3]))} == set(ids[:3])
    # 读向量、删除、整篇删除
    assert np.allclose(m.get_vectors([ids[7]])[ids[7]], v[7], atol=1e-5)
    m.remove(ids[:5])
    assert len(m) == 595 and ids[0] not in m.get_vectors([ids[0]])
    m.remove_doc("d001")
    assert len(m) == 575 and not any(cid.startswith("d001:") for cid, _ in m.search(v[25], 50))
    m.add(ids[:5], v[:5])                                                        # upsert 回来
    assert len(m) == 580 and set(m.ids) == {i for i in ids if not i.startswith("d001:")}
    # 重开：数据在文件里
    again = MilvusVectorStore(tmp_path / "milvus.db", 0)
    assert len(again) == 580 and again.dim == 32 and again.search(v[7], 1)[0][0] == ids[7]


def test_pipeline_with_milvus_backend(tmp_path: Path):
    def build(backend: str, d: Path) -> Index:
        st = Settings.lite(index_dir=d)
        st.vector_store.backend = backend
        p = IngestPipeline(st)
        rep = p.ingest_paths([ROOT / "data" / "raw"], save=True)
        assert not rep.failed, rep.failed
        return p.index

    mv, base = build("milvus", tmp_path / "mv"), build("numpy", tmp_path / "np")
    assert mv.signature()["vector_store"] == "milvus" and len(mv.vectors) == len(base.vectors)
    assert (tmp_path / "mv" / "milvus.db").exists()
    for q in ["年假可以跨年使用吗", "K8s Ingress 502 怎么排查", "出差住宿标准能不能超额"]:
        a = [c for c, _ in HybridRetriever(mv.settings, mv).retrieve_dense(q, 5)]
        b = [c for c, _ in HybridRetriever(base.settings, base).retrieve_dense(q, 5)]
        assert a[0] == b[0], q
    res = HybridRetriever(mv.settings, mv).retrieve("年假可以跨年使用吗", top_k=3)
    assert res.contexts and res.trace.routes.get("dense:q0")
    # 删除一篇文档再重开
    doc = next(d for d, md in mv.store.docs.items() if "K8s" in (md.title or ""))
    n_before = len(mv.vectors)
    mv.remove_document(doc)
    assert len(mv.vectors) < n_before
    mv.save()
    again = Index.load(mv.settings, tmp_path / "mv")
    assert len(again.vectors) == len(mv.vectors) and doc not in again.store.docs
