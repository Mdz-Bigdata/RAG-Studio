"""LanceDB 集成：向量后端（vector_store.backend: lancedb）与原生全文 / 混合检索、时间旅行、ANN 索引。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lancedb")

from prism.config import LanceDBConfig, Settings  # noqa: E402
from prism.index import Index  # noqa: E402
from prism.ingest import IngestPipeline  # noqa: E402
from prism.lance import LanceDBVectorStore  # noqa: E402
from prism.multimodal import BaseImageEmbedder, PerceptualImageEmbedder  # noqa: E402
from prism.retrieval import HybridRetriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _unit(n: int, dim: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_vector_store_upsert_search_filter_and_time_travel(tmp_path: Path):
    s = LanceDBVectorStore(tmp_path / "db", 8)
    v = _unit(4, 8)
    s.add(["a", "b", "c"], v[:3], texts=["员工出差住宿标准为一线城市每晚 600 元", "远程办公员工每月可申请办公室租金报销", "订单 SO-8817 已发货"], doc_ids=["d1", "d1", "d2"])
    assert len(s) == 3 and s.dim == 8
    first = s.versions()[0]["version"]
    assert s.search(v[1], 2)[0][0] == "b" and abs(s.search(v[1], 1)[0][1] - 1.0) < 1e-4   # 余弦相似度，与 numpy 后端同尺度
    assert [h[0] for h in s.search(v[1], 3, allowed_ids={"a", "c"})] in (["a", "c"], ["c", "a"])   # SQL 预过滤：不可见的块不占名额
    assert s.search(v[1], 3, allowed_ids=set()) == []

    s.add(["b"], v[3:4], texts=["办公室租金报销上限每月 1500 元"], doc_ids=["d1"])   # 同 id upsert，不重复
    assert len(s) == 3 and np.allclose(s.get_vectors(["b"])["b"], v[3], atol=1e-6)
    assert s.fts_search("租金报销", 3)[0][0] == "b" and s.fts_search("SO-8817", 3)[0][0] == "c"   # ICU 分词：中文与编号都能命中
    assert s.hybrid_search(v[0], "住宿标准", 3)[0][0] == "a"

    s.remove(["c"])
    assert len(s) == 2 and "c" not in s.get_vectors(["c"])
    assert {h[0] for h in s.search_at_version(first, v[2], 3)} == {"a", "b", "c"}         # 时间旅行：旧版本里还在
    assert s.info()["rows"] == 2 and any(ix["columns"] == ["text"] for ix in s.info()["indices"])


def test_ann_index_build_keeps_nearest_neighbour(tmp_path: Path):
    s = LanceDBVectorStore(tmp_path / "db", 16, cfg=LanceDBConfig(index_type="auto", index_min_rows=10**9))
    v = _unit(400, 16, seed=1)
    s.add([f"id{i}" for i in range(400)], v)
    s.save()
    assert not s.indexed                                      # auto：没到 index_min_rows 就精确扫描
    assert s.build_ann_index("IVF_HNSW_SQ") == "IVF_HNSW_SQ" and s.indexed
    s.configure(ef_search=200, nprobes=20, refine_factor=10)
    assert s.search(v[123], 1)[0][0] == "id123"
    assert s.search_exact(v[7], 1)[0][0] == "id7"


def test_pipeline_with_lancedb_backend_matches_numpy(tmp_path: Path):
    def build(backend: str, d: Path) -> Index:
        st = Settings.lite(index_dir=d)
        st.vector_store.backend = backend
        p = IngestPipeline(st)
        p.ingest_paths([ROOT / "data" / "raw"], save=True)
        return p.index

    lance_idx, numpy_idx = build("lancedb", tmp_path / "lance"), build("numpy", tmp_path / "numpy")
    assert lance_idx.signature()["vector_store"] == "lancedb" and len(lance_idx.vectors) == len(numpy_idx.vectors)
    queries = ["年假可以跨年使用吗", "K8s Ingress 502 怎么排查", "出差住宿标准能不能超额"]
    for q in queries:
        a = [cid for cid, _ in HybridRetriever(lance_idx.settings, lance_idx).retrieve_dense(q, 5)]
        b = [cid for cid, _ in HybridRetriever(numpy_idx.settings, numpy_idx).retrieve_dense(q, 5)]
        assert a == b, q                                       # 小表精确扫描：与 numpy 暴力余弦结果一致

    reloaded = Index.load(lance_idx.settings, tmp_path / "lance")   # 重新打开同一目录：数据在 LanceDB 里持久化
    assert len(reloaded.vectors) == len(lance_idx.vectors)

    st = reloaded.settings.model_copy(deep=True)
    st.retrieval.sparse_backend = "lancedb_fts"
    res = HybridRetriever(st, reloaded).retrieve("MySQL 1062 主键冲突有几种处理方案", top_k=3)
    assert res.trace.routes.get("sparse:q0") and res.contexts                  # 关键词路换成 LanceDB 原生全文检索
    assert any("1062" in c.chunk.text for c in res.contexts)


def test_multimodal_table_sync_and_cross_search(tmp_path: Path):
    from PIL import Image, ImageDraw

    from prism.lance.multimodal import LanceMultimodalTable

    (tmp_path / "img").mkdir()
    bar = Image.new("RGB", (400, 300), "white")
    d = ImageDraw.Draw(bar)
    for i, h in enumerate([120, 200, 160, 90]):
        d.rectangle([40 + i * 90, 280 - h, 100 + i * 90, 280], fill=(40, 90, 200))
    bar.save(tmp_path / "img" / "sales.png")
    pie = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(pie).pieslice([100, 40, 320, 260], 0, 130, fill=(220, 50, 50))
    pie.save(tmp_path / "img" / "share.png")
    (tmp_path / "report.md").write_text("# 销售报告\n\n## 季度销售\n\n各区域季度销售额如下图。\n\n![图1 各区域季度销售额柱状图](img/sales.png)\n\n"
                                        "## 市场份额\n\n各产品线市场份额。\n\n![图2 市场份额饼图](img/share.png)\n", encoding="utf-8")
    st = Settings.lite(index_dir=tmp_path / "index")
    p = IngestPipeline(st)
    p.ingest_paths([tmp_path / "report.md"], save=True)

    mm = LanceMultimodalTable(tmp_path / "lancedb")
    out = mm.sync_from_index(p.index)
    assert out["images"] == 2 and out["crossModal"] is False            # 离线感知特征：没有文本塔，正文行不进 CLIP 空间
    emb = p.index.images.embedder
    hits = mm.search(embedder=emb, image=bar, target="image", k=2)["hits"]
    assert hits[0]["title"].endswith("各区域季度销售额柱状图") and hits[0]["hasImage"]       # 以图搜图
    res = mm.search(embedder=emb, text="饼图", target="image", mode="vector", k=2)
    assert res["note"] and res["hits"][0]["title"].endswith("市场份额饼图")                 # 无文本塔时自动退到全文检索
    thumb = mm.thumbnail(hits[0]["id"])
    assert thumb and thumb[:2] == b"\xff\xd8"                                              # 缩略图字节存在 LanceDB 里（JPEG）
    assert mm.info()["byModality"] == {"image": 2, "text": 0}
    again = mm.sync_from_index(p.index)
    assert again["images"] == 2 and again["written"] == 0 and mm.info()["rows"] == 2       # 再同步：内容没变就不写，不重复


def test_aux_indexes_on_lancedb_and_legacy_fallback(tmp_path: Path):
    import json

    from prism.retrieval.rerank import NoopReranker

    def build(d: Path, backend: str) -> Index:
        st = Settings.lite(index_dir=d).with_overrides({"vector_store": {"backend": backend}, "enrich": {"summary_index": True}, "chunking": {"multi_granularity": True}})
        p = IngestPipeline(st, index=Index(st, directory=d))
        p.ingest_paths([ROOT / "data" / "raw" / "员工手册_2026版.md", ROOT / "data" / "raw" / "售后政策.md"], save=True)
        return p.index

    names = ("summary", "section", "sentence")
    lance, numpy_idx = build(tmp_path / "lance", "lancedb"), build(tmp_path / "numpy", "numpy")
    for n in names:
        assert lance.aux[n].backend == "lancedb" and len(lance.aux[n]) == len(numpy_idx.aux[n]) > 0
    assert json.loads((tmp_path / "lance" / "index_meta.json").read_text(encoding="utf-8"))["aux_store"] == dict.fromkeys(names, "lancedb")
    assert not (tmp_path / "lance" / "summary_vectors.npy").exists()            # 辅助索引不再写 numpy 文件

    r = HybridRetriever(lance.settings, lance, llm=None, reranker=NoopReranker())
    assert r.retrieve("年假可以跨年使用吗", top_k=3).trace.routes.get("summary:q0")
    detail = r.retrieve("出差住宿每晚多少钱", top_k=3)                           # 细节问题 → 句子级（LanceDB 混合检索）
    assert any(k.startswith("sentence:") and v for k, v in detail.trace.routes.items()) and any("600 元" in c.chunk.text for c in detail.contexts)

    loaded = Index.load(lance.settings, tmp_path / "lance", embedder=lance.embedder)
    assert all(loaded.aux[n].backend == "lancedb" and len(loaded.aux[n]) == len(lance.aux[n]) for n in names)
    doc_id = next(d for d, md in loaded.store.docs.items() if "售后" in (md.title or ""))
    before = len(loaded.aux["sentence"])
    loaded.remove_document(doc_id)
    assert len(loaded.aux["sentence"]) < before                                  # 删除文档时 LanceDB 辅助表同步清理

    # 已切到 LanceDB、但还没重建的知识库：沿用 numpy 建的辅助索引，不会出现空的辅助路
    legacy = Index.load(lance.settings, tmp_path / "numpy", embedder=lance.embedder)
    assert all(legacy.aux[n].backend == "numpy" and len(legacy.aux[n]) == len(numpy_idx.aux[n]) for n in names)


def test_image_index_on_lancedb_and_legacy_fallback(tmp_path: Path):
    from PIL import Image, ImageDraw

    from prism.multimodal import MultimodalRetriever

    (tmp_path / "img").mkdir()
    bar = Image.new("RGB", (400, 300), "white")
    d = ImageDraw.Draw(bar)
    for i, h in enumerate([120, 200, 160, 90]):
        d.rectangle([40 + i * 90, 280 - h, 100 + i * 90, 280], fill=(40, 90, 200))
    bar.save(tmp_path / "img" / "sales.png")
    pie = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(pie).pieslice([100, 40, 320, 260], 0, 130, fill=(220, 50, 50))
    pie.save(tmp_path / "img" / "share.png")
    (tmp_path / "report.md").write_text("# 销售报告\n\n## 季度销售\n\n各区域季度销售额如下图。\n\n![图1 各区域季度销售额柱状图](img/sales.png)\n\n"
                                        "## 市场份额\n\n各产品线市场份额。\n\n![图2 市场份额饼图](img/share.png)\n", encoding="utf-8")

    def build(d: Path, backend: str) -> Index:
        st = Settings.lite(index_dir=d).with_overrides({"vector_store": {"backend": backend}})
        p = IngestPipeline(st, index=Index(st, directory=d))
        p.ingest_paths([tmp_path / "report.md"], save=True)
        return p.index

    lance, numpy_idx = build(tmp_path / "lance", "lancedb"), build(tmp_path / "numpy", "numpy")
    imgs = lance.images
    assert imgs.store_backend == "lancedb" and len(imgs.visual) == len(imgs.desc) == len(imgs.records) == 2
    assert not (tmp_path / "lance" / "image_index" / "visual" / "vectors.npy").exists()       # 图片向量不再写 numpy 文件
    ids, mat = imgs.visual.ids, imgs.visual.matrix                                            # 图文混合检索要用的 ids / matrix 同序
    assert len(ids) == mat.shape[0] == 2 and np.allclose(mat[0], imgs.visual.get_vectors([ids[0]])[ids[0]])

    r = MultimodalRetriever(lance.settings, lance)
    assert r.image_to_image(bar).images[0].record.caption.endswith("柱状图")                      # 以图搜图（LanceDB 视觉向量）
    assert r.text_to_image("市场份额饼图").images[0].record.caption.endswith("饼图")               # 文搜图（描述向量 + BM25）

    loaded = Index.load(lance.settings, tmp_path / "lance", embedder=lance.embedder)
    assert loaded.images.store_backend == "lancedb" and len(loaded.images.visual) == 2 and len(loaded.images.records) == 2
    # 已切到 LanceDB、但还没重建的知识库：沿用 numpy 文件里的图片向量
    legacy = Index.load(lance.settings, tmp_path / "numpy", embedder=lance.embedder)
    assert legacy.images.store_backend == "numpy" and len(legacy.images.visual) == len(numpy_idx.images.visual) == 2


class _ToyClip(BaseImageEmbedder):
    """玩具双塔：图片走感知特征；文本里出现关键词就映射到对应参考图的向量（模拟图文对齐的空间）。"""

    name = "toy-clip"
    model_name = "toy"
    cross_modal = True

    def __init__(self, refs: dict) -> None:
        self.inner = PerceptualImageEmbedder()
        self.dim = self.inner.dim
        self.refs = refs

    def embed_images(self, images):
        return self.inner.embed_images(images)

    def embed_texts(self, texts):
        out = []
        for t in texts:
            hit = [self.inner.embed_image(im) for k, im in self.refs.items() if k in t]
            v = np.mean(hit, axis=0) if hit else np.full(self.dim, 1e-3, dtype=np.float32)
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.asarray(out, dtype=np.float32)


def test_multimodal_table_online_routes_and_incremental_sync(tmp_path: Path):
    from PIL import Image, ImageDraw

    from prism.lance.multimodal import LanceMultimodalTable, MultimodalLanceRoute
    from prism.multimodal import MultimodalRetriever

    (tmp_path / "img").mkdir()
    bar = Image.new("RGB", (400, 300), "white")
    d = ImageDraw.Draw(bar)
    for i, h in enumerate([120, 200, 160, 90]):
        d.rectangle([40 + i * 90, 280 - h, 100 + i * 90, 280], fill=(40, 90, 200))
    bar.save(tmp_path / "img" / "sales.png")
    pie = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(pie).pieslice([100, 40, 320, 260], 0, 130, fill=(220, 50, 50))
    pie.save(tmp_path / "img" / "share.png")
    (tmp_path / "report.md").write_text("# 销售报告\n\n## 季度销售\n\n各区域季度销售额如下图。\n\n![图1 各区域季度销售额柱状图](img/sales.png)\n\n"
                                        "## 市场份额\n\n各产品线市场份额。\n\n![图2 市场份额饼图](img/share.png)\n", encoding="utf-8")
    st = Settings.lite(index_dir=tmp_path / "index").with_overrides({"multimodal": {"lance_table": True}})
    index = Index(st, directory=tmp_path / "index", image_embedder=_ToyClip({"销售": bar, "柱状": bar, "份额": pie, "饼图": pie}))
    IngestPipeline(st, index=index).ingest_paths([tmp_path / "report.md"], save=True)     # 保存索引时自动同步多模态表

    mm = LanceMultimodalTable.of_index(index)
    assert mm.info()["byModality"] == {"image": 2, "text": len(index.store.child_ids())}   # 图片行 + 正文行（CLIP 文本向量）
    version = mm.info()["version"]
    assert mm.ensure_synced(index) == "fresh"
    assert mm.sync_from_index(index, embed="none")["written"] == 0 and mm.info()["version"] == version   # 没变化：不写新版本

    # 线上图文检索：文搜图 / 图搜文各多一路 LanceDB 多模态表，参与 RRF 融合
    r = MultimodalRetriever(st, index, lance=MultimodalLanceRoute(index, mm))
    res = r.text_to_image("市场份额饼图")
    assert res.images[0].record.caption.endswith("饼图") and res.routes["lance_mm"][0].chunk_id == res.images[0].record.image_id
    assert "lance_mm" in res.images[0].routes
    res = r.image_to_text(pie, query="市场份额")
    assert "lance_mm:image→text" in res.routes and "市场份额" in res.contexts[0].chunk.text
    assert "lance_mm:image→text" not in r.image_to_text(pie).routes              # 没有文字时与图文直达路相同，不重复计票
    assert "lance_mm" not in MultimodalRetriever(st, index).text_to_image("市场份额饼图").routes   # 不接入时与原来一致

    # 增量同步：改一张图的图注 → 只重写这一行
    pid = next(i for i, rec in index.images.records.items() if rec.caption.endswith("饼图"))
    index.images.records[pid] = index.images.records[pid].model_copy(update={"caption": "图2 产品线毛利率饼图"})
    out = mm.sync_from_index(index, embed="none")
    assert out["written"] == 1 and out["removed"] == 0 and mm.ensure_synced(index) == "fresh"
    # 查询前发现有变化会自动同步：改柱状图的图注，下一次文搜图就能用新图注检索到
    bid = next(i for i in index.images.records if i != pid)
    index.images.records[bid] = index.images.records[bid].model_copy(update={"caption": "图1 各区域净利润率"})
    assert r.text_to_image("净利润率").routes["lance_mm"][0].chunk_id == bid

    # 删除文档 → 保存索引时对应的行一起删掉
    index.remove_document(next(iter(index.store.docs)))
    index.save()
    assert mm.info()["rows"] == 0


def test_multimodal_table_keeps_text_rows_of_imageless_kb(tmp_path: Path):
    from prism.lance.multimodal import LanceMultimodalTable

    st = Settings.lite(index_dir=tmp_path / "index").with_overrides({"vector_store": {"backend": "lancedb"}, "multimodal": {"lance_table": True}})
    index = Index(st, directory=tmp_path / "index", image_embedder=_ToyClip({}))
    IngestPipeline(st, index=index).ingest_paths([ROOT / "data" / "raw" / "员工手册_2026版.md"], save=True)
    mm = LanceMultimodalTable.of_index(index)
    # 没有图片的知识库（也相当于整库重建的新目录）：保存索引时就建好正文行，供以图搜文
    assert mm.info()["byModality"] == {"image": 0, "text": len(index.store.child_ids())} and len(index.store.child_ids()) > 0
    out = mm.sync_from_index(index)                                  # 手动同步：没有变化就不写
    assert out["written"] == 0 and out["images"] == 0 and out["texts"] == len(index.store.child_ids())

    # 另一个进程（图文模型还没加载）再入库一份文档：保存索引时给新切片补图文向量，正文行跟着更新
    loaded = Index.load(st, tmp_path / "index", embedder=index.embedder, image_embedder=_ToyClip({}))
    assert not loaded.images.embedder_loaded
    IngestPipeline(st, index=loaded).ingest_paths([ROOT / "data" / "raw" / "售后政策.md"], save=True)
    assert mm.info()["byModality"]["text"] == len(loaded.store.child_ids()) > out["texts"]
    loaded.remove_document(next(d for d, md in loaded.store.docs.items() if "售后" in (md.title or "")))
    loaded.save()
    assert mm.info()["byModality"]["text"] == len(loaded.store.child_ids()) == out["texts"]
