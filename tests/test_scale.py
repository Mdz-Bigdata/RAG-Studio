"""千万级配置：文档级检索范围（ChunkScope）、SQLite 块存储、LanceDB jieba 全文索引替代内存 BM25、
过滤下推、自适应 nprobes、ANN 索引按数据量重训。

正确性以默认配置（内存块存储 + numpy + bm25s）为基准：同一份语料，两种配置检索结果应当一致。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lancedb")

from prism.config import LanceDBConfig, Settings  # noqa: E402
from prism.index import ChunkScope, ChunkStore, Index, SqliteChunkStore  # noqa: E402
from prism.ingest import IngestPipeline  # noqa: E402
from prism.lance import LanceDBVectorStore  # noqa: E402
from prism.lance.keyword import LanceKeywordIndex  # noqa: E402
from prism.retrieval import HybridRetriever  # noqa: E402
from prism.schemas import Chunk, DocMetadata  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ["年假可以跨年使用吗", "K8s Ingress 502 怎么排查", "出差住宿标准能不能超额", "MySQL 1062 主键冲突有几种处理方案", "无理由退货谁承担"]


def scale_settings(d: Path) -> Settings:
    st = Settings.lite(index_dir=d)
    st.storage.chunk_store = "sqlite"
    st.vector_store.backend = "lancedb"
    st.vector_store.lancedb.fts_tokenizer = "jieba"
    st.retrieval.sparse_backend = "lancedb_fts"
    return st


def build(st: Settings) -> Index:
    p = IngestPipeline(st)
    report = p.ingest_paths([ROOT / "data" / "raw"], save=True)
    assert not report.failed, report.failed
    return p.index


@pytest.fixture(scope="module")
def pair(tmp_path_factory) -> tuple[Index, Index]:
    base = tmp_path_factory.mktemp("scale")
    return build(Settings.lite(index_dir=base / "memory")), build(scale_settings(base / "scale"))


def parents(res) -> list[str]:
    return [c.chunk.chunk_id for c in res.contexts]


# --------------------------------------------------------------------------------------
def test_chunk_scope_semantics():
    sc = ChunkScope.of_docs(["a"], ["a", "b", "c"])
    assert sc.include == {"a"} and "a:p0001:c000" in sc and "b:p0001" not in sc and sc.sql() == "doc_id IN ('a')"
    wide = ChunkScope.of_docs(["a", "b"], ["a", "b", "c"])            # 允许的多：记排除的那份
    assert wide.include is None and wide.exclude == {"c"} and wide.sql() == "doc_id NOT IN ('c')"
    assert "zz:p0001" in wide                                          # 排除模式：不认识的文档算允许
    assert ChunkScope.of_docs(["a", "b"], ["a", "b"]).unrestricted
    empty = ChunkScope.of_docs([], ["a"])
    assert not empty and empty.sql() == "false"
    assert wide.restrict(["b", "c", "d"]).include == {"b", "d"}
    assert ChunkScope(include=frozenset(["o'k"])).sql() == "doc_id IN ('o''k')"   # SQL 引号转义
    with pytest.raises(TypeError):
        len(sc)                                                        # 没给 expand 就不能展开


def test_scale_profile_matches_default(pair):
    mem, big = pair
    assert isinstance(big.store, SqliteChunkStore) and isinstance(big.bm25, LanceKeywordIndex)
    assert len(big.store) == len(mem.store) and big.store.count_by_kind() == mem.store.count_by_kind()
    assert not (big.directory / "bm25_corpus.jsonl").exists() and not (big.directory / "chunks.jsonl").exists()   # 千万级配置：不再整份落盘
    for q in QUERIES:
        a = HybridRetriever(mem.settings, mem).retrieve(q, top_k=3)
        b = HybridRetriever(big.settings, big).retrieve(q, top_k=3)
        assert a.contexts and b.contexts, q
        assert parents(a)[0] == parents(b)[0], q                       # 第一名一致（关键词路换成了同一套 jieba 分词的全文索引）
        assert b.trace.routes.get("sparse:q0"), q


def test_expired_and_permission_filters_push_down(pair, monkeypatch):
    mem, big = pair
    r = HybridRetriever(big.settings, big)
    scope = r._allowed_ids(None, date(2026, 6, 1), big.settings.retrieval)
    old = next(d for d, md in big.store.docs.items() if md.valid_to)  # 2024 版员工手册已过期
    assert isinstance(scope, ChunkScope) and not scope.allows_doc(old) and scope.sql() == f"doc_id NOT IN ('{old}')"
    seen: list[str] = []
    orig = big.vectors._filter
    monkeypatch.setattr(big.vectors, "_filter", lambda allowed: (seen.append(repr(allowed)), orig(allowed))[1])
    res = r.retrieve("年假可以跨年使用吗", top_k=5, on_date=date(2026, 6, 1))
    assert res.contexts and all(c.chunk.doc_id != old for c in res.contexts)
    assert seen and all("exclude=1" in s for s in seen)                 # 交给向量 / 全文索引的是文档级范围，不是段落 id 集合
    # 结果与默认配置一致
    base = HybridRetriever(mem.settings, mem).retrieve("年假可以跨年使用吗", top_k=5, on_date=date(2026, 6, 1))
    assert parents(base)[0] == parents(res)[0]
    # 问历史规定时放行过期文档
    hist = r.retrieve("2024 版员工手册里年假怎么规定", top_k=5, on_date=date(2026, 6, 1))
    assert any(c.chunk.doc_id == old for c in hist.contexts)


def test_permission_scope_and_metadata_change_invalidates_cache(pair):
    _, big = pair
    st = big.settings.model_copy(deep=True)
    st.retrieval.access.enabled = True
    r = HybridRetriever(st, big)
    secret = sorted(big.store.docs)[0]
    user = {"level": 0}
    assert r._allowed_ids({"user": user}, None, st.retrieval).allows_doc(secret)
    big.store.docs[secret].access_level = 2                            # 原地改元数据（语料治理就是这么改的）
    big.store.touch_doc(secret)
    try:
        sc = r._allowed_ids({"user": user}, None, st.retrieval)
        assert not sc.allows_doc(secret)                               # 缓存按 generation 失效
        assert all(c.chunk.doc_id != secret for c in r.retrieve("员工手册", filters={"user": user}, top_k=5).contexts)
        big.store.save(big.directory)
        assert SqliteChunkStore.load(big.directory).docs[secret].access_level == 2   # 写回磁盘
    finally:
        big.store.docs[secret].access_level = 0
        big.store.touch_doc(secret)
        big.store.save(big.directory)


def test_explicit_doc_level_filters(pair):
    mem, big = pair
    doc = next(d for d, md in big.store.docs.items() if "K8s" in (md.title or ""))
    for idx in (mem, big):
        r = HybridRetriever(idx.settings, idx)
        res = r.retrieve("怎么排查", filters={"doc_id": doc}, top_k=5)
        assert res.contexts and {c.chunk.doc_id for c in res.contexts} == {doc}


def test_remove_and_readd_document(tmp_path: Path):
    idx = build(scale_settings(tmp_path / "idx"))
    doc = next(d for d, md in idx.store.docs.items() if "售后" in (md.title or ""))
    n_vec, n_chunks = len(idx.vectors), len(idx.store)
    chunks = idx.store.chunks_of_doc(doc)
    md = idx.store.docs[doc]
    idx.remove_document(doc)
    assert len(idx.vectors) == n_vec - sum(1 for c in chunks if c.kind == "child") and len(idx.store) == n_chunks - len(chunks)
    assert not idx.bm25.search("无理由退货", 10) or all(not cid.startswith(doc) for cid, _ in idx.bm25.search("无理由退货", 10))
    idx.add_document(doc, md, chunks)                                   # 追加写（fresh）：不重复
    idx.add_document(doc, md, chunks)                                   # 同一文档再入库：先整篇删再追加
    assert len(idx.vectors) == n_vec and len(idx.store) == n_chunks
    idx.save()
    again = Index.load(idx.settings, idx.directory)
    assert len(again.store) == n_chunks and again.bm25.search("无理由退货", 3)[0][0].startswith(doc)


def test_sqlite_store_parity_and_migration(tmp_path: Path):
    st = Settings.lite(index_dir=tmp_path / "m")
    m_idx = build(st)
    store: ChunkStore = m_idx.store
    child = next(c for c in store.chunks.values() if c.kind == "child")
    # 内存 → SQLite：加载时自动导入，不需要重新解析
    st2 = st.model_copy(deep=True)
    st2.storage.chunk_store = "sqlite"
    s_idx = Index.load(st2, tmp_path / "m")
    sq = s_idx.store
    assert isinstance(sq, SqliteChunkStore) and (tmp_path / "m" / "chunks.jsonl.migrated").exists()
    assert len(sq) == len(store) and sq.docs.keys() == store.docs.keys()
    assert sq.get(child.chunk_id) == child and sq.parent_of(child.chunk_id) == store.parent_of(child.chunk_id)
    assert [c.chunk_id for c in sq.neighbors(child.chunk_id, 2)] == [c.chunk_id for c in store.neighbors(child.chunk_id, 2)]
    assert sq.children_of(child.parent_id) == store.children_of(child.parent_id)
    assert sorted(sq.child_ids()) == sorted(store.child_ids())
    assert child.chunk_id in sq.chunks and sq.chunks[child.chunk_id] == child and sum(1 for _ in sq.chunks.values()) == len(store)
    # 语料治理剔除段落：parent 的 child_ids 同步清理并落盘
    parent = sq.parent_of(child.chunk_id)
    assert sq.remove_chunks({child.chunk_id}) == [child.chunk_id]
    assert child.chunk_id not in SqliteChunkStore.load(tmp_path / "m").get(parent.chunk_id).child_ids
    # 检索结果与内存版一致
    for q in QUERIES[:3]:
        a = HybridRetriever(st, m_idx).retrieve(q, top_k=3)
        b = HybridRetriever(st2, s_idx).retrieve(q, top_k=3)
        assert parents(a)[0] == parents(b)[0] or child.parent_id in parents(a)[:1], q
    # SQLite → 内存：切回去也能自动迁移
    s_idx.store.close()
    back = Index.load(st, tmp_path / "m")
    assert isinstance(back.store, ChunkStore) and len(back.store) == len(store) - 1
    assert not (tmp_path / "m" / "chunks.sqlite").exists() and (tmp_path / "m" / "chunks.sqlite.old").exists()


def test_sqlite_store_save_to_other_directory(tmp_path: Path):
    s = SqliteChunkStore(tmp_path / "a")
    s.add_document("d", DocMetadata(title="t"))
    s.add_chunks([Chunk(chunk_id="d:p0000", doc_id="d", kind="parent", text="x", child_ids=["d:p0000:c000"]),
                  Chunk(chunk_id="d:p0000:c000", doc_id="d", kind="child", text="x", parent_id="d:p0000")])
    s.save(tmp_path / "b")                                              # 整库重建的暂存目录：整库复制过去
    t = SqliteChunkStore.load(tmp_path / "b")
    assert len(t) == 2 and t.docs["d"].title == "t" and t.parent_of("d:p0000:c000").chunk_id == "d:p0000"


# --------------------------------------------------------------------------------------
def _unit(n: int, dim: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_hnsw_refine_does_not_crash_and_probes_scale(tmp_path: Path):
    """HNSW 要求 ef ≥ 实际取回数：refine_factor × limit 超过 ef_search 时以前会直接报错。"""
    s = LanceDBVectorStore(tmp_path / "db", 16, cfg=LanceDBConfig(index_type="IVF_HNSW_SQ", index_min_rows=300))
    v = _unit(3000, 16, seed=2)
    s.add([f"d{i // 10}:p0000:c{i % 10:03d}" for i in range(3000)], v)
    s.save()
    assert s.indexed and s._ann_meta()["partitions"] == s.partitions_for(3000, "IVF_HNSW_SQ")
    s.configure(ef_search=32, nprobes=1, refine_factor=5, nprobes_ratio=0.5)
    assert s._probes() == max(1, -(-s._ann_meta()["partitions"] // 2))
    hits = s.search(v[42], 50)                                          # limit 50 × refine 5 = 250 > ef 32
    assert len(hits) == 50 and hits[0][0] == "d4:p0000:c002"
    scope = ChunkScope(include=frozenset(["d4", "d7"]))
    assert {h[0].split(":")[0] for h in s.search(v[42], 20, scope)} == {"d4", "d7"}   # 文档级预过滤（doc_id 从 id 前缀补齐）


def test_ann_index_retrains_when_data_grows(tmp_path: Path):
    s = LanceDBVectorStore(tmp_path / "db", 8, cfg=LanceDBConfig(index_type="IVF_FLAT", index_min_rows=300))
    s.add([f"a{i}" for i in range(400)], _unit(400, 8, 3))
    s.save()
    assert s._ann_meta()["rows"] == 400
    s.add([f"b{i}" for i in range(1300)], _unit(1300, 8, 4), fresh=True)
    s.save()
    assert s._ann_meta()["rows"] == 1700 and s._ann_meta()["partitions"] == s.partitions_for(1700, "IVF_FLAT")


# --------------------------------------------------------------------------------------
# 批量导入
def _bulk_build(st: Settings, workers: int = 0) -> Index:
    from prism.ingest.bulk import BulkIndexer

    idx = Index(st, directory=st.paths.index_dir)
    p = IngestPipeline(st, index=idx)
    bulk = BulkIndexer(idx, batch_chunks=16, tokenize_workers=workers)          # 小批次：多批写入、跨批续传都走到
    for path in sorted((ROOT / "data" / "raw").iterdir()):
        doc, chunks, tables = p.chunk_document(p.parse(path))
        bulk.add(doc.doc_id, doc.metadata, chunks, tables=tables)
    stats = bulk.finish()
    assert stats.batches > 1 and stats.docs == 5
    return idx


def test_bulk_import_matches_per_document_ingest(pair, tmp_path: Path):
    _, big = pair
    bulk = _bulk_build(scale_settings(tmp_path / "bulk"), workers=2)               # 多进程分词
    assert len(bulk.store) == len(big.store) and len(bulk.vectors) == len(big.vectors) and len(bulk.entities) == len(big.entities)
    assert not (tmp_path / "bulk" / "bulk_import.journal").exists()
    again = Index.load(bulk.settings, tmp_path / "bulk")
    for q in QUERIES:
        a = HybridRetriever(big.settings, big).retrieve(q, top_k=3)
        b = HybridRetriever(again.settings, again).retrieve(q, top_k=3)
        assert parents(a)[0] == parents(b)[0], q


def test_bulk_finish_does_not_rewrite_table(tmp_path: Path):
    """批次够大时 finish 不合并，save() 也不能按 optimize_every 偷偷合并（整表重写 + 旧版本保留 = 磁盘翻倍）。"""
    st = scale_settings(tmp_path / "idx")
    st.vector_store.lancedb.optimize_every = 2
    idx = _bulk_build_with(st, batch_chunks=40)
    data = idx.directory / "lancedb" / "children.lance" / "data"
    files_before = {p.name for p in data.iterdir()}
    assert idx.vectors._writes == 0 and files_before
    idx.save()
    assert {p.name for p in data.iterdir()} == files_before          # 没有重写出新的数据文件
    # 从 LanceDB 正文列重建内存 BM25、流式遍历分词结果：不依赖 pylance
    from prism.index import BM25Store

    assert len(BM25Store.from_lance(idx.vectors, st.sparse)) == len(idx.vectors)
    assert sum(1 for _ in idx.bm25.corpus_tokens) == len(idx.vectors)


def _bulk_build_with(st: Settings, *, batch_chunks: int) -> Index:
    from prism.ingest.bulk import BulkIndexer

    idx = Index(st, directory=st.paths.index_dir)
    p = IngestPipeline(st, index=idx)
    bulk = BulkIndexer(idx, batch_chunks=batch_chunks)
    for path in sorted((ROOT / "data" / "raw").iterdir()):
        doc, chunks, tables = p.chunk_document(p.parse(path))
        bulk.add(doc.doc_id, doc.metadata, chunks, tables=tables)
    bulk.finish(compact=False)
    return idx


def test_bulk_import_recovers_half_written_batch(tmp_path: Path):
    from prism.ingest.bulk import BulkIndexer

    st = scale_settings(tmp_path / "idx")
    idx = Index(st, directory=st.paths.index_dir)
    p = IngestPipeline(st, index=idx)
    doc, chunks, _ = p.chunk_document(p.parse(ROOT / "data" / "raw" / "售后政策.md"))
    children = [c for c in chunks if c.kind == "child"]
    # 模拟崩溃：向量写进去了、段落写了一半、文档行没写，journal 里只有 pending
    idx.vectors.add([c.chunk_id for c in children], idx.embedder.embed_documents([c.text for c in children]), texts=[c.text for c in children], fresh=True)
    idx.store.add_chunks(chunks[:3])
    (idx.directory / "bulk_import.journal").write_text(json.dumps({"batch": 1, "pending": [doc.doc_id]}) + "\n", encoding="utf-8")
    bulk = BulkIndexer(idx)
    assert len(idx.vectors) == 0 and len(idx.store) == 0 and not bulk.has(doc.doc_id)   # 半批数据清掉了
    bulk.add(doc.doc_id, doc.metadata, chunks)
    bulk.finish()
    assert len(idx.vectors) == len(children) and len(idx.store) == len(chunks)          # 重导一遍，不重复


def _write_jsonl(path: Path) -> list[dict]:
    recs = []
    for f in sorted((ROOT / "data" / "raw").glob("*.md")):
        text = f.read_text(encoding="utf-8")
        recs.append({"id": f.stem, "title": f.stem, "text": text, "tags": "制度,内部", "access_level": 1 if "K8s" in f.stem else 0,
                     "valid_to": "2025-12-31" if "2024" in f.stem else None})
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    return recs


def _run_cli(argv: list[str], monkeypatch, capsys) -> dict:
    import runpy

    monkeypatch.setattr("sys.argv", ["bulk_import.py", *argv])
    runpy.run_path(str(ROOT / "scripts" / "bulk_import.py"), run_name="__main__")
    out = capsys.readouterr().out
    return json.loads(out[out.index("{"):])


def test_bulk_import_cli_index_dir_and_resume(tmp_path: Path, monkeypatch, capsys):
    import yaml

    cfg = scale_settings(tmp_path / "unused").model_dump(mode="json")
    cfg["storage"]["chunk_store"] = "sqlite"
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    recs = _write_jsonl(tmp_path / "docs.jsonl")
    argv = [str(tmp_path / "docs.jsonl"), "--index-dir", str(tmp_path / "idx"), "--config", str(tmp_path / "config.yaml"), "--batch-chunks", "8", "--tokenize-workers", "1"]
    first = _run_cli(argv, monkeypatch, capsys)
    assert first["read"] == len(recs) and first["docs"] == len(recs) and first["failed"] == 0 and first["batches"] > 1
    second = _run_cli(argv, monkeypatch, capsys)
    assert second["skipped"] == len(recs) and second["docs"] == 0                     # 重跑：全部跳过
    idx = Index.load(Settings.load(tmp_path / "config.yaml"), tmp_path / "idx", strict=False)
    k8s = next(d for d, md in idx.store.docs.items() if "K8s" in md.title)
    assert idx.store.docs[k8s].access_level == 1 and idx.store.docs[k8s].tags == ["制度", "内部"]   # 元数据带进来了
    old = next(d for d, md in idx.store.docs.items() if "2024" in md.title)
    assert idx.store.docs[old].valid_to == date(2025, 12, 31)
    res = HybridRetriever(idx.settings, idx).retrieve("K8s Ingress 502 怎么排查", top_k=3)
    assert res.contexts and res.contexts[0].chunk.doc_id == k8s


def test_bulk_import_cli_into_web_kb(tmp_path: Path, monkeypatch, capsys):
    from fastapi.testclient import TestClient

    from app import config as app_config_mod

    cfg = app_config_mod.app_config
    saved = (cfg.data_dir, cfg.upload_dir, cfg.index_root)
    cfg.data_dir, cfg.upload_dir, cfg.index_root = tmp_path, tmp_path / "uploads", tmp_path / "indexes"
    for d in (cfg.upload_dir, cfg.index_root):
        d.mkdir(parents=True, exist_ok=True)
    from app.main import app

    try:
        with TestClient(app) as c:
            r = c.put("/api/v1/settings/", json={"overrides": {"storage": {"chunk_store": "sqlite"}, "retrieval": {"sparse_backend": "lancedb_fts"},
                                                               "vector_store": {"backend": "lancedb", "lancedb": {"fts_tokenizer": "jieba"}}}})
            assert r.status_code == 200, r.text
            kb = c.post("/api/v1/knowledge-bases/", json={"name": "批量导入库"}).json()["id"]
            recs = _write_jsonl(tmp_path / "docs.jsonl")
            out = _run_cli([str(tmp_path / "docs.jsonl"), "--kb", kb, "--tokenize-workers", "1"], monkeypatch, capsys)
            assert out["docs"] == len(recs)
            docs = c.get(f"/api/v1/documents/?kbId={kb}").json()
            assert len(docs) == len(recs) and all(d["status"] == "COMPLETED" and d["chunkCount"] > 0 for d in docs)
            res = c.post(f"/api/v1/retrieval/{kb}/playground", json={"query": "年假可以跨年使用吗"})
            assert res.status_code == 200, res.text
            assert res.json()["reranked"] and res.json()["routes"].get("sparse:q0")      # 应用加载的是批量导入建好的索引
            again = _run_cli([str(tmp_path / "docs.jsonl"), "--kb", kb, "--tokenize-workers", "1"], monkeypatch, capsys)
            assert again["skipped"] == len(recs)

            # 太大的知识库不自动整库重建：只记待重建
            from app.services.reindex_service import PENDING_KEY, reindex_service
            from app.db import Repository, SessionLocal

            c.put("/api/v1/settings/", json={"overrides": {"storage": {"chunk_store": "sqlite", "auto_rebuild_max_chunks": 1}, "chunking": {"child_tokens": 120},
                                                           "retrieval": {"sparse_backend": "lancedb_fts"}, "vector_store": {"backend": "lancedb", "lancedb": {"fts_tokenizer": "jieba"}}}})
            reindex_service._check_and_enqueue("测试")
            assert not reindex_service.is_active(kb)
            with SessionLocal() as db:
                assert "bulk_import.py --kb" in (Repository(db).get_setting(PENDING_KEY) or {}).get(kb, "")
            # 批量重建：新索引换上，旧索引留底，待重建记录清除
            rebuilt = _run_cli(["--kb", kb, "--rebuild", "--tokenize-workers", "1"], monkeypatch, capsys)
            assert rebuilt["docs"] == len(recs) and rebuilt["failed"] == 0
            assert list(cfg.index_root.glob(f"{kb}.old-*")) and not (cfg.index_root / f"{kb}.rebuild").exists()
            with SessionLocal() as db:
                assert kb not in (Repository(db).get_setting(PENDING_KEY) or {})
            res = c.post(f"/api/v1/retrieval/{kb}/playground", json={"query": "K8s Ingress 502 怎么排查"})
            assert res.status_code == 200 and res.json()["contexts"]
            c.post("/api/v1/settings/reset")
    finally:
        cfg.data_dir, cfg.upload_dir, cfg.index_root = saved


def test_document_page_endpoint(tmp_path: Path):
    """知识库几十万篇文档时「知识库」页分页取：offset / limit / 按文件名或标题搜索，processing 是整库的处理中数。"""
    from fastapi.testclient import TestClient

    from app import config as app_config_mod
    from app.db import Document, SessionLocal

    cfg = app_config_mod.app_config
    saved = (cfg.data_dir, cfg.upload_dir, cfg.index_root)
    cfg.data_dir, cfg.upload_dir, cfg.index_root = tmp_path, tmp_path / "uploads", tmp_path / "indexes"
    for d in (cfg.upload_dir, cfg.index_root):
        d.mkdir(parents=True, exist_ok=True)
    from app.main import app

    try:
        with TestClient(app) as c:
            kb = c.post("/api/v1/knowledge-bases/", json={"name": "分页库"}).json()["id"]
            with SessionLocal() as db:   # 直接造 250 行：不走解析
                for i in range(250):
                    db.add(Document(id=f"pg{i:04d}", kb_id=kb, file_name=f"文件{i:04d}.md", title=("季度报告" if i % 50 == 0 else f"标题{i}"),
                                    status="PROCESSING" if i < 3 else "COMPLETED", chunk_count=1))
                db.commit()
            first = c.get(f"/api/v1/documents/page?kbId={kb}&offset=0&limit=100").json()
            assert first["total"] == 250 and first["processing"] == 3 and len(first["items"]) == 100
            last = c.get(f"/api/v1/documents/page?kbId={kb}&offset=200&limit=100").json()
            assert len(last["items"]) == 50 and not {d["id"] for d in last["items"]} & {d["id"] for d in first["items"]}
            hit = c.get(f"/api/v1/documents/page?kbId={kb}&q=季度").json()
            assert hit["total"] == 5 and all("季度" in d["title"] for d in hit["items"]) and hit["processing"] == 3   # 搜索不影响处理中数
            assert c.get(f"/api/v1/documents/page?kbId={kb}&q=文件0001").json()["total"] == 1
            assert len(c.get(f"/api/v1/documents/page?kbId={kb}&limit=9999").json()["items"]) == 250   # limit 封顶 500，这里只有 250 篇
    finally:
        cfg.data_dir, cfg.upload_dir, cfg.index_root = saved


def test_bm25_scope_mask_matches_membership():
    """内存 BM25 对文档级范围用 numpy 算掩码：结果与逐段落 `cid in scope` 一致。"""
    from prism.config import SparseConfig
    from prism.index import BM25Store

    st = BM25Store(SparseConfig(use_jieba=False))
    ids = [f"d{i % 7}:p{i:04d}:c000" for i in range(300)]
    st.add(ids, [f"段落 {i} 关于年假与报销" for i in range(300)])
    for scope in (ChunkScope(include=frozenset(["d1", "d3"])), ChunkScope(exclude=frozenset(["d0"])), ChunkScope.of_docs([], ["d0"]), ChunkScope()):
        got = {cid for cid, _ in st.search("年假", 300, scope)}
        want = {cid for cid in ids if cid in scope}
        assert got == want, scope
    st.remove({ids[0]})
    assert ids[0] not in {cid for cid, _ in st.search("年假", 300, ChunkScope(include=frozenset(["d0"])))}
