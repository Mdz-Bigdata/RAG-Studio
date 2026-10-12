"""LanceDB 控制台服务：全局库与各知识库的表 / 版本 / 索引，检索实验（向量 / 精确 / 全文 / 混合 / 时间旅行），多模态表。

知识库的向量后端是 LanceDB（vector_store.backend: lancedb）时直接操作它；还是 numpy 等后端时，
可以一键把现有向量 + 正文"镜像"进全局 LanceDB（表 kb_<id>），不重建、不影响线上检索，就能体验 LanceDB 的全部能力。
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from prism.lance import LanceDBVectorStore, connect, list_table_names, open_table, table_info, version_list
from prism.lance.multimodal import LanceMultimodalTable
from prism.multimodal import decode_image_b64

from ..db import Repository, SessionLocal
from .index_service import index_service
from .memory_service import lance_dir
from .retrieval_service import retrieval_service
from .settings_service import settings_service

logger = logging.getLogger(__name__)


def _mirror_name(kb_id: str) -> str:
    return "kb_" + re.sub(r"[^0-9A-Za-z_]", "_", kb_id)


class LanceService:
    # --- 总览 -------------------------------------------------------------------------
    def overview(self) -> dict[str, Any]:
        import lancedb

        db = connect(lance_dir())
        tables = []
        for name in list_table_names(db):
            t = open_table(db, name)
            if t is not None:
                info = table_info(t)
                info["purpose"] = self._purpose(name)
                tables.append(info)
        s = settings_service.get()
        with SessionLocal() as session:
            kbs = [(kb.id, kb.name) for kb in Repository(session).list_kbs()]
        kb_rows = []
        for kb_id, name in kbs:
            row: dict[str, Any] = {"kbId": kb_id, "name": name}
            try:
                idx = index_service.get(kb_id)
                row.update(backend=getattr(idx.vectors, "name", "numpy"), vectors=len(idx.vectors), images=len(idx.images))
                row["mirror"] = open_table(db, _mirror_name(kb_id)) is not None
                row["multimodal"] = LanceMultimodalTable.of_index(idx)._table() is not None
            except Exception as exc:
                row["error"] = str(exc)
            kb_rows.append(row)
        return {"version": lancedb.__version__, "dir": str(lance_dir()), "backend": s.vector_store.backend, "sparseBackend": s.retrieval.sparse_backend,
                "lancedb": s.vector_store.lancedb.model_dump(), "search": {"nprobes": s.vector_store.search.nprobes, "refineFactor": s.vector_store.search.refine_factor, "ef": s.vector_store.search.ef_search},
                "tables": tables, "kbs": kb_rows}

    @staticmethod
    def _purpose(name: str) -> str:
        if name == "memories":
            return "长期记忆"
        if name in ("rec_items", "rec_events"):
            return "个性化推荐"
        if name.startswith("agent_board"):
            return "多智能体黑板"
        if name.startswith("agent_lake"):
            return "Agent 湖"
        if name == "mm" or name.startswith("mm_"):
            return "多模态表"
        if name.startswith("kb_"):
            return "知识库镜像"
        return ""

    def table_versions(self, name: str) -> list[dict[str, Any]]:
        t = open_table(connect(lance_dir()), name)
        if t is None:
            raise KeyError(name)
        return version_list(t)

    # --- 知识库向量（后端或镜像）--------------------------------------------------------------
    def kb_store(self, kb_id: str, *, create: bool = False) -> tuple[LanceDBVectorStore, str]:
        idx = index_service.get(kb_id)
        if isinstance(idx.vectors, LanceDBVectorStore):
            return idx.vectors, "backend"
        s = settings_service.get().vector_store
        store = LanceDBVectorStore(lance_dir(), idx.embedder.dim or 0, table=_mirror_name(kb_id), cfg=s.lancedb)
        store.configure(ef_search=s.search.ef_search, nprobes=s.search.nprobes, refine_factor=s.search.refine_factor)
        if store._table() is None and not create:
            raise LookupError("这个知识库的向量后端不是 LanceDB，也还没有镜像：先点「镜像到 LanceDB」")
        return store, "mirror"

    def mirror_kb(self, kb_id: str) -> dict[str, Any]:
        """把知识库现有的 child 向量 + 正文写进全局 LanceDB 的 kb_<id> 表（不重新向量化）。"""
        idx = index_service.get(kb_id)
        if isinstance(idx.vectors, LanceDBVectorStore):
            return {"kbId": kb_id, "mode": "backend", "rows": len(idx.vectors), "note": "向量后端已经是 LanceDB，无需镜像"}
        t0 = time.perf_counter()
        children = [c for c in idx.store.chunks.values() if c.kind == "child"]
        vecs = idx.vectors.get_vectors([c.chunk_id for c in children])
        children = [c for c in children if c.chunk_id in vecs]
        store, _ = self.kb_store(kb_id, create=True)
        db = connect(lance_dir())
        if store._table() is not None:
            db.drop_table(store.table_name)
            store._tbl, store._count, store._indexed = None, None, None
        if children:
            import numpy as np

            store.add([c.chunk_id for c in children], np.vstack([vecs[c.chunk_id] for c in children]), texts=[c.embed_text() for c in children], doc_ids=[c.doc_id for c in children])
            store.ensure_indexes()
        return {"kbId": kb_id, "mode": "mirror", "rows": len(children), "table": store.table_name, "ms": round((time.perf_counter() - t0) * 1000)}

    def kb_info(self, kb_id: str) -> dict[str, Any]:
        try:
            store, mode = self.kb_store(kb_id)
        except LookupError as exc:
            return {"kbId": kb_id, "mode": "none", "note": str(exc)}
        aux = {}
        if mode == "backend":
            for name, a in index_service.get(kb_id).aux.items():
                aux[name] = {"backend": a.backend, "rows": len(a)}
        return {"kbId": kb_id, "mode": mode, **store.info(), "versionsList": store.versions(20), "aux": aux}

    def kb_search(self, kb_id: str, query: str, *, mode: str = "vector", k: int = 8, version: int | None = None,
                  nprobes: int | None = None, refine_factor: int | None = None, explain: bool = False) -> dict[str, Any]:
        store, where = self.kb_store(kb_id)
        idx = index_service.get(kb_id)
        if nprobes:
            store.nprobes = nprobes
        if refine_factor is not None:
            store.refine_factor = refine_factor or None
        t0 = time.perf_counter()
        qvec = idx.embedder.embed_query(query) if mode != "fts" else None
        t_embed = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        if version:
            hits = store.search_at_version(version, qvec, k)
        elif mode == "fts":
            hits = store.fts_search(query, k)
        elif mode == "hybrid":
            hits = store.hybrid_search(qvec, query, k)
        else:
            hits = store.search(qvec, k, exact=(mode == "exact"))
        t_search = (time.perf_counter() - t1) * 1000
        names = retrieval_service.doc_names(kb_id)
        out = []
        for cid, score in hits:
            c = idx.store.get(cid)
            out.append({"chunkId": cid, "score": round(float(score), 4), "breadcrumb": c.breadcrumb_text if c else "", "text": (c.text[:300] if c else "（该块已不在当前版本）"),
                        "docId": c.doc_id if c else "", "fileName": names.get(c.doc_id) if c else None})
        plan = None
        if explain and qvec is not None and not version:
            try:
                b = store._table().search(qvec).distance_type("cosine").limit(k)
                plan = b.explain_plan(verbose=True)
            except Exception as exc:
                plan = f"（无法生成查询计划：{exc}）"
        return {"kbId": kb_id, "source": where, "mode": mode, "version": version, "hits": out, "timingsMs": {"embed": round(t_embed, 1), "search": round(t_search, 1)},
                "annIndexed": store.indexed, "plan": plan}

    def kb_build_index(self, kb_id: str, index_type: str) -> dict[str, Any]:
        store, mode = self.kb_store(kb_id)
        t0 = time.perf_counter()
        store.build_ann_index(index_type)
        return {"kbId": kb_id, "mode": mode, "indexType": index_type, "ms": round((time.perf_counter() - t0) * 1000), **store.info()}

    def kb_optimize(self, kb_id: str) -> dict[str, Any]:
        store, mode = self.kb_store(kb_id)
        t0 = time.perf_counter()
        store.optimize()
        return {"kbId": kb_id, "mode": mode, "ms": round((time.perf_counter() - t0) * 1000), **store.info()}

    # --- 多模态表（每个知识库一张：data/indexes/<kb>/lancedb 的表 mm）-------------------------------
    def _mm(self, kb_id: str) -> LanceMultimodalTable:
        return LanceMultimodalTable.of_index(index_service.get(kb_id), keep_versions_days=settings_service.get().vector_store.lancedb.keep_versions_days)

    def mm_sync(self, kb_id: str) -> dict[str, Any]:
        """手动同步：先补齐缺的向量（含还没建的正文图文向量），再增量写表。"""
        with index_service.lock(kb_id):
            return self._mm(kb_id).sync_from_index(index_service.get(kb_id), embed="all")

    def mm_info(self, kb_id: str) -> dict[str, Any]:
        # online：是否已接入线上图文检索（打开时保存索引 / 查询前自动增量同步）
        return {**self._mm(kb_id).info(), "online": settings_service.get().multimodal.lance_table}

    def mm_search(self, kb_id: str, *, text: str | None = None, image_b64: str | None = None, target: str = "image", mode: str = "vector", k: int = 8) -> dict[str, Any]:
        idx = index_service.get(kb_id)
        image = decode_image_b64(image_b64) if image_b64 else None
        t0 = time.perf_counter()
        out = self._mm(kb_id).search(embedder=idx.images.embedder, text=text, image=image, target=target, mode=mode, k=k)
        out["ms"] = round((time.perf_counter() - t0) * 1000)
        names = retrieval_service.doc_names(kb_id)
        for h in out["hits"]:
            h["fileName"] = names.get(h["docId"])
        return out

    def mm_thumbnail(self, kb_id: str, row_id: str) -> bytes | None:
        return self._mm(kb_id).thumbnail(row_id)


lance_service = LanceService()
