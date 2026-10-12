"""LanceDB 多模态表 mm：知识库的图片与正文进同一张表——CLIP 向量、缩略图字节、图注 / 描述 / OCR / 正文、所在文档，一处管理。

每个知识库一张，和它的其他 LanceDB 表放在一起（data/indexes/<kb>/lancedb 里的表 mm），随知识库删除、整库重建一起走。

- 线上图文检索（multimodal.lance_table 打开时）多出「LanceDB 多模态表」一路，与图文向量 / 描述向量 / 描述 BM25 等各路一起 RRF 融合：
  · 文搜图 / 问答附图：image 行的全文检索（图注 / 描述 / OCR / 上下文，ICU 分词）。不再叠 CLIP 向量——图文向量已经是单独一路，
    再算一次只会放大 CLIP 对细节（区域 / 季度）不敏感的噪声。120 张同风格图表压测（hit@1，完整图注 / 口语改写 / 关键词）：
    不接入 0.983 / 0.742 / 0.800；接入全文 1.000 / 0.767 / 0.825；接入"CLIP + 全文"混合反而降到 0.958 / 0.758 / 0.767；
  · 图搜文：text 行上的原生混合检索（查询图的 CLIP 向量 + 图中文字 / 图片转文字的全文，表内 RRF）。
- 增量同步：每行带内容签名 sig。保存索引时（以及查询前发现知识库有变化时）只重写变了的行、删掉已不存在的行，
  缩略图只为新图 / 变了的图生成。正文行要 CLIP 文本向量：手动同步一次补齐；保存索引时只给新切片补（单次有上限）；
  查询前的兜底同步从不现算向量。开关打开时，保存索引会为所有知识库（包括没有图片的）建好并维护正文行，
  整库重建（新目录从零开始）后表也随之重建。
  每次有变化的同步是一个新版本，图片集的变化可以回看。
- LanceDB 控制台的多模态检索实验（以文搜图 / 以图搜图 / 以图搜文，向量 / 全文 / 混合）也用这张表。
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .db import connect, has_index, open_table, sql_in, sql_str, table_info, write_lock

logger = logging.getLogger(__name__)

TABLE = "mm"
_COLS = ["id", "modality", "doc_id", "doc_title", "ref_id", "title", "text", "page"]
_IN_LIMIT = 5000                        # 候选集合超过这么多改为后过滤（IN 列表太长反而慢）
_synced: dict[str, int] = {}            # 表路径 → 上次同步完成时知识库内容的摘要（进程内，查询前据此判断要不要同步）
_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def _sync_lock(key: str) -> threading.Lock:
    with _guard:
        return _locks.setdefault(key, threading.Lock())


def _schema(dim: int):
    import pyarrow as pa

    return pa.schema([pa.field("id", pa.string()), pa.field("modality", pa.string()), pa.field("doc_id", pa.string()), pa.field("doc_title", pa.string()),
                      pa.field("ref_id", pa.string()), pa.field("title", pa.string()), pa.field("text", pa.string()), pa.field("page", pa.int32()),
                      pa.field("sig", pa.string()), pa.field("image", pa.binary()), pa.field("clip", pa.list_(pa.float32(), dim))])


def _thumbnail(image: Any, side: int = 256) -> bytes | None:
    try:
        img = image.convert("RGB").copy()
        img.thumbnail((side, side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()
    except Exception:
        return None


def _embedder_key(imgs) -> tuple[str, bool | None]:
    """图文向量模型的身份与是否跨模态。不触发加载：模型没加载时取索引里记录的模型；
    两边都没有（纯文本知识库、本进程还没用过图文模型）时跨模态记 None = 不知道，这时不动正文行。"""
    s = imgs.stats()
    cross = s.get("cross_modal")
    return f"{s.get('image_embedder')}:{s.get('image_model')}", None if cross is None else bool(cross)


def digest(index) -> int:
    """知识库里与多模态表有关的内容的摘要：只用来判断"自上次同步后有没有变"，每次查询前算一遍（字符串的 hash 有缓存，很便宜）。"""
    imgs, store = index.images, index.store
    key, cross = _embedder_key(imgs)
    images = tuple((r.image_id, r.doc_id, r.doc_title, r.caption, r.description, r.ocr_text, r.context_text, tuple(r.breadcrumbs), r.page, r.order, r.sha1, r.file)
                   for r in imgs.records.values())
    texts: tuple = ()
    if cross:
        chunks = store.chunks
        texts = tuple((cid, chunks[cid].text, chunks[cid].doc_id, chunks[cid].page_start, tuple(chunks[cid].breadcrumbs)) for cid in store.child_ids())
        texts += tuple((d, m.title) for d, m in store.docs.items())
    return hash((key, cross, len(imgs.visual), len(imgs.chunk_clip), images, texts))


def _sig(emb_key: str, row: dict[str, Any], *extra: Any) -> str:
    h = hashlib.sha1()
    for part in (emb_key, *(row[k] for k in ("modality", "doc_id", "doc_title", "ref_id", "title", "text", "page")), *extra):
        h.update(str(part).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:20]


def _wanted(index, emb_key: str, cross: bool | None) -> dict[str, dict[str, Any]]:
    """这张表"应该有"的行（还不含向量 / 缩略图），每行带内容签名 sig。"""
    imgs, store = index.images, index.store
    want: dict[str, dict[str, Any]] = {}
    for r in imgs.records.values():
        row = {"id": f"img:{r.image_id}", "modality": "image", "doc_id": r.doc_id, "doc_title": r.doc_title, "ref_id": r.image_id,
               "title": r.caption or r.label(), "text": r.search_text() or r.caption or r.doc_title, "page": r.page}
        row["sig"] = _sig(emb_key, row, r.sha1, r.file)
        want[row["id"]] = row
    if cross:
        titles = {d: (m.title or d) for d, m in store.docs.items()}
        for cid in store.child_ids():
            c = store.chunks[cid]
            row = {"id": f"txt:{cid}", "modality": "text", "doc_id": c.doc_id, "doc_title": titles.get(c.doc_id, ""), "ref_id": cid,
                   "title": c.breadcrumb_text or titles.get(c.doc_id, ""), "text": c.text[:2000], "page": c.page_start}
            row["sig"] = _sig(emb_key, row)
            want[row["id"]] = row
    return want


class LanceMultimodalTable:
    def __init__(self, directory: str | Path, *, name: str = TABLE, keep_versions_days: int = 7) -> None:
        self.directory = Path(directory)
        self.name = name
        self.keep_versions_days = keep_versions_days
        self._key = f"{self.directory.resolve()}::{name}"

    @classmethod
    def of_index(cls, index, *, keep_versions_days: int = 7) -> "LanceMultimodalTable":
        """知识库自己的那张（和 children / aux_* / img_* 在同一个 LanceDB 库里）。"""
        return cls(Path(index.directory) / "lancedb", keep_versions_days=keep_versions_days)

    def _table(self):
        return open_table(connect(self.directory), self.name)

    # --- 同步 -----------------------------------------------------------------------------
    def ensure_synced(self, index, *, wait: bool = True) -> str:
        """查询前调用：知识库自上次同步后有变化才（增量）同步。
        返回 fresh（无需同步）/ synced（刚同步完）/ busy（别的线程正在同步且 wait=False，这次先用现有的表）。"""
        if _synced.get(self._key) == digest(index) and self._table() is not None:
            return "fresh"
        lock = _sync_lock(self._key)
        if not lock.acquire(blocking=wait):
            return "busy"
        try:
            if _synced.get(self._key) == digest(index) and self._table() is not None:
                return "fresh"
            self._sync(index, embed="none")
            return "synced"
        except (RuntimeError, KeyError) as exc:   # 正赶上入库线程在改这个知识库（字典在变）：这次先用现有的表，保存索引时会再同步
            logger.info("多模态表同步让路给正在进行的入库：%s", exc)
            return "busy"
        finally:
            lock.release()

    def sync_from_index(self, index, *, embed: str = "all") -> dict[str, Any]:
        """把知识库的图片（CLIP 向量 + 缩略图 + 文字）与正文块（CLIP 文本向量）增量写进表。
        embed：all = 先补齐缺的向量（换过模型的图片向量、全部还没建的正文图文向量）——控制台手动同步用；
        new = 只给还没有图文向量的新切片补建，单次不超过 multimodal.chunk_sync_per_query——保存索引时用；
        none = 不现算向量，只同步已有向量的行——查询前的兜底同步用（不让一次查询卡在编码上）。"""
        with _sync_lock(self._key):
            return self._sync(index, embed=embed)

    def _sync(self, index, *, embed: str) -> dict[str, Any]:
        import pyarrow as pa

        t0 = time.perf_counter()
        imgs = index.images
        if embed == "all":
            _ = imgs.embedder                               # 手动同步：加载图文模型，没有图片的知识库也能建正文行（供以图搜文）
            imgs.ensure_fresh()
        emb_key, cross = _embedder_key(imgs)
        if cross is None and embed == "new":
            # 没有图片的知识库 / 整库重建的新目录：索引里没记录图文模型。加载它（全局共享、启动时已预热，通常只是取现成的）
            # 确认能不能做文本，正文行才建得起来、随知识库变化维护
            _ = imgs.embedder
            emb_key, cross = _embedder_key(imgs)
        if cross and embed != "none":
            children = [index.store.chunks[cid] for cid in index.store.child_ids()]
            if embed == "all":
                imgs.sync_chunk_vectors(children)
            else:
                have = set(imgs.chunk_clip.ids)
                if any(c.chunk_id not in have for c in children):
                    imgs.sync_chunk_vectors(children, limit=index.settings.multimodal.chunk_sync_per_query)
        want = _wanted(index, emb_key, cross)

        def managed(row_id: str) -> bool:                      # 跨模态未知时正文行原样保留
            return cross is not None or not row_id.startswith("txt:")

        db = connect(self.directory)
        with write_lock(self.directory, self.name):
            tbl = open_table(db, self.name)
            existing = self._signatures(tbl)                    # None：表不存在 / 结构是旧的 → 整表重写
            changed = [k for k, row in want.items() if existing is None or existing.get(k) != row["sig"]]
            rows = self._materialize(index, [want[k] for k in changed])
            dim = len(rows[0]["clip"]) if rows else None
            if existing is not None and dim is not None and tbl.schema.field("clip").type.list_size != dim:
                existing = None                                 # 换了向量维度：整表重写
                rows = self._materialize(index, list(want.values()))
            if existing is None:
                stale: list[str] = []
                if rows:
                    tbl = db.create_table(self.name, pa.Table.from_pylist(rows, schema=_schema(len(rows[0]["clip"]))), mode="overwrite")
                elif tbl is not None:
                    db.drop_table(self.name)
                    tbl = None
            else:
                ready = {r["id"] for r in rows}
                stale = [k for k in existing if managed(k) and (k not in want or (existing[k] != want[k]["sig"] and k not in ready))]
                if rows:
                    tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(pa.Table.from_pylist(rows, schema=_schema(dim)))
                for i in range(0, len(stale), 500):
                    tbl.delete(sql_in("id", stale[i:i + 500]))
            if tbl is not None and (rows or stale):
                self._index_text(tbl)
                if existing is not None:
                    try:                                          # 合并小文件 + 新行并入全文索引 + 清理过期旧版本
                        tbl.optimize(cleanup_older_than=timedelta(days=self.keep_versions_days))
                    except Exception as exc:
                        logger.info("多模态表 optimize 失败（不影响检索）：%s", exc)
        _synced[self._key] = digest(index)
        n_img = sum(1 for r in want.values() if r["modality"] == "image")
        out = {"table": self.name, "path": str(self.directory), "images": n_img, "texts": len(want) - n_img, "written": len(rows), "removed": len(stale),
               "crossModal": cross, "embedder": emb_key, "version": tbl.version if tbl is not None else None, "ms": round((time.perf_counter() - t0) * 1000)}
        if rows or stale:
            logger.info("LanceDB 多模态表 %s 同步：写入 %d 行、删除 %d 行（%d ms）", self.directory, len(rows), len(stale), out["ms"])
        return out

    @staticmethod
    def _signatures(tbl) -> dict[str, str] | None:
        if tbl is None or "sig" not in tbl.schema.names:
            return None
        t = tbl.search().select(["id", "sig"]).limit(max(1, tbl.count_rows())).to_arrow()
        return dict(zip(t.column("id").to_pylist(), t.column("sig").to_pylist()))

    @staticmethod
    def _materialize(index, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """补上向量与缩略图。没有向量的行（图片向量还没建 / 正文的图文向量还没补）先跳过，下次同步再写。"""
        imgs = index.images
        img_ids = [r["ref_id"] for r in rows if r["modality"] == "image"]
        txt_ids = [r["ref_id"] for r in rows if r["modality"] == "text"]
        vecs = {"image": imgs.visual.get_vectors(img_ids) if img_ids else {}, "text": imgs.chunk_clip.get_vectors(txt_ids) if txt_ids else {}}
        out: list[dict[str, Any]] = []
        for row in rows:
            vec = vecs[row["modality"]].get(row["ref_id"])
            if vec is None:
                continue
            thumb = None
            if row["modality"] == "image":
                rec = imgs.records.get(row["ref_id"])
                try:
                    thumb = _thumbnail(imgs.load_image(rec)) if rec is not None else None
                except Exception:
                    thumb = None
            out.append({**row, "image": thumb, "clip": np.asarray(vec, dtype=np.float32)})
        return out

    @staticmethod
    def _index_text(tbl) -> None:
        if not has_index(tbl, "text"):
            from lancedb.index import FTS

            tbl.create_index("text", config=FTS(base_tokenizer="icu", stem=False, remove_stop_words=False), replace=True)

    # --- 线上检索（多模态检索器的「LanceDB 多模态表」路）-----------------------------------------
    def search_images(self, *, vector: Any = None, text: str = "", n: int = 30, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """image 行。向量 + 文字 → 原生混合检索（RRF）；只有文字 → 全文；只有向量 → 余弦。返回 [(image_id, 分数)]。"""
        return self._route("image", vector, text, n, allowed)

    def search_texts(self, *, vector: Any = None, text: str = "", n: int = 30, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """图搜文：text 行（正文 child 块）。返回 [(chunk_id, 分数)]。"""
        return self._route("text", vector, text, n, allowed)

    def _route(self, modality: str, vector: Any, text: str, n: int, allowed: set[str] | None) -> list[tuple[str, float]]:
        from lancedb.rerankers import RRFReranker

        tbl = self._table()
        if tbl is None or n <= 0 or (allowed is not None and not allowed):
            return []
        text = (text or "").strip()
        post = allowed is not None and len(allowed) > _IN_LIMIT
        where = f"modality = {sql_str(modality)}" + (f" AND {sql_in('ref_id', sorted(allowed))}" if allowed is not None and not post else "")
        limit = n * 4 if post else n
        vec = None if vector is None else np.asarray(vector, dtype=np.float32).reshape(-1)
        hits: list[tuple[str, float]] | None = None
        if vec is not None and text:
            try:
                t = (tbl.search(query_type="hybrid", vector_column_name="clip", fts_columns="text").vector(vec).text(text).distance_type("cosine")
                     .rerank(RRFReranker()).where(where, prefilter=True).select(["ref_id"]).limit(limit).to_arrow())
                hits = list(zip(t.column("ref_id").to_pylist(), (float(x) for x in t.column("_relevance_score").to_pylist())))
            except Exception as exc:                         # 全文那一半用不了（查询语法字符 / 没有全文索引）：退回纯向量，分数换成同尺度的名次分
                logger.info("多模态表混合检索的全文部分失败（%s），本次只用向量", exc)
                t = tbl.search(vec, vector_column_name="clip").distance_type("cosine").where(where, prefilter=True).select(["ref_id", "_distance"]).limit(limit).to_arrow()
                hits = [(i, 1.0 / (60 + rank)) for rank, i in enumerate(t.column("ref_id").to_pylist(), start=1)]
        elif vec is not None:
            t = tbl.search(vec, vector_column_name="clip").distance_type("cosine").where(where, prefilter=True).select(["ref_id", "_distance"]).limit(limit).to_arrow()
            hits = [(i, 1.0 - float(d)) for i, d in zip(t.column("ref_id").to_pylist(), t.column("_distance").to_pylist())]
        elif text:
            try:
                t = tbl.search(text, query_type="fts", fts_columns="text").where(where, prefilter=True).select(["ref_id", "_score"]).limit(limit).to_arrow()
                hits = list(zip(t.column("ref_id").to_pylist(), (float(x) for x in t.column("_score").to_pylist())))
            except Exception as exc:
                logger.info("多模态表全文检索失败：%s", exc)
        hits = hits or []
        if post:
            hits = [h for h in hits if h[0] in allowed]
        return hits[:n]

    # --- 控制台 ---------------------------------------------------------------------------
    def search(self, *, embedder, text: str | None = None, image: Any = None, target: str = "image", mode: str = "vector", k: int = 8) -> dict[str, Any]:
        """target：image（找图）/ text（找正文）/ all；mode：vector / fts / hybrid。有 image 时按图检索（以图搜图 / 以图搜文）。"""
        tbl = self._table()
        if tbl is None:
            return {"hits": [], "note": "还没有同步：先把知识库的图片 / 正文同步到 LanceDB"}
        where = None if target == "all" else f"modality = {sql_str(target)}"
        note = ""
        if image is not None:
            q = embedder.embed_images([image])[0]
            b = tbl.search(np.asarray(q, dtype=np.float32), vector_column_name="clip").distance_type("cosine")
            how = "以图搜" + ("图" if target == "image" else "文" if target == "text" else "全部")
        elif mode == "fts" or not getattr(embedder, "cross_modal", False):
            if mode != "fts":
                note = "图文向量模型不支持文本查询（离线感知特征），改用全文检索"
            b = tbl.search(text or "", query_type="fts", fts_columns="text")
            how = "全文检索"
        elif mode == "hybrid":
            q = embedder.embed_texts([text or ""])[0]
            b = tbl.search(query_type="hybrid", vector_column_name="clip", fts_columns="text").vector(np.asarray(q, dtype=np.float32)).text(text or "").distance_type("cosine")
            how = "混合检索（CLIP 向量 + 全文，RRF）"
        else:
            q = embedder.embed_texts([text or ""])[0]
            b = tbl.search(np.asarray(q, dtype=np.float32), vector_column_name="clip").distance_type("cosine")
            how = "以文搜" + ("图" if target == "image" else "文" if target == "text" else "全部")
        if where:
            b = b.where(where, prefilter=True)
        rows = b.select(_COLS + ["image"]).limit(k).to_list()
        hits = []
        for r in rows:
            score = (1.0 - float(r["_distance"])) if "_distance" in r else float(r.get("_relevance_score", r.get("_score", 0.0)))
            hits.append({"id": r["id"], "modality": r["modality"], "docId": r["doc_id"], "docTitle": r["doc_title"], "refId": r["ref_id"], "title": r["title"],
                         "text": (r["text"] or "")[:240], "page": r["page"], "hasImage": bool(r.get("image")), "score": round(score, 4)})
        return {"hits": hits, "how": how, "note": note, "version": tbl.version}

    def thumbnail(self, row_id: str) -> bytes | None:
        tbl = self._table()
        if tbl is None:
            return None
        rows = tbl.search().where(f"id = {sql_str(row_id)}").select(["image"]).limit(1).to_list()
        return rows[0]["image"] if rows and rows[0]["image"] else None

    def info(self) -> dict[str, Any]:
        tbl = self._table()
        if tbl is None:
            return {"name": self.name, "path": str(self.directory), "rows": 0, "exists": False}
        counts = {"image": tbl.count_rows("modality = 'image'"), "text": tbl.count_rows("modality = 'text'")}
        return {**table_info(tbl), "path": str(self.directory), "exists": True, "byModality": counts}


class MultimodalLanceRoute:
    """多模态检索器里的「LanceDB 多模态表」路：查询前按需增量同步，再在表上做原生混合检索。
    这一路出错只记一条说明、返回空排名，不影响其他各路；返回的编号都过滤到当前索引里还存在的图片 / 正文块。"""

    def __init__(self, index, table: LanceMultimodalTable | None = None) -> None:
        self.index = index
        self.table = table or LanceMultimodalTable.of_index(index)

    def _prepare(self, notes: list[str]) -> None:
        if self.table.ensure_synced(self.index, wait=False) == "busy":
            notes.append("LanceDB 多模态表正在同步，这一路暂用同步前的数据")

    def text_to_image(self, query: str, n: int, allowed: set[str] | None, notes: list[str]) -> list[tuple[str, float]]:
        """图片行的全文检索（CLIP 向量已是检索器里单独的一路，这里不重复计票）。"""
        try:
            self._prepare(notes)
            hits = self.table.search_images(text=query, n=n, allowed=allowed)
        except Exception as exc:
            logger.warning("LanceDB 多模态表路（文搜图）出错：%s", exc)
            notes.append(f"LanceDB 多模态表路出错，本次跳过：{exc}")
            return []
        records = self.index.images.records
        return [(i, s) for i, s in hits if i in records]

    def image_to_text(self, vector: Any, text: str, n: int, allowed: set[str] | None, notes: list[str]) -> list[tuple[str, float]]:
        try:
            self._prepare(notes)
            hits = self.table.search_texts(vector=vector, text=text, n=n, allowed=allowed)
        except Exception as exc:
            logger.warning("LanceDB 多模态表路（图搜文）出错：%s", exc)
            notes.append(f"LanceDB 多模态表路出错，本次跳过：{exc}")
            return []
        chunks = self.index.store.chunks
        return [(i, s) for i, s in hits if i in chunks]
