"""LanceDB 向量后端（vector_store.backend: lancedb）：每个知识库一个 LanceDB 库 <index_dir>/lancedb，表 children。

列：id / doc_id / text（child 的检索文本，供原生全文检索）/ vector（fixed_size_list<float32>）。
- 写：merge_insert 按 id upsert、delete；每次写入都是一个新版本，可回看（时间旅行）；
- 读：行数不到 index_min_rows 时精确扫描（召回 100%）；够了自动建 ANN 索引（默认 IVF_HNSW_SQ），
  查询期的 nprobes / refine_factor / ef 在 vector_store.search 里调；
- allowed_ids（权限 / 元数据过滤）走 SQL 预过滤：先过滤再算距离，不会被不可见的块挤掉名额；
  传进来的是 ChunkScope（文档级范围，千万级知识库用）时直接下推成 doc_id IN / NOT IN，不展开成段落 id；
- IVF 分区数随数据量定（√行数 / 4），查询时至少探查 nprobes_ratio 比例的分区——固定 nprobes 在大库上召回会悄悄变差；
  数据量涨到建索引时的 4 倍就重训索引（分区中心是按当时的数据训练的）；
- fts_tokenizer: jieba 时，写入时用 BM25 同一套 jieba 分词把正文切好存进 tokens 列，全文索引建在 tokens 上（whitespace），
  千万级知识库用它替代内存里的 BM25；
- 原生全文检索（ICU 分词，中文可用、无需下载模型）与混合检索（向量 + 全文 + RRF），
  可以替换关键词路（retrieval.sparse_backend: lancedb_fts），也在 LanceDB 控制台里直接对比。
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ..config import LanceDBConfig
from ..index.scope import ChunkScope, doc_prefix
from ..index.vector_store import widen_until_filled
from .db import connect, has_index, open_table, sql_in, sql_str, table_info, version_list, write_lock

logger = logging.getLogger(__name__)

_IN_LIMIT = 5000        # allowed_ids 超过该数量时改为"多取再在内存过滤，不够就加倍重取"，避免超长 IN 列表
_BATCH = 500


def _vector_column(vecs: np.ndarray):
    import pyarrow as pa

    return pa.FixedSizeListArray.from_arrays(pa.array(vecs.reshape(-1), type=pa.float32()), vecs.shape[1])


class LanceDBVectorStore:
    name = "lancedb"
    stores_text = True          # Index.add_document 会把 child 文本一起写进来（原生全文检索要用）

    def __init__(self, directory: str | Path, dim: int, *, table: str = "children", cfg: LanceDBConfig | None = None) -> None:
        self.directory = Path(directory)
        self.table_name = table
        self.cfg = cfg or LanceDBConfig()
        self._dim = dim
        self._tbl = None
        self._count: int | None = None
        self._indexed: bool | None = None
        self._writes = 0
        self._snap: tuple[int, list[str], np.ndarray] | None = None
        self._lock = threading.RLock()
        # 查询期参数（configure 设置，来自 vector_store.search）
        self.nprobes = 20
        self.nprobes_ratio = 0.3
        self.refine_factor: int | None = 5
        self.ef = 128
        self.tokenizer: Callable[[str], list[str]] | None = None   # fts_tokenizer: jieba 时由 Index 注入（与 BM25 同一套分词）
        self._has_doc_ids: tuple[int, bool] | None = None

    # ----------------------------------------------------------------------------------
    @property
    def db(self):
        return connect(self.directory)

    def _table(self):
        if self._tbl is None:
            self._tbl = open_table(self.db, self.table_name)
            if self._tbl is not None:
                self._dim = self._tbl.schema.field("vector").type.list_size
        return self._tbl

    @property
    def dim(self) -> int:
        self._table()
        return self._dim

    def __len__(self) -> int:
        if self._count is None:
            tbl = self._table()
            self._count = tbl.count_rows() if tbl is not None else 0
        return self._count

    def _snapshot(self) -> tuple[list[str], np.ndarray]:
        """全部 id 与对应的向量矩阵（同一次读取，顺序一致）；按表版本缓存，写入后失效。"""
        tbl = self._table()
        if tbl is None:
            return [], np.zeros((0, self._dim or 0), dtype=np.float32)
        if self._snap is None or self._snap[0] != tbl.version:
            t = tbl.search().select(["id", "vector"]).limit(max(1, len(self))).to_arrow()
            mat = t.column("vector").combine_chunks().flatten().to_numpy().reshape(t.num_rows, -1).astype(np.float32) if t.num_rows else np.zeros((0, self.dim), dtype=np.float32)
            self._snap = (tbl.version, t.column("id").to_pylist(), mat)
        return self._snap[1], self._snap[2]

    @property
    def ids(self) -> list[str]:
        """全部 id（与 NumpyVectorStore.ids 对齐：图片索引按它做增量同步）。"""
        return list(self._snapshot()[0])

    @property
    def matrix(self) -> np.ndarray:
        """与 ids 同序的全部向量（与 NumpyVectorStore.matrix 对齐：图文混合检索要在全部图片上做标准化）。"""
        return self._snapshot()[1]

    def drop(self) -> None:
        """整表删除（换了模型、向量维度变了时重建用）。"""
        with write_lock(self.directory, self.table_name):
            if self._table() is not None:
                self.db.drop_table(self.table_name)
            self._tbl, self._count, self._indexed, self._dim, self._snap = None, None, None, 0, None

    def configure(self, *, ef_search: int = 128, nprobes: int = 20, refine_factor: int | None = 5, nprobes_ratio: float = 0.3, **_: object) -> None:
        self.ef, self.nprobes, self.refine_factor, self.nprobes_ratio = ef_search, nprobes, refine_factor, nprobes_ratio

    # --- ANN 索引的元信息（分区数 / 建索引时的行数）----------------------------------------
    @property
    def _ann_meta_path(self) -> Path:
        return self.directory / f"{self.table_name}.ann.json"

    def _ann_meta(self) -> dict[str, Any]:
        try:
            return json.loads(self._ann_meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _probes(self) -> int:
        """查询探查的分区数：配置的 nprobes 与"分区数 × nprobes_ratio"取大者。老索引没有记分区数时按 Lance 的默认（√行数）估。"""
        parts = self._ann_meta().get("partitions") or max(1, int(math.sqrt(max(1, len(self)))))
        return max(self.nprobes, math.ceil(parts * self.nprobes_ratio))

    @staticmethod
    def partitions_for(rows: int, index_type: str) -> int:
        """IVF 分区数：IVF_HNSW_SQ 分区内还有 HNSW 图，分区可以大一些（√n / 4）；IVF_PQ / IVF_FLAT 用 √n。"""
        root = math.sqrt(max(1, rows))
        return max(1, int(root / 4) if "HNSW" in index_type else int(root))

    @property
    def indexed(self) -> bool:
        """是否已有 ANN 向量索引（没有时精确扫描）。"""
        if self._indexed is None:
            tbl = self._table()
            self._indexed = bool(tbl is not None and has_index(tbl, "vector"))
        return self._indexed

    @property
    def tuned(self) -> bool:
        return self.indexed

    # ----------------------------------------------------------------------------------
    @property
    def jieba_fts(self) -> bool:
        return self.cfg.fts_tokenizer == "jieba"

    @property
    def fts_column(self) -> str:
        return "tokens" if self.jieba_fts else "text"

    def _tokens(self, text: str) -> str:
        if self.tokenizer is None:
            raise RuntimeError("fts_tokenizer: jieba 需要先注入分词器（Index 会自动注入）")
        return " ".join(self.tokenizer(text))

    def add(self, ids: list[str], vectors: np.ndarray, *, texts: list[str] | None = None, doc_ids: list[str] | None = None, fresh: bool = False,
            tokens: list[str] | None = None) -> None:
        """fresh=True：调用方保证这些 id 都是新的（同一文档重新入库前已整篇删除）——直接追加，不做按 id 的 upsert。
        merge_insert 要拿新数据和整张表按 id 做连接，千万行时每传一个文档要好几秒；追加是常数时间。
        tokens：已经切好、空格连接的分词结果（批量导入在多进程里分词后传进来）；不给则按 tokenizer 现切。"""
        if not ids:
            return
        import pyarrow as pa

        vecs = np.asarray(vectors, dtype=np.float32)
        n = len(ids)
        cols = {
            "id": pa.array(ids, type=pa.string()),
            # 文档级过滤（ChunkScope）按 doc_id 下推：调用方没给就从段落 id 的前缀取
            "doc_id": pa.array(doc_ids or [doc_prefix(i) for i in ids], type=pa.string()),
            "text": pa.array(texts or [""] * n, type=pa.string()),
        }
        if self.jieba_fts:
            cols["tokens"] = pa.array(tokens if tokens is not None else [self._tokens(t) for t in (texts or [""] * n)], type=pa.string())
        cols["vector"] = _vector_column(vecs)
        data = pa.table(cols)
        with write_lock(self.directory, self.table_name):
            tbl = self._table()
            if tbl is None:
                self._tbl = self.db.create_table(self.table_name, data)
                self._dim = vecs.shape[1]
            else:
                if vecs.shape[1] != self._dim:
                    raise ValueError(f"向量维度不匹配：{vecs.shape[1]} != {self._dim}（换了向量模型需要重建索引）")
                if fresh:
                    tbl.add(data)
                else:
                    tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(data)
            self._after_write()

    def remove_doc(self, doc_id: str) -> None:
        """整篇删除：按 doc_id 一条语句删掉（不用先列出全部段落 id）。"""
        tbl = self._table()
        if tbl is None:
            return
        with write_lock(self.directory, self.table_name):
            tbl.delete(f"doc_id = {sql_str(doc_id)}")
            self._after_write()

    def remove_docs(self, doc_ids: list[str]) -> None:
        tbl = self._table()
        if tbl is None or not doc_ids:
            return
        with write_lock(self.directory, self.table_name):
            for i in range(0, len(doc_ids), _BATCH):
                tbl.delete(sql_in("doc_id", doc_ids[i:i + _BATCH]))
            self._after_write()

    def remove(self, ids: list[str]) -> None:
        tbl = self._table()
        if tbl is None or not ids:
            return
        with write_lock(self.directory, self.table_name):
            for i in range(0, len(ids), _BATCH):
                tbl.delete(sql_in("id", ids[i:i + _BATCH]))
            self._after_write()

    def _after_write(self) -> None:
        self._count = None
        self._snap = None
        self._writes += 1

    def get_vectors(self, ids: list[str]) -> dict[str, np.ndarray]:
        tbl = self._table()
        if tbl is None or not ids:
            return {}
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(ids), _BATCH):
            part = ids[i:i + _BATCH]
            t = tbl.search().where(sql_in("id", part)).select(["id", "vector"]).limit(len(part)).to_arrow()
            if not t.num_rows:
                continue
            mat = t.column("vector").combine_chunks().flatten().to_numpy().reshape(t.num_rows, -1).astype(np.float32)
            out.update(zip(t.column("id").to_pylist(), mat))
        return out

    # ----------------------------------------------------------------------------------
    def _doc_ids_complete(self) -> bool:
        """表里每行都有 doc_id 才能按文档下推（早期写入的辅助索引表没有填 doc_id）。按表版本缓存。"""
        tbl = self._table()
        if tbl is None:
            return True
        if self._has_doc_ids is None or self._has_doc_ids[0] != tbl.version:
            self._has_doc_ids = (tbl.version, tbl.count_rows("doc_id = '' OR doc_id IS NULL") == 0)
        return self._has_doc_ids[1]

    def _filter(self, allowed_ids) -> tuple[str | None, object | None]:
        """(预过滤 SQL, 后置过滤用的集合)。两者都是 None = 不过滤。"""
        if allowed_ids is None:
            return None, None
        if isinstance(allowed_ids, ChunkScope):
            if allowed_ids.unrestricted:
                return None, None
            # 排除模式意味着至少一半文档可见（ChunkScope.of_docs 取小的那份）：多取一些再按 doc 过滤（O(1)）比每次让
            # LanceDB 解析一条几千项的 NOT IN 更快——百万段落、排除 7500 篇时 32 ms → 13 ms。允许列表才值得下推
            if allowed_ids.include is not None and allowed_ids.pushable and self._doc_ids_complete():
                return allowed_ids.sql("doc_id"), None
            return None, allowed_ids
        if len(allowed_ids) > _IN_LIMIT:
            return None, allowed_ids
        return sql_in("id", allowed_ids), None

    def search(self, query: np.ndarray, top_n: int, allowed_ids: set[str] | None = None, *, exact: bool = False) -> list[tuple[str, float]]:
        """余弦相似度降序（与 numpy 后端同一尺度：归一化向量的内积）。"""
        tbl = self._table()
        if tbl is None or top_n <= 0 or (allowed_ids is not None and not allowed_ids):
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        where, post = self._filter(allowed_ids)

        def fetch(limit: int) -> list[tuple[str, float]]:
            b = tbl.search(q, vector_column_name="vector").distance_type("cosine").select(["id", "_distance"]).limit(limit)
            if where:
                b = b.where(where, prefilter=True)
            if exact:
                b = b.bypass_vector_index()
            elif self.indexed:
                b = b.nprobes(self._probes())
                if self.refine_factor:
                    b = b.refine_factor(self.refine_factor)
                if "HNSW" in (self.cfg.index_type if self.cfg.index_type != "auto" else "IVF_HNSW_SQ"):
                    # HNSW 要求 ef ≥ 实际取回数（refine 时是 limit × refine_factor），否则 Lance 直接报错
                    b = b.ef(max(self.ef, limit * (self.refine_factor or 1)))
            t = b.to_arrow()
            return [(cid, 1.0 - float(d)) for cid, d in zip(t.column("id").to_pylist(), t.column("_distance").to_pylist())]

        return widen_until_filled(fetch, top_n, len(self), post) if post is not None else fetch(top_n)

    def search_exact(self, query: np.ndarray, top_n: int, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        return self.search(query, top_n, allowed_ids, exact=True)

    def fts_search(self, text: str, top_n: int, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        """原生全文检索（BM25）：ICU 分词，中文与编号（SO-8817 / 502）都能命中。"""
        tbl = self._table()
        if tbl is None or not text.strip() or (allowed_ids is not None and not allowed_ids):
            return []
        self.ensure_fts()
        if self.jieba_fts:
            text = self._tokens(text)
            if not text:
                return []
        where, post = self._filter(allowed_ids)

        def fetch(limit: int) -> list[tuple[str, float]]:
            b = tbl.search(text, query_type="fts", fts_columns=self.fts_column).select(["id", "_score"]).limit(limit)
            if where:
                b = b.where(where, prefilter=True)
            t = b.to_arrow()
            return [(cid, float(s)) for cid, s in zip(t.column("id").to_pylist(), t.column("_score").to_pylist())]

        try:
            return widen_until_filled(fetch, top_n, len(self), post) if post is not None else fetch(top_n)
        except Exception as exc:   # 查询里有全文检索语法字符等
            logger.info("LanceDB 全文检索失败（%s），本次关键词路为空", exc)
            return []

    def hybrid_search(self, query: np.ndarray, text: str, top_n: int, allowed_ids: set[str] | None = None, *, reranker: str = "rrf") -> list[tuple[str, float]]:
        """LanceDB 原生混合检索：向量 + 全文，两路用 RRF（或线性加权）融合。"""
        from lancedb.rerankers import LinearCombinationReranker, RRFReranker

        tbl = self._table()
        if tbl is None or (allowed_ids is not None and not allowed_ids):
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)

        def vector_only() -> list[tuple[str, float]]:
            # 全文那一半用不了（空文本 / 查询语法字符）时退回纯向量，分数换成同一尺度的 RRF 名次分
            return [(cid, 1.0 / (60 + rank)) for rank, (cid, _) in enumerate(self.search(q, top_n, allowed_ids), start=1)]

        if not (text or "").strip():
            return vector_only()
        self.ensure_fts()
        if self.jieba_fts:
            text = self._tokens(text)
            if not text:
                return vector_only()
        where, post = self._filter(allowed_ids)
        rr = LinearCombinationReranker(weight=0.7) if reranker == "linear" else RRFReranker()
        b = tbl.search(query_type="hybrid", fts_columns=self.fts_column).vector(q).text(text).distance_type("cosine").rerank(rr).select(["id"]).limit(top_n * 4 if post is not None else top_n)
        if where:
            b = b.where(where, prefilter=True)
        try:
            t = b.to_arrow()
        except Exception as exc:
            logger.info("LanceDB 混合检索的全文部分失败（%s），本次只用向量", exc)
            return vector_only()
        hits = list(zip(t.column("id").to_pylist(), (float(x) for x in t.column("_relevance_score").to_pylist())))
        if post is not None:
            hits = [h for h in hits if h[0] in post]
        return hits[:top_n]

    # ----------------------------------------------------------------------------------
    def ensure_fts(self) -> None:
        tbl = self._table()
        col = self.fts_column
        if tbl is None or has_index(tbl, col):
            return
        with write_lock(self.directory, self.table_name):
            if has_index(tbl, col):
                return
            # jieba：正文写入时已经切好词、空格分隔，索引按空白切即可
            base = "whitespace" if self.jieba_fts else self.cfg.fts_tokenizer
            kw: dict[str, Any] = {"base_tokenizer": base, "stem": False, "remove_stop_words": False, "lower_case": True, "replace": True}
            if self.jieba_fts:
                kw.update(with_position=False, max_token_length=None)
            if self.cfg.fts_tokenizer == "ngram":
                kw.update(ngram_min_length=2, ngram_max_length=2)
            tbl.create_fts_index(col, **kw)

    def ensure_indexes(self) -> dict[str, Any]:
        """按数据量补齐索引：全文索引（有正文时）、id 的 BTREE（IN 预过滤）、够量时的 ANN 向量索引。"""
        tbl = self._table()
        done: dict[str, Any] = {}
        if tbl is None:
            return done
        rows = len(self)
        if rows:
            self.ensure_fts()
        with write_lock(self.directory, self.table_name):
            if rows >= 2000:
                for col in ("id", "doc_id"):          # id：按 id 取向量 / 删除；doc_id：文档级预过滤与整篇删除
                    if not has_index(tbl, col):
                        tbl.create_scalar_index(col, index_type="BTREE")
                        done[col] = "BTREE"
            kind = self.cfg.index_type
            want_ann = kind not in ("none",) and (kind != "auto" or rows >= self.cfg.index_min_rows)
            if want_ann and rows >= 256:
                index_type = "IVF_HNSW_SQ" if kind == "auto" else kind
                built = self._ann_meta().get("rows") or 0
                if not has_index(tbl, "vector"):
                    done["vector"] = self._build_ann(index_type, rows)
                elif built and rows >= 4 * built:
                    # 分区中心是按建索引时的数据训练的：数据涨到 4 倍，新数据只能硬塞进老分区，召回和速度都变差
                    logger.info("%s 行数 %d 已是建索引时（%d）的 4 倍以上，重训 ANN 索引", self.table_name, rows, built)
                    done["vector"] = self._build_ann(index_type, rows)
        return done

    def _build_ann(self, index_type: str, rows: int) -> str:
        tbl = self._table()
        parts = self.partitions_for(rows, index_type)
        kw: dict[str, Any] = {"num_partitions": parts}
        if index_type == "IVF_PQ":
            kw["num_sub_vectors"] = max(1, self.dim // 16)
        tbl.create_index(metric="cosine", vector_column_name="vector", index_type=index_type, replace=True, **kw)
        self._ann_meta_path.write_text(json.dumps({"index_type": index_type, "partitions": parts, "rows": rows}), encoding="utf-8")
        self._indexed = True
        return index_type

    def build_ann_index(self, index_type: str = "IVF_HNSW_SQ") -> str:
        """控制台手动建 / 重建 ANN 索引（不受 index_min_rows 限制，但至少要 256 行才能训练 IVF）。"""
        tbl = self._table()
        if tbl is None or len(self) < 256:
            raise ValueError("至少需要 256 行向量才能训练 IVF 索引；数据量小时精确扫描本来就最快、召回 100%")
        with write_lock(self.directory, self.table_name):
            return self._build_ann(index_type, len(self))

    def optimize(self, *, drop_history: bool = False) -> None:
        """合并小文件 + 把新写入并入索引 + 清理过期旧版本。
        drop_history：批量导入刚建完的表，旧版本只是一批一个的碎片（合并前的数据），留着会让磁盘占用翻倍——直接清掉。"""
        tbl = self._table()
        if tbl is None:
            return
        with write_lock(self.directory, self.table_name):
            if drop_history:
                tbl.optimize(cleanup_older_than=timedelta(0), delete_unverified=True)
            else:
                tbl.optimize(cleanup_older_than=timedelta(days=self.cfg.keep_versions_days))
        self._writes = 0

    def save(self, directory: Path | None = None, prefix: str = "") -> None:
        """LanceDB 写入即持久化；这里只做索引维护。"""
        if self._table() is None:
            return
        try:
            self.ensure_indexes()
            if self._writes >= self.cfg.optimize_every:
                self.optimize()
        except Exception as exc:   # 索引维护失败不影响数据本身
            logger.warning("LanceDB 索引维护失败：%s", exc)

    # ----------------------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        tbl = self._table()
        if tbl is None:
            return {"name": self.table_name, "rows": 0, "exists": False, "path": str(self.directory)}
        return {**table_info(tbl), "exists": True, "path": str(self.directory), "annIndexed": self.indexed, "indexMinRows": self.cfg.index_min_rows, "writesSinceOptimize": self._writes}

    def versions(self, limit: int = 50) -> list[dict[str, Any]]:
        tbl = self._table()
        return version_list(tbl, limit) if tbl is not None else []

    def search_at_version(self, version: int, query: np.ndarray, top_n: int) -> list[tuple[str, float]]:
        """时间旅行：在历史版本上检索（只读，不影响当前表）。"""
        tbl = self.db.open_table(self.table_name)
        tbl.checkout(int(version))
        t = tbl.search(np.asarray(query, dtype=np.float32).reshape(-1)).distance_type("cosine").select(["id", "_distance"]).limit(top_n).to_arrow()
        return [(cid, 1.0 - float(d)) for cid, d in zip(t.column("id").to_pylist(), t.column("_distance").to_pylist())]

    def coarse_info(self) -> dict[str, Any]:
        """向量检索调优卡片用：LanceDB 的存储占用（含索引）。"""
        info = self.info()
        return {"dim": self.dim, "bytes": info.get("bytes") or 0, "full_bytes": len(self) * self.dim * 4}
