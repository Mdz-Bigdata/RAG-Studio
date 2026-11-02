"""Milvus 向量后端（vector_store.backend: milvus）：亿级段落时把向量放到分布式的 Milvus 集群上。

- milvus_uri 为本地文件路径（默认 <index_dir>/milvus.db）：嵌入式 Milvus Lite，不用起服务，开发 / 测试 / 千万级以内可用；
  为 http(s):// 地址：连 Milvus Standalone / 集群 / Zilliz Cloud，每个知识库一个 collection（<collection_prefix>_<知识库目录名>）；
- 列：id（段落 id，主键）/ doc_id / vector；索引 AUTOINDEX（集群上即 HNSW / DiskANN 由服务端选），余弦度量；
- 文档级过滤（ChunkScope）下推成 `doc_id in [...]` / `doc_id not in [...]` 表达式；段落 id 集合（老路径）下推成 `id in [...]`，
  太长时多取再过滤；整篇删除 `doc_id == "..."` 一条语句；
- 正文不存在 Milvus 里（关键词路仍用 BM25 / LanceDB 全文）；亿级的全文检索建议接 Elasticsearch，本后端不覆盖。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .scope import ChunkScope
from .vector_store import IN_LIMIT, widen_until_filled

logger = logging.getLogger(__name__)

_BATCH = 2000
_EXPR_LIMIT = 20000       # 表达式里最多放多少个值


def _quote(v: str) -> str:
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _in_expr(column: str, values, negate: bool = False) -> str:
    return f"{column} {'not in' if negate else 'in'} [{', '.join(_quote(v) for v in sorted(values))}]"


def milvus_available() -> bool:
    try:
        import pymilvus  # noqa: F401
    except ImportError:
        return False
    return True


class MilvusVectorStore:
    name = "milvus"
    stores_text = False

    def __init__(self, uri: str | Path, dim: int, *, collection: str = "children", token: str | None = None, db_name: str | None = None) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("需要 pymilvus：pip install 'pymilvus[milvus_lite]'（本地文件）或 pip install pymilvus（连集群）") from exc
        self.uri = str(uri)
        self.local = not self.uri.startswith(("http://", "https://", "tcp://", "unix:"))
        if self.local:
            Path(self.uri).parent.mkdir(parents=True, exist_ok=True)
        kw: dict[str, Any] = {"uri": self.uri}
        if token:
            kw["token"] = token
        if db_name and not self.local:
            kw["db_name"] = db_name
        self.client = MilvusClient(**kw)
        self.collection = collection
        self._dim = dim
        self._count: int | None = None
        self._ensure()

    # --- 建表 -----------------------------------------------------------------------------
    def _exists(self) -> bool:
        return bool(self.client.has_collection(self.collection))

    def _ensure(self) -> None:
        if self._exists():
            # 新进程里打开的 collection 处于 released 状态：search / get / query 之前要先 load（幂等；Lite 与集群都一样）
            try:
                self.client.load_collection(self.collection)
            except Exception as exc:
                logger.warning("milvus load_collection(%s) 失败：%s", self.collection, exc)
            return
        if not self._dim:
            return
        from pymilvus import DataType

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)
        ip = self.client.prepare_index_params()
        ip.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(self.collection, schema=schema, index_params=ip)

    @property
    def dim(self) -> int:
        if not self._dim and self._exists():
            for f in self.client.describe_collection(self.collection).get("fields", []):
                if f.get("name") == "vector":
                    self._dim = int((f.get("params") or {}).get("dim") or 0)
        return self._dim

    def configure(self, **_: object) -> None:
        return None

    @property
    def tuned(self) -> bool:
        return True

    def __len__(self) -> int:
        if self._count is None:
            if not self._exists():
                self._count = 0
            else:
                rows = self.client.query(self.collection, filter="", output_fields=["count(*)"])
                self._count = int(rows[0]["count(*)"]) if rows else 0
        return self._count

    def drop(self) -> None:
        if self._exists():
            self.client.drop_collection(self.collection)
        self._count = None

    # --- 写 -------------------------------------------------------------------------------
    def add(self, ids: list[str], vectors: np.ndarray, *, texts: list[str] | None = None, doc_ids: list[str] | None = None, fresh: bool = False, **_: object) -> None:
        if not ids:
            return
        vecs = np.asarray(vectors, dtype=np.float32)
        if not self._dim:
            self._dim = int(vecs.shape[1])
            self._ensure()
        elif vecs.shape[1] != self.dim:
            raise ValueError(f"向量维度不匹配：{vecs.shape[1]} != {self.dim}（换了向量模型需要重建索引）")
        docs = doc_ids or [i.split(":", 1)[0] for i in ids]
        for k in range(0, len(ids), _BATCH):
            rows = [{"id": ids[j], "doc_id": docs[j], "vector": vecs[j].tolist()} for j in range(k, min(k + _BATCH, len(ids)))]
            if fresh:
                self.client.insert(self.collection, rows)
            else:
                self.client.upsert(self.collection, rows)
        self._count = None

    def remove(self, ids: list[str]) -> None:
        if not ids or not self._exists():
            return
        for k in range(0, len(ids), _BATCH):
            self.client.delete(self.collection, ids=ids[k:k + _BATCH])
        self._count = None

    def remove_doc(self, doc_id: str) -> None:
        if self._exists():
            self.client.delete(self.collection, filter=f"doc_id == {_quote(doc_id)}")
            self._count = None

    def remove_docs(self, doc_ids: list[str]) -> None:
        if not doc_ids or not self._exists():
            return
        for k in range(0, len(doc_ids), _BATCH):
            self.client.delete(self.collection, filter=_in_expr("doc_id", doc_ids[k:k + _BATCH]))
        self._count = None

    # --- 读 -------------------------------------------------------------------------------
    def get_vectors(self, ids: list[str]) -> dict[str, np.ndarray]:
        if not ids or not self._exists():
            return {}
        out: dict[str, np.ndarray] = {}
        for k in range(0, len(ids), _BATCH):
            for row in self.client.get(self.collection, ids=ids[k:k + _BATCH], output_fields=["vector"]):
                out[row["id"]] = np.asarray(row["vector"], dtype=np.float32)
        return out

    @property
    def ids(self) -> list[str]:
        if not self._exists():
            return []
        it = self.client.query_iterator(self.collection, filter="", output_fields=["id"], batch_size=8192)
        out: list[str] = []
        while True:
            rows = it.next()
            if not rows:
                it.close()
                break
            out.extend(r["id"] for r in rows)
        return out

    def _filter(self, allowed_ids) -> tuple[str | None, Any]:
        """(下推的过滤表达式, 后置过滤用的集合)。两者都是 None = 不过滤。"""
        if allowed_ids is None:
            return None, None
        if isinstance(allowed_ids, ChunkScope):
            if allowed_ids.unrestricted:
                return None, None
            if allowed_ids.include is not None:
                return (_in_expr("doc_id", allowed_ids.include) if len(allowed_ids.include) <= _EXPR_LIMIT else None), (None if len(allowed_ids.include) <= _EXPR_LIMIT else allowed_ids)
            if len(allowed_ids.exclude) <= _EXPR_LIMIT:
                return _in_expr("doc_id", allowed_ids.exclude, negate=True), None
            return None, allowed_ids
        if len(allowed_ids) <= IN_LIMIT:
            return _in_expr("id", allowed_ids), None
        return None, allowed_ids

    def search(self, query: np.ndarray, top_n: int, allowed_ids=None, *, exact: bool = False) -> list[tuple[str, float]]:
        """余弦相似度降序（Milvus COSINE 度量的 distance 就是相似度）。"""
        if top_n <= 0 or not self._exists() or len(self) == 0 or (allowed_ids is not None and not allowed_ids):
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1).tolist()
        expr, post = self._filter(allowed_ids)

        def fetch(limit: int) -> list[tuple[str, float]]:
            res = self.client.search(self.collection, data=[q], limit=min(limit, 16384), output_fields=["id"], filter=expr or "")
            return [(h["id"], float(h["distance"])) for h in (res[0] if res else [])]

        return widen_until_filled(fetch, top_n, len(self), post) if post is not None else fetch(top_n)

    def search_exact(self, query: np.ndarray, top_n: int, allowed_ids=None) -> list[tuple[str, float]]:
        return self.search(query, top_n, allowed_ids, exact=True)

    # --- 持久化 ---------------------------------------------------------------------------
    def save(self, directory: Path | None = None, prefix: str = "") -> None:
        """Milvus 写入即持久化；这里只把缓冲刷到存储（集群上 flush 是异步的）。"""
        if self._exists():
            try:
                self.client.flush(self.collection)
            except Exception as exc:   # Milvus Lite 没有 flush，也不需要
                logger.debug("milvus flush 跳过：%s", exc)

    def info(self) -> dict[str, Any]:
        return {"name": self.collection, "uri": self.uri, "local": self.local, "rows": len(self), "dim": self.dim}

    def coarse_info(self) -> dict[str, Any]:
        return {"dim": self.dim, "bytes": 0, "full_bytes": len(self) * self.dim * 4}
