"""向量存储：统一接口 add / search / get_vectors / remove / save / load。

- NumpyVectorStore：零依赖、可持久化（npy + json），几万条以内暴力余弦足够快，也便于测试
- QdrantVectorStore：内存 / 本地文件 / 远程服务三种模式（pip install qdrant-client）
- ChromaVectorStore：chromadb 持久化集合（pip install chromadb）
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np


IN_LIMIT = 5000   # allowed_ids 不超过该数量时下推给向量库做预过滤；更多时改为"多取再过滤，不够就加倍重取"


def widen_until_filled(fetch: Callable[[int], list[tuple[str, float]]], top_n: int, total: int, allowed_ids: set[str]) -> list[tuple[str, float]]:
    """后置过滤的补救：过滤后不足 top_n 就把取回数量翻倍重查，直到够数或取完全库。
    只取固定的前几倍会漏掉排在后面的可见块：库里明明有，却被排在前面的不可见块挤掉。"""
    limit = min(max(top_n * 4, top_n), total)
    while True:
        hits = [h for h in fetch(limit) if h[0] in allowed_ids]
        if len(hits) >= top_n or limit >= total:
            return hits[:top_n]
        limit = min(limit * 4, total)


class NumpyVectorStore:
    name = "numpy"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.ids: list[str] = []
        self.matrix = np.zeros((0, dim), dtype=np.float32)
        self._row: dict[str, int] = {}
        # 查询期调优（configure 设置）：PCA 降维 + int8 量化做粗筛，再用全精度向量精算
        self.pca_dim: int | None = None
        self.quantize = "none"
        self.rescore = True
        self.oversample = 4.0
        self._coarse: dict | None = None     # 缓存的降维 / 量化矩阵；增删向量后失效

    def configure(self, *, pca_dim: int | None = None, quantize: str = "none", rescore: bool = True, oversample: float = 4.0, **_: object) -> None:
        if (pca_dim, quantize) != (self.pca_dim, self.quantize):
            self._coarse = None
        self.pca_dim, self.quantize, self.rescore, self.oversample = pca_dim, quantize, rescore, max(1.0, oversample)

    @property
    def tuned(self) -> bool:
        return bool(self.pca_dim) or self.quantize != "none"

    def _build_coarse(self) -> dict:
        """PCA：对当前向量做 SVD 取前 k 个主成分（降维后重新归一化，保持余弦语义）；int8：逐维对称量化。"""
        x = self.matrix
        info: dict = {"mean": None, "comps": None, "scale": None}
        reduced = x
        if self.pca_dim and 1 < self.pca_dim < x.shape[1] and len(x) >= 2:
            k = min(self.pca_dim, len(x) - 1, x.shape[1])
            mean = x.mean(axis=0)
            _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
            comps = vt[:k].astype(np.float32)
            reduced = (x - mean) @ comps.T
            reduced /= np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-9
            info.update(mean=mean.astype(np.float32), comps=comps)
        if self.quantize == "int8":
            scale = np.abs(reduced).max(axis=0) / 127.0 + 1e-12
            info["scale"] = scale.astype(np.float32)
            info["matrix"] = np.round(reduced / scale).astype(np.int8)
        else:
            info["matrix"] = reduced.astype(np.float32)
        info["dim"] = info["matrix"].shape[1] if info["matrix"].ndim == 2 else 0
        info["bytes"] = int(info["matrix"].nbytes)
        return info

    def coarse_info(self) -> dict:
        if not self.ids:
            return {"dim": self.dim, "bytes": 0, "full_bytes": 0}
        if self._coarse is None:
            self._coarse = self._build_coarse()
        return {"dim": self._coarse["dim"], "bytes": self._coarse["bytes"], "full_bytes": int(self.matrix.nbytes)}

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if not ids:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            if len(self.ids) == 0:
                self.dim = vectors.shape[1]
                self.matrix = np.zeros((0, self.dim), dtype=np.float32)
            else:
                raise ValueError(f"向量维度不匹配：{vectors.shape[1]} != {self.dim}")
        keep = [i for i, cid in enumerate(ids) if cid not in self._row]
        for i, cid in enumerate(ids):
            if cid in self._row:
                self.matrix[self._row[cid]] = vectors[i]
        if keep:
            start = len(self.ids)
            self.matrix = np.vstack([self.matrix, vectors[keep]])
            for j, i in enumerate(keep):
                self.ids.append(ids[i])
                self._row[ids[i]] = start + j
        self._coarse = None

    def remove(self, ids: list[str]) -> None:
        drop = {cid for cid in ids if cid in self._row}
        if not drop:
            return
        keep = [i for i, cid in enumerate(self.ids) if cid not in drop]
        self.matrix = self.matrix[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
        self.ids = [self.ids[i] for i in keep]
        self._row = {cid: i for i, cid in enumerate(self.ids)}
        self._coarse = None

    def get_vectors(self, ids: list[str]) -> dict[str, np.ndarray]:
        return {cid: self.matrix[self._row[cid]] for cid in ids if cid in self._row}

    def search(self, query: np.ndarray, top_n: int, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if self.tuned and self.ids:
            return self._search_tuned(query, top_n, allowed_ids)
        return self.search_exact(query, top_n, allowed_ids)

    def _search_tuned(self, query: np.ndarray, top_n: int, allowed_ids: set[str] | None) -> list[tuple[str, float]]:
        if self._coarse is None:
            self._coarse = self._build_coarse()
        c = self._coarse
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        qc = q
        if c["comps"] is not None:
            qc = (q - c["mean"]) @ c["comps"].T
            qc /= np.linalg.norm(qc) + 1e-9
        if c["scale"] is not None:
            scores = c["matrix"].astype(np.float32) @ (qc * c["scale"])
        else:
            scores = c["matrix"] @ qc
        if allowed_ids is not None:
            mask = np.fromiter((cid in allowed_ids for cid in self.ids), dtype=bool, count=len(self.ids))
            scores = np.where(mask, scores, -np.inf)
        n_cand = min(len(self.ids), max(top_n, int(top_n * self.oversample)) if self.rescore else top_n)
        idx = np.argpartition(-scores, n_cand - 1)[:n_cand]
        idx = idx[np.isfinite(scores[idx])]
        if self.rescore:
            exact = self.matrix[idx] @ q                      # 粗筛候选用全精度向量回算
            order = np.argsort(-exact)[:top_n]
            return [(self.ids[idx[i]], float(exact[i])) for i in order]
        idx = idx[np.argsort(-scores[idx])][:top_n]
        return [(self.ids[i], float(scores[i])) for i in idx]

    def search_exact(self, query: np.ndarray, top_n: int, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        scores = self.matrix @ q
        if allowed_ids is not None:
            mask = np.fromiter((cid in allowed_ids for cid in self.ids), dtype=bool, count=len(self.ids))
            scores = np.where(mask, scores, -np.inf)
        k = min(top_n, len(self.ids))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.ids[i], float(scores[i])) for i in idx if np.isfinite(scores[i])]

    def save(self, directory: Path, prefix: str = "") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"{prefix}vectors.npy", self.matrix)
        (directory / f"{prefix}vector_ids.json").write_text(json.dumps(self.ids, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, prefix: str = "") -> "NumpyVectorStore":
        matrix = np.load(directory / f"{prefix}vectors.npy")
        ids = json.loads((directory / f"{prefix}vector_ids.json").read_text(encoding="utf-8"))
        store = cls(dim=matrix.shape[1] if matrix.ndim == 2 else 0)
        store.matrix = matrix.astype(np.float32)
        store.ids = ids
        store._row = {cid: i for i, cid in enumerate(ids)}
        return store


def _point_id(chunk_id: str) -> str:
    """Qdrant 要求 point id 为 uuid / 整数：用确定性 uuid5 映射 chunk_id。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorStore:
    """内存 / 本地文件 / 远程三种模式；payload 保存原 chunk_id。"""

    name = "qdrant"

    def __init__(self, collection: str, dim: int, *, url: str | None = None, path: str | Path | None = None, hnsw_m: int = 16, hnsw_ef_construct: int = 100, quantization: str = "none", ef_search: int = 128, oversample: float = 4.0) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("需要 qdrant-client：pip install qdrant-client") from exc
        if url:
            self.client = QdrantClient(url=url)
        elif path:
            self.client = QdrantClient(path=str(path))
        else:
            self.client = QdrantClient(":memory:")
        self.collection = collection
        self.dim = dim
        self.hnsw_m, self.hnsw_ef_construct, self.quantization = hnsw_m, hnsw_ef_construct, quantization
        self.ef_search, self.oversample = ef_search, oversample
        self._ensure()

    def configure(self, *, ef_search: int = 128, oversample: float = 4.0, **_: object) -> None:
        self.ef_search, self.oversample = ef_search, oversample

    @property
    def tuned(self) -> bool:
        return True

    def _search_params(self, exact: bool = False):
        from qdrant_client.http import models as rest

        quant = rest.QuantizationSearchParams(rescore=True, oversampling=self.oversample) if self.quantization != "none" else None
        return rest.SearchParams(hnsw_ef=self.ef_search, exact=exact, quantization=quant)

    def _ensure(self) -> None:
        from qdrant_client.http import models as rest

        if self.dim and not self.client.collection_exists(self.collection):
            quant = None
            if self.quantization == "int8":
                quant = rest.ScalarQuantization(scalar=rest.ScalarQuantizationConfig(type=rest.ScalarType.INT8, always_ram=True))
            elif self.quantization == "binary":
                quant = rest.BinaryQuantization(binary=rest.BinaryQuantizationConfig(always_ram=True))
            self.client.create_collection(
                self.collection,
                vectors_config=rest.VectorParams(size=self.dim, distance=rest.Distance.COSINE),
                hnsw_config=rest.HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct),
                quantization_config=quant,
            )

    def __len__(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        from qdrant_client.http import models as rest

        if not ids:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if not self.dim:
            self.dim = vectors.shape[1]
        self._ensure()
        points = [rest.PointStruct(id=_point_id(cid), vector=vectors[i].tolist(), payload={"chunk_id": cid}) for i, cid in enumerate(ids)]
        self.client.upsert(self.collection, points=points)

    def remove(self, ids: list[str]) -> None:
        from qdrant_client.http import models as rest

        if ids and self.client.collection_exists(self.collection):
            self.client.delete(self.collection, points_selector=rest.PointIdsList(points=[_point_id(c) for c in ids]))

    def get_vectors(self, ids: list[str]) -> dict[str, np.ndarray]:
        if not ids or not self.client.collection_exists(self.collection):
            return {}
        records = self.client.retrieve(self.collection, ids=[_point_id(c) for c in ids], with_vectors=True, with_payload=True)
        out: dict[str, np.ndarray] = {}
        for r in records:
            cid = (r.payload or {}).get("chunk_id")
            if cid and r.vector is not None:
                out[cid] = np.asarray(r.vector, dtype=np.float32)
        return out

    def search_exact(self, query: np.ndarray, top_n: int, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        return self.search(query, top_n, allowed_ids, exact=True)

    def search(self, query: np.ndarray, top_n: int, allowed_ids: set[str] | None = None, exact: bool = False) -> list[tuple[str, float]]:
        if not self.client.collection_exists(self.collection) or len(self) == 0:
            return []
        from qdrant_client.http import models as rest

        if allowed_ids is not None and not allowed_ids:
            return []
        vector = np.asarray(query, dtype=np.float32).tolist()

        def fetch(limit: int, query_filter=None) -> list[tuple[str, float]]:
            res = self.client.query_points(self.collection, query=vector, limit=limit, with_payload=True,
                                           query_filter=query_filter, search_params=self._search_params(exact))
            return [(cid, float(p.score)) for p in res.points if (cid := (p.payload or {}).get("chunk_id"))]

        if allowed_ids is None:
            return fetch(top_n)
        if len(allowed_ids) <= IN_LIMIT:
            # 过滤条件下推：先按 chunk_id 圈定范围，再在范围内找最近邻
            match = rest.FieldCondition(key="chunk_id", match=rest.MatchAny(any=sorted(allowed_ids)))
            return fetch(top_n, rest.Filter(must=[match]))
        return widen_until_filled(fetch, top_n, len(self), allowed_ids)

    def save(self, directory: Path) -> None:  # Qdrant 自行持久化
        return None


class ChromaVectorStore:
    """chromadb 持久化集合（pip install 'rag-agent-studio[chroma]'）。"""

    name = "chroma"

    def __init__(self, directory: Path, collection: str = "rag_children") -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("需要 chromadb：pip install 'rag-agent-studio[chroma]'") from exc
        self.client = chromadb.PersistentClient(path=str(directory))
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        self.dim = 0

    def __len__(self) -> int:
        return self.col.count()

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if ids:
            self.col.upsert(ids=ids, embeddings=np.asarray(vectors, dtype=np.float32).tolist())

    def remove(self, ids: list[str]) -> None:
        if ids:
            self.col.delete(ids=ids)

    def get_vectors(self, ids: list[str]) -> dict[str, np.ndarray]:
        if not ids:
            return {}
        res = self.col.get(ids=ids, include=["embeddings"])
        return {cid: np.asarray(v, dtype=np.float32) for cid, v in zip(res["ids"], res["embeddings"] or [])}

    def search(self, query: np.ndarray, top_n: int, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        n = self.col.count()
        if n == 0:
            return []
        if allowed_ids is not None and not allowed_ids:
            return []
        vector = np.asarray(query, dtype=np.float32).tolist()

        def fetch(limit: int) -> list[tuple[str, float]]:
            res = self.col.query(query_embeddings=[vector], n_results=limit)
            return [(cid, 1.0 - float(dist)) for cid, dist in zip(res["ids"][0], res["distances"][0])]

        if allowed_ids is None:
            return fetch(min(top_n, n))
        return widen_until_filled(fetch, top_n, n, allowed_ids)

    def save(self, directory: Path) -> None:
        return None
