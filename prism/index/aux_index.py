"""辅助索引：摘要索引 / 章节级 / 句子级。

- numpy 后端：各自一份"向量 + BM25"，与主索引（child）共用一个目录，按前缀区分文件；
  查询时在辅助索引内部先做一次小 RRF（向量 + BM25），作为一路结果交给主融合层。
- LanceDB 后端（vector_store.backend: lancedb）：每个辅助索引一张表 aux_<name>（向量 + 正文 + ICU 全文索引），
  查询直接用 LanceDB 原生混合检索（向量 + 全文，RRF k=60），与 numpy 版的融合方式一致。
  切到 LanceDB 后、知识库重建之前，已有的 numpy 文件照常加载使用——不会出现空的辅助路；重建后自然换成 LanceDB 表。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import LanceDBConfig, SparseConfig
from .bm25_store import BM25Store
from .vector_store import NumpyVectorStore


class AuxIndex:
    def __init__(self, name: str, dim: int, sparse_cfg: SparseConfig, *, directory: Path | None = None, lance_cfg: LanceDBConfig | None = None) -> None:
        self.name = name
        self.sparse_cfg = sparse_cfg
        self.lance = None
        if directory is not None and lance_cfg is not None:
            from ..lance import LanceDBVectorStore   # 延迟导入：没装 lancedb 时 numpy 版照常可用

            self.lance = LanceDBVectorStore(Path(directory) / "lancedb", dim, table=f"aux_{name}", cfg=lance_cfg)
            self.vectors = self.lance
            self.bm25: BM25Store | None = None
        else:
            self.vectors = NumpyVectorStore(dim=dim)
            self.bm25 = BM25Store(sparse_cfg)

    @property
    def backend(self) -> str:
        return "lancedb" if self.lance is not None else "numpy"

    def __len__(self) -> int:
        return len(self.vectors)

    @property
    def prefix(self) -> str:
        return f"{self.name}_"

    def add(self, ids: list[str], texts: list[str], vectors: np.ndarray) -> None:
        """texts 进全文（BM25 / FTS），vectors 可以由截短的文本算出（章节级只拿开头算向量）。"""
        if not ids:
            return
        if self.lance is not None:
            self.lance.add(ids, vectors, texts=texts)
            return
        self.vectors.add(ids, vectors)
        self.bm25.add(ids, texts)

    def remove(self, ids: set[str]) -> None:
        self.vectors.remove(list(ids))
        if self.bm25 is not None:
            self.bm25.remove(ids)

    def search(self, query: str, query_vec: np.ndarray, top_n: int, allowed_ids: set[str] | None = None, k: int = 60) -> list[tuple[str, float]]:
        if not len(self):
            return []
        if self.lance is not None:
            return self.lance.hybrid_search(query_vec, query, top_n, allowed_ids)
        dense = self.vectors.search(query_vec, top_n, allowed_ids)
        sparse = self.bm25.search(query, top_n, allowed_ids)
        fused: dict[str, float] = {}
        for ranking in (dense, sparse):
            for rank, (cid, _) in enumerate(ranking, start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def save(self, directory: Path) -> None:
        if self.lance is not None:
            self.lance.save()        # 写入即持久化，这里只做索引维护
            return
        self.vectors.save(directory, prefix=self.prefix)
        self.bm25.save(directory, prefix=self.prefix)

    @classmethod
    def load(cls, name: str, directory: Path, dim: int, sparse_cfg: SparseConfig, *, lance_cfg: LanceDBConfig | None = None) -> "AuxIndex":
        legacy = (directory / f"{name}_vectors.npy").exists()
        if lance_cfg is not None:
            aux = cls(name, dim, sparse_cfg, directory=directory, lance_cfg=lance_cfg)
            if len(aux) or not legacy:
                return aux
            # 已切到 LanceDB、但这个知识库还没重建：先沿用旧的 numpy 文件
        aux = cls(name, dim, sparse_cfg)
        if legacy:
            aux.vectors = NumpyVectorStore.load(directory, prefix=aux.prefix)
        aux.bm25 = BM25Store.load(directory, sparse_cfg, prefix=aux.prefix)
        return aux
