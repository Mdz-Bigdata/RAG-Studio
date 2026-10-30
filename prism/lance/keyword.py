"""关键词路的 LanceDB 版：全文索引建在 children 表的 tokens 列上（jieba 预分词，与 BM25 同一套分词）。

接口和 BM25Store 一样（tokenize / search / add / remove / save），给 Index 当 bm25 用。区别：
- BM25Store 把全部段落的分词结果放在内存里，增删任何文档都要整库重建（bm25s 不支持增量）；
- 这里正文随向量一起写进 LanceDB（Index.add_document 写向量时带上 texts，tokens 列由向量存储按同一分词器切好），
  索引增量维护、文档级过滤下推，内存里什么都不留。
启用条件：vector_store.backend: lancedb + lancedb.fts_tokenizer: jieba + retrieval.sparse_backend: lancedb_fts。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..config import SparseConfig
from ..index.bm25_store import BM25Store
from .vector_store import LanceDBVectorStore


class _TokenCorpus:
    """corpus_tokens 的流式替身（覆盖率评测要统计全库词频）：不一次装进内存。"""

    def __init__(self, store: LanceDBVectorStore) -> None:
        self._store = store

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[list[str]]:
        tbl = self._store._table()
        if tbl is None:
            return
        # 流式读（不依赖 pylance，也不一次装进内存）
        for batch in tbl.search().select(["tokens"]).limit(None).to_batches(8192):
            for s in batch.column("tokens").to_pylist():
                yield (s or "").split()


class LanceKeywordIndex:
    backend = "lancedb_fts"

    def __init__(self, vectors: LanceDBVectorStore, cfg: SparseConfig) -> None:
        self.vectors = vectors
        self._tok = BM25Store(cfg)            # 只用它的分词器（jieba + 编号精确 token + 停用词），不存语料
        vectors.tokenizer = self._tok.tokenize

    def tokenize(self, text: str) -> list[str]:
        return self._tok.tokenize(text)

    def __len__(self) -> int:
        return len(self.vectors)

    @property
    def corpus_tokens(self) -> _TokenCorpus:
        return _TokenCorpus(self.vectors)

    def _ensure_index(self) -> None:
        self.vectors.ensure_fts()

    def search(self, query: str, top_n: int, allowed_ids=None) -> list[tuple[str, float]]:
        return self.vectors.fts_search(query, top_n, allowed_ids)

    # 正文随向量一起写入 / 删除（Index.add_document / remove_document），这里不必再做
    def add(self, ids: list[str], texts: list[str]) -> None:
        return None

    def remove(self, ids: set[str]) -> None:
        return None

    def save(self, directory: Path, prefix: str = "") -> None:
        return None
