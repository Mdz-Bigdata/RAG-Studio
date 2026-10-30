"""批量写入：千万级知识库的导入通道（scripts/bulk_import.py 与压测 scripts/scale_bench.py 都用它）。

逐个上传时每篇文档单独向量化、单独写入、写完保存一次索引；千万级段落这样做，光是"每篇保存一次"就要好几天。
BulkIndexer 攒够一批（默认 4096 个 child）再一起处理：
- 向量化：跨文档拼成大批次（GPU 吃得满），API 向量模型可以多线程并发请求；也可以直接传入算好的向量；
- 分词：LanceDB jieba 全文索引需要的 tokens 在多进程里切（jieba 是纯 Python，单核约 0.4 MB/s）；
- 写入：向量 + 正文 + tokens 一次追加进 LanceDB，段落 + 文档元数据一个 SQLite 事务；
- 索引：ANN / 全文 / BTREE 不在导入过程中反复重建，finish() 时整理碎片并一次建好。

断点续传：已经在块存储里的文档跳过（has()）。每批写入前在 journal 里记下这批文档，SQLite 提交（最后一步）后记"已提交"；
中途崩溃时，下次打开先把没提交的那批从 LanceDB / SQLite 里清掉，再从头导入这批——不会留下半批数据或重复向量。

切片与逐个上传完全一致（IngestPipeline.chunk_document）；不处理文档里的图片（批量导入只建文本索引）。
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..config import SparseConfig
from ..graph.unweaver import extract_definitions
from ..index import Index
from ..index.table_store import link_chunks
from ..schemas import Chunk, DocMetadata

logger = logging.getLogger(__name__)

_TOKENIZER = None


def _init_tokenizer(cfg: dict) -> None:
    global _TOKENIZER
    from ..index.bm25_store import BM25Store

    _TOKENIZER = BM25Store(SparseConfig.model_validate(cfg)).tokenize


def _tokenize_many(texts: list[str]) -> list[str]:
    return [" ".join(_TOKENIZER(t)) for t in texts]


@dataclass
class BulkStats:
    docs: int = 0
    chunks: int = 0
    children: int = 0
    batches: int = 0
    seconds: dict[str, float] = field(default_factory=lambda: {"embed": 0.0, "tokenize": 0.0, "write_vectors": 0.0, "write_store": 0.0, "indexes": 0.0})

    def as_dict(self) -> dict[str, Any]:
        return {"docs": self.docs, "chunks": self.chunks, "children": self.children, "batches": self.batches, "seconds": {k: round(v, 1) for k, v in self.seconds.items()}}


class BulkIndexer:
    def __init__(self, index: Index, *, batch_chunks: int = 4096, embed_batch: int = 256, embed_workers: int = 1, tokenize_workers: int = 0,
                 on_commit: Callable[[list[tuple[str, DocMetadata, list[Chunk]]]], None] | None = None) -> None:
        """on_commit：每批写入提交后回调 [(doc_id, 元数据, 段落)]——Web 知识库导入时用它登记文档行。"""
        self.index = index
        self.on_commit = on_commit
        self.batch_chunks = batch_chunks
        self.embed_batch = embed_batch
        self.embed_workers = max(1, embed_workers)
        self.stats = BulkStats()
        self._buf: list[tuple[str, DocMetadata, list[Chunk], list, np.ndarray | None]] = []
        self._pending_children = 0
        self._journal = Path(index.directory) / "bulk_import.journal"
        self._jieba = bool(getattr(index.vectors, "jieba_fts", False))
        self._pool = None
        if self._jieba and tokenize_workers > 1:
            self._pool = ProcessPoolExecutor(max_workers=tokenize_workers, initializer=_init_tokenizer, initargs=(index.settings.sparse.model_dump(mode="json"),))
        index.directory.mkdir(parents=True, exist_ok=True)
        self._recover()

    # --- 断点续传 -------------------------------------------------------------------------
    def has(self, doc_id: str) -> bool:
        return doc_id in self.index.store.docs

    def _log(self, rec: dict) -> None:
        with self._journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _recover(self) -> None:
        """上次中途退出时没提交的批次：从 LanceDB / SQLite 里清掉，这些文档会被重新导入。"""
        if not self._journal.exists():
            return
        pending: dict[int, list[str]] = {}
        for line in self._journal.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if "pending" in rec:
                pending[rec["batch"]] = rec["pending"]
            elif "committed" in rec:
                pending.pop(rec["committed"], None)
        dirty = [d for docs in pending.values() for d in docs]
        if dirty:
            logger.warning("批量导入：上次有 %d 篇文档写到一半，先清掉再重新导入", len(dirty))
            remove_docs = getattr(self.index.vectors, "remove_docs", None)
            if callable(remove_docs):
                remove_docs(dirty)
            for d in dirty:
                self.index.store.remove_document(d)
        self._journal.unlink()

    # --- 写入 -----------------------------------------------------------------------------
    def add(self, doc_id: str, metadata: DocMetadata, chunks: list[Chunk], *, tables: list | None = None, vectors: np.ndarray | None = None) -> None:
        """vectors：按 chunks 里 child 的顺序算好的向量（压测 / 外部已向量化时用），不给就用知识库的向量模型算。"""
        if self.has(doc_id):
            self.index.remove_document(doc_id)
        self._buf.append((doc_id, metadata, chunks, tables or [], vectors))
        self._pending_children += sum(1 for c in chunks if c.kind == "child")
        if self._pending_children >= self.batch_chunks:
            self.flush()

    def _embed(self, texts: list[str]) -> np.ndarray:
        parts = [texts[i:i + self.embed_batch] for i in range(0, len(texts), self.embed_batch)]
        if self.embed_workers > 1 and len(parts) > 1:
            with ThreadPoolExecutor(max_workers=self.embed_workers) as pool:   # API 向量模型：并发请求
                out = list(pool.map(self.index.embedder.embed_documents, parts))
        else:
            out = [self.index.embedder.embed_documents(p) for p in parts]
        return np.concatenate(out) if out else np.zeros((0, self.index.embedder.dim), dtype=np.float32)

    def _tokenize(self, texts: list[str]) -> list[str]:
        if self._pool is not None:
            step = max(64, len(texts) // (self._pool._max_workers * 4) or 1)
            return [s for part in self._pool.map(_tokenize_many, [texts[i:i + step] for i in range(0, len(texts), step)]) for s in part]
        tok = self.index.vectors.tokenizer
        return [" ".join(tok(t)) for t in texts]

    def flush(self) -> None:
        if not self._buf:
            return
        idx, st = self.index, self.stats
        batch, self._buf, self._pending_children = self._buf, [], 0
        st.batches += 1
        docs = [b[0] for b in batch]
        self._log({"batch": st.batches, "pending": docs})

        children = [c for _, _, chunks, _, _ in batch for c in chunks if c.kind == "child"]
        texts = [c.embed_text() for c in children]
        ids = [c.chunk_id for c in children]
        t = time.perf_counter()
        if all(b[4] is not None for b in batch):
            vecs = np.concatenate([b[4] for b in batch]).astype(np.float32) if children else np.zeros((0, 1), dtype=np.float32)
        else:
            vecs = self._embed(texts)
        st.seconds["embed"] += time.perf_counter() - t
        if len(vecs) != len(children):
            raise ValueError(f"向量数 {len(vecs)} 与 child 数 {len(children)} 不一致")

        if children:
            t = time.perf_counter()
            tokens = self._tokenize(texts) if self._jieba else None
            st.seconds["tokenize"] += time.perf_counter() - t
            t = time.perf_counter()
            if getattr(idx.vectors, "stores_text", False):
                idx.vectors.add(ids, vecs, texts=texts, doc_ids=[c.doc_id for c in children], fresh=True, tokens=tokens)
            else:
                idx.vectors.add(ids, vecs)
            idx.bm25.add(ids, texts)
            st.seconds["write_vectors"] += time.perf_counter() - t

        t = time.perf_counter()
        all_chunks: list[Chunk] = []
        relations = []
        for doc_id, _md, chunks, tables, _ in batch:
            levels: list[Chunk] = []
            if idx.settings.chunking.multi_granularity:
                from .chunking.hierarchy import build_hierarchy

                levels = build_hierarchy(doc_id, chunks, idx.settings.chunking)
            all_chunks.extend(chunks + levels)
            if levels or any(c.summary for c in chunks):
                idx._add_aux(chunks, levels)
            if idx.settings.retrieval.graph.unweaver_enabled:
                for p in chunks:
                    if p.kind == "parent":
                        relations.extend(extract_definitions(p.text, p.chunk_id, doc_id))
            if tables:
                link_chunks(tables, chunks)
                idx.tables.add(tables)
        if relations:
            idx.entities.add(relations)
        idx.store.add_chunks(all_chunks)
        idx.store.add_documents({doc_id: md for doc_id, md, *_ in batch})   # 最后写文档：它在，这批就算提交了
        st.seconds["write_store"] += time.perf_counter() - t
        self._log({"committed": st.batches})
        if self.on_commit is not None:
            self.on_commit([(doc_id, md, chunks) for doc_id, md, chunks, *_ in batch])
        st.docs += len(batch)
        st.chunks += len(all_chunks)
        st.children += len(children)

    def finish(self, *, build_indexes: bool = True, compact: bool | None = None) -> BulkStats:
        """写完最后一批；整理 LanceDB 碎片、一次建好 ANN / 全文 / BTREE 索引，保存索引元数据。
        compact：合并碎片。默认只在每批较小（平均 < 2 万行）时做——合并要把数据整个重写一遍，临时多占一倍磁盘，
        批次够大时（几百个几万行的文件）没必要。"""
        self.flush()
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        t = time.perf_counter()
        optimize = getattr(self.index.vectors, "optimize", None)
        if compact is None:
            compact = self.stats.batches > 1 and self.stats.children / self.stats.batches < 20000
        if build_indexes and compact and callable(optimize):
            optimize(drop_history=True)      # 合并一批一个的小文件；合并前的旧版本没用了，清掉（否则磁盘翻倍）
        elif hasattr(self.index.vectors, "_writes"):
            # 不合并时也别让下面的 save() 按"写入满 optimize_every 次"自己去合并：那会把整表重写一遍、旧版本还按天数保留，
            # 磁盘占用翻倍（千万级实测 LanceDB 30 GB → 69 GB，把磁盘写满）
            self.index.vectors._writes = 0
        if build_indexes:
            self.index.save()                # LanceDB 在 save 里补齐索引（数据量到了建 ANN，涨到 4 倍重训）
        else:
            self.index.store.save(self.index.directory)
        self.stats.seconds["indexes"] += time.perf_counter() - t
        self._journal.unlink(missing_ok=True)
        return self.stats
