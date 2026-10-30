"""块存储与文档元数据：parent / child 映射、相邻块查询、元数据过滤。

内存版：全部段落常驻内存，保存时整份重写 chunks.jsonl（万~十万级足够）。千万级用 SQLite 版（sqlite_store.py），接口一致。"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path

from ..schemas import Chunk, DocMetadata
from .scope import ChunkScope, doc_prefix


class ChunkStore:
    backend = "memory"

    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.docs: dict[str, DocMetadata] = {}
        self._children_by_doc: dict[str, list[str]] = {}
        self.generation = 0          # 文档增删 / 元数据改动时递增：检索范围缓存按它失效

    def __len__(self) -> int:
        return len(self.chunks)

    def add_document(self, doc_id: str, metadata: DocMetadata) -> None:
        self.docs[doc_id] = metadata
        self.generation += 1

    def add_documents(self, docs: dict[str, DocMetadata]) -> None:
        self.docs.update(docs)
        self.generation += 1

    def touch_doc(self, doc_id: str) -> None:
        """原地改过 DocMetadata 后调用：检索范围缓存失效。"""
        self.generation += 1

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self.chunks[c.chunk_id] = c
        self._rebuild_order()

    def _rebuild_order(self) -> None:
        self._children_by_doc = {}
        for c in sorted(self.chunks.values(), key=lambda x: (x.doc_id, x.order)):
            if c.kind == "child":
                self._children_by_doc.setdefault(c.doc_id, []).append(c.chunk_id)

    def get(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def get_many(self, ids: list[str]) -> list[Chunk]:
        return [self.chunks[i] for i in ids if i in self.chunks]

    def parent_of(self, chunk_id: str) -> Chunk | None:
        c = self.chunks.get(chunk_id)
        if c is None:
            return None
        if c.kind == "parent":
            return c
        return self.chunks.get(c.parent_id) if c.parent_id else None

    def children_of(self, parent_id: str) -> list[Chunk]:
        p = self.chunks.get(parent_id)
        return self.get_many(p.child_ids) if p else []

    def neighbors(self, chunk_id: str, window: int) -> list[Chunk]:
        """同一文档内按顺序前后各 window 个 child（句子窗口检索用）。"""
        c = self.chunks.get(chunk_id)
        if c is None or window <= 0:
            return []
        seq = self._children_by_doc.get(c.doc_id, [])
        if chunk_id not in seq:
            return []
        i = seq.index(chunk_id)
        return self.get_many(seq[max(0, i - window):i] + seq[i + 1:i + 1 + window])

    def child_ids(self) -> list[str]:
        return [cid for ids in self._children_by_doc.values() for cid in ids]

    def iter_chunks(self, kind: str | None = None) -> Iterator[Chunk]:
        return (c for c in list(self.chunks.values()) if kind is None or c.kind == kind)

    def chunks_of_doc(self, doc_id: str, kind: str | None = None) -> list[Chunk]:
        return sorted((c for c in self.chunks.values() if c.doc_id == doc_id and (kind is None or c.kind == kind)), key=lambda c: (c.order, c.chunk_id))

    def count_by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.chunks.values():
            out[c.kind] = out.get(c.kind, 0) + 1
        return out

    def count_doc(self, doc_id: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.chunks.values():
            if c.doc_id == doc_id:
                out[c.kind] = out.get(c.kind, 0) + 1
        return out

    def doc_of(self, chunk_id: str) -> str | None:
        d = doc_prefix(chunk_id)
        if d in self.docs:
            return d
        c = self.chunks.get(chunk_id)
        return c.doc_id if c is not None else None

    def scope_chunk_ids(self, scope: ChunkScope, kinds: tuple[str, ...] = ("child",)) -> Iterator[str]:
        """把文档级范围展开成段落 id（只给认不得 ChunkScope 的老后端用）。"""
        if kinds == ("child",):
            return (cid for d, ids in self._children_by_doc.items() if scope.allows_doc(d) for cid in ids)
        return (c.chunk_id for c in self.chunks.values() if c.kind in kinds and scope.allows_doc(c.doc_id))

    def remove_document(self, doc_id: str) -> list[str]:
        removed = [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]
        for cid in removed:
            del self.chunks[cid]
        self.docs.pop(doc_id, None)
        self.generation += 1
        self._rebuild_order()
        return removed

    def remove_chunks(self, ids: set[str]) -> list[str]:
        """移除若干块（语料治理剔除段落）；parent 的 child_ids 同步清理。"""
        removed = [cid for cid in ids if cid in self.chunks]
        for cid in removed:
            del self.chunks[cid]
        for c in self.chunks.values():
            if c.child_ids and any(x in ids for x in c.child_ids):
                c.child_ids = [x for x in c.child_ids if x not in ids]
        self._rebuild_order()
        return removed

    # ----------------------------------------------------------------------------------
    def filter_child_ids(self, predicate: Callable[[Chunk, DocMetadata | None], bool]) -> set[str]:
        out: set[str] = set()
        for cid in self.child_ids():
            c = self.chunks[cid]
            if predicate(c, self.docs.get(c.doc_id)):
                out.add(cid)
        return out

    def effective_child_ids(self, on: date | None = None) -> set[str]:
        return self.filter_child_ids(lambda c, md: md is None or md.is_effective(on))

    # ----------------------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks.values():
                f.write(c.model_dump_json() + "\n")
        (directory / "docs.json").write_text(
            json.dumps({k: json.loads(v.model_dump_json()) for k, v in self.docs.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "ChunkStore":
        store = cls()
        p = directory / "chunks.jsonl"
        if p.exists():
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        c = Chunk.model_validate_json(line)
                        store.chunks[c.chunk_id] = c
        d = directory / "docs.json"
        if d.exists():
            raw = json.loads(d.read_text(encoding="utf-8"))
            store.docs = {k: DocMetadata.model_validate(v) for k, v in raw.items()}
        store._rebuild_order()
        return store
