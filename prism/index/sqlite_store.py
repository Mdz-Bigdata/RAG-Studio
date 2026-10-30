"""块存储的 SQLite 版（storage.chunk_store: sqlite）：千万级段落的知识库用。

内存版 ChunkStore 把全部段落常驻内存、保存时整份重写 chunks.jsonl——千万级段落要十几 GB 内存，每传一个文档写几 GB。
这里段落存在 <index_dir>/chunks.sqlite：
- 按 id 读（主键），parent / 同文档相邻段落走索引；最近读过的段落留在 LRU 缓存里；
- 写入按文档一个事务、立即提交（向量 / 全文写进 LanceDB 也是立即持久化的，两边不会长时间不一致）；
- 段落 JSON 用 zlib 压缩（中文正文约省一半磁盘）；
- 文档元数据（DocMetadata）仍常驻内存：过滤要用、数量通常是段落的 1/20~1/50。原地改了元数据要调用 touch_doc，
  save() 时写回。

对外接口与内存版 ChunkStore 一致；chunks 是一个按需读盘的只读映射（`store.chunks[cid]` / `in` / `values()` 照常可用，
values() 是流式遍历，体检 / 评测这类全库操作仍然可以跑，只是不再一次装进内存）。
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
import zlib
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import date
from pathlib import Path

from ..schemas import Chunk, DocMetadata
from .doc_index import DocIndex
from .scope import ChunkScope, doc_prefix

logger = logging.getLogger(__name__)

FILE = "chunks.sqlite"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, kind TEXT NOT NULL, parent_id TEXT, ord INTEGER NOT NULL, body BLOB NOT NULL) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id, kind, ord);
CREATE INDEX IF NOT EXISTS chunks_parent ON chunks(parent_id);
CREATE INDEX IF NOT EXISTS chunks_kind ON chunks(kind);
CREATE TABLE IF NOT EXISTS docs (doc_id TEXT PRIMARY KEY, body TEXT NOT NULL) WITHOUT ROWID;
"""
_BATCH = 900          # SQLite 变量个数上限（999）以内


def _pack(c: Chunk) -> bytes:
    return zlib.compress(c.model_dump_json().encode("utf-8"), 1)


def _unpack(b: bytes) -> Chunk:
    return Chunk.model_validate_json(zlib.decompress(b))


class _DocDict(dict):
    """记下增删过的文档，save() 只写这些。"""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.dirty: set[str] = set()
        self.removed: set[str] = set()
        self.on_change: Callable[[], None] | None = None

    def __setitem__(self, k, v) -> None:
        super().__setitem__(k, v)
        self.dirty.add(k)
        self.removed.discard(k)
        if self.on_change:
            self.on_change()

    def pop(self, k, *default):
        if k in self:
            self.removed.add(k)
            self.dirty.discard(k)
            if self.on_change:
                self.on_change()
        return super().pop(k, *default)

    def __delitem__(self, k) -> None:
        self.pop(k)


class _ChunkView(Mapping):
    """store.chunks 的兼容视图：按需读盘。"""

    def __init__(self, store: "SqliteChunkStore") -> None:
        self._s = store

    def __getitem__(self, k: str) -> Chunk:
        c = self._s.get(k)
        if c is None:
            raise KeyError(k)
        return c

    def __contains__(self, k: object) -> bool:
        return isinstance(k, str) and self._s.get(k) is not None

    def __len__(self) -> int:
        return len(self._s)

    def __iter__(self) -> Iterator[str]:
        for (cid,) in self._s._rows("SELECT id FROM chunks"):
            yield cid

    def values(self):          # 流式：不逐个按 id 回查
        return self._s.iter_chunks()

    def items(self):
        return ((c.chunk_id, c) for c in self._s.iter_chunks())

    def get(self, k, default=None):
        c = self._s.get(k)
        return default if c is None else c


class SqliteChunkStore:
    backend = "sqlite"

    def __init__(self, directory: Path, *, cache_size: int = 20000) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / FILE
        self._local = threading.local()
        self._wlock = threading.RLock()
        self._cache: OrderedDict[str, Chunk] = OrderedDict()
        self._cache_size = cache_size
        self._clock = threading.Lock()
        self._count: int | None = None
        self.generation = 0
        with self._wlock:
            self._conn().executescript(_SCHEMA)
        self.docs = _DocDict({doc_id: DocMetadata.model_validate_json(body) for doc_id, body in self._rows("SELECT doc_id, body FROM docs")})
        self.docs.dirty.clear()
        self.docs.on_change = self._bump
        self.doc_index = DocIndex.build(self.docs.items())     # 增量维护：增删 / 改元数据时同步
        self.chunks = _ChunkView(self)

    # --- 连接：每个线程一个（WAL 模式下读写互不阻塞）------------------------------------
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-65536")        # 64 MB 页缓存
            conn.execute("PRAGMA mmap_size=268435456")
            self._local.conn = conn
        return conn

    def _rows(self, sql: str, params: Iterable = ()) -> Iterator[tuple]:
        cur = self._conn().execute(sql, tuple(params))
        while True:
            rows = cur.fetchmany(2000)
            if not rows:
                return
            yield from rows

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _bump(self) -> None:
        self.generation += 1

    # --- 缓存 -----------------------------------------------------------------------------
    def _remember(self, chunks: Iterable[Chunk]) -> None:
        with self._clock:
            for c in chunks:
                self._cache[c.chunk_id] = c
                self._cache.move_to_end(c.chunk_id)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def _forget(self, ids: Iterable[str] | None = None) -> None:
        with self._clock:
            if ids is None:
                self._cache.clear()
            else:
                for i in ids:
                    self._cache.pop(i, None)

    # --- 读 -------------------------------------------------------------------------------
    def __len__(self) -> int:
        if self._count is None:
            self._count = self._conn().execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return self._count

    def get(self, chunk_id: str) -> Chunk | None:
        with self._clock:
            c = self._cache.get(chunk_id)
        if c is not None:
            return c
        row = self._conn().execute("SELECT body FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None:
            return None
        c = _unpack(row[0])
        self._remember([c])
        return c

    def get_many(self, ids: list[str]) -> list[Chunk]:
        found: dict[str, Chunk] = {}
        missing: list[str] = []
        with self._clock:
            for i in ids:
                c = self._cache.get(i)
                if c is not None:
                    found[i] = c
                else:
                    missing.append(i)
        for k in range(0, len(missing), _BATCH):
            part = missing[k:k + _BATCH]
            rows = self._conn().execute(f"SELECT body FROM chunks WHERE id IN ({','.join('?' * len(part))})", part).fetchall()
            got = [_unpack(b) for (b,) in rows]
            self._remember(got)
            found.update((c.chunk_id, c) for c in got)
        return [found[i] for i in ids if i in found]

    def parent_of(self, chunk_id: str) -> Chunk | None:
        c = self.get(chunk_id)
        if c is None:
            return None
        if c.kind == "parent":
            return c
        return self.get(c.parent_id) if c.parent_id else None

    def children_of(self, parent_id: str) -> list[Chunk]:
        p = self.get(parent_id)
        return self.get_many(p.child_ids) if p else []

    def neighbors(self, chunk_id: str, window: int) -> list[Chunk]:
        c = self.get(chunk_id)
        if c is None or window <= 0 or c.kind != "child":
            return []
        seq = [cid for (cid,) in self._rows("SELECT id FROM chunks WHERE doc_id = ? AND kind = 'child' ORDER BY ord, id", (c.doc_id,))]
        if chunk_id not in seq:
            return []
        i = seq.index(chunk_id)
        return self.get_many(seq[max(0, i - window):i] + seq[i + 1:i + 1 + window])

    def child_ids(self) -> list[str]:
        """全部 child id（全库遍历：只给老代码路径 / 小库用）。"""
        return [cid for (cid,) in self._rows("SELECT id FROM chunks WHERE kind = 'child' ORDER BY doc_id, ord, id")]

    def iter_chunks(self, kind: str | None = None) -> Iterator[Chunk]:
        sql, params = ("SELECT body FROM chunks WHERE kind = ?", (kind,)) if kind else ("SELECT body FROM chunks", ())
        for (b,) in self._rows(sql, params):
            yield _unpack(b)

    def chunks_of_doc(self, doc_id: str, kind: str | None = None) -> list[Chunk]:
        sql = "SELECT body FROM chunks WHERE doc_id = ?" + (" AND kind = ?" if kind else "") + " ORDER BY ord, id"
        return [_unpack(b) for (b,) in self._rows(sql, (doc_id, kind) if kind else (doc_id,))]

    def count_by_kind(self) -> dict[str, int]:
        return dict(self._rows("SELECT kind, COUNT(*) FROM chunks GROUP BY kind"))

    def count_doc(self, doc_id: str) -> dict[str, int]:
        return dict(self._rows("SELECT kind, COUNT(*) FROM chunks WHERE doc_id = ? GROUP BY kind", (doc_id,)))

    def doc_of(self, chunk_id: str) -> str | None:
        d = doc_prefix(chunk_id)
        if d in self.docs:
            return d
        c = self.get(chunk_id)
        return c.doc_id if c is not None else None

    def scope_chunk_ids(self, scope: ChunkScope, kinds: tuple[str, ...] = ("child",)) -> Iterator[str]:
        """把文档级范围展开成段落 id（只给认不得 ChunkScope 的老后端用）。"""
        marks = ",".join("?" * len(kinds))
        if scope.include is not None:
            docs = sorted(scope.include)
            for k in range(0, len(docs), _BATCH):
                part = docs[k:k + _BATCH]
                yield from (cid for (cid,) in self._rows(f"SELECT id FROM chunks WHERE kind IN ({marks}) AND doc_id IN ({','.join('?' * len(part))})", [*kinds, *part]))
        else:
            yield from (cid for (cid, d) in self._rows(f"SELECT id, doc_id FROM chunks WHERE kind IN ({marks})", kinds) if d not in scope.exclude)

    # --- 写（每次调用一个事务，立即提交）---------------------------------------------------
    def add_document(self, doc_id: str, metadata: DocMetadata) -> None:
        self.docs[doc_id] = metadata
        self.doc_index.upsert(doc_id, metadata)
        self._flush_docs()

    def add_documents(self, docs: dict[str, DocMetadata]) -> None:
        """批量导入：一批文档一个事务。"""
        for doc_id, md in docs.items():
            dict.__setitem__(self.docs, doc_id, md)
            self.doc_index.upsert(doc_id, md)
        self.docs.dirty.update(docs)
        self.docs.removed.difference_update(docs)
        self._bump()
        self._flush_docs()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        with self._wlock:
            conn = self._conn()
            with conn:
                conn.executemany("INSERT OR REPLACE INTO chunks(id, doc_id, kind, parent_id, ord, body) VALUES (?, ?, ?, ?, ?, ?)",
                                 [(c.chunk_id, c.doc_id, c.kind, c.parent_id, c.order, _pack(c)) for c in chunks])
            self._count = None
        self._forget(c.chunk_id for c in chunks)

    def remove_document(self, doc_id: str) -> list[str]:
        with self._wlock:
            conn = self._conn()
            removed = [cid for (cid,) in conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()]
            with conn:
                conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self._count = None
        self._forget(removed)
        self.docs.pop(doc_id, None)
        self.doc_index.remove(doc_id)
        self._flush_docs()
        return removed

    def remove_chunks(self, ids: set[str]) -> list[str]:
        """移除若干块（语料治理剔除段落）；所在 parent 的 child_ids 同步清理。"""
        ids = set(ids)
        present = [c for c in self.get_many(sorted(ids))]
        if not present:
            return []
        removed = [c.chunk_id for c in present]
        parents = {c.parent_id for c in present if c.parent_id and c.parent_id not in ids}
        touched = []
        for p in self.get_many(sorted(parents)):
            kept = [x for x in p.child_ids if x not in ids]
            if kept != p.child_ids:
                touched.append(p.model_copy(update={"child_ids": kept}))
        with self._wlock:
            conn = self._conn()
            with conn:
                for k in range(0, len(removed), _BATCH):
                    part = removed[k:k + _BATCH]
                    conn.execute(f"DELETE FROM chunks WHERE id IN ({','.join('?' * len(part))})", part)
            self._count = None
        self._forget(removed)
        self.add_chunks(touched)
        return removed

    def touch_doc(self, doc_id: str) -> None:
        """原地改过 DocMetadata 后调用：save() 时写回，检索范围缓存失效。"""
        if doc_id in self.docs:
            self.docs.dirty.add(doc_id)
            self.doc_index.upsert(doc_id, self.docs[doc_id])
            self._bump()

    def _flush_docs(self) -> None:
        dirty, removed = list(self.docs.dirty), list(self.docs.removed)
        if not dirty and not removed:
            return
        with self._wlock:
            conn = self._conn()
            with conn:
                conn.executemany("INSERT OR REPLACE INTO docs(doc_id, body) VALUES (?, ?)", [(d, self.docs[d].model_dump_json()) for d in dirty if d in self.docs])
                conn.executemany("DELETE FROM docs WHERE doc_id = ?", [(d,) for d in removed])
        self.docs.dirty.difference_update(dirty)
        self.docs.removed.difference_update(removed)

    # --- 过滤（老接口：全库遍历）-----------------------------------------------------------
    def filter_child_ids(self, predicate: Callable[[Chunk, DocMetadata | None], bool]) -> set[str]:
        return {c.chunk_id for c in self.iter_chunks("child") if predicate(c, self.docs.get(c.doc_id))}

    def effective_child_ids(self, on: date | None = None) -> set[str]:
        return self.filter_child_ids(lambda c, md: md is None or md.is_effective(on))

    # --- 持久化 ---------------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        """写入是即时提交的；这里补写改过的文档元数据。保存到别的目录（重建时的暂存目录）就整库复制过去。"""
        self._flush_docs()
        directory = Path(directory)
        if directory.resolve() != self.directory.resolve():
            directory.mkdir(parents=True, exist_ok=True)
            dst = sqlite3.connect(directory / FILE)
            with dst:
                self._conn().backup(dst)
            dst.close()

    @classmethod
    def load(cls, directory: Path, *, cache_size: int = 20000) -> "SqliteChunkStore":
        """打开 <directory>/chunks.sqlite；只有内存版的 chunks.jsonl 时一次性导入（导入后 jsonl 改名留底）。"""
        directory = Path(directory)
        store = cls(directory, cache_size=cache_size)
        legacy = directory / "chunks.jsonl"
        if legacy.exists() and len(store) == 0:
            from .doc_store import ChunkStore

            old = ChunkStore.load(directory)
            logger.info("段落存储迁移到 SQLite：%d 个段落、%d 篇文档（%s）", len(old), len(old.docs), directory)
            batch: list[Chunk] = []
            for c in old.chunks.values():
                batch.append(c)
                if len(batch) >= 5000:
                    store.add_chunks(batch)
                    batch = []
            store.add_chunks(batch)
            store.add_documents(old.docs)
            legacy.rename(directory / "chunks.jsonl.migrated")
            store._forget()
        return store

    def export_to_memory(self):
        """切回内存版时用：读出全部段落与文档。"""
        from .doc_store import ChunkStore

        mem = ChunkStore()
        mem.docs = dict(self.docs)
        for c in self.iter_chunks():
            mem.chunks[c.chunk_id] = c
        mem._rebuild_order()
        return mem


def move_aside(directory: Path) -> None:
    """切回内存版后，旧的 chunks.sqlite 改名留底（否则下次切回 SQLite 时会读到过期数据）。"""
    p = Path(directory) / FILE
    if p.exists():
        for suffix in ("", "-wal", "-shm"):
            q = p.with_name(p.name + suffix)
            if q.exists():
                shutil.move(str(q), str(q.with_name(q.name + ".old")))
