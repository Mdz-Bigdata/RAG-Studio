"""每个知识库一个 prism.Index（目录 data/indexes/<kb_id>），带缓存与写锁。"""

from __future__ import annotations

import logging
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from prism.corpus.aliases import AliasDict
from prism.index import Index
from prism.ingest import IngestPipeline

from ..config import app_config
from ..db import Repository, SessionLocal
from .model_service import model_service
from .settings_service import settings_service

logger = logging.getLogger(__name__)


class _ReadWriteLock:
    """检索持共享读锁，重建收尾的目录替换持独占写锁。

    LanceDB 等磁盘后端按路径懒读数据文件：一个查询拿着旧索引对象执行到一半时目录被替换，就会读到新索引的文件清单之外
    （Not found）。替换只需几毫秒，等在途查询结束再换即可；重建本身（慢的部分）不持这把锁，不会挡住检索。
    读锁按线程可重入：检索内部再调检索不会和排队中的写锁互相等待；写锁对持有者可重入，持有者也可直接读。

    加锁顺序固定为「先这把锁，再知识库写锁 index_service.lock」：检索拿到读锁后会在 get() 里短暂取知识库锁，
    写方若先持知识库锁再等读者，就会互相等死。"""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._readers = 0
        self._writer: int | None = None             # 持有写锁的线程 id
        self._write_depth = 0
        self._tls = threading.local()

    @contextmanager
    def read(self) -> Iterator[None]:
        depth = getattr(self._tls, "depth", 0)
        if depth == 0 and self._writer == threading.get_ident():
            yield                                    # 写锁持有者读自己的数据
            return
        if depth == 0:
            with self._cv:
                self._cv.wait_for(lambda: self._writer is None)
                self._readers += 1
        self._tls.depth = depth + 1
        try:
            yield
        finally:
            self._tls.depth = depth
            if depth == 0:
                with self._cv:
                    self._readers -= 1
                    self._cv.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        me = threading.get_ident()
        with self._cv:
            if self._writer != me:
                self._cv.wait_for(lambda: self._writer is None)
                self._writer = me                    # 先挡住新的读者，再等在途读者结束
                self._cv.wait_for(lambda: self._readers == 0)
            self._write_depth += 1
        try:
            yield
        finally:
            with self._cv:
                self._write_depth -= 1
                if self._write_depth == 0:
                    self._writer = None
                    self._cv.notify_all()


def _forget_lance(*paths: Path) -> None:
    try:
        from prism.lance.db import forget_connection
    except Exception:
        return
    for p in paths:
        for sub in (p, p / "lancedb"):
            forget_connection(sub)


def embedder_signature(emb) -> tuple[str, str | None]:
    """向量模型的身份，与 index_meta.json 里的 embedder / model 两个字段对应。"""
    return emb.name, getattr(emb, "model_name", None)


class IndexService:
    def __init__(self) -> None:
        self._indexes: dict[str, Index] = {}
        self._mtimes: dict[str, float | None] = {}   # 加载 / 保存时 index_meta.json 的修改时间
        self._locks: dict[str, threading.RLock] = {}
        self._global = threading.RLock()
        self._staging: dict[str, Index] = {}         # 整库重建中的新索引（只对重建线程可见）
        self._builds: dict[str, tuple[str, str | None] | None] = {}   # 整库重建所用向量模型的签名，见 rebuild_embedder
        self._tls = threading.local()
        self._rw: dict[str, _ReadWriteLock] = {}

    def _rwlock(self, kb_id: str) -> _ReadWriteLock:
        with self._global:
            return self._rw.setdefault(kb_id, _ReadWriteLock())

    @contextmanager
    def swap_guard(self, kb_id: str) -> Iterator[None]:
        """整库重建收尾：挡住新检索并等在途检索结束。必须在 lock(kb_id) 之前获取（见 _ReadWriteLock）。"""
        with self._rwlock(kb_id).write():
            yield

    @contextmanager
    def reading(self, kb_id: str) -> Iterator[None]:
        """检索期间持有：保证这次检索用到的索引文件不会被重建收尾的目录替换换掉。"""
        with self._rwlock(kb_id).read():
            yield

    def lock(self, kb_id: str) -> threading.RLock:
        with self._global:
            return self._locks.setdefault(kb_id, threading.RLock())

    def directory(self, kb_id: str) -> Path:
        return app_config.index_root / kb_id

    def _disk_mtime(self, kb_id: str) -> float | None:
        meta = self.directory(kb_id) / "index_meta.json"
        return meta.stat().st_mtime if meta.exists() else None

    def get(self, kb_id: str) -> Index:
        if self._rebuilding(kb_id) and kb_id in self._staging:
            return self._staging[kb_id]
        return self._get_live(kb_id)

    def _get_live(self, kb_id: str) -> Index:
        with self.lock(kb_id):
            idx = self._indexes.get(kb_id)
            embedder = model_service.embedder()
            # 磁盘上的索引被别的进程（另一个 worker / CLI）改写过 → 丢弃内存缓存重新加载，避免用旧缓存覆盖新数据
            if idx is not None and idx.embedder is embedder and self._mtimes.get(kb_id) == self._disk_mtime(kb_id):
                return idx
            settings = settings_service.get()
            directory = self.directory(kb_id)
            llm = model_service.llm(aux=True) if settings.retrieval.graph.llm_extraction else None
            if Index.exists(directory):
                idx = Index.load(settings, directory, embedder=embedder, llm=llm, strict=False, image_embedder_factory=model_service.image_embedder)
            else:
                idx = Index(settings, embedder=embedder, directory=directory, llm=llm, image_embedder_factory=model_service.image_embedder)
            idx.aliases = self.aliases(kb_id)
            self._indexes[kb_id] = idx
            self._mtimes[kb_id] = self._disk_mtime(kb_id)
            return idx

    # --- 语料治理：知识库级的别名词典与剔除清单（存数据库，重建索引不丢）-----------------------
    @staticmethod
    def _setting(key: str, default=None):
        with SessionLocal() as db:
            return Repository(db).get_setting(key, default)

    def aliases(self, kb_id: str) -> AliasDict | None:
        groups = (self._setting(f"aliases:{kb_id}") or {}).get("groups") or []
        return AliasDict(groups) or None

    def excluded(self, kb_id: str) -> set[str]:
        """剔除清单的键（「doc_id:指纹」）。早期只按指纹记录的条目用其中的 docId 补成同样的形式。"""
        raw = (self._setting(f"governance:{kb_id}") or {}).get("excluded") or {}
        return {k if ":" in k else f"{v.get('docId')}:{k}" for k, v in raw.items()}

    def pipeline(self, kb_id: str) -> IngestPipeline:
        settings = settings_service.get()
        llm = model_service.llm(aux=True) if (settings.enrich.contextual_summary or settings.enrich.hypothetical_questions or settings.enrich.summary_index) else None
        vlm = model_service.vision_llm(aux=True) if (settings.multimodal.enabled and settings.multimodal.vlm_caption) else None
        idx = self.get(kb_id)
        return IngestPipeline(settings, index=idx, llm=llm, vlm=vlm, aliases=idx.aliases, excluded=self.excluded(kb_id))

    def save(self, kb_id: str, idx: Index | None = None) -> None:
        """idx：调用方刚改过的索引对象。入库途中保存策略配置 / 切换模型会清空缓存（invalidate），只按缓存保存就会静默跳过——
        文档标成 COMPLETED 却不在索引里。"""
        with self.lock(kb_id):
            if self._rebuilding(kb_id) and kb_id in self._staging:
                self._staging[kb_id].save(self._staging_dir(kb_id))
                return
            cached = self._indexes.get(kb_id)
            if idx is None:
                idx = cached
            if idx is not None:
                idx.save(self.directory(kb_id))
                if idx is cached:
                    self._mtimes[kb_id] = self._disk_mtime(kb_id)
                else:                                    # 缓存已失效：下次 get() 按新配置从磁盘重新加载（含这次写入）
                    self._indexes.pop(kb_id, None)
                    self._mtimes.pop(kb_id, None)
                    try:
                        self._note_embedder_change(kb_id, idx)
                    except Exception:
                        logger.exception("知识库 %s：检查入库途中是否切换了向量模型失败", kb_id)

    def _note_embedder_change(self, kb_id: str, idx: Index) -> None:
        """缓存在入库途中被清空，可能是切换了向量模型：那刚写入的就是旧模型的向量，得重建。光靠之后的自动检查不够——这份索引接下来
        会被新模型加载、index_meta 里的模型字段被改写，"向量模型变更"再也查不出来，新旧向量混在一起。所以记一笔待重建（持久化）；
        不在这里直接排队：切换模型触发的那次检查可能也正要给它排队，两边一撞就是两遍整库重建。"""
        from .reindex_service import reindex_service      # 延迟导入：reindex_service 依赖本模块

        if embedder_signature(idx.embedder) != embedder_signature(model_service.embedder()):
            reindex_service.defer(kb_id, "向量模型变更")

    def drop(self, kb_id: str) -> None:
        with self.lock(kb_id):
            self._indexes.pop(kb_id, None)
            self._mtimes.pop(kb_id, None)
            d = self.directory(kb_id)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            self._staging.pop(kb_id, None)
            for extra in (self._backup_dir(kb_id), self._staging_dir(kb_id), self._old_dir(kb_id)):
                shutil.rmtree(extra, ignore_errors=True)

    # --- 整库重建：在临时目录建新索引，建好后原子替换 ---------------------------------------------
    # 重建线程（document_service.reindex_kb 所在线程）的 get / pipeline / save 指向临时索引，其余线程（检索、问答、
    # 上传）照常读写正式索引——重建期间检索看到的始终是完整的旧索引，而不是正在逐份填充的半成品。
    # 被中断（进程重启 / 热重载）时正式索引从未被动过，删掉临时目录、重新排队即可。
    # Qdrant 的集合名取自目录名，临时目录会建到另一个集合上：这种后端退回"先备份、原地重建"。
    def _staging_dir(self, kb_id: str) -> Path:
        return app_config.index_root / f".{kb_id}.rebuild-staging"

    def _old_dir(self, kb_id: str) -> Path:
        return app_config.index_root / f".{kb_id}.rebuild-old"

    def _backup_dir(self, kb_id: str) -> Path:
        return app_config.index_root / f".{kb_id}.rebuild-backup"

    def _rebuilding(self, kb_id: str) -> bool:
        return getattr(self._tls, "rebuilding", None) == kb_id

    def target_dir(self, kb_id: str) -> Path:
        """当前线程写入的目录：重建线程是临时目录，其余线程是正式目录。"""
        if self._rebuilding(kb_id) and kb_id in self._staging:
            return self._staging_dir(kb_id)
        return self.directory(kb_id)

    def live(self, kb_id: str) -> Index:
        """正式索引（无视当前线程是否在重建）。"""
        return self._get_live(kb_id)

    def begin_rebuild(self, kb_id: str) -> None:
        with self._global:
            self._builds[kb_id] = None               # 先登记"已开始"，再读配置 / 模型 / 别名词典（见 rebuild_embedder）
        settings = settings_service.get()
        with self.lock(kb_id):
            self._tls.rebuilding = kb_id
            if settings.vector_store.backend == "qdrant":
                # 原地重建：旧索引挪到备份；已有备份说明上次被中断过，那份才是最后一个完整索引
                self._indexes.pop(kb_id, None)
                self._mtimes.pop(kb_id, None)
                d, bak = self.directory(kb_id), self._backup_dir(kb_id)
                if bak.exists():
                    shutil.rmtree(d, ignore_errors=True)
                elif d.exists():
                    d.rename(bak)
                return
            staging = self._staging_dir(kb_id)
            shutil.rmtree(staging, ignore_errors=True)
            llm = model_service.llm(aux=True) if settings.retrieval.graph.llm_extraction else None
            embedder = model_service.embedder()
            idx = Index(settings, embedder=embedder, directory=staging, llm=llm, image_embedder_factory=model_service.image_embedder)
            idx.aliases = self.aliases(kb_id)
            self._staging[kb_id] = idx
            with self._global:
                self._builds[kb_id] = embedder_signature(embedder)

    def rebuild_embedder(self, kb_id: str) -> tuple[bool, tuple[str, str | None] | None]:
        """（整库重建是否已开始，它所用向量模型的签名）。自动重建据此判断重建途中又来的变更要不要跑完再来一遍。
        begin_rebuild 一进来就登记"已开始"，之后才读配置 / 模型 / 别名词典：还没登记，说明此前落库的变更它都用得上。
        临时目录里的新索引从头到尾只用一个模型，定下来就记上签名。替换完成后记录仍保留——任务此刻还没标记完成，正式索引
        已经是它建的——直到 forget_rebuild 或下一次 begin_rebuild。
        签名为 None：模型还没定下来，或是原地重建（Qdrant：逐份文档按当时的模型写入，可能新旧混杂），只能按"可能过期"处理。"""
        with self._global:
            return kb_id in self._builds, self._builds.get(kb_id)

    def forget_rebuild(self, kb_id: str) -> None:
        with self._global:
            self._builds.pop(kb_id, None)

    def finish_rebuild(self, kb_id: str) -> None:
        """新索引替换正式索引。两次目录改名之间被打断由 recover_interrupted 收拾。
        加锁顺序：swap_guard（读写锁）→ lock（知识库锁）；调用方已持有 swap_guard 时这里是重入。"""
        with self.swap_guard(kb_id), self.lock(kb_id):
            try:
                idx = self._staging.pop(kb_id, None)
                if idx is None:                                   # Qdrant：原地重建，删掉备份即可
                    shutil.rmtree(self._backup_dir(kb_id), ignore_errors=True)
                    return
                staging, live, old = self._staging_dir(kb_id), self.directory(kb_id), self._old_dir(kb_id)
                if not staging.exists():                          # 一份文档都没写入（知识库里没有可重建的文档）
                    idx.save(staging)
                shutil.rmtree(old, ignore_errors=True)
                if live.exists():                                 # 已持 swap_guard：没有在途检索，替换只是两次改名
                    live.rename(old)
                staging.rename(live)
                self._indexes.pop(kb_id, None)
                self._mtimes.pop(kb_id, None)
                _forget_lance(live, staging, old)
                shutil.rmtree(old, ignore_errors=True)
            finally:
                self._tls.rebuilding = None

    def abort_rebuild(self, kb_id: str) -> bool:
        """放弃本次重建，保留最后一份完整索引。返回是否有可保留的索引（重建前本来就没有索引时为 False）。"""
        with self.lock(kb_id):
            if getattr(self._tls, "rebuilding", None) == kb_id:
                self._tls.rebuilding = None
            if self._staging.pop(kb_id, None) is not None or self._staging_dir(kb_id).exists():
                shutil.rmtree(self._staging_dir(kb_id), ignore_errors=True)
                return self.directory(kb_id).exists()
            bak = self._backup_dir(kb_id)
            if not bak.exists():
                return False
            d = self.directory(kb_id)
            shutil.rmtree(d, ignore_errors=True)
            bak.rename(d)
            self._indexes.pop(kb_id, None)
            self._mtimes.pop(kb_id, None)
            return True

    def recover_interrupted(self, skip: set[str] | None = None) -> list[str]:
        """收拾被中断的重建，返回这些知识库 id，由调用方重新排队。
        - 临时目录：正式索引没动过，删掉即可；
        - 替换做到一半（正式目录已挪成 .rebuild-old、新目录还没就位）：挪回去；
        - 原地重建的备份（Qdrant）：用备份覆盖残缺的正式目录。
        skip：本进程里正在重建的知识库，它们的临时目录 / 备份是有效的进行中状态，不能动。"""
        root = app_config.index_root
        if not root.exists():
            return []
        out: list[str] = []
        for p in sorted(root.glob(".*.rebuild-*")):
            kb_id, _, kind = p.name[1:].rpartition(".rebuild-")
            if (skip and kb_id in skip) or kb_id in out:
                continue
            with self.lock(kb_id):
                live = self.directory(kb_id)
                old = self._old_dir(kb_id)
                if old.exists():
                    if live.exists():
                        shutil.rmtree(old, ignore_errors=True)       # 替换其实已完成
                    else:
                        old.rename(live)
                shutil.rmtree(self._staging_dir(kb_id), ignore_errors=True)
                self._staging.pop(kb_id, None)
            if self._backup_dir(kb_id).exists():
                self.abort_rebuild(kb_id)
            self._indexes.pop(kb_id, None)
            self._mtimes.pop(kb_id, None)
            logger.warning("知识库 %s 上次整库重建被中断（%s）：正式索引保持重建前的版本，稍后重新重建", kb_id, kind)
            out.append(kb_id)
        return out

    def indexed_docs(self, kb_id: str) -> int:
        meta = Index.read_meta(self.directory(kb_id)) or {}
        return int(meta.get("docs") or 0)

    def needs_reindex(self, kb_id: str) -> bool:
        meta = Index.read_meta(self.directory(kb_id))
        if not meta or not meta.get("chunks"):
            return False
        return (meta.get("embedder"), meta.get("model")) != embedder_signature(model_service.embedder())

    def invalidate(self) -> None:
        with self._global:
            self._indexes = {}
            self._mtimes = {}

    def stats(self, kb_id: str) -> dict:
        meta = Index.read_meta(self.directory(kb_id)) or {}
        if kb_id in self._indexes:
            return {**meta, **self._indexes[kb_id].stats(), "needsReindex": self.needs_reindex(kb_id)}
        return {**meta, "needsReindex": self.needs_reindex(kb_id)}


index_service = IndexService()
