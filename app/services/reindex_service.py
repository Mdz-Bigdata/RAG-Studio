"""自动重建索引：配置或向量模型变了，就在后台把受影响的知识库重建，不需要用户手动点。

两种"索引过期"：
1. 入库参数变了（解析 / 清洗 / 切片 / 索引增强 / BM25）——用入库指纹判断，影响所有有文档的知识库；
2. 向量模型变了——索引记录的 embedder 与当前不一致，只影响对应知识库。

触发点：保存 / 重置策略配置、激活 / 停用 / 删除向量模型、服务启动。
单线程串行执行（解析与向量化都吃资源），状态存在内存里供前端轮询。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import deque
from typing import Any

from ..db import Repository, SessionLocal
from .document_service import document_service
from .index_service import embedder_signature, index_service
from .model_service import model_service
from .settings_service import settings_service

logger = logging.getLogger(__name__)
FINGERPRINT_KEY = "ingest_fingerprint"
PENDING_KEY = "reindex_pending"     # {kb_id: reason}：排队 / 进行中的重建，完成才移除；进程重启后据此继续
MODEL_CHANGED = "向量模型变更"


def ingest_fingerprint() -> str:
    """影响"已入库文档长什么样"的全部参数。检索期参数（融合 / 精排 / 压缩…）不在其中，改了立即生效。"""
    s = settings_service.get()
    parts: dict[str, Any] = {
        # mode / cloud 是后加的字段：本机解析时不进指纹（否则升级后所有索引都被判过期），见下面的 mineru_remote
        "parsing": s.parsing.model_dump(mode="json", exclude={"mineru": {"mode", "cloud"}}),
        "cleaning": s.cleaning.model_dump(mode="json"),
        "chunking": s.chunking.model_dump(mode="json"),
        "enrich": s.enrich.model_dump(mode="json"),
        "sparse": s.sparse.model_dump(mode="json"),
        # 向量库后端与 HNSW / 量化是建索引期参数；search（降维 / 粗筛量化 / ef_search）是查询期参数，不触发重建
        "vector_store": s.vector_store.model_dump(mode="json", exclude={"search"}),
        "graph_llm_extraction": s.retrieval.graph.llm_extraction,
        "graph_unweaver": s.retrieval.graph.unweaver_enabled,
    }
    if s.parsing.mineru.resolved_mode == "cloud":
        # 换成云端解析（或换了云端地址）结果会变，要重建；换密钥不用
        parts["mineru_remote"] = ["cloud", s.parsing.mineru.cloud.api_url]
    if s.corpus.keywords:
        # 只在开启时加入：新增配置段不应让所有已有索引被判为过期
        parts["corpus_keywords"] = s.corpus.keywords_per_chunk
    return hashlib.sha1(json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


class ReindexService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: deque[str] = deque()
        self._jobs: dict[str, dict[str, Any]] = {}     # kb_id -> {name, status, reason, error, startedAt, finishedAt}
        self._worker: threading.Thread | None = None
        self._checking = False
        self._recovered: list[str] = []

    _pending_lock = threading.Lock()      # 待重建记录的读-改-写串行（工作线程清除 / 请求线程添加会互相覆盖）

    def _mark_pending(self, kb_id: str, reason: str | None) -> None:
        with self._pending_lock, SessionLocal() as db:
            repo = Repository(db)
            pending = repo.get_setting(PENDING_KEY) or {}
            if reason is None:
                if kb_id not in pending:
                    return
                pending.pop(kb_id)
            else:
                pending.setdefault(kb_id, reason)
            repo.set_setting(PENDING_KEY, pending)

    # --- 对外 -------------------------------------------------------------------------
    def fingerprint_changed(self) -> bool:
        with SessionLocal() as db:
            stored = Repository(db).get_setting(FINGERPRINT_KEY)
        return stored is not None and stored != ingest_fingerprint()

    def request(self, reason: str) -> None:
        """异步检查哪些知识库过期并排队重建（检查本身可能要加载向量模型，所以不放在请求线程里）。
        被中断的重建先同步恢复成重建前的索引——只是目录改名，很快，且要赶在第一个查询之前。"""
        with self._lock:
            active = {k for k, j in self._jobs.items() if j["status"] in ("queued", "running")}
        try:
            self._recovered = index_service.recover_interrupted(skip=active)
        except Exception:
            logger.exception("恢复被中断的重建失败")
            self._recovered = []
        with self._lock:
            self._checking = True
        threading.Thread(target=self._check_and_enqueue, args=(reason,), daemon=True, name="reindex-check").start()

    def enqueue(self, kb_id: str, name: str, reason: str) -> None:
        self._mark_pending(kb_id, reason)
        current = embedder_signature(model_service.embedder()) if reason == MODEL_CHANGED else None   # 可能要加载模型：不在锁里取
        with self._lock:
            job = self._jobs.get(kb_id)
            if job and job["status"] in ("queued", "running"):
                if job["status"] == "running":       # 排队中的还没开始，自然用上这次的变更
                    job["rerun"] = job["rerun"] or self._outdated_by(kb_id, reason, current)   # 跑完再来一遍；已定的不撤销
                return
            self._jobs[kb_id] = {"kbId": kb_id, "name": name, "status": "queued", "reason": reason, "error": None, "failedDocs": [], "startedAt": None, "finishedAt": None, "rerun": False}
            self._queue.append(kb_id)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, daemon=True, name="reindex-worker")
                self._worker.start()

    def defer(self, kb_id: str, reason: str) -> None:
        """记一笔待重建（持久化）但不马上排队：下一次自动检查（启动 / 保存配置 / 切换模型）按"上次重建被中断"重建它；
        在那之前这个知识库若正常重建完成，记录随之清除。"""
        self._mark_pending(kb_id, reason)

    def is_active(self, kb_id: str) -> bool:
        job = self._jobs.get(kb_id)
        return bool(job and job["status"] in ("queued", "running"))

    def status(self) -> dict[str, Any]:
        with self._lock:
            jobs = [dict(j) for j in self._jobs.values()]
            checking = self._checking
        active = [j for j in jobs if j["status"] in ("queued", "running")]
        return {
            "busy": checking or bool(active),
            "checking": checking,
            "running": [j for j in jobs if j["status"] == "running"],
            "queued": [j for j in jobs if j["status"] == "queued"],
            "recent": sorted((j for j in jobs if j["status"] in ("done", "failed")), key=lambda j: j["finishedAt"] or 0, reverse=True)[:5],
        }

    # --- 内部 -------------------------------------------------------------------------
    @staticmethod
    def _outdated_by(kb_id: str, reason: str, current: tuple[str, str | None] | None) -> bool:
        """正在跑的重建遇上新的变更：它跑完是否仍然过期（是就再来一遍）。current：当前向量模型的签名（换模型时才有）。
        - 还没到 begin_rebuild：配置 / 模型 / 别名词典都还没读，会用上这次的变更；
        - 已开始（含已替换、任务还没收尾）：换模型只看它用的是不是当前模型——替换之前正式索引一直记着旧模型，光看那个，
          重建途中任何一次检查（保存任意配置都会触发）都会被当成"模型变了"，整库再重建一遍；签名未知（还没定下来 /
          Qdrant 原地重建）按过期算。其余原因（入库参数 / 别名词典 / 手动重建……）一律再来一遍。"""
        started, built = index_service.rebuild_embedder(kb_id)
        if not started:
            return False
        if reason != MODEL_CHANGED:
            return True
        return built is None or built != current

    def _running_build_stale(self, kb_id: str) -> bool:
        """正在跑的重建用的已不是当前模型，正式索引却看不出来：重建途中模型又换回了正式索引记着的那个（A→B→A），
        needs_reindex 说不用重建，可这次重建一替换，正式索引就成了 B。签名未知的交给 needs_reindex（原地重建写的就是正式目录）。"""
        with self._lock:
            job = self._jobs.get(kb_id)
            if not job or job["status"] != "running":
                return False
        started, built = index_service.rebuild_embedder(kb_id)
        return started and built is not None and built != embedder_signature(model_service.embedder())

    def _check_and_enqueue(self, reason: str) -> None:
        try:
            current = ingest_fingerprint()
            with SessionLocal() as db:
                repo = Repository(db)
                stored = repo.get_setting(FINGERPRINT_KEY)
                pending = repo.get_setting(PENDING_KEY) or {}
                kbs = []
                for kb in repo.list_kbs():
                    docs = repo.list_documents(kb.id)
                    kbs.append((kb.id, kb.name, len(docs), sum(1 for d in docs if d.status == "COMPLETED")))
            config_changed = stored is not None and stored != current
            with self._lock:
                active = {k for k, j in self._jobs.items() if j["status"] in ("queued", "running")}
            # 待重建记录里本进程正在跑的不算"中断"：否则重建期间任何一次配置保存都会再排一次完整重建
            interrupted = (set(pending) | set(self._recovered)) - active
            for kb_id, name, n_docs, n_completed in kbs:
                if n_docs == 0:
                    if kb_id in pending:
                        self._mark_pending(kb_id, None)
                    continue
                if kb_id in interrupted:
                    self.enqueue(kb_id, name, "上次重建被中断，继续重建")
                elif config_changed:
                    self.enqueue(kb_id, name, "入库参数变更")
                elif index_service.needs_reindex(kb_id) or self._running_build_stale(kb_id):
                    self.enqueue(kb_id, name, MODEL_CHANGED)
                elif not self.is_active(kb_id) and index_service.indexed_docs(kb_id) < n_completed:
                    self._enqueue_if_incomplete(kb_id, name)     # 快照里少了文档：可能只是在途入库让计数错开，核对后再定
            for kb_id in set(pending) - {k for k, *_ in kbs}:      # 知识库已删除
                self._mark_pending(kb_id, None)
            # 新指纹最后才写：此前受影响的知识库都已记入 PENDING_KEY，中途进程退出下次启动仍会判为配置变更
            with SessionLocal() as db:
                Repository(db).set_setting(FINGERPRINT_KEY, current)
            self._recovered = []
        except Exception:
            logger.exception("自动重建索引检查失败（%s）", reason)
        finally:
            with self._lock:
                self._checking = False

    def _enqueue_if_incomplete(self, kb_id: str, name: str, attempt: int = 0) -> None:
        """索引里的文档数少于已完成的文档数 → 排队整库重建。
        两个数一个来自数据库、一个来自索引文件，不是同一时刻读的：中间有入库 / 删除（例如保存配置时用户正在上传）就可能错开，
        把好好的知识库判成"索引不完整"、整库重建（文档全部回到处理中）。所以只在两次读之间本进程没有写过这个知识库时下结论；
        否则等它的在途写入都结束再核对——真缺文档照样重建，只是晚一点。"""
        if self.is_active(kb_id):
            return
        before = document_service.index_writes(kb_id)
        settled = before[0] == before[1]                 # 有进行中的写入就不必读了，直接等它结束
        if settled:
            with SessionLocal() as db:
                completed = sum(1 for d in Repository(db).list_documents(kb_id) if d.status == "COMPLETED")
            indexed = index_service.indexed_docs(kb_id)
            settled = document_service.index_writes(kb_id) == before
        if not settled:
            if attempt < 3:
                logger.info("知识库 %s 的索引文档数少于已完成文档数，但期间有在途入库 / 删除：结束后再核对", name)
                document_service.when_idle(kb_id, lambda: self._enqueue_if_incomplete(kb_id, name, attempt + 1))
            else:
                logger.info("知识库 %s：连续 %d 次核对都碰上在途入库 / 删除，留给下次检查", name, attempt + 1)
            return
        if indexed < completed:
            self.enqueue(kb_id, name, f"索引不完整（{indexed}/{completed} 份文档）")

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self._worker = None      # 线程退出前就让出位置：此后的 enqueue 会启动新的工作线程，任务不会卡在"排队中"
                    return
                kb_id = self._queue.popleft()
                job = self._jobs[kb_id]
                job.update(status="running", startedAt=time.time(), rerun=False)
                index_service.forget_rebuild(kb_id)      # 上一次重建的记录作废：这次 begin_rebuild 之前的变更它都用得上
            logger.info("自动重建索引：%s（%s）", job["name"], job["reason"])
            failed_docs: list[str] = []
            try:
                failed_docs = document_service.reindex_kb(kb_id) or []
                error = None
            except Exception as exc:
                logger.exception("重建索引失败 %s", kb_id)
                error = f"{type(exc).__name__}: {exc}"[:300]
            if failed_docs:
                logger.warning("重建索引 %s 完成，但 %d 份文档失败：%s", job["name"], len(failed_docs), "、".join(failed_docs))
            with self._lock:
                rerun = job.get("rerun")
                job.update(status="failed" if error else "done", error=error, failedDocs=failed_docs, finishedAt=time.time())
            if not error and not rerun:
                self._mark_pending(kb_id, None)
            if rerun and not error:
                self.enqueue(kb_id, job["name"], job["reason"])


reindex_service = ReindexService()
