"""长期记忆服务：LanceDB 记忆库（data/lancedb/memories）+ 对话召回 / 抽取。"""

from __future__ import annotations

import logging
import threading
from typing import Any

from prism.memory import LanceMemoryStore, MemoryRecord, extract_memories

from ..config import app_config
from .model_service import model_service
from .settings_service import settings_service

logger = logging.getLogger(__name__)

def lance_dir():
    """全局 LanceDB 目录（每次取：测试会在运行时改 app_config.data_dir）。"""
    return app_config.data_dir / "lancedb"


LANCE_DIR = lance_dir()   # 兼容旧导入（lake_service 等）；新代码请用 lance_dir()，测试隔离数据目录时才会跟着变


def public(rec: MemoryRecord) -> dict[str, Any]:
    return {"id": rec.id, "content": rec.content, "userId": rec.user_id, "agentId": rec.agent_id, "kind": rec.kind, "importance": rec.importance,
            "source": rec.source, "sessionId": rec.session_id, "tags": rec.tags, "createdAt": rec.created_at, "updatedAt": rec.updated_at,
            "lastAccess": rec.last_access, "accessCount": rec.access_count, "expiresAt": rec.expires_at}


class MemoryService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: tuple[tuple, LanceMemoryStore] | None = None

    def store(self) -> LanceMemoryStore:
        emb = model_service.embedder()
        cfg = settings_service.get().memory
        key = (str(lance_dir()), emb.name, getattr(emb, "model_name", None), emb.dim, cfg.model_dump_json())
        with self._lock:
            if self._store is None or self._store[0] != key:
                self._store = (key, LanceMemoryStore(lance_dir(), emb, cfg))
            return self._store[1]

    # --- 对话集成 ---------------------------------------------------------------------
    def recall(self, user_id: str, query: str, *, agent_id: str = "assistant") -> list[dict[str, Any]]:
        """回答前召回：返回 [{id, content, kind, score, similarity, recency}]；记忆关闭或失败时返回空。"""
        if not settings_service.get().memory.enabled or not user_id or not query.strip():
            return []
        try:
            hits = self.store().search(query, user_id=user_id, agent_id=agent_id)
        except Exception as exc:
            logger.warning("记忆召回失败：%s", exc)
            return []
        return [{"id": h["record"].id, "content": h["record"].content, "kind": h["record"].kind, "score": round(h["score"], 4),
                 "similarity": round(h["similarity"], 4), "recency": round(h["recency"], 4), "importance": h["importance"]} for h in hits]

    def extract_async(self, user_id: str, user_text: str, answer: str = "", *, session_id: str = "") -> None:
        """回答结束后在后台抽取记忆（不拖慢回答）。"""
        cfg = settings_service.get().memory
        if not cfg.enabled or cfg.extract == "off" or not user_id or not user_text.strip():
            return
        threading.Thread(target=self._extract, args=(user_id, user_text, answer, session_id), daemon=True, name="memory-extract").start()

    def _extract(self, user_id: str, user_text: str, answer: str, session_id: str) -> list[dict[str, Any]]:
        cfg = settings_service.get().memory
        try:
            cands = extract_memories(user_text, answer, llm=model_service.llm(aux=True), mode=cfg.extract)
            out = []
            for c in cands:
                rec, how = self.store().add(c.content, user_id=user_id, kind=c.kind, importance=c.importance, source="chat", session_id=session_id)
                out.append({**public(rec), "action": how})
            if out:
                logger.info("记住了 %d 条关于 %s 的信息", len(out), user_id)
            return out
        except Exception as exc:
            logger.warning("记忆抽取失败：%s", exc)
            return []

    def extract_now(self, user_id: str, user_text: str, answer: str = "", *, session_id: str = "") -> list[dict[str, Any]]:
        return self._extract(user_id, user_text, answer, session_id)


memory_service = MemoryService()
