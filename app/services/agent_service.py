"""多智能体协作服务：AgentTeam + LanceDB 黑板，SSE 推送每一步。"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from typing import Any

from prism.agents import AgentTeam, Blackboard

from .memory_service import lance_dir, memory_service
from .model_service import model_service
from .retrieval_service import retrieval_service
from .settings_service import settings_service

logger = logging.getLogger(__name__)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class AgentService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._board: tuple[tuple, Blackboard] | None = None

    def board(self) -> Blackboard:
        emb = model_service.embedder()
        key = (str(lance_dir()), emb.name, getattr(emb, "model_name", None), emb.dim)
        with self._lock:
            if self._board is None or self._board[0] != key:
                self._board = (key, Blackboard(lance_dir(), emb))
            return self._board[1]

    def team(self, kb_id: str, user_id: str) -> AgentTeam:
        settings = settings_service.get()
        retriever = retrieval_service.retriever(kb_id)
        names = retrieval_service.doc_names(kb_id)

        def search(query: str, top_k: int) -> list[dict[str, Any]]:
            # 子问题已经由规划者拆好：研究员直接检索，不再做查询改写（省掉每个子问题一次 LLM 调用）
            res = retriever.retrieve(query, top_k=top_k, use_transform=False)
            return [{"chunkId": c.chunk.chunk_id, "docId": c.chunk.doc_id, "fileName": names.get(c.chunk.doc_id), "label": c.citation_label(),
                     "text": c.text_for_llm().strip()[:900], "score": round(float(c.score), 4)} for c in res.contexts]

        def recall(query: str) -> list[str]:
            return [m["content"] for m in memory_service.recall(user_id, query)]

        return AgentTeam(self.board(), search, llm=model_service.llm(), fast_llm=model_service.llm(aux=True), cfg=settings.agents,
                         memory_recall=recall if settings.memory.enabled else None)

    def stream(self, kb_id: str, question: str, user_id: str) -> Iterator[str]:
        try:
            team = self.team(kb_id, user_id)
            for ev in team.run(question, user_id=user_id, kb_id=kb_id):
                yield _sse(ev["type"], ev)
        except Exception as exc:
            logger.exception("多智能体运行失败")
            yield _sse("error", {"type": "error", "agent": "system", "content": f"{type(exc).__name__}: {exc}", "data": {}, "ms": 0})


agent_service = AgentService()
