"""个性化推荐服务：知识库 → 推荐条目（文档 / 章节 / 生成的问题 / 图片），对话与反馈 → 交互事件。"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any

import numpy as np

from prism.recommend import LanceRecommender, RecItem

from .index_service import index_service
from .memory_service import lance_dir, memory_service
from .model_service import model_service
from .settings_service import settings_service

logger = logging.getLogger(__name__)


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-9 else v.astype(np.float32)


class RecommendService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine: tuple[tuple, LanceRecommender] | None = None

    def engine(self) -> LanceRecommender:
        emb = model_service.embedder()
        cfg = settings_service.get().recommend
        key = (str(lance_dir()), emb.name, getattr(emb, "model_name", None), emb.dim, cfg.model_dump_json())
        with self._lock:
            if self._engine is None or self._engine[0] != key:
                self._engine = (key, LanceRecommender(lance_dir(), emb, cfg))
            return self._engine[1]

    # --- 知识库 → 推荐条目 --------------------------------------------------------------
    def sync_kb(self, kb_id: str) -> dict[str, Any]:
        """文档 / 章节（parent）/ 生成的问题 / 图片 → rec_items。章节向量 = 其 child 向量的均值（不重新调用模型）。"""
        idx = index_service.get(kb_id)
        store = idx.store
        items: list[RecItem] = []
        vecs: list[np.ndarray] = []
        questions: list[tuple[RecItem, str]] = []
        doc_vecs: dict[str, list[np.ndarray]] = {}
        parents = [c for c in store.chunks.values() if c.kind == "parent"]
        child_vecs = idx.vectors.get_vectors([cid for p in parents for cid in p.child_ids])
        for p in parents:
            cv = [child_vecs[c] for c in p.child_ids if c in child_vecs]
            if not cv:
                continue
            v = _norm(np.mean(cv, axis=0))
            title = p.breadcrumb_text or (store.docs.get(p.doc_id).title if store.docs.get(p.doc_id) else "")
            items.append(RecItem(f"{kb_id}:sec:{p.chunk_id}", kb_id, "section", p.chunk_id, title, (p.summary or p.text)[:300], list(p.metadata.get("tags") or [])))
            vecs.append(v)
            doc_vecs.setdefault(p.doc_id, []).append(v)
            for q in p.hypothetical_questions[:3]:
                qid = hashlib.blake2b(q.encode("utf-8"), digest_size=8).hexdigest()
                questions.append((RecItem(f"{kb_id}:q:{qid}", kb_id, "question", p.chunk_id, q, title), q))
        for doc_id, dv in doc_vecs.items():
            meta = store.docs.get(doc_id)
            items.append(RecItem(f"{kb_id}:doc:{doc_id}", kb_id, "doc", doc_id, (meta.title if meta else "") or doc_id, "", list(getattr(meta, "tags", []) or [])))
            vecs.append(_norm(np.mean(dv, axis=0)))
        if questions:
            qv = model_service.embedder().embed_queries([q for _, q in questions])
            for (it, _), v in zip(questions, qv):
                items.append(it)
                vecs.append(np.asarray(v, dtype=np.float32))
        recs = list(idx.images.records.values())
        if recs:
            dv = idx.images.desc.get_vectors([r.image_id for r in recs])
            for r in recs:
                if r.image_id in dv:
                    items.append(RecItem(f"{kb_id}:img:{r.image_id}", kb_id, "image", r.image_id, r.caption or r.label(), r.brief(200)))
                    vecs.append(np.asarray(dv[r.image_id], dtype=np.float32))
        eng = self.engine()
        n = eng.upsert_items(items, np.vstack(vecs)) if items else 0
        removed = eng.remove_items(kb_id=kb_id, keep_ids={it.id for it in items})
        by_kind: dict[str, int] = {}
        for it in items:
            by_kind[it.kind] = by_kind.get(it.kind, 0) + 1
        return {"kbId": kb_id, "items": n, "removed": removed, "byKind": by_kind}

    # --- 推荐 / 交互 -------------------------------------------------------------------
    def recommend(self, user_id: str, *, kb_id: str | None = None, kinds: list[str] | None = None, k: int = 8, use_memory: bool = True) -> dict[str, Any]:
        extra: list[np.ndarray] = []
        if use_memory and settings_service.get().memory.enabled:
            try:
                extra = memory_service.store().vectors(user_id=user_id, kinds=["preference", "profile"])
            except Exception as exc:
                logger.info("读取记忆偏好失败：%s", exc)
        out = self.engine().recommend(user_id, kb_id=kb_id, kinds=kinds, k=k, extra_interest=extra)
        out["memoryInterests"] = len(extra)
        return out

    def log_chat(self, user_id: str, kb_id: str | None, question: str, cited_chunk_ids: list[str]) -> None:
        """对话里的提问记为 ask，回答引用到的章节记为 cite（后台执行，失败不影响对话）。"""
        cfg = settings_service.get().recommend
        if not cfg.enabled or not user_id or not question.strip():
            return

        def work() -> None:
            try:
                eng = self.engine()
                eng.log_event(user_id, kind="ask", text=question, kb_id=kb_id or "")
                if kb_id:
                    ids = [f"{kb_id}:sec:{cid}" for cid in cited_chunk_ids]
                    for item_id in eng.get_items(ids):
                        eng.log_event(user_id, kind="cite", item_id=item_id, kb_id=kb_id)
            except Exception as exc:
                logger.info("记录推荐交互失败：%s", exc)

        threading.Thread(target=work, daemon=True, name="rec-log").start()


recommend_service = RecommendService()
