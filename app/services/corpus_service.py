"""语料治理服务：质量体检与剔除、别名词典、覆盖缺口 FAQ 增强、自动元数据。

知识库级的治理状态都存在 app_settings 里（重建索引会删掉索引目录，但不能丢掉人工治理的成果）：
    governance:<kb_id>   {"excluded": {指纹: {reason, docId, breadcrumb, preview, at}}}
    aliases:<kb_id>      {"groups": [[词, 词, ...], ...], "updatedAt"}
    augment:<kb_id>      {"items": [...], "docId": 增强文档 id, "publishedAt"}
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from prism.corpus import AliasDict, CorpusHealthChecker, build_faq_markdown, discover_aliases, draft_faq, exclusion_key, extract_metadata
from prism.corpus.metadata import FIELDS

from ..db import Document, Repository, SessionLocal
from .document_service import _doc_metadata, _loads, document_service
from .eval_service import eval_service
from .index_service import index_service
from .model_service import model_service
from .retrieval_service import retrieval_service
from .settings_service import settings_service

logger = logging.getLogger(__name__)

FAQ_FILE_NAME = "增强语料-FAQ.md"
ITEM_STATUSES = ("pending", "approved", "rejected", "gap")


def _norm(v: Any) -> str:
    if isinstance(v, list):
        v = ",".join(map(str, v))
    return re.sub(r"[\s:：_\-]+", "", str(v if v is not None else "")).lower()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class CorpusService:
    def __init__(self) -> None:
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    @contextmanager
    def _state(self, key: str):
        """app_settings 里一条治理状态的读-改-写串行化：两次快速点击 / 生成草稿期间的审核不能互相覆盖。
        加锁顺序：这把锁 → index_service.lock（不能反过来）。"""
        with self._locks_guard:
            lock = self._locks.setdefault(key, threading.RLock())
        with lock:
            yield

    @staticmethod
    def _excluded_map(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """剔除清单按「doc_id:指纹」索引；早期只按指纹记录的条目用其中的 docId 补成同样的键。"""
        raw = state.get("excluded") or {}
        return {k if ":" in k else f"{v.get('docId')}:{k}": v for k, v in raw.items()}

    # --- 公共 ---------------------------------------------------------------------------
    @staticmethod
    def _get(key: str, default: Any = None) -> Any:
        with SessionLocal() as db:
            return Repository(db).get_setting(key, default)

    @staticmethod
    def _set(key: str, value: Any) -> None:
        with SessionLocal() as db:
            Repository(db).set_setting(key, value)

    @staticmethod
    def _kb_name(kb_id: str) -> str:
        with SessionLocal() as db:
            kb = Repository(db).get_kb(kb_id)
            return kb.name if kb else kb_id

    def _reingest(self, doc_id: str) -> None:
        """按文档来源重新入库：人工校对过的 / 没有源文件的用保存的 Markdown，其余重新解析源文件。"""
        with SessionLocal() as db:
            d = Repository(db).get_document(doc_id)
            if d is None:
                return
            edited, path, md = "edited" in (d.parser or ""), d.file_path, d.markdown_content
        if (edited or not (path and Path(path).exists())) and md:
            document_service.update_markdown(doc_id, md)
        elif path and Path(path).exists():
            document_service.process(doc_id)

    def _sync_counts(self, kb_id: str, doc_ids: set[str]) -> None:
        idx = index_service.get(kb_id)
        with SessionLocal() as db:
            repo = Repository(db)
            for doc_id in doc_ids:
                d = repo.get_document(doc_id)
                if d is not None:
                    chunks = [c for c in idx.store.chunks.values() if c.doc_id == doc_id]
                    repo.update_document(d, chunk_count=sum(1 for c in chunks if c.kind == "child"), parent_count=sum(1 for c in chunks if c.kind == "parent"))

    def _refresh_index_meta(self, kb_id: str, doc_ids: list[str]) -> None:
        """元数据改了只需要更新索引里的 DocMetadata（过滤用），不必重新向量化。"""
        with SessionLocal() as db:
            repo = Repository(db)
            fresh = {d.id: _doc_metadata(d) for d in (repo.get_document(x) for x in doc_ids) if d is not None}
        with index_service.lock(kb_id):
            idx = index_service.get(kb_id)
            for doc_id, md in fresh.items():
                cur = idx.store.docs.get(doc_id)
                if cur is None:
                    continue
                for key in ("title", "version", "valid_from", "valid_to", "department", "tags", "access_level", "allowed_departments"):
                    setattr(cur, key, getattr(md, key))
            index_service.save(kb_id, idx)

    # --- 质量体检 ------------------------------------------------------------------------
    def health(self, kb_id: str) -> dict[str, Any]:
        cfg = settings_service.get().corpus
        idx = index_service.get(kb_id)
        names = retrieval_service.doc_names(kb_id)
        report = CorpusHealthChecker(idx, near_dup_threshold=cfg.near_dup_threshold, garbled_threshold=cfg.garbled_threshold).run()
        faq_id = (self._get(f"augment:{kb_id}") or {}).get("docId")
        for d in report["docs"]:
            d["fileName"] = names.get(d["docId"])
            d["synthetic"] = d["docId"] == faq_id
        for c in report["chunks"]:
            c["fileName"] = names.get(c["docId"])
        excluded = self._excluded_map(self._get(f"governance:{kb_id}") or {})
        # "fingerprint" 字段是剔除清单的键（doc_id:指纹），恢复时原样传回
        report["excluded"] = [{**v, "fingerprint": key, "fileName": names.get(v.get("docId"))} for key, v in sorted(excluded.items(), key=lambda kv: kv[1].get("at") or "", reverse=True)]
        report["totals"]["excluded"] = len(excluded)
        return report

    def exclude(self, kb_id: str, parent_ids: list[str], reason: str = "") -> dict[str, Any]:
        key = f"governance:{kb_id}"
        with self._state(key):
            state = self._get(key) or {}
            excluded = self._excluded_map(state)
            with index_service.lock(kb_id):
                idx = index_service.get(kb_id)
                targets = [p for p in (idx.store.get(pid) for pid in parent_ids) if p is not None and p.kind == "parent"]
                for p in targets:
                    excluded[exclusion_key(p.doc_id, p.text)] = {"reason": reason, "docId": p.doc_id, "breadcrumb": p.breadcrumb_text, "preview": p.text[:220], "at": _now()}
                removed = idx.remove_chunks({p.chunk_id for p in targets})
                index_service.save(kb_id, idx)
            self._set(key, {**state, "excluded": excluded})
        self._sync_counts(kb_id, {p.doc_id for p in targets})
        return {"parents": len(targets), "chunks": len(removed)}

    def restore(self, kb_id: str, fingerprints: list[str]) -> list[str]:
        """撤销剔除：从清单里移除，受影响的文档重新入库（后台）。fingerprints 是清单的键（doc_id:指纹）。
        返回需要重新入库的文档 id。"""
        key = f"governance:{kb_id}"
        with self._state(key):
            state = self._get(key) or {}
            excluded = self._excluded_map(state)
            doc_ids = {excluded[k].get("docId") for k in fingerprints if k in excluded}
            for k in fingerprints:
                excluded.pop(k, None)
            self._set(key, {**state, "excluded": excluded})
        return [d for d in doc_ids if d]

    def reingest_many(self, doc_ids: list[str]) -> None:
        for doc_id in doc_ids:
            try:
                self._reingest(doc_id)
            except Exception:
                logger.exception("重新入库失败 %s", doc_id)

    def retire(self, kb_id: str, doc_ids: list[str], valid_to: str | None = None) -> str:
        """给旧版本设置失效日期（默认昨天）：检索时自动过滤，不删除文档。"""
        day = valid_to or (date.today() - timedelta(days=1)).isoformat()
        date.fromisoformat(day)
        with SessionLocal() as db:
            repo = Repository(db)
            for doc_id in doc_ids:
                d = repo.get_document(doc_id)
                if d is not None and d.kb_id == kb_id:
                    repo.update_document(d, valid_to=day)
        self._refresh_index_meta(kb_id, doc_ids)
        return day

    # --- 别名词典 ------------------------------------------------------------------------
    def aliases(self, kb_id: str) -> dict[str, Any]:
        return self._get(f"aliases:{kb_id}") or {"groups": [], "updatedAt": None}

    def save_aliases(self, kb_id: str, groups: list[list[str]]) -> dict[str, Any]:
        from .reindex_service import reindex_service

        clean = AliasDict(groups).groups
        data = {"groups": clean, "updatedAt": _now()}
        self._set(f"aliases:{kb_id}", data)
        with index_service.lock(kb_id):
            index_service.get(kb_id).aliases = AliasDict(clean) or None     # 查询扩展立即生效
        with SessionLocal() as db:
            has_docs = bool(Repository(db).list_documents(kb_id))
        if has_docs:
            reindex_service.enqueue(kb_id, self._kb_name(kb_id), "别名词典变更")   # 文档扩展需要重建
        return {**data, "reindexing": has_docs}

    def discover_aliases(self, kb_id: str) -> list[dict[str, Any]]:
        idx = index_service.get(kb_id)
        texts = [c.text for c in idx.store.chunks.values() if c.kind == "parent"]
        return discover_aliases(texts, AliasDict(self.aliases(kb_id).get("groups") or []))

    # --- 覆盖缺口 FAQ 增强 -----------------------------------------------------------------
    def augment_state(self, kb_id: str) -> dict[str, Any]:
        state = self._get(f"augment:{kb_id}") or {"items": [], "docId": None, "publishedAt": None}
        cov = eval_service.last_coverage(kb_id)
        known = {it["question"] for it in state["items"]} | {it.get("intent") for it in state["items"]}
        gaps = [r for r in (cov or {}).get("intents", []) if r["bucket"] in ("none", "partial")]
        state["coverage"] = {
            "available": cov is not None, "n": (cov or {}).get("n", 0),
            "gaps": len(gaps), "pendingGaps": [{"intent": r["intent"], "bucket": r["bucket"], "score": r["score"]} for r in gaps if r["intent"] not in known],
        }
        state["llm"] = model_service.llm(aux=True) is not None
        return state

    def _save_augment(self, kb_id: str, state: dict[str, Any]) -> None:
        self._set(f"augment:{kb_id}", {k: state.get(k) for k in ("items", "docId", "publishedAt")})

    def generate(self, kb_id: str, intents: list[str] | None = None, limit: int = 30) -> dict[str, Any]:
        state = self.augment_state(kb_id)
        if intents is None:
            todo = [(g["intent"], g["bucket"], g["score"]) for g in state["coverage"]["pendingGaps"]]
        else:
            known = {it.get("intent") or it["question"] for it in state["items"]}
            todo = [(q, "manual", None) for q in dict.fromkeys(x.strip() for x in intents) if q and q not in known]
        todo = todo[:limit]
        if not todo:
            return state
        llm = model_service.llm(aux=True)
        faq_doc = state.get("docId")
        drafts: list[dict[str, Any]] = []
        for intent, bucket, score in todo:
            try:
                # 用原问题检索（不做查询改写），结论与覆盖分析一致
                res = retrieval_service.retrieve(kb_id, intent, top_k=4, use_transform=False)
                contexts = [c for c in res.contexts if c.chunk.doc_id != faq_doc]   # 不拿 FAQ 自己回答自己
            except Exception as exc:
                logger.warning("增强检索失败 %s：%s", intent, exc)
                contexts = []
            item = draft_faq(intent, contexts, llm)
            drafts.append({"id": uuid.uuid4().hex[:10], "intent": intent, "bucket": bucket, "coverageScore": score, "published": False, **item})
        # 生成期间可能有人审核 / 修改 / 删除了条目：合并进最新状态，而不是用开始时的快照覆盖
        with self._state(f"augment:{kb_id}"):
            latest = self.augment_state(kb_id)
            known = {it.get("intent") or it["question"] for it in latest["items"]}
            latest["items"].extend(d for d in drafts if d["intent"] not in known)
            self._save_augment(kb_id, latest)
        return self.augment_state(kb_id)

    def add_item(self, kb_id: str, question: str, answer: str) -> dict[str, Any]:
        with self._state(f"augment:{kb_id}"):
            return self._add_item(kb_id, question, answer)

    def _add_item(self, kb_id: str, question: str, answer: str) -> dict[str, Any]:
        state = self.augment_state(kb_id)
        state["items"].append({"id": uuid.uuid4().hex[:10], "intent": question, "question": question, "answer": answer, "status": "approved" if answer.strip() else "gap",
                               "method": "manual", "bucket": "manual", "coverageScore": None, "sources": [], "note": "", "published": False, "createdAt": _now()})
        self._save_augment(kb_id, state)
        return self.augment_state(kb_id)

    def update_item(self, kb_id: str, item_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        with self._state(f"augment:{kb_id}"):
            return self._update_item(kb_id, item_id, fields)

    def _update_item(self, kb_id: str, item_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        state = self.augment_state(kb_id)
        for it in state["items"]:
            if it["id"] == item_id:
                for k in ("question", "answer"):
                    if fields.get(k) is not None:
                        it[k] = str(fields[k])
                status = fields.get("status")
                if status:
                    if status not in ITEM_STATUSES:
                        raise ValueError(f"未知状态 {status}")
                    if status == "approved" and not it["answer"].strip():
                        raise ValueError("答案为空，不能通过：请先补写答案")
                    it["status"] = status
                it["published"] = False if (fields.get("answer") is not None or fields.get("question") is not None) else it.get("published", False)
                it["updatedAt"] = _now()
                break
        else:
            raise KeyError(item_id)
        self._save_augment(kb_id, state)
        return self.augment_state(kb_id)

    def delete_item(self, kb_id: str, item_id: str) -> dict[str, Any]:
        with self._state(f"augment:{kb_id}"):
            return self._delete_item(kb_id, item_id)

    def _delete_item(self, kb_id: str, item_id: str) -> dict[str, Any]:
        state = self.augment_state(kb_id)
        state["items"] = [it for it in state["items"] if it["id"] != item_id]
        self._save_augment(kb_id, state)
        return self.augment_state(kb_id)

    def publish(self, kb_id: str) -> dict[str, Any]:
        with self._state(f"augment:{kb_id}"):
            return self._publish(kb_id)

    def _publish(self, kb_id: str) -> dict[str, Any]:
        """把通过的条目写成「增强语料 · FAQ」文档入库；没有通过的条目时撤下该文档。"""
        state = self.augment_state(kb_id)
        approved = [it for it in state["items"] if it["status"] == "approved" and it["answer"].strip()]
        doc_id = state.get("docId")
        with SessionLocal() as db:
            repo = Repository(db)
            doc = repo.get_document(doc_id) if doc_id else None
            if approved and doc is None:
                doc = repo.create_document(id=str(uuid.uuid4()), kb_id=kb_id, file_name=FAQ_FILE_NAME, file_path=None, file_size=0, file_format="md", status="PENDING", parser="augment",
                                           title="增强语料 · FAQ", tags="增强语料")
            doc_id = doc.id if doc is not None else None
        if not approved:
            if doc_id:
                document_service.delete(doc_id)
            doc_id = None
        else:
            markdown = build_faq_markdown(approved, self._kb_name(kb_id))
            with SessionLocal() as db:
                repo = Repository(db)
                repo.update_document(repo.get_document(doc_id), file_size=len(markdown.encode("utf-8")))
            document_service.update_markdown(doc_id, markdown)
        for it in state["items"]:
            it["published"] = it in approved
        state["docId"], state["publishedAt"] = doc_id, _now()
        self._save_augment(kb_id, state)
        return self.augment_state(kb_id)

    # --- 自动元数据 ---------------------------------------------------------------------
    @staticmethod
    def _current(d: Document) -> dict[str, Any]:
        return {"title": d.title, "version": d.version, "valid_from": d.valid_from, "valid_to": d.valid_to, "department": d.department,
                "tags": [t for t in (d.tags or "").split(",") if t], "access_level": int(d.access_level or 0)}

    def extract_metadata(self, kb_id: str, doc_ids: list[str] | None = None, use_llm: bool = False) -> list[dict[str, Any]]:
        llm = model_service.llm(aux=True) if use_llm else None
        with SessionLocal() as db:
            docs = [d for d in Repository(db).list_documents(kb_id) if (not doc_ids or d.id in doc_ids) and d.markdown_content and d.status == "COMPLETED"]
            rows = [(d.id, d.file_name, d.markdown_content, self._current(d), _loads(d.auto_meta), d.parser) for d in docs]
        out = []
        for doc_id, file_name, markdown, current, stored, parser in rows:
            if parser and parser.startswith("augment"):
                continue
            stem = Path(file_name).stem
            try:
                found = extract_metadata(markdown, llm, file_stem=stem)
            except Exception as exc:
                logger.warning("元数据抽取失败 %s：%s", file_name, exc)
                found = {}
            sugg = {}
            for key in FIELDS:
                if key not in found:
                    continue
                cur, val = current[key], found[key]["value"]
                empty = cur in (None, "", [], 0) or (key == "title" and cur == stem)
                a, b = _norm(val), _norm(cur)
                # 只差空格 / 前缀（"表：产品价格表" vs "产品价格表"、"2026" vs "2026.1"）不算新建议；已有标签不建议替换
                same = a == b or (bool(a) and bool(b) and (a in b or b in a)) or (key == "tags" and not empty)
                sugg[key] = {**found[key], "empty": empty, "same": same}
            out.append({"docId": doc_id, "fileName": file_name, "current": current, "suggestions": sugg, "autoMeta": stored})
        return out

    def apply_metadata(self, kb_id: str, items: list[dict[str, Any]]) -> int:
        changed: list[str] = []
        with SessionLocal() as db:
            repo = Repository(db)
            for item in items:
                d = repo.get_document(item.get("docId") or "")
                fields = item.get("fields") or {}
                if d is None or d.kb_id != kb_id or not fields:
                    continue
                upd: dict[str, Any] = {}
                for key, v in fields.items():
                    if key not in FIELDS:
                        continue
                    if key in ("valid_from", "valid_to"):
                        upd[key] = date.fromisoformat(str(v)).isoformat() if v else None
                    elif key == "tags":
                        upd["tags"] = ",".join(str(t).strip() for t in (v or []) if str(t).strip())
                    elif key == "access_level":
                        upd["access_level"] = max(0, min(3, int(v or 0)))
                    else:
                        upd[key] = str(v).strip() or None
                if not upd:
                    continue
                auto = _loads(d.auto_meta)
                for key in upd:
                    # 人工确认过的值标为 manual：重新解析 / 整库重建时优先于文件自带的元数据（document_service._keep_manual）
                    prev = auto.get(key) if isinstance(auto.get(key), dict) else {}
                    auto[key] = {"value": fields[key], "source": "manual", "evidence": prev.get("evidence", ""), "suggestedBy": prev.get("source"), "applied": True}
                upd["auto_meta"] = json.dumps(auto, ensure_ascii=False)
                repo.update_document(d, **upd)
                changed.append(d.id)
        if changed:
            self._refresh_index_meta(kb_id, changed)
        return len(changed)


corpus_service = CorpusService()
