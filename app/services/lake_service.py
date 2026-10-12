"""Agent 湖服务：问答自动沉淀、历史问答检索增强、从业务库回填、湖分析、版本 / Tag / 分支 / 派生列。

湖表 agent_lake 和长期记忆 / 推荐 / 黑板在同一个全局 LanceDB 目录（data/lancedb）。
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any

from prism.lake import AgentLake, LakeAnalyst, format_experience, recall_experience
from prism.lake import analytics, run_retest
from prism.lake.changes import QUIET_WHEN_UNCHANGED, REASONS, describe_change, markdown_diff
from prism.lance import connect, open_table

from ..config import app_config
from ..db import Repository, SessionLocal
from .model_service import model_service
from .settings_service import settings_service

logger = logging.getLogger(__name__)

def lance_dir():
    """与长期记忆 / 推荐 / 黑板共用的全局 LanceDB 目录（每次取：测试会在运行时改 app_config.data_dir）。"""
    return app_config.data_dir / "lancedb"


def _ts(dt: datetime | None) -> float:
    if dt is None:
        return time.time()
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()     # SQLite 里存的是 utcnow()


def _loads(text: str | None, default: Any) -> Any:
    try:
        return json.loads(text) if text else default
    except ValueError:
        return default


def turn_records(*, session_id: str, kb_id: str | None, user_id: str, question_id: str, question: str, q_ts: float, answer_id: str, answer: str, a_ts: float,
                 sources: list[dict[str, Any]], citations: list[int], plan: dict[str, Any], model: str | None, mode: str | None = None, error: bool = False) -> list[dict[str, Any]]:
    """一轮问答 → 三条湖记录：question、retrieval（检索日志）、answer。实时沉淀与历史回填共用。"""
    from prism.generation.answer import looks_like_refusal

    kb = kb_id or ""
    timings = plan.get("timingsMs") or {}
    scores = [float(s.get("score") or 0.0) for s in sources]
    contexts = [{"docId": s.get("docId"), "fileName": s.get("fileName"), "chunkId": s.get("chunkId"), "score": s.get("score")} for s in sources[:8]]
    ret_text = " | ".join(plan.get("queries") or [question]) + ("\n命中：" + "、".join(dict.fromkeys(c["fileName"] or "" for c in contexts if c["fileName"])) if contexts else "")
    cited_docs = list(dict.fromkeys(sources[i - 1].get("docId") for i in citations if 0 < i <= len(sources) and sources[i - 1].get("docId")))
    a_status = "error" if error else ("refused" if (not sources or looks_like_refusal(answer)) else "ok")
    return [
        {"id": question_id, "kind": "question", "session_id": session_id, "user_id": user_id, "agent": "user", "kb_id": kb, "content": question,
         "meta": {"mode": mode or plan.get("strategy")}, "ts": q_ts},
        {"id": f"ret-{answer_id}", "kind": "retrieval", "session_id": session_id, "user_id": user_id, "agent": "retriever", "kb_id": kb, "parent_id": question_id,
         "title": question[:120], "content": ret_text, "status": "ok" if sources else "no_context", "score": max(scores) if scores else None,
         "latency_ms": timings.get("total"), "refs": [c["docId"] for c in contexts if c["docId"]], "ts": q_ts + 0.001,
         "meta": {"strategy": plan.get("strategy"), "queries": plan.get("queries"), "notes": (plan.get("notes") or [])[:12], "timingsMs": timings,
                  "funnel": plan.get("funnel"), "contexts": contexts, "experience": [e.get("questionId") for e in plan.get("experience") or []],
                  "routeFallback": plan.get("routeFallback")}},
        {"id": answer_id, "kind": "answer", "session_id": session_id, "user_id": user_id, "agent": "assistant", "kb_id": kb, "parent_id": question_id,
         "title": question[:120], "content": answer, "status": a_status, "refs": cited_docs, "ts": a_ts, "meta": {"model": model, "citations": citations}},
    ]


class LakeService:
    def __init__(self) -> None:
        self._lake: tuple[str, AgentLake] | None = None
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._writer: threading.Thread | None = None

    # ------------------------------------------------------------------ 后台写入（单线程、按提交顺序）
    def _enqueue(self, fn) -> None:
        """湖的后台写入排成一队依次执行：同一份文档"校对 → 删除"这样的先后关系不会被线程调度打乱。"""
        with self._lock:
            if self._writer is None or not self._writer.is_alive():
                self._writer = threading.Thread(target=self._drain, name="lake-writer", daemon=True)
                self._writer.start()
        self._queue.put(fn)

    def _drain(self) -> None:
        while True:
            fn = self._queue.get()
            try:
                fn()
            except Exception:
                logger.exception("Agent 湖后台写入失败")
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 30.0) -> bool:
        """等后台写入全部完成（测试 / 需要立即读到结果时用）。"""
        deadline = time.time() + timeout
        while self._queue.unfinished_tasks and time.time() < deadline:
            time.sleep(0.02)
        return not self._queue.unfinished_tasks

    def cfg(self):
        return settings_service.get().lake

    def lake(self) -> AgentLake:
        emb, info = model_service.embedder_info()
        key = f"{lance_dir()}:{id(emb)}:{info.get('model')}"
        with self._lock:
            if self._lake is None or self._lake[0] != key:
                self._lake = (key, AgentLake(lance_dir(), emb, model_name=str(info.get("model") or type(emb).__name__)))
            return self._lake[1]

    # ------------------------------------------------------------------ 问答沉淀 / 检索增强
    def experience(self, query: str, *, kb_id: str | None, session_id: str | None) -> tuple[str, list[dict[str, Any]]]:
        cfg = self.cfg()
        if not (cfg.enabled and cfg.augment) or not query:
            return "", []
        try:
            items = recall_experience(self.lake(), query, kb_id=kb_id, session_id=session_id, top_k=cfg.augment_top_k, min_score=cfg.augment_min_score)
        except Exception as exc:
            logger.warning("Agent 湖历史问答召回失败：%s", exc)
            return "", []
        return format_experience(items), items

    def record_turn_async(self, **kw: Any) -> None:
        cfg = self.cfg()
        if not (cfg.enabled and cfg.record_chat):
            return
        lake = self.lake()                     # 在调用时就绑定目录：测试会在运行时切换数据目录，后台线程晚执行也不会写错库
        self._enqueue(lambda: self._record_turn(lake, **kw))

    def _record_turn(self, lake: AgentLake, **kw: Any) -> None:
        try:
            lake.append(turn_records(**kw))
        except Exception:
            logger.exception("Agent 湖：问答沉淀失败")

    def feedback(self, answer_id: str, value: int) -> bool:
        return self.lake().update(answer_id, feedback=max(-1, min(1, int(value))))

    # ------------------------------------------------------------------ 回填
    def sync(self, sources: list[str] | None = None) -> dict[str, Any]:
        """把业务库里已有的数据回填进湖（按 id 去重，可重复执行）：历史对话、知识库文档、长期记忆、智能体黑板。"""
        sources = sources or ["chat", "documents", "memories", "board"]
        lake = self.lake()
        out: dict[str, Any] = {}
        if "chat" in sources:
            out["chat"] = lake.append(self._chat_records())
        if "documents" in sources:
            out["documents"] = lake.append(self._document_records())
        db = connect(lance_dir())
        if "memories" in sources:
            out["memories"] = lake.append(self._lance_rows(db, "memories", self._memory_record))
        if "board" in sources:
            out["board"] = lake.append(self._lance_rows(db, "agent_board", self._board_record))
        return out

    def _chat_records(self) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []
        with SessionLocal() as db:
            repo = Repository(db)
            for s in repo.list_sessions():
                msgs = repo.list_messages(s.id)
                for prev, m in zip(msgs, msgs[1:]):
                    if prev.role != "user" or m.role != "assistant":
                        continue
                    plan = _loads(m.trace_json, {})
                    recs += turn_records(session_id=s.id, kb_id=s.kb_id, user_id="default", question_id=prev.id, question=prev.content, q_ts=_ts(prev.created_at),
                                         answer_id=m.id, answer=m.content, a_ts=_ts(m.created_at), sources=_loads(m.sources_json, []), citations=_loads(m.citations_json, []),
                                         plan=plan, model=m.model, error="[生成失败]" in (m.content or ""))
        return recs

    @staticmethod
    def _document_record(d, kb_name: str) -> dict[str, Any]:
        """文档快照行（id = doc-<文档 id>）：标题 + 正文前 1500 字；文档变更时按最新内容覆盖。"""
        body = (d.markdown_content or "")[:1500]
        return {"id": f"doc-{d.id}", "kind": "document", "agent": "ingest", "kb_id": d.kb_id, "title": d.title or d.file_name,
                "content": f"{d.title or d.file_name}\n{body}", "status": "ok" if d.status == "COMPLETED" else d.status.lower(),
                "meta": {"fileName": d.file_name, "format": d.file_format, "parser": d.parser, "chunks": d.chunk_count, "version": d.version,
                         "tags": d.tags, "kbName": kb_name}, "ts": _ts(d.created_at)}

    def _document_records(self) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            repo = Repository(db)
            return [self._document_record(d, kb.name) for kb in repo.list_kbs() for d in repo.list_documents(kb.id)]

    # ------------------------------------------------------------------ 文档变更记录
    def record_doc_event_async(self, doc_id: str, reason: str, before: str | None, after: str | None = None, *, status: str = "ok",
                               error: str | None = None, info: dict[str, Any] | None = None) -> None:
        """文档服务在上传解析 / 校对 / 重新解析 / 恢复 / 删除之后调用（后台线程，失败只记日志，不影响文档处理）。"""
        if not self.cfg().enabled:
            return
        try:
            lake = self.lake()                 # 调用时绑定目录（同 record_turn_async）
            snapshot = None
            if info is None:                   # 入队时就读文档信息：排在后面执行时文档可能已经被删了
                with SessionLocal() as db:
                    repo = Repository(db)
                    d = repo.get_document(doc_id)
                    if d is None:
                        return
                    kb = repo.get_kb(d.kb_id)
                    info = {"fileName": d.file_name, "kbId": d.kb_id, "parser": d.parser, "chunks": d.chunk_count, "status": d.status}
                    snapshot = self._document_record(d, kb.name if kb else d.kb_id)
                    if after is None:
                        after = d.markdown_content
        except Exception:
            logger.exception("Agent 湖：读取文档信息失败（%s %s）", reason, doc_id)
            return
        self._enqueue(lambda: self._record_doc_event(doc_id, reason, before, after, status=status, error=error, info=info, snapshot=snapshot, lake=lake))

    def _record_doc_event(self, doc_id: str, reason: str, before: str | None, after: str | None, *, status: str = "ok", error: str | None = None,
                          info: dict[str, Any], snapshot: dict[str, Any] | None = None, lake: AgentLake | None = None) -> dict[str, Any] | None:
        try:
            diff = markdown_diff(before, after if reason != "delete" else "")
            if status == "ok" and reason in QUIET_WHEN_UNCHANGED and not diff["changed"]:
                return None
            rec = {"kind": "doc_event", "agent": "document-service", "kb_id": info.get("kbId"), "parent_id": f"doc-{doc_id}", "title": info.get("fileName") or doc_id,
                   "content": describe_change(info.get("fileName") or doc_id, reason, diff, status=status, error=error), "status": status,
                   "meta": {"event": reason, "eventLabel": REASONS.get(reason, reason), "docId": doc_id, "parser": info.get("parser"), "chunks": info.get("chunks"),
                            "error": error, **diff}}
            lake = lake or self.lake()
            lake.append([rec])
            if snapshot is not None and status == "ok":
                lake.append([snapshot], upsert=True)             # 快照跟着最新内容走
            elif reason == "delete":
                lake.update(f"doc-{doc_id}", status="deleted")
            return rec
        except Exception:
            logger.exception("Agent 湖：文档变更记录写入失败（%s %s）", reason, doc_id)
            return None

    # ------------------------------------------------------------------ 缺口复测
    def _retest_one(self, question: str, kb_id: str, check_answer: bool) -> dict[str, Any]:
        """用现在的知识库重新检索（与智能问答同一条检索链路）；check_answer 时再生成一次回答，看是否拒答。"""
        from prism.generation import AnswerGenerator

        from .retrieval_service import retrieval_service

        result = retrieval_service.retrieve(kb_id, question)
        names = retrieval_service.doc_names(kb_id)
        contexts = [{"docId": c.chunk.doc_id, "fileName": names.get(c.chunk.doc_id), "score": round(float(c.score), 4)} for c in result.contexts]
        out: dict[str, Any] = {"score": max((c["score"] for c in contexts), default=None), "contexts": contexts, "refused": None, "answer": None}
        if check_answer:
            ans = AnswerGenerator(settings_service.get().generation, model_service.llm()).answer(result)
            out.update(refused=bool(ans.refused), answer=ans.text)
        # 本库仍答不上：看别的知识库能不能答——能答说明是选错了库 / 路由问题，而不是缺语料
        if not contexts or (out["score"] or 0.0) < self.cfg().low_score or out["refused"]:
            out["altKb"] = retrieval_service.suggest_kb(question, kb_id, full=True, check_answer=check_answer)
        return out

    def retest(self, *, limit: int = 10, include_weak: bool = False, check_answer: bool = True) -> dict[str, Any]:
        lake = self.lake()
        return run_retest(lake, analytics.frame(lake.to_arrow()), self._retest_one, low_score=self.cfg().low_score, include_weak=include_weak,
                          check_answer=check_answer, limit=limit)

    @staticmethod
    def _lance_rows(db, table: str, convert) -> list[dict[str, Any]]:
        tbl = open_table(db, table)
        if tbl is None:
            return []
        rows = tbl.to_arrow()
        rows = rows.drop_columns([c for c in ("vector",) if c in rows.column_names]).to_pylist()
        return [r for r in (convert(x) for x in rows) if r]

    @staticmethod
    def _memory_record(r: dict[str, Any]) -> dict[str, Any] | None:
        if not r.get("content"):
            return None
        return {"id": f"mem-{r['id']}", "kind": "memory", "agent": r.get("agent_id") or "assistant", "user_id": r.get("user_id") or "default",
                "session_id": r.get("session_id") or "", "content": r["content"], "score": r.get("importance"),
                "meta": {"memoryKind": r.get("kind"), "source": r.get("source"), "accessCount": r.get("access_count")}, "ts": r.get("updated_at") or r.get("created_at")}

    @staticmethod
    def _board_record(r: dict[str, Any]) -> dict[str, Any] | None:
        if not r.get("content"):
            return None
        return {"id": f"board-{r['id']}", "kind": "board", "agent": r.get("agent") or "agent", "session_id": r.get("run_id") or "", "content": r["content"],
                "refs": r.get("refs") or "[]", "meta": {"boardKind": r.get("kind"), "step": r.get("step")}, "ts": r.get("ts")}

    # ------------------------------------------------------------------ 分析
    def _frame(self, version=None, branch=None, with_vector=False):
        lake = self.lake()
        return analytics.frame(lake.to_arrow(view=lake.view(version=version, branch=branch), with_vector=with_vector))

    def overview(self, version=None, branch=None, days: int = 30) -> dict[str, Any]:
        lake = self.lake()
        return {"info": lake.info(), "stats": analytics.overview(self._frame(version, branch), days=days), "cfg": self.cfg().model_dump()}

    def quality(self, version=None, branch=None, days: int | None = None) -> dict[str, Any]:
        return analytics.retrieval_quality(self._frame(version, branch), low_score=self.cfg().low_score, days=days)

    def topics(self, *, k: int | None = None, version=None, branch=None, write_column: str | None = None) -> dict[str, Any]:
        df = self._frame(version, branch, with_vector=True)
        q = analytics.retrieval_quality(df.drop(columns=["vector"]), low_score=self.cfg().low_score, gap_limit=100_000)
        r = df[df["kind"] == "retrieval"]
        scores = {row.parent_id: float(row.score) for row in r.itertuples() if row.score == row.score and row.score is not None}
        result = analytics.topics(df, k=k, gaps={g["questionId"] for g in q["gaps"]}, scores=scores, with_ids=True)
        if write_column:
            result["written"] = self.lake().add_column(write_column, values=result["assignments"], branch=branch)
        result.pop("assignments", None)
        return result

    def sql(self, query: str, version=None, branch=None) -> dict[str, Any]:
        lake = self.lake()
        return analytics.run_sql({"lake": lake.to_arrow(view=lake.view(version=version, branch=branch))}, query, limit=self.cfg().sql_row_limit)

    def schema(self, version=None, branch=None) -> dict[str, Any]:
        lake = self.lake()
        view = lake.view(version=version, branch=branch)
        cols = [{"name": f.name, "type": str(f.type)} for f in (view.schema if view is not None else []) if f.name != "vector"]
        return {"columns": cols, "doc": analytics.schema_doc(cols)}

    def analyze(self, question: str, *, version=None, branch=None, user_id: str = "default") -> dict[str, Any]:
        llm = model_service.llm(aux=True)
        # 复测只对主表最新版本有意义（要和现在的知识库比）；在历史版本 / 分支上分析时不提供
        retest_fn = self._retest_one if not version and not branch else None
        return LakeAnalyst(self.lake(), llm, self.cfg(), version=version, branch=branch, retest_fn=retest_fn, kb_info_fn=self.kb_profiles).run(question, user_id=user_id)

    @staticmethod
    def kb_profiles() -> list[dict[str, Any]]:
        """各知识库当前的画像（名称 / 描述 / 分类 / 文档清单），给分析智能体判断路由问题、知识库描述是否已经完善。"""
        with SessionLocal() as db:
            repo = Repository(db)
            out = []
            for kb in repo.list_kbs():
                docs = repo.list_documents(kb.id)
                out.append({"kbId": kb.id, "name": kb.name, "description": kb.description or "", "category": kb.category or "",
                            "docCount": len(docs), "documents": [d.title or d.file_name for d in docs[:15]],
                            "updatedAt": kb.updated_at.isoformat() if kb.updated_at else None})
            return out


lake_service = LakeService()
