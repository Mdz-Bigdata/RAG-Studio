"""多智能体共享黑板（LanceDB 表 agent_board）：每条记录 = 某次运行里某个智能体写下的一条内容 + 向量。

智能体之间不直接传话，而是读写同一块黑板：研究员把证据贴上去，分析师 / 撰稿人按语义取回与自己子任务相关的证据；
整次运行（问题、计划、证据、分析、审校意见、最终回答）都留在黑板上，可回放，也能跨运行检索"做过的类似任务"。
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ..index.embeddings import BaseEmbedder
from ..lance import connect, open_table, sql_in, sql_str, table_info, write_lock

TABLE = "agent_board"
_COLS = ["id", "run_id", "step", "agent", "kind", "content", "refs", "ts"]


def _schema(dim: int):
    import pyarrow as pa

    return pa.schema([pa.field("id", pa.string()), pa.field("run_id", pa.string()), pa.field("step", pa.int32()), pa.field("agent", pa.string()),
                      pa.field("kind", pa.string()), pa.field("content", pa.string()), pa.field("refs", pa.string()), pa.field("ts", pa.float64()),
                      pa.field("vector", pa.list_(pa.float32(), dim))])


class Blackboard:
    def __init__(self, directory: str | Path, embedder: BaseEmbedder) -> None:
        self.directory = Path(directory)
        self.embedder = embedder
        self._step: dict[str, int] = {}
        self.dim = int(embedder.dim or len(embedder.embed_documents(["维度探测"])[0]))

    def _table(self, create: bool = False):
        db = connect(self.directory)
        t = open_table(db, TABLE)
        if t is not None and t.schema.field("vector").type.list_size != self.dim:
            # 换了向量模型：黑板是工作记忆，历史运行的向量不再可比，直接换新表（运行记录的原文不再参与语义检索）
            db.create_table(f"{TABLE}_archive_{int(time.time())}", t.to_arrow(), mode="overwrite")
            t = db.create_table(TABLE, schema=_schema(self.dim), mode="overwrite")
        if t is None and create:
            t = db.create_table(TABLE, schema=_schema(self.dim), exist_ok=True)
        return t

    def post(self, run_id: str, agent: str, kind: str, content: str, refs: Any = None) -> dict[str, Any]:
        import pyarrow as pa

        step = self._step.get(run_id, 0) + 1
        self._step[run_id] = step
        row = {"id": uuid.uuid4().hex, "run_id": run_id, "step": step, "agent": agent, "kind": kind, "content": content,
               "refs": json.dumps(refs or [], ensure_ascii=False), "ts": time.time()}
        vec = self.embedder.embed_documents([content[:2000] or kind])[0]
        tbl = self._table(create=True)
        with write_lock(self.directory, TABLE):
            tbl.add(pa.Table.from_pylist([{**row, "vector": np.asarray(vec, dtype=np.float32)}], schema=_schema(self.dim)))
        return row

    def post_many(self, run_id: str, agent: str, kind: str, items: list[tuple[str, Any]]) -> list[dict[str, Any]]:
        """一次贴多条（研究员的一批证据）：批量向量化、一次写入。"""
        import pyarrow as pa

        if not items:
            return []
        vecs = self.embedder.embed_documents([c[:2000] for c, _ in items])
        rows = []
        for (content, refs), vec in zip(items, vecs):
            step = self._step.get(run_id, 0) + 1
            self._step[run_id] = step
            rows.append({"id": uuid.uuid4().hex, "run_id": run_id, "step": step, "agent": agent, "kind": kind, "content": content,
                         "refs": json.dumps(refs or [], ensure_ascii=False), "ts": time.time(), "vector": np.asarray(vec, dtype=np.float32)})
        tbl = self._table(create=True)
        with write_lock(self.directory, TABLE):
            tbl.add(pa.Table.from_pylist(rows, schema=_schema(self.dim)))
        return [{k: r[k] for k in _COLS} for r in rows]

    def recall(self, run_id: str, query: str, *, kinds: list[str] | None = None, k: int = 5) -> list[dict[str, Any]]:
        """在本次运行的黑板上做语义检索（只看指定类型，例如 evidence）。"""
        tbl = self._table()
        if tbl is None:
            return []
        where = f"run_id = {sql_str(run_id)}" + (f" AND {sql_in('kind', kinds)}" if kinds else "")
        vec = self.embedder.embed_queries([query])[0]
        rows = tbl.search(np.asarray(vec, dtype=np.float32)).distance_type("cosine").where(where, prefilter=True).select(_COLS + ["_distance"]).limit(k).to_list()
        return [{**self._public(r), "similarity": round(1.0 - float(r["_distance"]), 4)} for r in rows]

    def history(self, run_id: str) -> list[dict[str, Any]]:
        tbl = self._table()
        if tbl is None:
            return []
        rows = tbl.search().where(f"run_id = {sql_str(run_id)}").select(_COLS).limit(10_000).to_list()
        return [self._public(r) for r in sorted(rows, key=lambda r: (r["ts"], r["step"]))]

    def runs(self, limit: int = 30) -> list[dict[str, Any]]:
        tbl = self._table()
        if tbl is None:
            return []
        rows = tbl.search().where("kind IN ('question', 'answer')").select(_COLS).limit(100_000).to_list()
        runs: dict[str, dict[str, Any]] = {}
        for r in rows:
            run = runs.setdefault(r["run_id"], {"runId": r["run_id"], "question": "", "answer": "", "ts": r["ts"]})
            if r["kind"] == "question":
                run["question"], run["ts"] = r["content"], r["ts"]
                refs = json.loads(r["refs"] or "{}")
                run.update({"kbId": refs.get("kbId"), "userId": refs.get("userId")} if isinstance(refs, dict) else {})
            else:
                run["answer"] = r["content"][:160]
        return sorted(runs.values(), key=lambda x: x["ts"], reverse=True)[:limit]

    def delete_runs(self, run_ids: list[str]) -> int:
        """删掉若干次运行在黑板上的全部记录（问题、计划、证据、分析、审校、回答），返回删除的行数。"""
        tbl = self._table()
        if tbl is None or not run_ids:
            return 0
        with write_lock(self.directory, TABLE):
            deleted = tbl.delete(sql_in("run_id", run_ids)).num_deleted_rows
        for rid in run_ids:
            self._step.pop(rid, None)
        return deleted

    def clear(self) -> int:
        """清空黑板上的全部运行记录（表结构保留）。"""
        tbl = self._table()
        if tbl is None:
            return 0
        with write_lock(self.directory, TABLE):
            deleted = tbl.delete("true").num_deleted_rows
        self._step.clear()
        return deleted

    def similar_runs(self, question: str, *, exclude_run: str = "", k: int = 3) -> list[dict[str, Any]]:
        """跨运行的经验检索：以前做过的相似任务（只看问题行）。"""
        tbl = self._table()
        if tbl is None:
            return []
        where = "kind = 'question'" + (f" AND run_id != {sql_str(exclude_run)}" if exclude_run else "")
        vec = self.embedder.embed_queries([question])[0]
        rows = tbl.search(np.asarray(vec, dtype=np.float32)).distance_type("cosine").where(where, prefilter=True).select(_COLS + ["_distance"]).limit(k).to_list()
        return [{"runId": r["run_id"], "question": r["content"], "similarity": round(1.0 - float(r["_distance"]), 4)} for r in rows]

    def info(self) -> dict[str, Any]:
        tbl = self._table()
        return table_info(tbl) if tbl is not None else {"name": TABLE, "rows": 0}

    @staticmethod
    def _public(r: dict[str, Any]) -> dict[str, Any]:
        try:
            refs = json.loads(r.get("refs") or "[]")
        except ValueError:
            refs = []
        return {"id": r["id"], "runId": r["run_id"], "step": r["step"], "agent": r["agent"], "kind": r["kind"], "content": r["content"], "refs": refs, "ts": r["ts"]}
