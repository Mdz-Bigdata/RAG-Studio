"""Agent 湖：LanceDB 表 agent_lake —— 一张带版本的宽表，装下 Agent 运行时产生的全部上下文资产。

思路来自"从多模态数据湖到 Agent 湖"（Lance 格式）：记忆只是上下文里提炼出的一小部分，真正的资产是上下文本身——
问题、检索日志、回答、反馈、工具调用、分析记录、长期记忆、黑板、文档。它们放在同一张表里：
- 一行 = 一条上下文资产（kind 区分类型），结构化列 + JSON 元数据 + 正文 + 向量，`parent_id` / `refs` 记录跨会话的引用关系；
- 向量 + 全文（ICU 中文分词）+ SQL 预过滤：同一份数据既能检索增强，也能做分析；
- 每次写入都是一个新版本：可时间旅行、打 Tag 冻结一个快照；
- 派生列（主题、意图、打分……）按主键追加，不重写已有数据；实验在分支上做，验证后把列合并回主表——
  对应文章里"主表 → 冻结 → 拉分支加打分列 → 验证后打 Tag 发布"的数据处理时间线。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute  # noqa: F401  （pa.compute）

from ..index.embeddings import BaseEmbedder
from ..lance import connect, list_table_names, open_table, sql_in, sql_str, table_info, version_list, write_lock
from ..lance.db import has_index

logger = logging.getLogger(__name__)

TABLE = "agent_lake"
_GUARD_TAG = "_cleanup_guard_"      # 清理版本时临时打在下一个版本上的保护 Tag，列表里不展示
KINDS = ("question", "retrieval", "answer", "analysis", "tool_call", "memory", "board", "document", "doc_event", "retest", "note")
KIND_LABELS = {
    "question": "问题", "retrieval": "检索日志", "answer": "回答", "analysis": "湖分析", "tool_call": "工具调用",
    "memory": "长期记忆", "board": "智能体黑板", "document": "文档", "doc_event": "文档变更", "retest": "缺口复测", "note": "笔记",
}
# 基础列（固定 schema）；之外的列都是派生列，可以随时追加 / 删除
BASE_COLUMNS = ["id", "kind", "session_id", "user_id", "agent", "kb_id", "title", "content", "meta", "refs", "parent_id",
                "status", "score", "latency_ms", "feedback", "ts", "day", "embed_model"]
_TEXT_COLS = {"id", "kind", "session_id", "user_id", "agent", "kb_id", "title", "content", "meta", "refs", "parent_id", "status", "day", "embed_model"}
_COL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_COL_TYPES = {"string": "string", "double": "float64", "int": "int64", "bool": "bool"}
_OPTIMIZE_EVERY = 20


def _schema(dim: int, extra: list[Any] | None = None):
    import pyarrow as pa

    fields = [pa.field(c, pa.string()) for c in ("id", "kind", "session_id", "user_id", "agent", "kb_id", "title", "content", "meta", "refs", "parent_id", "status")]
    fields += [pa.field("score", pa.float64()), pa.field("latency_ms", pa.float64()), pa.field("feedback", pa.int32()), pa.field("ts", pa.float64()),
               pa.field("day", pa.string()), pa.field("embed_model", pa.string()), pa.field("vector", pa.list_(pa.float32(), dim))]
    return pa.schema(fields + list(extra or []))


def _json(value: Any, default: str) -> str:
    if value is None:
        return default
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _embed_text(rec: dict[str, Any]) -> str:
    title, content = rec.get("title") or "", rec.get("content") or ""
    text = f"{title}\n{content}" if title and title not in content else content or title
    return text[:2000] or rec.get("kind", "")


class AgentLake:
    """Agent 湖的读写入口。directory 是全局 LanceDB 目录（与记忆 / 推荐 / 黑板同库，不同表）。"""

    def __init__(self, directory: str | Path, embedder: BaseEmbedder, *, model_name: str = "", table: str = TABLE) -> None:
        self.directory = Path(directory)
        self.embedder = embedder
        self.model_name = model_name or type(embedder).__name__
        self.table_name = table
        self.dim = int(embedder.dim or len(embedder.embed_documents(["维度探测"])[0]))
        self._writes = 0
        self._checked = False
        self._check_lock = threading.Lock()

    # ------------------------------------------------------------------ 表与版本
    def _db(self):
        return connect(self.directory)

    def _table(self, *, create: bool = False):
        db = self._db()
        tbl = open_table(db, self.table_name)
        if tbl is None and create:
            with write_lock(self.directory, self.table_name):
                tbl = open_table(db, self.table_name) or db.create_table(self.table_name, schema=_schema(self.dim), exist_ok=True)
        if tbl is not None and not self._checked:
            self._ensure_model(tbl)
        return tbl

    def exists(self) -> bool:
        return self.table_name in list_table_names(self._db()) or open_table(self._db(), self.table_name) is not None

    def view(self, *, version: int | str | None = None, branch: str | None = None):
        """只读视图：某个版本号 / Tag / 分支。每次打开新的表句柄，checkout 不影响共享句柄。"""
        tbl = self._table()
        if tbl is None:
            return None
        if branch and branch != "main":
            return tbl.branches.checkout(branch)
        if version not in (None, "", "latest"):
            fresh = self._db().open_table(self.table_name)
            fresh.checkout(int(version) if str(version).isdigit() else str(version))
            return fresh
        return tbl

    def _ensure_model(self, tbl) -> None:
        """向量模型换了（维度或模型名变化）：用保存的正文重算全部向量，写成新版本（旧版本 / Tag 仍可回看）。"""
        with self._check_lock:
            if self._checked:
                return
            self._checked = True
            vec_type = tbl.schema.field("vector").type
            dim = getattr(vec_type, "list_size", self.dim)
            if tbl.count_rows() == 0:
                if dim != self.dim:
                    self._rewrite(tbl, [])
                return
            sample = tbl.search().select(["embed_model"]).limit(1).to_list()
            same_model = sample and sample[0].get("embed_model") == self.model_name
            if dim == self.dim and same_model:
                return
            logger.info("Agent 湖：向量模型变化（%s 维 → %s 维，%s），重算向量", dim, self.dim, self.model_name)
            rows = tbl.to_arrow().drop_columns(["vector"]).to_pylist()
            self._rewrite(tbl, rows)

    def _rewrite(self, tbl, rows: list[dict[str, Any]]) -> None:
        import pyarrow as pa

        extra = [f for f in tbl.schema if f.name not in BASE_COLUMNS and f.name != "vector"]
        schema = _schema(self.dim, extra)
        vecs = self.embedder.embed_documents([_embed_text(r) for r in rows]) if rows else np.zeros((0, self.dim), dtype=np.float32)
        for r, v in zip(rows, vecs):
            r["vector"] = np.asarray(v, dtype=np.float32).tolist()
            r["embed_model"] = self.model_name
        with write_lock(self.directory, self.table_name):
            self._db().create_table(self.table_name, data=pa.Table.from_pylist(rows, schema=schema), schema=schema, mode="overwrite")

    def _after_write(self, tbl) -> None:
        self._writes += 1
        if not has_index(tbl, "content"):
            try:
                tbl.create_fts_index("content", base_tokenizer="icu", stem=False, remove_stop_words=False, lower_case=True, replace=True)
            except Exception as exc:  # 没有 ICU 分词时退回空白分词
                logger.info("Agent 湖：ICU 全文索引不可用（%s），改用 whitespace", exc)
                tbl.create_fts_index("content", replace=True)
        elif self._writes % _OPTIMIZE_EVERY == 0:
            try:
                tbl.optimize()       # 合并小文件 + 新数据并入全文索引
            except Exception as exc:
                logger.debug("optimize 跳过：%s", exc)

    # ------------------------------------------------------------------ 写入
    def normalize(self, rec: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        ts = float(rec.get("ts") or now)
        kind = rec.get("kind") or "note"
        if kind not in KINDS:
            raise ValueError(f"未知的记录类型：{kind}（可选：{', '.join(KINDS)}）")
        return {
            "id": str(rec.get("id") or uuid.uuid4().hex),
            "kind": kind,
            "session_id": str(rec.get("session_id") or ""),
            "user_id": str(rec.get("user_id") or "default"),
            "agent": str(rec.get("agent") or "assistant"),
            "kb_id": str(rec.get("kb_id") or ""),
            "title": str(rec.get("title") or "")[:300],
            "content": str(rec.get("content") or ""),
            "meta": _json(rec.get("meta"), "{}"),
            "refs": _json(rec.get("refs"), "[]"),
            "parent_id": str(rec.get("parent_id") or ""),
            "status": str(rec.get("status") or "ok"),
            "score": None if rec.get("score") is None else float(rec["score"]),
            "latency_ms": None if rec.get("latency_ms") is None else float(rec["latency_ms"]),
            "feedback": int(rec.get("feedback") or 0),
            "ts": ts,
            "day": _day(ts),
            "embed_model": self.model_name,
        }

    def append(self, records: Iterable[dict[str, Any]], *, upsert: bool = False) -> dict[str, int]:
        """写入一批记录。默认按 id 去重（已存在的跳过，派生列不受影响）；upsert=True 时覆盖基础列。"""
        import pyarrow as pa

        rows = [self.normalize(r) for r in records]
        if not rows:
            return {"inserted": 0, "skipped": 0}
        rows = list({r["id"]: r for r in rows}.values())
        dead = self._tombstones()
        removed = 0
        if dead:                                   # 用户删过的记录：回填历史数据 / 文档快照不再把它写回来
            removed = len(rows)
            rows = [r for r in rows if r["id"] not in dead]
            removed -= len(rows)
            if not rows:
                return {"inserted": 0, "skipped": removed}
        vecs = self.embedder.embed_documents([_embed_text(r) for r in rows])
        for r, v in zip(rows, vecs):
            r["vector"] = np.asarray(v, dtype=np.float32).tolist()
        tbl = self._table(create=True)
        with write_lock(self.directory, self.table_name):
            existing = self._existing_ids(tbl, [r["id"] for r in rows])
            new_rows = rows if upsert else [r for r in rows if r["id"] not in existing]
            if new_rows:
                schema = tbl.schema
                data = pa.Table.from_pylist([{f.name: r.get(f.name) for f in schema} for r in new_rows], schema=schema)
                if upsert and existing:
                    base = data.select(BASE_COLUMNS + ["vector"])
                    tbl.merge_insert("id").when_matched_update_all().execute(base.filter(pa.compute.is_in(base["id"], pa.array(list(existing)))))
                    fresh = [r for r in new_rows if r["id"] not in existing]
                    if fresh:
                        tbl.add(pa.Table.from_pylist([{f.name: r.get(f.name) for f in schema} for r in fresh], schema=schema))
                else:
                    tbl.add(data)
                self._after_write(tbl)
        inserted = len([r for r in new_rows if r["id"] not in existing])
        return {"inserted": inserted, "updated": len(new_rows) - inserted, "skipped": len(rows) - len(new_rows) + removed}

    # ------------------------------------------------------------------ 删除记录
    def _tombstone_path(self) -> Path:
        return self.directory / f"{self.table_name}.deleted.json"

    def _tombstones(self) -> set[str]:
        p = self._tombstone_path()
        try:
            return set(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else set()
        except (OSError, ValueError):
            return set()

    def delete_records(self, ids: list[str], *, cascade: bool = True, dry_run: bool = False) -> list[str]:
        """按 id 删除记录；cascade 时连同 parent_id 指向它的后代（问题 → 检索日志 / 回答 → …）一起删。
        删掉的 id 记进墓碑文件：这些记录多半来自对话历史 / 文档，不记的话下次「回填历史数据」又会被写回来。
        dry_run 只返回将被删除的 id。"""
        tbl = self._table()
        if tbl is None or not ids:
            return []
        with write_lock(self.directory, self.table_name):
            found = self._existing_ids(tbl, list(ids))
            seen, todo = set(found), list(found)
            while cascade and todo:
                kids: list[str] = []
                for i in range(0, len(todo), 500):
                    part = todo[i:i + 500]
                    kids += tbl.search().where(sql_in("parent_id", part)).select(["id"]).limit(100_000).to_arrow().column("id").to_pylist()
                todo = [k for k in dict.fromkeys(kids) if k not in seen]
                seen.update(todo)
            out = sorted(seen)
            if dry_run or not out:
                return out
            for i in range(0, len(out), 500):
                tbl.delete(sql_in("id", out[i:i + 500]))
            dead = self._tombstones() | seen
            tmp = self._tombstone_path().with_suffix(".tmp")
            tmp.write_text(json.dumps(sorted(dead), ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._tombstone_path())
            self._after_write(tbl)
        logger.info("湖表删除记录 %d 条（%s）", len(out), "含关联记录" if cascade else "不含关联记录")
        return out

    def set_gap_ignored(self, question_id: str, ignored: bool = True) -> bool:
        """把一个问题标记为「忽略」（不再算知识缺口，记录保留，可恢复）。问题记录的 status 平时都是 ok。"""
        rec = self.get(question_id)
        if rec is None or rec.get("kind") != "question":
            return False
        return self.update(question_id, status="ignored" if ignored else "ok")

    def _existing_ids(self, tbl, ids: list[str]) -> set[str]:
        if not ids or tbl.count_rows() == 0:
            return set()
        out: set[str] = set()
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            got = tbl.search().where(sql_in("id", part)).select(["id"]).limit(len(part)).to_arrow()
            out.update(got.column("id").to_pylist())
        return out

    def update(self, record_id: str, **values: Any) -> bool:
        tbl = self._table()
        if tbl is None or not self._existing_ids(tbl, [record_id]):
            return False
        clean = {k: (_json(v, "") if k in ("meta", "refs") else v) for k, v in values.items() if k in BASE_COLUMNS and k not in ("id", "embed_model")}
        if clean:
            with write_lock(self.directory, self.table_name):
                tbl.update(where=f"id = {sql_str(record_id)}", values=clean)
        return True

    def delete(self, where: str) -> int:
        tbl = self._table()
        if tbl is None:
            return 0
        before = tbl.count_rows()
        with write_lock(self.directory, self.table_name):
            tbl.delete(where)
        return before - tbl.count_rows()

    # ------------------------------------------------------------------ 读取
    def columns(self, tbl=None) -> list[str]:
        tbl = tbl if tbl is not None else self._table()
        return [] if tbl is None else [f.name for f in tbl.schema if f.name != "vector"]

    def derived_columns(self, tbl=None) -> list[dict[str, str]]:
        tbl = tbl if tbl is not None else self._table()
        return [] if tbl is None else [{"name": f.name, "type": str(f.type)} for f in tbl.schema if f.name not in BASE_COLUMNS and f.name != "vector"]

    def get(self, record_id: str) -> dict[str, Any] | None:
        tbl = self._table()
        if tbl is None:
            return None
        got = tbl.search().where(f"id = {sql_str(record_id)}").select(self.columns(tbl)).limit(1).to_list()
        return got[0] if got else None

    def where_clause(self, *, kinds: list[str] | None = None, session_id: str | None = None, user_id: str | None = None, kb_id: str | None = None,
                     parent_ids: list[str] | None = None, extra: str | None = None) -> str | None:
        parts = []
        if kinds:
            parts.append(sql_in("kind", kinds))
        if session_id:
            parts.append(f"session_id = {sql_str(session_id)}")
        if user_id:
            parts.append(f"user_id = {sql_str(user_id)}")
        if kb_id:
            parts.append(f"kb_id = {sql_str(kb_id)}")
        if parent_ids is not None:
            parts.append(sql_in("parent_id", parent_ids))
        if extra:
            parts.append(f"({extra})")
        return " AND ".join(parts) or None

    def list(self, *, where: str | None = None, limit: int = 50, offset: int = 0, view=None) -> tuple[list[dict[str, Any]], int]:
        """按时间倒序列出（Lance 扫描不带 ORDER BY：先取过滤后的 id + ts 排序，再按页取整行）。"""
        tbl = view if view is not None else self._table()
        if tbl is None:
            return [], 0
        q = tbl.search()
        if where:
            q = q.where(where)
        keys = q.select(["id", "ts"]).limit(None).to_arrow()
        total = keys.num_rows
        if not total:
            return [], 0
        order = np.argsort(-np.asarray(keys.column("ts").to_numpy(zero_copy_only=False), dtype=float))[offset:offset + limit]
        ids = [keys.column("id")[int(i)].as_py() for i in order]
        rows = tbl.search().where(sql_in("id", ids)).select(self.columns(tbl)).limit(len(ids)).to_list() if ids else []
        by_id = {r["id"]: r for r in rows}
        return [by_id[i] for i in ids if i in by_id], total

    def to_arrow(self, *, view=None, with_vector: bool = False):
        tbl = view if view is not None else self._table()
        if tbl is None:
            import pyarrow as pa

            return _schema(self.dim).empty_table() if with_vector else _schema(self.dim).empty_table().drop_columns(["vector"])
        data = tbl.to_arrow()
        return data if with_vector else data.drop_columns(["vector"])

    # ------------------------------------------------------------------ 检索：向量 + 全文 + RRF
    def search(self, query: str, *, top_k: int = 10, mode: str = "hybrid", where: str | None = None, view=None, rrf_k: int = 60) -> list[dict[str, Any]]:
        """mode: hybrid（向量 + 全文 RRF）/ vector / fts。返回每条命中的各路分数，便于解释。"""
        tbl = view if view is not None else self._table()
        if tbl is None or not query.strip() or tbl.count_rows() == 0:
            return []
        cols = self.columns(tbl)
        n = max(top_k * 4, 20)
        routes: dict[str, list[dict[str, Any]]] = {}
        if mode in ("hybrid", "vector"):
            qv = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
            b = tbl.search(qv, vector_column_name="vector").distance_type("cosine").select(cols).limit(n)
            if where:
                b = b.where(where, prefilter=True)
            hits = b.to_list()
            for h in hits:
                h["_vector"] = max(0.0, 1.0 - float(h.pop("_distance", 1.0)))
            routes["vector"] = hits
        if mode in ("hybrid", "fts") and has_index(tbl, "content"):
            try:
                b = tbl.search(query, query_type="fts", fts_columns="content").select(cols).limit(n)
                if where:
                    b = b.where(where, prefilter=True)
                hits = b.to_list()
                for h in hits:
                    h["_fts"] = float(h.pop("_score", 0.0))
                routes["fts"] = hits
            except Exception as exc:
                logger.debug("全文检索失败：%s", exc)
        merged: dict[str, dict[str, Any]] = {}
        for route, hits in routes.items():
            for rank, h in enumerate(hits, start=1):
                item = merged.setdefault(h["id"], {**{k: v for k, v in h.items() if not k.startswith("_")}, "vectorScore": None, "ftsScore": None, "rrf": 0.0, "routes": []})
                item["rrf"] += 1.0 / (rrf_k + rank)
                item["routes"].append(route)
                if route == "vector":
                    item["vectorScore"] = round(h["_vector"], 4)
                else:
                    item["ftsScore"] = round(h["_fts"], 4)
        out = sorted(merged.values(), key=lambda x: -x["rrf"])[:top_k]
        for x in out:
            x["rrf"] = round(x["rrf"], 5)
        return out

    def vector_search(self, query: str, *, top_k: int, where: str | None = None) -> list[dict[str, Any]]:
        return self.search(query, top_k=top_k, mode="vector", where=where)

    # ------------------------------------------------------------------ 版本 / Tag / 分支
    def versions(self, limit: int = 50) -> list[dict[str, Any]]:
        tbl = self._table()
        if tbl is None:
            return []
        tags_by_version: dict[int, list[str]] = {}
        for name, info in (tbl.tags.list() or {}).items():
            if not name.startswith(_GUARD_TAG):
                tags_by_version.setdefault(int(info.get("version")), []).append(name)
        out = version_list(tbl, limit)
        for v in out:
            v["tags"] = sorted(tags_by_version.get(int(v["version"]), []))
        return out

    def tags(self) -> list[dict[str, Any]]:
        tbl = self._table()
        if tbl is None:
            return []
        return sorted(({"name": k, "version": int(v.get("version"))} for k, v in (tbl.tags.list() or {}).items() if not k.startswith(_GUARD_TAG)), key=lambda t: -t["version"])

    def create_tag(self, name: str, version: int | None = None) -> dict[str, Any]:
        tbl = self._table(create=True)
        name = self._check_ref_name(name)
        v = int(version or tbl.version)
        with write_lock(self.directory, self.table_name):
            tbl.tags.create(name, v)
        return {"name": name, "version": v}

    def delete_tag(self, name: str) -> None:
        tbl = self._table()
        if tbl is not None:
            tbl.tags.delete(name)

    def delete_versions(self, upto: int) -> dict[str, Any]:
        """删除主表 v{upto} 及更早的全部版本并释放磁盘（不影响当前数据）。最新版本不可删；被删范围内有 Tag 时拒绝（先删 Tag）；
        分支的起点版本 Lance 会自动保留。

        Lance 只能"清理早于某时刻的版本"，而且截止时刻在 optimize 内部按"调用时的 now - older_than（毫秒取整）"计算，
        optimize 本身要几毫秒——连续写入的相邻版本只差 1～4 ms，一次算准做不到。所以先给下一个版本临时打一个保护 Tag：
        截止时刻越界时 Lance 整体拒绝、什么都不删；没删够就往后挪。二分几次即可精确删到 v{upto}。
        （社区版没装 pylance，cleanup_old_versions 用不了，只能走 optimize，顺带合并小文件，可能多出一个新版本。）"""
        tbl = self._table()
        if tbl is None:
            raise ValueError("湖表还不存在")
        with write_lock(self.directory, self.table_name):
            stamps = {int(v["version"]): v["timestamp"].astimezone(timezone.utc) for v in tbl.list_versions()}
            if upto not in stamps:
                raise ValueError(f"版本 v{upto} 不存在（可能已被清理）")
            newer = sorted(v for v in stamps if v > upto)
            if not newer or upto >= tbl.version:
                raise ValueError("最新版本不能删除")
            tagged = sorted((name, int(info.get("version"))) for name, info in (tbl.tags.list() or {}).items() if int(info.get("version")) <= upto)
            if tagged:
                raise ValueError("要删除的版本上有 Tag：" + "、".join(f"{n}（v{v}）" for n, v in tagged) + "。先删除这些 Tag 再清理")
            branch_points = {int(info["parent_version"]) for info in (tbl.branches.list() or {}).values() if info.get("parent_version") is not None}
            before = set(stamps)
            removable = [v for v in stamps if v <= upto and v not in branch_points]
            if removable:
                self._cleanup_through(tbl, max(removable), guard_version=newer[0], stamps=stamps)
            after = {int(v["version"]) for v in tbl.list_versions()}
        deleted = sorted(before - after)
        kept = sorted(v for v in after if v <= upto)          # 分支起点
        logger.info("湖表清理版本 ≤ v%s：删除 %d 个，保留 %s", upto, len(deleted), kept)
        return {"deleted": deleted, "kept": kept, "version": tbl.version}

    @staticmethod
    def _cleanup_through(tbl, target: int, *, guard_version: int, stamps: dict[int, datetime]) -> None:
        guard = f"{_GUARD_TAG}{uuid.uuid4().hex[:8]}"
        tbl.tags.create(guard, guard_version)
        try:
            a = stamps[target]
            lo, hi = -0.5, (stamps[guard_version] - a).total_seconds()     # 相对 v{target} 时间戳的偏移（秒）
            for _ in range(24):
                mid = (lo + hi) / 2
                try:
                    tbl.optimize(cleanup_older_than=datetime.now(timezone.utc) - (a + timedelta(seconds=mid)))
                except Exception as exc:
                    if guard not in str(exc):
                        raise
                    hi = mid                                   # 越界碰到保护 Tag：整体被拒，什么都没删
                    continue
                if target not in {int(v["version"]) for v in tbl.list_versions()}:
                    return
                lo = mid                                       # 还没删到 target：截止时刻往后挪
            raise RuntimeError(f"未能精确清理到 v{target}（相邻版本间隔过近），请重试")
        finally:
            tbl.tags.delete(guard)

    def branches(self) -> list[dict[str, Any]]:
        tbl = self._table()
        if tbl is None:
            return []
        out = []
        for name, info in (tbl.branches.list() or {}).items():
            b = tbl.branches.checkout(name)
            out.append({"name": name, "parentBranch": info.get("parent_branch") or "main", "parentVersion": info.get("parent_version"),
                        "version": b.version, "rows": b.count_rows(), "derivedColumns": [c["name"] for c in self.derived_columns(b)]})
        return out

    def create_branch(self, name: str, *, from_tag: str | None = None, from_version: int | None = None) -> dict[str, Any]:
        tbl = self._table(create=True)
        name = self._check_ref_name(name)
        if from_tag:
            from_version = int(tbl.tags.get_version(from_tag))
        with write_lock(self.directory, self.table_name):
            b = tbl.branches.create(name, from_version=int(from_version or tbl.version))
        return {"name": name, "version": b.version, "rows": b.count_rows()}

    def delete_branch(self, name: str) -> None:
        tbl = self._table()
        if tbl is not None:
            tbl.branches.delete(name)

    def diff_branch(self, name: str, sample: int = 8) -> dict[str, Any]:
        """分支 vs 主表：新增 / 删除的列、只在一边的行、派生列的取值分布（社区版 LanceDB 没有 branch diff，这里按 id 自己比）。"""
        main, br = self._table(), self.view(branch=name)
        mc, bc = set(self.columns(main)), set(self.columns(br))
        mids = set(main.search().select(["id"]).limit(None).to_arrow().column("id").to_pylist())
        bdata = br.to_arrow().drop_columns(["vector"])
        bids = set(bdata.column("id").to_pylist())
        added_cols = sorted(bc - mc)
        dist: dict[str, list[dict[str, Any]]] = {}
        for c in added_cols:
            vals: dict[str, int] = {}
            for v in bdata.column(c).to_pylist():
                key = "（空）" if v is None else str(v)
                vals[key] = vals.get(key, 0) + 1
            dist[c] = [{"value": k, "count": n} for k, n in sorted(vals.items(), key=lambda kv: -kv[1])[:sample]]
        return {"branch": name, "addedColumns": added_cols, "removedColumns": sorted(mc - bc), "rowsOnlyInBranch": len(bids - mids),
                "rowsOnlyInMain": len(mids - bids), "mainRows": len(mids), "branchRows": len(bids), "columnValues": dist}

    def promote_branch(self, name: str, columns: list[str] | None = None) -> dict[str, Any]:
        """把分支上验证过的派生列合并回主表（按 id 追加列，不动主表其他数据）；只在分支里新增的行也一并写入。"""
        import pyarrow as pa

        main, br = self._table(), self.view(branch=name)
        cols = columns or [c["name"] for c in self.derived_columns(br)]
        bschema = br.schema
        missing = [c for c in cols if c not in bschema.names]
        if missing:
            raise ValueError(f"分支 {name} 上没有这些列：{', '.join(missing)}")
        with write_lock(self.directory, self.table_name):
            for c in cols:
                if c not in main.schema.names:
                    main.add_columns(bschema.field(c))
            data = br.to_arrow()
            mids = set(main.search().select(["id"]).limit(None).to_arrow().column("id").to_pylist())
            in_main = pa.compute.is_in(data["id"], pa.array(list(mids), type=pa.string()))
            updated = 0
            if cols:
                part = data.filter(in_main).select(["id", *cols])
                if part.num_rows:
                    main.merge_insert("id").when_matched_update_all().execute(part)
                    updated = part.num_rows
            fresh = data.filter(pa.compute.invert(in_main))
            if fresh.num_rows:
                main.add(fresh.select(main.schema.names).cast(main.schema))
        return {"branch": name, "columns": cols, "updatedRows": updated, "insertedRows": fresh.num_rows, "version": main.version}

    @staticmethod
    def _check_ref_name(name: str) -> str:
        name = (name or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
            raise ValueError("名称只能包含字母、数字、点、下划线、连字符（不超过 64 个字符）")
        return name

    # ------------------------------------------------------------------ 派生列（宽表加列）
    def add_column(self, name: str, *, values: dict[str, Any] | None = None, expr: str | None = None, dtype: str = "string", branch: str | None = None) -> dict[str, Any]:
        """追加一列：expr 是按已有列计算的 SQL 表达式（DuckDB 方言，可用 CASE / json_extract_string 等）；
        values 是 {id: 值}（模型打分 / 聚类标签等外部结果）。新列作为新的数据文件附加到现有分片上，历史数据不重写。"""
        import pyarrow as pa

        if not _COL_NAME.match(name or "") or name in BASE_COLUMNS or name == "vector":
            raise ValueError("列名需为小写字母开头的 snake_case，且不能与基础列重名")
        tbl = self.view(branch=branch) if branch else self._table()
        if tbl is None:
            raise ValueError("Agent 湖还是空的")
        if expr:
            data = self._eval_expr(tbl, name, expr)
        else:
            arrow_type = pa.string() if dtype == "string" else pa.from_numpy_dtype(np.dtype(_COL_TYPES[dtype]))
            if name in tbl.schema.names:
                arrow_type = tbl.schema.field(name).type
            values = values or {}
            data = pa.table({"id": pa.array(list(values), type=pa.string()), name: pa.array(list(values.values()), type=arrow_type)})
        with write_lock(self.directory, self.table_name):
            if expr and name in tbl.schema.names:
                tbl.drop_columns([name])          # 表达式列重算：类型可能变了
            if name not in tbl.schema.names:
                tbl.add_columns(data.schema.field(name))
            if data.num_rows:
                tbl.merge_insert("id").when_matched_update_all().execute(data.cast(pa.schema([pa.field("id", pa.string()), tbl.schema.field(name)])))
        filled = sum(1 for v in data.column(name).to_pylist() if v is not None)
        return {"column": name, "branch": branch or "main", "filled": filled, "version": tbl.version}

    @staticmethod
    def _eval_expr(tbl, name: str, expr: str):
        """在 DuckDB 里对整表求值表达式，得到 (id, 新列)。禁用外部访问，表达式只能引用表里的列。"""
        import duckdb

        from .analytics import check_sql

        check_sql(f"SELECT {expr} FROM lake")          # 同一套只读校验：单条语句、无写操作关键字
        con = duckdb.connect(config={"enable_external_access": False, "lock_configuration": True})
        try:
            con.register("lake", tbl.to_arrow().drop_columns(["vector"]))
            return con.execute(f'SELECT id, ({expr}) AS "{name}" FROM lake').fetch_arrow_table()
        except duckdb.Error as exc:
            raise ValueError(f"表达式无法计算：{exc}") from exc
        finally:
            con.close()

    def drop_column(self, name: str, *, branch: str | None = None) -> None:
        if name in BASE_COLUMNS or name == "vector":
            raise ValueError("基础列不能删除")
        tbl = self.view(branch=branch) if branch else self._table()
        if tbl is None or name not in tbl.schema.names:
            raise ValueError(f"{'分支 ' + branch if branch else '主表'}上没有列 {name}")
        with write_lock(self.directory, self.table_name):
            tbl.drop_columns([name])

    # ------------------------------------------------------------------ 概况
    def info(self) -> dict[str, Any]:
        tbl = self._table()
        if tbl is None:
            return {"exists": False, "rows": 0, "dim": self.dim, "embedModel": self.model_name}
        info = table_info(tbl)
        info.update({"exists": True, "dim": self.dim, "embedModel": self.model_name, "derivedColumns": self.derived_columns(tbl),
                     "tags": sum(1 for k in (tbl.tags.list() or {}) if not k.startswith(_GUARD_TAG)), "branches": len(tbl.branches.list() or {})})
        return info

    def optimize(self) -> dict[str, Any]:
        tbl = self._table()
        if tbl is None:
            return {}
        with write_lock(self.directory, self.table_name):
            tbl.optimize()
        return table_info(tbl)
