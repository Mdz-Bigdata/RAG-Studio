"""长期记忆：LanceDB 表 memories（用户 / 智能体维度），向量召回 × 新近度 × 重要度综合排序。

一条记忆 = 一句可以独立理解的陈述（"用户在财务部"、"用户希望回答先给结论"）。
- 写入去重：与同一用户 / 智能体已有记忆的余弦相似度 ≥ dedup_threshold 时合并更新（新内容覆盖旧内容），不重复堆积；
- 召回：SQL 预过滤（用户、智能体、类型、未过期）→ 向量近邻 → score = w_sim·相似度 + w_rec·新近度 + w_imp·重要度；
- 遗忘：过期删除；长期没被用到、重要度又低的记忆按衰减分清理；
- 每次写入都是 LanceDB 的一个新版本：记忆的增删改可回看。
向量模型换了（维度 / 模型名变化）时自动用保存的原文重算向量。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import MemoryConfig
from ..index.embeddings import BaseEmbedder
from ..lance import connect, open_table, sql_in, sql_str, table_info, version_list, write_lock

logger = logging.getLogger(__name__)

TABLE = "memories"
KINDS = ("profile", "preference", "fact", "episode", "task")
_DAY = 86400.0


@dataclass
class MemoryRecord:
    id: str
    content: str
    user_id: str = "default"
    agent_id: str = "assistant"
    kind: str = "fact"
    importance: float = 0.5
    source: str = "manual"
    session_id: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    last_access: float = 0.0
    access_count: int = 0
    expires_at: float = 0.0          # 0 = 不过期

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _schema(dim: int):
    import pyarrow as pa

    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("user_id", pa.string()),
        pa.field("agent_id", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("importance", pa.float32()),
        pa.field("source", pa.string()),
        pa.field("session_id", pa.string()),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("created_at", pa.float64()),
        pa.field("updated_at", pa.float64()),
        pa.field("last_access", pa.float64()),
        pa.field("access_count", pa.int32()),
        pa.field("expires_at", pa.float64()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])


_COLUMNS = [f for f in MemoryRecord.__dataclass_fields__]


def _row_to_record(row: dict[str, Any]) -> MemoryRecord:
    rec = MemoryRecord(**{k: row[k] for k in _COLUMNS if k in row and row[k] is not None})
    rec.importance = round(float(rec.importance), 4)    # float32 列读回来是 0.800000011920929
    return rec


class LanceMemoryStore:
    def __init__(self, directory: str | Path, embedder: BaseEmbedder, cfg: MemoryConfig | None = None) -> None:
        self.directory = Path(directory)
        self.embedder = embedder
        self.cfg = cfg or MemoryConfig()
        self._meta_path = self.directory / f"{TABLE}.meta.json"
        self._tbl = None
        self._check_embedder()

    # ----------------------------------------------------------------------------------
    @property
    def db(self):
        return connect(self.directory)

    def _signature(self) -> dict[str, Any]:
        return {"embedder": self.embedder.name, "model": getattr(self.embedder, "model_name", None), "dim": self._dim()}

    def _dim(self) -> int:
        return int(self.embedder.dim or len(self.embedder.embed_documents(["维度探测"])[0]))

    def _table(self, create: bool = False):
        if self._tbl is None:
            self._tbl = open_table(self.db, TABLE)
            if self._tbl is None and create:
                self._tbl = self.db.create_table(TABLE, schema=_schema(self._dim()), exist_ok=True)
                self._write_meta()
        return self._tbl

    def _write_meta(self) -> None:
        self._meta_path.write_text(json.dumps(self._signature(), ensure_ascii=False), encoding="utf-8")

    def _check_embedder(self) -> None:
        """向量模型换了：用保存的原文重算全部记忆的向量（记忆量小，直接整表重写）。"""
        tbl = self._table()
        if tbl is None:
            return
        built = json.loads(self._meta_path.read_text(encoding="utf-8")) if self._meta_path.exists() else {}
        sig = self._signature()
        if (built.get("embedder"), built.get("model"), built.get("dim")) == (sig["embedder"], sig["model"], sig["dim"]):
            return
        rows = tbl.to_arrow().to_pylist()
        logger.info("向量模型已更换（%s → %s），重算 %d 条记忆的向量", built.get("model"), sig["model"], len(rows))
        vecs = self.embedder.embed_documents([r["content"] for r in rows]) if rows else np.zeros((0, sig["dim"]), dtype=np.float32)
        with write_lock(self.directory, TABLE):
            self._tbl = self.db.create_table(TABLE, self._to_arrow([_row_to_record(r) for r in rows], vecs), schema=_schema(sig["dim"]), mode="overwrite")
        self._write_meta()

    def _to_arrow(self, records: list[MemoryRecord], vectors: np.ndarray):
        import pyarrow as pa

        dim = vectors.shape[1] if vectors.ndim == 2 and vectors.shape[0] else self._dim()
        cols: dict[str, list[Any]] = {k: [getattr(r, k) for r in records] for k in _COLUMNS}
        cols["vector"] = list(np.asarray(vectors, dtype=np.float32).reshape(len(records), dim)) if records else []
        return pa.Table.from_pydict({k: v for k, v in cols.items()}, schema=_schema(dim))

    # ----------------------------------------------------------------------------------
    def add(self, content: str, *, user_id: str = "default", agent_id: str = "assistant", kind: str = "fact", importance: float = 0.5,
            source: str = "manual", session_id: str = "", tags: list[str] | None = None, ttl_days: float | None = None) -> tuple[MemoryRecord, str]:
        """写入一条记忆；与已有记忆近似重复时合并更新。返回 (记录, "created" | "updated")。"""
        content = " ".join(content.split()).strip()
        if not content:
            raise ValueError("记忆内容为空")
        kind = kind if kind in KINDS else "fact"
        importance = float(min(1.0, max(0.0, importance)))
        now = time.time()
        vec = self.embedder.embed_documents([content])[0]
        tbl = self._table(create=True)
        with write_lock(self.directory, TABLE):
            dup = self._nearest(vec, user_id=user_id, agent_id=agent_id)
            if dup is not None and dup[1] >= self.cfg.dedup_threshold:
                old = dup[0]
                rec = MemoryRecord(**{**old.to_dict(), "content": content, "importance": max(old.importance, importance), "updated_at": now,
                                      "tags": sorted(set(old.tags) | set(tags or [])), "kind": kind if importance >= old.importance else old.kind,
                                      "expires_at": now + ttl_days * _DAY if ttl_days else old.expires_at})
                tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(self._to_arrow([rec], vec[None, :]))
                return rec, "updated"
            rec = MemoryRecord(id=uuid.uuid4().hex, content=content, user_id=user_id, agent_id=agent_id, kind=kind, importance=importance,
                               source=source, session_id=session_id or "", tags=sorted(set(tags or [])), created_at=now, updated_at=now,
                               last_access=0.0, access_count=0, expires_at=now + ttl_days * _DAY if ttl_days else 0.0)
            tbl.add(self._to_arrow([rec], vec[None, :]))
            return rec, "created"

    def _nearest(self, vec: np.ndarray, *, user_id: str, agent_id: str) -> tuple[MemoryRecord, float] | None:
        tbl = self._table()
        if tbl is None or not tbl.count_rows():
            return None
        t = (tbl.search(np.asarray(vec, dtype=np.float32)).distance_type("cosine").where(f"user_id = {sql_str(user_id)} AND agent_id = {sql_str(agent_id)}", prefilter=True)
             .select(_COLUMNS + ["_distance"]).limit(1).to_list())
        if not t:
            return None
        return _row_to_record(t[0]), 1.0 - float(t[0]["_distance"])

    def search(self, query: str, *, user_id: str = "default", agent_id: str | None = None, kinds: list[str] | None = None,
               top_k: int | None = None, min_score: float | None = None, touch: bool = True) -> list[dict[str, Any]]:
        """召回：返回 [{record, score, similarity, recency, importance}]，按综合分降序。"""
        tbl = self._table()
        if tbl is None or not query.strip():
            return []
        top_k = top_k or self.cfg.recall_top_k
        min_score = self.cfg.min_score if min_score is None else min_score
        now = time.time()
        where = [f"user_id = {sql_str(user_id)}", f"(expires_at = 0 OR expires_at > {now})"]
        if agent_id:
            where.append(f"agent_id = {sql_str(agent_id)}")
        if kinds:
            where.append(sql_in("kind", kinds))
        vec = self.embedder.embed_queries([query])[0]
        rows = (tbl.search(np.asarray(vec, dtype=np.float32)).distance_type("cosine").where(" AND ".join(where), prefilter=True)
                .select(_COLUMNS + ["_distance"]).limit(max(top_k * 4, 20)).to_list())
        hits = []
        for r in rows:
            rec = _row_to_record(r)
            sim = 1.0 - float(r["_distance"])
            recency = 0.5 ** (max(0.0, now - max(rec.updated_at, rec.last_access)) / (self.cfg.half_life_days * _DAY))
            score = self.cfg.w_similarity * sim + self.cfg.w_recency * recency + self.cfg.w_importance * rec.importance
            hits.append({"record": rec, "score": score, "similarity": sim, "recency": recency, "importance": rec.importance})
        hits.sort(key=lambda h: h["score"], reverse=True)
        hits = [h for h in hits if h["score"] >= min_score][:top_k]
        if touch and hits:
            self.touch([h["record"].id for h in hits])
        return hits

    def vectors(self, *, user_id: str, kinds: list[str] | None = None, limit: int = 20) -> list[np.ndarray]:
        """某用户的记忆向量（推荐系统把偏好 / 身份记忆当作长期兴趣）。"""
        tbl = self._table()
        if tbl is None:
            return []
        where = f"user_id = {sql_str(user_id)} AND (expires_at = 0 OR expires_at > {time.time()})" + (f" AND {sql_in('kind', kinds)}" if kinds else "")
        rows = tbl.search().where(where).select(["vector", "importance"]).limit(limit).to_list()
        return [np.asarray(r["vector"], dtype=np.float32) for r in rows]

    def touch(self, ids: list[str]) -> None:
        tbl = self._table()
        if tbl is None or not ids:
            return
        with write_lock(self.directory, TABLE):
            tbl.update(where=sql_in("id", ids), values_sql={"access_count": "access_count + 1", "last_access": repr(time.time())})

    def get(self, memory_id: str) -> MemoryRecord | None:
        tbl = self._table()
        if tbl is None:
            return None
        rows = tbl.search().where(f"id = {sql_str(memory_id)}").limit(1).to_list()
        return _row_to_record(rows[0]) if rows else None

    def list(self, *, user_id: str | None = None, agent_id: str | None = None, kind: str | None = None, keyword: str | None = None, limit: int = 200) -> list[MemoryRecord]:
        tbl = self._table()
        if tbl is None:
            return []
        where = []
        if user_id:
            where.append(f"user_id = {sql_str(user_id)}")
        if agent_id:
            where.append(f"agent_id = {sql_str(agent_id)}")
        if kind:
            where.append(f"kind = {sql_str(kind)}")
        if keyword:
            where.append(f"content LIKE {sql_str('%' + keyword.replace('%', '').replace('_', '') + '%')}")
        q = tbl.search()
        if where:
            q = q.where(" AND ".join(where))
        rows = q.select(_COLUMNS).limit(10_000).to_list()
        recs = sorted((_row_to_record(r) for r in rows), key=lambda r: r.updated_at, reverse=True)
        return recs[:limit]

    def update(self, memory_id: str, *, content: str | None = None, kind: str | None = None, importance: float | None = None, tags: list[str] | None = None) -> MemoryRecord:
        rec = self.get(memory_id)
        if rec is None:
            raise KeyError(memory_id)
        if content is not None and content.strip():
            rec.content = " ".join(content.split())
        if kind in KINDS:
            rec.kind = kind
        if importance is not None:
            rec.importance = float(min(1.0, max(0.0, importance)))
        if tags is not None:
            rec.tags = sorted(set(tags))
        rec.updated_at = time.time()
        vec = self.embedder.embed_documents([rec.content])
        with write_lock(self.directory, TABLE):
            self._table().merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(self._to_arrow([rec], vec))
        return rec

    def delete(self, memory_id: str) -> bool:
        tbl = self._table()
        if tbl is None:
            return False
        with write_lock(self.directory, TABLE):
            return tbl.delete(f"id = {sql_str(memory_id)}").num_deleted_rows > 0

    def clear(self, user_id: str) -> int:
        tbl = self._table()
        if tbl is None:
            return 0
        with write_lock(self.directory, TABLE):
            return tbl.delete(f"user_id = {sql_str(user_id)}").num_deleted_rows

    def forget(self, *, user_id: str | None = None, min_keep_score: float = 0.15) -> dict[str, int]:
        """清理：过期的直接删；其余按"重要度 × 新近度"衰减分，低于 min_keep_score 的删掉（用户明确要求记住的不受影响：重要度高）。"""
        tbl = self._table()
        if tbl is None:
            return {"expired": 0, "decayed": 0}
        now = time.time()
        scope = f" AND user_id = {sql_str(user_id)}" if user_id else ""
        with write_lock(self.directory, TABLE):
            expired = tbl.delete(f"expires_at > 0 AND expires_at <= {now}{scope}").num_deleted_rows
            stale = [r for r in self.list(user_id=user_id, limit=100_000)
                     if r.importance * 0.5 ** ((now - max(r.updated_at, r.last_access)) / (self.cfg.half_life_days * _DAY)) < min_keep_score]
            decayed = tbl.delete(sql_in("id", [r.id for r in stale])).num_deleted_rows if stale else 0
        return {"expired": expired, "decayed": decayed}

    # ----------------------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        tbl = self._table()
        if tbl is None:
            return {"rows": 0, "users": 0, "byKind": {}, "byUser": {}}
        rows = tbl.search().select(["user_id", "kind", "agent_id"]).limit(1_000_000).to_list()
        by_kind: dict[str, int] = {}
        by_user: dict[str, int] = {}
        by_agent: dict[str, int] = {}
        for r in rows:
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
            by_user[r["user_id"]] = by_user.get(r["user_id"], 0) + 1
            by_agent[r["agent_id"]] = by_agent.get(r["agent_id"], 0) + 1
        return {"rows": len(rows), "users": len(by_user), "byKind": by_kind, "byUser": by_user, "byAgent": by_agent, "table": table_info(tbl), **self._signature()}

    def versions(self, limit: int = 30) -> list[dict[str, Any]]:
        tbl = self._table()
        return version_list(tbl, limit) if tbl is not None else []
