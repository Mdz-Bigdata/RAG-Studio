"""Agent 湖 API：概况 / 记录 / 混合检索 / 检索质量与知识缺口 / 主题 / SQL / 分析智能体 / 版本 · Tag · 分支 / 派生列 / 回填 / 反馈。

查询类接口都支持 ?version=（版本号或 Tag）与 ?branch=，在历史快照或实验分支上分析。
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..services.lake_service import lake_service

router = APIRouter()


def _err(exc: Exception) -> HTTPException:
    return HTTPException(400, str(exc))


class SearchBody(BaseModel):
    query: str
    kinds: list[str] | None = None
    mode: Literal["hybrid", "vector", "fts"] = "hybrid"
    topK: int = Field(10, ge=1, le=100)
    sessionId: str | None = None
    userId: str | None = None
    kbId: str | None = None
    version: str | None = None
    branch: str | None = None


class SqlBody(BaseModel):
    sql: str
    version: str | None = None
    branch: str | None = None


class TopicsBody(BaseModel):
    k: int | None = Field(None, ge=1, le=20)
    writeColumn: str | None = None
    version: str | None = None
    branch: str | None = None


class AgentBody(BaseModel):
    question: str
    version: str | None = None
    branch: str | None = None
    userId: str = "default"


class TagBody(BaseModel):
    name: str
    version: int | None = None


class BranchBody(BaseModel):
    name: str
    fromTag: str | None = None
    fromVersion: int | None = None


class PromoteBody(BaseModel):
    columns: list[str] | None = None


class ColumnBody(BaseModel):
    name: str
    expr: str | None = None              # SQL 表达式，例如 "CASE WHEN score < 0.05 THEN 'low' ELSE 'ok' END"
    values: dict[str, Any] | None = None
    dtype: Literal["string", "double", "int", "bool"] = "string"
    branch: str | None = None


class NoteBody(BaseModel):
    content: str
    title: str = ""
    kbId: str | None = None
    userId: str = "default"


class FeedbackBody(BaseModel):
    answerId: str
    value: int = Field(..., ge=-1, le=1)


class RetestBody(BaseModel):
    limit: int = Field(10, ge=1, le=50)
    includeWeak: bool = False            # 连同"答了但检索分偏低"的问题一起复测
    checkAnswer: bool = True             # 复测时再生成一次回答，看是否拒答（更准，但每题多一次模型调用）


class SyncBody(BaseModel):
    sources: list[Literal["chat", "documents", "memories", "board"]] | None = None


@router.get("/overview")
def overview(version: str | None = None, branch: str | None = None, days: int = 30):
    try:
        return lake_service.overview(version, branch, days)
    except Exception as exc:
        raise _err(exc) from exc


@router.get("/records")
def records(kind: str | None = None, sessionId: str | None = None, parentId: str | None = None, limit: int = Query(50, le=500), offset: int = 0,
            version: str | None = None, branch: str | None = None):
    lake = lake_service.lake()
    kinds = [k.strip() for k in (kind or "").split(",") if k.strip()]       # kind=question,answer 多选
    where = lake.where_clause(kinds=kinds or None, session_id=sessionId, parent_ids=[parentId] if parentId else None)
    items, total = lake.list(where=where, limit=limit, offset=offset, view=lake.view(version=version, branch=branch))
    return {"items": items, "total": total}


@router.get("/records/{record_id}")
def record(record_id: str):
    lake = lake_service.lake()
    rec = lake.get(record_id)
    if rec is None:
        raise HTTPException(404, "记录不存在")
    children, _ = lake.list(where=lake.where_clause(parent_ids=[record_id]), limit=50)
    parent = lake.get(rec["parent_id"]) if rec.get("parent_id") else None
    return {"record": rec, "parent": parent, "children": children}


@router.delete("/records/{record_id}")
def delete_record(record_id: str, cascade: bool = True, dryRun: bool = False):
    """删除一条记录（默认连同它的关联后代记录）；dryRun=true 只返回将被删除的 id。删除后回填历史数据不会再写回来。"""
    deleted = lake_service.lake().delete_records([record_id], cascade=cascade, dry_run=dryRun)
    if not deleted:
        raise HTTPException(404, "记录不存在（可能已被删除）")
    return {"deleted": deleted, "dryRun": dryRun}


class DeleteRecordsBody(BaseModel):
    ids: list[str] = Field(..., min_length=1, max_length=10_000)
    cascade: bool = True
    dryRun: bool = False


@router.post("/records/delete")
def delete_records(body: DeleteRecordsBody):
    """批量删除记录（如"删除某个主题下的全部问题"），默认连同关联后代；dryRun 只返回将被删除的 id。"""
    deleted = lake_service.lake().delete_records(body.ids, cascade=body.cascade, dry_run=body.dryRun)
    if not deleted:
        raise HTTPException(404, "这些记录都不存在（可能已被删除）")
    return {"deleted": deleted, "dryRun": body.dryRun}


class IgnoreBody(BaseModel):
    ignored: bool = True


@router.post("/gaps/{question_id}/ignore")
def ignore_gap(question_id: str, body: IgnoreBody):
    """忽略 / 恢复一个知识缺口：问题记录标记为 ignored，不再计入缺口（记录保留）。"""
    if not lake_service.lake().set_gap_ignored(question_id, body.ignored):
        raise HTTPException(404, "问题记录不存在")
    return {"ok": True, "ignored": body.ignored}


@router.post("/records")
def add_note(body: NoteBody):
    rec = {"kind": "note", "title": body.title, "content": body.content, "kb_id": body.kbId, "user_id": body.userId, "agent": "user"}
    return lake_service.lake().append([rec])


@router.post("/search")
def search(body: SearchBody):
    lake = lake_service.lake()
    where = lake.where_clause(kinds=body.kinds, session_id=body.sessionId, user_id=body.userId, kb_id=body.kbId)
    try:
        return lake.search(body.query, top_k=body.topK, mode=body.mode, where=where, view=lake.view(version=body.version, branch=body.branch))
    except Exception as exc:
        raise _err(exc) from exc


@router.get("/quality")
def quality(version: str | None = None, branch: str | None = None, days: int | None = None):
    return lake_service.quality(version, branch, days)


@router.post("/topics")
def topics(body: TopicsBody):
    try:
        return lake_service.topics(k=body.k, version=body.version, branch=body.branch, write_column=body.writeColumn)
    except Exception as exc:
        raise _err(exc) from exc


@router.get("/schema")
def schema(version: str | None = None, branch: str | None = None):
    return lake_service.schema(version, branch)


@router.post("/sql")
def sql(body: SqlBody):
    try:
        return lake_service.sql(body.sql, body.version, body.branch)
    except Exception as exc:
        raise _err(exc) from exc


@router.post("/analyze")
def analyze(body: AgentBody):
    try:
        return lake_service.analyze(body.question, version=body.version, branch=body.branch, user_id=body.userId)
    except Exception as exc:
        raise _err(exc) from exc


@router.post("/retest")
def retest(body: RetestBody):
    """复测缺口：用现在的知识库重新检索（并生成回答）未解决的缺口问题，结果写回湖里；之后的缺口统计以复测结果为准。"""
    try:
        return lake_service.retest(limit=body.limit, include_weak=body.includeWeak, check_answer=body.checkAnswer)
    except Exception as exc:
        raise _err(exc) from exc


@router.post("/feedback")
def feedback(body: FeedbackBody):
    if not lake_service.feedback(body.answerId, body.value):
        raise HTTPException(404, "这条回答还没有进入 Agent 湖（可能还在写入，稍后重试）")
    return {"ok": True}


@router.post("/sync")
def sync(body: SyncBody):
    return lake_service.sync(body.sources)


@router.post("/optimize")
def optimize():
    return lake_service.lake().optimize()


# --- 版本 / Tag / 分支 -----------------------------------------------------------------
@router.get("/versions")
def versions(limit: int = 50):
    lake = lake_service.lake()
    return {"versions": lake.versions(limit), "tags": lake.tags(), "branches": lake.branches(), "derivedColumns": [c["name"] for c in lake.derived_columns()]}


@router.delete("/versions/{version}")
def delete_versions(version: int):
    """删除主表 v{version} 及更早的全部版本（Lance 只能按时间整段清理，不能单删中间某个版本）。"""
    try:
        return lake_service.lake().delete_versions(version)
    except Exception as exc:
        raise _err(exc) from exc


@router.post("/tags")
def create_tag(body: TagBody):
    try:
        return lake_service.lake().create_tag(body.name, body.version)
    except Exception as exc:
        raise _err(exc) from exc


@router.delete("/tags/{name}")
def delete_tag(name: str):
    try:
        lake_service.lake().delete_tag(name)
    except Exception as exc:
        raise _err(exc) from exc
    return {"ok": True}


@router.post("/branches")
def create_branch(body: BranchBody):
    try:
        return lake_service.lake().create_branch(body.name, from_tag=body.fromTag, from_version=body.fromVersion)
    except Exception as exc:
        raise _err(exc) from exc


@router.delete("/branches/{name}")
def delete_branch(name: str):
    try:
        lake_service.lake().delete_branch(name)
    except Exception as exc:
        raise _err(exc) from exc
    return {"ok": True}


@router.get("/branches/{name}/diff")
def diff_branch(name: str):
    try:
        return lake_service.lake().diff_branch(name)
    except Exception as exc:
        raise _err(exc) from exc


@router.post("/branches/{name}/promote")
def promote_branch(name: str, body: PromoteBody):
    try:
        return lake_service.lake().promote_branch(name, body.columns)
    except Exception as exc:
        raise _err(exc) from exc


# --- 派生列 --------------------------------------------------------------------------
@router.post("/columns")
def add_column(body: ColumnBody):
    if not body.expr and body.values is None:
        raise HTTPException(400, "需要 expr（SQL 表达式）或 values（{id: 值}）")
    try:
        return lake_service.lake().add_column(body.name, values=body.values, expr=body.expr, dtype=body.dtype, branch=body.branch)
    except Exception as exc:
        raise _err(exc) from exc


@router.delete("/columns/{name}")
def drop_column(name: str, branch: str | None = None):
    try:
        lake_service.lake().drop_column(name, branch=branch)
    except Exception as exc:
        raise _err(exc) from exc
    return {"ok": True}
