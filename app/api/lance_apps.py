"""LanceDB 应用接口：控制台（/lancedb）、长期记忆（/memory）、个性化推荐（/recommend）、多智能体协作（/agents）。"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from prism.agents import ROLES
from prism.recommend import AlreadyVoted

from ..services.agent_service import agent_service
from ..services.lance_service import lance_service
from ..services.memory_service import memory_service
from ..services.recommend_service import recommend_service

lance_router = APIRouter()
memory_router = APIRouter()
recommend_router = APIRouter()
agents_router = APIRouter()


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except AlreadyVoted as exc:     # 每人每条只能评价一次：409，带上之前那次评价，前端据此显示选中状态
        raise HTTPException(409, {"message": str(exc), "vote": exc.vote, "itemId": exc.item_id}) from exc
    except KeyError as exc:
        raise HTTPException(404, f"不存在：{exc.args[0] if exc.args else exc}") from exc
    except (LookupError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


# --- LanceDB 控制台 ------------------------------------------------------------------------
class LanceSearchRequest(BaseModel):
    query: str
    mode: Literal["vector", "exact", "fts", "hybrid"] = "vector"
    k: int = Field(8, ge=1, le=50)
    version: int | None = None
    nprobes: int | None = Field(None, ge=1, le=500)
    refineFactor: int | None = Field(None, ge=0, le=50)
    explain: bool = False


class BuildIndexRequest(BaseModel):
    indexType: Literal["IVF_HNSW_SQ", "IVF_PQ", "IVF_FLAT"] = "IVF_HNSW_SQ"


class MMSearchRequest(BaseModel):
    text: str | None = None
    imageB64: str | None = None
    target: Literal["image", "text", "all"] = "image"
    mode: Literal["vector", "fts", "hybrid"] = "vector"
    k: int = Field(8, ge=1, le=50)


@lance_router.get("/overview")
def lance_overview():
    return lance_service.overview()


@lance_router.get("/tables/{name}/versions")
def lance_table_versions(name: str):
    return _call(lance_service.table_versions, name)


@lance_router.get("/kb/{kb_id}")
def lance_kb_info(kb_id: str):
    return _call(lance_service.kb_info, kb_id)


@lance_router.post("/kb/{kb_id}/mirror")
def lance_kb_mirror(kb_id: str):
    return _call(lance_service.mirror_kb, kb_id)


@lance_router.post("/kb/{kb_id}/search")
def lance_kb_search(kb_id: str, body: LanceSearchRequest):
    return _call(lance_service.kb_search, kb_id, body.query, mode=body.mode, k=body.k, version=body.version, nprobes=body.nprobes, refine_factor=body.refineFactor, explain=body.explain)


@lance_router.post("/kb/{kb_id}/index")
def lance_kb_build_index(kb_id: str, body: BuildIndexRequest):
    return _call(lance_service.kb_build_index, kb_id, body.indexType)


@lance_router.post("/kb/{kb_id}/optimize")
def lance_kb_optimize(kb_id: str):
    return _call(lance_service.kb_optimize, kb_id)


@lance_router.get("/mm/{kb_id}")
def lance_mm_info(kb_id: str):
    return _call(lance_service.mm_info, kb_id)


@lance_router.post("/mm/{kb_id}/sync")
def lance_mm_sync(kb_id: str):
    return _call(lance_service.mm_sync, kb_id)


@lance_router.post("/mm/{kb_id}/search")
def lance_mm_search(kb_id: str, body: MMSearchRequest):
    if not (body.text or "").strip() and not body.imageB64:
        raise HTTPException(400, "请输入文字或上传图片")
    return _call(lance_service.mm_search, kb_id, text=body.text, image_b64=body.imageB64, target=body.target, mode=body.mode, k=body.k)


@lance_router.get("/mm/{kb_id}/thumb/{row_id}")
def lance_mm_thumb(kb_id: str, row_id: str):
    data = lance_service.mm_thumbnail(kb_id, row_id)
    if not data:
        raise HTTPException(404, "没有缩略图")
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


# --- 长期记忆 -------------------------------------------------------------------------------
class MemoryCreate(BaseModel):
    userId: str = "default"
    content: str
    kind: Literal["profile", "preference", "fact", "episode", "task"] = "fact"
    importance: float = Field(0.5, ge=0, le=1)
    agentId: str = "assistant"
    tags: list[str] = Field(default_factory=list)
    ttlDays: float | None = Field(None, gt=0)
    mediaB64: str | None = None              # 附带的图片 / 短视频（data URL 或裸 base64），需要多模态记忆
    mediaName: str | None = None             # 原文件名（取扩展名）；data URL 时可省略


class MemoryUpdate(BaseModel):
    content: str | None = None
    kind: Literal["profile", "preference", "fact", "episode", "task"] | None = None
    importance: float | None = Field(None, ge=0, le=1)
    tags: list[str] | None = None


class MemorySearch(BaseModel):
    userId: str = "default"
    query: str = ""
    imageB64: str | None = None              # 以图召回（需要多模态记忆）
    agentId: str | None = None
    kinds: list[str] | None = None
    topK: int = Field(5, ge=1, le=50)
    minScore: float = 0.0


class MemoryExtract(BaseModel):
    userId: str = "default"
    text: str
    answer: str = ""


class ForgetRequest(BaseModel):
    userId: str | None = None
    minKeepScore: float = Field(0.15, ge=0, le=1)


@memory_router.get("")
def memory_list(userId: str | None = None, agentId: str | None = None, kind: str | None = None, keyword: str | None = None, limit: int = 200):
    return memory_service.publics(memory_service.store().list(user_id=userId, agent_id=agentId, kind=kind, keyword=keyword, limit=limit))


@memory_router.post("")
def memory_add(body: MemoryCreate):
    media = _decode_media(body.mediaB64, body.mediaName) if body.mediaB64 else None
    rec, how = _call(memory_service.store().add, body.content, user_id=body.userId, agent_id=body.agentId, kind=body.kind, importance=body.importance,
                     tags=body.tags, ttl_days=body.ttlDays, source="manual", media=media)
    return {**memory_service.publics([rec])[0], "action": how}


def _decode_media(data: str, name: str | None) -> tuple[bytes, str]:
    """data URL / 裸 base64 → (字节, 扩展名)。扩展名取文件名，没有就看 data URL 的 MIME。"""
    import base64
    import binascii

    head, _, body = data.partition(",") if data.startswith("data:") else ("", "", data)
    ext = (name or "").rsplit(".", 1)[-1].lower() if name and "." in name else head[5:].split(";")[0].split("/")[-1].replace("jpeg", "jpg").replace("quicktime", "mov")
    try:
        raw = base64.b64decode(body, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(400, "媒体文件不是有效的 base64") from exc
    if len(raw) > 200 * 1024 * 1024:
        raise HTTPException(413, "媒体文件超过 200MB")
    return raw, ext or "jpg"


@memory_router.get("/{memory_id}/media", include_in_schema=False)
def memory_media(memory_id: str):
    from fastapi.responses import FileResponse

    path = memory_service.media_path(memory_id)
    if path is None:
        raise HTTPException(404, "这条记忆没有附带媒体")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})


@memory_router.patch("/{memory_id}")
def memory_update(memory_id: str, body: MemoryUpdate):
    return memory_service.publics([_call(memory_service.store().update, memory_id, content=body.content, kind=body.kind, importance=body.importance, tags=body.tags)])[0]


@memory_router.delete("/{memory_id}")
def memory_delete(memory_id: str):
    if not memory_service.store().delete(memory_id):
        raise HTTPException(404, "记忆不存在")
    return {"ok": True}


@memory_router.post("/search")
def memory_search(body: MemorySearch):
    image = None
    if body.imageB64:
        from prism.multimodal import decode_image_b64

        try:
            image = decode_image_b64(body.imageB64)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    if not body.query.strip() and image is None:
        raise HTTPException(400, "需要文字或图片")
    store = memory_service.store()
    if image is not None and store.media is None:
        raise HTTPException(400, "以图召回需要开启多模态记忆（memory.multimodal）并配置跨模态的图文向量模型")
    hits = store.search(body.query, user_id=body.userId, agent_id=body.agentId, kinds=body.kinds, top_k=body.topK, min_score=body.minScore, touch=False, image=image)
    views = memory_service.publics([h["record"] for h in hits])
    return [{**v, "score": round(h["score"], 4), "similarity": round(h["similarity"], 4), "recency": round(h["recency"], 4),
             "mmSimilarity": round(h["mmSimilarity"], 4) if h.get("mmSimilarity") is not None else None} for v, h in zip(views, hits)]


@memory_router.post("/extract")
def memory_extract(body: MemoryExtract):
    return memory_service.extract_now(body.userId, body.text, body.answer)


@memory_router.post("/forget")
def memory_forget(body: ForgetRequest):
    return memory_service.store().forget(user_id=body.userId, min_keep_score=body.minKeepScore)


@memory_router.delete("/user/{user_id}")
def memory_clear(user_id: str):
    return {"deleted": memory_service.store().clear(user_id)}


@memory_router.get("/stats")
def memory_stats():
    return memory_service.store().stats()


@memory_router.get("/versions")
def memory_versions():
    return memory_service.store().versions()


# --- 个性化推荐 -----------------------------------------------------------------------------
class RecEventRequest(BaseModel):
    userId: str = "default"
    kind: Literal["view", "click", "like", "dislike", "ask", "cite"]
    itemId: str = ""
    kbId: str = ""
    text: str = ""


@recommend_router.post("/sync/{kb_id}")
def recommend_sync(kb_id: str):
    return _call(recommend_service.sync_kb, kb_id)


@recommend_router.get("")
def recommend(userId: str = "default", kbId: str | None = None, kinds: str | None = None, k: int = 8, useMemory: bool = True) -> dict[str, Any]:
    kind_list = [x for x in (kinds or "").split(",") if x] or None
    return recommend_service.recommend(userId, kb_id=kbId, kinds=kind_list, k=max(1, min(k, 50)), use_memory=useMemory)


@recommend_router.post("/events")
def recommend_event(body: RecEventRequest):
    return _call(recommend_service.engine().log_event, body.userId, kind=body.kind, item_id=body.itemId, kb_id=body.kbId, text=body.text)


@recommend_router.get("/similar/{item_id:path}")
def recommend_similar(item_id: str, k: int = 6):
    return _call(recommend_service.engine().similar, item_id, k)


@recommend_router.get("/stats")
def recommend_stats():
    return recommend_service.engine().stats()


# --- 多智能体协作 ---------------------------------------------------------------------------
class AgentRunRequest(BaseModel):
    kbId: str
    question: str
    userId: str = "default"


@agents_router.get("/roles")
def agent_roles():
    return ROLES


@agents_router.post("/run")
def agent_run(body: AgentRunRequest):
    if not body.question.strip():
        raise HTTPException(400, "问题不能为空")
    return StreamingResponse(agent_service.stream(body.kbId, body.question, body.userId), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@agents_router.get("/runs")
def agent_runs(limit: int = 30):
    return agent_service.board().runs(limit)


@agents_router.get("/runs/{run_id}")
def agent_run_history(run_id: str):
    rows = agent_service.board().history(run_id)
    if not rows:
        raise HTTPException(404, "运行记录不存在")
    return rows


class AgentRunsDelete(BaseModel):
    runIds: list[str] = Field(min_length=1)


@agents_router.delete("/runs/{run_id}")
def agent_run_delete(run_id: str):
    deleted = agent_service.board().delete_runs([run_id])
    if not deleted:
        raise HTTPException(404, "运行记录不存在")
    return {"runs": 1, "rows": deleted}


@agents_router.post("/runs/delete")
def agent_runs_delete(body: AgentRunsDelete):
    return {"rows": agent_service.board().delete_runs(body.runIds)}


@agents_router.delete("/runs")
def agent_runs_clear():
    return {"rows": agent_service.board().clear()}


@agents_router.get("/board")
def agent_board_info():
    return agent_service.board().info()
