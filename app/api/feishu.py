"""飞书文档接口：应用配置、同步进知识库、事件订阅（自动同步）、远程增删改查、文档智能体（SSE）。"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from prism.feishu import FeishuError

from ..services.feishu_service import feishu_service

router = APIRouter()


def _schedule(background: BackgroundTasks):
    return lambda ids: background.add_task(feishu_service.process_many, ids)


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(404, f"不存在：{exc.args[0] if exc.args else ''}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FeishuError as exc:
        raise HTTPException(502 if exc.status and exc.status >= 500 else 400, str(exc)) from exc


# --- 配置 -------------------------------------------------------------------------
class FeishuConfigUpdate(BaseModel):
    app_id: str | None = None
    app_secret: str | None = None
    domain: Literal["feishu", "lark"] | None = None
    web_base: str | None = None
    folder_token: str | None = None
    default_kb_id: str | None = None
    use_convert_api: bool | None = None
    agent_allow_write: bool | None = None
    new_doc_share: Literal["tenant_editable", "tenant_readable", "closed"] | None = None
    agent_max_steps: int | None = Field(default=None, ge=1, le=12)
    read_images: bool | None = None
    auto_sync_minutes: int | None = Field(default=None, ge=0, le=1440)
    event_verification_token: str | None = None
    event_encrypt_key: str | None = None


@router.get("/config")
def get_config():
    return feishu_service.public_settings()


@router.put("/config")
def update_config(body: FeishuConfigUpdate):
    return _call(feishu_service.update_settings, body.model_dump(exclude_none=True))


@router.post("/test")
def test_connection():
    return feishu_service.test()


# --- 同步进知识库 -------------------------------------------------------------------
class ImportRequest(BaseModel):
    kbId: str
    refs: list[str] = Field(min_length=1)


@router.get("/docs")
def list_links(kbId: str | None = None):
    return feishu_service.links(kbId)


@router.post("/import")
def import_docs(body: ImportRequest, background: BackgroundTasks):
    return _call(feishu_service.import_refs, body.kbId, body.refs, schedule=_schedule(background))


@router.post("/docs/{link_id}/sync")
def sync_link(link_id: str, background: BackgroundTasks, force: bool = False):
    return _call(feishu_service.sync_link, link_id, force=force, schedule=_schedule(background))


@router.post("/sync")
def sync_kb(kbId: str, background: BackgroundTasks, force: bool = False):
    return _call(feishu_service.sync_kb, kbId, force=force, schedule=_schedule(background))


@router.post("/sync/auto")
def run_auto_sync_now():
    """立即跑一轮自动同步（与定时任务同一逻辑）；已有一轮在跑时返回 skipped。"""
    result = feishu_service.auto_sync_round()
    return result if result is not None else {"skipped": True}


# --- 事件订阅：飞书上改了文档 → 只同步那一篇 -------------------------------------------
@router.post("/events")
async def feishu_event(request: Request, background: BackgroundTasks):
    """在飞书开发者后台「事件与回调」把请求地址配成 https://<域名>/api/v1/feishu/events，订阅「文档编辑」(drive.file.edit_v1)。"""
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(400, "请求体不是 JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(400, "请求体不是 JSON 对象")
    try:
        result = feishu_service.handle_event(payload)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if "documentId" in result:
        background.add_task(feishu_service.sync_changed, result["documentId"])   # 飞书要求 3 秒内回 200，同步放后台
    return result


@router.post("/events/subscribe")
def subscribe_events():
    return _call(feishu_service.subscribe_all)


@router.delete("/docs/{link_id}")
def unlink(link_id: str, deleteLocal: bool = True):
    return _call(feishu_service.unlink, link_id, delete_local=deleteLocal)


# --- 远程增删改查 -------------------------------------------------------------------
class CreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=800)
    markdown: str = ""
    folderToken: str | None = None
    kbId: str | None = None            # 给出时新建后直接同步进该知识库


class WriteRequest(BaseModel):
    ref: str
    markdown: str
    mode: Literal["append", "replace"] = "append"
    kbId: str | None = None


class ReplaceTextRequest(BaseModel):
    ref: str
    find: str = Field(min_length=1)
    replace: str = ""
    kbId: str | None = None


@router.get("/folder")
def list_folder(folderToken: str | None = None):
    return _call(feishu_service.list_remote, folderToken)


@router.get("/remote")
def read_remote(ref: str, kbId: str | None = None, view: bool = False):
    """view=true：页面展示用，多返回按飞书原样排版的 html。"""
    return _call(feishu_service.read, ref, kbId, view=view)


@router.get("/media/{token}")
def media(token: str):
    """文档里的图片（服务端代理下载并缓存，前端不直连飞书）。"""
    path = _call(feishu_service.media_file, token)
    return FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})


@router.post("/remote")
def create_remote(body: CreateRequest, background: BackgroundTasks):
    return _call(feishu_service.create, body.title, body.markdown, folder_token=body.folderToken, kb_id=body.kbId, schedule=_schedule(background))


@router.put("/remote")
def write_remote(body: WriteRequest, background: BackgroundTasks):
    return _call(feishu_service.write, body.ref, body.markdown, mode=body.mode, kb_id=body.kbId, schedule=_schedule(background))


@router.post("/remote/replace-text")
def replace_text(body: ReplaceTextRequest, background: BackgroundTasks):
    return _call(feishu_service.replace_text, body.ref, body.find, body.replace, kb_id=body.kbId, schedule=_schedule(background))


@router.delete("/remote")
def delete_remote(ref: str, kbId: str | None = None, deleteLocal: bool = True):
    return _call(feishu_service.delete_remote, ref, kb_id=kbId, delete_local=deleteLocal)


# --- 智能体 -------------------------------------------------------------------------
class AgentRunRequest(BaseModel):
    question: str
    kbId: str | None = None
    allowWrite: bool | None = None     # 为空时用配置里的 agent_allow_write


class AgentActionRequest(BaseModel):
    tool: str
    args: dict[str, Any] = {}
    kbId: str | None = None


@router.post("/agent/run")
def agent_run(body: AgentRunRequest):
    if not body.question.strip():
        raise HTTPException(400, "问题不能为空")
    return StreamingResponse(feishu_service.stream(body.kbId, body.question, allow_write=body.allowWrite), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/agent/execute")
def agent_execute(body: AgentActionRequest):
    return {"tool": body.tool, "result": _call(feishu_service.execute_action, body.kbId, body.tool, body.args)}
