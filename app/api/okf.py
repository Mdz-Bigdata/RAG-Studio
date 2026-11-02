"""OKF 知识包：导入 / 重新同步 / 状态 / 命中预览 / Git 自动同步。条目导入后与普通文档一样出现在「知识库」页（标签 OKF）。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.okf_service import okf_service

router = APIRouter()


class OKFImportRequest(BaseModel):
    source: str                          # Git 地址（https / ssh / .git）或服务器上的目录
    ref: str | None = None               # 分支 / 标签（Git 来源）
    subdir: str | None = None            # 知识包在仓库里的子目录


class OKFAutoSyncRequest(BaseModel):
    minutes: int = Field(ge=0, le=1440)  # 0 = 关闭


# 固定路径要排在 /{kb_id} 前面，否则 "auto-sync" 会被当成知识库 id
@router.get("/auto-sync")
def auto_sync_settings():
    """Git 来源的定时自动同步：间隔、是否在跑、上一轮结果。"""
    return okf_service.auto_sync_settings()


@router.put("/auto-sync")
def update_auto_sync(body: OKFAutoSyncRequest):
    return okf_service.update_auto_sync(body.minutes)


@router.post("/auto-sync/run")
def run_auto_sync_now():
    """立刻跑一轮（远端提交号没变的知识库跳过）。"""
    result = okf_service.auto_sync_round()
    if result is None:
        raise HTTPException(409, "上一轮自动同步还没结束")
    return result


@router.get("/{kb_id}")
def status(kb_id: str):
    return okf_service.status(kb_id)


@router.post("/{kb_id}/import")
def import_bundle(kb_id: str, body: OKFImportRequest):
    try:
        return okf_service.import_bundle(kb_id, body.source, ref=body.ref or None, subdir=body.subdir or None)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/{kb_id}/sync")
def sync(kb_id: str):
    try:
        return okf_service.sync(kb_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{kb_id}/match")
def match(kb_id: str, q: str, limit: int = 5):
    """这个问题会命中哪些条目（确定性路由的预览）。"""
    return okf_service.match(kb_id, q, limit=limit)


@router.delete("/{kb_id}")
def forget(kb_id: str, deleteDocs: bool = True):
    return okf_service.forget(kb_id, delete_docs=deleteDocs)
