"""语料治理：质量体检与剔除 / 别名词典 / 覆盖缺口 FAQ 增强 / 自动元数据。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import Repository, get_db
from ..services.corpus_service import corpus_service

router = APIRouter()


class ExcludeRequest(BaseModel):
    chunkIds: list[str]
    reason: str = ""


class RestoreRequest(BaseModel):
    fingerprints: list[str]


class RetireRequest(BaseModel):
    docIds: list[str]
    validTo: str | None = None


class AliasesRequest(BaseModel):
    groups: list[list[str]]


class GenerateRequest(BaseModel):
    intents: list[str] | None = None     # 不传：取最近一次覆盖分析里"部分匹配 / 无匹配"且尚未处理的意图
    limit: int = Field(30, ge=1, le=100)


class ItemCreate(BaseModel):
    question: str
    answer: str = ""


class ItemUpdate(BaseModel):
    question: str | None = None
    answer: str | None = None
    status: str | None = None


class MetaExtractRequest(BaseModel):
    docIds: list[str] | None = None
    useLlm: bool = False


class MetaApplyRequest(BaseModel):
    items: list[dict[str, Any]]          # [{docId, fields: {field: value}}]


def _kb(kb_id: str, db: Session) -> None:
    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")


# --- 质量体检 ------------------------------------------------------------------------------
@router.get("/{kb_id}/health")
def health(kb_id: str, db: Session = Depends(get_db)):
    """整库体检：乱码 / 低信息量 / 过短 / 超长 / 重复 / 近似重复段落，过期 / 多版本冲突 / 解析差 / 缺元数据文档。"""
    _kb(kb_id, db)
    return corpus_service.health(kb_id)


@router.post("/{kb_id}/health/exclude")
def exclude(kb_id: str, body: ExcludeRequest, db: Session = Depends(get_db)):
    """把段落移出检索（连同 child）。按内容指纹记录，重新解析 / 重建索引后仍然生效。"""
    _kb(kb_id, db)
    return corpus_service.exclude(kb_id, body.chunkIds, body.reason)


@router.post("/{kb_id}/health/restore")
def restore(kb_id: str, body: RestoreRequest, background: BackgroundTasks, db: Session = Depends(get_db)):
    _kb(kb_id, db)
    doc_ids = corpus_service.restore(kb_id, body.fingerprints)
    background.add_task(corpus_service.reingest_many, doc_ids)
    return {"reingesting": doc_ids}


@router.post("/{kb_id}/health/retire")
def retire(kb_id: str, body: RetireRequest, db: Session = Depends(get_db)):
    """给旧版本设置失效日期（默认昨天），检索时自动过滤。"""
    _kb(kb_id, db)
    try:
        return {"validTo": corpus_service.retire(kb_id, body.docIds, body.validTo)}
    except ValueError as exc:
        raise HTTPException(400, f"日期格式错误：{exc}") from exc


# --- 别名词典 ------------------------------------------------------------------------------
@router.get("/{kb_id}/aliases")
def aliases(kb_id: str, db: Session = Depends(get_db)):
    _kb(kb_id, db)
    return corpus_service.aliases(kb_id)


@router.put("/{kb_id}/aliases")
def save_aliases(kb_id: str, body: AliasesRequest, db: Session = Depends(get_db)):
    """保存后查询扩展立即生效；文档扩展（切片里写入别名）在后台重建索引后生效。"""
    _kb(kb_id, db)
    return corpus_service.save_aliases(kb_id, body.groups)


@router.post("/{kb_id}/aliases/discover")
def discover(kb_id: str, db: Session = Depends(get_db)):
    """从语料里找"全称（简称）""X，又称 Y"等写法，推荐别名组。"""
    _kb(kb_id, db)
    return corpus_service.discover_aliases(kb_id)


# --- 覆盖缺口 FAQ 增强 -----------------------------------------------------------------------
@router.get("/{kb_id}/augment")
def augment(kb_id: str, db: Session = Depends(get_db)):
    _kb(kb_id, db)
    return corpus_service.augment_state(kb_id)


@router.post("/{kb_id}/augment/generate")
def generate(kb_id: str, body: GenerateRequest, db: Session = Depends(get_db)):
    _kb(kb_id, db)
    try:
        return corpus_service.generate(kb_id, body.intents, body.limit)
    except Exception as exc:
        raise HTTPException(500, f"生成失败：{exc}") from exc


@router.post("/{kb_id}/augment/items")
def add_item(kb_id: str, body: ItemCreate, db: Session = Depends(get_db)):
    _kb(kb_id, db)
    if not body.question.strip():
        raise HTTPException(400, "问题不能为空")
    return corpus_service.add_item(kb_id, body.question.strip(), body.answer)


@router.put("/{kb_id}/augment/items/{item_id}")
def update_item(kb_id: str, item_id: str, body: ItemUpdate, db: Session = Depends(get_db)):
    _kb(kb_id, db)
    try:
        return corpus_service.update_item(kb_id, item_id, body.model_dump(exclude_none=True))
    except KeyError as exc:
        raise HTTPException(404, "条目不存在") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/{kb_id}/augment/items/{item_id}")
def delete_item(kb_id: str, item_id: str, db: Session = Depends(get_db)):
    _kb(kb_id, db)
    return corpus_service.delete_item(kb_id, item_id)


@router.post("/{kb_id}/augment/publish")
def publish(kb_id: str, db: Session = Depends(get_db)):
    """已通过的条目写成「增强语料 · FAQ」文档并入库；没有通过的条目时撤下该文档。"""
    _kb(kb_id, db)
    try:
        return corpus_service.publish(kb_id)
    except Exception as exc:
        raise HTTPException(500, f"发布失败：{exc}") from exc


# --- 自动元数据 ------------------------------------------------------------------------------
@router.post("/{kb_id}/metadata/extract")
def extract(kb_id: str, body: MetaExtractRequest, db: Session = Depends(get_db)):
    """对已入库文档（按保存的 Markdown）重新抽取元数据，只返回建议，不写入。"""
    _kb(kb_id, db)
    return corpus_service.extract_metadata(kb_id, body.docIds, body.useLlm)


@router.post("/{kb_id}/metadata/apply")
def apply(kb_id: str, body: MetaApplyRequest, db: Session = Depends(get_db)):
    """写入选中的字段；只更新索引里的文档元数据（用于过滤），不重新向量化。"""
    _kb(kb_id, db)
    try:
        return {"updated": corpus_service.apply_metadata(kb_id, body.items)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
