from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import Repository, get_db
from ..schemas import CoverageRequest, EvalRunRequest, HardNegativeMiningRequest, MultimodalEvalRequest
from ..services import eval_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/sample")
def sample():
    return eval_service.sample()


@router.post("/{kb_id}/run")
def run(kb_id: str, body: EvalRunRequest, db: Session = Depends(get_db)):
    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    examples = [e.model_dump() for e in (body.examples or [])]
    if body.useSample or not examples:
        examples = eval_service.sample()
    if not examples:
        raise HTTPException(400, "没有评测样例：请上传黄金集或使用示例集")
    try:
        return eval_service.run(kb_id, examples, ks=body.ks, name=body.name, overrides=body.overrides)
    except Exception as exc:
        raise HTTPException(500, f"评测失败：{exc}") from exc


@router.get("/{kb_id}/reports")
def reports(kb_id: str, db: Session = Depends(get_db)):
    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    return eval_service.reports(kb_id)


@router.post("/{kb_id}/hard-negatives")
def hard_negatives(kb_id: str, body: HardNegativeMiningRequest, db: Session = Depends(get_db)):
    """显式训练标注挖掘难负例；heldOut 中的问题只用于阻止进入训练数据。"""
    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    try:
        return eval_service.mine_hard_negatives(
            kb_id, [e.model_dump() for e in body.examples],
            held_out=[e.model_dump() for e in body.heldOut], config=body.config,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("难负样本挖掘失败（知识库 %s）", kb_id)
        raise HTTPException(500, "难负样本挖掘失败，请检查服务日志") from exc


@router.get("/{kb_id}/multimodal/sample")
def multimodal_sample(kb_id: str, db: Session = Depends(get_db)):
    """从图片库自动生成多模态样例：图注 / 图中文字 → 文搜图，图片自身（加扰动）→ 图搜图 / 图搜文，外加不应附图的负例。"""
    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    return eval_service.multimodal_samples(kb_id)


@router.post("/{kb_id}/multimodal/run")
def multimodal_run(kb_id: str, body: MultimodalEvalRequest, db: Session = Depends(get_db)):
    """多模态评测：文搜图 / 图搜图 Hit@K·MRR（含召回路消融）、图搜文、回答附图的精确率 / 召回率 / 负例不附图率。"""
    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    examples = list(body.examples or [])
    if body.auto or not examples:
        examples = eval_service.multimodal_samples(kb_id)
    if not examples:
        raise HTTPException(400, "这个知识库还没有图片，无法生成多模态样例")
    try:
        return eval_service.multimodal_run(kb_id, examples, ks=body.ks, name=body.name)
    except Exception as exc:
        raise HTTPException(500, f"多模态评测失败：{exc}") from exc


@router.get("/{kb_id}/multimodal/reports")
def multimodal_reports(kb_id: str, db: Session = Depends(get_db)):
    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    return eval_service.multimodal_reports(kb_id)


@router.post("/{kb_id}/coverage")
def coverage(kb_id: str, body: CoverageRequest, db: Session = Depends(get_db)):
    """语料准备：把每条意图与现有语料的匹配程度分到 高匹配 / 部分匹配 / 无匹配，并找出未被命中 / 已过期的文档。"""
    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    intents = list(body.intents or [])
    if body.useSample or not intents:
        intents = [{"question": e["question"], "category": e.get("category") or "未分类"} for e in eval_service.sample()]
    if not intents:
        raise HTTPException(400, "请提供意图列表（每行一个用户问题）")
    try:
        return eval_service.coverage(kb_id, intents, high=body.high, partial=body.partial)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"覆盖分析失败：{exc}") from exc


@router.get("/{kb_id}/coverage")
def last_coverage(kb_id: str, db: Session = Depends(get_db)):
    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    return eval_service.last_coverage(kb_id)
