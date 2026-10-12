from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import Repository, get_db
from ..schemas import RetrievalRequest, RetrievalResponse, RouteRequest, VectorBenchmarkRequest
from ..services import retrieval_service

router = APIRouter()


@router.post("/{kb_id}/playground", response_model=RetrievalResponse)
def playground(kb_id: str, body: RetrievalRequest, db: Session = Depends(get_db)):
    """检索实验室：完整可解释链路（查询层 → 各路召回 → 融合 → 精排 → MMR → 上下文 → 织网 → 压缩）。"""
    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    try:
        filters = {**(body.filters or {}), **({"user": body.user.model_dump()} if body.user else {})} or None
        result = retrieval_service.retrieve(kb_id, body.query, mode=body.mode, top_k=body.topK, use_transform=body.useTransform, overrides=body.overrides, filters=filters)
    except Exception as exc:
        raise HTTPException(500, f"检索失败：{exc}") from exc
    return retrieval_service.to_response(kb_id, result)


@router.get("/{kb_id}/facets")
def facets(kb_id: str, level: int | None = None, department: str | None = None, db: Session = Depends(get_db)):
    """元数据过滤的可选项。传 level / department 时只统计该身份可见的文档。"""
    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    user = {"level": level, "department": department} if level is not None else None
    return retrieval_service.facets(kb_id, user)


@router.post("/route")
def route(body: RouteRequest):
    return retrieval_service.route(body.query, top_n=body.topN, mode=body.mode)


@router.post("/{kb_id}/vector-benchmark")
def vector_benchmark(kb_id: str, body: VectorBenchmarkRequest, db: Session = Depends(get_db)):
    """向量索引调优对比：精确检索 vs 当前（或本次传入的）降维 / 量化配置的召回率与延迟。不修改已保存的配置。"""
    from prism.index.vector_bench import benchmark

    from ..services import index_service, settings_service

    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    index = index_service.get(kb_id)
    store = index.vectors
    configure = getattr(store, "configure", None)
    base = settings_service.get().vector_store.search
    trial = {"pca_dim": body.pcaDim if body.pcaDim is not None else base.pca_dim, "quantize": body.quantize or base.quantize,
             "rescore": base.rescore if body.rescore is None else body.rescore, "oversample": body.oversample or base.oversample, "ef_search": base.ef_search}
    with index_service.lock(kb_id):
        if callable(configure):
            configure(**trial)
        try:
            return {**benchmark(index, n_queries=body.queries, top_k=body.topK), "config": trial}
        finally:
            if callable(configure):
                configure(pca_dim=base.pca_dim, quantize=base.quantize, rescore=base.rescore, oversample=base.oversample, ef_search=base.ef_search)
