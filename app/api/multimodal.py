"""多模态检索 API：文搜图 / 图搜图 / 图文混合 / 图搜文，图片库与图片文件。端到端问答走 /chat/stream。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..schemas import MultimodalSearchRequest
from ..services import multimodal_service
from ..services.multimodal_service import kb_exists

router = APIRouter()


def _require_kb(kb_id: str) -> None:
    if not kb_exists(kb_id):
        raise HTTPException(404, "知识库不存在")


@router.get("/query-images/{name}", include_in_schema=False)
def query_image(name: str):
    path = multimodal_service.query_image_path(name)
    if path is None:
        raise HTTPException(404, "图片不存在")
    return FileResponse(path)


@router.get("/{kb_id}/status")
def status(kb_id: str):
    """图片索引概况与能力：图文向量模型是否跨模态、OCR 引擎、是否有可看图的模型。"""
    _require_kb(kb_id)
    return multimodal_service.status(kb_id)


@router.post("/{kb_id}/search")
def search(kb_id: str, body: MultimodalSearchRequest):
    """mode = text2image（文搜图）| image2image（图搜图）| hybrid（图文混合）| image2text（图搜文）。"""
    _require_kb(kb_id)
    try:
        return multimodal_service.search(kb_id, body.mode, query=body.query or "", image_b64=body.imageB64, top_k=body.topK, alpha=body.alpha, min_score=body.minScore, doc_ids=body.docIds)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"多模态检索失败：{exc}") from exc


@router.get("/{kb_id}/images")
def list_images(kb_id: str, docId: str | None = None):
    _require_kb(kb_id)
    return multimodal_service.list_images(kb_id, docId)


@router.get("/{kb_id}/images/{image_id}/file")
def image_file(kb_id: str, image_id: str):
    _require_kb(kb_id)
    path = multimodal_service.image_path(kb_id, image_id)
    if path is None:
        raise HTTPException(404, "图片不存在")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})
