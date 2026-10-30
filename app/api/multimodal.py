"""多模态检索 API：文搜图 / 图搜图 / 图文混合 / 图搜文 / 文搜视频 / 图搜视频，查重与相似推荐，图片库、图片文件与视频文件。端到端问答走 /chat/stream。"""

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
    """mode = text2image（文搜图）| image2image（图搜图）| hybrid（图文混合）| image2text（图搜文）| text2video（文搜视频）| image2video（图搜视频）。"""
    _require_kb(kb_id)
    try:
        return multimodal_service.search(kb_id, body.mode, query=body.query or "", image_b64=body.imageB64, top_k=body.topK, alpha=body.alpha, min_score=body.minScore, doc_ids=body.docIds)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"多模态检索失败：{exc}") from exc


@router.get("/{kb_id}/similar")
def similar(kb_id: str, imageId: str | None = None, docId: str | None = None, topK: int = 10, threshold: float | None = None):
    """相似内容推荐 / 单条查重：imageId（图片或视频片段）找相似图片与片段，docId（视频）找相似视频（含片段重合度）。"""
    _require_kb(kb_id)
    if not imageId and not docId:
        raise HTTPException(400, "需要 imageId 或 docId")
    try:
        return multimodal_service.similar(kb_id, image_id=imageId, doc_id=docId, top_k=max(1, min(topK, 50)), threshold=threshold)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{kb_id}/duplicates")
def duplicates(kb_id: str, threshold: float | None = None, kinds: str | None = None):
    """全库内容查重：kinds = image,video（默认都查）；threshold 不给时按图文向量模型自动。"""
    _require_kb(kb_id)
    return multimodal_service.duplicates(kb_id, threshold=threshold, kinds=[k for k in (kinds or "").split(",") if k in ("image", "video")] or None)


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


@router.get("/{kb_id}/videos/{doc_id}/file")
def video_file(kb_id: str, doc_id: str):
    """视频原文件（或转好的预览 mp4）。支持 Range 请求，播放器可以直接跳到片段起点。"""
    _require_kb(kb_id)
    path = multimodal_service.video_path(kb_id, doc_id)
    if path is None:
        raise HTTPException(404, "视频不存在，或浏览器无法播放该格式（需要 ffmpeg 转出预览）")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})
