"""多模态检索服务：文搜图 / 图搜图 / 图文混合 / 图搜文 / 文搜视频 / 图搜视频，内容查重与相似推荐，图片库浏览，视频播放，以及问答里的图片侧。"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from prism.multimodal import ImageDescriber, ImageHit, ImageRecord, MultimodalResult, MultimodalRetriever, decode_image_b64, image_to_base64
from prism.multimodal.schemas import VideoSegment
from prism.multimodal.video import is_video, playable_path, video_backend
from prism.schemas import Context

from ..config import app_config
from ..db import Repository, SessionLocal
from .index_service import index_service
from .model_service import model_service
from .settings_service import settings_service

logger = logging.getLogger(__name__)

ROUTE_LABELS = {
    "clip": "图文向量（CLIP 文本塔 → 图片）",
    "clip_fused": "图文向量（文本 + 图片早融合）",
    "visual": "视觉相似（图像塔 → 图片）",
    "ocr_sparse": "图中文字（OCR → 图片文字 BM25，认定是哪张图）",
    "desc_dense": "图片描述 · 向量",
    "desc_sparse": "图片描述 · BM25",
    "clip:image→chunk": "图文向量直达（图片 → 正文）",
    "neighbor:image→image→chunk": "相似图片所在段落",
    "caption:dense": "图片转文字 · 向量",
    "caption:sparse": "图片转文字 · BM25",
    "lance_mm": "LanceDB 多模态表 · 图片文字全文（ICU 分词）",
    "lance_mm:image→text": "LanceDB 多模态表 · 图 → 正文（CLIP + 图中文字）",
    "video": "视频向量（文本 → 整段视频，WeMM）",
    "video_joint": "视频 + 字幕向量（文本 → 画面与台词，WeMM）",
    "video:image→video": "图 → 视频（图片 → 整段视频，WeMM）",
}


class MultimodalService:
    def retriever(self, kb_id: str) -> MultimodalRetriever:
        settings = settings_service.get()
        index = index_service.get(kb_id)
        describer = ImageDescriber(settings.multimodal, model_service.vision_llm(aux=True), cache_path=index.directory / "image_index" / "query_desc_cache.json")
        llm = model_service.llm(aux=True) if settings.multimodal.cross_lingual else None   # 跨语言补译用辅助模型（关思考、可换快模型）
        return MultimodalRetriever(settings, index, describer=describer, lance=self._lance_route(index, settings), llm=llm)

    @staticmethod
    def _lance_route(index, settings):
        """multimodal.lance_table 打开时接入 LanceDB 多模态表路（每库一张表 mm，查询前按需增量同步）；没装 lancedb 就不接。"""
        if not settings.multimodal.lance_table:
            return None
        from prism.lance import lancedb_available

        if not lancedb_available():
            logger.warning("multimodal.lance_table 已打开，但没有安装 lancedb：多模态表路跳过")
            return None
        from prism.lance.multimodal import LanceMultimodalTable, MultimodalLanceRoute

        return MultimodalLanceRoute(index, LanceMultimodalTable.of_index(index, keep_versions_days=settings.vector_store.lancedb.keep_versions_days))

    # --- 检索 -----------------------------------------------------------------------------
    def search(self, kb_id: str, mode: str, *, query: str = "", image_b64: str | None = None, top_k: int | None = None, alpha: float | None = None, min_score: float | None = None, doc_ids: list[str] | None = None) -> dict[str, Any]:
        with index_service.reading(kb_id):        # 整库重建收尾替换目录时，等这次检索结束
            return self._search(kb_id, mode, query=query, image_b64=image_b64, top_k=top_k, alpha=alpha, min_score=min_score, doc_ids=doc_ids)

    def _search(self, kb_id: str, mode: str, *, query: str, image_b64: str | None, top_k: int | None, alpha: float | None, min_score: float | None, doc_ids: list[str] | None) -> dict[str, Any]:
        query = (query or "").strip()
        image = decode_image_b64(image_b64) if image_b64 else None
        r = self.retriever(kb_id)
        if mode == "text2image":
            if not query:
                raise ValueError("文搜图需要输入文字")
            result = r.text_to_image(query, top_k=top_k, doc_ids=doc_ids)
        elif mode == "image2image":
            if image is None:
                raise ValueError("图搜图需要上传图片")
            result = r.image_to_image(image, top_k=top_k, min_score=min_score, doc_ids=doc_ids)
        elif mode == "hybrid":
            if image is None or not query:
                raise ValueError("图文混合检索需要同时提供文字和图片")
            result = r.hybrid(query, image, alpha=alpha, top_k=top_k, doc_ids=doc_ids)
        elif mode == "image2text":
            if image is None:
                raise ValueError("图搜文需要上传图片")
            result = r.image_to_text(image, query=query, top_k=top_k)
        elif mode == "text2video":
            if not query:
                raise ValueError("文搜视频需要输入文字")
            result = r.text_to_video(query, top_k=top_k, doc_ids=doc_ids)
        elif mode == "image2video":
            if image is None:
                raise ValueError("图搜视频需要上传图片")
            result = r.image_to_video(image, top_k=top_k, doc_ids=doc_ids)
        else:
            raise ValueError(f"未知检索模式：{mode}")
        return self.to_response(kb_id, result)

    # --- 序列化 ---------------------------------------------------------------------------
    def image_url(self, kb_id: str, image_id: str) -> str:
        return f"{app_config.api_prefix}/multimodal/{kb_id}/images/{image_id}/file"

    def video_url(self, kb_id: str, doc_id: str) -> str:
        return f"{app_config.api_prefix}/multimodal/{kb_id}/videos/{doc_id}/file"

    def record_view(self, kb_id: str, rec: ImageRecord, names: dict[str, str]) -> dict[str, Any]:
        view = {
            "imageId": rec.image_id, "docId": rec.doc_id, "fileName": names.get(rec.doc_id), "url": self.image_url(kb_id, rec.image_id),
            "page": rec.page, "width": rec.width, "height": rec.height, "origin": rec.origin, "caption": rec.caption, "description": rec.description,
            "ocrText": rec.ocr_text, "contextText": rec.context_text, "breadcrumb": rec.breadcrumb_text, "anchorChunkId": rec.anchor_chunk_id, "label": rec.label(),
        }
        if rec.is_video:
            view.update({"time": rec.time, "timeStart": rec.time_start, "timeEnd": rec.time_end, "videoUrl": self.video_url(kb_id, rec.doc_id)})
        return view

    def segment_view(self, kb_id: str, seg: VideoSegment, names: dict[str, str], durations: dict[str, float]) -> dict[str, Any]:
        return {
            "docId": seg.doc_id, "fileName": names.get(seg.doc_id), "videoUrl": self.video_url(kb_id, seg.doc_id), "duration": durations.get(seg.doc_id),
            "start": seg.start, "end": seg.end, "time": seg.time, "score": round(seg.score, 4), "rank": seg.rank,
            "frames": [self.hit_view(kb_id, h, names) for h in seg.hits],
        }

    @staticmethod
    def video_durations(index) -> dict[str, float]:
        """每个视频的时长：取最后一个片段的结束时间（关键帧只覆盖分析过的部分，足够画时间轴）。"""
        out: dict[str, float] = {}
        for r in index.images.records.values():
            if r.is_video:
                out[r.doc_id] = max(out.get(r.doc_id, 0.0), r.time_end or r.time or 0.0)
        return out

    def hit_view(self, kb_id: str, hit: ImageHit, names: dict[str, str], index: int | None = None) -> dict[str, Any]:
        view = self.record_view(kb_id, hit.record, names)
        view.update({"score": round(hit.score, 4), "rank": hit.rank, "linked": hit.linked, "routes": {k: {"rank": v.rank, "score": round(v.score, 4)} for k, v in hit.routes.items()}})
        if index is not None:
            view["index"] = index
        return view

    def to_response(self, kb_id: str, result: MultimodalResult) -> dict[str, Any]:
        from .retrieval_service import retrieval_service

        index = index_service.get(kb_id)
        names = retrieval_service.doc_names(kb_id)
        durations = self.video_durations(index) if result.segments else {}

        def route_item(chunk_id: str, score: float, rank: int) -> dict[str, Any]:
            rec = index.images.get(chunk_id)
            if rec is not None:
                return {"id": chunk_id, "kind": "image", "rank": rank, "score": round(score, 4), "url": self.image_url(kb_id, chunk_id), "text": rec.brief(80), "label": rec.label()}
            c = index.store.get(chunk_id)
            return {"id": chunk_id, "kind": "chunk", "rank": rank, "score": round(score, 4), "url": None, "text": (c.text[:160] if c else ""), "label": (c.breadcrumb_text if c else "")}

        return {
            "mode": result.mode,
            "query": result.query,
            "hasImage": result.has_image,
            "images": [self.hit_view(kb_id, h, names) for h in result.images],
            "contexts": [context_view(c, names) for c in result.contexts],
            "segments": [self.segment_view(kb_id, s, names, durations) for s in result.segments],
            "routes": {k: [route_item(x.chunk_id, x.score, x.rank) for x in v[:10]] for k, v in result.routes.items()},
            "routeLabels": {k: ROUTE_LABELS.get(k, k) for k in result.routes},
            "queryDescription": result.query_description,
            "notes": result.notes,
            "timingsMs": {k: round(v, 1) for k, v in result.timings_ms.items()},
            "embedder": result.embedder or index.images.stats(),
        }

    # --- 查重 / 相似 ----------------------------------------------------------------------
    def similar(self, kb_id: str, *, image_id: str | None = None, doc_id: str | None = None, top_k: int = 10, threshold: float | None = None) -> dict[str, Any]:
        """某张图 / 某个视频片段（image_id）或某个视频（doc_id）的相似内容：相似图片与片段、相似视频（含片段重合度）。"""
        from .retrieval_service import retrieval_service

        with index_service.reading(kb_id):
            index = index_service.get(kb_id)
            d = self.retriever(kb_id).deduper(threshold)
            names = retrieval_service.doc_names(kb_id)
            if image_id and index.images.get(image_id) is None:
                raise ValueError("图片不存在")
            if doc_id and not any(r.doc_id == doc_id for r in index.images.records.values()):
                raise ValueError("该文档没有图片 / 视频")
            rec = index.images.get(image_id) if image_id else None
            video_doc = doc_id or (rec.doc_id if rec is not None and rec.is_video else None)
            images = []
            if image_id:
                for n, (i, sc) in enumerate(d.similar_images(image_id, top_k)):
                    r = index.images.get(i)
                    if r is not None:
                        images.append({**self.record_view(kb_id, r, names), "score": round(sc, 4), "rank": n + 1, "duplicate": sc >= d.threshold})
            videos = [self.similar_video_view(kb_id, index, s, names) for s in d.similar_videos(video_doc, top_k)] if video_doc else []
            return {"imageId": image_id, "docId": video_doc, "threshold": d.threshold, "embedder": index.images.embedder.signature(), "images": images, "videos": videos}

    def similar_video_view(self, kb_id: str, index, s, names: dict[str, str]) -> dict[str, Any]:
        cover = index.images.get(s.pairs[0][1]) if s.pairs else next((r for r in index.images.of_document(s.doc_id)), None)
        return {"docId": s.doc_id, "fileName": names.get(s.doc_id), "videoUrl": self.video_url(kb_id, s.doc_id), "score": round(s.score, 4), "overlap": round(s.overlap, 4),
                "duplicate": s.duplicate, "cover": self.image_url(kb_id, cover.image_id) if cover else None,
                "pairs": [{"source": self.record_view(kb_id, index.images.get(a), names), "match": self.record_view(kb_id, index.images.get(b), names), "score": round(sc, 4)}
                          for a, b, sc in s.pairs[:20] if index.images.get(a) and index.images.get(b)]}

    def duplicates(self, kb_id: str, *, threshold: float | None = None, kinds: list[str] | None = None) -> dict[str, Any]:
        """全库查重：重复的图片组、重复（片段高度重合）的视频组。"""
        from .retrieval_service import retrieval_service

        with index_service.reading(kb_id):
            index = index_service.get(kb_id)
            d = self.retriever(kb_id).deduper(threshold)
            names = retrieval_service.doc_names(kb_id)
            groups = []
            for g in d.duplicate_groups(tuple(kinds or ("image", "video"))):
                if g.kind == "image":
                    items = [self.record_view(kb_id, index.images.get(i), names) for i in g.ids if index.images.get(i)]
                else:
                    items = [{"docId": i, "fileName": names.get(i), "videoUrl": self.video_url(kb_id, i),
                              "cover": next((self.image_url(kb_id, r.image_id) for r in index.images.of_document(i)), None)} for i in g.ids]
                groups.append({"kind": g.kind, "score": round(g.score, 4), "items": items})
            return {"threshold": d.threshold, "overlap": d.overlap, "embedder": index.images.embedder.signature(), "groups": groups,
                    "images": sum(1 for r in index.images.records.values() if not r.is_video), "videos": len(d.videos())}

    # --- 图片库 ---------------------------------------------------------------------------
    def list_images(self, kb_id: str, doc_id: str | None = None) -> list[dict[str, Any]]:
        from .retrieval_service import retrieval_service

        index = index_service.get(kb_id)
        names = retrieval_service.doc_names(kb_id)
        recs = sorted(index.images.records.values(), key=lambda r: (r.doc_id, r.order))
        return [self.record_view(kb_id, r, names) for r in recs if not doc_id or r.doc_id == doc_id]

    def video_path(self, kb_id: str, doc_id: str) -> Path | None:
        """视频文档的可播放文件（浏览器放不了的格式用入库时转好的预览 mp4）。"""
        with SessionLocal() as db:
            doc = Repository(db).get_document(doc_id)
            if doc is None or doc.kb_id != kb_id or not doc.file_path or not is_video(doc.file_path):
                return None
            path = Path(doc.file_path)
        root = app_config.upload_dir.resolve()
        p = playable_path(path)
        return p if p is not None and root in p.resolve().parents else None

    def image_path(self, kb_id: str, image_id: str) -> Path | None:
        index = index_service.get(kb_id)
        rec = index.images.get(image_id)
        if rec is None:
            return None
        p = index.images.path_of(rec).resolve()
        return p if p.is_file() and index.directory.resolve() in p.parents else None

    def status(self, kb_id: str) -> dict[str, Any]:
        index = index_service.get(kb_id)
        s = settings_service.get().multimodal
        rt = model_service.status()
        frames = [r for r in index.images.records.values() if r.is_video]
        return {
            "enabled": s.enabled, "images": len(index.images), "documentsWithImages": len({r.doc_id for r in index.images.records.values()}),
            "videos": len({r.doc_id for r in frames}), "videoFrames": len(frames),
            "video": {"enabled": s.video.enabled, "backend": video_backend(), "vectors": len(index.images.video),
                      "jointVectors": sum(1 for r in frames if r.subtitle.strip() and r.clip) if len(index.images.video_joint) else 0},   # 带字幕（联合向量与纯视频向量不同）的片段数
            "described": sum(1 for r in index.images.records.values() if r.description), "withOcr": sum(1 for r in index.images.records.values() if r.ocr_text),
            "embedder": rt["imageEmbedding"], "indexEmbedder": index.images.stats(), "vision": rt["vision"], "ocr": rt["ocr"],
            "defaults": {"topK": s.top_k, "alpha": s.hybrid_alpha, "minScore": s.min_score, "rrfK": s.rrf_k},
            "routes": self.active_routes(s, bool(rt["imageEmbedding"].get("crossModal")), len(index.images) > 0, video_native=bool(rt["imageEmbedding"].get("videoNative"))),
        }

    @staticmethod
    def active_routes(s, cross_modal: bool, has_images: bool, *, video_native: bool = False) -> dict[str, list[str]]:
        """文搜图 / 图搜文实际参与 RRF 融合的召回路，前端按它生成说明。与 MultimodalRetriever 的分支保持一致：
        没有文本塔就没有图文向量 / 直达路；权重为 0 的路不参与融合；LanceDB 多模态表路要开关打开、权重 > 0 且装了 lancedb。"""
        from prism.lance import lancedb_available

        lance = s.lance_table and s.weight_lance_mm > 0 and lancedb_available()
        t2i = [k for k, on in (("clip", cross_modal and s.weight_clip > 0), ("desc_dense", s.weight_desc_dense > 0),
                               ("desc_sparse", s.weight_desc_sparse > 0), ("lance_mm", lance)) if on]
        i2t = [k for k, on in (("clip:image→chunk", cross_modal and s.weight_clip > 0), ("neighbor:image→image→chunk", has_images),
                               ("caption:dense", True), ("caption:sparse", True), ("lance_mm:image→text", lance and cross_modal)) if on]
        t2v = t2i + [k for k, on in (("video", video_native and s.video.weight_video > 0), ("video_joint", video_native and s.video.weight_video_joint > 0)) if on]
        i2v = ["visual"] + (["video:image→video"] if video_native and s.video.weight_video > 0 else [])
        return {"text2image": t2i, "image2text": i2t, "text2video": t2v, "image2video": i2v}

    # --- 问答 -----------------------------------------------------------------------------
    def images_for_answer(self, kb_id: str, query: str, contexts: list[Context], *, doc_ids: set[str] | None = None) -> tuple[list[ImageHit], MultimodalResult]:
        """doc_ids：本轮可见的文档（过期 / 权限过滤后，见 retrieval_service.visible_doc_ids）；None = 不过滤。"""
        with index_service.reading(kb_id):
            return self.retriever(kb_id).images_for_answer(query, contexts, doc_ids=doc_ids)

    def payloads(self, kb_id: str, hits: list[ImageHit], n: int) -> list[str]:
        """前 n 张检索图片的 data URL（缩小后的 JPEG），交给看得了图的模型。"""
        index = index_service.get(kb_id)
        max_side = settings_service.get().multimodal.caption_max_side
        out: list[str] = []
        for h in hits[:n]:
            try:
                out.append("data:image/jpeg;base64," + image_to_base64(index.images.load_image(h.record), max_side=max_side))
            except OSError as exc:
                logger.warning("图片文件读取失败 %s：%s", h.record.image_id, exc)
                break  # 附图顺序必须与 [图n] 对齐，读不到就到此为止
        return out

    def save_query_image(self, image_b64: str) -> str | None:
        """把用户随问题上传的图片落盘，历史会话里才能回看。返回可访问的 URL。"""
        try:
            im = decode_image_b64(image_b64)
        except ValueError:
            return None
        d = app_config.upload_dir / "_chat"
        d.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}.jpg"
        im.thumbnail((1280, 1280))
        im.save(d / name, format="JPEG", quality=88)
        return f"{app_config.api_prefix}/multimodal/query-images/{name}"

    def query_image_path(self, name: str) -> Path | None:
        d = (app_config.upload_dir / "_chat").resolve()
        p = (d / name).resolve()
        return p if p.is_file() and p.parent == d else None


def context_view(cx: Context, names: dict[str, str]) -> dict[str, Any]:
    c = cx.chunk
    return {
        "chunkId": c.chunk_id, "docId": c.doc_id, "fileName": names.get(c.doc_id), "breadcrumb": c.breadcrumb_text, "text": c.text,
        "compressedText": cx.compressed_text, "score": round(cx.score, 6), "tokenCount": c.token_count, "pageStart": c.page_start, "pageEnd": c.page_end,
        "matchedChildIds": cx.matched_child_ids, "weaved": cx.weaved, "weavedEntity": cx.weaved_entity,
    }


def kb_exists(kb_id: str) -> bool:
    with SessionLocal() as db:
        return Repository(db).get_kb(kb_id) is not None


multimodal_service = MultimodalService()
