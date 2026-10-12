"""对话服务：SSE 事件流（meta → plan → sources → relations → images → message* → done），会话持久化，多模态 / 临时附件。"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from itertools import zip_longest
from typing import Any

from prism.generation import AnswerGenerator
from prism.generation.answer import parse_citations, parse_image_citations
from prism.generation.citecheck import check_citations, check_numbers, strip_citations
from prism.generation.prompts import format_context
from prism.multimodal import ImageHit, decode_image_b64
from prism.retrieval import detect_language
from prism.schemas import Context, RetrievalResult, RetrievalTrace

from ..db import Repository, SessionLocal
from .lake_service import lake_service
from .model_service import model_service
from .multimodal_service import multimodal_service
from .memory_service import memory_service
from .recommend_service import recommend_service
from .retrieval_service import retrieval_service
from .settings_service import settings_service

logger = logging.getLogger(__name__)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ChatService:
    def stream(self, *, kb_id: str | None, session_id: str | None, message: str, image_b64: str | None = None, temp_doc_content: str | None = None, mode: str = "hybrid", overrides: dict[str, Any] | None = None, top_k: int | None = None, user: dict[str, Any] | None = None) -> Iterator[str]:
        settings = settings_service.get()
        mm = settings.multimodal
        message = (message or "").strip()
        query_image_url = multimodal_service.save_query_image(image_b64) if image_b64 else None
        with SessionLocal() as db:
            repo = Repository(db)
            session = repo.get_session(session_id) if session_id else None
            if session is None:
                session = repo.create_session(kb_id if kb_id and kb_id != "auto" else None, None)
            session_id = session.id
            history = [{"role": m.role, "content": m.content} for m in repo.list_messages(session_id)][-10:]
            extra = {"trace_json": {"queryImage": query_image_url}} if query_image_url else {}
            user_msg = repo.add_message(session, "user", message or "（图片检索）", **extra)
            user_msg_id, user_ts = user_msg.id, time.time()
            resolved_kb = session.kb_id
            first_kb = next((kb.id for kb in repo.list_kbs()), None)

        # 查询路由：kb 未指定 / auto（只发图片、没有文字时无从路由，沿用会话的知识库）
        route_info: list[dict[str, Any]] = []
        if not kb_id or kb_id == "auto":
            if message:
                route_info = retrieval_service.route(message, top_n=1, mode=settings.retrieval.router.mode)
            resolved_kb = route_info[0]["kb_id"] if route_info else (resolved_kb or first_kb)
        else:
            resolved_kb = kb_id

        yield _sse("meta", {"sessionId": session_id, "kbId": resolved_kb, "mode": mode, "language": detect_language(message), "route": route_info, "queryImage": query_image_url})

        # 带图提问：先认图（图片描述 + 图中文字，顺带以图搜文）。识别结果要交给文本检索——查询改写、多跳判断"还缺什么"
        # 都得知道图里有什么；否则"这段代码违反了哪些规范"在检索侧只是一句没有内容的话
        i2t: tuple[Any, Any, Any] | None = None
        image_error = ""
        if image_b64 and resolved_kb and mm.enabled:
            try:
                i2t = self._image_to_text(resolved_kb, message, image_b64, top_k or settings.retrieval.rerank.top_k, user=user)
            except Exception as exc:
                logger.exception("以图搜文失败")
                image_error = f"图片检索失败：{exc}"
        query_description = (i2t[2].query_description or "") if i2t is not None else ""

        # 文本检索。历史里上一轮回答的引用编号去掉（只用来补全指代，编号在本轮已失效）；带图时把图片内容作为本轮的一条上下文
        retrieval_history = [{**m, "content": strip_citations(m["content"])} if m["role"] == "assistant" else m for m in history]
        if query_description:
            retrieval_history.append({"role": "user", "content": f"（随本条问题上传的图片，识别出的内容）{query_description[:600]}"})
        result = RetrievalResult(query=message, contexts=[], trace=RetrievalTrace(original_query=message, strategy="none", notes=[] if resolved_kb else ["没有可用知识库"]))
        if resolved_kb and message:
            try:
                result = retrieval_service.retrieve(resolved_kb, message, mode=mode, top_k=top_k, overrides=overrides, history=retrieval_history, filters={"user": user} if user else None,
                                                    image_context=query_description or None)
            except Exception as exc:
                logger.exception("检索失败")
                result.trace.notes.append(f"检索失败：{exc}")

        # 自动路由兜底：路由选中的库答不上（没资料，或精排最高分低于阈值——含兜底规则保留下来的低分段落）时，
        # 用同样的参数试路由排名接下来的两个库，换成分数达标且最高的那个。带图提问换库后，以图搜文按新库重做（图片识别结果沿用）。
        route_fallback = None
        if message and resolved_kb and (not kb_id or kb_id == "auto"):
            try:
                fb = retrieval_service.route_fallback(message, resolved_kb, result, mode=mode, top_k=top_k, overrides=overrides, history=retrieval_history,
                                                      filters={"user": user} if user else None, image_context=query_description or None)
                if fb is not None:
                    resolved_kb, result, route_fallback = fb
                    if i2t is not None:
                        # 以图搜文按新库重做，图片识别结果（图中文字、图片描述）沿用第一次的：与知识库无关，不再 OCR / 调多模态模型。
                        # 重做失败也不能沿用旧库的结果——会把两个库的资料、图片混进同一轮
                        first = i2t[2]
                        i2t = None
                        try:
                            i2t = self._image_to_text(resolved_kb, message, image_b64, top_k or settings.retrieval.rerank.top_k, recognized=(first.ocr_text, first.caption), user=user)
                        except Exception as exc:
                            logger.exception("换库后以图搜文失败")
                            image_error = f"图片检索失败：{exc}"
            except Exception as exc:
                logger.warning("自动路由兜底失败：%s", exc)

        # 图片侧：带图提问 → 图搜文结果并入上下文 + 图搜图 / 图文混合；纯文字提问 → 文搜图 + 图文关联
        images: list[ImageHit] = []
        image_search: dict[str, Any] = {}
        if resolved_kb and mm.enabled:
            try:
                if i2t is not None:
                    images, image_search = self._merge_image_side(message, *i2t, result, top_k or settings.retrieval.rerank.top_k, mm.chat_images,
                                                                  doc_ids=retrieval_service.visible_doc_ids(resolved_kb, message, user))
                elif image_error:
                    image_search = {"mode": "error", "routes": [], "notes": [image_error], "timingsMs": {}}
                elif message and not image_b64:
                    # 附图与文本检索用同一套文档可见性：已失效的旧版文档、当前用户无权查看的文档里的图不能带进回答
                    visible = retrieval_service.visible_doc_ids(resolved_kb, message, user)
                    images, mm_result = multimodal_service.images_for_answer(resolved_kb, message, result.contexts, doc_ids=visible)
                    image_search = {"mode": "text2image", "routes": list(mm_result.routes), "notes": mm_result.notes, "timingsMs": {k: round(v, 1) for k, v in mm_result.timings_ms.items()}}
            except Exception as exc:
                logger.exception("图片检索失败")
                image_search = {"mode": "error", "routes": [], "notes": [f"图片检索失败：{exc}"], "timingsMs": {}}

        if resolved_kb:
            payload = retrieval_service.to_response(resolved_kb, result)
        else:
            payload = {"queries": [message], "hydeDocument": None, "notes": result.trace.notes, "strategy": "none", "hops": [], "timingsMs": {}, "funnel": {}, "contexts": [], "weaved": [], "compression": {}}
        plan = {k: payload[k] for k in ("strategy", "queries", "hydeDocument", "notes", "hops", "timingsMs", "funnel", "compression")}
        plan["routeFallback"] = route_fallback
        plan["imageSearch"] = image_search or None
        # 长期记忆（LanceDB）：按用户召回，只用于个性化；随 plan 下发给前端并存进本轮链路
        user_id = str((user or {}).get("userId") or "default")
        memories = memory_service.recall(user_id, message) if message else []
        plan["memories"] = memories
        # Agent 湖检索增强：相似的历史问答作为参考信息（本轮检索到资料时才带，不作为引用来源）
        experience_notes, experience = lake_service.experience(message, kb_id=resolved_kb, session_id=session_id) if result.contexts else ("", [])
        plan["experience"] = [{k: e[k] for k in ("questionId", "question", "score")} for e in experience]
        # 用户手动选的知识库里一条资料都没检索到：看看别的知识库能不能答（快速探测），给前端"可能在 X 里有答案"的提示
        plan["kbSuggestion"] = None
        if message and kb_id and kb_id != "auto" and resolved_kb and not result.contexts:
            try:
                plan["kbSuggestion"] = retrieval_service.suggest_kb(message, resolved_kb)
            except Exception as exc:
                logger.warning("知识库建议失败：%s", exc)
        yield _sse("plan", plan)
        sources = [{"index": i + 1, **c} for i, c in enumerate(payload["contexts"])]
        yield _sse("sources", sources)
        yield _sse("relations", payload["weaved"])
        image_views: list[dict[str, Any]] = []
        if images and resolved_kb:
            names = retrieval_service.doc_names(resolved_kb)
            image_views = [multimodal_service.hit_view(resolved_kb, h, names, index=i + 1) for i, h in enumerate(images)]
        yield _sse("images", image_views)

        # 生成：需要看图（用户上传了图片，或要把检索到的图片交给模型）且有可看图的模型时，用多模态模型
        llm, llm_info = model_service.llm_info()
        vision = model_service.vision_llm() if (image_b64 or (images and mm.chat_attach_images > 0)) else None
        if vision is not None and vision is not llm:
            llm, llm_info = vision, model_service.llm_info(multimodal=True)[1]
        can_see = llm is not None and llm is vision
        payloads = multimodal_service.payloads(resolved_kb, images, mm.chat_attach_images) if (can_see and images and resolved_kb) else []
        note = None
        if image_b64 and not can_see:
            note = query_description or "（未能识别图片内容：没有配置多模态模型，也没有安装 OCR）"
        gen = AnswerGenerator(settings.generation, llm)
        full = ""
        failed = False
        try:
            for token in gen.stream(result, history=history, image_b64=image_b64 if can_see else None, extra_context=temp_doc_content, images=images or None, image_payloads=payloads or None, query_image_note=note, memories=[m["content"] for m in memories] or None, reference_notes=experience_notes or None):
                full += token
                yield _sse("message", {"content": token})
        except Exception as exc:
            logger.exception("生成失败")
            err = f"\n\n[生成失败] {exc}"
            full += err
            failed = True
            yield _sse("error", {"error": str(exc)})

        citations = parse_citations(full, len(sources))
        answer_id = None
        image_citations = parse_image_citations(full, len(image_views))
        # 引用校验：[n] 所在的句子在本轮第 n 段资料里找不到依据（串号 / 沿用了上一轮的编号）→ 随回答提示
        # 按交给模型时的样子校验（含来源 / 版本 / 生效期行）：只用正文时，引用版本号的句子会被误报为无依据
        source_texts = [format_context(c) for c in result.contexts] + ([temp_doc_content] if temp_doc_content else [])
        citation_check = check_citations(full, source_texts, question=message) if (citations and not failed) else []
        if not failed:
            # 数字依据：回答里的数值在本轮资料（含图片的图注 / 描述 / 图中文字、用户图片的识别结果）和问题里都找不到 → 提示核实。
            # 没有引用编号的回答也要查；写出了算式的推导值不算
            image_texts = ["\n".join(x for x in (h.record.caption, h.record.description, h.record.ocr_text) if x) for h in images]
            citation_check += check_numbers(full, source_texts + image_texts + ([query_description] if query_description else []), question=message)
        with SessionLocal() as db:
            repo = Repository(db)
            session = repo.get_session(session_id)
            if session is not None:
                if not session.kb_id and resolved_kb:
                    session.kb_id = resolved_kb
                trace = {**plan, "images": image_views, "imageCitations": image_citations, "citationCheck": citation_check}
                answer_id = repo.add_message(session, "assistant", full, sources_json=sources, relations_json=payload["weaved"], citations_json=citations, trace_json=trace, model=llm_info.get("model") or "extractive").id
        if message:
            # 后台：从这轮对话抽取长期记忆；提问 + 被引用的章节记为推荐交互（都不拖慢回答）
            memory_service.extract_async(user_id, message, full, session_id=session_id)
            if answer_id:
                # 一轮问答沉淀进 Agent 湖（后台：问题 + 检索日志 + 回答），供湖分析与之后的检索增强
                lake_service.record_turn_async(session_id=session_id, kb_id=resolved_kb, user_id=user_id, question_id=user_msg_id, question=message, q_ts=user_ts,
                                               answer_id=answer_id, answer=full, a_ts=time.time(), sources=sources, citations=citations, plan=plan,
                                               model=llm_info.get("model") or "extractive", mode=mode, error=failed)
            recommend_service.log_chat(user_id, resolved_kb, message, [sources[i - 1]["chunkId"] for i in citations if 0 < i <= len(sources)])
        yield _sse("done", {"citations": citations, "imageCitations": image_citations, "citationCheck": citation_check, "model": llm_info.get("model") or "extractive", "offline": llm is None, "vision": can_see, "messageId": answer_id})

    def _image_to_text(self, kb_id: str, message: str, image_b64: str, top_k: int, recognized: tuple[str, str] | None = None, user: dict[str, Any] | None = None) -> tuple[Any, Any, Any]:
        """带图提问第一步（在文本检索之前）：以图搜文——图片 →（图文向量直达 / 相似图片所在段落 / OCR·VLM 描述再检索）→ 正文，
        顺带得到图片的识别内容（query_description）。返回 (多模态检索器, 解码后的图片, 图搜文结果)。"""
        r = multimodal_service.retriever(kb_id)
        image = decode_image_b64(image_b64)
        visible = retrieval_service.visible_doc_ids(kb_id, message, user)
        return r, image, r.image_to_text(image, query=message, top_k=top_k, recognized=recognized, doc_ids=visible)

    def _merge_image_side(self, message: str, r: Any, image: Any, i2t: Any, result: RetrievalResult, top_k: int, n_images: int, doc_ids: set[str] | None = None) -> tuple[list[ImageHit], dict[str, Any]]:
        """带图提问第二步（文本检索之后）：图搜文的上下文并入本轮上下文；
        以图搜图：没有文字时按视觉相似度；有文字时做图文混合检索（"像这张图，但要……"）。"""
        docs = sorted(doc_ids) if doc_ids is not None else None
        sim = r.hybrid(message, image, top_k=n_images, doc_ids=docs) if message else r.image_to_image(image, top_k=n_images, doc_ids=docs)

        # 文本检索与图搜文的上下文交替合并、去重
        merged: list[Context] = []
        seen: set[str] = set()
        a, b = list(result.contexts), list(i2t.contexts)
        for pair in zip_longest(b, a) if not message else zip_longest(a, b):
            for c in pair:
                if c is not None and c.chunk.chunk_id not in seen:
                    seen.add(c.chunk.chunk_id)
                    merged.append(c)
        # 上限取 top_k 与文本检索条数中较大的：多跳 / Agentic 已经合并了多跳的上下文（例如 8 段），不能在这里又截回 top_k
        result.contexts = merged[:max(top_k, len(result.contexts))]
        tr = result.trace
        tr.strategy = "image" if not message else f"{tr.strategy}+image"
        tr.notes.extend(n for n in i2t.notes + sim.notes if n not in tr.notes)
        for route, items in i2t.routes.items():
            tr.routes[f"image:{route}"] = items
        info = {
            "mode": f"image2text+{sim.mode}",
            "routes": list(i2t.routes) + list(sim.routes),
            "notes": i2t.notes + sim.notes,
            "queryDescription": i2t.query_description,
            "timingsMs": {"image2text": round(i2t.timings_ms.get("total", 0.0), 1), sim.mode: round(sim.timings_ms.get("total", 0.0), 1)},
        }
        # 相似图片也要过门槛：图文混合按相对相关度（≥ 第一名的一半）；纯图搜图按"相似"阈值（CLIP 0.9 / 离线特征 0.7）。
        # 第一名总是保留——用户上传图片本身就是在问"库里有没有类似的"。
        hits = sim.images
        if hits:
            if message:
                keep = [h for h in hits if h.rank == 1 or h.score >= 0.5 * hits[0].score]
            else:
                near = r.images.embedder.similar_threshold if r.cfg.min_score is None else r.cfg.min_score
                keep = [h for h in hits if h.rank == 1 or h.score >= near]
            if len(keep) < len(hits):
                info["notes"].append(f"相似图片：{len(hits)} 张候选中 {len(keep)} 张通过相关性门槛")
            hits = keep
        return hits[:n_images], info


chat_service = ChatService()
