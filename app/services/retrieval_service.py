"""检索服务：为知识库构建 HybridRetriever（+ 多跳 / Agentic），输出可解释链路；跨库查询路由。"""

from __future__ import annotations

import logging

from typing import Any

from prism.retrieval import AgenticRAG, HybridRetriever, KBDescriptor, KBRouter, MultiHopRetriever
from prism.schemas import Context, RetrievalResult, ScoredChunk

from ..db import Repository, SessionLocal
from .index_service import index_service
from .model_service import model_service
from .settings_service import settings_service


logger = logging.getLogger(__name__)


class RetrievalService:
    def retriever(self, kb_id: str, overrides: dict[str, Any] | None = None) -> HybridRetriever:
        settings = settings_service.get()
        if overrides:
            settings = settings.with_overrides({"retrieval": overrides})
        index = index_service.get(kb_id)
        llm = model_service.llm(aux=True)
        reranker = model_service.reranker()
        return HybridRetriever(settings, index, llm=llm, reranker=reranker)

    def retrieve(self, kb_id: str, query: str, *, mode: str = "hybrid", top_k: int | None = None, use_transform: bool = True, overrides: dict[str, Any] | None = None, filters: dict[str, Any] | None = None, history: list[dict[str, str]] | None = None, image_context: str | None = None) -> RetrievalResult:
        """image_context：带图提问时图片识别出的内容（图片描述 + 图中文字），多跳判断"还缺什么"时要用。"""
        with index_service.reading(kb_id):        # 整库重建收尾替换目录时，等这次检索结束
            return self._retrieve(kb_id, query, mode=mode, top_k=top_k, use_transform=use_transform, overrides=overrides, filters=filters, history=history, image_context=image_context)

    def _retrieve(self, kb_id: str, query: str, *, mode: str, top_k: int | None, use_transform: bool, overrides: dict[str, Any] | None, filters: dict[str, Any] | None, history: list[dict[str, str]] | None, image_context: str | None = None) -> RetrievalResult:
        retriever = self.retriever(kb_id, overrides)
        llm = model_service.llm(aux=True)
        kwargs = dict(top_k=top_k, use_transform=use_transform, filters=filters, history=history)
        settings = settings_service.get()
        if mode == "multihop":
            return MultiHopRetriever(retriever, llm, cfg=settings.retrieval.multihop.model_copy(update={"enabled": True})).retrieve(query, image_context=image_context, **kwargs)
        if mode == "agentic":
            return AgenticRAG(retriever, llm, cfg=settings.retrieval.agentic.model_copy(update={"enabled": True})).run(query, **kwargs)
        return retriever.retrieve(query, **kwargs)

    def facets(self, kb_id: str, user: dict[str, Any] | None = None) -> dict[str, Any]:
        """元数据过滤的可选项：当前身份可见的现行文档里出现过的部门 / 标签 / 年份 / 文件类型。"""
        with index_service.reading(kb_id):
            return self.retriever(kb_id).facets({"user": user} if user else None)

    def visible_doc_ids(self, kb_id: str, query: str, user: dict[str, Any] | None = None) -> set[str] | None:
        """本轮问题能看到的文档（过期 / 权限，与文本检索同一套规则）：回答附图等旁路用。None = 不过滤。"""
        return self.retriever(kb_id).visible_doc_ids(query, {"user": user} if user else None)

    # --- 序列化为前端结构 -------------------------------------------------------------
    def to_response(self, kb_id: str, result: RetrievalResult) -> dict[str, Any]:
        index = index_service.get(kb_id)
        names = self.doc_names(kb_id)

        def item(sc: ScoredChunk) -> dict[str, Any]:
            c = index.store.get(sc.chunk_id)
            return {
                "chunkId": sc.chunk_id, "score": round(sc.score, 6), "rank": sc.rank, "route": sc.route,
                "text": (c.text[:400] if c else ""), "breadcrumb": (c.breadcrumb_text if c else ""), "docId": (c.doc_id if c else ""),
                "fileName": names.get(c.doc_id) if c else None,
            }

        def ctx(cx: Context) -> dict[str, Any]:
            c = cx.chunk
            return {
                "chunkId": c.chunk_id, "docId": c.doc_id, "fileName": names.get(c.doc_id), "breadcrumb": c.breadcrumb_text, "text": c.text,
                "compressedText": cx.compressed_text, "score": round(cx.score, 6), "tokenCount": c.token_count, "pageStart": c.page_start, "pageEnd": c.page_end,
                "matchedChildIds": cx.matched_child_ids, "weaved": cx.weaved, "weavedEntity": cx.weaved_entity,
            }

        tr = result.trace
        return {
            "query": result.query,
            "strategy": tr.strategy,
            "queries": tr.queries,
            "hydeDocument": tr.hyde_document,
            "notes": tr.notes,
            "routes": {k: [item(x) for x in v[:20]] for k, v in tr.routes.items()},
            "fused": [item(x) for x in tr.fused[:50]],
            "reranked": [item(x) for x in tr.reranked],
            "diversified": [item(x) for x in tr.diversified],
            "contexts": [ctx(x) for x in result.contexts],
            "weaved": [w.model_dump() for w in tr.weaved],
            "hops": [h.model_dump() for h in tr.hops],
            "compression": tr.compression,
            "metadataFilter": tr.metadata_filter or None,
            "timingsMs": {k: round(v, 1) for k, v in tr.timings_ms.items()},
            "funnel": {"recall": sum(len(v) for v in tr.routes.values()), "fused": len(tr.fused), "reranked": len(tr.reranked), "contexts": len(result.contexts)},
        }

    def doc_names(self, kb_id: str) -> dict[str, str]:
        with SessionLocal() as db:
            return {d.id: d.file_name for d in Repository(db).list_documents(kb_id)}

    # --- 跨库路由 -------------------------------------------------------------------
    def route(self, query: str, *, top_n: int = 1, mode: str = "embedding", kb_ids: list[str] | None = None) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            repo = Repository(db)
            kbs = repo.list_kbs()
            descriptors = []
            for kb in kbs:
                if kb_ids and kb.id not in kb_ids:
                    continue
                titles = [d.title or d.file_name for d in repo.list_documents(kb.id)[:12]]
                descriptors.append(KBDescriptor(kb_id=kb.id, name=kb.name, description=kb.description or "", tags=[t for t in [kb.category] if t], sample_titles=titles))
        router = KBRouter(embedder=model_service.embedder(), llm=model_service.llm(aux=True))
        return router.route(query, descriptors, top_n=top_n, mode=mode)

    # --- 自动路由兜底 -------------------------------------------------------------------
    @staticmethod
    def best_score(result: RetrievalResult) -> float | None:
        """精排最高分（取阈值过滤之前的 trace.reranked；没有精排时看上下文分数）。"""
        scores = [s.score for s in result.trace.reranked] if result.trace.reranked else [c.score for c in result.contexts]
        return float(max(scores)) if scores else None

    def is_weak(self, result: RetrievalResult, min_score: float | None) -> bool:
        """答不上的结果：没有上下文；或精排最高分低于阈值——兜底规则（min_score_agreement）保留下来的低分段落也算。"""
        if not result.contexts:
            return True
        if min_score is None or not result.trace.reranked:
            return False
        return (self.best_score(result) or 0.0) < min_score

    def route_fallback(self, query: str, current_kb: str, result: RetrievalResult, *, candidates: int = 2, **retrieve_kwargs: Any) -> tuple[str, RetrievalResult, dict[str, Any]] | None:
        """自动路由选中的库答不上时，到路由排名接下来的几个库里用同样的参数检索，换成分数达标且最高的那个。
        返回 (新知识库 id, 新检索结果, 兜底说明)；不需要兜底或别的库也答不上时返回 None。"""
        overrides = retrieve_kwargs.get("overrides")
        settings = settings_service.get().with_overrides({"retrieval": overrides}) if overrides else settings_service.get()
        min_score = settings.retrieval.rerank.min_score
        if not self.is_weak(result, min_score):
            return None
        try:
            routed = [r for r in self.route(query, top_n=candidates + 1) if r["kb_id"] != current_kb][:candidates]
        except Exception as exc:
            logger.warning("路由兜底：路由失败（%s）", exc)
            return None
        best: tuple[float, dict[str, Any], RetrievalResult] | None = None
        for cand in routed:
            try:
                res = self.retrieve(cand["kb_id"], query, **retrieve_kwargs)
            except Exception as exc:
                logger.warning("路由兜底：在 %s 里检索失败（%s）", cand["name"], exc)
                continue
            if self.is_weak(res, min_score):
                continue
            score = self.best_score(res) or 0.0
            if best is None or score > best[0]:
                best = (score, cand, res)
        if best is None:
            return None
        score, cand, res = best
        with SessionLocal() as db:
            kb = Repository(db).get_kb(current_kb)
        cur = self.best_score(result) if result.contexts else None
        info = {"fromKbId": current_kb, "fromName": kb.name if kb else current_kb, "fromScore": None if cur is None else round(cur, 4),
                "toKbId": cand["kb_id"], "toName": cand["name"], "toScore": round(score, 4),
                "reason": "没有检索到资料" if not result.contexts else f"精排最高分 {cur:.3f} 低于阈值 {min_score}"}
        res.trace.notes.append(f"自动路由兜底：「{info['fromName']}」{info['reason']}，改用「{info['toName']}」（精排最高分 {score:.3f}）")
        return cand["kb_id"], res, info

    def suggest_kb(self, query: str, exclude_kb: str | None, *, top_n: int = 2, full: bool = False, check_answer: bool = False) -> dict[str, Any] | None:
        """问题在 exclude_kb 里答不上时，找能回答它的其它知识库：先按知识库描述路由出候选，再到候选库里实际检索确认。

        full=False：快速探测（不做查询改写 / HyDE），给问答页"这个问题可能在 X 里有答案"的提示用；
        full=True：与智能问答相同的检索链路，check_answer 时再生成一次回答看是否拒答（缺口复测用）。
        返回检索分最高（check_answer 时还要回答不拒答）的库：{kbId, name, routeScore, score, fileName, refused, answer}；都没有返回 None。"""
        if not query.strip():
            return None
        try:
            candidates = [r for r in self.route(query, top_n=top_n + 1) if r["kb_id"] != exclude_kb][:top_n]
        except Exception as exc:
            logger.warning("知识库建议：路由失败（%s）", exc)
            return None
        low = settings_service.get().lake.low_score
        probes = []                                   # 每个候选库都实际检索一次，取分数最高的（重排分在库之间可比）
        for order, cand in enumerate(candidates):
            try:
                result = self.retrieve(cand["kb_id"], query, use_transform=full)
            except Exception as exc:
                logger.warning("知识库建议：在 %s 里检索失败（%s）", cand["name"], exc)
                continue
            if not result.contexts:
                continue
            top = max(result.contexts, key=lambda c: c.score)
            if top.score >= low:
                probes.append((float(top.score), -order, cand, result, top))
        for score, _, cand, result, top in sorted(probes, key=lambda x: (x[0], x[1]), reverse=True):
            out = {"kbId": cand["kb_id"], "name": cand["name"], "routeScore": round(float(cand.get("score") or 0.0), 4), "score": round(score, 4),
                   "fileName": self.doc_names(cand["kb_id"]).get(top.chunk.doc_id), "refused": None, "answer": None}
            if check_answer:
                from prism.generation import AnswerGenerator

                ans = AnswerGenerator(settings_service.get().generation, model_service.llm()).answer(result)
                if ans.refused:
                    continue                          # 分数最高的库也答不上：试下一个
                out.update(refused=False, answer=ans.text)
            return out
        return None

    def entities(self, kb_id: str) -> list[dict[str, Any]]:
        idx = index_service.get(kb_id)
        return [r.model_dump() for r in idx.entities.relations]

    def graph(self, kb_id: str, *, section_depth: int = 2, max_entities: int = 200) -> dict[str, Any]:
        from prism.graph.topology import build_topology

        idx = index_service.get(kb_id)
        g = settings_service.get().retrieval.graph
        out = build_topology(idx.store, idx.entities, section_depth=section_depth, max_entities=max_entities)
        out["config"] = {"unweaver": g.unweaver_enabled, "llmExtraction": g.llm_extraction}
        return out


retrieval_service = RetrievalService()
