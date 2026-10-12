"""混合检索器：漏斗式管理的完整链路。

    用户 Query
      │  查询层：（多轮）问题归并 / 改写 / Multi-Query / 分解 / HyDE（精确查询跳过）
      ▼
      ├── 路 1：向量检索（每条查询 top-N=50）
      ├── 路 2：BM25 关键词（每条查询 top-N=50）
      ├── 路 3：HyDE 假设文档 → 向量检索
      ├── 路 4：SQL 结构化检索（库里有 Excel / CSV 表时：text-to-SQL 或规则 → 命中的行）
      ├── 路 5：摘要索引（入库时生成了 parent 摘要：命中摘要 → 返回原文）
      └── 路 6：分层索引（按问题类型：概念型加章节级、细节型加句子级）
         检索前先按元数据过滤：过期文档、用户无权查看的文档（权限级别 / 部门），
         以及从问题里识别出的范围限定（年份 / 部门 / 标签 / 文件类型，见 metadata_filter）
         各路并发执行；多条查询 × 多路的结果一起做 RRF（即 RAG-Fusion）
      ▼  融合层：RRF k=60（或 α=0.5 凸组合）→ 粗召回池 recall_top_k ≥ 50
      ▼  精排层：Cross-Encoder / API / LLM / 词法 → rerank_top_n = 10
      ▼  多样化：MMR λ=0.7（可选）
      ▼  粒度层：child → parent，自动合并，Lost-in-the-middle 排序 → context_top_n = 3-5
      ▼  织网层：UnWeaver 实体定义织入（可选）
      ▼  压缩层：规则 / LLM 上下文压缩（可选）
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

from ..config import RetrievalConfig, Settings
from ..generation.llm import BaseLLM
from ..graph.unweaver import weave_contexts
from ..index import Index
from ..ingest.chunking.tokenizer import count_tokens
from ..schemas import Chunk, Context, DocMetadata, RetrievalResult, RetrievalTrace, ScoredChunk
from .compress import LLMCompressor, compress_rule
from .diversify import mmr_select
from .fusion import convex_fusion, merge_same_route, rrf_fusion
from .granularity import expand_contexts
from .hierarchy import LEVEL_LABEL, levels_for
from .metadata_filter import describe, extract_llm, extract_rule, facets, matches
from .query_transform import QueryPlan, QueryTransformer
from .rerank import BaseReranker, build_reranker
from .temporal import history_intent

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, settings: Settings, index: Index, llm: BaseLLM | None = None, reranker: BaseReranker | None = None) -> None:
        self.settings = settings
        self.index = index
        self.llm = llm
        self.cfg: RetrievalConfig = settings.retrieval
        self.transformer = QueryTransformer(self.cfg.query_transform, llm)
        self.reranker = reranker or build_reranker(self.cfg.rerank, llm=llm)
        self._sql_aggregate: Context | None = None

    # ----------------------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        on_date: date | None = None,
        top_k: int | None = None,
        use_transform: bool = True,
        history: list[dict[str, str]] | None = None,
        overrides: dict[str, Any] | None = None,
        auto_filter: dict[str, list[Any]] | bool | None = None,
    ) -> RetrievalResult:
        """auto_filter：None = 按配置从问题里识别元数据条件；False = 本次不识别；
        dict = 直接使用上游已经识别好的条件（多跳的后续查询沿用首跳的，不重复调用模型）。"""
        cfg = self.cfg if not overrides else RetrievalConfig.model_validate(_merge(self.cfg.model_dump(mode="json"), overrides))
        transformer = self.transformer if cfg.query_transform == self.cfg.query_transform else QueryTransformer(cfg.query_transform, self.llm)
        t0 = time.perf_counter()
        trace = RetrievalTrace(original_query=query)
        top_k = top_k or cfg.rerank.top_k

        include_expired = False
        if cfg.filter_expired and cfg.history_intent:
            docs = self.index.store.docs.values()
            hit = history_intent(query, docs, on_date) if any(not md.is_effective(on_date) for md in docs) else None
            if hit:
                include_expired = True
                trace.notes.append(f"时效：问题在问历史规定（“{hit}”），本次检索包含已失效的文档，资料中会标注失效")
        allowed = self._allowed_ids(filters, on_date, cfg, include_expired=include_expired)
        user = (filters or {}).get("user") if cfg.access.enabled else None
        if user is not None:
            hidden = sum(1 for md in self.index.store.docs.values() if not md.is_visible_to(user))
            if hidden:
                trace.notes.append(f"权限过滤：当前身份（级别 {user.get('level', 0)}{' · ' + str(user['department']) if user.get('department') else ''}）无权查看 {hidden} 篇文档，已排除")
        if allowed is not None and not allowed:
            trace.notes.append("元数据过滤后无候选")
            return RetrievalResult(query=query, contexts=[], trace=trace)

        # 自查询在后台识别条件，与查询变换并行：两边各调一次模型，耗时不叠加
        resolve_auto = self._start_self_query(query, history, auto_filter, allowed, cfg)
        plan = transformer.plan(query, history=history) if use_transform else QueryPlan(original=query, queries=[query])
        allowed = self._apply_auto_filter(resolve_auto(), allowed, cfg, trace)
        trace.queries = list(plan.queries)
        trace.hyde_document = plan.hyde_document
        trace.notes.extend(plan.notes)
        trace.timings_ms["query_transform"] = (time.perf_counter() - t0) * 1000

        # 召回层
        t1 = time.perf_counter()
        rankings = self._recall(plan, allowed, trace, cfg)
        trace.timings_ms["recall"] = (time.perf_counter() - t1) * 1000

        # 融合层
        t2 = time.perf_counter()
        fused = self._fuse(rankings, cfg)
        candidates_n = max(cfg.rerank.candidates, cfg.rerank.keep_n, top_k)
        fused = fused[:candidates_n]
        trace.fused = [ScoredChunk(chunk_id=cid, score=s, rank=i + 1, route="fused") for i, (cid, s) in enumerate(fused)]
        trace.timings_ms["fusion"] = (time.perf_counter() - t2) * 1000
        if cfg.rerank.enabled and 0 < len(fused) < 50:
            trace.notes.append(f"融合候选仅 {len(fused)} 条（<50），Reranker 选择空间不足")

        # 精排层：保留 keep_n（rerank_top_n）
        t3 = time.perf_counter()
        keep_n = max(cfg.rerank.keep_n, top_k)
        reranked = self._rerank(query, fused, keep_n)
        pop_note = getattr(self.reranker, "pop_note", None)
        note = pop_note() if callable(pop_note) else None
        if note:
            trace.notes.append(note)
        trace.reranked = [ScoredChunk(chunk_id=cid, score=s, rank=i + 1, route="rerank") for i, (cid, s) in enumerate(reranked)]
        trace.timings_ms["rerank"] = (time.perf_counter() - t3) * 1000

        if cfg.rerank.min_score is not None:
            above = [(cid, s) for cid, s in reranked if s >= cfg.rerank.min_score]
            if not above and reranked and cfg.rerank.min_score_agreement:
                above = self._agreement_rescue(rankings, reranked)
                if above:
                    trace.notes.append(f"精排最高分低于阈值 {cfg.rerank.min_score}，但向量与 BM25 两路第一名指向同一段落：交叉验证通过，不判无答案")
            if not above:
                trace.notes.append(f"精排最高分低于阈值 {cfg.rerank.min_score}，判定为库中无答案")
            reranked = above

        if cfg.rerank.min_score_ratio and reranked:
            # 动态 top-K：上下文不是越多越好，和最优候选差距过大的不凑数
            cutoff = reranked[0][1] * cfg.rerank.min_score_ratio
            kept = [(cid, s) for cid, s in reranked if s >= cutoff]
            if len(kept) < len(reranked):
                trace.notes.append(f"相对阈值 {cfg.rerank.min_score_ratio}：{len(reranked)} → {len(kept)} 条（低于 top-1 × ratio 的候选被丢弃）")
            reranked = kept

        # 多样化：MMR
        if cfg.diversify.enabled and reranked:
            t4 = time.perf_counter()
            vectors = self.index.vectors.get_vectors([cid for cid, _ in reranked])
            reranked = mmr_select(reranked, vectors, lambda_=cfg.diversify.mmr_lambda, k=len(reranked))
            trace.diversified = [ScoredChunk(chunk_id=cid, score=s, rank=i + 1, route="mmr") for i, (cid, s) in enumerate(reranked)]
            trace.timings_ms["mmr"] = (time.perf_counter() - t4) * 1000
        scored = [ScoredChunk(chunk_id=cid, score=s, rank=i + 1, route="final") for i, (cid, s) in enumerate(reranked)]

        # 粒度层：child → parent，取 context_top_n
        contexts = expand_contexts(scored, self.index.store, cfg.granularity)
        contexts = _take_top(contexts, top_k, cfg.granularity.order)
        if self._sql_aggregate is not None:
            # 聚合类 SQL（COUNT / AVG …）没有对应的行：把结果表作为一条上下文放在最前面
            contexts.insert(0, self._sql_aggregate)

        # 织网层
        if cfg.graph.unweaver_enabled and contexts:
            contexts, relations = weave_contexts(contexts, self.index.entities, self.index.store.parent_of, max_weaved=cfg.graph.max_weaved, allow=self._weave_allow(allowed))
            contexts, relations = self._gate_weaved(query, contexts, relations, cfg, trace)
            trace.weaved = relations

        # 压缩层
        if cfg.compression.enabled and contexts:
            before = sum(count_tokens(c.chunk.text) for c in contexts)
            compressor = LLMCompressor(self.llm) if (cfg.compression.mode == "llm" and self.llm is not None) else None
            for c in contexts:
                if compressor is not None:
                    c.compressed_text = compressor.compress(query, c.chunk.text, min_tokens=cfg.compression.min_tokens)
                else:
                    c.compressed_text = compress_rule(query, c.chunk.text, ratio=cfg.compression.ratio, min_tokens=cfg.compression.min_tokens)
                if c.compressed_text == c.chunk.text:
                    c.compressed_text = None
            after = sum(count_tokens(c.text_for_llm()) for c in contexts)
            trace.compression = {"mode": cfg.compression.mode, "tokens_before": before, "tokens_after": after}

        if trace.metadata_filter.get("applied") and not contexts and cfg.metadata_filter.fallback:
            # 识别出的条件是推测：按它查不到东西时，宁可退回全库，也不要把"条件猜错"变成"库中无答案"
            retry = self.retrieve(query, filters=filters, on_date=on_date, top_k=top_k, use_transform=use_transform,
                                  history=history, overrides=overrides, auto_filter=False)
            retry.trace.metadata_filter = {**trace.metadata_filter, "applied": False, "reason": "no_results"}
            retry.trace.notes.insert(0, f"元数据过滤：按“{describe(trace.metadata_filter['conditions'])}”没有检索到结果，已去掉该条件重新检索")
            return retry

        trace.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        return RetrievalResult(query=query, contexts=contexts, trace=trace)

    # ----------------------------------------------------------------------------------
    def _scope_docs(self, allowed: set[str] | None) -> dict[str, DocMetadata]:
        """当前还在检索范围内的文档（已过时效 / 权限 / 显式过滤）。"""
        store = self.index.store
        if allowed is None:
            return dict(store.docs)
        ids = {store.chunks[c].doc_id for c in allowed if c in store.chunks}
        return {doc_id: md for doc_id, md in store.docs.items() if doc_id in ids}

    def _start_self_query(self, query: str, history: list[dict[str, str]] | None, auto_filter: dict[str, list[Any]] | bool | None,
                          allowed: set[str] | None, cfg: RetrievalConfig) -> Callable[[], tuple[dict[str, list[Any]], str, str | None]]:
        """返回一个取结果的函数：(条件, 来源, 备注)。模型调用放在后台线程里。"""
        mf = cfg.metadata_filter
        if isinstance(auto_filter, dict):
            return lambda: (auto_filter, "upstream", None)
        if auto_filter is False or mf.mode == "off":
            return lambda: ({}, "off", None)
        available = facets(self._scope_docs(allowed).values())
        # 每个属性最多只有一种取值时没有可缩小的范围，不必识别
        if not any(len(available[f]) > 1 for f in mf.fields):
            return lambda: ({}, "none", None)
        if mf.mode == "rule" or self.llm is None:
            return lambda: (extract_rule(query, available, mf.fields), "rule", None)

        def run() -> tuple[dict[str, list[Any]], str, str | None]:
            try:
                return extract_llm(query, available, mf.fields, self.llm, history), "llm", None
            except Exception as exc:
                logger.warning("自查询识别条件失败，退回规则：%s", type(exc).__name__)
                return extract_rule(query, available, mf.fields), "rule", "元数据过滤：模型识别条件失败，本次只按规则识别年份和部门"

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="self-query")
        future = pool.submit(run)
        pool.shutdown(wait=False)
        return future.result

    def _apply_auto_filter(self, resolved: tuple[dict[str, list[Any]], str, str | None], allowed: set[str] | None,
                           cfg: RetrievalConfig, trace: RetrievalTrace) -> set[str] | None:
        conditions, source, note = resolved
        if note:
            trace.notes.append(note)
        conditions = {f: list(v) for f, v in conditions.items() if v}
        if not conditions:
            return allowed
        scope = self._scope_docs(allowed)
        keep = {doc_id for doc_id, md in scope.items() if matches(md, conditions, include_missing=cfg.metadata_filter.include_missing)}
        info = {"conditions": conditions, "source": source, "docs_before": len(scope), "docs_after": len(keep), "applied": False}
        trace.metadata_filter = info
        label = "沿用首次检索的条件" if source == "upstream" else "从问题中识别"
        if not keep:
            info["reason"] = "no_match"
            trace.notes.append(f"元数据过滤（{label}）：{describe(conditions)}，没有匹配的文档，本次不使用该条件")
            return allowed
        if len(keep) == len(scope):
            info["reason"] = "no_narrowing"   # 范围内的文档本来就都满足，不必提示
            return allowed
        info["applied"] = True
        trace.notes.append(f"元数据过滤（{label}）：{describe(conditions)} → 候选文档 {len(scope)} 篇缩小到 {len(keep)} 篇"
                           + ("（未填该属性的文档保留）" if cfg.metadata_filter.include_missing else ""))
        store = self.index.store
        return {cid for cid in (allowed if allowed is not None else store.child_ids()) if store.chunks[cid].doc_id in keep}

    # ----------------------------------------------------------------------------------
    def _allowed_ids(self, filters: dict[str, Any] | None, on_date: date | None, cfg: RetrievalConfig, *, include_expired: bool = False) -> set[str] | None:
        need_effective = cfg.filter_expired and not include_expired
        if not filters and not need_effective:
            return None
        user = (filters or {}).get("user") if cfg.access.enabled else None

        def pred(c: Chunk, md: DocMetadata | None) -> bool:
            if need_effective and md is not None and not md.is_effective(on_date):
                return False
            if user is not None and md is not None and not md.is_visible_to(user):
                return False
            if filters:
                for key, want in filters.items():
                    if key == "user":
                        continue
                    if key == "doc_id":
                        if c.doc_id not in _as_list(want):
                            return False
                    elif key in ("year", "file_format") or (key in ("department", "tags") and md is not None):
                        # 显式指定的条件按字面执行：没填该属性的文档不算匹配
                        if not matches(md, {key: _as_list(want)}, include_missing=False):
                            return False
                    elif key == "tags":
                        if not set(_as_list(want)) & set(c.metadata.get("tags") or []):
                            return False
                    elif key in ("department", "version", "source", "title"):
                        if str(c.metadata.get(key)) not in {str(x) for x in _as_list(want)}:
                            return False
                    elif key == "sheet" and c.metadata.get("sheet") != want:
                        return False
            return True

        return self.index.store.filter_child_ids(pred)

    def facets(self, filters: dict[str, Any] | None = None, on_date: date | None = None) -> dict[str, Any]:
        """当前身份能检索到的文档里，各元数据属性有哪些取值、各几篇：给界面的过滤选项用。"""
        scope = self._scope_docs(self._allowed_ids({"user": (filters or {}).get("user")} if (filters or {}).get("user") else None, on_date, self.cfg))
        return {"docs": len(scope), **{f: [{"value": v, "count": n} for v, n in values.items()] for f, values in facets(scope.values()).items()}}

    def visible_doc_ids(self, query: str, filters: dict[str, Any] | None = None, on_date: date | None = None) -> set[str] | None:
        """文档级可见性（过期 / 权限），给文本检索之外的旁路用（回答附图）：与 retrieve 同一套规则，
        问历史规定时放行过期文档。None = 不需要过滤。"""
        cfg = self.cfg
        docs = self.index.store.docs
        user = (filters or {}).get("user") if cfg.access.enabled else None
        expired_ok = not cfg.filter_expired or bool(cfg.history_intent and history_intent(query, docs.values(), on_date))
        if expired_ok and user is None:
            return None
        return {doc_id for doc_id, md in docs.items() if (expired_ok or md.is_effective(on_date)) and (user is None or md.is_visible_to(user))}

    def _weave_allow(self, allowed: set[str] | None):
        """织入块（parent）是否通过了本次检索的元数据过滤：allowed 是 child id，换算成它们所属的 parent。"""
        if allowed is None:
            return None
        store = self.index.store
        parents = {store.chunks[c].parent_id or c for c in allowed if c in store.chunks}
        return lambda chunk: chunk.chunk_id in allowed or chunk.chunk_id in parents

    def _gate_weaved(self, query: str, contexts: list[Context], relations: list[Any], cfg: RetrievalConfig, trace: RetrievalTrace) -> tuple[list[Context], list[Any]]:
        """织入块过一遍精排：与问题的相关度低于门槛的不织入，留下的用真实精排分（而不是固定的 0.5）。"""
        woven = [c for c in contexts if c.weaved]
        threshold = cfg.graph.weave_min_score if cfg.graph.weave_min_score is not None else cfg.rerank.min_score
        if not woven or not cfg.graph.weave_relevance or threshold is None or not cfg.rerank.enabled:
            return contexts, relations
        cands = [(c.chunk.chunk_id, (f"{c.chunk.breadcrumb_text}\n" if c.chunk.breadcrumbs else "") + c.chunk.text) for c in woven]
        try:
            scores = dict(self.reranker.rerank(query, cands, len(cands)))
        except Exception as exc:
            logger.warning("织入块精排失败，保留全部织入：%s", exc)
            return contexts, relations
        keep = {cid for cid, s in scores.items() if s >= threshold}
        out: list[Context] = []
        for c in contexts:
            if not c.weaved:
                out.append(c)
            elif c.chunk.chunk_id in keep:
                c.score = float(scores[c.chunk.chunk_id])
                out.append(c)
        if len(keep) < len(woven):
            trace.notes.append(f"织入门槛：{len(woven)} 段候选中 {len(woven) - len(keep)} 段与问题的精排分低于 {threshold}，未织入")
        store = self.index.store

        def woven_id(chunk_id: str) -> str:
            p = store.parent_of(chunk_id)
            return p.chunk_id if p is not None else chunk_id

        return out, [r for r in relations if woven_id(r.source_chunk_id) in keep]

    def _agreement_rescue(self, rankings: dict[str, list[tuple[str, float]]], reranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """原问题的向量路与 BM25 路第一名落在同一个 parent → 返回该 parent 下的精排结果（没有就用向量路第一名）。"""
        dense, sparse = rankings.get("dense:q0") or [], rankings.get("sparse:q0") or []
        if not dense or not sparse:
            return []

        def parent_id(cid: str) -> str | None:
            p = self.index.store.parent_of(cid)
            return p.chunk_id if p is not None else None

        agreed = parent_id(dense[0][0])
        if agreed is None or agreed != parent_id(sparse[0][0]):
            return []
        same = [(cid, s) for cid, s in reranked if parent_id(cid) == agreed]
        return same or [(dense[0][0], 0.0)]

    def _recall(self, plan: QueryPlan, allowed: set[str] | None, trace: RetrievalTrace, cfg: RetrievalConfig) -> dict[str, list[tuple[str, float]]]:
        top_n = cfg.top_n_per_route
        self.index.bm25._ensure_index()   # 并发前先建好，避免多个线程同时触发首次构建
        # 关键词路：默认 bm25s + jieba；向量后端是 LanceDB 时可改用它的原生全文检索（ICU 分词）
        fts = getattr(self.index.vectors, "fts_search", None) if cfg.sparse_backend == "lancedb_fts" else None
        if cfg.sparse_backend == "lancedb_fts" and fts is None:
            trace.notes.append("sparse_backend=lancedb_fts 需要 vector_store.backend: lancedb，本次关键词路仍用 BM25")
        elif fts is not None:
            self.index.vectors.ensure_fts()
        q_vecs = self.index.embedder.embed_queries(plan.queries)
        jobs: dict[str, Any] = {}
        for i, (q, vec) in enumerate(zip(plan.queries, q_vecs)):
            jobs[f"dense:q{i}"] = lambda vec=vec: self.index.vectors.search(vec, top_n, allowed)
            if fts is not None:
                jobs[f"sparse:q{i}"] = lambda q=q: fts(self.index.sparse_query(q), top_n, allowed)
            else:
                jobs[f"sparse:q{i}"] = lambda q=q: self.index.bm25.search(self.index.sparse_query(q), top_n, allowed)
        if plan.hyde_document:
            jobs["dense:hyde"] = lambda: self.index.vectors.search(self.index.embedder.embed_documents([plan.hyde_document])[0], top_n, allowed)
        # 摘要索引 / 分层索引：辅助索引内部先做向量 + BM25 小融合，作为一路交给主融合层
        aux_allowed = self._aux_allowed(allowed)
        if cfg.summary.enabled and len(self.index.aux["summary"]):
            for i, (q, vec) in enumerate(zip(plan.queries, q_vecs)):
                jobs[f"summary:q{i}"] = lambda q=q, vec=vec: self.index.aux["summary"].search(q, vec, top_n, aux_allowed["summary"])
        if cfg.hierarchy.enabled and (len(self.index.aux["section"]) or len(self.index.aux["sentence"])):
            levels, kind = levels_for(plan.original, cfg.hierarchy.routing)
            trace.notes.append(f"分层路由：{LEVEL_LABEL.get(kind, '章节级 + 句子级全部参与')}")
            for level in levels:
                if len(self.index.aux[level]):
                    for i, (q, vec) in enumerate(zip(plan.queries, q_vecs)):
                        jobs[f"{level}:q{i}"] = lambda q=q, vec=vec, level=level: self.index.aux[level].search(q, vec, top_n, aux_allowed[level])
        sql_box: dict[str, Any] = {}
        if cfg.sql.enabled and len(self.index.tables):
            def run_sql() -> list[tuple[str, float]]:
                res = self.index.tables.search(plan.original, llm=self.llm if cfg.sql.text_to_sql else None, top_n=cfg.sql.top_n, allowed_ids=allowed)
                sql_box["res"] = res
                return res.ranking
            jobs["sql:q0"] = run_sql

        if cfg.parallel_recall and len(jobs) > 1:
            # 多路召回并发：各路互不依赖，耗时取最慢的一路而不是求和（远程向量库 / text-to-SQL 时收益明显）
            with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
                futures = {k: pool.submit(fn) for k, fn in jobs.items()}
                rankings = {k: f.result() for k, f in futures.items()}
        else:
            rankings = {k: fn() for k, fn in jobs.items()}

        self._sql_aggregate = None
        res = sql_box.get("res")
        if res is not None:
            if res.note:
                trace.notes.append(f"SQL 路：{res.note}")
            if res.rows:
                t = self.index.tables.tables.get(res.table)
                self._sql_aggregate = Context(chunk=Chunk(chunk_id=f"sql:{abs(hash(res.sql)) % 10**8}", doc_id=t.doc_id if t else "", kind="parent", text=res.aggregate_text, breadcrumbs=[f"表：{t.sheet}" if t else "SQL", "SQL 计算结果"], block_types=["table"], token_count=count_tokens(res.aggregate_text)), score=1.0)
            if not res.ranking:
                rankings.pop("sql:q0", None)
        for route, r in rankings.items():
            trace.routes[route] = [ScoredChunk(chunk_id=cid, score=s, rank=k + 1, route=route) for k, (cid, s) in enumerate(r)]
        return rankings

    def _aux_allowed(self, allowed: set[str] | None) -> dict[str, set[str] | None]:
        """主过滤结果是 child id；换算成摘要（parent id）/ 句子 / 章节单元各自可见的 id。"""
        if allowed is None:
            return {"summary": None, "section": None, "sentence": None}
        store = self.index.store
        parents = {store.chunks[c].parent_id for c in allowed if c in store.chunks}
        out: dict[str, set[str] | None] = {"summary": set(parents), "section": set(), "sentence": set()}
        for c in store.chunks.values():
            if c.kind == "sentence" and c.parent_id in parents:
                out["sentence"].add(c.chunk_id)
            elif c.kind == "section" and parents.intersection(c.child_ids):
                out["section"].add(c.chunk_id)
        return out

    def _fuse(self, rankings: dict[str, list[tuple[str, float]]], cfg: RetrievalConfig) -> list[tuple[str, float]]:
        fcfg = cfg.fusion
        if fcfg.method == "convex":
            dense = merge_same_route([r for k, r in rankings.items() if k.startswith(("dense", "summary", "section", "sentence"))])
            # 凸组合只有两路：SQL 命中属于"精确匹配"，并入稀疏路
            sparse = merge_same_route([r for k, r in rankings.items() if k.startswith(("sparse", "sql"))])
            return convex_fusion(dense, sparse, fcfg.convex_alpha)
        def weight(route: str) -> float:
            if route.startswith("dense"):
                return fcfg.weight_dense
            if route.startswith("sql"):
                return cfg.sql.weight
            if route.startswith("summary"):
                return cfg.summary.weight
            if route.startswith(("section", "sentence")):
                return cfg.hierarchy.weight
            return fcfg.weight_sparse

        return rrf_fusion([(r, weight(k)) for k, r in rankings.items() if r], k=fcfg.rrf_k)

    def _rerank(self, query: str, fused: list[tuple[str, float]], keep_n: int) -> list[tuple[str, float]]:
        if not fused:
            return []
        candidates: list[tuple[str, str]] = []
        for cid, _ in fused:
            c = self.index.store.get(cid)
            if c is not None:
                # 给 reranker 的文本带上面包屑：让它知道"这行来自哪张表 / 哪个章节"
                text = (f"{c.breadcrumb_text}\n" if c.breadcrumbs else "") + c.text
                candidates.append((cid, text))
        return self.reranker.rerank(query, candidates, keep_n)

    # ----------------------------------------------------------------------------------
    # 消融评测 / 游乐场用：单路 / 无重排
    def retrieve_dense(self, query: str, top_n: int | None = None) -> list[tuple[str, float]]:
        vec = self.index.embedder.embed_query(query)
        return self.index.vectors.search(vec, top_n or self.cfg.top_n_per_route, self._allowed_ids(None, None, self.cfg))

    def retrieve_sparse(self, query: str, top_n: int | None = None) -> list[tuple[str, float]]:
        return self.index.bm25.search(self.index.sparse_query(query), top_n or self.cfg.top_n_per_route, self._allowed_ids(None, None, self.cfg))

    def retrieve_fused(self, query: str) -> list[tuple[str, float]]:
        rankings = {"dense:q0": self.retrieve_dense(query), "sparse:q0": self.retrieve_sparse(query)}
        return self._fuse(rankings, self.cfg)


def _take_top(contexts, top_k: int, order: str):
    if len(contexts) <= top_k:
        return contexts
    best = sorted(contexts, key=lambda c: c.score, reverse=True)[:top_k]
    if order == "lost_in_the_middle":
        head, tail = [], []
        for i, ctx in enumerate(best):
            (head if i % 2 == 0 else tail).append(ctx)
        return head + tail[::-1]
    return best


def _as_list(v: Any) -> list[Any]:
    return list(v) if isinstance(v, (list, tuple, set)) else [v]


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out
