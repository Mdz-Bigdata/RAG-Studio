"""Agentic RAG：让模型自主决定检索策略并迭代验证。

流程（最多 max_steps 步）：
1. 规划：LLM 根据问题选择策略 —— precise（精确编号 / 型号：关 HyDE / Multi-Query，加大 BM25 权重）、
   hybrid（默认双路 + 重排）、multihop（多跳）、none（寒暄 / 与知识库无关，不检索）；
2. 执行对应策略；
3. 反思：资料是否足够？不够则给出新的检索问句再来一轮，够则结束。
无 LLM 时按启发式规则选策略（精确 token → precise；多跳句式 → multihop；否则 hybrid）。
"""

from __future__ import annotations

import json
import re

from ..config import AgenticConfig
from ..schemas import Context, HopTrace, RetrievalResult, RetrievalTrace
from .multihop import MultiHopRetriever, _MULTI, _carry_filter
from .query_transform import has_precise_tokens

_PLAN_PROMPT = """你是检索策略规划器。根据用户问题选择最合适的检索策略：
- precise：问题包含订单号 / 错误码 / 型号 / 版本号 / 具体日期等精确 token，需要关键词精确匹配
- multihop：需要对比、分步骤或串联多个信息点才能回答
- hybrid：一般的语义 / 政策 / 流程类问题
- none：寒暄、闲聊或明显与知识库无关，不需要检索

用户问题：{query}
只输出 JSON：{{"strategy": "precise|multihop|hybrid|none", "query": "（可选）更适合检索的改写", "reason": "一句话"}}"""

_REFLECT_PROMPT = """用户问题：{query}

已检索到的资料：
{summary}

这些资料是否足以回答问题？如果足够，只输出 DONE；否则输出一个新的检索问句（只输出问句）。"""


class AgenticRAG:
    def __init__(self, retriever, llm=None, cfg: AgenticConfig | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.cfg = cfg or AgenticConfig(enabled=True)

    def run(self, query: str, **kwargs) -> RetrievalResult:
        plan = self._plan(query)
        strategy = plan["strategy"]
        search_query = plan.get("query") or query
        notes = [f"agentic 规划：{strategy}（{plan.get('reason', '')}）"]

        if strategy == "none":
            trace = RetrievalTrace(original_query=query, strategy="none", queries=[query], notes=notes + ["判定无需检索"])
            return RetrievalResult(query=query, contexts=[], trace=trace)

        if strategy == "multihop":
            result = MultiHopRetriever(self.retriever, self.llm, cfg=None).retrieve(search_query, **kwargs)
        elif strategy == "precise":
            overrides = dict(kwargs.get("overrides") or {})
            overrides.setdefault("query_transform", {}).update({"hyde": False, "multi_query": 0})
            overrides.setdefault("fusion", {}).update({"weight_sparse": 1.5})
            kwargs["overrides"] = overrides
            result = self.retriever.retrieve(search_query, **kwargs)
        else:
            result = self.retriever.retrieve(search_query, **kwargs)

        result.trace.strategy = f"agentic:{strategy}"
        result.trace.notes = notes + result.trace.notes
        if not result.trace.hops:
            result.trace.hops.append(HopTrace(hop=1, query=search_query, found_chunk_ids=[c.chunk.chunk_id for c in result.contexts], note=strategy))

        kwargs.setdefault("auto_filter", _carry_filter(result))
        # 反思循环
        step = 2
        merged: dict[str, Context] = {c.chunk.chunk_id: c for c in result.contexts}
        while self.llm is not None and step <= self.cfg.max_steps:
            nxt = self._reflect(query, list(merged.values()))
            if not nxt:
                result.trace.notes.append(f"agentic 反思：第 {step - 1} 轮后判定资料充分")
                break
            more = self.retriever.retrieve(nxt, **kwargs)
            new_ids = [c.chunk.chunk_id for c in more.contexts if c.chunk.chunk_id not in merged]
            for c in more.contexts:
                merged.setdefault(c.chunk.chunk_id, c)
            result.trace.hops.append(HopTrace(hop=step, query=nxt, found_chunk_ids=new_ids, note="反思补充检索"))
            if not new_ids:
                break
            step += 1
        contexts = sorted(merged.values(), key=lambda c: c.score, reverse=True)
        top_k = kwargs.get("top_k") or self.retriever.cfg.rerank.top_k
        result.contexts = contexts[: max(top_k, len(result.contexts))]
        return result

    # ----------------------------------------------------------------------------------
    def _plan(self, query: str) -> dict:
        if self.llm is not None:
            try:
                raw = self.llm.complete(_PLAN_PROMPT.format(query=query), max_tokens=200)
                m = re.search(r"\{.*\}", raw, re.S)
                data = json.loads(m.group(0)) if m else {}
                strategy = str(data.get("strategy", "hybrid")).lower()
                if strategy in ("precise", "multihop", "hybrid", "none"):
                    return {"strategy": strategy, "query": (data.get("query") or "").strip() or None, "reason": str(data.get("reason", ""))}
            except Exception:
                pass
        if has_precise_tokens(query):
            return {"strategy": "precise", "reason": "检测到精确 token（规则）"}
        if _MULTI.search(query):
            return {"strategy": "multihop", "reason": "多跳句式（规则）"}
        return {"strategy": "hybrid", "reason": "默认（规则）"}

    def _reflect(self, query: str, contexts: list[Context]) -> str | None:
        summary = "\n".join(f"- {c.chunk.breadcrumb_text}: {c.text_for_llm()[:200]}" for c in contexts[:6]) or "（无）"
        try:
            raw = self.llm.complete(_REFLECT_PROMPT.format(query=query, summary=summary), max_tokens=120).strip()
        except Exception:
            return None
        if not raw or raw.upper().startswith("DONE"):
            return None
        q = raw.splitlines()[0].strip().strip('"“”')
        return q if q and q != query else None
