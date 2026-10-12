"""多跳检索：迭代检索串联信息。

"A 和 B 有什么不同"、"X 的负责人所在部门的报销标准"这类问题，一次检索拿不全证据。
做法：第 1 跳按原问题检索 → LLM 判断"已有资料是否足够回答，缺什么" → 生成下一跳检索问句 → 再检索 → 合并，
最多 max_hops 跳。无 LLM 时退化为启发式拆分子问题各查一次再合并。
"""

from __future__ import annotations

import re

from ..config import MultiHopConfig
from ..schemas import Context, HopTrace, RetrievalResult

_SPLIT = re.compile(r"(?:和|与|以及|及|、|跟|对比|相比|分别)")
_MULTI = re.compile(r"(有什么(不同|区别)|分别|对比|相比|区别|差异|之间)")

_FOLLOWUP_PROMPT = """用户问题：{query}
{image}
已经检索到的资料摘要：
{summary}

判断：仅凭以上资料能否完整回答用户问题？
- 如果能，只输出：DONE
- 如果不能，输出下一步应该检索的一个具体问句（只输出问句本身），用于补齐缺失的信息。"""


def _carry_filter(result: RetrievalResult):
    info = result.trace.metadata_filter
    return info["conditions"] if info.get("applied") else False


class MultiHopRetriever:
    def __init__(self, retriever, llm=None, cfg: MultiHopConfig | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.cfg = cfg or MultiHopConfig(enabled=True)

    def retrieve(self, query: str, *, image_context: str | None = None, **kwargs) -> RetrievalResult:
        """image_context：用户随问题上传的图片识别出的内容（图片描述 + 图中文字）。判断"缺什么"时一起看，
        否则"这段代码违反了哪些规范"这类问题，下一跳只会说"资料里没有代码内容"，补查不到点子上。"""
        max_hops = max(1, self.cfg.max_hops)
        first = self.retriever.retrieve(query, **kwargs)
        # 后续查询沿用首跳识别出的元数据条件：子问题往往丢了"2023 年""财务部"这类限定，也省掉重复的模型调用
        kwargs.setdefault("auto_filter", _carry_filter(first))
        merged: dict[str, Context] = {c.chunk.chunk_id: c for c in first.contexts}
        trace = first.trace
        trace.strategy = "multihop"
        trace.hops.append(HopTrace(hop=1, query=query, found_chunk_ids=list(merged), note="初始检索"))

        followups = self._plan_followups(query, list(merged.values()), image_context=image_context) if self.llm is not None else self._heuristic_subqueries(query)
        hop = 2
        for fq in followups:
            if hop > max_hops:
                break
            res = self.retriever.retrieve(fq, **kwargs)
            new_ids = []
            for c in res.contexts:
                if c.chunk.chunk_id not in merged:
                    merged[c.chunk.chunk_id] = c
                    new_ids.append(c.chunk.chunk_id)
                elif c.score > merged[c.chunk.chunk_id].score:
                    merged[c.chunk.chunk_id].score = c.score
            trace.hops.append(HopTrace(hop=hop, query=fq, found_chunk_ids=new_ids, note=f"新增 {len(new_ids)} 条"))
            for route, items in res.trace.routes.items():
                trace.routes[f"hop{hop}:{route}"] = items
            hop += 1
            if self.llm is not None and hop <= max_hops:
                nxt = self._plan_followups(query, list(merged.values()), max_n=1, image_context=image_context)
                if not nxt:
                    break
                followups = followups + nxt

        contexts = sorted(merged.values(), key=lambda c: c.score, reverse=True)
        top_k = kwargs.get("top_k") or self.retriever.cfg.rerank.top_k
        contexts = contexts[: max(top_k, len(first.contexts))]
        trace.notes.append(f"multihop: 共 {len(trace.hops)} 跳，合并 {len(contexts)} 条上下文")
        return RetrievalResult(query=query, contexts=contexts, trace=trace)

    # ----------------------------------------------------------------------------------
    def _plan_followups(self, query: str, contexts: list[Context], max_n: int = 2, image_context: str | None = None) -> list[str]:
        summary = "\n".join(f"- {c.chunk.breadcrumb_text}: {c.text_for_llm()[:200]}" for c in contexts[:6]) or "（无）"
        image = f"\n用户随问题上传了一张图片，识别出的内容：\n{image_context.strip()[:800]}\n" if (image_context or "").strip() else ""
        out: list[str] = []
        try:
            raw = self.llm.complete(_FOLLOWUP_PROMPT.format(query=query, image=image, summary=summary), max_tokens=120).strip()
        except Exception:
            return self._heuristic_subqueries(query)[:max_n]
        if not raw or raw.upper().startswith("DONE"):
            return []
        q = raw.splitlines()[0].strip().strip('"“”')
        if q and q != query:
            out.append(q)
        return out[:max_n]

    @staticmethod
    def _heuristic_subqueries(query: str) -> list[str]:
        if not _MULTI.search(query):
            return []
        core = re.sub(r"(有什么(不同|区别)|有何(不同|区别)|的区别|的差异|分别是什么|分别)", "", query)
        parts = [p.strip("？?，, ") for p in _SPLIT.split(core) if p.strip("？?，, ")]
        parts = [p for p in parts if 1 < len(p) < 40]
        return parts[:3] if len(parts) >= 2 else []
