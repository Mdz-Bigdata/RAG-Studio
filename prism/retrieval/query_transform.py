"""查询层：用户怎么问 ≠ 系统应该怎么搜。

- 多轮归并：结合对话历史把"那它多少钱"改写成独立完整的问题
- 改写 / 压缩：口语、指代不清、噪音词 → 规范检索表达（"那啥我想问一下…" → "办公室租金 报销政策"）
- Multi-Query（RAG Fusion）：3~5 个不同角度问法分别检索，结果合并；原始问题必须保留
- 分解：多跳问题（"A 和 B 有什么不同"）拆成可独立检索的子问题
- HyDE：先生成假设答案再用其向量检索；短 query 有效（+15~25%），
  但 query 含精确数字 / 编号 / 型号 / 日期时必须跳过（假设答案里的数字是编的，会把检索带偏）
- Step-back：问题太具体、库里只有通用背景时，先退一步问背景

所有变换都可能"理解错原问题"，所以结果里始终保留原始 query。
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..config import QueryTransformConfig
from ..generation.llm import BaseLLM
from ..index.bm25_store import PRECISE_TOKEN

_MULTI_HOP = re.compile(r"(有什么(不同|区别)|分别|对比|相比|和.+的.+(区别|差异)|以及|同时.+吗)")
_DATE = re.compile(r"\d{4}\s*年|\d{1,2}\s*月|\d{4}-\d{1,2}|Q[1-4]", re.I)
_PRONOUN = re.compile(r"(它|他|她|这个|那个|该|其|上面|上述|刚才|前面)")

CONDENSE_PROMPT = """下面是一段对话历史和用户的最新问题。请把最新问题改写成一个不依赖上下文、可以独立理解的完整问题（补全指代、保留关键实体）。只输出改写后的问题。
对话历史：
{history}
最新问题：{query}
独立问题："""

REWRITE_PROMPT = """把用户的问题改写成适合知识库检索的简短书面表达：去掉口语和客套，保留核心实体、动作和限定条件；指代不明的词根据上下文补全。只输出改写后的一句话。
用户问题：{query}
改写："""

MULTI_QUERY_PROMPT = """针对下面的问题，生成 {n} 个含义相同但角度或措辞不同的检索问句（例如换用同义词、改成陈述式关键词、从制度名 / 流程名角度问）。每行一个，不要编号。
问题：{query}"""

DECOMPOSE_PROMPT = """把下面的问题拆成 1-3 个可以独立检索的子问题，每行一个，不要编号。如果问题本身就是单一问题，只输出原问题。
问题：{query}"""

HYDE_PROMPT = """根据下面的问题，写一段约 100 字的参考答案。不需要保证准确，但要尽量覆盖可能相关的关键词、制度名、流程步骤和概念，用陈述句。不要出现具体数字。
问题：{query}
答案："""

STEP_BACK_PROMPT = """把下面这个具体问题往上抽象一层，改写成一个更通用的背景问题（回答它有助于回答原问题）。只输出一句话。
问题：{query}"""


@dataclass
class QueryPlan:
    original: str
    queries: list[str] = field(default_factory=list)   # 用于 dense + sparse 的所有查询（含原始）
    hyde_document: str | None = None                  # 仅用于 dense
    notes: list[str] = field(default_factory=list)


def has_precise_tokens(query: str) -> bool:
    return bool(PRECISE_TOKEN.search(query)) or bool(_DATE.search(query))


class QueryTransformer:
    def __init__(self, cfg: QueryTransformConfig, llm: BaseLLM | None) -> None:
        self.cfg = cfg
        self.llm = llm

    def plan(self, query: str, history: list[dict[str, str]] | None = None) -> QueryPlan:
        query = query.strip()
        plan = QueryPlan(original=query, queries=[query] if self.cfg.keep_original else [])
        if self.llm is None:
            plan.notes.append("no_llm: 跳过查询变换")
            if not plan.queries:
                plan.queries.append(query)
            return plan

        # 多轮归并：只在有历史且问题像追问（短 / 含指代）时触发
        if self.cfg.condense_history and history and (len(query) <= 20 or _PRONOUN.search(query)):
            hist = "\n".join(f"{m.get('role', 'user')}: {str(m.get('content', ''))[:300]}" for m in history[-6:])
            condensed = self._ask(CONDENSE_PROMPT.format(history=hist, query=query), 150)
            if condensed and condensed != query:
                plan.queries.insert(0, condensed) if not self.cfg.keep_original else plan.queries.append(condensed)
                plan.notes.append(f"condense: {condensed}")
                query = condensed

        precise = has_precise_tokens(query)
        if precise:
            plan.notes.append("检测到精确 token（编号 / 数字 / 日期）")

        # 改写 / 分解 / Multi-Query / Step-back / HyDE 互不依赖：并发调用，耗时取最慢的一个而不是求和
        # （推理模型每次 1~5 秒，串行时查询层动辄十几秒）。结果按固定顺序合并，保证可复现。
        jobs: dict[str, Any] = {}
        if self.cfg.rewrite:
            jobs["rewrite"] = lambda: self._ask(REWRITE_PROMPT.format(query=query), 120)
        if self.cfg.decompose and _MULTI_HOP.search(query):
            jobs["decompose"] = lambda: self._ask_lines(DECOMPOSE_PROMPT.format(query=query), 300)
        if self.cfg.multi_query > 0:
            jobs["multi_query"] = lambda: self._ask_lines(MULTI_QUERY_PROMPT.format(query=query, n=self.cfg.multi_query), 400)
        if self.cfg.step_back:
            jobs["step_back"] = lambda: self._ask(STEP_BACK_PROMPT.format(query=query), 120)
        if self.cfg.hyde:
            if precise and self.cfg.skip_hyde_on_precise_tokens:
                plan.notes.append("hyde: 跳过（精确查询反模式）")
            elif len(query) > self.cfg.hyde_max_query_chars:
                plan.notes.append("hyde: 跳过（query 足够长）")
            else:
                jobs["hyde"] = lambda: self._ask(HYDE_PROMPT.format(query=query), 300)
        if len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                futures = {k: pool.submit(fn) for k, fn in jobs.items()}
                out = {k: f.result() for k, f in futures.items()}
        else:
            out = {k: fn() for k, fn in jobs.items()}

        rewritten = out.get("rewrite")
        if rewritten and rewritten != query:
            plan.queries.append(rewritten)
            plan.notes.append(f"rewrite: {rewritten}")
        subs = [x for x in out.get("decompose") or [] if x != query]
        if subs:
            plan.queries.extend(subs[:3])
            plan.notes.append(f"decompose: {subs[:3]}")
        alts = out.get("multi_query") or []
        plan.queries.extend(alts[: self.cfg.multi_query])
        if alts:
            plan.notes.append(f"multi_query: {len(alts[: self.cfg.multi_query])} 条")
        if out.get("step_back"):
            plan.queries.append(out["step_back"])
            plan.notes.append(f"step_back: {out['step_back']}")
        for key in ("rewrite", "multi_query", "hyde"):
            if key in out and not out[key]:
                plan.notes.append(f"{key}: 模型没有返回内容（已忽略）")
        if out.get("hyde"):
            plan.hyde_document = out["hyde"]
            plan.notes.append("hyde: 已生成假设文档")

        # 去重、保序
        seen: set[str] = set()
        plan.queries = [q for q in plan.queries if q and not (q in seen or seen.add(q))]
        if not plan.queries:
            plan.queries = [query]
        return plan

    # ----------------------------------------------------------------------------------
    def _ask(self, prompt: str, max_tokens: int) -> str:
        try:
            return self.llm.complete(prompt, max_tokens=max_tokens).strip().strip('"“”')
        except Exception:  # 查询变换失败不能阻断检索
            return ""

    def _ask_lines(self, prompt: str, max_tokens: int) -> list[str]:
        raw = self._ask(prompt, max_tokens)
        out: list[str] = []
        for ln in raw.splitlines():
            ln = re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", ln).strip()
            if ln:
                out.append(ln)
        return out
