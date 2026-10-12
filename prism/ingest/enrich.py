"""索引增强（可选，索引阶段一次性 LLM 成本）：

1. Contextual Retrieval（Anthropic 2024）：为每个 parent 生成"这段在整篇文档里讲什么"的一两句说明，
   与正文一起 embed。arXiv 2604.01733：Contextual Dense 比普通 Dense 的 Recall@5 高 2.8pp。
2. 假设性问题（QA 对索引 / 反向 HyDE）：为每个 parent 生成 N 个用户可能会问的问题，拉近"问题 ↔ 文档"两种文体的向量距离。

两者都写在 chunk 上，并通过 Chunk.embed_text() 进入索引；child 继承 parent 的增强内容。

3. 摘要索引：为每个 parent 生成一段摘要，写在 parent.summary 上，由 Index 单独建一路"摘要召回"；
   命中摘要后返回原文。无 LLM 时退化为抽取式摘要（面包屑 + 首两句），流程照样可用。
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor

from ..config import EnrichConfig
from ..generation.llm import BaseLLM
from ..schemas import Chunk, ParsedDocument

logger = logging.getLogger(__name__)

_CONTEXT_PROMPT = """<document>
{document}
</document>

下面是文档中的一段：
<chunk>
{chunk}
</chunk>

请用一到两句话说明这段内容在整篇文档中的位置和主题（例如属于哪个制度 / 哪个产品 / 哪个时间段），
目的是帮助检索系统更好地理解这段文字。只输出说明文字，不要其他内容。"""

_QUESTIONS_PROMPT = """下面是一段知识库内容：
<chunk>
{chunk}
</chunk>

请站在用户角度，写出 {n} 个这段内容能够直接回答的问题。要求：问法自然、角度不同、包含关键实体。
每行一个问题，不要编号，不要其他内容。"""

_SUMMARY_PROMPT = """下面是知识库中的一段内容（所在章节：{crumb}）：
<chunk>
{chunk}
</chunk>

请用不超过 {n} 个字概括这段内容讲了什么：保留关键实体、数字、条件和结论，用陈述句，不要评价。只输出摘要。"""


class Enricher:
    def __init__(self, cfg: EnrichConfig, llm: BaseLLM | None) -> None:
        self.cfg = cfg
        self.llm = llm

    def enabled(self) -> bool:
        return self.cfg.summary_index or (self.llm is not None and (self.cfg.contextual_summary or self.cfg.hypothetical_questions))

    def enrich(self, doc: ParsedDocument, chunks: list[Chunk]) -> list[Chunk]:
        if not self.enabled():
            return chunks
        parents = [c for c in chunks if c.kind == "parent"]
        doc_text = _document_digest(doc)

        def work(p: Chunk) -> None:
            if self.cfg.summary_index:
                p.summary = self._summary(p)
            if self.llm is None:
                return
            try:
                if self.cfg.contextual_summary:
                    p.context_summary = self.llm.complete(_CONTEXT_PROMPT.format(document=doc_text, chunk=p.text), max_tokens=200).strip()
                if self.cfg.hypothetical_questions:
                    raw = self.llm.complete(_QUESTIONS_PROMPT.format(chunk=p.text, n=self.cfg.questions_per_chunk), max_tokens=400)
                    p.hypothetical_questions = _parse_lines(raw)[: self.cfg.questions_per_chunk]
            except Exception as exc:  # 单块失败不影响整体入库
                logger.warning("enrich 失败 %s: %s", p.chunk_id, exc)

        with ThreadPoolExecutor(max_workers=max(1, self.cfg.max_concurrency)) as ex:
            list(ex.map(work, parents))

        if self.llm is None:
            return chunks
        by_id = {p.chunk_id: p for p in parents}
        for c in chunks:
            if c.kind == "child" and c.parent_id in by_id:
                p = by_id[c.parent_id]
                c.context_summary = p.context_summary
                c.hypothetical_questions = list(p.hypothetical_questions)
        return chunks


    def _summary(self, p: Chunk) -> str:
        if self.llm is not None:
            try:
                s = self.llm.complete(_SUMMARY_PROMPT.format(crumb=p.breadcrumb_text or "-", chunk=p.text[:4000], n=self.cfg.summary_max_chars), max_tokens=400).strip()
                if s:
                    return s[: self.cfg.summary_max_chars * 2]
            except Exception as exc:
                logger.warning("摘要生成失败 %s，改用抽取式摘要：%s", p.chunk_id, exc)
        return extractive_summary(p, self.cfg.summary_max_chars)


def extractive_summary(p: Chunk, max_chars: int = 200) -> str:
    """无 LLM 的兜底摘要：章节路径 + 开头两句。"""
    from .chunking.tokenizer import split_sentences

    lead = "".join(split_sentences(p.text)[:2]) or p.text
    head = (p.breadcrumbs[-1] + "：") if p.breadcrumbs else ""
    return (head + lead)[:max_chars]


def _document_digest(doc: ParsedDocument, max_chars: int = 6000) -> str:
    """整篇太长时只取开头（标题 + 前几节），够模型判断"这是什么文档"。"""
    text = doc.markdown or "\n".join(b.text for b in doc.blocks)
    return text[:max_chars]


def _parse_lines(raw: str) -> list[str]:
    out = []
    for ln in raw.splitlines():
        ln = re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", ln).strip()
        if ln:
            out.append(ln)
    return out
