"""上下文压缩（Context Compression）。

无关噪声与多版本冲突会干扰模型判断，并放大 Lost-in-the-Middle。两种压缩：
- rule：按"与 query 的词汇重叠 + 位置"给句子打分，保留比例 ratio 的高分句（保序），
  表格行只保留含关键词的行并始终保留表头；短文本不压缩。零成本，确定性。
- llm：让轻量模型逐字抽取与问题相关的句子（不许改写），适合高价值请求。
"""

from __future__ import annotations

import re

from ..ingest.chunking.tokenizer import count_tokens, split_sentences
from .rerank import LexicalReranker

_LLM_PROMPT = """从下面的资料中，逐字摘出与问题直接相关的句子（可以有多句），保持原文顺序，不要改写、不要补充、不要总结。
如果没有任何相关句子，输出"无"。

问题：{query}

资料：
{text}"""


def compress_rule(query: str, text: str, *, ratio: float = 0.6, min_tokens: int = 120) -> str:
    if count_tokens(text) <= min_tokens:
        return text
    if text.lstrip().startswith("|"):
        return _compress_table(query, text)
    sentences = split_sentences(text)
    if len(sentences) <= 2:
        return text
    terms = LexicalReranker()._terms(query)
    scored: list[tuple[int, float]] = []
    for i, s in enumerate(sentences):
        low = s.lower()
        overlap = sum((2.0 if len(t) > 1 else 0.5) for t in terms if t in low)
        position = 1.0 if i == 0 else 0.3 / (1 + i)   # 首句常含主题
        scored.append((i, overlap + position))
    keep_n = max(2, int(round(len(sentences) * ratio)))
    keep = {i for i, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:keep_n]}
    keep.add(0)
    kept = [s for i, s in enumerate(sentences) if i in keep]
    return _join(kept)


def _compress_table(query: str, md: str) -> str:
    lines = md.splitlines()
    if len(lines) <= 6:
        return md
    header, rows = lines[:2], lines[2:]
    terms = LexicalReranker()._terms(query)
    hit = [r for r in rows if any(t in r.lower() for t in terms)]
    if not hit or len(hit) >= len(rows):
        return md
    return "\n".join(header + hit + [f"…（已省略 {len(rows) - len(hit)} 行不相关记录）"])


def _join(sentences: list[str]) -> str:
    out = ""
    for s in sentences:
        if not out:
            out = s
        elif out[-1:].isascii() and s[:1].isascii():
            out += " " + s
        else:
            out += s
    return out


class LLMCompressor:
    def __init__(self, llm) -> None:
        self.llm = llm

    def compress(self, query: str, text: str, *, min_tokens: int = 120) -> str:
        if count_tokens(text) <= min_tokens:
            return text
        try:
            out = self.llm.complete(_LLM_PROMPT.format(query=query, text=text), max_tokens=800).strip()
        except Exception:
            return compress_rule(query, text, min_tokens=min_tokens)
        if not out or out in ("无", "无。", "None"):
            return compress_rule(query, text, min_tokens=min_tokens)
        # 防止模型"改写"：抽取的句子必须大部分出现在原文里
        sents = [s for s in re.split(r"(?<=[。！？!?；;])", out) if s.strip()]
        verified = [s for s in sents if s.strip()[:20] in text]
        return "".join(verified) if verified and len(verified) >= len(sents) * 0.6 else compress_rule(query, text, min_tokens=min_tokens)
