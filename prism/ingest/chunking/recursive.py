"""递归切分：段落 → 句子 → 字符，作为超长文本块的兜底切法。"""

from __future__ import annotations

from .tokenizer import count_tokens, split_sentences


def recursive_split(text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text.strip()] if text.strip() else []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        units = split_sentences(text, max_tokens=max_tokens)
        joiner = ""
    else:
        units = paragraphs
        joiner = "\n\n"
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for u in units:
        ut = count_tokens(u)
        if ut > max_tokens:
            if buf:
                chunks.append(joiner.join(buf))
                buf, buf_tokens = [], 0
            chunks.extend(recursive_split(u, max_tokens) if joiner else _hard_split(u, max_tokens))
            continue
        if buf and buf_tokens + ut > max_tokens:
            chunks.append(joiner.join(buf))
            if overlap_tokens > 0:
                keep: list[str] = []
                kept = 0
                for prev in reversed(buf):
                    pt = count_tokens(prev)
                    if kept + pt > overlap_tokens:
                        break
                    keep.insert(0, prev)
                    kept += pt
                buf, buf_tokens = keep, kept
            else:
                buf, buf_tokens = [], 0
        buf.append(u)
        buf_tokens += ut
    if buf:
        chunks.append(joiner.join(buf))
    return [c for c in chunks if c.strip()]


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """既无段落也无句读的超长文本（如超长表格行 / URL 列表）按字符硬切。"""
    approx_chars = max(20, max_tokens)  # 中文 1 字 ≈ 1 token
    return [text[i:i + approx_chars] for i in range(0, len(text), approx_chars)]
