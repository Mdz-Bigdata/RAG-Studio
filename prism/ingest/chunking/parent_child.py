"""Parent-Child（Small-to-Big）：小块检索、大块使用。

- child ≈ 150 token，语义聚焦，向量与 BM25 都建在 child 上
- parent ≈ 500-800 token，命中 child 后取回 parent 交给 LLM，上下文完整
- 表格 child 按行组切分，并且每个 child 都重复表头，保证"这行来自哪张表 / 哪一列"不丢
"""

from __future__ import annotations

from ...schemas import BlockType
from ..parsers.html_table import markdown_table_rows
from .tokenizer import count_tokens, split_sentences


def build_child_texts(text: str, child_tokens: int, overlap_ratio: float, block_types: list[str]) -> list[str]:
    if count_tokens(text) <= child_tokens:
        return [text]
    if block_types == [BlockType.TABLE.value] and text.lstrip().startswith("|"):
        return _split_table(text, child_tokens)
    if block_types == [BlockType.CODE.value]:
        return _split_lines(text, child_tokens)
    return _sentence_windows(text, child_tokens, overlap_ratio)


def _sentence_windows(text: str, child_tokens: int, overlap_ratio: float) -> list[str]:
    sentences = split_sentences(text, max_tokens=child_tokens)
    if not sentences:
        return [text]
    overlap_budget = int(child_tokens * overlap_ratio)
    windows: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for s in sentences:
        st = count_tokens(s)
        if buf and buf_tokens + st > child_tokens:
            windows.append(_join(buf))
            keep: list[str] = []
            kept = 0
            for prev in reversed(buf):
                pt = count_tokens(prev)
                if kept + pt > overlap_budget:
                    break
                keep.insert(0, prev)
                kept += pt
            buf, buf_tokens = keep, kept
        buf.append(s)
        buf_tokens += st
    if buf:
        windows.append(_join(buf))
    # 去掉与前一个窗口完全相同的尾巴窗口
    dedup: list[str] = []
    for w in windows:
        if not dedup or w != dedup[-1]:
            dedup.append(w)
    return dedup


def _join(sentences: list[str]) -> str:
    out = ""
    for s in sentences:
        if not out:
            out = s
        elif out[-1:].isascii() and out[-1:].isalnum() and s[:1].isascii():
            out += " " + s
        else:
            out += s
    return out


def _split_table(md: str, child_tokens: int) -> list[str]:
    header, rows = markdown_table_rows(md)
    if not rows:
        return [md]
    lead = "\n".join(header)
    lead_tokens = count_tokens(lead)
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = lead_tokens
    for r in rows:
        rt = count_tokens(r)
        if buf and buf_tokens + rt > child_tokens:
            chunks.append(lead + "\n" + "\n".join(buf))
            buf, buf_tokens = [], lead_tokens
        buf.append(r)
        buf_tokens += rt
    if buf:
        chunks.append(lead + "\n" + "\n".join(buf))
    return chunks


def _split_lines(text: str, child_tokens: int) -> list[str]:
    lines = text.split("\n")
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for ln in lines:
        lt = count_tokens(ln) + 1
        if buf and buf_tokens + lt > child_tokens:
            chunks.append("\n".join(buf))
            buf, buf_tokens = [], 0
        buf.append(ln)
        buf_tokens += lt
    if buf:
        chunks.append("\n".join(buf))
    return chunks
