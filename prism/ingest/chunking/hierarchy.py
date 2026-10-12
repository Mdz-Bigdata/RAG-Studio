"""多粒度分层索引：在 parent（段落级）/ child（检索块）之外，再生成两层检索单元。

- 章节级 section：同一章（按 section_depth 级标题归并）下的连续 parent 合成一个单元，回答"XX 是什么 / 概述一下 / 有哪些方面"这类宽泛问题；
- 句子级 sentence：parent 按句切开，回答"几天 / 多少钱 / 哪一条"这类细节问题，命中后仍返回所属 parent 作上下文。

检索时由 retrieval.hierarchy 按问题类型决定加哪一层（见 retrieval/hierarchy.py）。
"""

from __future__ import annotations

from ...config import ChunkingConfig
from ...schemas import Chunk
from .tokenizer import count_tokens, split_sentences

_SKIP_SENTENCE_TYPES = {"table", "code", "equation"}


def build_hierarchy(doc_id: str, chunks: list[Chunk], cfg: ChunkingConfig) -> list[Chunk]:
    parents = sorted((c for c in chunks if c.kind == "parent"), key=lambda c: c.order)
    return _sections(doc_id, parents, cfg) + _sentences(parents, cfg)


def _sections(doc_id: str, parents: list[Chunk], cfg: ChunkingConfig) -> list[Chunk]:
    depth = 1 + max(1, cfg.section_depth)            # breadcrumbs[0] 是文档标题
    key = lambda c: (c.doc_id, tuple(c.breadcrumbs[:depth]))
    groups: list[list[Chunk]] = []
    for p in parents:
        if groups and key(groups[-1][0]) == key(p):
            groups[-1].append(p)
        else:
            groups.append([p])
    out: list[Chunk] = []
    for si, group in enumerate(groups):
        texts, used = [], 0
        for p in group:
            t = p.token_count or count_tokens(p.text)
            if texts and used + t > cfg.section_max_tokens:
                break
            texts.append(p.text)
            used += t
        pages = [x for p in group for x in (p.page_start, p.page_end) if x is not None]
        out.append(Chunk(
            chunk_id=f"{doc_id}:s{si:03d}", doc_id=doc_id, kind="section", text="\n\n".join(texts),
            child_ids=[p.chunk_id for p in group], order=group[0].order, breadcrumbs=list(group[0].breadcrumbs[:depth]),
            page_start=min(pages) if pages else None, page_end=max(pages) if pages else None, token_count=used,
            block_types=sorted({b for p in group for b in p.block_types}), metadata=dict(group[0].metadata),
        ))
    return out


def _sentences(parents: list[Chunk], cfg: ChunkingConfig) -> list[Chunk]:
    out: list[Chunk] = []
    for p in parents:
        if set(p.block_types) & _SKIP_SENTENCE_TYPES and len(p.block_types) == 1:
            continue   # 纯表格 / 代码块按句切没有意义，child 已按行切好
        merged: list[str] = []
        for s in split_sentences(p.text):
            s = s.strip()
            if not s:
                continue
            if merged and len(s) < cfg.sentence_min_chars:
                merged[-1] += s
            else:
                merged.append(s)
        for j, s in enumerate(merged):
            out.append(Chunk(
                chunk_id=f"{p.chunk_id}:t{j:03d}", doc_id=p.doc_id, kind="sentence", text=s, parent_id=p.chunk_id,
                order=p.order, breadcrumbs=list(p.breadcrumbs), page_start=p.page_start, page_end=p.page_end,
                token_count=count_tokens(s), block_types=list(p.block_types), metadata=dict(p.metadata),
            ))
    return out
