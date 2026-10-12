"""粒度层：用于搜索的块 ≠ 用于回答的上下文。

- Small-to-Big：命中 child → 返回所属 parent（检索精确，生成完整）
- 句子窗口：以命中 child 为中心，向前后各扩展 N 个 child
- 自动合并：同一 parent 内多个 child 命中 → 只保留一个 parent，分数取最大
- 上下文预算：按 token 上限截断
- Lost in the Middle：LLM 对上下文中部注意力下降，把最相关的放两头（1, 3, 5, ..., 6, 4, 2）
"""

from __future__ import annotations

from ..config import GranularityConfig
from ..index.doc_store import ChunkStore
from ..ingest.chunking.tokenizer import count_tokens
from ..schemas import Chunk, Context, ScoredChunk


def expand_contexts(reranked: list[ScoredChunk], store: ChunkStore, cfg: GranularityConfig) -> list[Context]:
    contexts: dict[str, Context] = {}
    order: list[str] = []
    for sc in reranked:
        child = store.get(sc.chunk_id)
        if child is None:
            continue
        if cfg.return_parents:
            target = store.parent_of(child.chunk_id) or child
        else:
            target = child
        if cfg.sentence_window > 0 and not cfg.return_parents:
            target = _window_chunk(child, store.neighbors(child.chunk_id, cfg.sentence_window))

        key = target.chunk_id
        if key in contexts:
            if cfg.auto_merge_adjacent:
                ctx = contexts[key]
                ctx.score = max(ctx.score, sc.score)
                if child.chunk_id not in ctx.matched_child_ids:
                    ctx.matched_child_ids.append(child.chunk_id)
            continue
        contexts[key] = Context(chunk=target, score=sc.score, matched_child_ids=[child.chunk_id])
        order.append(key)

    # 章节级单元已经包含了它下面的段落：同时命中时只保留章节，分数取两者较大值，避免同一段内容重复进上下文
    for sec in [contexts[k] for k in order if contexts[k].chunk.kind == "section"]:
        for pid in sec.chunk.child_ids:
            covered = contexts.pop(pid, None)
            if covered is not None:
                sec.score = max(sec.score, covered.score)
                sec.matched_child_ids.extend(x for x in covered.matched_child_ids if x not in sec.matched_child_ids)
    order = [k for k in order if k in contexts]
    ranked = sorted((contexts[k] for k in order), key=lambda c: c.score, reverse=True)

    # 上下文预算
    budget = cfg.max_context_tokens
    kept: list[Context] = []
    used = 0
    for ctx in ranked:
        t = ctx.chunk.token_count or count_tokens(ctx.chunk.text)
        if kept and used + t > budget:
            continue
        kept.append(ctx)
        used += t

    if cfg.order == "lost_in_the_middle":
        return _lost_in_the_middle(kept)
    return kept


def _window_chunk(center: Chunk, neighbors: list[Chunk]) -> Chunk:
    ordered = sorted(neighbors + [center], key=lambda c: c.order)
    text = "\n".join(c.text for c in ordered)
    merged = center.model_copy(deep=True)
    merged.text = text
    merged.token_count = count_tokens(text)
    merged.page_start = min((c.page_start for c in ordered if c.page_start is not None), default=center.page_start)
    merged.page_end = max((c.page_end for c in ordered if c.page_end is not None), default=center.page_end)
    return merged


def _lost_in_the_middle(items: list[Context]) -> list[Context]:
    """最相关的放开头，其次放结尾，依次向中间填充。"""
    head: list[Context] = []
    tail: list[Context] = []
    for i, ctx in enumerate(items):
        (head if i % 2 == 0 else tail).append(ctx)
    return head + tail[::-1]
