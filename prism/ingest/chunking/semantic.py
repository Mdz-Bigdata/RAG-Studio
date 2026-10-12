"""语义切块：相邻句向量相似度骤降处即"话题转折点"。

做法与 LangChain SemanticChunker 一致：每句连同前后各一句拼成窗口再 embed（缓解单句信息量太少），
计算相邻窗口余弦距离，取第 P 百分位为阈值，距离超过阈值处下刀。
"""

from __future__ import annotations

import numpy as np

from .tokenizer import count_tokens


def semantic_split(
    sentences: list[str],
    embedder,
    *,
    percentile: float = 80.0,
    min_sentences: int = 2,
    max_tokens: int | None = None,
    buffer: int = 1,
) -> list[list[str]]:
    n = len(sentences)
    if n <= min_sentences:
        return [sentences] if sentences else []

    windows = []
    for i in range(n):
        lo, hi = max(0, i - buffer), min(n, i + buffer + 1)
        windows.append(" ".join(sentences[lo:hi]))
    vecs = np.asarray(embedder.embed_documents(windows), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    vecs = vecs / norms
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
    distances = 1.0 - sims
    if len(distances) == 0:
        return [sentences]
    threshold = float(np.percentile(distances, percentile))

    groups: list[list[str]] = []
    cur: list[str] = [sentences[0]]
    cur_tokens = count_tokens(sentences[0])
    for i, d in enumerate(distances):
        s = sentences[i + 1]
        st = count_tokens(s)
        over = max_tokens is not None and cur_tokens + st > max_tokens
        if (d > threshold and len(cur) >= min_sentences) or over:
            groups.append(cur)
            cur, cur_tokens = [s], st
        else:
            cur.append(s)
            cur_tokens += st
    if cur:
        groups.append(cur)
    return groups
