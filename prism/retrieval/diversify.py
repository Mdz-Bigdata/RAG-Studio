"""MMR（Maximal Marginal Relevance）多样化。

精排后 top-K 往往是同一段落的多个切片（尤其 child 粒度）：相关但冗余，挤掉了其他角度的证据。
MMR 逐个挑选：score = λ · rel(d) − (1−λ) · max_{s∈已选} sim(d, s)。λ = 0.7 是常用起点。
"""

from __future__ import annotations

import numpy as np


def mmr_select(
    relevance: list[tuple[str, float]],
    vectors: dict[str, np.ndarray],
    *,
    lambda_: float = 0.7,
    k: int = 5,
) -> list[tuple[str, float]]:
    """relevance: [(id, score 0-1)] 已按分数降序；vectors: id → 归一化向量。返回 [(id, mmr_score)]。"""
    if not relevance:
        return []
    ids = [cid for cid, _ in relevance]
    rel = {cid: float(s) for cid, s in relevance}
    lo, hi = min(rel.values()), max(rel.values())
    if hi - lo > 1e-9:
        rel = {cid: (s - lo) / (hi - lo) for cid, s in rel.items()}
    else:
        rel = {cid: 1.0 for cid in rel}

    selected: list[tuple[str, float]] = []
    remaining = list(ids)
    while remaining and len(selected) < k:
        best_id, best_score = None, -1e9
        for cid in remaining:
            v = vectors.get(cid)
            redundancy = 0.0
            if v is not None and selected:
                sims = [float(np.dot(v, vectors[s])) for s, _ in selected if s in vectors]
                redundancy = max(sims) if sims else 0.0
            score = lambda_ * rel[cid] - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_id, best_score = cid, score
        selected.append((best_id, best_score))
        remaining.remove(best_id)
    return selected
