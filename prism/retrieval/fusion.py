"""融合层：多路排名 → 一份统一排名。

RRF（Reciprocal Rank Fusion，Cormack et al. 2009）：score(d) = Σ_i w_i / (k + rank_i(d))，k = 60。
不看分数只看排名——余弦相似度在 [0.3, 0.9]，BM25 无上界，两者直接相加会被 BM25 淹没；
归一化又受异常值和语料分布影响，换一批数据就要重调。RRF 五行代码、零参数、生产最稳。

凸组合（convex）：min-max 归一化后 α·dense + (1-α)·sparse。arXiv 2604.01733 在金融数据上等权 α=0.5
略优于 RRF（R@5 0.726 vs 0.695），值得在有评估集时 A/B。经验值：中文技术文档 α=0.6~0.7（向量为主），
FAQ / 合同 / 法规 α=0.4~0.5（BM25 权重更高）。
"""

from __future__ import annotations

from collections.abc import Sequence

Ranking = Sequence[tuple[str, float]]  # [(id, score)]，已按分数降序


def rrf_fusion(rankings: list[tuple[Ranking, float]], k: int = 60) -> list[tuple[str, float]]:
    """rankings: [(ranking, weight), ...]。返回按 RRF 分数降序的 (id, score)。"""
    fused: dict[str, float] = {}
    for ranking, weight in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def convex_fusion(dense: Ranking, sparse: Ranking, alpha: float = 0.5) -> list[tuple[str, float]]:
    d = _minmax(dense)
    s = _minmax(sparse)
    ids = set(d) | set(s)
    fused = {i: alpha * d.get(i, 0.0) + (1 - alpha) * s.get(i, 0.0) for i in ids}
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def merge_same_route(rankings: list[Ranking]) -> list[tuple[str, float]]:
    """同一路、多条查询（Multi-Query）的结果按最大分合并，用于 convex 融合前的路内合并。"""
    best: dict[str, float] = {}
    for r in rankings:
        for doc_id, score in r:
            if score > best.get(doc_id, float("-inf")):
                best[doc_id] = score
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def _minmax(ranking: Ranking) -> dict[str, float]:
    if not ranking:
        return {}
    scores = [s for _, s in ranking]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return {i: 1.0 for i, _ in ranking}
    return {i: (s - lo) / (hi - lo) for i, s in ranking}
