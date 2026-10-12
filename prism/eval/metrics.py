"""检索指标：Hit Rate@K（有没有找到）、Recall@K、MRR（排第几）、NDCG@K（排序质量）。"""

from __future__ import annotations

import math


def hit_rate(ranked_relevance: list[bool], k: int) -> float:
    return 1.0 if any(ranked_relevance[:k]) else 0.0


def recall_at_k(ranked_relevance: list[bool], k: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return min(1.0, sum(ranked_relevance[:k]) / total_relevant)


def mrr(ranked_relevance: list[bool]) -> float:
    for i, rel in enumerate(ranked_relevance, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def ndcg(ranked_relevance: list[bool], k: int, total_relevant: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, rel in enumerate(ranked_relevance[:k], start=1) if rel)
    ideal_n = min(k, max(total_relevant, 1))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0
