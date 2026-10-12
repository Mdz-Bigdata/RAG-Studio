from __future__ import annotations

import pytest

from prism.retrieval.fusion import convex_fusion, rrf_fusion


def test_rrf_article_example():
    # 「苹果14 售价」：向量路 [历史, 参数页, 比较...]，BM25 路 [参数页, 比较, 历史]
    dense = [("history", 0.9), ("spec", 0.8), ("x", 0.7), ("compare", 0.6)]
    sparse = [("spec", 12.0), ("compare", 8.0), ("history", 5.0)]
    fused = rrf_fusion([(dense, 1.0), (sparse, 1.0)], k=60)
    scores = dict(fused)
    assert scores["spec"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["history"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["compare"] == pytest.approx(1 / 64 + 1 / 62)
    assert fused[0][0] in ("spec", "history") and fused[2][0] == "compare"


def test_rrf_weights_shift_ranking():
    dense = [("a", 1.0), ("b", 0.9)]
    sparse = [("b", 3.0), ("a", 1.0)]
    assert rrf_fusion([(dense, 1.0), (sparse, 1.0)])[0][0] in ("a", "b")
    assert rrf_fusion([(dense, 0.5), (sparse, 2.0)])[0][0] == "b"


def test_convex_fusion_minmax():
    dense = [("a", 0.9), ("b", 0.5)]
    sparse = [("b", 20.0), ("c", 5.0)]
    fused = dict(convex_fusion(dense, sparse, alpha=0.5))
    assert fused["a"] == pytest.approx(0.5)
    assert fused["b"] == pytest.approx(0.5)
    assert fused["c"] == pytest.approx(0.0)
