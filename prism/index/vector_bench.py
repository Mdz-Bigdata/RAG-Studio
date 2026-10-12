"""向量检索调优的效果对比：同一批查询分别走精确检索与当前调优配置，比较召回率与延迟。

查询取自库内正文（每个块开头一句），能反映真实分布；召回率 = 调优结果与精确 top-K 的重合比例。
"""

from __future__ import annotations

import random
import time

import numpy as np


def benchmark(index, *, n_queries: int = 30, top_k: int = 10, seed: int = 0) -> dict:
    store = index.vectors
    children = [c for c in index.store.chunks.values() if c.kind == "child"]
    if not children or not len(store):
        return {"ok": False, "message": "知识库里还没有向量，先上传文档"}
    rng = random.Random(seed)
    sample = rng.sample(children, min(n_queries, len(children)))
    texts = [c.text.split("。")[0][:60] or c.text[:60] for c in sample]
    qvecs = index.embedder.embed_queries(texts)

    def run(fn) -> tuple[list[list[str]], float]:
        t = time.perf_counter()
        out = [[cid for cid, _ in fn(q)] for q in qvecs]
        return out, (time.perf_counter() - t) * 1000 / len(qvecs)

    exact, t_exact = run(lambda q: store.search_exact(q, top_k))
    tuned, t_tuned = run(lambda q: store.search(q, top_k))
    recall = float(np.mean([len(set(a) & set(b)) / max(1, len(a)) for a, b in zip(exact, tuned)]))
    info = store.coarse_info() if hasattr(store, "coarse_info") else {}
    return {
        "ok": True, "backend": getattr(store, "name", "numpy"), "vectors": len(store), "dim": getattr(store, "dim", None),
        "queries": len(qvecs), "topK": top_k, "tuned": bool(getattr(store, "tuned", False)),
        "exactMs": round(t_exact, 3), "tunedMs": round(t_tuned, 3), "recall": round(recall, 4),
        "coarseDim": info.get("dim"), "coarseBytes": info.get("bytes"), "fullBytes": info.get("full_bytes"),
    }
