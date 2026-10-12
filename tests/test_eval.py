from __future__ import annotations

from pathlib import Path

import pytest

from prism.eval import EvalRunner, hit_rate, load_golden_set, mrr, ndcg, recall_at_k
from prism.retrieval import HybridRetriever
from prism.retrieval.rerank import NoopReranker

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "eval" / "golden_qa.jsonl"


def test_metrics():
    rel = [False, True, False, True]
    assert hit_rate(rel, 1) == 0.0 and hit_rate(rel, 2) == 1.0
    assert recall_at_k(rel, 4, 2) == 1.0 and recall_at_k(rel, 2, 2) == 0.5
    assert mrr(rel) == 0.5
    assert 0 < ndcg(rel, 4, 2) < 1


def test_eval_runner_on_golden_set(built_index):
    examples = load_golden_set(GOLDEN)
    retriever = HybridRetriever(built_index.settings, built_index, llm=None, reranker=NoopReranker())
    report = EvalRunner(retriever, ks=[1, 3, 5, 10], ablation=True).run(examples)
    assert report.refusal_total == 1
    assert report.n == len(examples) - report.refusal_total
    assert "无答案" not in report.category_metrics
    assert set(report.stage_metrics) == {"dense", "sparse", "fused", "final"}
    # 哈希向量 + BM25 + 无重排：混合召回在这套小语料上应有较高命中
    assert report.stage_metrics["final"]["hit@5"] >= 0.8
    assert report.stage_metrics["fused"]["hit@10"] >= report.stage_metrics["dense"]["hit@10"] - 1e-9
    md = report.to_markdown()
    assert "分阶段消融" in md and "错误归因" in md


def test_coverage_buckets_intents_and_flags_docs(built_index):
    """语料准备：有答案的意图落在高 / 部分匹配，库里没有的落在无匹配；过期文档被标记。"""
    from prism.eval import CoverageAnalyzer

    retriever = HybridRetriever(built_index.settings, built_index, llm=None, reranker=NoopReranker())
    intents = [
        {"question": "订单 SO-8817 的状态", "category": "精确编号"},
        {"question": "K8s Ingress 502 怎么排查", "category": "精确编号"},
        "年假可以跨年使用吗",
        "公司食堂几点开饭",
        "笔记本电脑保修多久",
        "年假可以跨年使用吗",  # 重复意图只算一次
    ]
    report = CoverageAnalyzer(retriever).run(intents)
    by_intent = {r.intent: r for r in report.intents}
    assert report.n == 5 and sum(report.buckets.values()) == 5
    assert by_intent["订单 SO-8817 的状态"].bucket == "high"
    assert by_intent["K8s Ingress 502 怎么排查"].bucket == "high"
    assert by_intent["年假可以跨年使用吗"].bucket == "high"
    assert by_intent["公司食堂几点开饭"].bucket == "none" and "食堂" in by_intent["公司食堂几点开饭"].missing_terms
    assert by_intent["笔记本电脑保修多久"].bucket == "none"
    assert report.by_category["精确编号"]["high"] == 2
    assert 0 < report.coverage_rate <= report.answerable_rate < 1

    docs = {d.title: d for d in report.docs}
    assert docs["员工手册（2024 版）"].status == "expired" and docs["员工手册（2024 版）"].hits == 0
    assert docs["产品价格表"].status == "unused"          # 这批意图没有问到价格 → 提示确认是否无关
    assert docs["售后服务政策"].hits >= 1
    md = report.to_markdown()
    assert "意图空间" in md and "知识空间" in md and any("无匹配" in x for x in report.recommendations)

    with pytest.raises(ValueError):
        CoverageAnalyzer(retriever, high=0.3, partial=0.5)


def test_section_and_its_paragraph_are_not_double_counted():
    """多粒度索引：最终上下文里同时出现"章节"和其中的相关段落，只算一次相关——NDCG / Recall 不能超过 1。"""
    from types import SimpleNamespace

    from prism.eval.dataset import GoldenExample
    from prism.eval.runner import EvalRunner
    from prism.index.doc_store import ChunkStore
    from prism.schemas import Chunk

    store = ChunkStore()
    p1 = Chunk(chunk_id="d:p1", doc_id="d", kind="parent", text="年假可以跨年使用，最晚次年 3 月 31 日")
    p2 = Chunk(chunk_id="d:p2", doc_id="d", kind="parent", text="病假需提供诊断证明")
    c1 = Chunk(chunk_id="d:p1:c0", doc_id="d", kind="child", text=p1.text, parent_id="d:p1")
    sec = Chunk(chunk_id="d:s000", doc_id="d", kind="section", text=p1.text + "\n" + p2.text, child_ids=["d:p1", "d:p2"])
    store.add_chunks([p1, p2, c1, sec])
    runner = EvalRunner.__new__(EvalRunner)
    runner.store = store
    ex = GoldenExample(question="年假能跨年吗", answer_keywords=["跨年"])

    runner.ks = [1, 3, 5]
    rel = runner._relevance(ex, ["d:s000", "d:p1", "d:p2", "d:p1:c0"])
    assert rel == [True]                           # 章节覆盖 p1（相关）+ p2；之后的 p1 / p2 / c0 都已覆盖，跳过
    assert runner._relevance(ex, ["d:p2", "d:s000"]) == [False, True]   # 章节里只有 p1 是新的且相关
    m = runner._metrics(rel, runner._count_relevant_parents(ex))
    assert m["ndcg@5"] == 1.0 and m["recall@5"] == 1.0
