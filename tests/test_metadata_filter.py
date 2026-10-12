"""元数据过滤：属性取值目录、匹配语义、从问题里识别条件、检索前预过滤与退回。"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from prism.generation.llm import FakeLLM
from prism.index.vector_store import widen_until_filled
from prism.retrieval import HybridRetriever
from prism.retrieval.metadata_filter import doc_years, extract_llm, extract_rule, facets, matches
from prism.retrieval.multihop import MultiHopRetriever
from prism.schemas import DocMetadata, RetrievalResult, RetrievalTrace

TODAY = date(2026, 10, 11)
DOCS = [
    DocMetadata(title="员工手册（2024 版）", version="2024.2", valid_from=date(2024, 1, 1), valid_to=date(2025, 12, 31),
                department="人力资源部", tags=["制度", "假期"], file_format="md"),
    DocMetadata(title="员工手册（2026 版）", version="2026.1", valid_from=date(2026, 1, 1), department="人力资源部",
                tags=["制度", "假期"], file_format="md"),
    DocMetadata(title="云平台扩容指南", department="基础架构部", tags=["云计算", "Kubernetes"], file_format="PDF", extra={"year": 2023}),
    DocMetadata(title="产品价格表", file_format="csv"),
]


def test_doc_years_cover_title_version_validity_and_open_ended_documents():
    assert doc_years(DOCS[0], TODAY) == {2024, 2025}
    assert doc_years(DOCS[1], date(2028, 3, 1)) == {2026, 2027, 2028}   # 未写失效日期：到今年都算
    assert doc_years(DOCS[2], TODAY) == {2023}
    assert doc_years(DocMetadata(title="CVE-2024-3094 处置记录"), TODAY) == set()   # 编号里的数字不是年份
    assert doc_years(DOCS[3], TODAY) == set()


def test_facets_count_documents_and_merge_case_variants():
    available = facets([*DOCS, DocMetadata(tags=["kubernetes", "kubernetes"], file_format="pdf")], TODAY)
    assert available["department"] == {"人力资源部": 2, "基础架构部": 1}
    assert available["tags"]["Kubernetes"] == 2 and "kubernetes" not in available["tags"]
    assert available["file_format"] == {"md": 2, "pdf": 2, "csv": 1}
    assert list(available["tags"])[:2] == ["Kubernetes", "假期"]   # 文档数多的在前，同数按名称


def test_match_is_and_across_fields_or_within_field_and_missing_is_configurable():
    cloud = DOCS[2]
    assert matches(cloud, {"tags": ["云计算", "售后"], "year": [2023]}, include_missing=False, today=TODAY)
    assert not matches(cloud, {"tags": ["云计算"], "year": [2024]}, include_missing=False, today=TODAY)
    assert matches(cloud, {"tags": ["KUBERNETES"], "file_format": ["pdf"]}, include_missing=False, today=TODAY)
    prices = DOCS[3]   # 没有部门、标签、年份
    assert matches(prices, {"department": ["人力资源部"], "year": [2026]}, include_missing=True, today=TODAY)
    assert not matches(prices, {"department": ["人力资源部"]}, include_missing=False, today=TODAY)
    assert matches(prices, {"author": ["张三"]}, include_missing=False, today=TODAY)   # 不认识的字段不参与


def test_rule_extraction_only_uses_values_that_exist_in_this_kb():
    available = facets(DOCS, TODAY)
    fields = ["department", "tags", "year", "file_format"]
    assert extract_rule("2023 年关于云计算的进展", available, fields) == {"year": [2023]}   # 标签要靠模型判断
    assert extract_rule("基础架构部发布的扩容指南", available, fields) == {"department": ["基础架构部"]}
    assert extract_rule("人力资源部的制度里年假几天", available, fields) == {"department": ["人力资源部"]}
    # 只是提到部门，不是在限定文档来源
    assert extract_rule("连续病假超过几天需要报人力资源部备案", available, fields) == {}
    assert extract_rule("人力资源部的报销流程怎么走", available, fields) == {}
    assert extract_rule("2019 年的制度和 CVE-2024-3094", available, fields) == {}
    assert extract_rule("2024 年人力资源部的规定", available, ["tags"]) == {}


def test_llm_extraction_is_constrained_to_schema_and_drops_invented_values():
    available = facets(DOCS, TODAY)
    answer = '```json\n{"tags": ["云计算", "区块链"], "year": ["2023", 2031], "department": "基础架构部", "author": ["张三"], "file_format": [true]}\n```'
    llm = FakeLLM(lambda prompt: answer)
    history = [{"role": "user", "content": "云平台去年扩容了几次"}, {"role": "assistant", "content": "三次"}]
    got = extract_llm("那 2023 年的呢", available, ["department", "tags", "year", "file_format"], llm, history)
    assert got == {"department": ["基础架构部"], "tags": ["云计算"], "year": [2023]}
    prompt = llm.calls[0]
    assert "云计算" in prompt and "人力资源部" in prompt and "2023" in prompt and "云平台去年扩容了几次" in prompt
    assert "三次" not in prompt   # 只带用户的上文
    assert extract_llm("随便问问", available, ["tags"], FakeLLM(lambda prompt: "{}")) == {}
    with pytest.raises(ValueError):
        extract_llm("随便问问", available, ["tags"], FakeLLM(lambda prompt: "没有条件"))


# --- 接入检索器 ---------------------------------------------------------------------------
def _titles(index, result) -> set[str]:
    return {index.store.docs[c.chunk.doc_id].title for c in result.contexts}


def _self_query(answer: str) -> FakeLLM:
    return FakeLLM(lambda prompt: answer if "自查询" in prompt else prompt)


def test_year_in_question_narrows_candidates_before_recall(built_index):
    retriever = HybridRetriever(built_index.settings, built_index)
    result = retriever.retrieve("2024 版的住宿标准是多少", use_transform=False)
    info = result.trace.metadata_filter
    assert info["applied"] and info["source"] == "rule" and info["conditions"] == {"year": [2024]}
    assert info["docs_after"] < info["docs_before"]
    assert "员工手册（2024 版）" in _titles(built_index, result) and "员工手册（2026 版）" not in _titles(built_index, result)
    assert any("年份=2024" in note for note in result.trace.notes)
    # 预过滤：各路召回里都不会出现范围外的块
    seen = {built_index.store.docs[built_index.store.chunks[s.chunk_id].doc_id].title for items in result.trace.routes.values() for s in items}
    assert "员工手册（2026 版）" not in seen


def test_no_condition_or_no_narrowing_leaves_retrieval_untouched(built_index):
    retriever = HybridRetriever(built_index.settings, built_index)
    plain = retriever.retrieve("年假可以跨年使用吗", use_transform=False)
    assert plain.trace.metadata_filter == {} and not any("元数据" in n for n in plain.trace.notes)
    # 过期的 2024 版已被时效过滤排除，剩下带年份的文档都是 2026：条件不缩小范围，不提示也不生效
    same = retriever.retrieve("2026 年的年假规定", use_transform=False)
    assert same.trace.metadata_filter["reason"] == "no_narrowing" and not same.trace.metadata_filter["applied"]
    assert not any("元数据" in n for n in same.trace.notes)
    off = retriever.retrieve("2024 版的住宿标准是多少", use_transform=False, overrides={"metadata_filter": {"mode": "off"}})
    assert off.trace.metadata_filter == {}
    assert retriever.retrieve("2024 版的住宿标准是多少", use_transform=False, auto_filter=False).trace.metadata_filter == {}


def test_llm_conditions_filter_and_missing_attribute_policy(built_index):
    retriever = HybridRetriever(built_index.settings, built_index, llm=_self_query('{"tags": ["售后"], "year": [2031]}'))
    kept = retriever.retrieve("售后方面的规定", use_transform=False)
    assert kept.trace.metadata_filter["source"] == "llm" and kept.trace.metadata_filter["conditions"] == {"tags": ["售后"]}
    # 默认保留没打标签的文档（价格表），排除打了其他标签的
    assert kept.trace.metadata_filter["docs_after"] == 2
    assert _titles(built_index, kept) <= {"售后服务政策", "产品价格表"}
    strict = retriever.retrieve("售后方面的规定", use_transform=False, overrides={"metadata_filter": {"include_missing": False}})
    assert strict.trace.metadata_filter["docs_after"] == 1 and _titles(built_index, strict) == {"售后服务政策"}


def test_conditions_that_match_nothing_are_ignored(built_index):
    retriever = HybridRetriever(built_index.settings, built_index, llm=_self_query('{"tags": ["售后"], "department": ["基础架构部"]}'))
    result = retriever.retrieve("Ingress 502 怎么排查", use_transform=False, overrides={"metadata_filter": {"include_missing": False}})
    assert result.trace.metadata_filter["reason"] == "no_match" and not result.trace.metadata_filter["applied"]
    assert "K8s 运维排障手册" in _titles(built_index, result)
    assert any("没有匹配的文档" in n for n in result.trace.notes)


def test_wrong_guess_falls_back_instead_of_reporting_no_answer(built_index):
    class AbsoluteReranker:
        """分数不随候选集归一化：范围缩错时，候选里没有一条过得了阈值。"""

        name = "stub"

        def rerank(self, query, candidates, top_k):
            return sorted(((cid, 1.0 if "Ingress" in text else 0.0) for cid, text in candidates), key=lambda x: -x[1])[:top_k]

    settings = built_index.settings.model_copy(deep=True)
    settings.retrieval.rerank = settings.retrieval.rerank.model_copy(update={"enabled": True, "min_score": 0.5, "min_score_agreement": False})
    retriever = HybridRetriever(settings, built_index, llm=_self_query('{"tags": ["售后"]}'), reranker=AbsoluteReranker())
    strict = {"metadata_filter": {"include_missing": False}}
    result = retriever.retrieve("K8s Ingress 502 怎么排查", use_transform=False, overrides=strict)
    assert result.trace.metadata_filter["reason"] == "no_results" and not result.trace.metadata_filter["applied"]
    assert "K8s 运维排障手册" in _titles(built_index, result)
    assert "去掉该条件重新检索" in result.trace.notes[0]
    stay = retriever.retrieve("K8s Ingress 502 怎么排查", use_transform=False, overrides={"metadata_filter": {"include_missing": False, "fallback": False}})
    assert not stay.contexts and stay.trace.metadata_filter["applied"]


def test_llm_failure_degrades_to_rules_with_a_note(built_index):
    def broken(prompt: str) -> str:
        if "自查询" in prompt:
            raise RuntimeError("upstream 500")
        return prompt

    retriever = HybridRetriever(built_index.settings, built_index, llm=FakeLLM(broken))
    result = retriever.retrieve("2024 版的住宿标准是多少", use_transform=False)
    assert result.trace.metadata_filter["source"] == "rule" and result.trace.metadata_filter["applied"]
    assert any("模型识别条件失败" in n for n in result.trace.notes)


def test_explicit_filters_are_literal_and_combine_with_recognised_ones(built_index):
    retriever = HybridRetriever(built_index.settings, built_index)
    assert _titles(built_index, retriever.retrieve("规定", filters={"year": [2026]}, use_transform=False)) == {"员工手册（2026 版）", "售后服务政策"}
    assert _titles(built_index, retriever.retrieve("价格", filters={"file_format": ["CSV"]}, use_transform=False)) == {"产品价格表"}
    assert _titles(built_index, retriever.retrieve("规定", filters={"tags": ["退货"]}, use_transform=False)) == {"售后服务政策"}
    assert not retriever.retrieve("规定", filters={"year": [1999]}, use_transform=False).contexts
    options = retriever.facets()
    assert options["docs"] == 4 and {"value": "基础架构部", "count": 1} in options["department"]   # 过期的 2024 版不计入
    assert {o["value"] for o in options["year"]} == {2026}


def test_followup_hops_reuse_first_hop_conditions_without_asking_again():
    calls = []

    class Stub:
        def retrieve(self, query, **kwargs):
            calls.append(kwargs.get("auto_filter"))
            trace = RetrievalTrace(original_query=query)
            if len(calls) == 1:
                trace.metadata_filter = {"conditions": {"year": [2023]}, "applied": True}
            return RetrievalResult(query=query, contexts=[], trace=trace)

    MultiHopRetriever(Stub(), llm=None).retrieve("2023 年远程员工的通讯补贴和本地员工的交通补贴有什么不同", top_k=3)
    assert calls[0] is None and calls[1:] and all(c == {"year": [2023]} for c in calls[1:])
    calls.clear()
    Stub.retrieve = lambda self, query, **kwargs: (calls.append(kwargs.get("auto_filter")), RetrievalResult(query=query, contexts=[], trace=RetrievalTrace(original_query=query)))[1]
    MultiHopRetriever(Stub(), llm=None).retrieve("远程员工的通讯补贴和本地员工的交通补贴有什么不同", top_k=3)
    assert calls[1:] and all(c is False for c in calls[1:])


def test_facets_api_scopes_options_to_the_given_identity(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import retrieval as retrieval_api
    from app.db import get_db
    from app.services import retrieval_service

    seen = []
    monkeypatch.setattr(retrieval_api, "Repository", lambda db: type("Repo", (), {"get_kb": lambda self, kb_id: None if kb_id == "missing" else object()})())
    monkeypatch.setattr(retrieval_service, "facets", lambda kb_id, user=None: seen.append((kb_id, user)) or {"docs": 0})
    app = FastAPI()
    app.include_router(retrieval_api.router, prefix="/api/v1/retrieval")
    app.dependency_overrides[get_db] = lambda: object()
    with TestClient(app) as client:
        assert client.get("/api/v1/retrieval/demo/facets").json() == {"docs": 0}
        assert client.get("/api/v1/retrieval/demo/facets", params={"level": 1, "department": "财务部"}).status_code == 200
        assert client.get("/api/v1/retrieval/missing/facets").status_code == 404
    assert seen == [("demo", None), ("demo", {"level": 1, "department": "财务部"})]


# --- 向量库：过滤不能把库里有的结果挤掉 -----------------------------------------------------
def test_widening_fetches_more_until_enough_visible_hits():
    ranked = [(f"hidden{i}", 1 - i / 100) for i in range(40)] + [("visible", 0.1)]
    limits = []

    def fetch(limit):
        limits.append(limit)
        return ranked[:limit]

    assert widen_until_filled(fetch, 2, len(ranked), {"visible"}) == [("visible", 0.1)]
    assert limits == [8, 32, 41]
    limits.clear()
    assert widen_until_filled(fetch, 1, len(ranked), {"hidden0"}) == [("hidden0", 1.0)] and limits == [4]


_BACKEND_CHECK = """
import sys
import numpy as np
from prism.index.vector_store import ChromaVectorStore, QdrantVectorStore

# 查询方向是第 0 维：编号越小越相似，allowed 的块排在全库最后
ids = [f"c{i:03d}" for i in range(60)]
vectors = np.zeros((60, 4), dtype=np.float32)
vectors[:, 0] = 1.0
vectors[:, 1] = np.linspace(0.0, 2.0, 60)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
store = QdrantVectorStore("filter_test", 4) if sys.argv[1] == "qdrant" else ChromaVectorStore(sys.argv[2])
store.add(ids, vectors)
query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
assert [cid for cid, _ in store.search(query, 2)] == ["c000", "c001"]
# 固定取前 4 x top_n 再过滤，会一个都拿不到
assert [cid for cid, _ in store.search(query, 2, {"c057", "c058", "c059"})] == ["c057", "c058"]
assert store.search(query, 2, set()) == []
print("FILTER-OK", flush=True)
"""


@pytest.mark.parametrize("backend", ["qdrant", "chroma"])
def test_filtered_search_finds_allowed_chunks_ranked_far_below_top_k(backend, tmp_path):
    # 子进程里跑：这两个库的原生线程偶尔在解释器退出时崩溃，不能让它带走整个测试进程；只看断言是否走完
    done = subprocess.run([sys.executable, "-c", _BACKEND_CHECK, backend, str(tmp_path / "chroma")],
                          capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1], timeout=120)
    assert "FILTER-OK" in done.stdout, done.stderr[-2000:]
