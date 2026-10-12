from __future__ import annotations

from datetime import date

from prism.config import QueryTransformConfig
from prism.generation import AnswerGenerator, FakeLLM
from prism.retrieval import HybridRetriever, QueryTransformer, has_precise_tokens
from prism.retrieval.rerank import NoopReranker


def _retriever(index):
    return HybridRetriever(index.settings, index, llm=None, reranker=NoopReranker())


def test_index_stats(built_index):
    s = built_index.stats()
    assert s["docs"] == 5
    assert s["children"] == s["vectors"] == s["bm25_docs"] > 0


def test_precise_code_query_hits_via_bm25(built_index):
    r = _retriever(built_index)
    sparse = r.retrieve_sparse("K8s Ingress 502 怎么排查", top_n=5)
    top = built_index.store.get(sparse[0][0])
    assert "502" in top.text or "502" in top.breadcrumb_text
    result = r.retrieve("K8s Ingress 502 怎么排查", top_k=3)
    assert result.contexts
    assert any("Ingress 502" in c.chunk.breadcrumb_text for c in result.contexts)


def test_order_id_exact_match(built_index):
    r = _retriever(built_index)
    result = r.retrieve("订单 SO-8817 的状态", top_k=3)
    assert result.contexts
    assert any("SO-8817 因商品运输破损" in c.chunk.text for c in result.contexts)


def test_expired_document_filtered(built_index):
    r = _retriever(built_index)
    res = r.retrieve("出差住宿标准", top_k=5)
    versions = {c.chunk.metadata.get("version") for c in res.contexts}
    assert "2024.2" not in versions
    assert "2026.1" in versions
    # 回到 2025 年，则 2024 版生效、2026 版未生效；无生效区间的文档（K8s 手册 1.3）始终有效
    res_2025 = r.retrieve("出差住宿标准", top_k=5, on_date=date(2025, 6, 1))
    versions_2025 = {c.chunk.metadata.get("version") for c in res_2025.contexts}
    assert "2026.1" not in versions_2025
    assert "2024.2" in versions_2025


def test_parent_expansion_and_trace(built_index):
    r = _retriever(built_index)
    res = r.retrieve("年假可以跨年使用吗", top_k=3)
    assert all(c.chunk.kind == "parent" for c in res.contexts)
    assert any("次年 3 月 31 日" in c.chunk.text for c in res.contexts)
    tr = res.trace
    assert tr.queries == ["年假可以跨年使用吗"]
    assert any(k.startswith("dense") for k in tr.routes) and any(k.startswith("sparse") for k in tr.routes)
    assert tr.fused and tr.reranked
    assert "total" in tr.timings_ms


def test_metadata_filter(built_index):
    r = _retriever(built_index)
    res = r.retrieve("退货", top_k=5, filters={"department": "基础架构部"})
    assert all(c.chunk.metadata.get("department") == "基础架构部" for c in res.contexts)


def test_has_precise_tokens():
    assert has_precise_tokens("订单 SO-8817 的状态")
    assert has_precise_tokens("2024 年 Q3 财报净利润")
    assert has_precise_tokens("MySQL 1062 怎么办")
    assert not has_precise_tokens("怎么退货")


def test_query_transformer_with_fake_llm():
    def responder(prompt: str) -> str:
        if "改写成适合知识库检索" in prompt:
            return "退货流程"
        if "含义相同但角度" in prompt:
            return "1. 售后申请步骤\n2. 退换货规定\n3. 如何申请退款"
        if "参考答案" in prompt:
            return "用户可以在订单页申请售后，选择退货退款，客服审核后寄回商品。"
        return ""

    llm = FakeLLM(responder)
    t = QueryTransformer(QueryTransformConfig(multi_query=3, hyde=True, rewrite=True, decompose=True), llm)
    plan = t.plan("怎么退货")
    assert plan.queries[0] == "怎么退货"
    assert "退货流程" in plan.queries and "售后申请步骤" in plan.queries
    assert plan.hyde_document and "售后" in plan.hyde_document

    plan2 = t.plan("订单 SO-8817 的状态")
    assert plan2.hyde_document is None
    assert any("跳过" in n for n in plan2.notes)


def test_answer_generator_citations_and_refusal(built_index):
    r = _retriever(built_index)
    res = r.retrieve("年假可以跨年使用吗", top_k=3)
    llm = FakeLLM(lambda p: "可以跨年，但最晚次年 3 月 31 日前休完 [1]。")
    ans = AnswerGenerator(built_index.settings.generation, llm).answer(res)
    assert ans.citations == [1] and not ans.refused
    empty = res.model_copy(update={"contexts": []})
    ans2 = AnswerGenerator(built_index.settings.generation, llm).answer(empty)
    assert ans2.refused
    ans3 = AnswerGenerator(built_index.settings.generation, None).answer(res)
    assert ans3.model == "extractive" and "次年 3 月 31 日" in ans3.text


def test_min_score_ratio_drops_weak_contexts(built_index):
    """相对阈值（动态 top-K）：与最优候选差距过大的块不进入上下文。"""
    from prism.retrieval.rerank import LexicalReranker

    retriever = HybridRetriever(built_index.settings, built_index, llm=None, reranker=LexicalReranker())
    full = retriever.retrieve("订单 SO-8817 的状态", top_k=5)
    cut = retriever.retrieve("订单 SO-8817 的状态", top_k=5, overrides={"rerank": {"min_score_ratio": 0.6}})
    assert len(full.contexts) == 5
    assert 1 <= len(cut.contexts) < len(full.contexts)
    assert all("SO-8817" in c.chunk.text for c in cut.contexts)
    assert any("相对阈值" in n for n in cut.trace.notes)


# --- 混合检索第三路：SQL 结构化检索 -----------------------------------------------------------
def test_sql_route_rules(built_index):
    from prism.retrieval import HybridRetriever

    tables = built_index.tables
    assert len(tables) == 1
    t = next(iter(tables.tables.values()))
    assert "售价（元）" in t.columns and len(t.rows) == 5 and all(t.chunk_ids)

    # 数值比较：向量 / BM25 都表达不了"高于 9000"
    res = tables.search("售价高于 9000 的产品有哪些")
    # 多行可能落在同一个 child 块里：ranking 按块去重，行数看 note
    assert res.mode == "rule" and '"售价（元）" >= 9000' in res.sql and res.note.endswith("→ 2 行") and res.ranking
    texts = "\n".join(built_index.store.get(cid).text for cid, _ in res.ranking)
    assert "iPhone 15 Pro Max" in texts and "MacBook Pro 14" in texts
    # 排序取极值
    res = tables.search("库存最少的是哪个型号")
    assert "ORDER BY" in res.sql and "MacBook Pro 14" in built_index.store.get(res.ranking[0][0]).text
    # 单元格等值（最长匹配）
    res = tables.search("iPhone 15 Pro Max 用的什么芯片")
    assert res.note.endswith("→ 1 行") and "A17 Pro" in built_index.store.get(res.ranking[0][0]).text
    # 无关问题：这一路不出结果
    assert tables.search("年假可以跨年吗").ranking == []

    r = HybridRetriever(built_index.settings, built_index).retrieve("售价高于 9000 的产品有哪些", top_k=3)
    assert "sql:q0" in r.trace.routes and any(n.startswith("SQL 路") for n in r.trace.notes)
    joined = "\n".join(c.chunk.text for c in r.contexts)
    assert "iPhone 15 Pro Max" in joined and "MacBook Pro 14" in joined
    # 没有字段条件的问题：SQL 路不参与融合
    assert "sql:q0" not in HybridRetriever(built_index.settings, built_index).retrieve("年假可以跨年吗").trace.routes
    # 可按请求关闭 / 串行执行
    off = HybridRetriever(built_index.settings, built_index).retrieve("售价高于 9000 的产品有哪些", overrides={"sql": {"enabled": False}, "parallel_recall": False})
    assert "sql:q0" not in off.trace.routes


def test_sql_route_text_to_sql_is_read_only(built_index):
    import pytest

    from prism.generation import FakeLLM
    from prism.retrieval import HybridRetriever

    tables = built_index.tables
    name = next(iter(tables.tables))
    # LLM 生成 SELECT *：命中的行参与融合
    res = tables.search("哪些是 2024 年上市的", llm=FakeLLM(lambda p: f'```sql\nSELECT * FROM "{name}" WHERE "上市时间" LIKE \'2024%\'\n```'))
    assert res.mode == "llm" and res.note.endswith("→ 3 行") and res.ranking
    # 聚合：没有对应的行，结果表作为一条上下文交给 LLM
    llm = FakeLLM(lambda p: f'SELECT COUNT(*) AS 数量, AVG("售价（元）") AS 平均售价 FROM "{name}"' if "SQLite" in p else "x")
    res = tables.search("一共多少个产品，平均售价多少", llm=llm)
    assert res.rows == [{"数量": 5, "平均售价": pytest.approx(9999.0)}] and not res.ranking
    settings = built_index.settings.with_overrides({"retrieval": {"query_transform": {"rewrite": False, "multi_query": 0, "hyde": False, "decompose": False}}})
    r = HybridRetriever(settings, built_index, llm=llm).retrieve("一共多少个产品，平均售价多少", use_transform=False)
    assert r.contexts[0].chunk.breadcrumbs[-1] == "SQL 计算结果" and "9999" in r.contexts[0].chunk.text
    # 只读：写操作 / 多语句被拒绝 → 回退规则；表数据不变
    for bad in (f'DROP TABLE "{name}"', f'SELECT * FROM "{name}"; DELETE FROM "{name}"', f'DELETE FROM "{name}"'):
        assert tables.search("售价高于 9000", llm=FakeLLM(lambda p, bad=bad: bad)).mode == "rule"
    assert tables.search("售价高于 0 的产品").note.endswith("→ 5 行")
    # 与表无关：LLM 回答 NONE → 回退规则 → 无结果
    assert tables.search("年假可以跨年吗", llm=FakeLLM(lambda p: "NONE")).ranking == []


def test_sql_tables_persist_and_remove(tmp_path):
    from pathlib import Path

    from prism.config import Settings
    from prism.index import Index
    from prism.ingest import IngestPipeline

    settings = Settings.lite(index_dir=tmp_path / "idx")
    p = IngestPipeline(settings)
    p.ingest_paths([Path(__file__).resolve().parents[1] / "data" / "raw" / "产品价格表.csv"])
    loaded = Index.load(settings)
    assert len(loaded.tables) == 1 and loaded.tables.search("售价低于 6000 的产品").note.endswith("→ 1 行")
    loaded.remove_document(next(iter(loaded.store.docs)))
    assert len(loaded.tables) == 0 and loaded.stats()["tables"] == 0


def test_query_transform_calls_run_in_parallel():
    import time

    from prism.config import QueryTransformConfig
    from prism.generation import FakeLLM
    from prism.retrieval import QueryTransformer

    def slow(prompt: str) -> str:
        time.sleep(0.4)
        if "检索问句" in prompt:
            return "角度一\n角度二"
        return "" if "改写" in prompt else "假设答案"

    cfg = QueryTransformConfig(rewrite=True, multi_query=2, hyde=True, decompose=False, hyde_max_query_chars=50)
    t = time.perf_counter()
    plan = QueryTransformer(cfg, FakeLLM(slow)).plan("出差住宿怎么报销")
    elapsed = time.perf_counter() - t
    assert elapsed < 0.9, elapsed                      # 三个调用并发：约 0.4s 而不是 1.2s
    assert plan.queries == ["出差住宿怎么报销", "角度一", "角度二"] and plan.hyde_document == "假设答案"
    assert "rewrite: 模型没有返回内容（已忽略）" in plan.notes   # 空结果不再静默吞掉


def test_openai_compatible_aux_disables_thinking_and_retries_empty(monkeypatch):
    from types import SimpleNamespace as NS

    from prism.config import LLMConfig
    from prism.generation.llm import OpenAICompatibleLLM

    calls = []

    def resp(content, finish, reasoning):
        return NS(model="m", choices=[NS(finish_reason=finish, message=NS(content=content, reasoning_content="…" if reasoning else None, model_extra={}))],
                  usage=NS(prompt_tokens=1, completion_tokens=1, completion_tokens_details=NS(reasoning_tokens=5 if reasoning else 0)))

    class Completions:
        def __init__(self, script):
            self.script = script

        def create(self, **kw):
            calls.append(kw)
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    def llm(script, aux):
        x = OpenAICompatibleLLM(LLMConfig(provider="openai_compatible", model="m", base_url="http://x"), api_key="k", aux=aux)
        x.client = NS(chat=NS(completions=Completions(script)))
        return x

    # 辅助模型：请求里带 thinking=disabled
    assert llm([resp("改写结果", "stop", False)], aux=True).chat([{"role": "user", "content": "q"}], max_tokens=120).text == "改写结果"
    assert calls[-1]["extra_body"] == {"thinking": {"type": "disabled"}}
    # 端点不认识 thinking 参数：去掉后重试，之后不再发送
    err = Exception("400 Unrecognized request argument supplied: thinking")
    err.status_code = 400
    x = llm([err, resp("ok", "stop", False), resp("ok2", "stop", False)], aux=True)
    assert x.chat([{"role": "user", "content": "q"}]).text == "ok" and "extra_body" not in calls[-1]
    assert x.chat([{"role": "user", "content": "q"}]).text == "ok2" and calls[-1]["extra_body"] is None
    # 主模型保留思考；思考耗尽预算、回答为空 → 放宽预算重试一次
    calls.clear()
    assert llm([resp("", "length", True), resp("完整回答", "stop", True)], aux=False).chat([{"role": "user", "content": "q"}], max_tokens=100).text == "完整回答"
    assert [c["max_tokens"] for c in calls] == [100, 2148] and calls[0]["extra_body"] is None


def test_bm25_lazy_build_is_thread_safe():
    from concurrent.futures import ThreadPoolExecutor

    from prism.config import SparseConfig
    from prism.index.bm25_store import BM25Store

    store = BM25Store(SparseConfig())
    store.add([f"d{i}" for i in range(300)], [f"文档 {i} 关于报销 差旅 住宿 标准 第{i}条" for i in range(300)])
    # 16 个线程同时触发首次构建：都应该拿到完整索引的结果
    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(lambda _: store.search("住宿 标准", 5), range(16)))
    assert all(len(r) == 5 for r in results)


def test_precise_token_does_not_swallow_chinese():
    from prism.index.bm25_store import PRECISE_TOKEN

    found = lambda t: [m.group(0) for m in PRECISE_TOKEN.finditer(t)]  # noqa: E731
    assert found("2025年Q3各区域销售额是怎样的？") == ["2025", "Q3"]
    assert found("订单SO-8817的状态") == ["SO-8817"] and found("CVE-2024-3094 和 GPT-4o") == ["CVE-2024-3094", "GPT-4o"]


def test_min_score_agreement_rescues_only_cross_validated_answers(built_index):
    """精排把所有候选都打成极低分（大幅改写的问法常见）：向量与 BM25 第一名一致才保留该段，否则判无答案。"""
    from prism.retrieval.rerank import BaseReranker

    class LowScores(BaseReranker):
        name = "cross_encoder"

        def rerank(self, query, candidates, top_k):
            return [(cid, 0.001) for cid, _ in candidates[:top_k]]

    retriever = HybridRetriever(built_index.settings, built_index, llm=None, reranker=LowScores())
    strict = {"rerank": {"min_score": 0.1, "min_score_agreement": False}}
    rescue = {"rerank": {"min_score": 0.1, "min_score_agreement": True}}
    q = "CVE-2024-3094 xz liblzma 后门怎么修复"

    dense = retriever.retrieve_dense(q)[0][0]
    sparse = retriever.retrieve_sparse(q)[0][0]
    agree = built_index.store.parent_of(dense).chunk_id == built_index.store.parent_of(sparse).chunk_id
    assert retriever.retrieve(q, top_k=3, use_transform=False, overrides=strict).contexts == []      # 不交叉验证：直接拒答
    rescued = retriever.retrieve(q, top_k=3, use_transform=False, overrides=rescue)
    if agree:
        assert rescued.contexts and all(c.chunk.chunk_id == built_index.store.parent_of(dense).chunk_id for c in rescued.contexts)
        assert any("交叉验证通过" in n for n in rescued.trace.notes)
    else:                                                     # 离线哈希向量偶尔与 BM25 不一致：此时必须拒答
        assert rescued.contexts == []

    # 库里没有的内容：BM25 无命中 → 无从交叉验证 → 拒答
    none = retriever.retrieve("公司食堂几点开饭", top_k=3, use_transform=False, overrides=rescue)
    assert none.contexts == [] and any("判定为库中无答案" in n for n in none.trace.notes)
