"""回答的事实依据：过期 / 无权文档不能从旁路（实体织入、回答附图）混进来；问历史规定时放行并标注失效；
织入块要与问题相关；引用 / 数字校验不误报。"""

from __future__ import annotations

from datetime import date

from prism.generation.citecheck import check_citations, check_numbers
from prism.generation.prompts import build_answer_prompt, format_context, validity_status
from prism.graph.unweaver import EntityRelation, EntityStore, weave_contexts
from prism.retrieval import HybridRetriever
from prism.retrieval.rerank import BaseReranker, NoopReranker
from prism.retrieval.temporal import history_intent
from prism.schemas import Chunk, Context, DocMetadata


def _parent(index, title: str, needle: str) -> Chunk:
    doc_id = next(d for d, md in index.store.docs.items() if md.title == title)
    return next(c for c in index.store.chunks.values() if c.kind == "parent" and c.doc_id == doc_id and needle in c.text)


class KeywordReranker(BaseReranker):
    """"住宿标准"段落最高、其余含"住宿"的高分、无关的低分：用来验证织入门槛。"""

    name = "cross_encoder"

    def rerank(self, query, candidates, top_k):
        scored = [(cid, 0.95 if "住宿标准为" in text else 0.9 if "住宿" in text else 0.01) for cid, text in candidates]
        return sorted(scored, key=lambda x: -x[1])[:top_k]


# ------------------------------------------------------------------------------ 时效
def test_history_intent_detection():
    expired = DocMetadata(title="员工手册（2024 版）", version="2024.2", valid_from=date(2024, 1, 1), valid_to=date(2025, 12, 31))
    current = DocMetadata(title="员工手册（2026 版）", version="2026.1", valid_from=date(2026, 1, 1))
    docs = [expired, current]
    today = date(2026, 10, 1)
    assert history_intent("2024 版员工手册的住宿标准是多少", docs, today) == "2024"
    assert history_intent("2025 年的年假能跨年吗", docs, today) == "2025"           # 落在旧版的生效区间里
    assert history_intent("旧版规定年假要提前几天", docs, today)
    assert history_intent("去年没休完的年假今年还能休吗", docs, today)
    assert history_intent("CVE-2024-3094 怎么修复", docs, today) is None              # 编号里的年份不算
    assert history_intent("出差住宿标准能不能超额", docs, today) is None
    assert history_intent("2026 版的住宿标准", docs, today) is None                   # 现行版本的年份不算


def test_validity_status_and_prompt_date():
    assert validity_status("2024-01-01", "2025-12-31", date(2026, 10, 1)) == "已失效（2025-12-31 废止）"
    assert validity_status("2027-01-01", None, date(2026, 10, 1)).startswith("尚未生效")
    assert validity_status("2026-01-01", None, date(2026, 10, 1)) == "现行有效"
    assert validity_status(None, None) is None
    chunk = Chunk(chunk_id="c1", doc_id="d", kind="parent", text="一线城市每晚 500 元", metadata={"version": "2024.2", "valid_from": "2024-01-01", "valid_to": "2025-12-31"})
    ctx = Context(chunk=chunk, score=1.0)
    assert "状态：已失效（2025-12-31 废止）" in format_context(ctx, date(2026, 10, 1))
    prompt = build_answer_prompt("住宿标准", [ctx], today=date(2026, 10, 1))
    assert prompt.startswith("<当前日期>2026-10-01</当前日期>") and "状态：已失效" in prompt


def test_expired_docs_only_for_history_questions(built_index):
    r = HybridRetriever(built_index.settings, built_index, llm=None, reranker=NoopReranker())
    today = date(2026, 10, 1)
    current = r.retrieve("出差住宿标准", top_k=5, on_date=today)
    assert "2024.2" not in {c.chunk.metadata.get("version") for c in current.contexts}
    history = r.retrieve("2024 版员工手册的出差住宿标准", top_k=5, on_date=today)
    assert "2024.2" in {c.chunk.metadata.get("version") for c in history.contexts}
    assert any(n.startswith("时效") for n in history.trace.notes)
    # 旁路（回答附图）用的文档可见性与之一致
    expired_id = next(d for d, md in built_index.store.docs.items() if md.version == "2024.2")
    visible = r.visible_doc_ids("出差住宿标准", on_date=today)
    assert visible is not None and expired_id not in visible and len(visible) == len(built_index.store.docs) - 1
    assert r.visible_doc_ids("2024 版的住宿标准", on_date=today) is None
    # 历史意图开关关掉：问旧版也照样过滤
    off = r.retrieve("2024 版员工手册的出差住宿标准", top_k=5, on_date=today, overrides={"history_intent": False})
    assert "2024.2" not in {c.chunk.metadata.get("version") for c in off.contexts}


# ------------------------------------------------------------------------------ 实体织入
def test_weave_respects_allow_filter(built_index):
    current = _parent(built_index, "员工手册（2026 版）", "住宿标准为一线城市")
    old = _parent(built_index, "员工手册（2024 版）", "住宿标准")
    store = EntityStore()
    store.add([EntityRelation(entity="一线城市", definition="北上广深", chunk_id=old.chunk_id, doc_id=old.doc_id)])
    ctxs = [Context(chunk=current, score=0.9)]
    woven, rel = weave_contexts(ctxs, store, built_index.store.parent_of)
    assert any(c.chunk.chunk_id == old.chunk_id for c in woven)                 # 不过滤：旧版被织进来（原来的问题）
    woven, rel = weave_contexts(ctxs, store, built_index.store.parent_of, allow=lambda ch: ch.doc_id != old.doc_id)
    assert [c.chunk.chunk_id for c in woven] == [current.chunk_id] and rel == []


def test_retriever_weave_filters_expired_and_gates_relevance(built_index, monkeypatch):
    old = _parent(built_index, "员工手册（2024 版）", "住宿标准")
    flow = _parent(built_index, "员工手册（2026 版）", "住宿发票")          # 差旅报销流程：含"住宿"，精排高分
    k8s = _parent(built_index, "K8s 运维排障手册", "502")                   # 与住宿无关，精排低分
    store = EntityStore()
    store.add([
        EntityRelation(entity="一线城市", definition="北上广深", chunk_id=old.chunk_id, doc_id=old.doc_id),
        EntityRelation(entity="部门负责人", definition="审批人", chunk_id=k8s.chunk_id, doc_id=k8s.doc_id),
        EntityRelation(entity="北京", definition="一线城市", chunk_id=flow.chunk_id, doc_id=flow.doc_id),
    ])
    monkeypatch.setattr(built_index, "entities", store)
    r = HybridRetriever(built_index.settings, built_index, llm=None, reranker=KeywordReranker())
    over = {"rerank": {"enabled": True, "min_score": 0.1}, "graph": {"unweaver_enabled": True, "max_weaved": 3}}
    res = r.retrieve("出差住宿标准", top_k=1, use_transform=False, on_date=date(2026, 10, 1), overrides=over)
    woven = [c for c in res.contexts if c.weaved]
    assert old.doc_id not in {c.chunk.doc_id for c in res.contexts}           # 过期文档不会经织入混进来
    assert [c.chunk.chunk_id for c in woven] == [flow.chunk_id] and woven[0].score == 0.9
    assert any("织入门槛" in n for n in res.trace.notes)
    assert {w.entity for w in res.trace.weaved} == {"北京"}
    # 关掉门槛：无关的 K8s 段落会被织进来
    loose = r.retrieve("出差住宿标准", top_k=1, use_transform=False, on_date=date(2026, 10, 1), overrides={**over, "graph": {**over["graph"], "weave_relevance": False}})
    assert k8s.chunk_id in {c.chunk.chunk_id for c in loose.contexts if c.weaved}


# ------------------------------------------------------------------------------ 引用 / 数字校验
SRC = ["来源：员工手册 > 住宿标准\n版本：2026.1\n一线城市每晚 600 元，其他城市每晚 400 元，特殊情况可上浮 20%。"]


def test_citecheck_skips_missing_info_and_question_terms():
    assert check_citations("餐补标准：根据现有资料无法回答，资料中未包含餐补标准的信息 [1]。", SRC) == []
    q = "深圳出差住 550 元一晚的酒店能全额报销吗"
    assert check_citations("550 元未超过 600 元标准，因此可以全额报销 [1]。", SRC, question=q) == []
    # 真正串号的句子照样报
    assert check_citations("P95 延迟超过 500ms 持续 5 分钟要告警，错误率超过 1% 也要告警 [1]。", SRC)


def test_check_numbers():
    assert check_numbers("一线城市每晚 600 元 [1]，可上浮 20%。", SRC) == []
    issues = check_numbers("一线城市每晚 650 元 [1]。", SRC)
    assert [(i["reason"], i["value"], i["index"]) for i in issues] == [("number", "650", 1)]
    assert check_numbers("上浮后最多 600 × 1.2 = 720 元 [1]。", SRC) == []              # 写出算式的推导值
    assert check_numbers("可上浮 20%，即最多 720 元 [1]。", SRC) == []
    assert check_citations("计算过程：600 元 × (1 + 20%) = 600 × 1.2 = 720 元 [1]。", SRC) == []
    assert check_numbers("住 550 元的酒店没有超标 [1]。", SRC, question="住 550 元能报吗") == []
    assert check_numbers("共 3 个步骤：1. 申请 2. 审批 3. 打款", SRC) == []            # 计数与序号
    assert check_numbers("资料中没有餐补标准，无法确认是否为每天 100 元。", SRC) == []    # 说明缺失的句子


def test_images_for_answer_empty_doc_filter_returns_nothing():
    from prism.multimodal.retriever import MultimodalRetriever

    class Stub(MultimodalRetriever):
        def __init__(self):  # noqa: D107 - 只测过滤分支，不需要真实索引
            from prism.config import MultimodalConfig
            self.cfg = MultimodalConfig()

    hits, res = Stub().images_for_answer("各区域销售额", [], doc_ids=set())
    assert hits == [] and res.images == []
