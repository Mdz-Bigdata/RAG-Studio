"""按需 Multi-Query 与反问澄清：检索把握不大时补跑扩展问法；判定无答案但有相近主题时反问用户，而不是只回一句拒答。"""

from __future__ import annotations

from prism.config import ClarifyConfig
from prism.generation import AnswerGenerator, FakeLLM
from prism.generation.prompts import REFUSAL_TEXT
from prism.retrieval import HybridRetriever
from prism.retrieval.rerank import BaseReranker, NoopReranker
from prism.schemas import ScoredChunk

Q = "年假可以跨年使用吗"
MQ_OFF = {"rewrite": False, "hyde": False, "decompose": False, "multi_query": 2,
          "multi_query_when": "low_confidence", "multi_query_below": 0.3}
BASE = {"metadata_filter": {"mode": "off"}, "query_transform": MQ_OFF}


class Scripted(BaseReranker):
    """第 n 次调用给所有候选打 scores[n] 分（超出列表就用最后一个）。"""

    name = "cross_encoder"

    def __init__(self, *scores: float) -> None:
        self.scores = list(scores)
        self.calls = 0

    def rerank(self, query, candidates, top_k):
        s = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return [(cid, s - i * 1e-4) for i, (cid, _) in enumerate(candidates[:top_k])]


def _mq_llm() -> FakeLLM:
    return FakeLLM(lambda p: "年休假 结转 规定\n未休年假 次年使用" if "检索问句" in p else "")


def _expand_calls(llm: FakeLLM) -> int:
    return sum(1 for m in llm.calls if "检索问句" in str(m))


def test_adaptive_multi_query_skips_expansion_when_confident(built_index):
    llm = _mq_llm()
    res = HybridRetriever(built_index.settings, built_index, llm=llm, reranker=Scripted(0.9)).retrieve(Q, top_k=3, overrides=BASE)
    assert res.trace.queries == [Q] and _expand_calls(llm) == 0
    assert not any("按需 Multi-Query" in n for n in res.trace.notes)


def test_adaptive_multi_query_retries_when_top_score_low(built_index):
    llm = _mq_llm()
    reranker = Scripted(0.05, 0.8)
    res = HybridRetriever(built_index.settings, built_index, llm=llm, reranker=reranker).retrieve(Q, top_k=3, overrides=BASE)
    assert _expand_calls(llm) == 1 and reranker.calls == 2
    assert res.trace.queries == [Q, "年休假 结转 规定", "未休年假 次年使用"]
    # 补跑的查询接着编号，不覆盖首轮的路；HyDE / SQL 这类整次检索只跑一次的路不重跑
    assert {"dense:q0", "sparse:q0", "dense:q1", "sparse:q1", "dense:q2", "sparse:q2"} <= set(res.trace.routes)
    assert res.trace.reranked[0].score > 0.7 and res.contexts
    assert any("0.050 → 0.800" in n for n in res.trace.notes)
    assert "multi_query_retry" in res.trace.timings_ms


def test_adaptive_multi_query_keeps_first_round_when_not_better(built_index):
    llm = _mq_llm()
    res = HybridRetriever(built_index.settings, built_index, llm=llm, reranker=Scripted(0.2, 0.1)).retrieve(Q, top_k=3, overrides=BASE)
    assert _expand_calls(llm) == 1
    assert abs(res.trace.reranked[0].score - 0.2) < 1e-3
    assert any("沿用首轮结果" in n for n in res.trace.notes)


def test_adaptive_multi_query_falls_back_to_always_without_reranker(built_index):
    llm = _mq_llm()
    res = HybridRetriever(built_index.settings, built_index, llm=llm, reranker=NoopReranker()).retrieve(Q, top_k=3, overrides=BASE)
    assert _expand_calls(llm) == 1 and len(res.trace.queries) == 3
    assert any("需要开启精排" in n for n in res.trace.notes)


NO_ANSWER = {**BASE, "query_transform": {**MQ_OFF, "multi_query": 0}, "rerank": {"min_score": 0.5, "min_score_agreement": False}}


def _no_answer(built_index, score: float = 0.2):
    return HybridRetriever(built_index.settings, built_index, llm=None, reranker=Scripted(score)).retrieve(Q, top_k=3, overrides=NO_ANSWER)


def test_near_misses_recorded_when_judged_no_answer(built_index):
    res = _no_answer(built_index)
    assert res.contexts == []
    near = res.trace.near_misses
    assert 1 <= len(near) <= 5 and len({m["label"] for m in near}) == len(near)
    assert all(m["label"] and m["snippet"] and m["score"] < 0.5 for m in near)
    # 分数远低于拒答阈值（< min_score × 0.2）：库外问题，候选只是凑数，不拿来反问
    assert _no_answer(built_index, 0.05).trace.near_misses == []


def test_clarification_replaces_fixed_refusal(built_index):
    res = _no_answer(built_index)
    ask = "资料里没有找到能直接回答的内容。你想问的是不是：\n- 年假\n请确认或补充。"
    llm = FakeLLM(lambda p: ask if "相近主题" in p else "x")
    gen = AnswerGenerator(built_index.settings.generation, llm)
    ans = gen.answer(res)
    assert ans.refused and ans.text == ask and ans.contexts == []
    assert llm.thinking == [False]
    assert "".join(gen.stream(res)) == ask


def test_clarification_falls_back_to_refusal(built_index):
    res = _no_answer(built_index)
    llm = FakeLLM(lambda p: "NONE")
    assert AnswerGenerator(built_index.settings.generation, llm).answer(res).text == REFUSAL_TEXT

    off = built_index.settings.generation.model_copy(update={"clarify": ClarifyConfig(enabled=False)})
    quiet = FakeLLM(lambda p: "不该被调用")
    assert AnswerGenerator(off, quiet).answer(res).text == REFUSAL_TEXT and quiet.calls == []
    # 没有相近主题（例如元数据过滤后无候选）：不调用模型，直接拒答
    bare = res.model_copy(update={"trace": res.trace.model_copy(update={"near_misses": []})})
    assert AnswerGenerator(built_index.settings.generation, quiet).answer(bare).text == REFUSAL_TEXT and quiet.calls == []


def test_low_confidence_note_in_prompt(built_index):
    res = HybridRetriever(built_index.settings, built_index, llm=None, reranker=NoopReranker()).retrieve(Q, top_k=3, use_transform=False)
    gen_cfg = built_index.settings.generation.model_copy(update={"clarify": ClarifyConfig(below=0.5)})

    def prompt_for(score: float) -> str:
        r = res.model_copy(update={"trace": res.trace.model_copy(update={"reranked": [ScoredChunk(chunk_id="x", score=score, rank=1, route="rerank")]})})
        return AnswerGenerator(gen_cfg, FakeLLM()).build_messages(r)[-1]["content"]

    assert "<检索提示>" in prompt_for(0.2)
    assert "<检索提示>" not in prompt_for(0.9)
    assert "<检索提示>" not in AnswerGenerator(built_index.settings.generation, FakeLLM()).build_messages(res)[-1]["content"]   # 默认关闭
