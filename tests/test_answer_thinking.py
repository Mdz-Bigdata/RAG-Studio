"""回答前思考（generation.llm.thinking）：普通问答不让推理模型思考（首字快 2.5~4.5 秒），多跳 / Agentic 才思考。"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from prism.config import GenerationConfig, LLMConfig
from prism.generation import AnswerGenerator, FakeLLM
from prism.schemas import Chunk, Context, RetrievalResult, RetrievalTrace


def _result(strategy: str) -> RetrievalResult:
    ctx = Context(chunk=Chunk(chunk_id="c1", doc_id="d1", kind="parent", text="出差住宿一线城市每晚 600 元。"), score=0.9)
    return RetrievalResult(query="出差能住多贵的酒店", contexts=[ctx], trace=RetrievalTrace(original_query="出差能住多贵的酒店", strategy=strategy))


@pytest.mark.parametrize(("mode", "strategy", "expect"), [
    ("auto", "hybrid", False), ("auto", "precise", False), ("auto", "multihop", True), ("auto", "multihop+image", True), ("auto", "agentic:hybrid", True),
    ("off", "multihop", False), ("on", "hybrid", True),
])
def test_answer_passes_thinking_decision(mode, strategy, expect):
    llm = FakeLLM(lambda p: "一线城市每晚 600 元 [1]。")
    gen = AnswerGenerator(GenerationConfig(llm=LLMConfig(thinking=mode)), llm)
    assert gen.answer(_result(strategy)).text
    assert "".join(gen.stream(_result(strategy)))
    assert llm.thinking == [expect, expect]


def test_openai_compatible_per_call_thinking_overrides_instance_default():
    from prism.generation.llm import OpenAICompatibleLLM

    sent: list[dict | None] = []

    def create(**kw):
        sent.append(kw.get("extra_body"))
        return NS(model="m", choices=[NS(finish_reason="stop", message=NS(content="ok", reasoning_content=None, model_extra={}))], usage=None)

    def llm(aux: bool) -> OpenAICompatibleLLM:
        x = OpenAICompatibleLLM(LLMConfig(provider="openai_compatible", model="deepseek-v4-pro"), api_key="x", base_url="http://127.0.0.1:9/v1", aux=aux)
        x.client = NS(chat=NS(completions=NS(create=create)))
        return x

    off = {"thinking": {"type": "disabled"}}
    main, aux = llm(aux=False), llm(aux=True)
    main.chat([{"role": "user", "content": "q"}])                     # 回答模型默认：不动（模型自己决定）
    main.chat([{"role": "user", "content": "q"}], thinking=False)     # 本次关闭
    aux.chat([{"role": "user", "content": "q"}])                      # 辅助模型默认关闭
    aux.chat([{"role": "user", "content": "q"}], thinking=True)       # 显式允许时不再关
    assert sent == [None, off, off, None]


@pytest.mark.parametrize(("query", "notes", "expect"), [
    ("去年没休完的年假今年还能休吗", [], True),          # 相对时间：要按当前日期推算
    ("今年 1 月攒的调休现在还能用吗", [], True),
    ("2025 年 12 月出差住宿标准是多少", [], True),       # 具体年月：要判断当时生效的版本
    ("员工手册里旧版的住宿标准", ["时效：问题在问历史规定（“旧版”）……"], True),
    ("调休的有效期是多久", [], False),                   # 只问期限多长：查得到就行，不必思考
    ("出差住宿标准能不能超额", [], False),
])
def test_auto_thinks_for_date_reasoning(query, notes, expect):
    llm = FakeLLM(lambda p: "ok [1]")
    gen = AnswerGenerator(GenerationConfig(llm=LLMConfig(thinking="auto")), llm)
    res = _result("hybrid")
    res.query = query
    res.trace.notes = list(notes)
    gen.answer(res)
    assert llm.thinking == [expect]
