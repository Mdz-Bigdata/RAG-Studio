"""带图提问 / 多跳的三处修正：引用校验（防串号）、历史去掉引用编号、多跳判断"缺什么"时看得到图片内容。"""

from __future__ import annotations

from prism.config import GenerationConfig
from prism.generation import AnswerGenerator, FakeLLM
from prism.generation.citecheck import check_citations, strip_citations
from prism.retrieval.multihop import MultiHopRetriever
from prism.schemas import Chunk, Context, RetrievalResult, RetrievalTrace

SOURCES = [
    "铁律五：单一职责优先。一个 Agent 只负责一种稳定能力，不要让一个 Agent 同时承担规划、执行、审核、记忆、调度。",
    "铁律三：Dry Run 默认开启。任何会改变生产环境的写操作，必须默认先执行 Dry Run，并等待人工确认后才能真实执行。",
    "铁律十：最小权限 + 沙箱 = 信任基础。必须做到最小权限原则、敏感工具隔离、高风险操作拦截、关键执行环境放入沙箱。",
]


def test_correct_citations_pass():
    answer = "写操作必须默认先执行 Dry Run，并等待人工确认后才能真实执行 [2]。一个 Agent 只负责一种稳定能力，不要同时承担规划与执行 [1]。"
    assert check_citations(answer, SOURCES) == []


def test_carried_over_citation_is_unsupported():
    # 上一轮的 [3] 是"铁律十八：评测先行"，本轮的 [3] 是铁律十：模型把上一轮的句子连同编号一起搬了过来
    answer = "按铁律十八，应先搭最小评测集、定义通过标准、跑基线，再调 Prompt、工具与流程 [3]。"
    issues = check_citations(answer, SOURCES)
    assert [(i["index"], i["reason"]) for i in issues] == [(3, "unsupported")]


def test_wrong_number_is_mismatch_with_suggestion():
    answer = "高风险操作要拦截，关键执行环境放入沙箱，遵循最小权限原则 [1]。"
    issues = check_citations(answer, SOURCES)
    assert issues and issues[0]["reason"] == "mismatch" and issues[0]["index"] == 1 and issues[0]["suggest"] == 3


def test_image_clauses_and_out_of_range():
    # 同时引用了图片的句子，内容可能来自图片本身：不校验
    assert check_citations("结账单注明仅作结账凭证、发票另开，因此还需另行取得发票 [1][用户图片]。", SOURCES) == []
    issues = check_citations("写操作必须默认先执行 Dry Run，并等待人工确认后才能执行 [7]。", SOURCES)
    assert [(i["index"], i["reason"]) for i in issues] == [(7, "out_of_range")]


def test_history_citations_are_stripped_before_generation():
    assert strip_citations("每晚 600 元 [1][图2]，见 [用户图片]。") == "每晚 600 元 ，见 。"
    ctx = Context(chunk=Chunk(chunk_id="c1", doc_id="d1", kind="parent", text="出差住宿一线城市每晚 600 元。"), score=0.9)
    result = RetrievalResult(query="那二线城市呢", contexts=[ctx], trace=RetrievalTrace(original_query="那二线城市呢", strategy="hybrid"))
    history = [{"role": "user", "content": "一线城市住宿标准 [1] 是多少"}, {"role": "assistant", "content": "每晚 600 元 [1][图1]。"}]
    msgs = AnswerGenerator(GenerationConfig(), FakeLLM()).build_messages(result, history=history)
    assert msgs[0]["content"] == "一线城市住宿标准 [1] 是多少"          # 用户原话不动
    assert msgs[1]["content"] == "每晚 600 元 。"                       # 上一轮回答的编号在本轮已失效


class _StubRetriever:
    """多跳的底层检索：每次返回同一段资料。"""

    def __init__(self) -> None:
        from types import SimpleNamespace as NS

        self.cfg = NS(rerank=NS(top_k=5))
        self.queries: list[str] = []

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        self.queries.append(query)
        ctx = Context(chunk=Chunk(chunk_id=f"c{len(self.queries)}", doc_id="d1", kind="parent", text=SOURCES[1]), score=0.5)
        return RetrievalResult(query=query, contexts=[ctx], trace=RetrievalTrace(original_query=query, strategy="hybrid"))


def test_multihop_followup_planning_sees_the_image():
    llm = FakeLLM(lambda p: "直接执行 os.system 部署生产违反哪条最小权限规范" if "os.system" in p else "DONE")
    base = _StubRetriever()
    res = MultiHopRetriever(base, llm).retrieve("这段代码违反了开发规范里的哪些铁律？", image_context="图中文字：os.system(\"kubectl apply -f prod.yaml\") 直接部署生产环境")
    assert "用户随问题上传了一张图片" in llm.calls[0] and "os.system" in llm.calls[0]
    assert base.queries[1].startswith("直接执行 os.system")                       # 第二跳针对图里的内容补查
    assert [h.query for h in res.trace.hops][1] == base.queries[1]
    # 没有图片时提示词里不出现图片段落
    llm2 = FakeLLM(lambda p: "DONE")
    MultiHopRetriever(_StubRetriever(), llm2).retrieve("年假能跨年吗")
    assert "上传了一张图片" not in llm2.calls[0]
