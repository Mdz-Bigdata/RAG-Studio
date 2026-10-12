"""回答级评测的确定性打分：关键事实、拒答 / 部分可答 / 错误前提、无依据数字。"""

from prism.eval.answer_eval import AnswerCase, fact_hits, looks_like_full_refusal, mentions_missing, parse_judge, score_case, summarize, ungrounded_numbers


def test_fact_hits_ignore_spacing_width_and_markdown():
    assert fact_hits("请在 **10 个工作日** 内提交 [1]", [["10个工作日"]]) == [True]
    assert fact_hits("售价１５９９９元", [["15999"]]) == [True]
    assert fact_hits("ON DUPLICATE KEY UPDATE 最常用", [["onduplicatekeyupdate"]]) == [True]
    assert fact_hits("只有 5 天", [["10天", "十天"]]) == [False]


def test_refusal_detection():
    assert looks_like_full_refusal("根据现有资料无法回答这个问题。")
    assert not looks_like_full_refusal("一线城市每晚 600 元 [1]。资料中没有餐补标准。")
    assert mentions_missing("一线城市每晚 600 元 [1]。资料中没有餐补标准。")


def test_score_answer_partial_refuse_and_premise():
    ans = AnswerCase(question="住宿标准", facts=[["600"], ["400"]])
    assert score_case(ans, "一线 600 元，其他 400 元 [1]", ["一线城市每晚 600 元，其他城市每晚 400 元"])["correct"]
    s = score_case(ans, "根据现有资料无法回答。", ["一线城市每晚 600 元"])
    assert not s["correct"] and s["falseRefusal"]

    partial = AnswerCase(question="住宿和餐补", expect="partial", facts=[["600"]])
    assert score_case(partial, "住宿每晚 600 元 [1]；餐补标准资料中没有提到。", ["每晚 600 元"])["correct"]
    assert not score_case(partial, "住宿每晚 600 元 [1]，餐补每天 100 元。", ["每晚 600 元"])["correct"]

    refuse = AnswerCase(question="产假几天", expect="refuse")
    assert score_case(refuse, "根据现有资料无法回答：资料中没有产假规定。", [])["correct"]
    assert not score_case(refuse, "产假 98 天。", [])["correct"]

    premise = AnswerCase(question="为什么年假不能跨年", expect="correct", facts=[["可以跨年"]], forbidden=["不能跨年使用的原因是"])
    assert score_case(premise, "这个前提不对：现行规定年假可以跨年使用 [1]。", ["可以跨年使用"])["correct"]
    # 资料明明能答，第一句却是拒答：后文给出了事实也算错（用户读到的是"无法回答"）
    s = score_case(premise, "根据现有资料无法回答为什么不能跨年。资料规定年假可以跨年使用 [1]。", ["可以跨年使用"])
    assert not s["correct"] and s["falseRefusal"]


def test_start_forbidden_catches_self_contradiction():
    case = AnswerCase(question="1 月的调休现在还能用吗", facts=[["不能", "已过期"]], start_forbidden=["^能", "^可以"])
    assert not score_case(case, "能。有效期 6 个月，1 月的调休到 7 月，现已过期，因此不能再用 [1]。", [])["correct"]
    assert score_case(case, "1 月 + 6 个月 = 7 月，早于今天，已过期，不能再用 [1]。", [])["correct"]
    assert score_case(case, "不能。有效期 6 个月，已过期 [1]。", [])["correct"]


def test_ungrounded_numbers():
    src = ["一线城市每晚 600 元，可上浮 20%。生效期：2026-01-01 ~ -"]
    assert ungrounded_numbers("每晚 600 元，可上浮 20% [1]", src) == []
    assert ungrounded_numbers("每晚 650 元 [1]", src) == ["650"]
    # 行尾的数字（来源头里的"版本：3.0"）不能把换行吞进去
    assert ungrounded_numbers("售后政策 3.0 版规定 [1]", ["来源：售后政策\n版本：3.0\n正文"]) == []
    # 序号、小整数计数、引用编号、来源里的日期都不算
    assert ungrounded_numbers("1. 先申请\n2. 共 3 个步骤 [2]\n自 2026 年 1 月 1 日起", src) == []
    # 问题里出现过的数字也不算
    assert ungrounded_numbers("P6 对应一等座", [], question="P6 能坐什么") == []


def test_parse_judge_and_summary():
    j = parse_judge('思考……{"unsupported": [{"claim": "餐补 100 元", "reason": "资料没有"}], "answered": true, "addresses_question": true}')
    assert j["unsupported"][0]["claim"] == "餐补 100 元"
    assert "error" in parse_judge("没有 JSON")
    case = AnswerCase(question="q", facts=[["a"]])
    rows = [
        {"case": case.model_dump(), "answer": "a", "score": score_case(case, "a", ["a"], cited=True), "judge": {"unsupported": []}},
        {"case": case.model_dump(), "answer": "b", "score": score_case(case, "b", ["a"], cited=False), "judge": j},
    ]
    s = summarize(rows)
    assert s["accuracy"] == 0.5 and s["hallucinationRate"] == 0.5 and s["uncitedRate"] == 0.5
