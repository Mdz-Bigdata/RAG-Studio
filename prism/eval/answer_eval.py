"""回答级评测：检索指标好看不等于回答对、不等于没幻觉。

检索评测（runner.py）只回答"相关段落有没有排进前 K"。用户看到的是回答，这里直接评回答：

- 准确：回答里有没有给出标准答案的关键事实（facts：每条事实列出可接受的几种写法，全部命中才算答对）；
- 拒答：库里没有答案的问题（expect=refuse）有没有明确说资料里没有；有答案的问题有没有被误拒；
- 部分可答（expect=partial）：能答的部分答对，同时明确说出缺的那部分；
- 错误前提（expect=correct）：问题本身带着错误前提（"为什么年假不能跨年"），回答要按资料纠正；
- 幻觉（确定性）：回答里出现的数字，在交给模型的资料和问题里都找不到 → 无依据数字；
- 幻觉（LLM 裁判，可选）：逐条找出资料不支持的论断。

确定性指标是主指标：可复现、不受裁判模型偏好影响；裁判模型和回答模型是同一家时会偏宽，只作辅助。
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..generation.citecheck import ungrounded_numbers

Expect = Literal["answer", "refuse", "partial", "correct"]


class AnswerCase(BaseModel):
    question: str
    kb: str = ""                                     # 知识库名称或 id；为空时由调用方决定
    category: str = "事实"
    expect: Expect = "answer"
    facts: list[list[str]] = Field(default_factory=list)   # 每条事实的可接受写法（任一命中即算这条事实答出）
    forbidden: list[str] = Field(default_factory=list)     # 出现即判错的内容（明确的错误答案）
    start_forbidden: list[str] = Field(default_factory=list)  # 第一句不能这样开头（正则，匹配规范化后的第一句）：
                                                           # 抓"能。……因此不能再用"这类先下结论、后推算出相反结果的自相矛盾
    note: str = ""


def load_cases(path: Path) -> list[AnswerCase]:
    out: list[AnswerCase] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(AnswerCase.model_validate_json(line))
    return out


# ------------------------------------------------------------------------------------------ 规范化与匹配
_CITE_MARK = re.compile(r"\[(?:\d+|图\s*\d+|用户图片)\]")
_MD = re.compile(r"[*_`#>|]")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """全角转半角、去掉空白与 Markdown 符号、小写：'10 个工作日' 与 '10个工作日'、'１５９９９' 与 '15999' 视为相同。"""
    t = unicodedata.normalize("NFKC", text or "")
    t = _CITE_MARK.sub("", t)
    t = _MD.sub("", t)
    t = t.replace("～", "~").replace("—", "-").replace("–", "-")
    return _SPACE.sub("", t).lower()


def fact_hits(answer: str, facts: list[list[str]]) -> list[bool]:
    a = normalize(answer)
    return [any(normalize(alt) in a for alt in alts) for alts in facts]


# 拒答 / 资料缺失的说法。部分可答时这些话出现在回答中间，所以全文匹配；
# "完全拒答"另看开头（looks_like_full_refusal）
_MISSING = (
    "无法回答", "无法确定", "无法给出", "没有足够信息", "信息不足", "资料不足", "未找到", "没有找到", "未检索到", "没有检索到",
    "资料中没有", "资料里没有", "资料没有", "没有提到", "未提到", "未提及", "没有提及", "没有涉及", "未涉及", "没有相关", "无相关",
    "未包含", "没有包含", "不包含", "没有明确", "未明确", "没有规定", "未规定", "没有说明", "未说明", "没有记载", "未记载",
    "没有给出", "未给出", "并未", "暂无", "没有关于", "不涉及", "未列出", "没有列出", "查不到",
)


def mentions_missing(text: str) -> bool:
    t = normalize(text)
    return any(normalize(k) in t for k in _MISSING)


_FIRST_SENTENCE = re.compile(r"[^。！？!?\n]+")


def looks_like_full_refusal(text: str) -> bool:
    """整体拒答：第一句就说资料里没有 / 无法回答。部分可答的回答第一句也可能这么说（"只能回答住宿，无法回答餐补"），
    所以 score_case 里还要看关键事实有没有答出来。"""
    m = _FIRST_SENTENCE.search(unicodedata.normalize("NFKC", text or "").strip())
    head = normalize(m.group(0)) if m else ""
    return any(normalize(k) in head for k in _MISSING)


# 无依据数字：与线上回答的数字校验（generation.citecheck）是同一份实现


# ------------------------------------------------------------------------------------------ 打分
def score_case(case: AnswerCase, answer: str, sources: list[str], *, cited: bool | None = None) -> dict[str, Any]:
    """确定性打分。sources：交给模型的资料正文（含图片描述 / 图中文字）。"""
    hits = fact_hits(answer, case.facts)
    all_facts = all(hits) if hits else True
    bad = [f for f in case.forbidden if normalize(f) in normalize(answer)]
    first = _FIRST_SENTENCE.search(unicodedata.normalize("NFKC", answer or "").strip())
    head = normalize(first.group(0)) if first else ""
    bad += [f"开头：{p}" for p in case.start_forbidden if re.search(p, head)]
    opens_with_refusal = looks_like_full_refusal(answer)
    full_refusal = opens_with_refusal and not any(hits)
    missing = mentions_missing(answer)
    if case.expect in ("answer", "correct"):
        # 资料能答的问题，第一句却是"根据现有资料无法回答"——哪怕后文给出了事实，用户读到的也是拒答
        correct = all_facts and not bad and not opens_with_refusal
    elif case.expect == "partial":
        correct = all_facts and missing and not bad
    else:   # refuse
        correct = (full_refusal or (missing and not case.facts)) and not bad
    ungrounded = ungrounded_numbers(answer, sources, case.question) if not (case.expect == "refuse" and full_refusal) else []
    return {
        "correct": bool(correct),
        "factHits": hits,
        "factRecall": (sum(hits) / len(hits)) if hits else None,
        "forbiddenHit": bad,
        "fullRefusal": full_refusal,
        "mentionsMissing": missing,
        "falseRefusal": case.expect in ("answer", "correct") and opens_with_refusal,
        "ungroundedNumbers": ungrounded,
        "cited": cited,
    }


# ------------------------------------------------------------------------------------------ LLM 裁判
JUDGE_SYSTEM = """你是严格的 RAG 回答审校员，只判断"回答是否忠实于资料"，不判断资料本身对不对。

逐条检查回答中的事实性论断（数字、日期、条件、范围、例外、步骤、因果、比较、对象归属）：
- 资料中有直接依据，或可由资料直接推出（例如简单的加减比较）→ 有依据；
- 资料中没有、与资料矛盾、把资料的范围扩大或缩小（"所有员工"而资料只说"远程员工"）、
  把 A 对象的规定套到 B 对象、补充了资料外的常识或建议并当作制度/事实陈述 → 无依据；
- 回答前后自相矛盾（先说"能"、后面又推出"不能"），把与资料不符的那个结论列为无依据；
- 资料里注明了日期、期限时，结合 <当前日期> 判断回答的结论是否成立（期限已过却说"还能用"→ 无依据）；
- 回答说"资料中没有 / 无法回答"的部分不算论断；礼貌用语、复述问题不算论断。

只输出一个 JSON 对象，不要输出其他文字：
{"unsupported": [{"claim": "回答中的原句或要点", "reason": "为什么无依据"}], "answered": true 或 false, "addresses_question": true 或 false}
answered：回答是否给出了实质性答案（整体拒答为 false）。addresses_question：回答是否针对用户的问题。"""


def build_judge_prompt(question: str, sources: list[str], answer: str, today: str | None = None) -> str:
    from datetime import date

    parts = [f"<当前日期>{today or date.today().isoformat()}</当前日期>", "<资料>"]
    for i, s in enumerate(sources, start=1):
        parts.append(f"[{i}] {s.strip()[:3000]}")
    parts.append("</资料>")
    parts.append(f"<问题>\n{question}\n</问题>")
    parts.append(f"<回答>\n{answer}\n</回答>")
    return "\n\n".join(parts)


def parse_judge(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return {"error": "no_json", "raw": t[:300]}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"error": "bad_json", "raw": t[:300]}
    uns = data.get("unsupported") or []
    if not isinstance(uns, list):
        uns = []
    return {"unsupported": [u for u in uns if isinstance(u, dict)], "answered": bool(data.get("answered", True)), "addressesQuestion": bool(data.get("addresses_question", True))}


# ------------------------------------------------------------------------------------------ 汇总
def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows：每个样例 {case: AnswerCase.model_dump(), score: score_case(...), judge: parse_judge(...)?, expiredInContext: bool?}"""
    def rate(items: list[bool]) -> float | None:
        return round(sum(items) / len(items), 4) if items else None

    by_expect: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_expect[r["case"]["expect"]].append(r)
        by_cat[r["case"]["category"]].append(r)
    judged = [r for r in rows if r.get("judge") and "error" not in r["judge"]]
    answered = [r for r in rows if not r["score"]["fullRefusal"]]
    out: dict[str, Any] = {
        "n": len(rows),
        "accuracy": rate([r["score"]["correct"] for r in rows]),
        "answerAccuracy": rate([r["score"]["correct"] for r in by_expect["answer"]]),
        "refusalAccuracy": rate([r["score"]["correct"] for r in by_expect["refuse"]]),
        "partialAccuracy": rate([r["score"]["correct"] for r in by_expect["partial"]]),
        "premiseCorrection": rate([r["score"]["correct"] for r in by_expect["correct"]]),
        "falseRefusalRate": rate([r["score"]["falseRefusal"] for r in by_expect["answer"] + by_expect["correct"]]),
        "ungroundedNumberRate": rate([bool(r["score"]["ungroundedNumbers"]) for r in answered]),
        "hallucinationRate": rate([bool(r["judge"]["unsupported"]) for r in judged]),
        "unsupportedClaims": sum(len(r["judge"]["unsupported"]) for r in judged),
        "judged": len(judged),
        "citationIssueRate": rate([bool(r.get("citationIssues")) for r in answered if r.get("citationIssues") is not None]),
        "uncitedRate": rate([not r["score"]["cited"] for r in answered if r["score"]["cited"] is not None and r["case"]["expect"] != "refuse"]),
        "expiredInContext": sum(1 for r in rows if r.get("expiredInContext")),
        "byCategory": {c: {"n": len(rs), "accuracy": rate([r["score"]["correct"] for r in rs]),
                           "hallucinationRate": rate([bool(r["judge"]["unsupported"]) for r in rs if r.get("judge") and "error" not in r["judge"]])}
                       for c, rs in sorted(by_cat.items())},
    }
    return out


def to_markdown(summary: dict[str, Any], rows: list[dict[str, Any]], *, title: str = "回答评测") -> str:
    def pct(v: float | None) -> str:
        return "—" if v is None else f"{v * 100:.1f}%"

    lines = [f"# {title}（n={summary['n']}）", "", "| 指标 | 值 |", "|---|---|"]
    for key, label in (("accuracy", "总体正确率"), ("answerAccuracy", "可答题正确率"), ("refusalAccuracy", "无答案题正确拒答率"),
                       ("partialAccuracy", "部分可答题正确率"), ("premiseCorrection", "错误前提纠正率"), ("falseRefusalRate", "误拒率"),
                       ("hallucinationRate", "幻觉率（裁判）"), ("ungroundedNumberRate", "无依据数字率"), ("citationIssueRate", "引用问题率"),
                       ("uncitedRate", "无引用回答率")):
        lines.append(f"| {label} | {pct(summary.get(key))} |")
    lines.append(f"| 过期文档进入上下文 | {summary.get('expiredInContext', 0)} 题 |")
    lines += ["", "## 按类别", "| 类别 | n | 正确率 | 幻觉率 |", "|---|---|---|---|"]
    for c, d in summary["byCategory"].items():
        lines.append(f"| {c} | {d['n']} | {pct(d['accuracy'])} | {pct(d['hallucinationRate'])} |")
    fails = [r for r in rows if not r["score"]["correct"] or (r.get("judge") or {}).get("unsupported") or r["score"]["ungroundedNumbers"]]
    if fails:
        lines += ["", "## 需要关注的样例"]
        for r in fails:
            c, s = r["case"], r["score"]
            flags = []
            if not s["correct"]:
                flags.append("错误")
            if s["ungroundedNumbers"]:
                flags.append("无依据数字 " + "、".join(s["ungroundedNumbers"]))
            for u in (r.get("judge") or {}).get("unsupported") or []:
                flags.append(f"无依据：{str(u.get('claim'))[:60]}")
            lines.append(f"- **[{c['expect']}·{c['category']}] {c['question']}** — {'；'.join(flags)}")
            lines.append(f"  - 回答：{_SPACE.sub(' ', r['answer'])[:220]}")
    return "\n".join(lines)
