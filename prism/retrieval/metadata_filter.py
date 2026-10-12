"""元数据过滤（自查询）：从问题里读出对文档范围的限定，召回前先把候选缩小。

向量相似度分不清"话题对、但年份 / 部门 / 类型不对"的文档。这里把问题里的限定（"2023 年的""财务部发布的"
"云计算方面的"）翻译成文档属性上的条件，交给检索前的 allowed_ids 预过滤：先筛再算相似度，
不会出现"前 K 条全被过滤掉、库里明明有却搜不到"的情况。

- 取值只能来自本知识库里真实存在的属性值（facets）；模型编出来的字段和取值一律丢弃。
- 规则模式只认年份，以及后面跟着"发布 / 制定 / 的制度 / 的文档"这类说法的部门名（"报人力资源部备案"只是提到，不算）。
  标签、文件类型要判断"是在限定范围，还是只是提到"，交给 LLM。
- 自动识别的条件是推测：默认不排除没填该属性的文档；过滤后没有候选或没有检索结果时退回不过滤。
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterable
from datetime import date
from typing import Any

from ..schemas import DocMetadata
from .temporal import _YEAR

logger = logging.getLogger(__name__)

FIELDS = ("department", "tags", "year", "file_format")
LABELS = {"department": "部门", "tags": "标签", "year": "年份", "file_format": "文件类型"}
_DESCRIPTIONS = {
    "department": "发布 / 负责这份文档的部门",
    "tags": "文档的主题标签",
    "year": "文档所属的年份（版本年份或生效年份）",
    "file_format": "文件类型",
}
_MAX_PROMPT_VALUES = 60   # 每个字段给模型看的取值上限，按文档数从多到少取

_PROMPT = """你在为检索系统做“自查询”：从用户问题里找出对“要查哪些文档”的限定条件。

知识库里的文档有下面这些属性，取值只能从列出的值里选：
{schema}

规则：
- 只有当问题明确限定了要哪一年、哪个部门、哪个主题或哪种类型的文档时，才输出对应条件。
- 问题只是提到某个词、并不是在限定文档范围时，不要输出。例如“财务部的报销流程怎么走”是在问流程，不等于只看财务部发布的文档；
  “M4 Pro 芯片的机型多少钱”里的产品名、型号、术语是要查的内容，不是标签条件。
- 标签只在用户点明要某个领域或类别的资料时才用，例如“云计算方面的”“售后类的文档里”“关于安全的资料”。
- 取值必须来自上面的列表。列表里没有对应的值，就不输出这个字段。不要推测，拿不准就输出空对象。

只输出 JSON 对象，例如 {{"tags": ["云计算"], "year": [2023]}}；没有条件时输出 {{}}。
{history}
问题：{query}"""


def doc_years(md: DocMetadata, today: date | None = None) -> set[int]:
    """文档属于哪些年份：标题 / 版本号里的年份，加上生效区间覆盖的年份（未写失效日期的算到今年）。"""
    years = {int(y) for y in _YEAR.findall(" ".join(x for x in (md.version or "", md.title or "") if x))}
    extra = md.extra.get("year")
    if isinstance(extra, (int, str)) and str(extra).isdigit():
        years.add(int(extra))
    if md.valid_from:
        end = md.valid_to.year if md.valid_to else max(md.valid_from.year, (today or date.today()).year)
        years.update(range(md.valid_from.year, end + 1))
    elif md.valid_to:
        years.add(md.valid_to.year)
    return years


def doc_values(md: DocMetadata, field: str, today: date | None = None) -> list[Any]:
    if field == "department":
        return [md.department.strip()] if md.department and md.department.strip() else []
    if field == "tags":
        return [t.strip() for t in md.tags if isinstance(t, str) and t.strip()]
    if field == "year":
        return sorted(doc_years(md, today))
    if field == "file_format":
        return [md.file_format.strip().lower()] if md.file_format and md.file_format.strip() else []
    return []


def _key(value: Any) -> Any:
    return value.casefold() if isinstance(value, str) else value


def facets(docs: Iterable[DocMetadata], today: date | None = None) -> dict[str, dict[Any, int]]:
    """每个属性在这批文档里出现过哪些值、各有几篇文档；按文档数从多到少排。"""
    counts: dict[str, Counter] = {f: Counter() for f in FIELDS}
    shown: dict[str, dict[Any, Any]] = {f: {} for f in FIELDS}
    for md in docs:
        if md is None:
            continue
        for field in FIELDS:
            for value in {_key(v): v for v in doc_values(md, field, today)}.values():
                shown[field].setdefault(_key(value), value)
                counts[field][_key(value)] += 1
    return {f: {shown[f][k]: n for k, n in sorted(counts[f].items(), key=lambda kv: (-kv[1], str(kv[0])))} for f in FIELDS}


def matches(md: DocMetadata | None, conditions: dict[str, list[Any]], *, include_missing: bool, today: date | None = None) -> bool:
    """字段之间是“且”，同一字段的多个取值是“或”。include_missing：文档没填这个属性时算不算匹配。"""
    for field, want in conditions.items():
        if field not in FIELDS or not want:
            continue
        have = {_key(v) for v in doc_values(md, field, today)} if md is not None else set()
        if not have:
            if include_missing:
                continue
            return False
        if not have & {_key(int(v) if field == "year" and str(v).isdigit() else v) for v in want}:
            return False
    return True


def _validated(raw: Any, available: dict[str, dict[Any, int]], fields: Iterable[str]) -> dict[str, list[Any]]:
    """只保留知识库里真实存在的字段和取值，并换成库里的写法。"""
    out: dict[str, list[Any]] = {}
    if not isinstance(raw, dict):
        return out
    for field in fields:
        values = raw.get(field)
        if values is None or field not in available:
            continue
        known = {_key(v): v for v in available[field]}
        picked: list[Any] = []
        for value in values if isinstance(values, (list, tuple, set)) else [values]:
            if field == "year":
                value = int(value) if isinstance(value, (int, str)) and not isinstance(value, bool) and str(value).strip().isdigit() else None
            elif isinstance(value, str):
                value = value.strip()
            else:
                value = None
            if value is not None and _key(value) in known and known[_key(value)] not in picked:
                picked.append(known[_key(value)])
        if picked:
            out[field] = picked
    return out


# 部门名后面要跟着"是它发的 / 它的文档"这类说法才算限定范围
_DEPT_SCOPE = r"\s*(?:发布|制定|印发|编制|出台|起草|下发|发的|出的|(?:的|里的|那份|这份)?\s*(?:文档|文件|制度|规定|手册|政策|资料|通知|规范|办法|细则))"


def extract_rule(query: str, available: dict[str, dict[Any, int]], fields: Iterable[str]) -> dict[str, list[Any]]:
    """不调模型：问题里的独立年份，以及明确作为文档来源出现的部门名。"""
    q = (query or "").casefold()
    raw = {
        "year": [int(y) for y in _YEAR.findall(query or "")],
        "department": [d for d in available.get("department", {})
                       if len(str(d)) >= 2 and re.search(re.escape(str(d).casefold()) + _DEPT_SCOPE, q)],
    }
    return _validated(raw, available, [f for f in fields if f in raw])


def extract_llm(query: str, available: dict[str, dict[Any, int]], fields: Iterable[str], llm: Any,
                history: list[dict[str, str]] | None = None) -> dict[str, list[Any]]:
    """让模型在给定的属性和取值范围内挑条件；输出不合法或调用失败时抛异常，由调用方决定怎么退。"""
    fields = [f for f in fields if available.get(f)]
    schema = "\n".join(
        f"- {f}（{_DESCRIPTIONS[f]}）：" + "、".join(str(v) for v in list(available[f])[:_MAX_PROMPT_VALUES]) for f in fields)
    turns = [str(m.get("content") or "").strip()[:200] for m in (history or []) if m.get("role") == "user"][-2:]
    context = ("\n对话上文（用户之前的提问，仅用于理解指代）：\n" + "\n".join(f"- {t}" for t in turns if t) + "\n") if any(turns) else ""
    raw = llm.complete(_PROMPT.format(schema=schema, history=context, query=query), max_tokens=200)
    match = re.search(r"\{.*\}", re.sub(r"```(?:json)?", "", raw), re.S)
    if not match:
        raise ValueError("no json object")
    return _validated(json.loads(match.group(0)), available, fields)


def describe(conditions: dict[str, list[Any]]) -> str:
    return "、".join(f"{LABELS.get(f, f)}={' / '.join(str(v) for v in values)}" for f, values in conditions.items() if values)
