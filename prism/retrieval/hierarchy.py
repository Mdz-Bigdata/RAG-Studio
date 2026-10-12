"""按问题类型选检索粒度（多粒度分层索引的路由）。

- 概念型："什么是 / 介绍 / 概述 / 有哪些 / 原理 / 为什么 / 区别" → 章节级（需要完整上下文）
- 细节型：含数字 / 编号 / 日期，或"多少 / 几天 / 哪一 / 是否 / 能不能 / 什么时候" → 句子级（精确定位）
- 其他 → 只走默认段落级（child → parent）

规则分类零成本、可解释；分类结果写进检索链路的说明里。
"""

from __future__ import annotations

import re

from .query_transform import has_precise_tokens

_CONCEPT = re.compile(r"(什么是|是什么|介绍|概述|概括|总结|总体|整体|有哪些|包括哪些|哪几[类种个方面]|原理|为什么|为何|区别|差异|对比|作用|意义|流程是|如何理解|overview|what is|explain)", re.I)
_DETAIL = re.compile(r"(多少|几[天个次条号点月年岁元块]|哪一|哪天|哪个|是否|能不能|可不可以|可以.{0,6}吗|什么时候|何时|期限|上限|下限|标准是|金额|价格|费用|电话|邮箱|地址|编号|how many|how much|when)", re.I)

LEVEL_LABEL = {"concept": "概念型 → 章节级", "detail": "细节型 → 句子级", "general": "一般问题 → 段落级"}


def classify_query(query: str) -> str:
    q = query.strip()
    if _DETAIL.search(q) or has_precise_tokens(q):
        return "detail"
    if _CONCEPT.search(q):
        return "concept"
    return "general"


def levels_for(query: str, routing: str) -> tuple[list[str], str]:
    """返回要额外启用的层（section / sentence）与问题类型。"""
    if routing == "all":
        return ["section", "sentence"], "all"
    kind = classify_query(query)
    return ({"concept": ["section"], "detail": ["sentence"]}.get(kind, []), kind)
