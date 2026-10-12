"""时效：问题是不是在问历史规定（旧版 / 已废止的制度）。

默认检索会过滤过期语料（retrieval.filter_expired），"2024 版 vs 2026 版反复横跳"靠它解决。
但用户明确问旧规定时（"2024 版的住宿标准""旧版要提前几天""去年没休完的年假"），过滤掉旧版只会让模型
拿现行规定冒充旧规定作答。这里识别这类问题，让检索放行过期语料——资料里会标注"已失效"，由回答说明时效。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date

from ..schemas import DocMetadata

_HISTORY = [
    re.compile(r"旧版|老版|旧的?(?:制度|规定|手册|政策|标准|版本|条款)"),
    re.compile(r"(?:以前|之前|原来|过去|当时|往年|早先)的?(?:规定|制度|政策|标准|版本|手册|要求|条款)"),
    re.compile(r"历史版本|上一?个版本|上一版|前一版|前一个版本"),
    re.compile(r"去年|前年"),
    re.compile(r"(?:废止|失效|过期)的?(?:版本|制度|规定|手册|文件|政策|条款)"),
    re.compile(r"新旧(?:版本|制度|规定|手册|政策)?|版本(?:变化|差异|区别|对比|变更)|(?:新版|现行).{0,8}(?:旧版|以前|之前)"),
]
# 独立的年份（"2024 版""2025 年"）；"CVE-2024-3094""2024-05" 里的不算
_YEAR = re.compile(r"(?<![A-Za-z0-9\-])((?:19|20)\d{2})(?![0-9\-])")


def _doc_years(md: DocMetadata) -> set[str]:
    years = set(_YEAR.findall(" ".join(x for x in (md.version or "", md.title or "") if x)))
    if md.valid_from and md.valid_to:
        years.update(str(y) for y in range(md.valid_from.year, md.valid_to.year + 1))
    elif md.valid_to:
        years.add(str(md.valid_to.year))
    return years


def history_intent(query: str, docs: Iterable[DocMetadata], on_date: date | None = None) -> str | None:
    """问题在问历史规定 → 返回命中的说法（写进检索链路说明）；否则 None。
    年份只有落在某份已过期文档的版本 / 标题 / 生效区间里才算（"2025 年 Q3 销售额"在没有过期文档的库里不算）。"""
    q = query or ""
    for pat in _HISTORY:
        m = pat.search(q)
        if m:
            return m.group(0)
    years = set(_YEAR.findall(q))
    if not years:
        return None
    for md in docs:
        if md is not None and not md.is_effective(on_date) and years & _doc_years(md):
            return next(iter(sorted(years & _doc_years(md))))
    return None
