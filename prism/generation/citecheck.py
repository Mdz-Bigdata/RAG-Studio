"""回答的引用校验：每个 [n] 所在的句子，能不能在本轮第 n 段资料里找到词面依据。

只看词面覆盖（汉字二元组 + 英文词 / 数字），不做语义判断——目标是抓"串号"：
模型把对话历史里上一轮的引用编号带进本轮（"按铁律十八……[3]"，而本轮的 [3] 是铁律十），
或句子内容另有出处却标了别的编号。两类问题：
- mismatch：第 n 段资料几乎不含这句话的内容，另一段资料明显更吻合（建议改为那一段）；
- unsupported：本轮所有资料都几乎不含这句话的内容（可能来自对话历史或模型自身）。
同时引用了图片（[图n] / [用户图片]）的句子不校验：内容可能来自图片本身。
阈值用真实回答校准：正确引用的句子对所引资料的覆盖率普遍 ≥ 0.25，串号的那句只有 0.17、对其它资料也都 < 0.2。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_CITE = re.compile(r"\[(\d+)\]")
_MARKERS = re.compile(r"\[(?:\d+|图\s*\d+|用户图片)\]")
_IMAGE_REF = re.compile(r"\[(?:图\s*\d+|用户图片)\]")
_CLAUSE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
_ASCII = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+|\d+(?:\.\d+)?%?")
_CJK = re.compile(r"[一-鿿]+")
# 回答里的结构词 / 套话：由它们组成的二元组不算证据
_FILLER = set("结论根据资料显示说明以下如下其中这段代码违反应该怎么改需要注意补充可以以及并且对于")
# 说明资料缺失的句子（"餐补标准：资料中未包含……"）
_MISSING = re.compile(r"无法回答|无法确定|没有足够信息|资料(?:中|里)?(?:没有|未|并未|不)|未(?:包含|提及|提到|涉及|说明|规定|给出|列出|记载)|没有(?:提及|提到|涉及|说明|规定|给出|列出|记载|相关)")

# 数字依据：回答里的数值在本轮资料和问题里都找不到 → 很可能是编的（幻觉里最常见、危害最大的一类）
_NUM = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*[ \t]*%?")
_LIST_ORD = re.compile(r"(?m)^\s*\d+[.)、]\s")
_DATE_LIKE = re.compile(r"\d{4}[-/年.]\d{1,2}(?:[-/月.]\d{1,2}日?)?")
# 写出了算式 / 换算的句子：数值是推导出来的（600 × 1.2 = 720），不算无依据
_ARITH = re.compile(r"[×*÷/=＝+＋]|合计|共计|总计|相差|差值|差额|之和|乘以|除以|加上|减去|上浮后|折合|累计|计算|推算|"
                    r"即(?:最多|最高|最少|最低|为|约|共|每)")
# 写出了算式的句子（600 × 1.2 = 720）：是推导，不是对某段资料的转述，不做词面引用校验（操作数另由数字校验把关）
_FORMULA = re.compile(r"\d\s*[×*÷=＝]|[×*÷=＝]\s*\d")


def numbers_in(text: str) -> set[str]:
    """文本里出现的数值（去掉引用编号、列表序号；日期整体与各段都算；'1,500' 与 '1500'、'20 %' 与 '20%' 视为相同）。"""
    t = unicodedata.normalize("NFKC", text or "")
    t = _MARKERS.sub(" ", t)
    t = _LIST_ORD.sub(" ", t)
    out: set[str] = set()
    for m in _DATE_LIKE.finditer(t):
        out.update(re.findall(r"\d+", m.group(0)))
    for m in _NUM.finditer(t):
        raw = re.sub(r"\s+", "", m.group(0)).replace(",", "")
        bare = raw.rstrip("%")
        out.update((raw, bare))
        if "." in bare:
            out.add(bare.rstrip("0").rstrip("."))
    return {n for n in out if n}


def ungrounded_numbers(answer: str, sources: list[str], question: str = "") -> list[str]:
    """回答里在资料和问题中都没出现过的数值。0~10 的小整数常用于计数、序号（"两种 / 3 个步骤"），不计。"""
    have = numbers_in(question)
    for s in sources:
        have |= numbers_in(s)
    out: list[str] = []
    seen: set[str] = set()
    for n in sorted(numbers_in(answer)):
        bare = n.rstrip("%")
        if n in have or bare in have or bare in seen:
            continue
        try:
            if not n.endswith("%") and "." not in bare and float(bare) <= 10:
                continue
        except ValueError:
            continue
        seen.add(bare)
        out.append(n)
    return out


def check_numbers(answer: str, sources: list[str], *, question: str = "") -> list[dict[str, Any]]:
    """逐句找无依据的数值；写出了算式 / 换算的句子跳过（推导值本来就不在资料里）。
    reason = "number"，value = 数值，index = 句子引用的第一个资料编号（没有引用时为 0）。"""
    issues: list[dict[str, Any]] = []
    reported: set[str] = set()
    for m in _CLAUSE.finditer(answer or ""):
        clause = m.group(0)
        if _ARITH.search(strip_citations(clause)) or _MISSING.search(clause):
            continue
        for n in ungrounded_numbers(clause, sources, question):
            if n.rstrip("%") in reported:
                continue
            reported.add(n.rstrip("%"))
            cites = [int(x) for x in _CITE.findall(clause)]
            issues.append({"index": cites[0] if cites else 0, "reason": "number", "sentence": strip_citations(clause).strip(" -*#>|\t")[:80], "support": 0.0, "suggest": None, "value": n})
    return issues


def strip_citations(text: str) -> str:
    """去掉 [n] / [图n] / [用户图片] 引用标记。对话历史交给模型前用：上一轮的编号在本轮已经失效。"""
    return _MARKERS.sub("", text or "")


def _grams(text: str) -> set[str]:
    out: set[str] = set()
    for run in _CJK.findall(text):
        out.update(run[i:i + 2] for i in range(len(run) - 1) if not (run[i] in _FILLER and run[i + 1] in _FILLER))
    out.update(m.group(0).lower() for m in _ASCII.finditer(text))
    return out


def check_citations(answer: str, sources: list[str], *, question: str = "", min_support: float = 0.2, min_grams: int = 6) -> list[dict[str, Any]]:
    """sources：本轮交给模型的资料（按 prompts.format_context 的样子，含来源 / 版本行），顺序即 [1]…[N]。
    question：用户问题——句子里复述问题的部分（"550 元的酒店能全额报销"里的 550、全额报销）不需要资料依据，不计入覆盖率。
    返回发现的问题（空列表 = 没发现问题）。"""
    src = [_grams(s or "") for s in sources]
    asked = _grams(question or "")
    issues: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for m in _CLAUSE.finditer(answer or ""):
        clause = m.group(0)
        nums = sorted({int(n) for n in _CITE.findall(clause)})
        if not nums or _IMAGE_REF.search(clause) or _MISSING.search(clause) or _FORMULA.search(clause):
            continue                                   # 说明"资料里没有"的句子、算式不是对资料的转述，标了编号也无从校验
        claim = strip_citations(clause).strip(" -*#>|\t")
        grams = _grams(claim) - asked
        if len(grams) < min_grams:
            continue                                   # 太短的句子词面证据不够，不判
        cover = [len(grams & g) / len(grams) if g else 0.0 for g in src]
        best = max(range(len(src)), key=cover.__getitem__) if src else -1
        for n in nums:
            if (n, claim[:40]) in seen:
                continue
            seen.add((n, claim[:40]))
            if not 1 <= n <= len(src):
                issues.append({"index": n, "reason": "out_of_range", "sentence": claim[:80], "support": 0.0, "suggest": None})
                continue
            s = cover[n - 1]
            if s >= min_support:
                continue
            if best >= 0 and cover[best] >= max(min_support, 2 * s) and best + 1 not in nums:
                issues.append({"index": n, "reason": "mismatch", "sentence": claim[:80], "support": round(s, 2), "suggest": best + 1})
            elif best < 0 or cover[best] < min_support:
                issues.append({"index": n, "reason": "unsupported", "sentence": claim[:80], "support": round(s, 2), "suggest": None})
    return issues
