"""自动元数据抽取：标题、版本、生效 / 失效日期、发布部门、标签、密级。

元数据是检索前"硬过滤"的依据（过期制度不召回、无权限文档不可见），但靠人手填，大部分文档永远是空的。
这里先用规则从正文里找"文档自己写明的"信息，再（可选）让 LLM 补齐规则没找到的字段：

- 规则只认明确的写法（"自 2026 年 1 月 1 日起施行""有效期至……""密级：机密"），宁缺毋滥——
  日期抽错会让文档被当成过期而检索不到，比不填更糟；
- LLM 的结果要能在原文里找到依据（日期的年份、版本号、部门名必须出现在正文里），否则丢弃；
- 只填空字段，人工填过的值永远不覆盖。

每个字段返回 {value, source: rule | llm | tfidf, evidence}，前端据此展示"自动抽取"标记和依据原文。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any

from ..schemas import BlockType, DocMetadata, ParsedDocument
from .keywords import extract_keywords

logger = logging.getLogger(__name__)

FIELDS = ("title", "version", "valid_from", "valid_to", "department", "tags", "access_level")

_DATE = r"(\d{4})\s*[年\-/.]\s*(\d{1,2})\s*[月\-/.]\s*(\d{1,2})\s*日?"
_VALID_FROM = [
    re.compile(rf"(?:自|从|于)?\s*{_DATE}\s*起\s*(?:正式)?(?:施行|实施|生效|执行)"),
    re.compile(rf"(?:生效|施行|实施|执行)(?:日期|时间)\s*[:：]\s*{_DATE}"),
    re.compile(rf"本(?:制度|办法|规定|细则|规范|标准|政策|通知|方案|手册)\s*(?:自|于)\s*{_DATE}\s*(?:起)?\s*(?:施行|实施|生效|执行)"),
]
_VALID_TO = [
    re.compile(rf"有效期(?:至|到|截止(?:到|至)?)\s*[:：]?\s*{_DATE}"),
    re.compile(rf"(?:失效|废止|截止)(?:日期|时间)\s*[:：]\s*{_DATE}"),
    re.compile(rf"{_DATE}\s*(?:起)?\s*(?:废止|失效|停止执行)"),
]
_VERSION = [
    re.compile(r"(?:版本(?:号)?|Version|Ver\.?|修订版次)\s*[:：]?\s*[Vv]?(\d+(?:\.\d+){0,3}[A-Za-z]?)"),
    re.compile(r"(?<![A-Za-z0-9])[Vv](\d+\.\d+(?:\.\d+){0,2})(?![\d.])"),
    re.compile(r"[（(]?(20\d{2})\s*(?:年)?\s*(?:版|修订版|修订)[）)]?"),
]
_DEPT = [
    re.compile(r"(?:发布|制定|起草|归口|主管|编制|负责)(?:部门|单位)\s*[:：]\s*([一-鿿A-Za-z]{2,20}?)(?=[\s，。；,;、\n]|$)"),
    re.compile(r"([一-鿿]{2,12}(?:部|中心|办公室|委员会|事业部|管理处|处|科|组))\s*(?:发布|制定|印发|编制)(?![一-鿿]{0,2}(?:的|之))"),
]
_ACCESS = [
    (re.compile(r"(?:密级|保密等级|文件密级|保密级别)\s*[:：]\s*(公开|内部|秘密|机密|绝密)"), None),
    (re.compile(r"^[\s【\[（(★]*(绝密|机密|秘密|内部资料|内部文件|仅限内部(?:使用|传阅)?)[\s】\]）)★]*$", re.M), None),
]
_ACCESS_LEVEL = {"公开": 0, "内部": 1, "内部资料": 1, "内部文件": 1, "仅限内部": 1, "仅限内部使用": 1, "仅限内部传阅": 1, "秘密": 2, "机密": 2, "绝密": 3}

_LLM_PROMPT = """下面是一份企业文档的开头与结尾部分。请抽取文档自己写明的元数据，输出一个 JSON 对象，字段：
title（文档标题）、version（版本号，如 "2.1" 或 "2026"）、valid_from（生效日期 YYYY-MM-DD）、valid_to（失效日期 YYYY-MM-DD）、
department（发布 / 负责部门）、tags（3~6 个主题标签，数组）。
文中没有明确写出的字段一律填 null，不要推测；只输出 JSON。

<开头>
{head}
</开头>
<结尾>
{tail}
</结尾>"""


def _date(m: re.Match) -> str | None:
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
    except ValueError:
        return None


def _snippet(text: str, m: re.Match, pad: int = 24) -> str:
    return text[max(0, m.start() - pad): m.end() + pad].replace("\n", " ").strip()


def _first(patterns: list[re.Pattern], text: str, conv=None) -> tuple[Any, str] | None:
    for pat in patterns:
        m = pat.search(text)
        if m:
            v = conv(m) if conv else m.group(1)
            if v:
                return v, _snippet(text, m)
    return None


def doc_text(doc: ParsedDocument) -> str:
    return doc.markdown or "\n".join(b.text for b in doc.blocks if b.type not in (BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER))


def extract_metadata(doc: ParsedDocument | str, llm=None, *, file_stem: str = "") -> dict[str, dict[str, Any]]:
    """doc 可以是解析结果，也可以是 Markdown 全文（校对后的文档）。"""
    if isinstance(doc, str):
        text, title_block = doc, next((ln.lstrip("#").strip() for ln in doc.splitlines() if ln.startswith("# ")), "")
    else:
        text = doc_text(doc)
        title_block = next((b.text.strip() for b in doc.blocks if b.type == BlockType.TITLE and (b.level or 1) <= 1 and b.text.strip()), "")
    head, tail = text[:4000], text[-1500:] if len(text) > 4000 else ""
    scope = head + "\n" + tail
    out: dict[str, dict[str, Any]] = {}

    if title_block and len(title_block) <= 60:
        out["title"] = {"value": title_block, "source": "rule", "evidence": "一级标题"}
    if hit := _first(_VERSION, head):
        out["version"] = {"value": hit[0], "source": "rule", "evidence": hit[1]}
    if hit := _first(_VALID_FROM, scope, _date):
        out["valid_from"] = {"value": hit[0], "source": "rule", "evidence": hit[1]}
    if hit := _first(_VALID_TO, scope, _date):
        if "valid_from" not in out or hit[0] > out["valid_from"]["value"]:
            out["valid_to"] = {"value": hit[0], "source": "rule", "evidence": hit[1]}
    if hit := _first(_DEPT, scope):
        out["department"] = {"value": hit[0], "source": "rule", "evidence": hit[1]}
    for pat, _ in _ACCESS:
        m = pat.search(text[:800])
        if m:
            out["access_level"] = {"value": _ACCESS_LEVEL.get(m.group(1), 1), "source": "rule", "evidence": _snippet(text, m)}
            break
    tags = extract_keywords(text[:20000], 5, title=title_block or file_stem, precise=False)
    if tags:
        out["tags"] = {"value": tags, "source": "tfidf", "evidence": "正文 TF-IDF 主题词"}

    if llm is not None:
        _merge_llm(out, llm, head[:3000], tail[-1200:] if tail else "", text)
    return out


def _merge_llm(out: dict[str, dict[str, Any]], llm, head: str, tail: str, text: str) -> None:
    try:
        raw = llm.complete(_LLM_PROMPT.format(head=head, tail=tail or "（同上）"), max_tokens=400)
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
    except Exception as exc:
        logger.warning("LLM 元数据抽取失败：%s", exc)
        return
    for key in ("title", "version", "valid_from", "valid_to", "department"):
        v = data.get(key)
        if not v or key in out and out[key]["source"] == "rule":
            continue
        v = str(v).strip()
        if key in ("valid_from", "valid_to"):
            try:
                d = date.fromisoformat(v[:10])
            except ValueError:
                continue
            if str(d.year) not in text:          # 没有依据的日期不要
                continue
            v = d.isoformat()
        elif key in ("version", "department") and v not in text:
            continue
        out[key] = {"value": v, "source": "llm", "evidence": "LLM 抽取（已在原文中核对）"}
    tags = data.get("tags")
    if isinstance(tags, list) and tags:
        out["tags"] = {"value": [str(t).strip() for t in tags if str(t).strip()][:6], "source": "llm", "evidence": "LLM 主题标签"}


def is_empty(meta: DocMetadata, key: str, *, file_stem: str = "") -> bool:
    v = getattr(meta, key)
    if key == "title":
        return not v or v == file_stem
    if key == "access_level":
        return not v
    return v in (None, "", [])


def apply_metadata(meta: DocMetadata, found: dict[str, dict[str, Any]], *, file_stem: str = "", fields: tuple[str, ...] = FIELDS) -> list[str]:
    """只填空字段；返回实际写入的字段名。"""
    applied: list[str] = []
    for key in fields:
        if key not in found or not is_empty(meta, key, file_stem=file_stem):
            continue
        v = found[key]["value"]
        if key in ("valid_from", "valid_to"):
            v = date.fromisoformat(v)
        setattr(meta, key, v)
        applied.append(key)
    if meta.valid_from and meta.valid_to and meta.valid_to < meta.valid_from:
        # 规则 / 模型各抽一半、互相矛盾：失效日期不可信，撤回
        meta.valid_to = None
        if "valid_to" in applied:
            applied.remove("valid_to")
    return applied
