"""覆盖缺口驱动的语料增强：从"用户会问、但库里答不好"的问题出发，补一份经过人工审核的 FAQ。

语料覆盖分析给出了"部分匹配 / 无匹配"的意图。两类缺口的处理方式不同：

- 部分匹配：知识在库里，但分散在几段里、或用词和用户问法对不上（"年假能不能攒到明年" vs "未休年假可结转至次年"）。
  让模型只根据检索到的原文写一条 FAQ——问题用用户的说法，答案用原文的事实，并附上来源段落。
  FAQ 入库后同时补上了"问法"和"答案聚合"两件事。
- 无匹配：库里确实没有。模型会回答 NO_ANSWER，条目标为"知识缺口"，需要人工补写答案（或去补充源文档）。

所有条目都要人工通过才会发布；发布时把通过的条目写成一篇 Markdown（「增强语料 · FAQ」），
像普通文档一样切片 / 索引，可随时撤回。没有文本模型时生成抽取式草稿（最相关原文的前几句），必须人工改写。
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from ..schemas import Context

NO_ANSWER = "NO_ANSWER"

_PROMPT = """你是企业知识库的编辑，正在把用户的真实提问整理成 FAQ。

<资料>
{contexts}
</资料>

用户问题：{question}

要求：
1. 只使用<资料>中明确写出的信息，不得补充常识或推测；数字、条件、例外情况要保留。
2. 资料不足以回答这个问题时，只输出 {no_answer}，不要输出其他内容。
3. 否则按下面的格式输出：
问题：<保留用户的说法，只修正错别字和口语省略>
答案：<完整、可独立阅读的答案，2~6 句话>
依据：<用到的资料编号，如 1,3>"""


def _format_contexts(contexts: list[Context]) -> str:
    return "\n\n".join(f"[{i}] 《{c.chunk.breadcrumb_text or c.chunk.doc_id}》\n{c.chunk.text[:1500]}" for i, c in enumerate(contexts, 1))


def _sources(contexts: list[Context], used: list[int] | None = None) -> list[dict[str, Any]]:
    picked = [contexts[i - 1] for i in used if 0 < i <= len(contexts)] if used else contexts[:2]
    return [{"chunkId": c.chunk.chunk_id, "docId": c.chunk.doc_id, "breadcrumb": c.chunk.breadcrumb_text, "snippet": c.chunk.text[:160]} for c in picked]


def draft_faq(question: str, contexts: list[Context], llm=None) -> dict[str, Any]:
    """返回 {question, answer, status: pending | gap, method: llm | extractive | none, sources, note}。"""
    base = {"question": question, "answer": "", "sources": [], "createdAt": datetime.now().isoformat(timespec="seconds")}
    if not contexts:
        return {**base, "status": "gap", "method": "none", "note": "检索不到任何相关语料：需要人工补写答案，或补充源文档"}
    if llm is not None:
        try:
            raw = llm.complete(_PROMPT.format(contexts=_format_contexts(contexts), question=question, no_answer=NO_ANSWER), max_tokens=700).strip()
        except Exception as exc:
            raw = ""
            base["note"] = f"模型调用失败（{exc}），改用抽取式草稿"[:200]
        if raw:
            if NO_ANSWER in raw[:40]:
                return {**base, "status": "gap", "method": "llm", "sources": _sources(contexts), "note": "模型判断现有语料不足以回答：需要人工补写答案，或补充源文档"}
            q = re.search(r"问题[:：]\s*(.+)", raw)
            a = re.search(r"答案[:：]\s*(.+?)(?:\n\s*依据[:：]|$)", raw, re.S)
            used = [int(x) for x in re.findall(r"\d+", (re.search(r"依据[:：]\s*(.+)", raw) or [None, ""])[1])]
            if a and a.group(1).strip():
                return {**base, "question": (q.group(1).strip() if q else question), "answer": a.group(1).strip(), "status": "pending", "method": "llm", "sources": _sources(contexts, used), "note": ""}
    from ..ingest.chunking.tokenizer import split_sentences   # 延迟导入：ingest 包会反过来用到 corpus

    top = contexts[0].chunk
    lead = "".join(split_sentences(top.text)[:3]) or top.text
    return {**base, "answer": lead[:400], "status": "pending", "method": "extractive", "sources": _sources(contexts[:1]),
            "note": base.get("note") or "没有文本模型：这是最相关原文的前几句，请改写成针对问题的答案后再通过"}


def build_faq_markdown(items: list[dict[str, Any]], kb_name: str = "") -> str:
    """把已通过的条目写成一篇文档：一问一节，标题就是问题（切片后 breadcrumb = 问题，检索与引用都更清楚）。"""
    lines = [f"# 增强语料 · FAQ{('（' + kb_name + '）') if kb_name else ''}", "", "> 由语料覆盖分析发现的缺口整理，经人工审核后发布。每条答案只来自知识库原文或人工补写。", ""]
    for it in items:
        lines.append(f"## {it['question'].strip()}")
        lines.append("")
        lines.append(it["answer"].strip())
        srcs = [s.get("breadcrumb") for s in it.get("sources") or [] if s.get("breadcrumb")]
        if srcs:
            lines.append("")
            lines.append("来源：" + "；".join(dict.fromkeys(srcs)))
        lines.append("")
    return "\n".join(lines).strip() + "\n"
