"""用 Agent 湖做检索增强：召回与当前问题相似的历史问答，作为"参考信息"交给回答模型。

只取可信的历史回答：状态 ok（非拒答 / 非失败）、没有被点踩；同一会话的问答已经在多轮历史里，不重复带入。
它不进 <资料>、不编号、不可引用——结论与引用仍以本轮检索到的资料为准；本轮没有检索到资料时也不带（避免历史回答变成事实来源）。
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from ..lance import sql_in, sql_str
from .store import AgentLake

_CITE = re.compile(r"\[(?:图)?\d+\]")


def recall_experience(lake: AgentLake, query: str, *, kb_id: str | None = None, session_id: str | None = None, top_k: int = 2, min_score: float = 0.75) -> list[dict[str, Any]]:
    if not query.strip() or not lake.exists():
        return []
    where = lake.where_clause(kinds=["question"], extra=f"session_id != {sql_str(session_id)}" if session_id else None)
    if kb_id:
        where += f" AND (kb_id = {sql_str(kb_id)} OR kb_id = '')"
    hits = [h for h in lake.search(query, top_k=top_k * 4, mode="vector", where=where) if (h.get("vectorScore") or 0) >= min_score]
    if not hits:
        return []
    tbl = lake._table()
    ids = [h["id"] for h in hits]
    answers = tbl.search().where(f"kind = 'answer' AND status = 'ok' AND feedback >= 0 AND {sql_in('parent_id', ids)}").select(["id", "parent_id", "content", "feedback", "ts"]).limit(len(ids) * 2).to_list()
    by_q = {}
    for a in answers:
        if a["content"].strip() and (a["parent_id"] not in by_q or a["ts"] > by_q[a["parent_id"]]["ts"]):
            by_q[a["parent_id"]] = a
    out, seen = [], set()
    for h in hits:
        a = by_q.get(h["id"])
        key = h["content"].strip()
        if a is None or key in seen:
            continue
        seen.add(key)
        out.append({"questionId": h["id"], "question": h["content"], "answerId": a["id"], "answer": _CITE.sub("", a["content"]).strip(),
                    "score": h["vectorScore"], "feedback": a["feedback"], "ts": h["ts"]})
        if len(out) >= top_k:
            break
    return out


def format_experience(items: list[dict[str, Any]], max_chars: int = 600) -> str:
    if not items:
        return ""
    lines = ["<历史问答参考>",
             "以下是 Agent 湖里与本问题相似的历史问答，不是知识库资料：只用来理解提问意图、补充回答角度；"
             "事实、数字与引用必须以 <资料> 为准，与资料冲突时以资料为准，不要用 [n] 引用这里的内容。"]
    for i, it in enumerate(items, start=1):
        when = datetime.fromtimestamp(it["ts"]).strftime("%Y-%m-%d")
        lines.append(f"Q{i}（相似度 {it['score']:.2f}，{when}{'，用户点赞' if it['feedback'] > 0 else ''}）：{it['question']}")
        ans = it["answer"]
        lines.append(f"A{i}：{ans[:max_chars]}{'…' if len(ans) > max_chars else ''}")
    lines.append("</历史问答参考>")
    return "\n".join(lines)
