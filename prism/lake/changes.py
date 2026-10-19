"""文档变更记录：上传解析 / 校对回灌 / 重新解析 / 恢复备份 / 删除时，在 Agent 湖里留一条 doc_event，附正文的行级变化摘要。

知识缺口要能回答"之后补过语料吗"：缺口发生后同一知识库里变更过哪些文档、改了什么，都从这里查。
"""

from __future__ import annotations

import difflib
from typing import Any

REASONS = {
    "upload": "上传解析",
    "parse": "从源文件解析",
    "reprocess": "重新解析",
    "reprocess_source": "放弃校对，从源文件重新解析",
    "reprocess_keep_edits": "保留校对，按当前配置重建",
    "edit": "校对回灌",
    "restore_backup": "恢复校对版备份",
    "markdown": "按 Markdown 重建",
    "rebuild": "整库重建",
    "delete": "删除",
    "feishu_sync": "飞书同步",
}
# 只在内容真的变了时才记录的系统动作（整库重建 / 换模型重建会对每份文档各来一次，不变就不刷屏）
QUIET_WHEN_UNCHANGED = {"parse", "markdown", "rebuild"}


def markdown_diff(before: str | None, after: str | None, *, samples: int = 5, width: int = 120) -> dict[str, Any]:
    """行级变化：新增 / 删除的行数、字数变化、前几条新增 / 删除的行（空行不计）。"""
    a = [ln for ln in (before or "").splitlines() if ln.strip()]
    b = [ln for ln in (after or "").splitlines() if ln.strip()]
    added, removed = [], []
    for line in difflib.ndiff(a, b):
        if line.startswith("+ "):
            added.append(line[2:])
        elif line.startswith("- "):
            removed.append(line[2:])
    clip = lambda xs: [x if len(x) <= width else x[:width] + "…" for x in xs[:samples]]   # noqa: E731
    return {"changed": (before or "") != (after or ""), "addedLines": len(added), "removedLines": len(removed),
            "charsBefore": len(before or ""), "charsAfter": len(after or ""), "added": clip(added), "removed": clip(removed)}


def describe_change(file_name: str, reason: str, diff: dict[str, Any], *, status: str = "ok", error: str | None = None) -> str:
    label = REASONS.get(reason, reason)
    if status != "ok":
        return f"{file_name}：{label}失败（{error or status}）"
    if reason == "delete":
        return f"{file_name}：已删除（删除前 {diff['charsBefore']} 字）"
    if not diff["changed"]:
        head = f"{file_name}：{label}，正文没有变化（{diff['charsAfter']} 字）"
    else:
        head = f"{file_name}：{label}，+{diff['addedLines']} 行 / -{diff['removedLines']} 行（{diff['charsBefore']} → {diff['charsAfter']} 字）"
    parts = [head]
    if diff["added"]:
        parts.append("新增：" + " ｜ ".join(diff["added"]))
    if diff["removed"]:
        parts.append("删除：" + " ｜ ".join(diff["removed"]))
    return "\n".join(parts)
