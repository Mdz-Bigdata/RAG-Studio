"""黄金评估集（JSONL）。

每行一个样例：
{
  "question": "年假可以跨年使用吗",
  "relevant_doc_ids": ["<doc_id>"],            # 可选：命中该文档即相关
  "relevant_sources": ["data/raw/员工手册.md"], # 可选：按来源文件名匹配（比 doc_id 好写）
  "relevant_chunk_ids": ["<doc_id>:p0003"],     # 可选：精确到 parent
  "answer_keywords": ["次年 3 月 31 日"],        # 可选：块文本包含任一关键词即相关
  "category": "常见问题 | 长尾 | 多跳 | 精确编号 | 无答案",
  "expected_refusal": false
}
建议 ≥100 条：常见问题 50%、边缘 / 长尾 30%、多跳 / 模糊 20%，另加无答案与时效冲突样例。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ..schemas import Chunk


class GoldenExample(BaseModel):
    question: str
    relevant_doc_ids: list[str] = Field(default_factory=list)
    relevant_sources: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    answer_keywords: list[str] = Field(default_factory=list)
    category: str = "常见问题"
    expected_refusal: bool = False

    def is_relevant(self, chunk: Chunk) -> bool:
        parent_id = chunk.parent_id or chunk.chunk_id
        if self.relevant_chunk_ids and (chunk.chunk_id in self.relevant_chunk_ids or parent_id in self.relevant_chunk_ids):
            return True
        if self.answer_keywords and any(k.lower() in chunk.text.lower() for k in self.answer_keywords):
            return True
        if self.relevant_doc_ids and chunk.doc_id in self.relevant_doc_ids and not self.answer_keywords:
            return True
        if self.relevant_sources and not self.answer_keywords:
            src = str(chunk.metadata.get("source") or "")
            if any(src.endswith(s) or Path(src).name == Path(s).name for s in self.relevant_sources):
                return True
        return False

    def has_labels(self) -> bool:
        return bool(self.relevant_doc_ids or self.relevant_sources or self.relevant_chunk_ids or self.answer_keywords)


def load_golden_set(path: Path) -> list[GoldenExample]:
    out: list[GoldenExample] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(GoldenExample.model_validate_json(line))
    return out
