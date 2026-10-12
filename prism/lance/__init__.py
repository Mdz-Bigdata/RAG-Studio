"""LanceDB 集成：知识库向量后端、长期记忆、个性化推荐、多智能体黑板、多模态表共用的嵌入式多模态数据库。"""

from __future__ import annotations

import os

# Lance 的 Rust 日志默认打 WARN（例如"select 没带 _distance 列"的弃用提示），服务日志里只留错误。
# 要在第一次 import lancedb 之前设置（本包内对 lancedb 的导入都是延迟的）。
os.environ.setdefault("LANCEDB_LOG", "error")
os.environ.setdefault("LANCE_LOG", "error")

from .db import connect, ensure_table, lancedb_available, list_table_names, open_table, sql_in, sql_str, table_info, version_list, write_lock  # noqa: E402
from .vector_store import LanceDBVectorStore  # noqa: E402

__all__ = [
    "LanceDBVectorStore",
    "connect",
    "ensure_table",
    "lancedb_available",
    "list_table_names",
    "open_table",
    "sql_in",
    "sql_str",
    "table_info",
    "version_list",
    "write_lock",
]
