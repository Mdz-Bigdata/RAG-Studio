"""LanceDB 公共层：嵌入式连接复用、表的打开 / 创建、SQL 字面量、写锁、表信息（行数 / 版本 / 索引 / 存储）。

LanceDB 是嵌入式、列式、带版本的多模态数据库（Lance 格式）：没有服务进程，`lancedb.connect(目录)` 即用；
同一张表里放向量、全文、标量列和图片字节，支持向量 / 全文（BM25）/ 混合检索、SQL 预过滤与时间旅行。
本项目用它做：知识库向量后端（vector_store.backend: lancedb）、长期记忆、个性化推荐、多智能体黑板、多模态表。
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CONNS: dict[str, Any] = {}
_LOCK = threading.Lock()
_WRITE_LOCKS: dict[tuple[str, str], threading.RLock] = {}


def lancedb_available() -> bool:
    try:
        import lancedb  # noqa: F401

        return True
    except Exception:  # ImportError 或依赖损坏
        return False


def connect(path: str | Path):
    """同一目录在进程内复用一个连接。"""
    import lancedb

    key = str(Path(path).expanduser().resolve())
    with _LOCK:
        db = _CONNS.get(key)
        if db is None:
            Path(key).mkdir(parents=True, exist_ok=True)
            db = _CONNS[key] = lancedb.connect(key)
        return db


def forget_connection(path: str | Path) -> None:
    """目录被整体删除（知识库重建）后丢掉缓存的连接。"""
    with _LOCK:
        _CONNS.pop(str(Path(path).expanduser().resolve()), None)


def write_lock(db_path: str | Path, table: str) -> threading.RLock:
    """同一张表的写入串行：LanceDB 用乐观并发提交，进程内先排队可以避免提交冲突重试。"""
    key = (str(Path(db_path).expanduser().resolve()), table)
    with _LOCK:
        return _WRITE_LOCKS.setdefault(key, threading.RLock())


def open_table(db, name: str):
    """表存在就打开，否则返回 None（本地连接不支持 table_exists；table_names() 默认只列 10 张）。"""
    try:
        return db.open_table(name)
    except (ValueError, FileNotFoundError):
        return None


def ensure_table(db, name: str, schema):
    tbl = open_table(db, name)
    return tbl if tbl is not None else db.create_table(name, schema=schema, exist_ok=True)


def list_table_names(db) -> list[str]:
    try:
        return sorted(db.list_tables(limit=None).tables)
    except Exception:
        return sorted(db.table_names(limit=10_000))


def sql_str(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_in(column: str, values: Iterable[Any]) -> str:
    vals = list(dict.fromkeys(values))
    return f"{column} IN ({', '.join(sql_str(v) for v in vals)})" if vals else "false"


def has_index(tbl, column: str, kind: str | None = None) -> bool:
    for ix in tbl.list_indices():
        if column in (getattr(ix, "columns", None) or []) and (kind is None or str(getattr(ix, "index_type", "")).upper() == kind.upper()):
            return True
    return False


def table_info(tbl) -> dict[str, Any]:
    """表的概况：行数、当前版本、版本数、索引、存储大小、碎片数（给控制台页面用）。"""
    stats: dict[str, Any] = {}
    try:
        stats = tbl.stats() or {}
    except Exception:
        pass
    versions = tbl.list_versions()
    frag = stats.get("fragment_stats") or {}
    return {
        "name": tbl.name,
        "rows": tbl.count_rows(),
        "version": tbl.version,
        "versions": len(versions),
        "bytes": stats.get("total_bytes"),
        "fragments": frag.get("num_fragments"),
        "smallFragments": frag.get("num_small_fragments"),
        "indices": [{"name": ix.name, "type": str(ix.index_type), "columns": list(ix.columns)} for ix in tbl.list_indices()],
        "schema": [{"name": f.name, "type": str(f.type)} for f in tbl.schema],
    }


def version_list(tbl, limit: int = 50) -> list[dict[str, Any]]:
    out = []
    for v in tbl.list_versions()[-limit:]:
        ts = v.get("timestamp")
        if isinstance(ts, datetime):
            ts = ts.astimezone(timezone.utc).isoformat()
        out.append({"version": v.get("version"), "timestamp": ts, "metadata": v.get("metadata") or {}})
    return list(reversed(out))
