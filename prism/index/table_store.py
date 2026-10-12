"""结构化表存储 + SQL 检索路（混合检索的第三路）。

向量检索懂语义、BM25 懂关键词，但"售价高于 9000 的有哪些""库存最少的是哪个"这类按字段过滤 / 比较 / 排序的问题，
两者都答不好——这是数据库的活。做法：

- 入库：Excel / CSV 的每一行在写成文本块的同时，原样进一张 SQLite 表（列名保留原文，纯数字列转成数值）；
  每行记住它对应的 child 块，SQL 命中的行就能和向量 / BM25 的结果一起做 RRF。
- 查询：
  · 有 LLM → text-to-SQL（只给表结构与样例行；只允许单条 SELECT，SQLite authorizer 强制只读）；
  · 无 LLM → 规则兜底：问题里出现的单元格取值 → 等值过滤；"列名 + 大于 / 低于 + 数字" → 比较；"最高 / 最低 + 列名" → 排序取首行。
- 聚合类 SQL（COUNT / AVG / SUM …）没有对应的行：结果表作为一条"SQL 计算结果"上下文直接交给 LLM。

持久化为 tables.json，加载时重建内存 SQLite——表都不大，省掉文件锁与迁移。
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas import Block, Chunk

logger = logging.getLogger(__name__)

TEXT_TO_SQL_PROMPT = """你是 SQLite 查询生成器。根据表结构把用户问题翻译成**一条** SELECT 语句。

规则：
- 只输出 SQL，不要解释、不要 Markdown 代码块。
- 需要返回具体记录时用 `SELECT * FROM 表 WHERE ...`（必须保留所有列）；只有问"多少个 / 平均 / 合计 / 最大值"这类统计时才用聚合函数。
- 列名、表名按下面给出的原样使用，并用双引号括起来。
- 文本匹配优先用 LIKE '%关键词%'；数值列直接比较。
- 如果问题和这些表无关，只输出 NONE。

{schema}

用户问题：{query}
SQL："""

_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
_UNIT = re.compile(r"[（(].*?[）)]")
_GT = r"(?:大于|高于|超过|多于|不低于|不少于|至少|以上|>=?|≥)"
_LT = r"(?:小于|低于|少于|不超过|不高于|至多|最多不超过|以下|以内|<=?|≤)"
_MAX_WORDS = ("最高", "最大", "最多", "最贵", "最晚", "最新", "最长")
_MIN_WORDS = ("最低", "最小", "最少", "最便宜", "最早", "最短")
_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex)\b", re.I)


@dataclass
class TableData:
    doc_id: str
    sheet: str
    columns: list[str]
    rows: list[list[str]]
    chunk_ids: list[str | None] = field(default_factory=list)
    name: str = ""                       # SQLite 表名，入库时分配

    def to_json(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "sheet": self.sheet, "columns": self.columns, "rows": self.rows, "chunk_ids": self.chunk_ids, "name": self.name}


@dataclass
class SQLResult:
    ranking: list[tuple[str, float]] = field(default_factory=list)   # [(child chunk id, score)]
    sql: str = ""
    mode: str = ""                       # llm / rule
    table: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)          # 聚合结果（没有对应的行）
    note: str = ""

    @property
    def aggregate_text(self) -> str:
        if not self.rows:
            return ""
        cols = list(self.rows[0])
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in self.rows[:20]]
        return f"SQL：{self.sql}\n\n" + "\n".join(lines)


def tables_from_blocks(doc_id: str, blocks: list[Block]) -> list[TableData]:
    """ExcelParser 给每一行留了 meta.record / meta.sheet：按 sheet 还原成表。"""
    by_sheet: dict[str, TableData] = {}
    for b in blocks:
        record = b.meta.get("record")
        sheet = b.meta.get("sheet")
        if not isinstance(record, dict) or not sheet:
            continue
        t = by_sheet.setdefault(sheet, TableData(doc_id=doc_id, sheet=sheet, columns=[], rows=[]))
        for col in record:
            if col not in t.columns:
                t.columns.append(col)
                for r in t.rows:
                    r.append("")
        t.rows.append([str(record.get(c, "")) for c in t.columns])
        t.chunk_ids.append(b.text)       # 先暂存行文本，link_chunks 再换成 chunk id
    return [t for t in by_sheet.values() if t.rows and t.columns]


def link_chunks(tables: list[TableData], chunks: list[Chunk]) -> None:
    """行 → 它所在的 child 块（行文本是 child 文本的子串）。找不到的行不参与融合，但仍可被 SQL 查到。"""
    children = [c for c in chunks if c.kind == "child"]
    for t in tables:
        linked: list[str | None] = []
        for row_text in t.chunk_ids:
            probe = (row_text or "")[:60]
            hit = next((c.chunk_id for c in children if row_text and row_text in c.text), None) or next((c.chunk_id for c in children if probe and probe in c.text), None)
            linked.append(hit)
        t.chunk_ids = linked


class TableStore:
    def __init__(self) -> None:
        self.tables: dict[str, TableData] = {}
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = threading.RLock()
        self._seq = 0

    def __len__(self) -> int:
        return len(self.tables)

    # --- 写入 -----------------------------------------------------------------------------
    def add(self, tables: list[TableData]) -> None:
        with self._lock:
            for t in tables:
                if not t.name:
                    self._seq += 1
                    t.name = f"t{self._seq}"
                else:
                    self._seq = max(self._seq, int(t.name[1:]) if t.name[1:].isdigit() else self._seq)
                self.tables[t.name] = t
                self._create(t)

    def _create(self, t: TableData) -> None:
        numeric = [all(_NUM.match(r[i].replace(",", "")) for r in t.rows if r[i]) and any(r[i] for r in t.rows) for i in range(len(t.columns))]
        cols = ", ".join(f'{_q(c)} {"REAL" if numeric[i] else "TEXT"}' for i, c in enumerate(t.columns))
        self._conn.execute(f"DROP TABLE IF EXISTS {_q(t.name)}")
        self._conn.execute(f"CREATE TABLE {_q(t.name)} (_row INTEGER, _chunk_id TEXT, {cols})")
        marks = ", ".join("?" for _ in range(len(t.columns) + 2))
        data = []
        for n, row in enumerate(t.rows):
            vals: list[Any] = [(float(v.replace(",", "")) if numeric[i] and v else (None if numeric[i] else v)) for i, v in enumerate(row)]
            data.append([n, t.chunk_ids[n] if n < len(t.chunk_ids) else None, *vals])
        self._conn.executemany(f"INSERT INTO {_q(t.name)} VALUES ({marks})", data)
        self._conn.commit()

    def remove_document(self, doc_id: str) -> None:
        with self._lock:
            for name in [n for n, t in self.tables.items() if t.doc_id == doc_id]:
                self._conn.execute(f"DROP TABLE IF EXISTS {_q(name)}")
                del self.tables[name]
            self._conn.commit()

    # --- 查询 -----------------------------------------------------------------------------
    def schema_text(self, sample_rows: int = 3) -> str:
        parts = []
        for t in self.tables.values():
            head = ", ".join(_q(c) for c in t.columns)
            sample = "\n".join("  " + " | ".join(r) for r in t.rows[:sample_rows])
            parts.append(f'表 {_q(t.name)}（来自"{t.sheet}"，{len(t.rows)} 行）\n列：{head}\n样例行：\n{sample}')
        return "\n\n".join(parts)

    def search(self, query: str, llm=None, top_n: int = 20, allowed_ids: set[str] | None = None) -> SQLResult:
        if not self.tables or not query.strip():
            return SQLResult()
        res = self._search_llm(query, llm, top_n) if llm is not None else None
        if res is None:
            res = self._search_rule(query, top_n)
        if allowed_ids is not None:
            res.ranking = [(cid, s) for cid, s in res.ranking if cid in allowed_ids]
        return res

    def _search_llm(self, query: str, llm, top_n: int) -> SQLResult | None:
        try:
            raw = llm.complete(TEXT_TO_SQL_PROMPT.format(schema=self.schema_text(), query=query), max_tokens=400)
        except Exception as exc:
            logger.info("text-to-SQL 失败，回退规则：%s", exc)
            return None
        sql = _clean_sql(raw)
        if not sql:
            return None
        try:
            return self._execute(sql, top_n, mode="llm")
        except (sqlite3.Error, ValueError) as exc:
            logger.info("SQL 执行失败（%s），回退规则：%s", sql, exc)
            return None

    def _search_rule(self, query: str, top_n: int) -> SQLResult:
        """规则兜底：对每张表拼出过滤条件，取条件最多（最具体）的那张表。"""
        low = query.lower()
        best: tuple[int, str, str] | None = None
        for t in self.tables.values():
            conds: list[str] = []
            order = ""
            for i, col in enumerate(t.columns):
                # 取值必须是问题里的"完整词"：库存 60 不应该被 "6000" 里的 "60" 命中
                values = {r[i] for r in t.rows if len(r[i]) >= 2 and re.search(r"(?<![0-9a-z.])" + re.escape(r[i].lower()) + r"(?![0-9a-z])", low)}
                # 只保留最长匹配："iPhone 15 Pro Max" 命中时不再把 "iPhone 15" 也算进来
                values = {v for v in values if not any(v != o and v in o for o in values)}
                if values:
                    conds.append("(" + " OR ".join(f"{_q(col)} = {_lit(v)}" for v in sorted(values)) + ")")
                core = _UNIT.sub("", col).strip()
                if len(core) >= 2 and core.lower() in low:
                    esc = re.escape(core.lower())
                    m = re.search(esc + r"[^0-9<>≥≤]{0,8}?" + _GT + r"\s*(\d+(?:\.\d+)?)", low) or re.search(esc + r"\D{0,6}?(\d+(?:\.\d+)?)\D{0,3}?(?:以上|及以上)", low)
                    if m:
                        conds.append(f"{_q(col)} >= {m.group(1)}")
                    m = re.search(esc + r"[^0-9<>≥≤]{0,8}?" + _LT + r"\s*(\d+(?:\.\d+)?)", low) or re.search(esc + r"\D{0,6}?(\d+(?:\.\d+)?)\D{0,3}?(?:以下|以内)", low)
                    if m:
                        conds.append(f"{_q(col)} <= {m.group(1)}")
                    if any(w in query for w in _MAX_WORDS):
                        order = f" ORDER BY {_q(col)} DESC LIMIT 1"
                    elif any(w in query for w in _MIN_WORDS):
                        order = f" ORDER BY {_q(col)} ASC LIMIT 1"
            if not conds and not order:
                continue
            sql = f"SELECT * FROM {_q(t.name)}" + (" WHERE " + " AND ".join(conds) if conds else "") + order
            score = len(conds) * 2 + (1 if order else 0)
            if best is None or score > best[0]:
                best = (score, sql, t.name)
        if best is None:
            return SQLResult(mode="rule", note="问题里没有可用于字段过滤 / 比较 / 排序的条件，SQL 路跳过")
        try:
            return self._execute(best[1], top_n, mode="rule")
        except sqlite3.Error as exc:
            return SQLResult(mode="rule", sql=best[1], note=f"SQL 执行失败：{exc}")

    def _execute(self, sql: str, top_n: int, mode: str) -> SQLResult:
        if _FORBIDDEN.search(sql) or not re.match(r"^\s*(select|with)\b", sql, re.I) or ";" in sql.rstrip().rstrip(";"):
            raise ValueError("只允许单条 SELECT")
        sql = sql.rstrip().rstrip(";")
        with self._lock:
            self._conn.set_authorizer(_read_only)
            try:
                cur = self._conn.execute(sql)
                cols = [d[0] for d in cur.description or []]
                rows = cur.fetchmany(max(top_n, 50))
            finally:
                self._conn.set_authorizer(None)
        table = next((n for n in self.tables if re.search(rf'\b{re.escape(n)}\b', sql)), "")
        res = SQLResult(sql=sql, mode=mode, table=table)
        if "_chunk_id" in cols:
            k = cols.index("_chunk_id")
            seen: set[str] = set()
            for n, row in enumerate(rows):
                cid = row[k]
                if cid and cid not in seen:
                    seen.add(cid)
                    res.ranking.append((cid, 1.0 / (n + 1)))   # SQL 结果没有相关性分数：按返回顺序给名次分
            res.ranking = res.ranking[:top_n]
            res.note = f"{mode} → {sql} → {len(rows)} 行"
        else:
            res.rows = [dict(zip(cols, row)) for row in rows]
            res.note = f"{mode} → {sql} → 聚合结果 {len(rows)} 行"
        return res

    # --- 持久化 ---------------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        p = Path(directory) / "tables.json"
        if not self.tables and not p.exists():
            return
        Path(directory).mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([t.to_json() for t in self.tables.values()], ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "TableStore":
        store = cls()
        p = Path(directory) / "tables.json"
        if p.exists():
            store.add([TableData(**t) for t in json.loads(p.read_text(encoding="utf-8"))])
        return store


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _clean_sql(raw: str) -> str:
    text = re.sub(r"^```(?:sql)?|```$", "", (raw or "").strip(), flags=re.I | re.M).strip()
    if not text or text.upper().startswith("NONE"):
        return ""
    m = re.search(r"\b(select|with)\b.*", text, re.I | re.S)
    return m.group(0).strip() if m else ""


def _read_only(action: int, *_: Any) -> int:
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
