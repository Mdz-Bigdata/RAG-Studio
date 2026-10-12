"""Excel / CSV 解析器：行级结构化块。

"查表格"本质上是查数据库。把整张表压成一个块，检索时要么召回一坨乱码，要么错过具体行。
这里的做法：
- 每个 sheet 输出一个 TITLE 块（sheet 名），随后每一行输出一个 TEXT 块 `列名: 值 | 列名: 值`，
  并在 meta 里保留 sheet / row_index / 原始字典，便于精确过滤与 Function Calling 兜底；
- 同时输出一个 TABLE 块保存 Markdown 表格（表头 + 前若干行），给需要整体视图的问题用。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from ...schemas import Block, BlockType, DocMetadata, ParsedDocument
from .base import BaseParser
from .html_table import grid_to_markdown

MAX_TABLE_PREVIEW_ROWS = 50


class ExcelParser(BaseParser):
    name = "excel"
    extensions = frozenset({"xlsx", "xlsm", "xls", "csv", "tsv"})

    def parse(self, path: Path, metadata: DocMetadata | None = None) -> ParsedDocument:
        doc = self._new_document(path, metadata)
        suffix = path.suffix.lower()
        if suffix in (".csv", ".tsv"):
            # CSV 没有 sheet 名：优先用文档标题（Web 上传的文件以 uuid 落盘，stem 没有业务含义）
            name = (metadata.title if metadata and metadata.title else None) or path.stem
            sheets = {name: list(_read_csv(path, "\t" if suffix == ".tsv" else ","))}
        else:
            sheets = _read_xlsx(path)
        for sheet_name, rows in sheets.items():
            doc.blocks.extend(sheet_to_blocks(sheet_name, rows))
        return doc


def _read_csv(path: Path, delimiter: str) -> Iterable[list[str]]:
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.reader(f, delimiter=delimiter):
            yield [c.strip() for c in row]


def _read_xlsx(path: Path) -> dict[str, list[list[str]]]:
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("解析 xlsx 需要 openpyxl：pip install 'rag-agent-studio[office]'") from exc
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out: dict[str, list[list[str]]] = {}
    for ws in wb.worksheets:
        rows: list[list[str]] = []
        for row in ws.iter_rows(values_only=True):
            cells = ["" if v is None else str(v).strip() for v in row]
            if any(cells):
                rows.append(cells)
        out[ws.title] = rows
    return out


def sheet_to_blocks(sheet_name: str, rows: list[list[str]]) -> list[Block]:
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return []
    header_idx = _detect_header(rows)
    header = [h or f"列{i + 1}" for i, h in enumerate(rows[header_idx])]
    data_rows = rows[header_idx + 1:]
    blocks: list[Block] = [Block(type=BlockType.TITLE, level=1, text=f"表：{sheet_name}", meta={"sheet": sheet_name})]

    preview = [header] + [_pad(r, len(header)) for r in data_rows[:MAX_TABLE_PREVIEW_ROWS]]
    blocks.append(Block(type=BlockType.TABLE, text=grid_to_markdown(preview), meta={"sheet": sheet_name, "format": "markdown", "rows": len(data_rows)}))

    for i, row in enumerate(data_rows):
        row = _pad(row, len(header))
        record: dict[str, Any] = {h: v for h, v in zip(header, row) if v}
        if not record:
            continue
        text = f"【表：{sheet_name}】" + " | ".join(f"{h}: {v}" for h, v in record.items())
        blocks.append(Block(type=BlockType.TEXT, text=text, meta={"sheet": sheet_name, "row_index": i + header_idx + 2, "record": record}))
    return blocks


def _detect_header(rows: list[list[str]]) -> int:
    """表头 = 前 5 行中非空单元格最多且大多不是纯数字的那一行。"""
    best, best_score = 0, -1.0
    for i, r in enumerate(rows[:5]):
        non_empty = [c for c in r if c]
        if not non_empty:
            continue
        textual = sum(1 for c in non_empty if not _is_number(c))
        score = len(non_empty) + textual * 0.5
        if score > best_score:
            best, best_score = i, score
    return best


def _is_number(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def _pad(row: list[str], n: int) -> list[str]:
    return (row + [""] * n)[:n]
