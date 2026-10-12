"""HTML 表格 → Markdown 表格（处理 rowspan / colspan），供 MinerU / Docx 输出使用。

为什么要转：向量模型与 BM25 对 `<td>` 标签不友好；Markdown 表格保留了"表头-单元格"的语义对齐，
且切片时可以按行拆分并重复表头，避免"缴费标准表解析成一滩乱码碎片"。
"""

from __future__ import annotations

import re
from html.parser import HTMLParser


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[dict]] = []
        self._row: list[dict] | None = None
        self._cell: dict | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = {
                "rowspan": _to_int(a.get("rowspan"), 1),
                "colspan": _to_int(a.get("colspan"), 1),
                "header": tag == "th",
            }
            self._buf = []
        elif tag == "br" and self._cell is not None:
            self._buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._cell["text"] = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._buf.append(data)


def _to_int(v: str | None, default: int) -> int:
    try:
        return max(1, int(v)) if v else default
    except ValueError:
        return default


def html_table_to_grid(html: str) -> list[list[str]]:
    """展开 rowspan / colspan 为规则二维网格（被合并的单元格填充相同文本）。"""
    p = _TableParser()
    p.feed(html)
    grid: list[list[str | None]] = []
    pending: dict[tuple[int, int], tuple[str, int]] = {}  # (row, col) -> (text, remaining rows)

    for r, row in enumerate(p.rows):
        line: list[str | None] = []
        c = 0
        cells = iter(row)
        cell = next(cells, None)
        while cell is not None or any(k[0] == r for k in pending):
            if (r, c) in pending:
                text, remain = pending.pop((r, c))
                line.append(text)
                if remain > 1:
                    pending[(r + 1, c)] = (text, remain - 1)
                c += 1
                continue
            if cell is None:
                break
            for i in range(cell["colspan"]):
                line.append(cell["text"])
                if cell["rowspan"] > 1:
                    pending[(r + 1, c + i)] = (cell["text"], cell["rowspan"] - 1)
            c += cell["colspan"]
            cell = next(cells, None)
        grid.append(line)

    width = max((len(x) for x in grid), default=0)
    return [[(x or "") for x in line] + [""] * (width - len(line)) for line in grid]


def grid_to_markdown(grid: list[list[str]]) -> str:
    if not grid:
        return ""
    esc = lambda s: s.replace("|", "\\|")
    header = grid[0]
    lines = ["| " + " | ".join(esc(x) for x in header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in grid[1:]:
        lines.append("| " + " | ".join(esc(x) for x in row) + " |")
    return "\n".join(lines)


def html_table_to_markdown(html: str) -> str:
    return grid_to_markdown(html_table_to_grid(html))


def markdown_table_rows(md: str) -> tuple[list[str], list[str]]:
    """把 Markdown 表格拆成 (表头两行, 数据行)，用于超长表格按行切分并重复表头。"""
    lines = [ln for ln in md.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return lines, []
    return lines[:2], lines[2:]
