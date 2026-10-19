"""飞书新版文档（docx）的块 → HTML：在页面上按飞书的样子展示文档。

和 Markdown 不同，这里保留飞书的排版信息：标题级别与字号、加粗 / 斜体 / 删除线 / 下划线 / 文字颜色 / 背景色、对齐、
项目符号与有序编号（含 sequence 续号）、待办、代码块、引用 / 高亮块（背景色、边框色）、分栏、表格列宽与合并单元格、图片（原图）。
Markdown 表达不了颜色，CJK 的「**粗体：**正文」又常被 Markdown 解析器当成字面量——所以展示走这里，检索 / 智能体 / 编辑仍用 Markdown。

安全：所有文字都经过 html.escape；链接只保留 http(s) / mailto；样式值只来自下面固定的调色板和整数，不拼接外部字符串。
图片地址由调用方的 image_src(token) 给出（服务端代理下载，前端不直连飞书）。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from html import escape
from typing import Any
from urllib.parse import unquote

from .convert import (BULLET, CALLOUT, CODE, DIVIDER, GRID, GRID_COLUMN, HEADINGS, IMAGE, ORDERED, PAGE, QUOTE, QUOTE_CONTAINER, TABLE, TEXT, TODO,
                      _LANG_NAMES, _PLACEHOLDERS, elements_to_md, image_caption)

# 飞书 FontColor / FontBackgroundColor 枚举 → 颜色（与飞书网页版的取值相近）
FONT_COLORS = {1: "#D83931", 2: "#DE7802", 3: "#DC9B04", 4: "#2EA121", 5: "#245BDB", 6: "#6425D0", 7: "#8F959E"}
BG_COLORS = {1: "#FEE3E2", 2: "#FEEAD2", 3: "#FFF6C2", 4: "#D9F5D6", 5: "#E1EAFF", 6: "#ECE2FE", 7: "#EFF0F1",
             8: "#FBBFBC", 9: "#FED4A4", 10: "#FFF67A", 11: "#B7EDB1", 12: "#BACEFD", 13: "#CDB2FA", 14: "#DEE0E3", 15: "#F5F6F7"}
BORDER_COLORS = {1: "#F98E8B", 2: "#FFBA6B", 3: "#FFE928", 4: "#8EE085", 5: "#82A7FC", 6: "#AD82F7", 7: "#BBBFC4"}
ALIGN = {2: "center", 3: "right"}
EMOJI = {"bulb": "💡", "star": "⭐", "warning": "⚠️", "exclamation": "❗", "question": "❓", "white_check_mark": "✅", "x": "❌", "fire": "🔥",
         "memo": "📝", "pushpin": "📌", "rocket": "🚀", "smile": "😄", "thumbsup": "👍", "heart": "❤️", "point_right": "👉", "information_source": "ℹ️"}

ImageSrc = Callable[[str], str]


def blocks_to_html(blocks: list[dict[str, Any]], *, title: str | None = None, image_src: ImageSrc | None = None) -> str:
    by_id = {b.get("block_id"): b for b in blocks if b.get("block_id")}
    root = next((b for b in blocks if b.get("block_type") == PAGE), None)
    if title is None and root is not None:
        title = elements_to_md((root.get("page") or {}).get("elements"), plain=True).strip()
    out = [f'<h1 class="fs-title">{escape(title)}</h1>'] if title else []
    top = root.get("children") if root is not None else [b.get("block_id") for b in blocks if not b.get("parent_id")]
    out.append(_Renderer(by_id, image_src).children(top or [], depth=0))
    return '<div class="fs-doc">' + "".join(out) + "</div>"


class _Renderer:
    def __init__(self, by_id: dict[str, dict[str, Any]], image_src: ImageSrc | None) -> None:
        self.by_id = by_id
        self.image_src = image_src

    def children(self, ids: list[str], *, depth: int) -> str:
        parts, prev_no = [], 0
        for bid in ids:
            b = self.by_id.get(bid)
            if b is None:
                continue
            no = 0
            if b.get("block_type") == ORDERED:
                seq = str(((b.get("ordered") or {}).get("style") or {}).get("sequence") or "auto")
                no = int(seq) if seq.isdigit() else prev_no + 1
            prev_no = no
            parts.append(self.block(b, depth=depth, ordered_no=no))
        return "".join(parts)

    def block(self, b: dict[str, Any], *, depth: int, ordered_no: int = 0) -> str:
        t = b.get("block_type")
        kids = b.get("children") or []
        if t in HEADINGS:
            field = HEADINGS[t][0]
            data = b.get(field) or {}
            return f'<div class="fs-h fs-h{t - 2}"{_align(data)}>{inline(data.get("elements"))}</div>' + self.children(kids, depth=depth)
        if t == TEXT:
            data = b.get("text") or {}
            body = inline(data.get("elements")) or "<br>"
            return f'<p class="fs-p"{_align(data)}>{body}</p>' + self._nested(kids, depth)
        if t in (BULLET, ORDERED, TODO):
            field = {BULLET: "bullet", ORDERED: "ordered", TODO: "todo"}[t]
            data = b.get(field) or {}
            done = t == TODO and bool((data.get("style") or {}).get("done"))
            marker = _marker(t, depth, ordered_no, done)
            cls = "fs-li fs-todo-done" if done else "fs-li"
            return (f'<div class="{cls}"><span class="fs-marker{" fs-check" if t == TODO else ""}">{marker}</span>'
                    f'<div class="fs-li-body"{_align(data)}>{inline(data.get("elements")) or "<br>"}</div></div>' + self._nested(kids, depth))
        if t == CODE:
            data = b.get("code") or {}
            style = data.get("style") or {}
            lang = _LANG_NAMES.get(int(style.get("language") or 1), "")
            wrap = " fs-code-wrap" if style.get("wrap") else ""
            head = f'<div class="fs-code-lang">{escape(lang)}</div>' if lang else ""
            return f'<div class="fs-code{wrap}">{head}<pre><code>{escape(elements_to_md(data.get("elements"), plain=True))}</code></pre></div>'
        if t == QUOTE:
            return f'<blockquote class="fs-quote">{inline((b.get("quote") or {}).get("elements"))}</blockquote>'
        if t == QUOTE_CONTAINER:
            return f'<blockquote class="fs-quote">{self.children(kids, depth=0)}</blockquote>'
        if t == CALLOUT:
            data = b.get("callout") or {}
            style = []
            if (bg := BG_COLORS.get(_int(data.get("background_color")))) is not None:
                style.append(f"background:{bg}")
            if (bc := BORDER_COLORS.get(_int(data.get("border_color")))) is not None:
                style.append(f"border-color:{bc}")
            emoji = EMOJI.get(str(data.get("emoji_id") or ""), "")
            icon = f'<span class="fs-callout-emoji">{emoji}</span>' if emoji else ""
            return f'<div class="fs-callout" style="{";".join(style)}">{icon}<div class="fs-callout-body">{self.children(kids, depth=0)}</div></div>'
        if t == DIVIDER:
            return '<hr class="fs-hr">'
        if t == IMAGE:
            return self.image(b)
        if t == TABLE:
            return self.table(b)
        if t == GRID:
            return f'<div class="fs-grid">{self.children(kids, depth=0)}</div>'
        if t == GRID_COLUMN:
            ratio = _int((b.get("grid_column") or {}).get("width_ratio")) or 1
            return f'<div class="fs-grid-col" style="flex:{ratio} 1 0">{self.children(kids, depth=0)}</div>'
        if t in _PLACEHOLDERS:
            return f'<div class="fs-placeholder">[{_PLACEHOLDERS[t]}]</div>'
        return self.children(kids, depth=depth) if kids else ""

    def _nested(self, kids: list[str], depth: int) -> str:
        return f'<div class="fs-nested">{self.children(kids, depth=depth + 1)}</div>' if kids else ""

    def image(self, b: dict[str, Any]) -> str:
        data = b.get("image") or {}
        token = str(data.get("token") or "")
        caption = image_caption(b)
        align = ALIGN.get(_int(data.get("align")), "left")
        if not (token and self.image_src and re.fullmatch(r"[A-Za-z0-9_-]+", token)):
            return f'<div class="fs-placeholder">[图片{"：" + escape(caption) if caption else ""}]</div>'
        width = _int(data.get("width"))
        size = f' width="{width}"' if width else ""
        cap = f'<div class="fs-img-caption">{escape(caption)}</div>' if caption else ""
        return f'<div class="fs-img" style="text-align:{align}"><img src="{escape(self.image_src(token))}"{size} loading="lazy" alt="{escape(caption or "图片")}">{cap}</div>'

    def table(self, b: dict[str, Any]) -> str:
        table = b.get("table") or {}
        prop = table.get("property") or {}
        cols, rows = _int(prop.get("column_size")), _int(prop.get("row_size"))
        cells = table.get("cells") or b.get("children") or []
        if cols <= 0 or not cells:
            return ""
        rows = rows or -(-len(cells) // cols)
        merge = prop.get("merge_info") or []
        widths = prop.get("column_width") or []
        fixed = len(widths) >= cols and all(_int(w) > 0 for w in widths[:cols])     # 有列宽就按飞书的列宽排，放不下横向滚动（和飞书一样）
        colgroup = "".join(f'<col style="width:{_int(w)}px">' for w in widths[:cols]) if fixed else ""
        table_style = f' style="width:{sum(_int(w) for w in widths[:cols])}px"' if fixed else ""
        covered: set[tuple[int, int]] = set()
        body = []
        for r in range(rows):
            tds = []
            for c in range(cols):
                i = r * cols + c
                if (r, c) in covered or i >= len(cells):
                    continue
                m = merge[i] if i < len(merge) and isinstance(merge[i], dict) else {}
                rs, cs = max(1, _int(m.get("row_span")) or 1), max(1, _int(m.get("col_span")) or 1)
                covered.update((r + dr, c + dc) for dr in range(rs) for dc in range(cs))
                tag = "th" if (r == 0 and prop.get("header_row")) or (c == 0 and prop.get("header_column")) else "td"
                span = (f' rowspan="{rs}"' if rs > 1 else "") + (f' colspan="{cs}"' if cs > 1 else "")
                cell = self.by_id.get(cells[i]) or {}
                tds.append(f"<{tag}{span}>{self.children(cell.get('children') or [], depth=0)}</{tag}>")
            body.append("<tr>" + "".join(tds) + "</tr>")
        return f'<div class="fs-table-wrap"><table class="fs-table"{table_style}>{"<colgroup>" + colgroup + "</colgroup>" if colgroup else ""}<tbody>{"".join(body)}</tbody></table></div>'


# ---------------------------------------------------------------------------- 行内
def inline(elements: list[dict[str, Any]] | None) -> str:
    out: list[str] = []
    for el in elements or []:
        if "text_run" in el:
            run = el["text_run"] or {}
            out.append(_styled(escape(run.get("content") or "").replace("\n", "<br>"), run.get("text_element_style") or {}))
        elif "mention_doc" in el:
            m = el["mention_doc"] or {}
            out.append(_styled(_link(escape(m.get("title") or "文档"), m.get("url")), m.get("text_element_style") or {}))
        elif "mention_user" in el:
            out.append('<span class="fs-mention">@用户</span>')
        elif "equation" in el:
            out.append(f'<span class="fs-eq">{escape(((el["equation"] or {}).get("content") or "").strip())}</span>')
        elif "reminder" in el:
            out.append("⏰")
        elif "file" in el:
            out.append('<span class="fs-mention">[附件]</span>')
    return "".join(out)


def _styled(html: str, style: dict[str, Any]) -> str:
    if not html:
        return html
    if style.get("inline_code"):
        html = f"<code>{html}</code>"
    if style.get("bold"):
        html = f"<strong>{html}</strong>"
    if style.get("italic"):
        html = f"<em>{html}</em>"
    if style.get("strikethrough"):
        html = f"<s>{html}</s>"
    if style.get("underline"):
        html = f"<u>{html}</u>"
    css = []
    if (color := FONT_COLORS.get(_int(style.get("text_color")))) is not None:
        css.append(f"color:{color}")
    if (bg := BG_COLORS.get(_int(style.get("background_color")))) is not None:
        css.append(f"background:{bg}")
    if css:
        html = f'<span style="{";".join(css)}">{html}</span>'
    link = (style.get("link") or {}).get("url")
    return _link(html, link) if link else html


def _link(html: str, url: Any) -> str:
    href = unquote(str(url or "")).strip()
    if not re.match(r"^(https?://|mailto:)", href, re.I):
        return html
    return f'<a href="{escape(href)}" target="_blank" rel="noopener noreferrer">{html}</a>'


def _marker(t: int, depth: int, no: int, done: bool) -> str:
    if t == TODO:
        return "☑" if done else "☐"
    if t == BULLET:
        return ("•", "◦", "▪")[depth % 3]
    if depth % 3 == 1:
        return _letters(no) + "."
    if depth % 3 == 2:
        return _roman(no) + "."
    return f"{no}."


def _letters(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(97 + r) + s
    return s or "a"


def _roman(n: int) -> str:
    vals = ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"))
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out or "i"


def _align(data: dict[str, Any]) -> str:
    a = ALIGN.get(_int((data.get("style") or {}).get("align")))
    return f' style="text-align:{a}"' if a else ""


def _int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0
