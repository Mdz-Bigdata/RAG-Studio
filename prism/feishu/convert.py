"""飞书新版文档（docx）的块 ⇄ Markdown。

读：blocks_to_markdown() 把 list_blocks 返回的平铺块按父子关系还原成 Markdown——标题层级、列表嵌套、代码、引用 / 高亮块、
待办、表格都保留，切片时面包屑与表格结构不丢；图片 / 画板 / 内嵌表格等非文字块留一个占位，说明那里有什么。
写：markdown_to_blocks() 把 Markdown 转成可以直接交给 children 接口的块（标题 / 段落 / 列表 / 代码 / 引用 / 待办 / 分割线，
行内粗体 / 斜体 / 删除线 / 行内代码 / 链接）。表格写成每行一段的文本——要完整表格请用官方 convert 接口（FeishuClient.convert_markdown）。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, unquote

# block_type → (字段名, Markdown 前缀)
HEADINGS = {3 + i: (f"heading{i + 1}", "#" * min(6, i + 1)) for i in range(9)}
TEXT, BULLET, ORDERED, CODE, QUOTE, TODO, CALLOUT, DIVIDER, IMAGE, TABLE, TABLE_CELL, QUOTE_CONTAINER = 2, 12, 13, 14, 15, 17, 19, 22, 27, 31, 32, 34
PAGE, GRID, GRID_COLUMN = 1, 24, 25
_PLACEHOLDERS = {18: "多维表格", 20: "会话卡片", 21: "画板", 23: "附件", 26: "内嵌网页", 27: "图片", 28: "第三方块", 29: "思维笔记", 30: "电子表格", 33: "视图", 43: "画板"}

# 代码语言：飞书 CodeLanguage 枚举里常用的几种；其余按 PlainText
CODE_LANGS = {"plaintext": 1, "text": 1, "bash": 7, "csharp": 8, "c#": 8, "cpp": 9, "c++": 9, "c": 10, "css": 12, "dockerfile": 18, "go": 22, "html": 24,
              "json": 28, "java": 29, "javascript": 30, "js": 30, "kotlin": 32, "markdown": 39, "md": 39, "php": 43, "python": 49, "py": 49, "ruby": 52,
              "rust": 53, "sql": 56, "shell": 60, "sh": 60, "swift": 61, "typescript": 63, "ts": 63, "xml": 66, "yaml": 67, "yml": 67}
_LANG_NAMES = {v: k for k, v in reversed(list(CODE_LANGS.items())) if v != 1}

_TEXT_FIELDS = ["text", "bullet", "ordered", "code", "quote", "todo", "page"] + [f"heading{i}" for i in range(1, 10)]


# ======================================================================================
# 块 → Markdown
# ======================================================================================
def elements_to_md(elements: list[dict[str, Any]] | None, *, plain: bool = False) -> str:
    out: list[str] = []
    for el in elements or []:
        if "text_run" in el:
            run = el["text_run"] or {}
            text = run.get("content") or ""
            if plain or not text.strip():
                out.append(text)
                continue
            style = run.get("text_element_style") or {}
            core, lead, tail = text.strip(), text[: len(text) - len(text.lstrip())], text[len(text.rstrip()):]
            if style.get("inline_code"):
                core = f"`{core}`"
            else:
                if style.get("bold"):
                    core = f"**{core}**"
                if style.get("italic"):
                    core = f"*{core}*"
                if style.get("strikethrough"):
                    core = f"~~{core}~~"
            link = (style.get("link") or {}).get("url")
            if link:
                core = f"[{core}]({unquote(link)})"
            out.append(lead + core + tail)
        elif "mention_doc" in el:
            m = el["mention_doc"] or {}
            title = m.get("title") or "文档"
            out.append(title if plain or not m.get("url") else f"[{title}]({unquote(m['url'])})")
        elif "mention_user" in el:
            out.append("@用户")
        elif "equation" in el:
            content = ((el["equation"] or {}).get("content") or "").strip()
            out.append(content if plain else f"${content}$")
        elif "reminder" in el:
            out.append("⏰")
        elif "file" in el:
            out.append("[附件]")
    return "".join(out)


def block_text(block: dict[str, Any]) -> str:
    """块里的纯文字（没有文字的块返回空串）。"""
    for field in _TEXT_FIELDS:
        if field in block and isinstance(block[field], dict):
            return elements_to_md(block[field].get("elements"), plain=True)
    return ""


ImageRenderer = Callable[[dict[str, Any], str], str | None]   # (图片块, 图注) → 替代占位的 Markdown；返回 None 用默认占位


def image_caption(block: dict[str, Any]) -> str:
    caption = (block.get("image") or {}).get("caption")
    return elements_to_md(caption.get("elements"), plain=True).strip() if isinstance(caption, dict) else ""


def blocks_to_markdown(blocks: list[dict[str, Any]], *, title: str | None = None, image: ImageRenderer | None = None) -> str:
    """image：图片块的渲染回调（下载图片做 OCR / 看图描述由调用方负责，这里保持纯函数）。"""
    by_id = {b.get("block_id"): b for b in blocks if b.get("block_id")}
    root = next((b for b in blocks if b.get("block_type") == PAGE), None)
    lines: list[str] = []
    if title is None and root is not None:
        title = elements_to_md((root.get("page") or {}).get("elements"), plain=True).strip()
    if title:
        lines += [f"# {title}", ""]
    top = root.get("children") if root is not None else [b.get("block_id") for b in blocks if not b.get("parent_id")]
    _render_children(top or [], by_id, lines, indent=0, image=image)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def _render_children(ids: list[str], by_id: dict[str, dict[str, Any]], lines: list[str], *, indent: int, image: ImageRenderer | None = None) -> None:
    ordered_no = 0
    for bid in ids:
        b = by_id.get(bid)
        if b is None:
            continue
        ordered_no = ordered_no + 1 if b.get("block_type") == ORDERED else 0
        _render(b, by_id, lines, indent=indent, ordered_no=ordered_no, image=image)


def _render(b: dict[str, Any], by_id: dict[str, dict[str, Any]], lines: list[str], *, indent: int, ordered_no: int, image: ImageRenderer | None = None) -> None:
    t = b.get("block_type")
    pad = "  " * indent
    children = b.get("children") or []
    if t in HEADINGS:
        field, hashes = HEADINGS[t]
        text = elements_to_md((b.get(field) or {}).get("elements")).strip()
        if text:
            lines += [f"{hashes} {text}", ""]
        _render_children(children, by_id, lines, indent=indent, image=image)
    elif t == TEXT:
        text = elements_to_md((b.get("text") or {}).get("elements"))
        if text.strip():
            lines += [pad + text.strip(), ""] if indent == 0 else [pad + text.strip()]
        _render_children(children, by_id, lines, indent=indent, image=image)
    elif t in (BULLET, ORDERED, TODO):
        field = {BULLET: "bullet", ORDERED: "ordered", TODO: "todo"}[t]
        data = b.get(field) or {}
        mark = "- " if t == BULLET else f"{ordered_no}. " if t == ORDERED else ("- [x] " if (data.get("style") or {}).get("done") else "- [ ] ")
        lines.append(pad + mark + elements_to_md(data.get("elements")).strip())
        _render_children(children, by_id, lines, indent=indent + 1, image=image)
        if indent == 0:
            lines.append("")
    elif t == CODE:
        data = b.get("code") or {}
        lang = _LANG_NAMES.get(int((data.get("style") or {}).get("language") or 1), "")
        lines += [f"```{lang}", elements_to_md(data.get("elements"), plain=True).rstrip("\n"), "```", ""]
    elif t == QUOTE:
        text = elements_to_md((b.get("quote") or {}).get("elements")).strip()
        lines += [f"> {ln}" for ln in text.splitlines() or [""]] + [""]
    elif t in (CALLOUT, QUOTE_CONTAINER):
        inner: list[str] = []
        _render_children(children, by_id, inner, indent=0, image=image)
        body = list(inner)
        while body and not body[-1].strip():
            body.pop()
        lines += [f"> {ln}".rstrip() for ln in body] + [""]
    elif t == DIVIDER:
        lines += ["---", ""]
    elif t == TABLE:
        lines += _table(b, by_id) + [""]
    elif t in (GRID, GRID_COLUMN):
        _render_children(children, by_id, lines, indent=indent, image=image)
    elif t in _PLACEHOLDERS:
        name = _PLACEHOLDERS[t]
        caption = image_caption(b) if t == IMAGE else ""
        custom = image(b, caption) if image is not None and t == IMAGE else None
        if custom:
            lines += [ln if indent == 0 else pad + ln for ln in custom.strip("\n").splitlines()] + [""]
        else:
            lines += [f"[{name}{'：' + caption if caption else ''}]", ""]
    elif children:
        _render_children(children, by_id, lines, indent=indent, image=image)


def _table(b: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> list[str]:
    table = b.get("table") or {}
    prop = table.get("property") or {}
    cols = int(prop.get("column_size") or 0)
    cells = table.get("cells") or b.get("children") or []
    if cols <= 0 or not cells:
        return []

    def cell_text(cid: str) -> str:
        cell = by_id.get(cid) or {}
        parts = [block_text(by_id[c]).strip() for c in cell.get("children") or [] if c in by_id]
        return " ".join(p for p in parts if p).replace("|", "\\|").replace("\n", " ")

    rows = [[cell_text(c) for c in cells[i:i + cols]] for i in range(0, len(cells), cols)]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(" --- " for _ in range(cols)) + "|"]
    out += ["| " + " | ".join(r + [""] * (cols - len(r))) + " |" for r in rows[1:]]
    return out


# ======================================================================================
# Markdown → 块
# ======================================================================================
_INLINE = re.compile(r"(`[^`]+`)|(\*\*[^*]+\*\*)|(~~[^~]+~~)|(\*[^*\s][^*]*\*)|(\[[^\]]+\]\([^)\s]+\))")


def md_inline(text: str) -> list[dict[str, Any]]:
    """行内 Markdown → text_run 列表（粗体 / 斜体 / 删除线 / 行内代码 / 链接）。"""
    elements: list[dict[str, Any]] = []

    def run(content: str, **style: Any) -> None:
        if content:
            elements.append({"text_run": {"content": content, "text_element_style": style} if style else {"content": content}})

    pos = 0
    for m in _INLINE.finditer(text):
        run(text[pos:m.start()])
        tok = m.group(0)
        if m.group(1):
            run(tok[1:-1], inline_code=True)
        elif m.group(2):
            run(tok[2:-2], bold=True)
        elif m.group(3):
            run(tok[2:-2], strikethrough=True)
        elif m.group(4):
            run(tok[1:-1], italic=True)
        else:
            label, url = re.match(r"\[([^\]]+)\]\(([^)\s]+)\)", tok).groups()
            run(label, link={"url": quote(url, safe="")})
        pos = m.end()
    run(text[pos:])
    return elements or [{"text_run": {"content": ""}}]


def _text_block(block_type: int, field: str, text: str, **style: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"elements": md_inline(text)}
    if style:
        body["style"] = style
    return {"block_type": block_type, field: body}


def markdown_to_blocks(markdown: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    lines = (markdown or "").replace("\r\n", "\n").split("\n")
    i = 0
    para: list[str] = []

    def flush() -> None:
        if para:
            blocks.append(_text_block(TEXT, "text", "\n".join(s.strip() for s in para)))   # 段内换行保留（同飞书 Shift+Enter），不并成一行
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if i == 0 and stripped == "---":                       # YAML front matter：不写进飞书正文
            end = next((j for j in range(1, len(lines)) if lines[j].strip() == "---"), None)
            if end is not None:
                i = end + 1
                continue
        fence = re.match(r"^\s*(```|~~~)\s*([\w#+.-]*)", line)
        if fence:
            flush()
            lang, body = fence.group(2).lower(), []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence.group(1)):
                body.append(lines[i])
                i += 1
            i += 1
            blocks.append({"block_type": CODE, "code": {"elements": [{"text_run": {"content": "\n".join(body)}}], "style": {"language": CODE_LANGS.get(lang, 1), "wrap": False}}})
            continue
        if not stripped:
            flush()
            i += 1
            continue
        m = re.match(r"^(#{1,9})\s+(.*)$", stripped)
        if m:
            flush()
            level = len(m.group(1))
            blocks.append(_text_block(2 + level, f"heading{level}", m.group(2).strip().rstrip("#").strip()))
            i += 1
            continue
        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            flush()
            blocks.append({"block_type": DIVIDER, "divider": {}})
            i += 1
            continue
        m = re.match(r"^[-*+]\s+\[([ xX])\]\s+(.*)$", stripped)
        if m:
            flush()
            blocks.append(_text_block(TODO, "todo", m.group(2), done=m.group(1).lower() == "x"))
            i += 1
            continue
        m = re.match(r"^[-*+]\s+(.*)$", stripped)
        if m:
            flush()
            blocks.append(_text_block(BULLET, "bullet", m.group(1)))
            i += 1
            continue
        m = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if m:
            flush()
            blocks.append(_text_block(ORDERED, "ordered", m.group(1)))
            i += 1
            continue
        if stripped.startswith(">"):
            flush()
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip()[1:].strip())
                i += 1
            blocks.append(_text_block(QUOTE, "quote", "\n".join(q for q in quote_lines if q)))
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            flush()
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = lines[i].strip()
                if not re.fullmatch(r"\|[\s:|-]+\|", row):        # 跳过 |---|---| 分隔行
                    cells = [c.strip() for c in row.strip("|").split("|")]
                    blocks.append(_text_block(TEXT, "text", " ｜ ".join(cells)))
                i += 1
            continue
        para.append(stripped)
        i += 1
    flush()
    return blocks


def has_table(markdown: str) -> bool:
    return bool(re.search(r"^\s*\|.*\|\s*$\n^\s*\|[\s:|-]+\|\s*$", markdown or "", re.M))


# ======================================================================================
# 定点改字
# ======================================================================================
def replace_in_elements(elements: list[dict[str, Any]], find: str, replace: str) -> tuple[list[dict[str, Any]], int]:
    """在 text_run 里做替换（保留每段的样式）；跨样式边界的匹配不处理。返回 (新 elements, 替换次数)。"""
    out, n = [], 0
    for el in elements or []:
        if "text_run" in el and find in ((el["text_run"] or {}).get("content") or ""):
            run = dict(el["text_run"])
            n += run["content"].count(find)
            run["content"] = run["content"].replace(find, replace)
            out.append({**el, "text_run": run})
        else:
            out.append(el)
    return out, n


def text_field(block: dict[str, Any]) -> str | None:
    for field in _TEXT_FIELDS:
        if field != "page" and isinstance(block.get(field), dict):
            return field
    return None
