"""Markdown 解析器（必做）。

Markdown 是 RAG 语料最理想的中间格式：标题层级天然给出章节结构，围栏代码与表格是显式原子块。
MinerU 输出的 markdown 也可以回灌到这里做二次解析。

支持：
- YAML front matter → DocMetadata（title / version / valid_from / valid_to / tags / department）
- `#` 标题（级别 1-6）、围栏代码、表格、列表、引用、段落
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml

from ...schemas import Block, BlockType, DocMetadata, ParsedDocument
from .base import BaseParser

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^(```|~~~)\s*([\w+-]*)\s*$")
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_LIST_LINE = re.compile(r"^\s*(?:[-*+]|\d+[.)]|[一二三四五六七八九十]+[、.]|[①-⑳])\s+")
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_DATA_URI = re.compile(r"data:image/[\w.+-]+;base64,[A-Za-z0-9+/=\s]+")


def strip_data_uri_images(text: str) -> str:
    """内嵌图片的 base64 不是正文：留着会撑爆切片、污染向量与 BM25。像素由图片索引单独保存，这里只留占位。"""
    return _DATA_URI.sub("embedded-image", text)


class MarkdownParser(BaseParser):
    name = "markdown"
    extensions = frozenset({"md", "markdown"})

    def parse(self, path: Path, metadata: DocMetadata | None = None) -> ParsedDocument:
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.parse_text(text, path, metadata)

    def parse_text(self, text: str, path: Path, metadata: DocMetadata | None = None) -> ParsedDocument:
        doc = self._new_document(path, metadata)
        text = strip_data_uri_images(self._apply_front_matter(text, doc.metadata))
        doc.markdown = text
        doc.blocks = parse_markdown_blocks(text)
        # 首个一级标题作为文档标题
        if not (metadata and metadata.title):
            for b in doc.blocks:
                if b.type == BlockType.TITLE and b.level == 1:
                    doc.metadata.title = b.text
                    break
        return doc

    @staticmethod
    def _apply_front_matter(text: str, md: DocMetadata) -> str:
        m = _FRONT_MATTER.match(text)
        if not m:
            return text
        try:
            data = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            return text
        if not isinstance(data, dict):
            return text
        for key in ("title", "version", "department", "language"):
            if data.get(key):
                setattr(md, key, str(data[key]))
        for key in ("valid_from", "valid_to"):
            v = data.get(key)
            if isinstance(v, date):
                setattr(md, key, v)
            elif isinstance(v, str):
                try:
                    setattr(md, key, date.fromisoformat(v))
                except ValueError:
                    pass
        tags = data.get("tags")
        if isinstance(tags, list):
            md.tags = [str(t) for t in tags]
        elif isinstance(tags, str):
            md.tags = [t.strip() for t in tags.split(",") if t.strip()]
        known = {"title", "version", "department", "language", "valid_from", "valid_to", "tags"}
        md.extra.update({k: v for k, v in data.items() if k not in known})
        return text[m.end():]


def parse_markdown_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    lines = text.splitlines()
    i, n = 0, len(lines)
    para: list[str] = []

    def flush_para() -> None:
        if para:
            blocks.append(Block(type=BlockType.TEXT, text="\n".join(para).strip()))
            para.clear()

    while i < n:
        line = lines[i]
        fence = _FENCE.match(line)
        if fence:
            flush_para()
            marker, lang = fence.group(1), fence.group(2)
            j = i + 1
            code: list[str] = []
            while j < n and not lines[j].startswith(marker):
                code.append(lines[j])
                j += 1
            blocks.append(Block(type=BlockType.CODE, text="\n".join(code), meta={"language": lang or ""}))
            i = j + 1
            continue

        heading = _HEADING.match(line)
        if heading:
            flush_para()
            blocks.append(Block(type=BlockType.TITLE, level=len(heading.group(1)), text=heading.group(2).strip()))
            i += 1
            continue

        if _TABLE_LINE.match(line):
            flush_para()
            j = i
            rows: list[str] = []
            while j < n and _TABLE_LINE.match(lines[j]):
                rows.append(lines[j].strip())
                j += 1
            blocks.append(Block(type=BlockType.TABLE, text="\n".join(rows), meta={"format": "markdown"}))
            i = j
            continue

        if _LIST_LINE.match(line):
            flush_para()
            j = i
            items: list[str] = []
            while j < n and (_LIST_LINE.match(lines[j]) or (lines[j].startswith("  ") and lines[j].strip())):
                items.append(lines[j].rstrip())
                j += 1
            blocks.append(Block(type=BlockType.LIST, text="\n".join(items)))
            i = j
            continue

        if not line.strip():
            flush_para()
            i += 1
            continue

        para.append(line.lstrip("> ").rstrip() if line.lstrip().startswith(">") else line.rstrip())
        i += 1

    flush_para()
    return blocks
