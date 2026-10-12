"""PDF 兜底解析器（无 MinerU 时使用）。

优先 pdfplumber：逐页提取文本 + 表格（表格转 Markdown，避免"缴费标准表解析成一滩乱码"），其次 pypdf。
这是"能用"级别——没有版面分析、扫描件拿不到文本。生产环境请安装 MinerU：`pip install 'mineru[torch]'`。
"""

from __future__ import annotations

import logging
from pathlib import Path

from ...schemas import Block, BlockType, DocMetadata, ParsedDocument
from .base import BaseParser, split_paragraphs, text_blocks_from_paragraphs
from .html_table import grid_to_markdown


class PdfFallbackParser(BaseParser):
    name = "pdfplumber"
    extensions = frozenset({"pdf"})

    def parse(self, path: Path, metadata: DocMetadata | None = None) -> ParsedDocument:
        doc = self._new_document(path, metadata)
        try:
            import pdfplumber
        except ImportError:
            pdfplumber = None
        if pdfplumber is not None:
            # pdfminer 对每个缺 FontBBox 的字体都打一行 warning（真实 PDF 上一次几十行），不影响抽取
            logging.getLogger("pdfminer").setLevel(logging.ERROR)
            with pdfplumber.open(str(path)) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    tables = []
                    try:
                        tables = page.find_tables()
                    except Exception:
                        tables = []
                    text = page.extract_text() or ""
                    doc.blocks.extend(text_blocks_from_paragraphs(split_paragraphs(text), page=page_idx))
                    for t in tables:
                        try:
                            data = t.extract()
                        except Exception:
                            continue
                        grid = [[str(c or "").replace("\n", " ").strip() for c in row] for row in (data or []) if row]
                        if grid and any(any(c for c in row) for row in grid):
                            doc.blocks.append(Block(type=BlockType.TABLE, text=grid_to_markdown(grid), page=page_idx, meta={"format": "markdown"}))
            return doc
        doc.metadata.parser = "pypdf"
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("解析 PDF 需要 MinerU、pdfplumber 或 pypdf") from exc
        reader = PdfReader(str(path))
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            doc.blocks.extend(text_blocks_from_paragraphs(split_paragraphs(text), page=page_idx))
        return doc
