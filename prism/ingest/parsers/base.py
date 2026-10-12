"""解析器基类与公共工具。"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from ...schemas import Block, BlockType, DocMetadata, ParsedDocument


class BaseParser(ABC):
    name: str = "base"
    extensions: frozenset[str] = frozenset()

    def supports(self, path: Path) -> bool:
        return path.suffix.lower().lstrip(".") in self.extensions

    @abstractmethod
    def parse(self, path: Path, metadata: DocMetadata | None = None) -> ParsedDocument: ...

    def _new_document(self, path: Path, metadata: DocMetadata | None) -> ParsedDocument:
        md = metadata.model_copy() if metadata else DocMetadata()
        md.source = md.source or str(path)
        md.file_format = md.file_format or path.suffix.lower().lstrip(".")
        md.parser = self.name
        if not md.title:
            md.title = path.stem
        return ParsedDocument(doc_id=ParsedDocument.make_doc_id(str(path.resolve())), metadata=md)


_PARA_SPLIT = re.compile(r"\n\s*\n+")


def split_paragraphs(text: str) -> list[str]:
    """按空行拆段，保留段内换行（交给清洗阶段处理 PDF 断行）。"""
    return [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]


def text_blocks_from_paragraphs(paragraphs: list[str], page: int | None = None) -> list[Block]:
    return [Block(type=BlockType.TEXT, text=p, page=page) for p in paragraphs]
