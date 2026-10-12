"""纯文本解析器：按空行拆段。"""

from __future__ import annotations

from pathlib import Path

from ...schemas import DocMetadata, ParsedDocument
from .base import BaseParser, split_paragraphs, text_blocks_from_paragraphs


class TextParser(BaseParser):
    name = "text"
    extensions = frozenset({"txt", "log", "rst"})

    def parse(self, path: Path, metadata: DocMetadata | None = None) -> ParsedDocument:
        doc = self._new_document(path, metadata)
        text = path.read_text(encoding="utf-8", errors="replace")
        doc.blocks = text_blocks_from_paragraphs(split_paragraphs(text))
        return doc
