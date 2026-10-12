"""图片文件解析器（无 MinerU 时使用）：图片本身进图片索引，正文由录入流水线用 OCR / VLM 描述补齐。"""

from __future__ import annotations

from pathlib import Path

from ...schemas import Block, BlockType, DocMetadata, ParsedDocument
from .base import BaseParser


class ImageFileParser(BaseParser):
    name = "image"
    extensions = frozenset({"png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"})

    def parse(self, path: Path, metadata: DocMetadata | None = None) -> ParsedDocument:
        doc = self._new_document(path, metadata)
        # 标题用原始文件名（Web 上传后磁盘文件名是 uuid，标题由 metadata.title 带入）
        doc.blocks = [Block(type=BlockType.TITLE, level=1, text=doc.metadata.title or path.stem, page=0)]
        return doc
