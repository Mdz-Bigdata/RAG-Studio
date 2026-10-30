"""视频文件解析器：视频没有文字层，正文由录入流水线用每个镜头关键帧的 OCR / VLM 描述 / 字幕补齐（见 multimodal/video.py）。"""

from __future__ import annotations

from pathlib import Path

from ...multimodal.video import VIDEO_EXTENSIONS, probe
from ...schemas import Block, BlockType, DocMetadata, ParsedDocument
from .base import BaseParser


class VideoParser(BaseParser):
    name = "video"
    extensions = VIDEO_EXTENSIONS

    def parse(self, path: Path, metadata: DocMetadata | None = None) -> ParsedDocument:
        doc = self._new_document(path, metadata)
        info = probe(path)
        doc.metadata.extra["video"] = {"duration": round(info.duration, 2), "width": info.width, "height": info.height, "fps": round(info.fps, 2)}
        doc.blocks = [Block(type=BlockType.TITLE, level=1, text=doc.metadata.title or path.stem)]
        return doc
