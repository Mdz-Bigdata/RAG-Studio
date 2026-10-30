"""解析器注册表：按扩展名 + 可用依赖选择解析器。

优先级：
1. `.md` 永远走 MarkdownParser（必做，结构最干净）
2. 表格类（xlsx / csv）走 ExcelParser（行级块比整表压扁更适合检索）
3. PDF / 图片 / Office / EPUB / HTML → MinerU（若已安装且启用）
4. 兜底：docx → python-docx，pptx → python-pptx，pdf → pdfplumber / pypdf，txt → TextParser，
   图片文件 → ImageFileParser（图片进图片索引，正文用 OCR / VLM 描述）
5. 视频文件 → VideoParser（按镜头抽关键帧进图片索引，正文用关键帧描述 + 字幕；需要 ffmpeg 或 opencv）
"""

from __future__ import annotations

from pathlib import Path

from ...config import Settings
from ...multimodal.video import video_available
from .base import BaseParser
from .docx_parser import DocxParser
from .excel_parser import ExcelParser
from .image_parser import ImageFileParser
from .markdown_parser import MarkdownParser
from .mineru_parser import MINERU_EXTENSIONS, MinerUParser, mineru_available
from .pdf_parser import PdfFallbackParser
from .pptx_parser import PptxParser
from .text_parser import TextParser
from .video_parser import VideoParser

__all__ = [
    "BaseParser",
    "DocxParser",
    "ExcelParser",
    "ImageFileParser",
    "MarkdownParser",
    "MinerUParser",
    "PdfFallbackParser",
    "PptxParser",
    "TextParser",
    "VideoParser",
    "ParserRegistry",
]


class ParserRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.markdown = MarkdownParser()
        self.excel = ExcelParser()
        self.text = TextParser()
        self.docx = DocxParser()
        self.pptx = PptxParser()
        self.pdf = PdfFallbackParser()
        self.image = ImageFileParser()
        self.video = VideoParser()
        self._mineru: MinerUParser | None = None
        self.use_mineru = settings.parsing.mineru.enabled and mineru_available()

    @property
    def mineru(self) -> MinerUParser:
        if self._mineru is None:
            self._mineru = MinerUParser(self.settings.parsing.mineru, out_dir=self.settings.paths.mineru_out_dir)
        return self._mineru

    def supported_extensions(self) -> set[str]:
        exts = set(self.markdown.extensions | self.excel.extensions | self.text.extensions | self.docx.extensions | self.pptx.extensions | self.pdf.extensions)
        if self.settings.multimodal.enabled:
            exts |= set(self.image.extensions)
        if self.video_enabled:
            exts |= set(self.video.extensions)
        if self.use_mineru:
            exts |= set(MINERU_EXTENSIONS)
        return exts

    @property
    def video_enabled(self) -> bool:
        mm = self.settings.multimodal
        return mm.enabled and mm.video.enabled and video_available()

    def fallback_parser(self, path: Path) -> BaseParser | None:
        """MinerU 失败时可用的内置解析器；没有则返回 None。"""
        ext = path.suffix.lower().lstrip(".")
        for p in (self.docx, self.pptx, self.pdf):
            if ext in p.extensions:
                return p
        if ext in self.image.extensions and self.settings.multimodal.enabled:
            return self.image
        return None

    def get_parser(self, path: Path) -> BaseParser:
        ext = path.suffix.lower().lstrip(".")
        if ext in self.markdown.extensions:
            return self.markdown
        if ext in self.excel.extensions:
            return self.excel
        if ext in self.text.extensions:
            return self.text
        if self.use_mineru and ext in MINERU_EXTENSIONS:
            return self.mineru
        if ext in self.docx.extensions:
            return self.docx
        if ext in self.pptx.extensions:
            return self.pptx
        if ext in self.pdf.extensions:
            return self.pdf
        if ext in self.image.extensions and self.settings.multimodal.enabled:
            return self.image
        if ext in self.video.extensions:
            if not self.video_enabled:
                raise ValueError(f"视频 {path.name} 无法入库：需要开启多模态与视频检索（multimodal.video.enabled），并安装 ffmpeg 或 opencv-python")
            return self.video
        raise ValueError(f"不支持的文件类型：{path.name}（安装 MinerU 可扩展支持：pip install 'mineru[torch]'）")
