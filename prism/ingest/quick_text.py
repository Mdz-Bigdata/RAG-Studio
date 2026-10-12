"""临时文档文本提取：不入库、不建向量，只供单次会话分析（聊天附件）。"""

from __future__ import annotations

from pathlib import Path

from ..config import Settings
from .cleaning import clean_document
from .parsers import ParserRegistry


def extract_text(path: Path, settings: Settings, max_chars: int = 20000) -> str:
    registry = ParserRegistry(settings)
    doc = registry.get_parser(path).parse(path)
    doc = clean_document(doc, settings.cleaning)
    text = doc.to_markdown()
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n…（已截断，原文 {len(text)} 字符）"
    return text
