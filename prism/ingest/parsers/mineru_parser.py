"""MinerU 4.x 解析器（PDF / 图片 / Office / EPUB / OFD / HTML ...）。

MinerU 4.0 用四个质量档位取代了 3.x 的 pipeline/vlm/hybrid 后端：
- flash     原生 PDF 文本、Office 等数字文档，不加载推理模型 → 索引构建首选，速度最快
- basic     ONNX/Torch 小模型做 OCR / 公式 / 表格，CPU 可跑
- standard  小模型 + VLM，复杂版面（默认）
- advanced  更多推理算力，疑难文档

安装（Python 3.10-3.14）：
    uv pip install -U "mineru[torch]>=4.0,<5"     # Torch 小模型 + 更高吞吐 VLM；Apple Silicon 自动含 torch
    mineru-kit models download --tier standard    # 下载 MinerU-4 模型包
    mineru-kit api-server --port 8000 --tier standard   # 可选：作为服务部署，本模块通过 MinerUApiParser 连接

SDK（本模块使用的公开接口，来自 mineru.parser / mineru.render）：
    from mineru.parser import parse, MinerUApiParser, ParseResult
    from mineru.render import render_content_list          # Content List V1（扁平，带 text_level / page_idx / bbox）
    result = parse(path, tier="standard", ocr_mode="auto", image_analysis=True, page_range="")
    result.markdown() / result.to_json() / result.structured_content()
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from ...config import MinerUConfig
from ...gpu import auto_device, mps_guard
from ...schemas import Block, BlockType, DocMetadata, ParsedDocument
from .base import BaseParser
from .html_table import html_table_to_markdown
from .markdown_parser import strip_data_uri_images

logger = logging.getLogger(__name__)

MINERU_EXTENSIONS = frozenset(
    {
        "pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff",
        "doc", "docx", "ppt", "pptx", "xls", "xlsx", "rtf", "odt", "ods", "odp",
        "epub", "ofd", "html", "htm", "mhtml",
    }
)


def mineru_available() -> bool:
    try:
        import mineru.parser  # noqa: F401
        return True
    except Exception:  # ImportError 或依赖损坏
        return False


class MinerUParser(BaseParser):
    name = "mineru"
    extensions = MINERU_EXTENSIONS

    def __init__(self, cfg: MinerUConfig, out_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.out_dir = out_dir
        self._api_parser: Any = None

    # ----------------------------------------------------------------------------------
    def parse(self, path: Path, metadata: DocMetadata | None = None) -> ParsedDocument:
        result = self._run(path)
        from mineru.render import render_content_list

        content_list: list[dict[str, Any]] = render_content_list(result.middle_json)
        markdown = result.markdown()

        doc = self._new_document(path, metadata)
        doc.metadata.parser_tier = self.cfg.tier
        doc.markdown = strip_data_uri_images(markdown)   # 内嵌图片的 base64 不进正文；像素由图片索引保存
        doc.blocks = content_list_to_blocks(content_list)
        doc.metadata.extra["mineru_content_list"] = content_list  # 供 MinerU-Popo 桥接使用

        if self.cfg.save_outputs and self.out_dir is not None:
            self._save_outputs(doc.doc_id, path, result, content_list, markdown)
        return doc

    def _run(self, path: Path):
        if self.cfg.model_source:
            os.environ["MINERU_MODEL_SOURCE"] = self.cfg.model_source
        page_range = self.cfg.page_range if path.suffix.lower() == ".pdf" else ""
        if self.cfg.api_url:
            from mineru.parser import MinerUApiParser

            if self._api_parser is None:
                self._api_parser = MinerUApiParser(
                    api_url=self.cfg.api_url,
                    api_key=self.cfg.api_key,
                    tier=self.cfg.tier,
                    include_images=False,
                )
            return self._api_parser.parse(str(path), page_range=page_range)

        from mineru.parser import parse

        # 本地解析的版面 / OCR 小模型在本进程里用 torch 跑（MINERU_DEVICE_MODE 未设时有 MPS 就用 MPS），
        # 必须和向量 / 重排模型共用 GPU 锁（见 prism/gpu.py）。代价：解析大 PDF 期间，同进程的检索要等它跑完
        with mps_guard(os.environ.get("MINERU_DEVICE_MODE") or auto_device()):
            return parse(
                str(path),
                tier=self.cfg.tier,
                ocr_mode=self.cfg.ocr_mode,
                image_analysis=self.cfg.image_analysis,
                page_range=page_range,
            )

    def _save_outputs(self, doc_id: str, path: Path, result: Any, content_list: list[dict[str, Any]], markdown: str) -> None:
        d = self.out_dir / doc_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "markdown.md").write_text(markdown, encoding="utf-8")
        (d / "content_list.json").write_text(json.dumps(content_list, ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "middle_json.json").write_text(result.to_json(), encoding="utf-8")
        (d / "source.txt").write_text(str(path.resolve()), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Content List V1 → Block
# --------------------------------------------------------------------------------------
# MinerU 4.0.x 的页面辅助块类型是 header / footer / page_number / aside_text（mineru.types.BlockType）；
# page_header / page_footer / page_aside_text 是早期文档里的叫法，一并兼容。
_PAGE_AUX = {
    "header": BlockType.HEADER, "page_header": BlockType.HEADER,
    "footer": BlockType.FOOTER, "page_footer": BlockType.FOOTER,
    "page_number": BlockType.PAGE_NUMBER,
    "aside_text": BlockType.OTHER, "page_aside_text": BlockType.OTHER,
}
_INTERNAL_LINK = re.compile(r"\[([^\]]+)\]\(#[^)]*\)")
_DOUBLE_DASH = re.compile(r"^(\s*)-\s+-\s+")
_LIST_PREFIX = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def _list_line(item: Any) -> str:
    """列表项统一成 `- xxx`：index 的条目自带缩进和 `- `（可能还有内部链接），list 的条目不带前缀。"""
    line = _INTERNAL_LINK.sub(r"\1", str(item)).rstrip()
    # standard 档（VLM）会把原文的项目符号也转写成 `-`：`- - 第1集…` 折叠成一个标记
    line = _DOUBLE_DASH.sub(r"\1- ", line)
    return line if _LIST_PREFIX.match(line) else f"- {line.strip()}"


def content_list_to_blocks(items: list[dict[str, Any]]) -> list[Block]:
    """把 MinerU Content List V1 转成统一 Block（保留页码 / bbox / 标题层级）。"""
    blocks: list[Block] = []
    for it in items:
        t = it.get("type")
        page = it.get("page_idx")
        bbox = it.get("bbox")
        meta: dict[str, Any] = {"mineru_type": t}
        if t == "text":
            text = (it.get("text") or "").strip()
            if not text:
                continue
            if it.get("text_level"):
                # flash 档把 Word 标题渲染成 **加粗**：去掉 Markdown 强调符号，面包屑才干净
                title = text.strip("*_ ").strip() or text
                blocks.append(Block(type=BlockType.TITLE, level=int(it["text_level"]), text=title, page=page, bbox=bbox, meta=meta))
            else:
                blocks.append(Block(type=BlockType.TEXT, text=text, page=page, bbox=bbox, meta=meta))
        elif t in ("list", "index"):
            # index 是版面模型眼里的"目录"，但课程大纲、章节清单这类正文列表也会被归进来——
            # 丢掉它等于丢掉整页内容（真实 PDF 上踩到过），所以按列表保留；真正的目录页只是多几条短行。
            items_ = [x for x in it.get("list_items") or [] if str(x).strip()]
            if items_:
                meta["sub_type"] = it.get("sub_type")
                blocks.append(Block(type=BlockType.LIST, text="\n".join(_list_line(x) for x in items_), page=page, bbox=bbox, meta=meta))
        elif t == "table":
            caption = " ".join(it.get("table_caption") or [])
            footnote = " ".join(it.get("table_footnote") or [])
            body = it.get("table_body") or ""
            md = html_table_to_markdown(body) if "<table" in body.lower() else body
            text = "\n".join(x for x in (caption, md, footnote) if x)
            if text.strip():
                meta.update({"format": "markdown", "caption": caption, "img_path": it.get("img_path")})
                blocks.append(Block(type=BlockType.TABLE, text=text, page=page, bbox=bbox, meta=meta))
        elif t in ("image", "chart"):
            caption = " ".join(it.get(f"{t}_caption") or [])
            footnotes = list(it.get(f"{t}_footnote") or [])
            if not caption:
                # 实测（4.0.10 standard）：图下方的"图1 …"有时被归进 footnote 而不是 caption
                cap = next((x for x in footnotes if re.match(r"^\s*(?:图|圖|图表|Figure|Fig\.?|Chart)\s*[\dA-Za-z一二三四五六七八九十]+", str(x), re.I)), None)
                if cap is not None:
                    caption = str(cap)
                    footnotes.remove(cap)
            footnote = " ".join(footnotes)
            desc = it.get("content") or ""  # image_analysis=True 时 VLM 给出的图片描述 / 图表数据
            text = "\n".join(x for x in (caption, desc, footnote) if x)
            meta.update({"img_path": it.get("img_path"), "sub_type": it.get("sub_type"), "caption": caption, "description": desc})
            # 没有图注 / 描述的图片也保留：像素进图片索引（MinerU 4.x 的 img_path 是内嵌 data URI）
            if text.strip() or it.get("img_path"):
                blocks.append(Block(type=BlockType.IMAGE, text=text, page=page, bbox=bbox, meta=meta))
        elif t == "code":
            caption = " ".join(it.get("code_caption") or [])
            text = it.get("code_body") or ""
            meta.update({"language": "", "caption": caption})
            if text.strip():
                blocks.append(Block(type=BlockType.CODE, text=(caption + "\n" if caption else "") + text, page=page, bbox=bbox, meta=meta))
        elif t == "equation":
            text = it.get("text") or ""
            if text.strip():
                blocks.append(Block(type=BlockType.EQUATION, text=text, page=page, bbox=bbox, meta=meta))
        elif t == "page_footnote":
            text = (it.get("text") or "").strip()
            if text:
                blocks.append(Block(type=BlockType.FOOTNOTE, text=text, page=page, bbox=bbox, meta=meta))
        elif t in _PAGE_AUX:
            text = (it.get("text") or "").strip()
            blocks.append(Block(type=_PAGE_AUX[t], text=text, page=page, bbox=bbox, meta=meta))
        # 其余未知类型不进入语料
    return blocks
