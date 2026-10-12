"""录入流水线：解析 → （Popo 树替换）→ 图片抽取 + OCR / VLM 描述 → 清洗 → 切片 → 增强 → 建索引（文本 + 图片）。

调优顺序建议：先把切片做对（占 50% 时间），再开混合检索，然后上重排，最后做 Query 改写。
所以这条流水线的每一步都可以单独调用并单独评测（见 eval/）。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..corpus.aliases import AliasDict
from ..corpus.health import exclusion_key
from ..corpus.keywords import extract_keywords
from ..index import Index
from ..index.table_store import tables_from_blocks
from ..multimodal.describe import ImageDescriber
from ..multimodal.extract import IMAGE_EXTENSIONS, ExtractedImage, extract_images
from ..schemas import Block, BlockType, Chunk, DocMetadata, ParsedDocument
from .chunking import StructuralChunker
from .cleaning import clean_document
from .enrich import Enricher
from .parsers import ParserRegistry
from .parsers.markdown_parser import MarkdownParser
from .popo_bridge import export_for_popo, import_popo_tree

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    documents: int = 0
    parents: int = 0
    children: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0
    parsers: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return f"docs={self.documents} parents={self.parents} children={self.children} failed={len(self.failed)} time={self.seconds:.1f}s parsers={self.parsers}"


def _bbox_area(bbox: list[float] | None) -> float | None:
    """bbox 占页面的面积比例。坐标可能是 0~1（PyMuPDF 抽取时归一化）或 0~1000（MinerU）。"""
    if not bbox or len(bbox) != 4:
        return None
    scale = 1000.0 if max(bbox) > 1.5 else 1.0
    x0, y0, x1, y1 = (v / scale for v in bbox)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def _text_covered(ocr_text: str, body: str, threshold: float = 0.6) -> bool:
    """正文是否已经包含这段 OCR 文字（人工校对时已抄录 / 整理成表格）：按 OCR 的词段计覆盖率，忽略空白与标点差异。"""
    segments = [seg for seg in re.split(r"[^\w]+", _norm_text(ocr_text)) if len(seg) >= 2]
    if not segments:
        return False
    norm_body = _norm_text(body)
    return sum(1 for seg in segments if seg in norm_body) / len(segments) >= threshold


class IngestPipeline:
    def __init__(self, settings: Settings, index: Index | None = None, llm=None, vlm=None, *, aliases: AliasDict | None = None, excluded: set[str] | None = None) -> None:
        """vlm：看得了图的模型，用于给图片生成语义描述；不传则只用 OCR / 图注 / 上下文。
        aliases：别名词典（文档扩展）；excluded：语料治理剔除的段落，键为「doc_id:内容指纹」，重新解析时同样不入库。"""
        self.settings = settings
        self.index = index or Index(settings)
        self.aliases = aliases
        self.excluded = excluded or set()
        self.describer = ImageDescriber(settings.multimodal, vlm, cache_path=self.index.directory / "image_index" / "desc_cache.json")
        self.registry = ParserRegistry(settings)
        self.chunker = StructuralChunker(settings.chunking, embedder=self.index.embedder if settings.chunking.strategy == "semantic" else None)
        self.enricher = Enricher(settings.enrich, llm)

    # ----------------------------------------------------------------------------------
    def ingest_paths(self, paths: list[Path], metadata: DocMetadata | None = None, save: bool = True) -> IngestReport:
        t0 = time.perf_counter()
        report = IngestReport()
        for path in self._expand(paths):
            try:
                doc = self.parse(path, metadata)
                chunks = self.ingest_document(doc, source_path=path)
                report.documents += 1
                report.parents += sum(1 for c in chunks if c.kind == "parent")
                report.children += sum(1 for c in chunks if c.kind == "child")
                report.parsers[doc.metadata.parser] = report.parsers.get(doc.metadata.parser, 0) + 1
                logger.info("已录入 %s（%s）: %d parents / %d children", path.name, doc.metadata.parser, report.parents, report.children)
            except Exception as exc:
                logger.exception("录入失败 %s", path)
                report.failed.append((str(path), f"{type(exc).__name__}: {exc}"))
        if save:
            self.index.save()
        report.seconds = time.perf_counter() - t0
        return report

    def parse(self, path: Path, metadata: DocMetadata | None = None) -> ParsedDocument:
        parser = self.registry.get_parser(path)
        try:
            doc = parser.parse(path, metadata)
        except Exception as exc:
            # MinerU 解析失败（模型包没下全 / 依赖缺失 / 渲染进程异常）：有内置解析器的格式降级处理，而不是让文档直接失败
            fallback = self.registry.fallback_parser(path) if parser.name == "mineru" else None
            if fallback is None:
                raise
            logger.warning("MinerU 解析 %s 失败（%s），改用内置解析器 %s", path.name, exc, fallback.name)
            doc = fallback.parse(path, metadata)
            doc.metadata.extra["mineru_error"] = f"{type(exc).__name__}: {exc}"[:300]
        popo = self.settings.parsing.popo
        if popo.enabled and popo.use_tree_hierarchy:
            tree = self._popo_tree_path(doc.doc_id)
            if tree.exists():
                logger.info("使用 MinerU-Popo 文档树替换页面级结构：%s", tree)
                extra = doc.metadata.extra
                doc = import_popo_tree(tree, doc_id=doc.doc_id, metadata=doc.metadata, source=str(path))
                doc.metadata.extra.update(extra)
        return doc

    def parse_markdown_text(self, text: str, *, doc_id: str, source: str, metadata: DocMetadata | None = None) -> ParsedDocument:
        """人工校对后的 Markdown 重新解析（Human-in-the-loop）。"""
        doc = MarkdownParser().parse_text(text, Path(source), metadata)
        doc.doc_id = doc_id
        return doc

    def ingest_document(self, doc: ParsedDocument, source_path: Path | None = None, image_doc: ParsedDocument | None = None) -> list[Chunk]:
        """source_path：原始文件。给了就重新抽取图片；不给（如 Markdown 校对回灌）则保留已有图片。
        image_doc：抽图时用来定位页码 / 图注 / 上下文的文档结构。正文来自校对后的 Markdown 时，Markdown 里没有页码，
        "图N …"图注就对不上页，需要传原始文件解析出的结构。"""
        images = self.extract_images(source_path, image_doc or doc) if source_path is not None else None
        if source_path is None or image_doc is not None:
            # 正文来自校对后的 Markdown（不是这次从源文件解析出的结构）：extract_images 的扫描页回填没有作用到它身上，
            # 用这份文档的图片（沿用索引里的，或刚重新抽取的）再回填一次，否则扫描页的文字会从文本检索里消失。
            records = images if images is not None else self.index.images.of_document(doc.doc_id)
            self._backfill_scanned_pages(doc, records, page_text_known=False)
        doc = clean_document(doc, self.settings.cleaning)
        tables = tables_from_blocks(doc.doc_id, doc.blocks)   # Excel / CSV 的行 → SQL 检索路
        chunks = self._drop_excluded(self.chunker.chunk(doc))
        chunks = self.enricher.enrich(doc, chunks)
        self._expand_chunks(doc, chunks)
        # Content List 体积大，不进索引元数据
        doc.metadata.extra.pop("mineru_content_list", None)
        self.index.add_document(doc.doc_id, doc.metadata, chunks, images=images, tables=tables)
        return chunks

    def _drop_excluded(self, chunks: list[Chunk]) -> list[Chunk]:
        """语料治理里剔除过的段落连同其 child 一起不入库。按「文档 + 内容指纹」匹配：只剔除被选中的那一份，
        别的文档里内容相同的段落（体检时保留的那份副本）不受影响。"""
        if not self.excluded:
            return chunks
        drop = {c.chunk_id for c in chunks if c.kind == "parent" and exclusion_key(c.doc_id, c.text) in self.excluded}
        if drop:
            logger.info("按语料治理剔除清单跳过 %d 个段落", len(drop))
        return [c for c in chunks if c.chunk_id not in drop and c.parent_id not in drop]

    def _expand_chunks(self, doc: ParsedDocument, chunks: list[Chunk]) -> None:
        """文档扩展：关键词（parent 抽取，child 继承）+ 别名（每块按自身正文匹配词典）。"""
        cfg = self.settings.corpus
        if cfg.keywords:
            title = doc.metadata.title or ""
            by_parent = {}
            for p in (c for c in chunks if c.kind == "parent"):
                p.keywords = extract_keywords(p.text, cfg.keywords_per_chunk, title=p.breadcrumb_text or title)
                by_parent[p.chunk_id] = p.keywords
            for c in chunks:
                if c.kind == "child" and c.parent_id in by_parent:
                    c.keywords = list(by_parent[c.parent_id])
        if self.aliases:
            for c in chunks:
                c.aliases = self.aliases.expansions(c.breadcrumb_text + "\n" + c.text)

    def extract_images(self, path: Path, doc: ParsedDocument) -> list[ExtractedImage]:
        """抽取图片 → OCR / VLM 生成"图片文本摘要"。图片文件本身没有正文，用摘要当正文。"""
        mm = self.settings.multimodal
        if not mm.enabled:
            return []
        images = extract_images(path, doc, mm, mineru_out_dir=self.settings.paths.mineru_out_dir)
        for it in images:
            ocr_text, description = self.describer.describe(it.image, it.sha1, hint=it.caption or it.context[:200])
            it.ocr_text = it.ocr_text or ocr_text
            # 本系统 VLM 的描述放在前面，解析器自带的文字接在后面。MinerU flash 档给的"图片内容"是图中文字的转写
            # （CLIP 论文实测：图 1~22 全是英文转写），原来"解析器有就不用 VLM"会把 VLM 的中文语义描述整段丢掉，
            # 中文提问对英文图表几乎无从匹配；转写本身也要留着（PDF 文字层抽出的英文带空格，比 OCR 准）
            if description and it.description and description not in it.description:
                it.description = f"{description}\n{it.description}"
            else:
                it.description = it.description or description
        self._backfill_scanned_pages(doc, images)
        if images and not any(b.is_content() and b.type != BlockType.TITLE for b in doc.blocks):
            # 只有图片、没有正文的文档（图片文件、纯扫描 PDF 在 flash 档）：用每张图的文本摘要当正文，
            # 这样文档能被切片检索，图片也有可挂接的段落。
            is_file = path.suffix.lower().lstrip(".") in IMAGE_EXTENSIONS
            title = doc.metadata.title or path.stem
            if not any(b.type == BlockType.TITLE for b in doc.blocks):
                doc.blocks.insert(0, Block(type=BlockType.TITLE, level=1, text=title, page=0))
            for it in images:
                if is_file and it.caption in ("", path.stem):
                    it.caption = title
                where = "" if is_file or it.page is None else f"（第 {it.page + 1} 页）"
                body = "\n".join(x for x in (f"图注：{it.caption}" if it.caption and not is_file else "", it.description, f"图片中文字：{it.ocr_text}" if it.ocr_text else "") if x)
                it.context = it.context or body[: mm.context_chars]
                doc.blocks.append(Block(type=BlockType.TEXT, text=f"图片{where}：{it.caption or title}" + (f"\n{body}" if body else ""), page=it.page if it.page is not None else 0))
        if images:
            logger.info("%s：抽取 %d 张图片（OCR=%s，VLM=%s）", path.name, len(images), self.describer.capabilities["ocr"], self.describer.capabilities["vlm"])
        return images

    # ----------------------------------------------------------------------------------
    def export_popo_inputs(self, path: Path, metadata: DocMetadata | None = None) -> dict[str, Path]:
        """解析 PDF 并为 MinerU-Popo 准备输入（不入库）。"""
        parser = self.registry.get_parser(path)
        doc = parser.parse(path, metadata)
        return export_for_popo(doc, pdf_path=path, popo_dir=self.settings.paths.popo_dir)

    @staticmethod
    def _backfill_scanned_pages(doc: ParsedDocument, images: list, *, page_text_known: bool = True) -> None:
        """扫描页回填：整页是一张图（扫描件 / 截图页）、页面本身几乎没有文字层时，把 OCR 结果作为这一页的正文。
        否则这些文字只存在于图片索引里，文本检索与回答上下文都看不到。
        判定"整页是图"：知道图片位置（MinerU / PyMuPDF 给出 bbox）时要求覆盖 ≥ 50% 页面；不知道位置时要求这一页
        几乎没有正文（< 40 字，页眉这类短行不算数）。只看"正文少于 N 字"会误判：图表页的正文可能本来就短，
        实测一页 70 字的正文 + 一张柱状图，被当成了扫描页。
        （MinerU standard 档自己会 OCR 成正文，此时该页已有文字，不会重复回填。）

        page_text_known=False：正文来自校对后的 Markdown，没有页码，算不出每页有多少字——只回填有位置且覆盖 ≥ 50% 页面的图，
        并且正文里已经包含这段文字（校对时抄录过 / 整理成了表格）就不再回填，避免重复。images 可以是抽取结果（ExtractedImage）
        或索引里的图片记录（ImageRecord）。"""
        body = "\n".join(b.text for b in doc.blocks) if not page_text_known else ""
        page_chars: dict[int, int] = {}
        for b in doc.blocks:
            if b.page is not None and b.is_content() and b.type != BlockType.TITLE:
                page_chars[b.page] = page_chars.get(b.page, 0) + len(b.text.strip())
        for it in images:
            size = it.image.size if hasattr(it, "image") else (it.width, it.height)
            if it.page is None or len(it.ocr_text or "") < 20 or min(size) < 400:
                continue
            area = _bbox_area(it.bbox)
            if not page_text_known:
                if area is None or area < 0.5 or _text_covered(it.ocr_text, body):
                    continue
            elif area is not None:
                if area < 0.5 or page_chars.get(it.page, 0) >= 300:
                    continue
            elif page_chars.get(it.page, 0) >= 40:
                continue
            doc.blocks.append(Block(type=BlockType.TEXT, text=f"（第 {it.page + 1} 页为扫描图片，以下为 OCR 识别文字）\n{it.ocr_text}", page=it.page, meta={"ocr_backfill": True}))
            page_chars[it.page] = page_chars.get(it.page, 0) + len(it.ocr_text)

    def _popo_tree_path(self, doc_id: str) -> Path:
        return self.settings.paths.popo_dir / "outputs" / "build_tree" / "mineru" / f"{doc_id}.json"

    def _expand(self, paths: list[Path]) -> list[Path]:
        exts = self.registry.supported_extensions()
        out: list[Path] = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                out.extend(sorted(x for x in p.rglob("*") if x.is_file() and x.suffix.lower().lstrip(".") in exts and not x.name.startswith(".")))
            elif p.is_file():
                out.append(p)
            else:
                logger.warning("路径不存在：%s", p)
        return out
