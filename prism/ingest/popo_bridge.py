"""MinerU ⇄ MinerU-Popo 桥接。

MinerU-Popo（https://github.com/opendatalab/MinerU-Popo）是页面级 OCR 输出的后处理框架：
用一个 4B 模型完成四个子任务——跨页表格截断分析、文本截断分析、标题层级分析、图文关联分析，
把"一页一页的块"重建为"整篇文档的语义树"。对 MinerU 输出，层级结构 TEDS 从 53.7 提升到 90.6，
ViDoRe V3 上 RAG 准确率最高提升 3.7 个点。

对 RAG 的价值：
- 跨页表格被合并，不再是两个半截表格；
- 被分页打断的段落续接（<|txt_contd|>），切片不再拦腰斩断；
- 标题层级来自全文而不是单页，面包屑更准；
- 图片 / 表格与说明文字关联。

工作流（四步，Popo 侧命令来自其 README）：
1. 本模块 `export_for_popo()`：把 MinerU 的 Content List 写成 Popo 期待的目录布局，
   同时直接生成 label_normalization 的产物（可跳过官方第 2 步）
2. `bash scripts/run_label_normalization.sh`（可选）
3. `bash scripts/run_inference.sh`   → outputs/inference/mineru/<doc_id>.json
4. `bash scripts/build_tree.sh`      → outputs/build_tree/mineru/<doc_id>.json
5. 本模块 `import_popo_tree()`：把树转回 ParsedDocument（标题层级 / 续接段落 / 合并表格）

MinerU 各版本输出与 Popo 的兼容性（Popo 的 MineruReader 按此优先级读取）：
    1. {doc}_model.json          VLM 模式：页面块数组，bbox 为千分比（0~1000），内容在 lines[].spans[].content
    2. {doc}_middle.json         经典 pipeline / hybrid：pdf_info[].para_blocks[]，bbox 按 page_size 归一化
    3. {doc}_content_list.json   Content List 协议：扁平块列表，带 text_level 的视为标题（MinerU 4.x 走这一条）
Markdown / Structured Content 等纯渲染输出丢失了逐块 bbox 与元素类型，不能喂给 Popo。
`mineru_output_to_popo_input()` 统一处理这三种输入；`validate_popo_input()` 复刻 Popo 推理前的严格校验
（bbox 必须是 0~1 的 xyxy 顺序，否则推理会拒绝）。
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from ..schemas import Block, BlockType, DocMetadata, ParsedDocument

# Popo run_inference.py 的 ALLOWED_BLOCK_TYPES
_POPO_TYPES = {
    "title", "text", "list_item", "equation", "image", "table", "image_caption", "table_caption",
    "image_footnote", "table_footnote", "page_title", "page_number", "page_footnote", "header", "aside_text", "footer",
}
_SUPPLEMENT = {"page_title", "page_number", "page_footnote", "header", "aside_text", "footer"}


def export_for_popo(doc: ParsedDocument, pdf_path: Path, popo_dir: Path, model_name: str = "mineru") -> dict[str, Path]:
    """写出 Popo 需要的两套输入，返回各路径。要求 doc.metadata.extra['mineru_content_list'] 存在。"""
    content_list: list[dict[str, Any]] | None = doc.metadata.extra.get("mineru_content_list")
    if not content_list:
        raise ValueError("该文档没有 MinerU Content List（需用 MinerU 解析且 save_outputs=True）")
    doc_id = doc.doc_id
    popo_dir = Path(popo_dir)

    # 1) 官方 label_normalization 的输入布局：post-process/<model>/<doc_id>/vlm/<doc_id>_content_list.json
    raw_dir = popo_dir / "post-process" / model_name / doc_id / "vlm"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{doc_id}_content_list.json"
    raw_path.write_text(json.dumps(content_list, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) PDF 放到 eval_pdf_dir/<doc_id>.pdf（Popo 推理需要页面渲染图）
    pdf_dir = popo_dir / "eval_pdf_dir"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_target = pdf_dir / f"{doc_id}.pdf"
    if not pdf_target.exists():
        shutil.copyfile(pdf_path, pdf_target)

    # 3) 直接生成归一化输入：outputs/label_normalization/<model>/<doc_id>.json（可跳过官方第 2 步）
    norm_dir = popo_dir / "outputs" / "label_normalization" / model_name
    norm_dir.mkdir(parents=True, exist_ok=True)
    normalized = content_list_to_popo_input(content_list, input_label=str(pdf_target.resolve()), doc_id=doc_id)
    validate_popo_input(normalized)
    norm_path = norm_dir / f"{doc_id}.json"
    norm_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"content_list": raw_path, "pdf": pdf_target, "normalized": norm_path}


def content_list_to_popo_input(content_list: list[dict[str, Any]], *, input_label: str, doc_id: str) -> dict[str, Any]:
    """MinerU Content List V1 → Popo 归一化输入 {"input_label", "doc_id", "pages": {"1": [block...]}}。
    bbox 从 0-1000 归一到 0-1；page 从 0-based 转 1-based。"""
    pages: dict[str, list[dict[str, Any]]] = {}
    order = 0
    for it in content_list:
        bbox = it.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        mapped = _map_type(it)
        if mapped is None:
            continue
        popo_type, content, title_level = mapped
        if not content and popo_type in ("text", "title", "list_item", "image_caption", "table_caption"):
            continue
        page = str(int(it.get("page_idx", 0)) + 1)
        block: dict[str, Any] = {
            "type": popo_type,
            "content": content,
            "bbox": _norm_bbox(bbox, 1000.0, 1000.0),
            "source_label": str(it.get("type", "")),
            "source_id": f"{doc_id}:{order}",
        }
        if title_level is not None:
            block["title_level"] = title_level
        pages.setdefault(page, []).append(block)
        order += 1
    return {"input_label": input_label, "doc_id": doc_id, "pages": pages}


def _map_type(it: dict[str, Any]) -> tuple[str, str, int | None] | None:
    t = it.get("type")
    if t == "text":
        text = _norm(it.get("text"))
        if it.get("text_level"):
            return "title", text, int(it["text_level"])
        return "text", text, None
    if t in ("list", "index"):
        # index（版面模型眼里的"目录"）里常常是大纲这类正文：和 list 一样当文本交给 Popo，不能丢
        items = [str(x).strip() for x in it.get("list_items") or [] if str(x).strip()]
        return ("text", _norm(" ".join(items)), None) if items else None
    if t == "table":
        return "table", _norm(it.get("table_body") or ""), None
    if t in ("image", "chart"):
        return "image", _norm(it.get("content") or ""), None
    if t == "equation":
        return "equation", _norm(it.get("text") or ""), None
    if t == "code":
        return "text", _norm(it.get("code_body") or ""), None
    if t == "page_footnote":
        return "page_footnote", _norm(it.get("text")), None
    # MinerU 4.0.x 的页面辅助块叫 header / footer / page_number / aside_text；page_* 是早期叫法
    if t in ("header", "page_header"):
        return "header", _norm(it.get("text")), None
    if t in ("footer", "page_footer"):
        return "footer", _norm(it.get("text")), None
    if t == "page_number":
        return "page_number", _norm(it.get("text")), None
    if t in ("aside_text", "page_aside_text"):
        return "aside_text", _norm(it.get("text")), None
    return None


def _norm_bbox(bbox: list[Any], width: float, height: float) -> list[float]:
    """任意坐标系的 bbox → 0~1 的 xyxy（裁剪到范围内，并保证 x0 ≤ x1、y0 ≤ y1）。"""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0, x1 = sorted((x0 / width, x1 / width))
    y0, y1 = sorted((y0 / height, y1 / height))
    return [round(min(1.0, max(0.0, v)), 6) for v in (x0, y0, x1, y1)]


# --------------------------------------------------------------------------------------
# MinerU < 4.0 的 model.json / middle.json → Popo 归一化输入
# --------------------------------------------------------------------------------------
# MinerU 标签 → Popo 规范标签；None = 跳过（discarded / *_body 等）
_LEGACY_TYPE_MAP: dict[str, str | None] = {
    "title": "title", "text": "text", "list": "list_item", "list_item": "list_item", "index": None,
    "equation": "equation", "interline_equation": "equation", "image": "image", "chart": "image", "table": "table",
    "image_caption": "image_caption", "table_caption": "table_caption", "image_footnote": "image_footnote", "table_footnote": "table_footnote",
    "page_footnote": "page_footnote", "page_number": "page_number", "header": "header", "page_header": "header",
    "footer": "footer", "page_footer": "footer", "aside_text": "aside_text", "page_aside_text": "aside_text", "page_title": "page_title",
    "code": "text", "ref_text": "text", "phonetic": "text",
    "discarded": None, "image_body": None, "table_body": None,
}


def _block_content(block: dict[str, Any]) -> str:
    """content / text / html / words 或 lines[].spans[].content 任一存在即可。"""
    for key in ("content", "text", "html", "words", "table_body"):
        v = block.get(key)
        if isinstance(v, str) and v.strip():
            return _norm(v)
    parts: list[str] = []
    for line in block.get("lines") or []:
        for span in line.get("spans") or []:
            c = span.get("content") or span.get("html") or ""
            if c:
                parts.append(str(c))
    return _norm(" ".join(parts))


def _legacy_block(block: dict[str, Any], width: float, height: float, doc_id: str, order: int) -> list[dict[str, Any]]:
    """一个 MinerU 块 → 0..n 个 Popo 块（image / table 的 caption、footnote 子块单独成块）。"""
    out: list[dict[str, Any]] = []
    label = str(block.get("type", ""))
    popo_type = _LEGACY_TYPE_MAP.get(label)
    bbox = block.get("bbox")
    children = block.get("blocks") or []
    if popo_type is not None and bbox is not None and len(bbox) == 4:
        content = _block_content(block)
        if not content and children:  # 表格 / 图片的正文在 *_body 子块里
            content = " ".join(filter(None, (_block_content(ch) for ch in children if str(ch.get("type", "")).endswith("_body"))))
        if content or popo_type in ("image", "table"):
            item: dict[str, Any] = {"type": popo_type, "content": content, "bbox": _norm_bbox(bbox, width, height), "source_label": label, "source_id": f"{doc_id}:{order + len(out)}"}
            level = block.get("level") or block.get("text_level")
            if popo_type == "title" and level:
                item["title_level"] = int(level)
            out.append(item)
    for ch in children:
        if not str(ch.get("type", "")).endswith("_body"):
            out.extend(_legacy_block(ch, width, height, doc_id, order + len(out)))
    return out


def model_json_to_popo_input(model: Any, *, input_label: str, doc_id: str) -> dict[str, Any]:
    """VLM 模式的 {doc}_model.json：页面块数组，bbox 为千分比（0~1000）。
    兼容三种外形：[[block...], ...]（每页一个数组）/ [{"page_idx", "blocks" | "layout_dets"}] / 带 page_idx 的扁平块列表。"""
    pages: dict[str, list[dict[str, Any]]] = {}
    order = 0

    def add(page_no: int, blocks: list[dict[str, Any]]) -> None:
        nonlocal order
        for b in blocks:
            items = _legacy_block(b, 1000.0, 1000.0, doc_id, order)
            order += len(items)
            if items:
                pages.setdefault(str(page_no), []).extend(items)

    if isinstance(model, dict):
        model = model.get("pages") or model.get("pdf_info") or []
    for i, page in enumerate(model or []):
        if isinstance(page, list):
            add(i + 1, page)
        elif isinstance(page, dict) and ("blocks" in page or "layout_dets" in page or "para_blocks" in page):
            add(int(page.get("page_idx", i)) + 1, page.get("blocks") or page.get("layout_dets") or page.get("para_blocks") or [])
        elif isinstance(page, dict):
            add(int(page.get("page_idx", 0)) + 1, [page])
    return {"input_label": input_label, "doc_id": doc_id, "pages": pages}


def middle_json_to_popo_input(middle: dict[str, Any], *, input_label: str, doc_id: str) -> dict[str, Any]:
    """经典 pipeline / hybrid 的 {doc}_middle.json：pdf_info[].para_blocks[]，bbox 是页面物理坐标，按 page_size 归一化。"""
    pages: dict[str, list[dict[str, Any]]] = {}
    order = 0
    for i, page in enumerate(middle.get("pdf_info") or []):
        size = page.get("page_size") or [1.0, 1.0]
        width, height = float(size[0]) or 1.0, float(size[1]) or 1.0
        page_no = str(int(page.get("page_idx", i)) + 1)
        blocks = list(page.get("para_blocks") or page.get("preproc_blocks") or [])
        # 页眉 / 页脚 / 页码在 middle.json 里放在 discarded_blocks：保留已打标的，Popo 会挂到根节点
        blocks += [b for b in page.get("discarded_blocks") or [] if b.get("type") in ("header", "footer", "page_number", "page_footnote", "aside_text")]
        for b in blocks:
            items = _legacy_block(b, width, height, doc_id, order)
            order += len(items)
            if items:
                pages.setdefault(page_no, []).extend(items)
    return {"input_label": input_label, "doc_id": doc_id, "pages": pages}


def find_mineru_output(directory: Path, doc_name: str) -> tuple[str, Path] | None:
    """按 Popo MineruReader 的优先级在目录里找 MinerU 输出：model.json → middle.json → content_list.json。"""
    directory = Path(directory)
    for kind in ("model", "middle", "content_list"):
        for cand in (directory / f"{doc_name}_{kind}.json", directory / f"{kind}.json", directory / f"{kind}_json.json"):
            if cand.exists():
                return kind, cand
    return None


def mineru_output_to_popo_input(path: Path, *, input_label: str, doc_id: str, kind: str | None = None) -> dict[str, Any]:
    """三种 MinerU 输出的统一入口；kind 省略时按文件名后缀 / 数据外形自动判断。结果已通过 validate_popo_input。"""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if kind is None:
        name = path.stem
        if name.endswith("content_list") or (isinstance(data, list) and data and isinstance(data[0], dict) and "page_idx" in data[0] and "lines" not in data[0]):
            kind = "content_list"
        elif name.endswith("middle") or name.endswith("middle_json") or (isinstance(data, dict) and "pdf_info" in data):
            kind = "middle"
        else:
            kind = "model"
    if kind == "content_list":
        out = content_list_to_popo_input(data, input_label=input_label, doc_id=doc_id)
    elif kind == "middle":
        out = middle_json_to_popo_input(data, input_label=input_label, doc_id=doc_id)
    elif kind == "model":
        out = model_json_to_popo_input(data, input_label=input_label, doc_id=doc_id)
    else:
        raise ValueError(f"未知的 MinerU 输出类型：{kind}")
    validate_popo_input(out)
    return out


def validate_popo_input(data: dict[str, Any]) -> int:
    """Popo 推理前的严格校验：类型在允许集合内，bbox 为 0~1 的 xyxy。返回块总数；不合法抛 ValueError。"""
    if not isinstance(data.get("pages"), dict):
        raise ValueError("Popo 输入缺少 pages")
    n = 0
    for page, blocks in data["pages"].items():
        if not str(page).isdigit() or int(page) < 1:
            raise ValueError(f"页码必须是 1-based 整数：{page!r}")
        for b in blocks:
            if b.get("type") not in _POPO_TYPES:
                raise ValueError(f"第 {page} 页存在未归一化的类型：{b.get('type')!r}")
            bbox = b.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in bbox)):
                raise ValueError(f"第 {page} 页 bbox 必须归一化到 0~1：{bbox!r}")
            if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                raise ValueError(f"第 {page} 页 bbox 必须是 xyxy 顺序：{bbox!r}")
            n += 1
    return n


def _norm(text: Any) -> str:
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text.replace("　", " ").replace("\n", " ")).strip()


# --------------------------------------------------------------------------------------
# Popo 树 → ParsedDocument
# --------------------------------------------------------------------------------------
def import_popo_tree(tree_path: Path, *, doc_id: str | None = None, metadata: DocMetadata | None = None, source: str = "") -> ParsedDocument:
    tree = json.loads(Path(tree_path).read_text(encoding="utf-8"))
    md = metadata.model_copy() if metadata else DocMetadata()
    md.source = md.source or source or str(tree_path)
    md.parser = (md.parser + "+popo") if md.parser and "popo" not in md.parser else "mineru+popo"
    md.file_format = md.file_format or "pdf"
    doc = ParsedDocument(doc_id=doc_id or ParsedDocument.make_doc_id(md.source), metadata=md)
    blocks: list[Block] = []
    _walk(tree, blocks, depth=0)
    doc.blocks = blocks
    if not md.title:
        for b in blocks:
            if b.type == BlockType.TITLE:
                md.title = b.text
                break
    return doc


def _walk(node: dict[str, Any], out: list[Block], depth: int) -> None:
    ntype = node.get("type", "text")
    title = (node.get("title") or "").strip()
    content = node.get("content") or ""
    level = node.get("level", -1)
    page = _first_page(node)

    if ntype == "root":
        for ch in node.get("children", []):
            _walk(ch, out, depth)
        return
    if ntype in _SUPPLEMENT:
        return  # 页眉页脚页码：不进语料

    if ntype == "text":
        if title and title != "Default Title":
            out.append(Block(type=BlockType.TITLE, level=max(1, int(level)) if isinstance(level, int) and level > 0 else max(1, depth + 1), text=title, page=page, meta={"popo": True}))
        for para in _split_content(content):
            out.append(Block(type=BlockType.TEXT, text=para, page=page, meta={"popo": True}))
    elif ntype in ("table", "chart", "image", "seal", "image_block"):
        caption = title
        footnote = (node.get("metadata") or "").strip()
        body = content
        if ntype == "table" and "<table" in body.lower():
            from .parsers.html_table import html_table_to_markdown

            body = html_table_to_markdown(body)
        text = "\n".join(x for x in (caption, body, footnote) if x)
        btype = BlockType.TABLE if ntype == "table" else BlockType.IMAGE
        if text.strip():
            out.append(Block(type=btype, text=text, page=page, meta={"popo": True, "format": "markdown" if ntype == "table" else None, "caption": caption, "merged_pages": [loc.get("page") for loc in node.get("location", [])]}))
    for ch in node.get("children", []):
        _walk(ch, out, depth + 1)


def _split_content(content: str) -> list[str]:
    """<|txt_contd|> = 跨页续接（直接拼上），<|txt_split|> = 段落边界。"""
    if not content:
        return []
    joined = content.replace("<|txt_contd|>", "")
    return [p.strip() for p in joined.split("<|txt_split|>") if p.strip()]


def _first_page(node: dict[str, Any]) -> int | None:
    locs = node.get("location") or []
    for loc in locs:
        p = loc.get("page")
        if isinstance(p, int):
            return p - 1  # Popo 1-based → 内部 0-based
    return None
