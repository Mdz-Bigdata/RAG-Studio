"""真实 PDF 上踩到的问题的回归测试。

夹具的形状照抄 MinerU 4.0.10 对一份 4 页中文 PDF（浏览器导出、图标字体项目符号）的实际输出，
内容换成了中性文本。对应的坑：
- Content List 的 `index` 块里是课程大纲这类正文，不能当目录丢掉；
- 页面辅助块的类型名是 header / footer / aside_text，不是 page_header / page_footer；
- 首页顶部的文档标题会被版面模型标成页眉；
- 连续几行"标题"之间没有正文时，前面的标题会从索引里消失；
- 文本层里的康熙部首（⾳ ⼿ ⼀）和图标字体项目符号（私用区字符）。

`RAG_TEST_MINERU=1 pytest tests/test_pdf_real_world.py -k live` 会用本机安装的 MinerU 真实解析一份
现场生成的 PDF（flash 档；首次运行会下载约 230MB 的版面 / OCR 模型）。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from prism.config import ChunkingConfig, CleaningConfig, Settings
from prism.ingest.chunking import StructuralChunker
from prism.ingest.cleaning import clean_document, normalize_bullets, normalize_glyphs
from prism.ingest.parsers.mineru_parser import content_list_to_blocks, mineru_available
from prism.ingest.popo_bridge import content_list_to_popo_input, validate_popo_input
from prism.schemas import Block, BlockType, DocMetadata, ParsedDocument

# MinerU 4.0.10 Content List V1 的真实形状（ocr_mode=auto，OCR 路径）
OCR_CONTENT_LIST = [
    {"type": "text", "text": "设备运维实战课程", "text_level": 2, "bbox": [179, 85, 556, 115], "page_idx": 0},
    {"type": "text", "text": "8 讲完整教程，从零搭建巡检系统", "text_level": 2, "bbox": [179, 141, 599, 162], "page_idx": 0},
    {"type": "text", "text": "课程价格:¥299", "text_level": 2, "bbox": [182, 179, 355, 200], "page_idx": 0},
    {
        "type": "index",
        "list_items": ["- 课程大纲", "- • 第1讲:架构设计与选型☆", "- 。系统架构图", "- 。技术栈对比", "- • 第2讲:[数据采集](#sec-2)", "    - 采集协议"],
        "bbox": [179, 195, 594, 942],
        "page_idx": 0,
    },
    {"type": "text", "text": "。 巡检任务调度:定时与事件触发", "bbox": [200, 60, 700, 80], "page_idx": 1},
    {"type": "text", "text": "适合人群", "text_level": 2, "bbox": [180, 300, 300, 330], "page_idx": 1},
    {"type": "list", "sub_type": "text", "list_items": ["运维工程师", "后端开发者"], "bbox": [180, 340, 600, 400], "page_idx": 1},
    {"type": "image", "img_path": "data:image/jpeg;base64,/9j/4AAQSkZJRg==", "image_caption": [], "image_footnote": [], "bbox": [181, 63, 204, 81], "page_idx": 2},
    {"type": "footer", "text": "内部资料", "bbox": [400, 960, 600, 980], "page_idx": 2},
    {"type": "page_number", "text": "3", "bbox": [480, 980, 520, 995], "page_idx": 2},
    {"type": "aside_text", "text": "侧栏水印", "bbox": [960, 300, 990, 700], "page_idx": 2},
]


def _doc(blocks: list[Block], title: str = "运维课程") -> ParsedDocument:
    return ParsedDocument(doc_id="d1", blocks=blocks, metadata=DocMetadata(source="运维课程.pdf", title=title, parser="mineru"))


def test_index_block_is_kept_as_list_not_dropped():
    blocks = content_list_to_blocks(OCR_CONTENT_LIST)
    lists = [b for b in blocks if b.type == BlockType.LIST]
    assert [b.meta["mineru_type"] for b in lists] == ["index", "list"]
    outline = lists[0].text.split("\n")
    assert outline[0] == "- 课程大纲" and outline[1] == "- • 第1讲:架构设计与选型☆"   # 自带的 `- ` 不重复添加
    assert outline[4] == "- • 第2讲:数据采集"                                          # 内部链接只留文字
    assert outline[5] == "    - 采集协议"                                             # 缩进（嵌套层级）保留
    assert lists[1].text == "- 运维工程师\n- 后端开发者"                                # list 的条目本身不带前缀


def test_page_aux_types_use_mineru4_names():
    by_type = {b.meta["mineru_type"]: b.type for b in content_list_to_blocks(OCR_CONTENT_LIST)}
    assert by_type["footer"] == BlockType.FOOTER
    assert by_type["page_number"] == BlockType.PAGE_NUMBER
    assert by_type["aside_text"] == BlockType.OTHER
    legacy = content_list_to_blocks([{"type": "page_header", "text": "旧叫法", "page_idx": 0}, {"type": "header", "text": "新叫法", "page_idx": 0}])
    assert [b.type for b in legacy] == [BlockType.HEADER, BlockType.HEADER]


def test_image_without_caption_keeps_pixels_for_image_index():
    img = next(b for b in content_list_to_blocks(OCR_CONTENT_LIST) if b.type == BlockType.IMAGE)
    assert img.text == "" and img.meta["img_path"].startswith("data:image/jpeg;base64,")
    cleaned = clean_document(_doc(content_list_to_blocks(OCR_CONTENT_LIST)), CleaningConfig())
    assert not any(b.type == BlockType.IMAGE for b in cleaned.blocks)   # 没有文字的图片不进正文
    assert not any("base64" in b.text for b in cleaned.blocks)


def test_outline_title_and_subtitle_survive_the_full_pipeline():
    """解析 → 清洗 → 切片之后：大纲、主标题、副标题都还在，页脚 / 页码 / 侧栏不在。"""
    doc = clean_document(_doc(content_list_to_blocks(OCR_CONTENT_LIST)), CleaningConfig())
    chunks = StructuralChunker(ChunkingConfig()).chunk(doc)
    parents = [c for c in chunks if c.kind == "parent"]
    text = "\n".join(p.text for p in parents)
    for needle in ("设备运维实战课程", "8 讲完整教程", "课程大纲", "第1讲", "系统架构图", "第2讲:数据采集", "巡检任务调度", "运维工程师"):
        assert needle in text, needle
    for noise in ("内部资料", "侧栏水印"):
        assert noise not in text
    first = parents[0]
    assert first.breadcrumbs == ["运维课程", "课程价格:¥299"]
    assert first.text.startswith("设备运维实战课程\n8 讲完整教程，从零搭建巡检系统")   # 孤儿标题并入后一节
    assert "- 第1讲:架构设计与选型☆\n  - 系统架构图\n  - 技术栈对比" in first.text      # "。"= 空心圆点 → 子项
    assert "- 巡检任务调度:定时与事件触发" in first.text


def test_orphan_titles_adopted_but_ancestors_left_alone():
    blocks = [
        Block(type=BlockType.TITLE, level=1, text="第一章 总则"),          # 有子节：标题在子节面包屑里，不重复
        Block(type=BlockType.TITLE, level=2, text="1.1 范围"),
        Block(type=BlockType.TEXT, text="适用于全体员工。"),
        Block(type=BlockType.TITLE, level=2, text="1.2 空节"),             # 孤儿：并入后一节
        Block(type=BlockType.TITLE, level=2, text="1.3 定义"),
        Block(type=BlockType.TEXT, text="术语说明。"),
        Block(type=BlockType.TITLE, level=1, text="附录"),                 # 文档末尾的孤儿：并入前一节结尾
    ]
    parents = [c for c in StructuralChunker(ChunkingConfig()).chunk(_doc(blocks, title="制度")) if c.kind == "parent"]
    assert [p.breadcrumbs for p in parents] == [["制度", "第一章 总则", "1.1 范围"], ["制度", "第一章 总则", "1.3 定义"]]
    assert parents[0].text == "适用于全体员工。"
    assert parents[1].text == "1.2 空节\n\n术语说明。\n\n附录"
    only_titles = [Block(type=BlockType.TITLE, level=1, text="封面标题"), Block(type=BlockType.TITLE, level=1, text="副标题")]
    solo = [c for c in StructuralChunker(ChunkingConfig()).chunk(_doc(only_titles, title="封面")) if c.kind == "parent"]
    assert len(solo) == 1 and solo[0].text == "封面标题\n副标题"


def test_first_page_header_is_rescued_but_running_headers_are_dropped():
    # 文本层路径：首页标题被标成 header，只出现一次 → 留下
    once = [
        Block(type=BlockType.HEADER, text="设备运维实战课程", page=0, meta={"mineru_type": "header"}),
        Block(type=BlockType.TEXT, text="8 讲完整教程，从零搭建巡检系统。", page=0),
        Block(type=BlockType.TEXT, text="第二页的正文内容在这里。", page=1),
    ]
    kept = clean_document(_doc(once), CleaningConfig()).blocks
    assert kept[0].text == "设备运维实战课程" and kept[0].type == BlockType.TEXT and kept[0].meta["rescued_header"]
    # 真正的页眉：每页都有 → 照常丢弃；非首页的一次性页眉也丢
    running = [
        Block(type=BlockType.HEADER, text="某某公司 运维手册", page=0),
        Block(type=BlockType.TEXT, text="第一页的正文内容在这里。", page=0),
        Block(type=BlockType.HEADER, text="某某公司 运维手册", page=1),
        Block(type=BlockType.TEXT, text="第二页的正文内容在这里。", page=1),
        Block(type=BlockType.HEADER, text="只在第三页出现的页眉", page=2),
        Block(type=BlockType.TEXT, text="第三页的正文内容在这里。", page=2),
    ]
    texts = [b.text for b in clean_document(_doc(running), CleaningConfig()).blocks]
    assert texts == ["第一页的正文内容在这里。", "第二页的正文内容在这里。", "第三页的正文内容在这里。"]


def test_kangxi_radicals_are_normalized_to_ideographs():
    # 浏览器导出的 PDF：⾳ ⼿ ⼀ ⽂ ⻓ ⻛ 看着和正常汉字一样，码位却在部首区
    raw = "智能语⾳助⼿：从零到⼀，⽀持⻓对话与多种⻛格"
    fixed = normalize_glyphs(raw)
    assert fixed == "智能语音助手：从零到一，支持长对话与多种风格"
    assert "：" in fixed and "，" in fixed            # 全角标点不动（不是整体 NFKC）
    doc = clean_document(_doc([Block(type=BlockType.TEXT, text=raw, page=0)]), CleaningConfig())
    assert "语音助手" in doc.blocks[0].text


def test_icon_font_bullets_become_list_items_instead_of_gluing_lines():
    # pdfplumber 路径：每行以私用区字形开头。删掉字形再合并断行，会得到"系统架构图技术栈对比环境搭建"
    per_line = "课程大纲\n 第1讲：架构设计 ⭐\n 系统架构图\n 技术栈对比\n 环境搭建"
    block = clean_document(_doc([Block(type=BlockType.TEXT, text=per_line, page=0)]), CleaningConfig()).blocks[0]
    assert block.text == "课程大纲\n- 第1讲：架构设计 ⭐\n- 系统架构图\n- 技术栈对比\n- 环境搭建"
    # MinerU 文本层路径：同一段里挤着多个字形
    inline = " 第1讲:架构设计 ⭐  系统架构图 技术栈对比"
    assert normalize_glyphs(inline) == "- 第1讲:架构设计 ⭐\n- 系统架构图\n- 技术栈对比"
    # 孤立的一个图标不是列表：只留空格，不把前后词粘在一起
    assert normalize_glyphs("点击  按钮") == "点击 按钮"


def test_bullet_artifacts_from_ocr():
    assert normalize_bullets("。 任务调度\n● 已完成\n- • 第1讲\n- 。子项\n一句话。第二句。") == "  - 任务调度\n- 已完成\n- 第1讲\n  - 子项\n一句话。第二句。"
    assert normalize_bullets("3.5 版本\n·") == "3.5 版本\n·"   # 后面没有内容的孤立符号不动


def test_popo_input_keeps_index_and_maps_real_aux_names():
    out = content_list_to_popo_input(OCR_CONTENT_LIST, input_label="/x/doc.pdf", doc_id="doc")
    assert validate_popo_input(out) == sum(len(v) for v in out["pages"].values())
    page1 = out["pages"]["1"]
    outline = next(b for b in page1 if b["source_label"] == "index")
    assert outline["type"] == "text" and "第1讲" in outline["content"]
    page3 = {b["source_label"]: b["type"] for b in out["pages"]["3"]}
    assert page3 == {"image": "image", "footer": "footer", "page_number": "page_number", "aside_text": "aside_text"}


# --------------------------------------------------------------------------------------
# 真实 MinerU：需要显式打开（会加载模型，首次运行要下载）
# --------------------------------------------------------------------------------------
LIVE = os.environ.get("RAG_TEST_MINERU") == "1" and mineru_available()


def _make_pdf(path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    pages = [
        ("设备巡检制度", ["第一条 巡检周期为每周一次，由值班工程师负责执行。", "第二条 巡检发现故障时，必须在 2 小时内上报，故障单编号格式为 GZ-2026-0001。"]),
        ("备件管理", ["第三条 备件库存低于安全线时，由库管员发起采购申请。", "第四条 编号 BJ-7781 的备件属于关键备件，最低库存为 5 件。"]),
    ]
    for title, paras in pages:
        c.setFont("STSong-Light", 22)
        c.drawString(72, h - 100, title)
        c.setFont("STSong-Light", 12)
        y = h - 150
        for p in paras:
            c.drawString(72, y, p)
            y -= 28
        c.showPage()
    c.save()


@pytest.mark.skipif(not LIVE, reason="设置 RAG_TEST_MINERU=1 且已安装 MinerU 时才运行")
def test_live_mineru_flash_parse_end_to_end(tmp_path: Path):
    """生成 PDF → MinerU flash 档解析 → 清洗 → 切片 → 建索引 → 检索命中精确编号。"""
    from prism.ingest import IngestPipeline
    from prism.retrieval import HybridRetriever

    pdf = tmp_path / "设备管理制度.pdf"
    _make_pdf(pdf)
    settings = Settings.lite(index_dir=tmp_path / "index")
    settings.paths.mineru_out_dir = tmp_path / "mineru_out"
    settings.parsing.mineru.enabled = True
    settings.parsing.mineru.tier = "flash"
    settings.multimodal.enabled = False
    pipeline = IngestPipeline(settings)
    doc = pipeline.parse(pdf)
    assert doc.metadata.parser == "mineru" and doc.metadata.parser_tier == "flash"
    assert {b.page for b in doc.blocks} == {0, 1}
    assert all(b.bbox and max(b.bbox) <= 1000 for b in doc.blocks)
    assert (settings.paths.mineru_out_dir / doc.doc_id / "content_list.json").is_file()
    chunks = pipeline.ingest_document(doc)
    text = "\n".join(c.text for c in chunks if c.kind == "parent")
    for needle in ("巡检周期", "GZ-2026-0001", "BJ-7781", "关键备件"):
        assert needle in text, needle
    result = HybridRetriever(settings, pipeline.index).retrieve("BJ-7781 最低库存是多少", top_k=1, use_transform=False)
    assert result.contexts and "5 件" in result.contexts[0].chunk.text
