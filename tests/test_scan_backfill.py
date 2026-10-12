"""扫描页 OCR 回填在「Markdown 校对回灌」路径上也要生效：正文来自校对稿时，用索引里的图片记录回填整页扫描图的文字；
图表（只占半页以内）不回填；校对稿里已经抄录过这段文字就不重复；没有位置信息的图无法判断，不回填。"""

from __future__ import annotations

from pathlib import Path

from prism.config import Settings
from prism.ingest import IngestPipeline
from prism.multimodal.schemas import ImageRecord

SCAN_OCR = "附件：设备巡检记录 巡检单号：XJ-2025-0931 巡检日期：2025年9月18日 设备编号：PUMP-07 状态：轴承温度偏高 78℃ 处理意见：更换轴承，停机 4小时 复检结论：温度恢复至 52℃，运行正常"
CHART_OCR = "2025年Q3各区域销售额 (万元) 300 华北区 | 450 华东区 | 380 华南区 | 210 西部区"
MARKDOWN = "# 2025 年第三季度经营报告\n\n一、区域销售情况第三季度公司实现销售额 1340 万元，见图1。\n\n![](embedded-image)\n\n图1 2025年Q3各区域销售额\n\n![](embedded-image)\n"


def _rec(i: int, page: int, ocr: str, bbox, w=900, h=1200) -> ImageRecord:
    return ImageRecord(image_id=f"d1_img{i:03d}", doc_id="d1", file=f"images/d1/{i:03d}.png", sha1=str(i), width=w, height=h, page=page, bbox=bbox, origin="mineru", ocr_text=ocr)


SCAN = _rec(2, 2, SCAN_OCR, [67.0, 71.0, 933.0, 937.0])            # 覆盖 75% 页面
CHART = _rec(0, 0, CHART_OCR, [134.0, 245.0, 857.0, 568.0], 1197, 756)   # 覆盖 23% 页面


def _pipeline(tmp_path: Path) -> IngestPipeline:
    return IngestPipeline(Settings.lite(index_dir=tmp_path / "index"))


def _backfilled(doc) -> list[str]:
    return [b.text for b in doc.blocks if b.meta.get("ocr_backfill")]


def test_markdown_body_backfills_only_full_page_scans(tmp_path: Path):
    p = _pipeline(tmp_path)
    doc = p.parse_markdown_text(MARKDOWN, doc_id="d1", source="report.pdf")
    p._backfill_scanned_pages(doc, [CHART, SCAN, _rec(3, 1, SCAN_OCR.replace("PUMP-07", "PUMP-08"), None)], page_text_known=False)
    texts = _backfilled(doc)
    assert len(texts) == 1 and "PUMP-07" in texts[0] and texts[0].startswith("（第 3 页为扫描图片")   # 图表、没有位置的图都不回填


def test_no_duplicate_when_proofread_text_already_has_it(tmp_path: Path):
    p = _pipeline(tmp_path)
    table = "\n附件 巡检记录数据：\n\n| 项目 | 内容 |\n| --- | --- |\n| 巡检单号 | XJ-2025-0931 |\n| 巡检日期 | 2025 年 9 月 18 日 |\n| 设备编号 | PUMP-07 |\n| 状态 | 轴承温度偏高 78℃ |\n| 处理意见 | 更换轴承，停机 4 小时 |\n| 复检结论 | 温度恢复至 52℃，运行正常 |\n"
    doc = p.parse_markdown_text(MARKDOWN + table, doc_id="d1", source="report.pdf")
    p._backfill_scanned_pages(doc, [SCAN], page_text_known=False)
    assert not _backfilled(doc)


def test_ingest_from_proofread_markdown_keeps_scan_text_searchable(tmp_path: Path, monkeypatch):
    """校对回灌（ingest_document 不带 source_path）：沿用索引里的图片记录回填，扫描页文字进入切片，能被文本检索到。"""
    p = _pipeline(tmp_path)
    monkeypatch.setattr(p.index.images, "of_document", lambda doc_id: [CHART, SCAN] if doc_id == "d1" else [])
    chunks = p.ingest_document(p.parse_markdown_text(MARKDOWN, doc_id="d1", source="report.pdf"))
    text = " ".join(c.text for c in chunks)
    assert "PUMP-07" in text and "52℃" in text
    assert "300 华北区" not in text                      # 图表的 OCR 不当正文
