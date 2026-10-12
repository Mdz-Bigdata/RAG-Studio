from __future__ import annotations

from prism.config import CleaningConfig
from prism.ingest.cleaning import clean_document, dedup_blocks, drop_repeated_page_lines, fix_linebreaks
from prism.schemas import Block, BlockType, ParsedDocument


def test_fix_linebreaks_cjk_and_latin():
    text = "员工每年享有 5 天带薪年假，\n需提前在 OA 系统提交申请。\n1. 第一步\n2. 第二步"
    out = fix_linebreaks(text)
    assert out.splitlines()[0] == "员工每年享有 5 天带薪年假，需提前在 OA 系统提交申请。"
    assert out.splitlines()[1] == "1. 第一步"
    assert fix_linebreaks("The informa-\ntion is here") == "The information is here"
    assert fix_linebreaks("hello\nworld") == "hello world"


def test_drop_repeated_page_lines_positional():
    # 无 bbox：页首 / 页末块才是页眉页脚候选；正文行即使仅数字不同也要保留
    blocks = []
    for p in range(5):
        blocks.append(Block(type=BlockType.TEXT, text="XX 公司内部文件", page=p))
        blocks.append(Block(type=BlockType.TEXT, text=f"这是第 {p} 页的正文内容，讨论了不同的主题 {p * 17}。", page=p))
        blocks.append(Block(type=BlockType.TEXT, text=f"第 {p + 1} 页", page=p))
    out = drop_repeated_page_lines(blocks, 0.4)
    texts = [b.text for b in out]
    assert "XX 公司内部文件" not in texts
    assert not any(t.startswith("第 ") and t.endswith(" 页") for t in texts)
    assert len(out) == 5


def test_drop_repeated_page_lines_by_bbox():
    # 有 bbox（MinerU 0-1000）：按 y 坐标判断边缘；中部重复行（如每页都出现的同一句正文）不误删
    blocks = []
    for p in range(4):
        blocks.append(Block(type=BlockType.TEXT, text="机密 · 内部资料", page=p, bbox=[100, 20, 900, 60]))
        blocks.append(Block(type=BlockType.TEXT, text="本条款适用于全体员工。", page=p, bbox=[100, 400, 900, 450]))
        blocks.append(Block(type=BlockType.TEXT, text=f"正文 {p}", page=p, bbox=[100, 500, 900, 550]))
        blocks.append(Block(type=BlockType.TEXT, text=f"- {p + 1} -", page=p, bbox=[450, 950, 550, 990]))
    out = drop_repeated_page_lines(blocks, 0.4)
    texts = [b.text for b in out]
    assert "机密 · 内部资料" not in texts and "- 1 -" not in texts
    assert texts.count("本条款适用于全体员工。") == 4
    assert len(out) == 8


def test_dedup_exact_and_near():
    a = "员工出差住宿标准为一线城市每晚 600 元，其他城市每晚 400 元。部门负责人可审批上浮。"
    blocks = [Block(text=a), Block(text=a), Block(text=a + "！"), Block(text="完全不同的一段内容，关于数据库主键冲突的处理方案对比。")]
    out = dedup_blocks(blocks, 0.9)
    assert len(out) == 2


def test_clean_document_drops_garbled_and_headers():
    doc = ParsedDocument(doc_id="d", blocks=[
        Block(type=BlockType.HEADER, text="页眉"),
        Block(type=BlockType.TEXT, text="�����正文"),
        Block(type=BlockType.TEXT, text="正常的一段文字。"),
    ])
    out = clean_document(doc, CleaningConfig())
    assert [b.text for b in out.blocks] == ["正常的一段文字。"]
