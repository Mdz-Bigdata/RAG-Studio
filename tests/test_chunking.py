from __future__ import annotations

from pathlib import Path

from prism.config import ChunkingConfig
from prism.index.embeddings import HashEmbedder
from prism.ingest.chunking import StructuralChunker, count_tokens, split_sentences
from prism.ingest.chunking.parent_child import build_child_texts
from prism.ingest.chunking.semantic import semantic_split
from prism.ingest.parsers.markdown_parser import MarkdownParser
from prism.schemas import Block, BlockType, ParsedDocument

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_count_tokens_and_sentences():
    assert count_tokens("年假") == 2
    assert count_tokens("hello world") >= 2
    s = split_sentences("第一句。第二句！Third sentence. 第四句？")
    assert len(s) == 4


def test_structural_chunker_breadcrumbs_and_parent_child():
    doc = MarkdownParser().parse(RAW / "员工手册_2026版.md")
    chunks = StructuralChunker(ChunkingConfig(parent_max_tokens=200, child_tokens=60)).chunk(doc)
    parents = [c for c in chunks if c.kind == "parent"]
    children = [c for c in chunks if c.kind == "child"]
    assert parents and children
    by_id = {c.chunk_id: c for c in chunks}
    for ch in children:
        assert ch.parent_id in by_id
        assert ch.chunk_id in by_id[ch.parent_id].child_ids
        assert ch.breadcrumbs == by_id[ch.parent_id].breadcrumbs
        assert ch.token_count <= 60 * 1.6  # 单句超长时允许略超
    annual = [p for p in parents if "年假" in p.breadcrumbs]
    assert annual and annual[0].breadcrumbs == ["员工手册（2026 版）", "假期管理", "年假"]
    assert "[来源：" in annual[0].embed_text()
    table_parents = [p for p in parents if "table" in p.block_types]
    assert table_parents and table_parents[0].text.lstrip().startswith("| 职级")


def test_table_children_repeat_header():
    header = "| 型号 | 价格 |\n| --- | --- |\n"
    rows = "\n".join(f"| 型号{i} | {i * 100} |" for i in range(60))
    kids = build_child_texts(header + rows, child_tokens=80, overlap_ratio=0.1, block_types=[BlockType.TABLE.value])
    assert len(kids) > 1
    assert all(k.startswith("| 型号 | 价格 |") for k in kids)


def test_long_text_is_split_and_small_merged():
    long = "这是一个句子。" * 200
    doc = ParsedDocument(doc_id="d", blocks=[Block(type=BlockType.TITLE, level=1, text="标题"), Block(text=long), Block(text="短块。")])
    chunks = StructuralChunker(ChunkingConfig(parent_max_tokens=300, parent_min_tokens=50, child_tokens=100)).chunk(doc)
    parents = [c for c in chunks if c.kind == "parent"]
    assert len(parents) >= 4
    assert all(p.token_count <= 300 * 1.2 for p in parents)


def test_semantic_split_produces_groups():
    sentences = ["猫是一种宠物。", "猫喜欢晒太阳。", "猫会抓老鼠。", "Kubernetes 是容器编排系统。", "Pod 是最小调度单元。", "Service 提供稳定访问入口。"]
    groups = semantic_split(sentences, HashEmbedder(dim=256), percentile=60, min_sentences=2)
    assert sum(len(g) for g in groups) == len(sentences)
    assert len(groups) >= 2
