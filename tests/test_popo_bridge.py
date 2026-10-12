from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism.ingest.popo_bridge import content_list_to_popo_input, find_mineru_output, import_popo_tree, mineru_output_to_popo_input, validate_popo_input
from prism.schemas import BlockType


def test_content_list_to_popo_input():
    items = [
        {"type": "text", "text": "第一章", "text_level": 1, "page_idx": 0, "bbox": [100, 100, 900, 150]},
        {"type": "text", "text": "正文", "page_idx": 0, "bbox": [100, 200, 900, 300]},
        {"type": "table", "table_body": "<table><tr><td>a</td></tr></table>", "page_idx": 1, "bbox": [0, 0, 1000, 1000]},
        {"type": "page_number", "text": "2", "page_idx": 1, "bbox": [450, 950, 550, 990]},
        {"type": "index", "list_items": ["目录"], "page_idx": 0, "bbox": [0, 0, 1, 1]},
    ]
    out = content_list_to_popo_input(items, input_label="/x/doc.pdf", doc_id="doc")
    assert out["input_label"] == "/x/doc.pdf"
    assert set(out["pages"]) == {"1", "2"}
    first = out["pages"]["1"][0]
    assert first["type"] == "title" and first["title_level"] == 1
    assert first["bbox"] == [0.1, 0.1, 0.9, 0.15]
    types = [b["type"] for p in out["pages"].values() for b in p]
    assert "page_number" in types and "table" in types and "index" not in types


def test_import_popo_tree(tmp_path: Path):
    tree = {
        "type": "root", "title": "", "content": "", "level": 0, "location": [], "block_ids": [], "children": [
            {"type": "text", "title": "第一章 总则", "metadata": "", "content": "第一条 本制度<|txt_contd|>适用于全体员工。<|txt_split|>第二条 略。", "level": 1,
             "location": [{"bbox": [0, 0, 1, 1], "page": 1}], "block_ids": [1], "children": [
                {"type": "text", "title": "1.1 范围", "metadata": "", "content": "范围说明。", "level": 2, "location": [{"bbox": [0, 0, 1, 1], "page": 2}], "block_ids": [2], "children": []},
                {"type": "table", "title": "表 1 标准", "metadata": "注：单位元", "content": "<table><tr><td>a</td><td>b</td></tr></table>", "level": 1,
                 "location": [{"bbox": [0, 0, 1, 1], "page": 2}, {"bbox": [0, 0, 1, 1], "page": 3}], "block_ids": [3, 4], "children": []},
            ]},
            {"type": "page_number", "title": "Page 1 - page_number", "metadata": "1", "content": "1", "level": -1, "location": [{"bbox": [0, 0, 1, 1], "page": 1}], "block_ids": [9], "children": []},
        ],
    }
    p = tmp_path / "doc.json"
    p.write_text(json.dumps(tree, ensure_ascii=False), encoding="utf-8")
    doc = import_popo_tree(p, doc_id="doc")
    types = [(b.type, b.level, b.text) for b in doc.blocks]
    assert types[0] == (BlockType.TITLE, 1, "第一章 总则")
    assert types[1] == (BlockType.TEXT, None, "第一条 本制度适用于全体员工。")
    assert types[2] == (BlockType.TEXT, None, "第二条 略。")
    assert types[3] == (BlockType.TITLE, 2, "1.1 范围")
    table = [b for b in doc.blocks if b.type == BlockType.TABLE][0]
    assert table.text.startswith("表 1 标准\n| a | b |") and table.text.endswith("注：单位元")
    assert table.meta["merged_pages"] == [2, 3]
    assert not any(b.type == BlockType.PAGE_NUMBER for b in doc.blocks)
    assert doc.metadata.title == "第一章 总则"


def test_legacy_model_json(tmp_path: Path):
    """VLM 模式 {doc}_model.json：千分比 bbox + lines/spans。"""
    model = [
        [
            {"type": "title", "bbox": [120, 340, 650, 380], "lines": [{"spans": [{"content": "1. Introduction"}]}]},
            {"type": "text", "bbox": [120, 400, 880, 600], "lines": [{"spans": [{"content": "正文第一句。"}, {"content": "第二句。"}]}]},
            {"type": "discarded", "bbox": [0, 0, 10, 10], "lines": [{"spans": [{"content": "水印"}]}]},
        ],
        [{"type": "table", "bbox": [900, 100, 100, 500], "html": "<table><tr><td>a</td></tr></table>"}],
    ]
    p = tmp_path / "doc_model.json"
    p.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    assert find_mineru_output(tmp_path, "doc") == ("model", p)
    out = mineru_output_to_popo_input(p, input_label="/x/doc.pdf", doc_id="doc")
    assert [b["type"] for b in out["pages"]["1"]] == ["title", "text"]
    assert out["pages"]["1"][0]["bbox"] == [0.12, 0.34, 0.65, 0.38]
    assert out["pages"]["1"][1]["content"] == "正文第一句。 第二句。"
    assert out["pages"]["2"][0]["bbox"] == [0.1, 0.1, 0.9, 0.5]  # 反向坐标被纠正为 xyxy


def test_legacy_middle_json(tmp_path: Path):
    """经典 pipeline {doc}_middle.json：物理坐标按 page_size 归一化，caption 子块单独成块。"""
    middle = {"pdf_info": [{
        "page_idx": 0, "page_size": [600, 800],
        "para_blocks": [
            {"type": "title", "bbox": [60, 80, 540, 120], "level": 2, "lines": [{"spans": [{"content": "二、标准"}]}]},
            {"type": "table", "bbox": [60, 200, 540, 600], "blocks": [
                {"type": "table_caption", "bbox": [60, 200, 540, 220], "lines": [{"spans": [{"content": "表 1"}]}]},
                {"type": "table_body", "bbox": [60, 220, 540, 600], "lines": [{"spans": [{"html": "<table><tr><td>x</td></tr></table>"}]}]},
            ]},
        ],
        "discarded_blocks": [{"type": "page_number", "bbox": [290, 770, 310, 790], "lines": [{"spans": [{"content": "1"}]}]}],
    }]}
    p = tmp_path / "doc_middle.json"
    p.write_text(json.dumps(middle, ensure_ascii=False), encoding="utf-8")
    out = mineru_output_to_popo_input(p, input_label="/x/doc.pdf", doc_id="doc")
    blocks = out["pages"]["1"]
    assert [b["type"] for b in blocks] == ["title", "table", "table_caption", "page_number"]
    assert blocks[0]["bbox"] == [0.1, 0.1, 0.9, 0.15] and blocks[0]["title_level"] == 2
    assert "<table>" in blocks[1]["content"]
    assert len({b["source_id"] for b in blocks}) == 4


def test_validate_popo_input_rejects_bad_bbox():
    ok = {"pages": {"1": [{"type": "text", "content": "a", "bbox": [0.1, 0.1, 0.5, 0.5]}]}}
    assert validate_popo_input(ok) == 1
    with pytest.raises(ValueError):
        validate_popo_input({"pages": {"1": [{"type": "text", "content": "a", "bbox": [120, 340, 650, 380]}]}})
    with pytest.raises(ValueError):
        validate_popo_input({"pages": {"1": [{"type": "text", "content": "a", "bbox": [0.5, 0.1, 0.1, 0.5]}]}})
    with pytest.raises(ValueError):
        validate_popo_input({"pages": {"1": [{"type": "paragraph", "content": "a", "bbox": [0.1, 0.1, 0.5, 0.5]}]}})
