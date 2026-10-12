"""长期记忆（LanceDB）：去重合并、按用户隔离的召回与综合打分、访问统计、过期 / 衰减遗忘、换模型重算、对话抽取。"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("lancedb")

from prism.config import MemoryConfig  # noqa: E402
from prism.generation import FakeLLM  # noqa: E402
from prism.index.embeddings import HashEmbedder  # noqa: E402
from prism.memory import LanceMemoryStore, extract_by_rules, extract_memories  # noqa: E402


def _store(tmp_path: Path, dim: int = 64, **cfg) -> LanceMemoryStore:
    return LanceMemoryStore(tmp_path / "lancedb", HashEmbedder(dim=dim), MemoryConfig(min_score=0.0, **cfg))


def test_add_dedup_recall_and_isolation(tmp_path: Path):
    s = _store(tmp_path)
    a, how = s.add("用户在财务部，负责报销审核", user_id="u1", kind="profile", importance=0.8)
    assert how == "created"
    b, how = s.add("用户在财务部，负责报销审核", user_id="u1", kind="profile", importance=0.6)   # 同一句：合并，不重复写
    assert how == "updated" and b.id == a.id and b.importance == 0.8
    s.add("用户希望回答先给结论再讲细节", user_id="u1", kind="preference", importance=0.7)
    s.add("用户在研发部", user_id="u2", kind="profile")
    assert len(s.list(user_id="u1")) == 2 and s.stats()["users"] == 2

    hits = s.search("财务部 报销", user_id="u1", top_k=2)
    assert hits[0]["record"].content.startswith("用户在财务部") and 0 < hits[0]["score"] <= 1.0
    assert all(h["record"].user_id == "u1" for h in hits)                              # 按用户隔离
    assert s.search("研发部", user_id="u1", kinds=["preference"], top_k=3)[0]["record"].kind == "preference"
    assert s.get(a.id).access_count == 1                                               # 召回会记一次访问

    s.update(a.id, content="用户已调到审计部", importance=0.9)
    assert s.get(a.id).content == "用户已调到审计部"
    assert s.delete(a.id) and s.get(a.id) is None


def test_expiry_and_decay_forgetting(tmp_path: Path):
    s = _store(tmp_path, half_life_days=1.0)
    keep, _ = s.add("用户要求记住：报销单走季度提交", user_id="u1", importance=0.9)
    gone, _ = s.add("用户本周在出差", user_id="u1", importance=0.5, ttl_days=1e-9)       # 已过期
    old, _ = s.add("用户上次随口提到喜欢蓝色", user_id="u1", importance=0.1)
    tbl = s._table()
    tbl.update(where=f"id = '{old.id}'", values_sql={"updated_at": repr(time.time() - 30 * 86400)})   # 30 天没用过
    time.sleep(0.01)
    assert not any(h["record"].id == gone.id for h in s.search("出差", user_id="u1", top_k=5))   # 过期的不召回
    assert s.forget(user_id="u1") == {"expired": 1, "decayed": 1}
    assert [r.id for r in s.list(user_id="u1")] == [keep.id]


def test_reembeds_when_embedder_changes(tmp_path: Path):
    s = _store(tmp_path, dim=64)
    s.add("用户在财务部", user_id="u1")
    s2 = _store(tmp_path, dim=96)                       # 换了向量模型（维度变化）
    assert s2.stats()["dim"] == 96 and s2.search("财务部", user_id="u1")[0]["record"].content == "用户在财务部"


def test_extraction_rules_and_llm():
    got = {c.content for c in extract_by_rules("我叫李明，我在财务部。请记住我的报销单都走季度提交。以后回答先给结论")}
    assert {"用户叫李明", "用户在财务部", "用户要求记住：我的报销单都走季度提交", "用户希望以后先给结论"} <= got
    assert extract_by_rules("年假可以跨年使用吗") == []                                    # 一次性问题不是记忆
    llm = FakeLLM(lambda p: '[{"content": "用户在财务部", "kind": "profile", "importance": 0.8}]')
    assert [c.content for c in extract_memories("我在财务部", "好的", llm=llm)] == ["用户在财务部"]
    loose = FakeLLM(lambda p: '```json\n[{"text": "用户叫李明。", "kind": "profile", "importance": 0.8}]\n```')   # 实测模型会把字段写成 text
    assert [c.content for c in extract_memories("我叫李明", llm=loose)] == ["用户叫李明"]
    assert "用户在财务部" in {c.content for c in extract_by_rules("我叫李明，在财务部做报销审核")}
    bad = FakeLLM(lambda p: "（不是 JSON）")
    assert extract_memories("我在财务部", llm=bad) == []                                  # LLM 说没有就没有
    assert [c.content for c in extract_memories("我在财务部", mode="rule")] == ["用户在财务部"]
