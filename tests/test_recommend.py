"""个性化推荐（LanceDB）：冷启动热度、提问 / 点赞驱动的兴趣向量、已看过不重复推、推荐理由、相似条目、换模型重算。"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("lancedb")

from prism.config import RecommendConfig  # noqa: E402
from prism.index.embeddings import HashEmbedder  # noqa: E402
from prism.recommend import LanceRecommender, RecItem  # noqa: E402

ITEMS = [
    RecItem("i-hotel", "kb", "section", "p1", "差旅管理 > 住宿标准", "出差住宿标准：一线城市每晚 600 元，其他城市 400 元，超标需审批"),
    RecItem("i-trip", "kb", "section", "p2", "差旅管理 > 差旅报销流程", "出差结束后 10 个工作日内提交差旅报销单，附住宿发票与交通票据"),
    RecItem("i-k8s", "kb", "section", "p3", "K8s 运维 > Ingress 502", "Ingress 返回 502 时先检查 upstream 健康检查与 Service 端口"),
    RecItem("i-mysql", "kb", "section", "p4", "数据库 > MySQL 1062", "MySQL 1062 主键冲突：INSERT IGNORE / ON DUPLICATE KEY UPDATE / 重试"),
    RecItem("i-leave", "kb", "question", "q1", "年假可以跨年使用吗", "年假可以跨年，最晚次年 3 月 31 日前休完"),
    RecItem("i-other", "kb2", "doc", "d9", "另一个知识库的文档", "出差住宿相关的另一份制度"),
]


def _rec(tmp_path: Path, dim: int = 128) -> LanceRecommender:
    r = LanceRecommender(tmp_path / "lancedb", HashEmbedder(dim=dim), RecommendConfig(mmr_lambda=0.9))
    r.upsert_items(ITEMS)
    return r


def test_cold_start_then_personalized(tmp_path: Path):
    r = _rec(tmp_path)
    r.log_event("u0", kind="like", item_id="i-mysql")                # 别人的点赞让它变"热门"
    cold = r.recommend("new-user", kb_id="kb", k=3)
    assert cold["mode"] == "popular" and cold["items"][0]["id"] == "i-mysql"

    r.log_event("u1", kind="ask", text="出差住宿标准能不能超额", kb_id="kb")
    rec = r.recommend("u1", kb_id="kb", k=3)
    assert rec["mode"] == "personal" and rec["items"][0]["id"] in {"i-hotel", "i-trip"}
    assert rec["items"][0]["reason"].startswith("因为你问过「出差住宿标准")
    assert all(it["kbId"] == "kb" for it in rec["items"])            # 按知识库过滤

    r.log_event("u1", kind="like", item_id=rec["items"][0]["id"])
    again = r.recommend("u1", kb_id="kb", k=5)
    assert rec["items"][0]["id"] not in {it["id"] for it in again["items"]}   # 看过 / 点过的不重复推荐
    r.log_event("u1", kind="dislike", item_id="i-k8s")
    assert "i-k8s" not in {it["id"] for it in r.recommend("u1", kb_id="kb", k=5)["items"]}
    assert r.recommend("u1", kb_id="kb", k=5)["profile"]["byKind"] == {"ask": 1, "like": 1, "dislike": 1}


def test_popularity_survives_resync_and_similar_items(tmp_path: Path):
    r = _rec(tmp_path)
    r.log_event("u1", kind="click", item_id="i-hotel")
    r.upsert_items(ITEMS)                                             # 重新同步条目：热度不清零
    assert r.get_items(["i-hotel"])["i-hotel"]["popularity"] == pytest.approx(0.6)
    sims = r.similar("i-hotel", k=2)
    assert sims[0]["id"] in {"i-trip", "i-other"} and "i-hotel" not in {s["id"] for s in sims}
    assert r.remove_items(kb_id="kb2") == 1 and r.stats()["items"] == 5


def test_reembeds_items_and_events_when_embedder_changes(tmp_path: Path):
    r = _rec(tmp_path, dim=128)
    r.log_event("u1", kind="ask", text="MySQL 主键冲突怎么办", kb_id="kb")
    r.log_event("u1", kind="click", item_id="i-mysql")
    r2 = LanceRecommender(tmp_path / "lancedb", HashEmbedder(dim=96), RecommendConfig(exclude_seen=False))
    assert r2.stats()["dim"] == 96 and r2.stats()["events"] == 2
    assert r2.recommend("u1", kb_id="kb", k=1)["items"][0]["id"] == "i-mysql"


def test_one_vote_per_user_and_item(tmp_path: Path):
    """赞 / 踩是评价：同一用户对同一条只能一次，赞踩二选一；别的用户、点开（不是评价）不受影响。"""
    from prism.recommend import AlreadyVoted

    r = _rec(tmp_path)
    r.log_event("u1", kind="like", item_id="i-hotel")
    with pytest.raises(AlreadyVoted) as exc:
        r.log_event("u1", kind="like", item_id="i-hotel")              # 同一条再点赞
    assert exc.value.vote == "like"
    with pytest.raises(AlreadyVoted):
        r.log_event("u1", kind="dislike", item_id="i-hotel")           # 点过赞就不能再点踩
    r.log_event("u2", kind="dislike", item_id="i-hotel")
    r.log_event("u1", kind="click", item_id="i-hotel")
    assert r.vote_of("u1", "i-hotel") == "like" and r.vote_of("u2", "i-hotel") == "dislike" and r.vote_of("u3", "i-hotel") is None
    assert r.get_items(["i-hotel"])["i-hotel"]["popularity"] == pytest.approx(1.0 + 0.6)   # 热度只加了一次点赞（+ 一次点开）


def test_simultaneous_votes_are_recorded_once(tmp_path: Path):
    """连点、赞踩一起点：几个请求同时到达，也只记第一次。"""
    import threading

    from prism.recommend import AlreadyVoted

    r = _rec(tmp_path)
    outcome: list[str] = []
    start = threading.Barrier(6)

    def vote(kind: str) -> None:
        start.wait()
        try:
            r.log_event("u1", kind=kind, item_id="i-trip")
            outcome.append(kind)
        except AlreadyVoted:
            outcome.append("rejected")

    threads = [threading.Thread(target=vote, args=(k,)) for k in ("like", "dislike", "like", "dislike", "like", "like")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(outcome) == 6 and sum(1 for o in outcome if o != "rejected") == 1
    assert len([e for e in r.events("u1") if e["kind"] in ("like", "dislike")]) == 1


def test_legacy_duplicate_votes_count_once_and_vote_is_returned(tmp_path: Path):
    """限制上线前连点留下的重复评价：只算最早那次；推荐结果带上"我的评价"。"""
    r = LanceRecommender(tmp_path / "lancedb", HashEmbedder(dim=128), RecommendConfig(mmr_lambda=0.9, exclude_seen=False))
    r.upsert_items(ITEMS)
    r.log_event("u1", kind="like", item_id="i-k8s")
    for k in ("like", "like", "dislike"):                               # 绕过查重直接写入，模拟旧数据
        r._log_event("u1", kind=k, item_id="i-k8s", kb_id="kb", text="", weight=None)
    assert [e["kind"] for e in r.events("u1") if e["item_id"] == "i-k8s"] == ["like"]
    rec = r.recommend("u1", kb_id="kb", k=6)
    assert rec["profile"]["byKind"] == {"like": 1}
    mine = {it["id"]: it["myVote"] for it in rec["items"]}
    assert mine.get("i-k8s") == "like" and all(v is None for k, v in mine.items() if k != "i-k8s")


def test_already_voted_maps_to_409_with_previous_vote():
    from fastapi import HTTPException

    from app.api.lance_apps import _call
    from prism.recommend import AlreadyVoted

    def boom():
        raise AlreadyVoted("i-hotel", "dislike")

    with pytest.raises(HTTPException) as exc:
        _call(boom)
    assert exc.value.status_code == 409 and exc.value.detail["vote"] == "dislike" and exc.value.detail["itemId"] == "i-hotel"


def test_dedupe_legacy_votes_and_recompute_popularity(tmp_path: Path):
    """清理限制上线前的重复评价：每人每条只留最早那次，热度按剩下的交互重算；预演不写入，再跑一次无事可做。"""
    r = _rec(tmp_path)
    r.log_event("u1", kind="like", item_id="i-k8s")
    r.log_event("u1", kind="click", item_id="i-k8s")
    for k in ("like", "like", "dislike"):                               # 旧数据：绕过查重直接写入
        r._log_event("u1", kind=k, item_id="i-k8s", kb_id="kb", text="", weight=None)
    r.log_event("u2", kind="like", item_id="i-k8s")
    assert r.get_items(["i-k8s"])["i-k8s"]["popularity"] == pytest.approx(4.6)    # 重复的点赞当时都加过热度

    plan = r.dedupe_votes(dry_run=True)
    assert [e["kind"] for e in plan["removed"]] == ["like", "like", "dislike"] and plan["popularity"]["i-k8s"] == pytest.approx((4.6, 2.6))
    assert r.get_items(["i-k8s"])["i-k8s"]["popularity"] == pytest.approx(4.6)    # 预演不写入

    done = r.dedupe_votes()
    assert len(done["removed"]) == 3
    assert r.get_items(["i-k8s"])["i-k8s"]["popularity"] == pytest.approx(2.6)    # u1 赞 1 + 点开 0.6 + u2 赞 1
    assert [e["kind"] for e in r.events("u1")] == ["click", "like"]
    assert r.dedupe_votes() == {"removed": [], "popularity": {}}
