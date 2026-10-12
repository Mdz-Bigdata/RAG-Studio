"""Agent 湖：写入去重、混合检索、版本 / Tag / 分支 / 派生列、检索质量与缺口、主题、只读 SQL、分析智能体、历史问答增强、API。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("lancedb")

from prism.generation import FakeLLM  # noqa: E402
from prism.generation.prompts import build_answer_prompt  # noqa: E402
from prism.index.embeddings import HashEmbedder  # noqa: E402
from prism.lake import AgentLake, LakeAnalyst, format_experience, recall_experience  # noqa: E402
from prism.lake import analytics  # noqa: E402

QA = [
    ("年假可以休几天", "正式员工每年 5 天年假 [1]", 0.82, "ok", 0),
    ("年假最多可以休几天", "工作满 10 年最多 10 天 [1]", 0.75, "ok", 1),
    ("差旅住宿报销标准是多少", "一线城市 600 元/晚 [2]", 0.66, "ok", 0),
    ("出差住酒店能报多少钱", "根据现有资料无法回答", 0.01, "refused", 0),
    ("笔记本电脑保修多久", "整机保修两年 [1]", 0.4, "ok", -1),
    ("Kafka 消费积压怎么处理", "", None, "refused", 0),
]


def _lake(tmp_path: Path, dim: int = 128) -> AgentLake:
    return AgentLake(tmp_path / "lancedb", HashEmbedder(dim=dim), model_name=f"hash{dim}")


def _seed(lake: AgentLake, now: float | None = None) -> None:
    now = now or time.time()
    recs = []
    for i, (q, a, score, status, fb) in enumerate(QA):
        ts = now - (len(QA) - i) * 3600
        recs.append({"id": f"q{i}", "kind": "question", "session_id": f"s{i}", "kb_id": "kb1", "content": q, "ts": ts})
        recs.append({"id": f"r{i}", "kind": "retrieval", "session_id": f"s{i}", "kb_id": "kb1", "parent_id": f"q{i}", "content": q, "score": score,
                     "latency_ms": 100 + i * 50, "status": "ok" if score is not None else "no_context", "ts": ts + 1,
                     "meta": {"strategy": "hybrid", "contexts": [{"docId": "d1", "fileName": "员工手册.md", "score": score}] if score is not None else []}})
        recs.append({"id": f"a{i}", "kind": "answer", "session_id": f"s{i}", "kb_id": "kb1", "parent_id": f"q{i}", "content": a or "根据现有知识库资料无法回答",
                     "status": status, "feedback": fb, "refs": ["d1"] if status == "ok" else [], "ts": ts + 2})
    lake.append(recs)


def test_append_dedup_search_and_model_change(tmp_path: Path):
    lake = _lake(tmp_path)
    _seed(lake)
    assert lake.info()["rows"] == 18
    assert lake.append([{"id": "q0", "kind": "question", "content": "改过的内容"}]) == {"inserted": 0, "updated": 0, "skipped": 1}   # 默认按 id 去重
    assert lake.get("q0")["content"] == "年假可以休几天"
    lake.append([{"id": "q0", "kind": "question", "content": "年假可以休几天？", "session_id": "s0", "kb_id": "kb1"}], upsert=True)
    assert lake.get("q0")["content"] == "年假可以休几天？"

    hits = lake.search("年假 几天", top_k=3, where=lake.where_clause(kinds=["question"]))
    assert hits and all(h["kind"] == "question" for h in hits) and "年假" in hits[0]["content"]
    assert set(hits[0]["routes"]) <= {"vector", "fts"} and hits[0]["vectorScore"] is not None
    items, total = lake.list(where=lake.where_clause(kinds=["answer"]), limit=2)
    assert total == 6 and items[0]["ts"] >= items[1]["ts"]                                   # 按时间倒序
    with pytest.raises(ValueError):
        lake.append([{"kind": "bogus", "content": "x"}])

    lake2 = AgentLake(tmp_path / "lancedb", HashEmbedder(dim=64), model_name="hash64")      # 换向量模型：自动重算
    assert lake2.info()["rows"] == 18 and lake2.search("报销", top_k=1, mode="vector")


def test_versions_tags_branches_and_derived_columns(tmp_path: Path):
    lake = _lake(tmp_path)
    _seed(lake)
    tag = lake.create_tag("baseline")
    lake.append([{"kind": "note", "content": "新增一条笔记"}])
    assert lake.view(version="baseline").count_rows() == 18 and lake.info()["rows"] == 19     # 时间旅行
    assert any("baseline" in v["tags"] for v in lake.versions())

    lake.create_branch("exp", from_tag="baseline")
    lake.add_column("quality", expr="CASE WHEN score < 0.05 THEN 'low' ELSE 'ok' END", branch="exp")
    lake.add_column("topic", values={"q0": "休假", "q1": "休假", "q2": "差旅"}, branch="exp")
    assert "topic" not in lake.columns()                                                      # 主表不受影响
    diff = lake.diff_branch("exp")
    assert diff["addedColumns"] == ["quality", "topic"] and diff["rowsOnlyInMain"] == 1
    assert {"value": "休假", "count": 2} in diff["columnValues"]["topic"]

    res = lake.promote_branch("exp", ["topic"])
    assert res["updatedRows"] == 18 and lake.get("q2")["topic"] == "差旅" and lake.get("q2")["content"]
    assert "quality" not in lake.columns()
    lake.append([{"kind": "note", "content": "合并后继续写入"}])                                # 有派生列也能正常写入
    assert lake.info()["rows"] == 20
    with pytest.raises(ValueError):
        lake.add_column("content", values={})
    with pytest.raises(ValueError):
        lake.drop_column("kind")
    before_drop = lake.info()["version"]
    lake.drop_column("topic")
    assert "topic" not in lake.columns() and tag["version"] < lake.info()["version"]
    assert "topic" in lake.view(version=before_drop).schema.names                              # 删列前的版本仍能时间旅行查看
    with pytest.raises(ValueError, match="没有列"):
        lake.drop_column("topic")                                                              # 不存在：报错而不是悄悄成功



def test_delete_versions_upto(tmp_path: Path):
    """点某个历史版本的删除 = 删它及更早的全部版本：最新版本不可删，有 Tag 要先删 Tag，分支起点保留，数据不变。"""
    lake = _lake(tmp_path)
    _seed(lake)
    for i in range(4):                                       # 连续写入：相邻版本只差几毫秒
        lake.append([{"kind": "note", "content": f"笔记 {i}"}])
    vs = sorted(v["version"] for v in lake.versions(200))
    rows, latest = lake.info()["rows"], vs[-1]
    target = vs[-3]

    with pytest.raises(ValueError, match="最新版本"):
        lake.delete_versions(latest)
    lake.create_tag("keep", target)
    with pytest.raises(ValueError, match="keep"):
        lake.delete_versions(target)
    lake.delete_tag("keep")
    lake.create_branch("exp", from_version=vs[1])

    res = lake.delete_versions(target)
    left = {v["version"] for v in lake.versions(200)}
    assert set(res["deleted"]) == {v for v in vs if v <= target} - {vs[1]}       # 恰好删到 target，一个不多
    assert res["kept"] == [vs[1]] and vs[1] in left                              # 分支起点被保留
    assert {v for v in vs if v > target} <= left
    assert lake.info()["rows"] == rows and lake.view(branch="exp").count_rows() > 0
    assert not lake._table().tags.list()                                         # 临时保护 Tag 已移除
    with pytest.raises(ValueError, match="不存在"):
        lake.delete_versions(vs[0])



def test_delete_records_cascades_and_stays_deleted(tmp_path: Path):
    """删问题连同检索日志 / 回答；删掉的记录不会被「回填历史数据」（按 id 去重的 append）写回来；删回答不影响问题。"""
    lake = _lake(tmp_path)
    _seed(lake)
    rows = lake.info()["rows"]
    assert lake.delete_records(["q0"], dry_run=True) == ["a0", "q0", "r0"] and lake.info()["rows"] == rows   # 预览不删
    assert lake.delete_records(["q0"]) == ["a0", "q0", "r0"]
    assert lake.info()["rows"] == rows - 3 and lake.get("q0") is None and lake.get("a0") is None
    assert lake.delete_records(["a1"], cascade=True) == ["a1"] and lake.get("q1") is not None            # 只删后代，不删父记录
    assert lake.delete_records(["nope"]) == []

    _seed(lake)                                                       # 再回填一遍同样的数据
    assert lake.get("q0") is None and lake.get("a1") is None and lake.info()["rows"] == rows - 4
    res = lake.append([{"id": "q0", "kind": "question", "content": "又来了"}, {"kind": "note", "content": "全新的记录照常写入"}])
    assert res["inserted"] == 1 and res["skipped"] == 1


def test_quality_gaps_topics(tmp_path: Path):
    lake = _lake(tmp_path)
    _seed(lake)
    df = analytics.frame(lake.to_arrow())
    ov = analytics.overview(df)
    assert ov["rows"] == 18 and ov["kinds"]["question"] == 6 and ov["sessions"] == 6
    q = analytics.retrieval_quality(df, low_score=0.05)
    reasons = {g["question"]: g["reasons"] for g in q["gaps"]}
    assert "出差住酒店能报多少钱" in reasons and "回答拒答" in reasons["出差住酒店能报多少钱"]
    assert reasons["Kafka 消费积压怎么处理"][0] == "未检索到资料"
    assert reasons["笔记本电脑保修多久"] == ["用户点踩"]
    assert "年假可以休几天" not in reasons
    assert q["feedback"] == {"up": 1, "down": 1} and q["refused"] == 2 and q["topDocs"][0]["fileName"] == "员工手册.md"
    assert q["latency"]["p50"] is not None and q["byStrategy"][0]["strategy"] == "hybrid"

    t = analytics.topics(analytics.frame(lake.to_arrow(with_vector=True)), k=3, gaps={g["questionId"] for g in q["gaps"]})
    assert t["k"] == 3 and sum(c["size"] for c in t["clusters"]) == 6 and len(t["assignments"]) == 6
    assert all(c["keywords"] for c in t["clusters"])
    assert all("questionIds" not in c for c in t["clusters"])                                  # 默认不带 id（给智能体的结果要精简）
    t2 = analytics.topics(analytics.frame(lake.to_arrow(with_vector=True)), k=3, with_ids=True)
    assert sorted(i for c in t2["clusters"] for i in c["questionIds"]) == [f"q{i}" for i in range(6)]
    assert all(len(c["questionIds"]) == c["size"] for c in t2["clusters"])

    # 忽略缺口：不再算缺口（缺口数 / 缺口率都变），单独列在 ignoredGaps；可恢复
    assert lake.set_gap_ignored("q4") and not lake.set_gap_ignored("a4") and not lake.set_gap_ignored("nope")
    q2 = analytics.retrieval_quality(analytics.frame(lake.to_arrow()), low_score=0.05)
    assert q2["gapCount"] == q["gapCount"] - 1 and "q4" not in {g["questionId"] for g in q2["gaps"]}
    assert [g["questionId"] for g in q2["ignoredGaps"]] == ["q4"] and q2["ignoredCount"] == 1
    assert lake.set_gap_ignored("q4", False)
    q3 = analytics.retrieval_quality(analytics.frame(lake.to_arrow()), low_score=0.05)
    assert q3["gapCount"] == q["gapCount"] and q3["ignoredCount"] == 0


def test_sql_is_read_only(tmp_path: Path):
    lake = _lake(tmp_path)
    _seed(lake)
    tables = {"lake": lake.to_arrow()}
    r = analytics.run_sql(tables, "SELECT kind, count(*) AS n FROM lake GROUP BY kind ORDER BY kind")
    assert r["columns"] == ["kind", "n"] and dict(map(tuple, r["rows"]))["question"] == 6
    r = analytics.run_sql(tables, "select json_extract_string(meta, '$.strategy') s, avg(score) from lake where kind = 'retrieval' group by 1")
    assert r["rows"][0][0] == "hybrid"
    assert analytics.run_sql(tables, "SELECT * FROM lake", limit=5)["truncated"]
    assert analytics.run_sql(tables, "SELECT 'drop table lake' AS s")["rows"] == [["drop table lake"]]   # 字符串里的关键字不算
    for bad in ["DELETE FROM lake", "SELECT 1; SELECT 2", "COPY lake TO '/tmp/x.csv'", "ATTACH '/tmp/x.db'", "SELECT * FROM read_csv('/etc/hosts')", "PRAGMA version"]:
        with pytest.raises(Exception):
            analytics.run_sql(tables, bad)


def test_analyst_rule_and_llm(tmp_path: Path):
    lake = _lake(tmp_path)
    _seed(lake)
    out = LakeAnalyst(lake, None).run("最近哪些问题没答上来？")
    assert out["planner"] == "rule" and out["steps"][0]["tool"] == "knowledge_gaps" and "出差住酒店" in out["answer"]
    rec = lake.get(out["recordId"])
    assert rec["kind"] == "analysis" and lake.list(where=lake.where_clause(kinds=["tool_call"], parent_ids=[out["recordId"]]))[1] == 1

    def responder(prompt: str) -> str:
        if prompt.startswith("用户问题："):
            return "结论：共 6 个问题。"
        if "原 SQL" in prompt:
            return "```sql\nSELECT count(*) AS n FROM lake WHERE kind = 'question'\n```"
        return json.dumps({"steps": [{"tool": "sql", "args": {"query": "SELECT count(*) FROM lake WHERE knd = 'question'"}, "why": "统计"}]})

    llm = FakeLLM(responder)
    out = LakeAnalyst(lake, llm).run("一共有多少个问题？", record=False)
    step = out["steps"][0]
    assert out["planner"] == "llm" and step["error"] is None and step["result"]["rows"] == [[6]] and "fixedFrom" in step["args"]   # SQL 报错后由模型修正
    assert out["answer"] == "结论：共 6 个问题。" and all(t is False for t in llm.thinking)


def test_experience_recall_filters(tmp_path: Path):
    lake = _lake(tmp_path)
    _seed(lake)
    items = recall_experience(lake, "年假可以休几天", kb_id="kb1", session_id="other", top_k=3, min_score=0.3)
    qs = [i["question"] for i in items]
    assert "年假可以休几天" in qs and "[1]" not in items[0]["answer"]                        # 去掉了历史回答里的引用标记
    assert "出差住酒店能报多少钱" not in qs and "笔记本电脑保修多久" not in qs                     # 拒答 / 被点踩的不带
    assert not recall_experience(lake, "年假可以休几天", session_id="s0", min_score=0.99)       # 同一会话不重复带
    block = format_experience(items)
    prompt = build_answer_prompt("年假几天", [], reference_notes=block)
    assert "<历史问答参考>" in prompt and prompt.index("<历史问答参考>") > prompt.index("</资料>") and "[1] 来源" not in prompt


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    data_dir = tmp_path_factory.mktemp("lakeapp")
    os.environ["SQLITE_URL"] = f"sqlite:///{data_dir}/app.db"
    from fastapi.testclient import TestClient

    from app import config as app_config_mod

    old = app_config_mod.app_config.data_dir
    app_config_mod.app_config.data_dir = data_dir
    from app.main import app

    with TestClient(app) as c:
        yield c
    app_config_mod.app_config.data_dir = old


def test_lake_api(client):
    from app.services.lake_service import lance_dir, lake_service

    assert "lakeapp" in str(lance_dir())
    lake = lake_service.lake()
    _seed(lake)
    ov = client.get("/api/v1/lake/overview").json()
    assert ov["info"]["rows"] >= 18 and ov["stats"]["kinds"]["question"] >= 6
    assert client.get("/api/v1/lake/quality").json()["gapCount"] >= 3
    r = client.post("/api/v1/lake/search", json={"query": "报销", "kinds": ["question"], "topK": 3}).json()
    assert r and r[0]["kind"] == "question"
    assert client.post("/api/v1/lake/sql", json={"sql": "select count(*) n from lake"}).json()["rows"][0][0] >= 18
    assert client.post("/api/v1/lake/sql", json={"sql": "drop table lake"}).status_code == 400
    assert client.post("/api/v1/lake/feedback", json={"answerId": "a0", "value": 1}).json()["ok"]
    assert client.post("/api/v1/lake/feedback", json={"answerId": "nope", "value": 1}).status_code == 404
    assert client.post("/api/v1/lake/tags", json={"name": "t1"}).status_code == 200
    assert client.post("/api/v1/lake/branches", json={"name": "b1", "fromTag": "t1"}).status_code == 200
    t = client.post("/api/v1/lake/topics", json={"k": 2, "writeColumn": "topic", "branch": "b1"}).json()
    assert t["written"]["filled"] == 6
    assert client.get("/api/v1/lake/branches/b1/diff").json()["addedColumns"] == ["topic"]
    assert client.post("/api/v1/lake/branches/b1/promote", json={}).json()["columns"] == ["topic"]
    v = client.get("/api/v1/lake/versions").json()
    assert v["tags"][0]["name"] == "t1" and v["branches"][0]["name"] == "b1"
    assert "topic" in v["derivedColumns"]                                                      # 合并回主表的派生列
    assert client.delete("/api/v1/lake/columns/topic").status_code == 200
    assert client.delete("/api/v1/lake/columns/topic").status_code == 400
    assert "topic" not in client.get("/api/v1/lake/versions").json()["derivedColumns"]
    preview = client.delete("/api/v1/lake/records/q1", params={"dryRun": True}).json()
    assert preview["dryRun"] and set(preview["deleted"]) >= {"q1"} and client.get("/api/v1/lake/records/q1").status_code == 200
    assert "q1" in client.delete("/api/v1/lake/records/q1").json()["deleted"]
    assert client.get("/api/v1/lake/records/q1").status_code == 404 and client.delete("/api/v1/lake/records/q1").status_code == 404
    assert client.post("/api/v1/lake/gaps/q3/ignore", json={"ignored": True}).status_code == 200
    qual = client.get("/api/v1/lake/quality").json()
    assert "q3" in {g["questionId"] for g in qual["ignoredGaps"]} and "q3" not in {g["questionId"] for g in qual["gaps"]}
    assert client.post("/api/v1/lake/gaps/q3/ignore", json={"ignored": False}).status_code == 200
    assert client.post("/api/v1/lake/gaps/nope/ignore", json={}).status_code == 404
    assert all("questionIds" in c for c in client.post("/api/v1/lake/topics", json={"k": 2}).json()["clusters"])
    batch = client.post("/api/v1/lake/records/delete", json={"ids": ["q4", "q5"], "dryRun": True}).json()
    assert {"q4", "q5", "a4", "a5"} <= set(batch["deleted"]) and client.get("/api/v1/lake/records/q4").status_code == 200
    assert {"q4", "q5"} <= set(client.post("/api/v1/lake/records/delete", json={"ids": ["q4", "q5"]}).json()["deleted"])
    assert client.post("/api/v1/lake/records/delete", json={"ids": ["q4"]}).status_code == 404
    oldest = v["versions"][-1]["version"]
    assert client.delete(f"/api/v1/lake/versions/{v['versions'][0]['version']}").status_code == 400     # 最新版本不可删
    assert client.delete("/api/v1/lake/branches/b1").status_code == 200
    assert client.delete("/api/v1/lake/tags/t1").status_code == 200
    assert client.delete("/api/v1/lake/branches/b1").status_code == 400 and client.delete("/api/v1/lake/tags/t1").status_code == 400   # 不存在：400 而不是 500
    v = client.get("/api/v1/lake/versions").json()
    assert v["tags"] == [] and v["branches"] == []
    assert oldest in client.delete(f"/api/v1/lake/versions/{oldest}").json()["deleted"]
    detail = client.get("/api/v1/lake/records/q0").json()
    assert {c["kind"] for c in detail["children"]} == {"retrieval", "answer"}
    a = client.post("/api/v1/lake/analyze", json={"question": "问题的主题分布"}).json()
    assert a["steps"][0]["tool"] == "topics" and a["recordId"]
    assert "chat" in client.post("/api/v1/lake/sync", json={}).json()


def test_retest_marks_resolved_and_keeps_open_gaps(tmp_path: Path):
    from prism.lake import run_retest

    lake = _lake(tmp_path)
    _seed(lake)
    calls = []

    def fake(question, kb_id, check_answer):
        calls.append((question, kb_id, check_answer))
        if "出差住酒店" in question:                       # 补了语料：检索到、回答不拒答
            return {"score": 0.9, "contexts": [{"docId": "d2", "fileName": "差旅制度.md", "score": 0.9}], "refused": False, "answer": "一线城市 600 元/晚 [1]"}
        if "保修" in question:                           # 检索到了，但回答仍然拒答（资料里没有）
            return {"score": 0.5, "contexts": [{"docId": "d1", "fileName": "员工手册.md", "score": 0.5}], "refused": True, "answer": "根据现有资料无法回答"}
        return {"score": None, "contexts": [], "refused": True, "answer": ""}

    df = analytics.frame(lake.to_arrow())
    before = analytics.retrieval_quality(df)
    assert before["gapCount"] == 3 and before["resolvedCount"] == 0
    out = run_retest(lake, df, fake, include_weak=False, limit=10)
    assert out["tested"] == 3 and out["counts"] == {"resolved": 1, "improved": 0, "answerable": 0, "still_gap": 2, "wrong_kb": 0, "error": 0}
    assert all(kb == "kb1" and check for _, kb, check in calls)
    lake.append([{"kind": "doc_event", "kb_id": "kb1", "title": "差旅制度.md", "content": "差旅制度.md：校对回灌，+3 行"}])

    after = analytics.retrieval_quality(analytics.frame(lake.to_arrow()))
    assert after["gapCount"] == 2 and after["resolvedCount"] == 1 and after["historicalGapCount"] == 3
    assert after["resolvedGaps"][0]["question"] == "出差住酒店能报多少钱" and after["resolvedGaps"][0]["retest"]["status"] == "resolved"
    still = {g["question"]: g for g in after["gaps"]}
    assert still["笔记本电脑保修多久"]["retest"]["status"] == "still_gap"           # 检索到但仍拒答：不算解决
    assert still["Kafka 消费积压怎么处理"]["docChangesAfter"] == ["差旅制度.md"]
    assert len(lake.list(where=lake.where_clause(kinds=["retest"]))[0]) == 3

    # 低分问题：答了但检索分 < 0.3 的也可以一起复测；分数明显上升记为"有改善"
    weak = run_retest(lake, analytics.frame(lake.to_arrow()), lambda q, kb, c: {"score": 0.2, "contexts": [{"docId": "d1", "fileName": "x", "score": 0.2}], "refused": False, "answer": "ok"},
                      include_weak=True, check_answer=False, limit=20)
    assert any(r["type"] == "weak" and r["status"] == "improved" for r in weak["results"])


def test_analyst_retest_tool(tmp_path: Path):
    lake = _lake(tmp_path)
    _seed(lake)
    fn = lambda q, kb, c: {"score": 0.9, "contexts": [{"docId": "d", "fileName": "f.md", "score": 0.9}], "refused": False, "answer": "已能回答"}   # noqa: E731
    out = LakeAnalyst(lake, None, retest_fn=fn).run("之前的缺口现在补上了吗？", record=False)
    assert [s["tool"] for s in out["steps"]][:2] == ["retest_gaps", "knowledge_gaps"]
    assert out["steps"][0]["result"]["counts"]["resolved"] == 3 and out["steps"][1]["result"]["gapCount"] == 0
    assert "复测" in out["answer"]
    no_fn = LakeAnalyst(lake, None).run("缺口补上了吗", record=False)            # 没有检索服务：不规划复测
    assert "retest_gaps" not in [s["tool"] for s in no_fn["steps"]]


def test_markdown_diff_summary():
    from prism.lake.changes import describe_change, markdown_diff

    d = markdown_diff("# 报告\n\n毛利率见图2。\n", "# 报告\n\n毛利率见图2。\n\n| 季度 | 毛利率 |\n| 25Q3 | 45.3% |\n")
    assert d["changed"] and d["addedLines"] == 2 and d["removedLines"] == 0 and "| 25Q3 | 45.3% |" in d["added"]
    text = describe_change("经营报告.pdf", "edit", d)
    assert text.startswith("经营报告.pdf：校对回灌，+2 行 / -0 行") and "新增：" in text
    assert "没有变化" in describe_change("a.md", "reprocess_keep_edits", markdown_diff("x", "x"))


def test_retest_flags_answer_in_other_kb(tmp_path: Path):
    """本库答不上、别的知识库能答：记为 wrong_kb（选错库 / 路由问题），不算缺语料，单独计数。"""
    from prism.lake import run_retest

    lake = _lake(tmp_path)
    _seed(lake)

    def fake(question, kb_id, check_answer):
        alt = {"kbId": "kb2", "name": "Kafka 运维库", "score": 0.88, "fileName": "kafka.md", "answer": "扩容分区 [1]"} if "Kafka" in question else None
        return {"score": None, "contexts": [], "refused": True, "answer": "", "altKb": alt}

    out = run_retest(lake, analytics.frame(lake.to_arrow()), fake)
    assert out["counts"]["wrong_kb"] == 1 and out["counts"]["still_gap"] == 2
    q = analytics.retrieval_quality(analytics.frame(lake.to_arrow()))
    assert q["misroutedCount"] == 1 and q["gapCount"] == 2
    g = q["misroutedGaps"][0]
    assert g["question"] == "Kafka 消费积压怎么处理" and g["retest"]["status"] == "wrong_kb" and g["retest"]["altKb"]["name"] == "Kafka 运维库"
    kg = LakeAnalyst(lake, None).call("knowledge_gaps", {})
    assert kg["misroutedCount"] == 1 and kg["misrouted"][0]["answerInKb"] == "Kafka 运维库"


def test_weak_question_answerable_with_low_score():
    """当时检索分偏低的问题：分数仍低但检索到资料、回答不拒答 → 能回答（不算缺口）；拒答 / 没检查回答 → 仍是缺口。"""
    from prism.lake.retest import classify

    weak = {"type": "weak", "oldScore": 0.096}
    ok = {"score": 0.096, "contexts": [{"docId": "d"}], "refused": False}
    assert classify(weak, ok, low_score=0.05, weak_score=0.3) == "answerable"
    assert classify(weak, {**ok, "refused": None}, low_score=0.05, weak_score=0.3) == "still_gap"       # 没生成回答：判断不了
    assert classify(weak, {**ok, "refused": True}, low_score=0.05, weak_score=0.3) == "still_gap"
    assert classify(weak, {**ok, "score": 0.2}, low_score=0.05, weak_score=0.3) == "improved"           # 分数明显上升
    assert classify(weak, {**ok, "score": 0.6}, low_score=0.05, weak_score=0.3) == "resolved"
    assert classify(weak, {**ok, "score": 0.01}, low_score=0.05, weak_score=0.3) == "still_gap"         # 低于判定阈值：等于没检索到


def test_routing_tool_and_kb_profiles(tmp_path: Path):
    """路由工具：兜底次数与明细；"答案在其它知识库"的问题之后再问是否已答对；知识库画像由调用方提供。"""
    from prism.lake import run_retest

    lake = _lake(tmp_path)
    _seed(lake)
    now = time.time()
    # Kafka 问题在 kb1 答不上，复测发现在 kb2 能答 → 答案在其它知识库
    run_retest(lake, analytics.frame(lake.to_arrow()), lambda q, kb, c: {"score": None, "contexts": [], "refused": True, "answer": "",
                                                                        "altKb": {"kbId": "kb2", "name": "运维库", "score": 0.9} if "Kafka" in q else None})
    # 之后用自动路由再问：先路由到 kb1，兜底换到 kb2 并答对
    lake.append([
        {"id": "q9", "kind": "question", "kb_id": "kb2", "content": "Kafka 消费积压怎么处理？", "ts": now + 10},
        {"id": "r9", "kind": "retrieval", "kb_id": "kb2", "parent_id": "q9", "title": "Kafka 消费积压怎么处理？", "content": "x", "score": 0.9, "status": "ok", "ts": now + 11,
         "meta": {"routeFallback": {"fromKbId": "kb1", "fromName": "员工制度库", "toKbId": "kb2", "toName": "运维库", "fromScore": 0.003, "toScore": 0.9, "reason": "精排最高分 0.003 低于阈值 0.1"}}},
        {"id": "a9", "kind": "answer", "kb_id": "kb2", "parent_id": "q9", "content": "扩容分区 [1]", "status": "ok", "ts": now + 12},
    ])
    profiles = [{"kbId": "kb1", "name": "员工制度库", "description": "年假、报销", "category": "制度", "docCount": 1, "documents": ["员工手册.md"]},
                {"kbId": "kb2", "name": "运维库", "description": "Kafka、K8s", "category": "运维", "docCount": 1, "documents": ["kafka.md"]}]
    agent = LakeAnalyst(lake, None, kb_info_fn=lambda: profiles)
    rt = agent.call("routing", {})
    assert rt["fallbackCount"] == 1 and rt["fallbacks"][0]["from"] == "员工制度库" and rt["fallbacks"][0]["to"] == "运维库"
    mis = rt["misrouted"][0]
    assert mis["question"] == "Kafka 消费积压怎么处理" and mis["askedIn"] == ["员工制度库"] and mis["status"] == "之后再问已答对（靠路由兜底）"
    assert rt["questionsByKb"]["员工制度库"] == 6
    assert agent.call("kb_profiles", {})[1]["description"] == "Kafka、K8s"
    kg = agent.call("knowledge_gaps", {})
    assert kg["misrouted"][0]["kb"] == "员工制度库" and "retestScore" in kg["gaps"][0]

    out = agent.run("年假问题为什么会选错知识库？知识库描述要不要改？", record=False)
    assert [s["tool"] for s in out["steps"]][:2] == ["routing", "kb_profiles"]
    with pytest.raises(ValueError):
        LakeAnalyst(lake, None).call("kb_profiles", {})
