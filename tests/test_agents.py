"""多智能体协作（LanceDB 黑板）：离线也能跑完整流程；审校判定证据不足时改写重查；黑板可回放、可检索相似任务。"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("lancedb")

from prism.agents import AgentTeam, Blackboard  # noqa: E402
from prism.config import AgentsConfig  # noqa: E402
from prism.generation import FakeLLM  # noqa: E402
from prism.index.embeddings import HashEmbedder  # noqa: E402

CORPUS = {
    "远程": [{"chunkId": "c-remote", "docId": "d1", "label": "员工手册 > 通讯补贴", "text": "远程办公员工每月通讯补贴 200 元，随工资发放", "score": 0.9}],
    "通勤": [{"chunkId": "c-commute", "docId": "d1", "label": "员工手册 > 交通补贴", "text": "本地通勤员工每月交通补贴 300 元，需上传通勤记录", "score": 0.88}],
    "发放": [{"chunkId": "c-pay", "docId": "d1", "label": "员工手册 > 补贴发放", "text": "各类补贴次月 10 日随工资一并发放", "score": 0.8}],
}


def search(query: str, top_k: int) -> list[dict]:
    return [h for key, hits in CORPUS.items() if key in query for h in hits][:top_k]


def _events(team: AgentTeam, q: str, **kw) -> list[dict]:
    return list(team.run(q, user_id="u1", kb_id="kb", **kw))


def test_offline_team_runs_end_to_end(tmp_path: Path):
    board = Blackboard(tmp_path / "lancedb", HashEmbedder(dim=64))
    evs = _events(AgentTeam(board, search), "远程员工通讯补贴和本地通勤交通补贴有什么不同")
    kinds = [e["type"] for e in evs]
    assert kinds[0] == "run" and kinds[-1] == "final" and {"plan", "evidence", "recall", "analysis", "review"} <= set(kinds)
    final = evs[-1]
    assert final["data"]["sources"] and final["data"]["sources"][0]["n"] == 1 and "[1]" in final["content"]
    run_id = evs[0]["data"]["runId"]
    assert [r["kind"] for r in board.history(run_id)][:2] == ["question", "plan"] and board.history(run_id)[-1]["kind"] == "answer"
    assert board.runs()[0]["runId"] == run_id


def test_reviewer_triggers_requery_and_writer_cites(tmp_path: Path):
    reviews = iter(['{"sufficient": false, "missing": ["发放时间"], "followups": ["补贴什么时候发放"], "comment": "缺发放时间"}',
                    '{"sufficient": true, "missing": [], "followups": [], "comment": "证据充分"}'])

    def responder(p: str) -> str:
        if "你是规划者" in p:
            return '{"subtasks": ["远程员工通讯补贴标准", "本地通勤员工交通补贴标准"], "reason": "两类补贴分别查"}'
        if "你是审校" in p:
            return next(reviews)
        if "<分析>" in p:
            return "通讯补贴每月 200 元 [1]，交通补贴每月 300 元 [2]，都在次月 10 日发放 [3]。"
        return "- 远程：200 元 [1]\n- 通勤：300 元 [2]\n- 发放：次月 10 日 [3]"

    llm = FakeLLM(responder)
    board = Blackboard(tmp_path / "lancedb", HashEmbedder(dim=64))
    team = AgentTeam(board, search, llm=llm, cfg=AgentsConfig(max_review_rounds=2), memory_recall=lambda q: ["用户在财务部"])
    evs = _events(team, "远程员工和通勤员工的补贴有什么不同")
    kinds = [e["type"] for e in evs]
    assert kinds.count("review") == 2 and "requery" in kinds and "memory" in kinds
    assert next(e for e in evs if e["type"] == "requery")["data"]["queries"] == ["补贴什么时候发放"]
    final = evs[-1]
    assert final["data"]["citations"] == [1, 2, 3] and final["data"]["sufficient"] is True
    assert {s["chunkId"] for s in final["data"]["sources"]} == {"c-remote", "c-commute", "c-pay"}
    assert False in llm.thinking                                        # 分析 / 撰稿不开推理模型的思考

    again = _events(AgentTeam(board, search), "远程员工和通勤员工的补贴有什么区别")
    assert again[0]["data"]["similarRuns"][0]["runId"] == evs[0]["data"]["runId"]   # 跨运行检索相似任务


def test_delete_runs_single_batch_and_clear(tmp_path: Path):
    board = Blackboard(tmp_path / "lancedb", HashEmbedder(dim=64))
    team = AgentTeam(board, search)
    ids = [_events(team, q)[0]["data"]["runId"] for q in ("远程员工通讯补贴", "本地通勤交通补贴", "补贴怎么发放")]
    assert {r["runId"] for r in board.runs()} == set(ids)

    assert board.delete_runs([ids[0]]) > 0 and board.history(ids[0]) == []          # 单条：整次运行的黑板记录都删掉
    assert board.delete_runs([ids[0]]) == 0
    assert {r["runId"] for r in board.runs()} == set(ids[1:])
    assert board.delete_runs(ids[1:2]) > 0 and [r["runId"] for r in board.runs()] == [ids[2]]   # 批量
    assert board.delete_runs([]) == 0

    _events(team, "远程员工通讯补贴")
    assert board.clear() > 0 and board.runs() == [] and board.info()["rows"] == 0   # 全部
    assert _events(team, "补贴怎么发放")[-1]["type"] == "final"                     # 清空后还能继续写
