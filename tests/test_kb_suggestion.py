"""问答页的选库提示：手动选的知识库里一条资料都没检索到、别的知识库能答时，plan 里给出 kbSuggestion；选自动路由时不给。"""

from __future__ import annotations

import io
import json
import os

import pytest

LEAVE = "# 员工手册\n\n## 年假\n\n员工每年享有 5 天带薪年假，工作满 3 年后增加至 10 天。年假需提前 3 个工作日在 OA 系统提交申请。\n"
KAFKA = "# Kafka 运维\n\n## 消费积压\n\n消费积压时先扩容分区与消费者实例，再排查慢消费。\n"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("kbsuggest")
    os.environ.setdefault("SQLITE_URL", f"sqlite:///{data_dir}/app.db")
    from fastapi.testclient import TestClient

    from app import config as app_config_mod

    cfg = app_config_mod.app_config
    old = (cfg.data_dir, cfg.upload_dir, cfg.index_root)
    cfg.data_dir, cfg.upload_dir, cfg.index_root = data_dir, data_dir / "uploads", data_dir / "indexes"
    for d in (cfg.upload_dir, cfg.index_root):
        d.mkdir(parents=True, exist_ok=True)
    from app.main import app

    with TestClient(app) as c:
        yield c
    cfg.data_dir, cfg.upload_dir, cfg.index_root = old


def _plan(client, kb_id: str, message: str) -> dict:
    with client.stream("POST", "/api/v1/chat/stream", json={"kbId": kb_id, "message": message}) as r:
        ev = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:") and ev == "plan":
                return json.loads(line[5:])
    raise AssertionError("没有 plan 事件")


def test_suggests_other_kb_when_selected_kb_has_nothing(client, monkeypatch):
    hr = client.post("/api/v1/knowledge-bases/", json={"name": "人事制度库", "description": "员工手册、年假、请假、考勤等人事制度", "category": "制度"}).json()
    ops = client.post("/api/v1/knowledge-bases/", json={"name": "运维手册库", "description": "Kafka、K8s 运维排障", "category": "运维"}).json()
    client.post(f"/api/v1/documents/upload?kbId={hr['id']}", files={"file": ("员工手册.md", io.BytesIO(LEAVE.encode()), "text/markdown")})
    client.post(f"/api/v1/documents/upload?kbId={ops['id']}", files={"file": ("kafka.md", io.BytesIO(KAFKA.encode()), "text/markdown")})

    empty = client.post("/api/v1/knowledge-bases/", json={"name": "新建空库", "description": "还没有上传文档"}).json()

    # 离线哈希向量没法按语义给知识库排序：路由结果固定成"运维库在前"，验证建议取的是实际检索分最高的库，而不是路由的第一名
    from app.services.retrieval_service import retrieval_service

    monkeypatch.setattr(retrieval_service, "route", lambda q, top_n=1, **kw: [{"kb_id": ops["id"], "name": "运维手册库", "score": 0.5},
                                                                             {"kb_id": hr["id"], "name": "人事制度库", "score": 0.4},
                                                                             {"kb_id": empty["id"], "name": "新建空库", "score": 0.1}][:top_n])
    real_retrieve = retrieval_service.retrieve

    def retrieve(kb_id, query, **kw):           # 离线精排分不可比：运维库给一个低分，人事库给高分
        res = real_retrieve(kb_id, query, **kw)
        for c in res.contexts:
            c.score = 0.9 if kb_id == hr["id"] else 0.04
        return res

    monkeypatch.setattr(retrieval_service, "retrieve", retrieve)
    plan = _plan(client, empty["id"], "公司年假怎么休")               # 选了一个检索不到任何资料的库
    assert not plan["funnel"].get("contexts")
    sug = plan["kbSuggestion"]
    assert sug and sug["kbId"] == hr["id"] and sug["name"] == "人事制度库" and sug["fileName"] == "员工手册.md"

    assert _plan(client, hr["id"], "公司年假怎么休")["kbSuggestion"] is None       # 选对了库：不提示
    assert _plan(client, "auto", "公司年假怎么休")["kbSuggestion"] is None          # 自动路由：不提示


def _stream(client, kb_id: str, message: str) -> dict[str, object]:
    events: dict[str, object] = {}
    with client.stream("POST", "/api/v1/chat/stream", json={"kbId": kb_id, "message": message}) as r:
        ev = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:") and ev in ("meta", "plan", "sources"):
                events[ev] = json.loads(line[5:])
    return events


def test_auto_route_falls_back_when_routed_kb_cannot_answer(client, monkeypatch):
    """自动路由选中的库答不上（没资料）时，换到排名后面、分数达标的库作答；手动选库时不兜底（那里给选库提示）。"""
    from app.services.retrieval_service import retrieval_service

    hr = client.post("/api/v1/knowledge-bases/", json={"name": "人事制度库二", "description": "年假、请假"}).json()
    empty = client.post("/api/v1/knowledge-bases/", json={"name": "空库二", "description": "还没有文档"}).json()
    client.post(f"/api/v1/documents/upload?kbId={hr['id']}", files={"file": ("员工手册.md", io.BytesIO(LEAVE.encode()), "text/markdown")})
    monkeypatch.setattr(retrieval_service, "route", lambda q, top_n=1, **kw: [{"kb_id": empty["id"], "name": "空库二", "score": 0.6},
                                                                             {"kb_id": hr["id"], "name": "人事制度库二", "score": 0.5}][:top_n])
    ev = _stream(client, "auto", "公司年假怎么休")
    fb = ev["plan"]["routeFallback"]
    assert ev["meta"]["kbId"] == empty["id"]                                     # 路由先选了空库
    assert fb and fb["fromKbId"] == empty["id"] and fb["toKbId"] == hr["id"] and fb["reason"] == "没有检索到资料"
    assert ev["sources"] and all(s["fileName"] == "员工手册.md" for s in ev["sources"])
    assert any("自动路由兜底" in n for n in ev["plan"]["notes"])
    assert _stream(client, empty["id"], "公司年假怎么休")["plan"]["routeFallback"] is None   # 手动选库：不兜底


def test_weak_result_includes_rescued_low_scores():
    """兜底规则保留下来的段落（有上下文、但精排最高分低于阈值）也算答不上。"""
    from app.services.retrieval_service import retrieval_service
    from prism.schemas import RetrievalResult, RetrievalTrace, ScoredChunk

    def res(contexts: bool, rerank: list[float]):
        r = RetrievalResult(query="q", contexts=[], trace=RetrievalTrace(original_query="q", strategy="hybrid"))
        r.trace.reranked = [ScoredChunk(chunk_id=f"c{i}", score=s, rank=i + 1, route="rerank") for i, s in enumerate(rerank)]
        if contexts:
            r.contexts = [object()]          # is_weak 只看有没有上下文
        return r

    assert retrieval_service.is_weak(res(False, []), 0.1)
    assert retrieval_service.is_weak(res(True, [0.003, 0.001]), 0.1)            # 兜底保留的低分结果
    assert not retrieval_service.is_weak(res(True, [0.92]), 0.1)
    assert not retrieval_service.is_weak(res(True, [0.003]), None)               # 没设阈值：有上下文就算能答
    assert retrieval_service.best_score(res(True, [0.2, 0.7])) == 0.7
