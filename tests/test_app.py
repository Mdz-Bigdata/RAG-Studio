"""后端 API 端到端冒烟：离线模式（哈希向量 + BM25 + 词法精排 + 抽取式回答），不依赖网络与模型。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("appdata")
    os.environ["SQLITE_URL"] = f"sqlite:///{data_dir}/app.db"
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["ANTHROPIC_API_KEY"] = ""
    from fastapi.testclient import TestClient

    from app import config as app_config_mod

    app_config_mod.app_config.data_dir = data_dir
    app_config_mod.app_config.upload_dir = data_dir / "uploads"
    app_config_mod.app_config.index_root = data_dir / "indexes"
    for d in (app_config_mod.app_config.upload_dir, app_config_mod.app_config.index_root):
        d.mkdir(parents=True, exist_ok=True)
    (data_dir / "eval").mkdir(exist_ok=True)
    (data_dir / "eval" / "golden_qa.jsonl").write_text((ROOT / "data" / "eval" / "golden_qa.jsonl").read_text(encoding="utf-8"), encoding="utf-8")

    from app.main import app

    with TestClient(app) as c:
        yield c


def test_status(client):
    r = client.get("/api/v1/system/status")
    assert r.status_code == 200
    body = r.json()
    assert body["runtime"]["embedding"]["provider"] in ("hash", "sentence-transformers")
    assert "funnel" in body["pipeline"]


def test_kb_document_lifecycle(client):
    r = client.post("/api/v1/knowledge-bases/", json={"name": "员工制度", "description": "人事制度与差旅报销", "category": "制度"})
    assert r.status_code == 201
    kb = r.json()
    kb_id = kb["id"]

    for name in ("员工手册_2026版.md", "员工手册_2024版.md", "K8s运维手册.md", "售后政策.md", "产品价格表.csv"):
        path = ROOT / "data" / "raw" / name
        with path.open("rb") as f:
            r = client.post(f"/api/v1/documents/upload?kbId={kb_id}", files={"file": (name, f, "application/octet-stream")})
        assert r.status_code == 201, r.text
    docs = client.get(f"/api/v1/documents/?kbId={kb_id}").json()
    assert len(docs) == 5
    assert all(d["status"] == "COMPLETED" for d in docs), [(d["fileName"], d["status"], d["errorMessage"]) for d in docs]
    assert all(d["chunkCount"] > 0 for d in docs)

    kb = client.get(f"/api/v1/knowledge-bases/{kb_id}").json()
    assert kb["documentCount"] == 5 and kb["chunkCount"] > 0

    doc = next(d for d in docs if d["fileName"] == "售后政策.md")
    chunks = client.get(f"/api/v1/documents/{doc['id']}/chunks").json()
    assert any(c["kind"] == "parent" for c in chunks) and any(c["kind"] == "child" for c in chunks)
    md = client.get(f"/api/v1/documents/{doc['id']}/markdown").json()["markdownContent"]
    assert "退款流程" in md

    # 校对回灌：追加一段定义，验证 UnWeaver 抽取
    edited = md + "\n\n## 术语\n\n**RMA** 是指退货授权编号，申请退货时由客服生成。\n"
    r = client.put(f"/api/v1/documents/{doc['id']}/markdown", json={"markdownContent": edited})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "COMPLETED"
    ents = client.get(f"/api/v1/knowledge-bases/{kb_id}/entities").json()
    assert any(e["entity"] == "RMA" for e in ents)

    # 检索实验室：精确编号命中 + 漏斗
    r = client.post(f"/api/v1/retrieval/{kb_id}/playground", json={"query": "订单 SO-8817 的状态", "topK": 3})
    assert r.status_code == 200, r.text
    pg = r.json()
    assert pg["contexts"] and any("SO-8817" in c["text"] for c in pg["contexts"])
    assert set(pg["routes"]) >= {"dense:q0", "sparse:q0"}
    assert pg["funnel"]["contexts"] <= 3

    # 覆盖参数：MMR + 压缩 + 凸组合
    r = client.post(f"/api/v1/retrieval/{kb_id}/playground", json={"query": "出差住宿标准能不能超额", "topK": 3, "overrides": {"diversify": {"enabled": True}, "compression": {"enabled": True, "min_tokens": 10}, "fusion": {"method": "convex"}}})
    assert r.status_code == 200, r.text
    pg = r.json()
    assert pg["diversified"] and pg["compression"].get("tokens_before")
    assert any("上浮 20%" in c["text"] for c in pg["contexts"])
    assert all(c["fileName"] != "员工手册_2024版.md" for c in pg["contexts"])

    # 多跳 / agentic（无 LLM：启发式）
    for mode in ("multihop", "agentic"):
        r = client.post(f"/api/v1/retrieval/{kb_id}/playground", json={"query": "远程员工的通讯补贴和本地通勤员工的交通补贴有什么不同", "mode": mode, "topK": 4})
        assert r.status_code == 200, r.text
        assert r.json()["strategy"].startswith(mode) or r.json()["strategy"] == "multihop"

    # 路由
    r = client.post("/api/v1/retrieval/route", json={"query": "年假可以跨年吗"})
    assert r.status_code == 200 and r.json()[0]["kb_id"] == kb_id

    # 对话 SSE（离线抽取式）
    with client.stream("POST", "/api/v1/chat/stream", json={"kbId": kb_id, "message": "年假可以跨年使用吗", "topK": 3}) as resp:
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())
    events = [ln for ln in raw.split("\n\n") if ln.strip()]
    names = [e.split("\n")[0].replace("event: ", "") for e in events]
    assert names[:4] == ["meta", "plan", "sources", "relations"] and names[-1] == "done"
    text = "".join(json.loads(e.split("\ndata: ")[1])["content"] for e in events if e.startswith("event: message"))
    assert "次年 3 月 31 日" in text
    meta = json.loads(events[0].split("\ndata: ")[1])
    sessions = client.get(f"/api/v1/chat/sessions?kbId={kb_id}").json()
    assert sessions and sessions[0]["id"] == meta["sessionId"] and sessions[0]["messageCount"] == 2
    msgs = client.get(f"/api/v1/chat/sessions/{meta['sessionId']}/messages").json()
    assert msgs[1]["role"] == "assistant" and msgs[1]["sources"]

    # 评测
    r = client.post(f"/api/v1/eval/{kb_id}/run", json={"useSample": True, "ks": [1, 3, 5]})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["n"] >= 10 and rep["stageMetrics"]["final"]["hit@5"] >= 0.8
    assert client.get(f"/api/v1/eval/{kb_id}/reports").json()[0]["id"] == rep["id"]

    # 语料准备：意图 × 知识 覆盖分析
    r = client.post(f"/api/v1/eval/{kb_id}/coverage", json={"intents": ["订单 SO-8817 的状态", {"question": "公司食堂几点开饭", "category": "无答案"}]})
    assert r.status_code == 200, r.text
    cov = r.json()
    assert cov["buckets"] == {"high": 1, "partial": 0, "none": 1}
    assert all("file_name" in d for d in cov["docs"]) and cov["markdown"].startswith("# 意图")
    assert client.get(f"/api/v1/eval/{kb_id}/coverage").json()["n"] == 2
    assert client.post(f"/api/v1/eval/{kb_id}/coverage", json={"useSample": True}).json()["n"] >= 10

    # 临时解析
    r = client.post("/api/v1/documents/parse-temp", files={"file": ("note.md", "# 临时\n\n内容 A".encode("utf-8"), "text/markdown")})
    assert r.status_code == 200 and "内容 A" in r.json()["content"]

    # 解析能力矩阵 + MinerU-Popo 状态 + 重新解析
    parsing = client.get("/api/v1/system/parsing").json()
    assert {m["ext"]: m["parser"] for m in parsing["matrix"]}["md"] == "markdown"
    assert [t["tier"] for t in parsing["mineru"]["tiers"]] == ["flash", "basic", "standard", "advanced"]
    # 档位下拉：四档全部返回且都可选，带状态 / 推荐 / 下载命令
    tiers = client.get("/api/v1/system/mineru/tiers").json()
    assert [t["tier"] for t in tiers["tiers"]] == ["flash", "basic", "standard", "advanced"] and all(t["selectable"] for t in tiers["tiers"])
    assert sum(t["current"] for t in tiers["tiers"]) == 1 and sum(t["recommended"] for t in tiers["tiers"]) == 1
    assert tiers["tiers"][0]["download"] is None and tiers["tiers"][2]["download"] == tiers["tiers"][3]["download"] == "mineru-kit models download --tier standard"
    assert all(t["status"] in ("ready", "needs_models", "missing_deps", "not_installed") for t in tiers["tiers"])
    for tier in ("basic", "advanced"):
        assert client.put("/api/v1/settings/", json={"overrides": {"parsing": {"mineru": {"tier": tier}}}}).json()["effective"]["parsing"]["mineru"]["tier"] == tier
    client.post("/api/v1/settings/reset")
    popo = client.get(f"/api/v1/documents/{doc['id']}/popo").json()
    assert popo["applicable"] is False and popo["treeReady"] is False
    assert client.post(f"/api/v1/documents/{doc['id']}/popo-export").status_code == 400
    r = client.post(f"/api/v1/documents/{doc['id']}/reprocess")
    assert r.status_code == 200, r.text
    assert client.get(f"/api/v1/documents/{doc['id']}").json()["status"] == "COMPLETED"

    # 删除文档与知识库
    r = client.delete(f"/api/v1/documents/{doc['id']}")
    assert r.status_code == 200
    assert client.get(f"/api/v1/knowledge-bases/{kb_id}").json()["documentCount"] == 4
    assert client.delete(f"/api/v1/knowledge-bases/{kb_id}").status_code == 200


def test_settings_and_models(client):
    r = client.get("/api/v1/settings/")
    assert r.status_code == 200 and r.json()["effective"]["retrieval"]["fusion"]["rrf_k"] == 60
    r = client.put("/api/v1/settings/", json={"overrides": {"retrieval": {"fusion": {"rrf_k": 40}, "diversify": {"enabled": True}}}})
    assert r.status_code == 200 and r.json()["effective"]["retrieval"]["fusion"]["rrf_k"] == 40
    assert client.put("/api/v1/settings/", json={"overrides": {"retrieval": {"fusion": {"method": "bogus"}}}}).status_code == 400
    assert client.post("/api/v1/settings/reset").json()["effective"]["retrieval"]["fusion"]["rrf_k"] == 60

    # 场景预设：每个预设都是合法的 retrieval 覆盖
    from prism.config import Settings

    body = client.get("/api/v1/settings/presets").json()
    assert [p["key"] for p in body["presets"]] == ["enterprise", "regulated", "faq", "longdoc"] and len(body["guide"]) == 4
    for p in body["presets"]:
        eff = Settings().with_overrides({"retrieval": p["retrieval"]}).retrieval
        assert eff.rerank.enabled and eff.rerank.top_k == p["retrieval"]["rerank"]["top_k"]
    assert client.put("/api/v1/settings/", json={"overrides": {"retrieval": body["presets"][1]["retrieval"]}}).json()["effective"]["retrieval"]["compression"]["enabled"] is True
    client.post("/api/v1/settings/reset")

    assert any(p["type"] == "anthropic" for p in client.get("/api/v1/models/providers").json())
    r = client.post("/api/v1/models/configs", json={"providerName": "SiliconFlow", "providerType": "openai_compatible", "baseUrl": "https://api.siliconflow.cn/v1", "apiKey": "sk-test-1234567890", "category": "text", "modelName": "Qwen/Qwen3-32B"})
    assert r.status_code == 200
    cfg = r.json()
    assert cfg["apiKeyMasked"].startswith("sk-t") and "****" in cfg["apiKeyMasked"]
    assert client.post(f"/api/v1/models/configs/{cfg['id']}/activate").status_code == 200
    assert client.get("/api/v1/models/configs").json()[0]["isActive"] == 1
    assert client.post(f"/api/v1/models/configs/{cfg['id']}/deactivate").status_code == 200
    assert client.delete(f"/api/v1/models/configs/{cfg['id']}").status_code == 200
