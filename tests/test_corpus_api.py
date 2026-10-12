"""语料治理 API 端到端：离线模式（哈希向量 + BM25 + 抽取式），不依赖网络与模型。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("corpusdata")
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

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def no_reindex(monkeypatch):
    """别名词典保存会排队后台重建；测试里只记录，不真的在后台线程里删索引。"""
    from app.services.reindex_service import reindex_service

    calls: list[tuple] = []
    monkeypatch.setattr(reindex_service, "enqueue", lambda *a: calls.append(a))
    return calls


_CREATED: list[str] = []


@pytest.fixture(autouse=True)
def _cleanup_kbs(client):
    """测试模块共用一个数据库：结束后删掉本测试建的知识库，避免影响其他模块的跨库路由断言。"""
    yield
    while _CREATED:
        client.delete(f"/api/v1/knowledge-bases/{_CREATED.pop()}")


def _kb_with_docs(client, names: list[str]) -> tuple[str, list[dict]]:
    kb_id = client.post("/api/v1/knowledge-bases/", json={"name": "治理测试", "description": "", "category": "制度"}).json()["id"]
    _CREATED.append(kb_id)
    for name in names:
        with (ROOT / "data" / "raw" / name).open("rb") as f:
            assert client.post(f"/api/v1/documents/upload?kbId={kb_id}", files={"file": (name, f, "application/octet-stream")}).status_code == 201
    docs = client.get(f"/api/v1/documents/?kbId={kb_id}").json()
    assert all(d["status"] == "COMPLETED" for d in docs), [(d["fileName"], d["errorMessage"]) for d in docs]
    return kb_id, docs


def test_governance_flow(client, no_reindex):
    kb_id, docs = _kb_with_docs(client, ["员工手册_2026版.md", "员工手册_2024版.md", "K8s运维手册.md"])

    # 上传即自动抽取元数据：front matter 已填的字段不覆盖，只记录抽取结果
    hb = next(d for d in docs if d["fileName"] == "员工手册_2024版.md")
    assert hb["autoMeta"]["valid_from"]["value"] == "2024-01-01" and hb["validTo"] == "2025-12-31"

    # 质量体检
    r = client.get(f"/api/v1/corpus/{kb_id}/health").json()
    assert r["totals"]["docs"] == 3 and 0 <= r["score"] <= 100
    assert any(i["type"] == "expired" for d in r["docs"] if d["docId"] == hb["id"] for i in d["issues"])

    # 剔除一个段落 → 切片数下降；恢复 → 后台重新入库后回来
    k8s = next(d for d in docs if d["fileName"] == "K8s运维手册.md")
    parents = [c for c in client.get(f"/api/v1/documents/{k8s['id']}/chunks").json() if c["kind"] == "parent"]
    before = k8s["chunkCount"]
    out = client.post(f"/api/v1/corpus/{kb_id}/health/exclude", json={"chunkIds": [parents[0]["chunkId"]], "reason": "测试"}).json()
    assert out["parents"] == 1 and out["chunks"] >= 2
    after = client.get(f"/api/v1/documents/{k8s['id']}").json()["chunkCount"]
    assert after < before
    health = client.get(f"/api/v1/corpus/{kb_id}/health").json()
    assert len(health["excluded"]) == 1
    # 重新解析时剔除清单同样生效
    client.post(f"/api/v1/documents/{k8s['id']}/reprocess")
    assert client.get(f"/api/v1/documents/{k8s['id']}").json()["chunkCount"] == after
    client.post(f"/api/v1/corpus/{kb_id}/health/restore", json={"fingerprints": [health["excluded"][0]["fingerprint"]]})
    assert client.get(f"/api/v1/documents/{k8s['id']}").json()["chunkCount"] == before

    # 让旧版本失效：只更新元数据，不重新向量化
    new = next(d for d in docs if d["fileName"] == "员工手册_2026版.md")
    client.post(f"/api/v1/corpus/{kb_id}/health/retire", json={"docIds": [new["id"]], "validTo": "2030-12-31"})
    assert client.get(f"/api/v1/documents/{new['id']}").json()["validTo"] == "2030-12-31"

    # 别名词典：查询扩展立即生效，并排队重建索引做文档扩展
    r = client.put(f"/api/v1/corpus/{kb_id}/aliases", json={"groups": [["年假", "年休假", "带薪假"], ["单个"]]}).json()
    assert r["groups"] == [["年假", "年休假", "带薪假"]] and r["reindexing"] and no_reindex
    assert client.get(f"/api/v1/corpus/{kb_id}/aliases").json()["groups"] == r["groups"]
    assert isinstance(client.post(f"/api/v1/corpus/{kb_id}/aliases/discover").json(), list)
    pg = client.post(f"/api/v1/retrieval/{kb_id}/playground", json={"query": "年休假可以跨年吗", "topK": 3}).json()
    assert pg["contexts"]

    # 元数据：抽取建议 → 应用
    rows = client.post(f"/api/v1/corpus/{kb_id}/metadata/extract", json={}).json()
    row = next(x for x in rows if x["docId"] == k8s["id"])
    assert "tags" in row["suggestions"]
    assert client.post(f"/api/v1/corpus/{kb_id}/metadata/apply", json={"items": [{"docId": k8s["id"], "fields": {"department": "运维部", "valid_from": "2026-01-01"}}]}).json()["updated"] == 1
    d = client.get(f"/api/v1/documents/{k8s['id']}").json()
    assert d["department"] == "运维部" and d["validFrom"] == "2026-01-01" and d["autoMeta"]["department"]["applied"]


def test_augment_flow(client, no_reindex):
    kb_id, _ = _kb_with_docs(client, ["员工手册_2026版.md", "售后政策.md"])
    r = client.post(f"/api/v1/corpus/{kb_id}/augment/generate", json={"intents": ["年假能不能留到明年用", "年假能不能留到明年用"]}).json()
    assert len(r["items"]) == 1
    item = r["items"][0]
    assert item["method"] in ("extractive", "llm") and item["sources"]
    assert client.post(f"/api/v1/corpus/{kb_id}/augment/items", json={"question": "年终奖怎么发", "answer": ""}).json()["items"][-1]["status"] == "gap"

    # 空答案不能通过；改写后通过并发布
    gap_id = client.get(f"/api/v1/corpus/{kb_id}/augment").json()["items"][-1]["id"]
    assert client.put(f"/api/v1/corpus/{kb_id}/augment/items/{gap_id}", json={"status": "approved"}).status_code == 400
    client.put(f"/api/v1/corpus/{kb_id}/augment/items/{item['id']}", json={"answer": "可以。当年未休完的年假可跨年使用，最晚在次年 3 月 31 日前休完。", "status": "approved"})
    r = client.post(f"/api/v1/corpus/{kb_id}/augment/publish").json()
    assert r["docId"] and r["items"][0]["published"]
    faq = client.get(f"/api/v1/documents/{r['docId']}").json()
    assert faq["status"] == "COMPLETED" and faq["chunkCount"] > 0 and faq["fileName"] == "增强语料-FAQ.md"
    pg = client.post(f"/api/v1/retrieval/{kb_id}/playground", json={"query": "年假能不能留到明年用", "topK": 5}).json()
    assert any("增强语料" in (c.get("fileName") or "") or "FAQ" in " ".join(c.get("breadcrumbs") or []) for c in pg["contexts"])

    # 全部撤回 → 发布即下线增强文档
    client.put(f"/api/v1/corpus/{kb_id}/augment/items/{item['id']}", json={"status": "rejected"})
    r = client.post(f"/api/v1/corpus/{kb_id}/augment/publish").json()
    assert r["docId"] is None
    assert client.get(f"/api/v1/documents/{faq['id']}").status_code == 404


def test_coverage_records_models_and_comparability(client, no_reindex):
    kb_id, _ = _kb_with_docs(client, ["员工手册_2026版.md"])
    body = {"intents": ["年假可以跨年使用吗", "公司食堂几点开饭"]}
    first = client.post(f"/api/v1/eval/{kb_id}/coverage", json=body).json()
    assert first["models"]["embedderModel"] and first["models"]["rerankerEffective"] and first["models"]["scoring"] == first["scoring"]
    assert first["comparison"] is None and "- 模型：嵌入" in first["markdown"]

    second = client.post(f"/api/v1/eval/{kb_id}/coverage", json=body).json()
    assert second["comparison"]["comparable"] and second["comparison"]["sameIntents"]
    assert second["comparison"]["previous"]["buckets"] == first["buckets"]

    third = client.post(f"/api/v1/eval/{kb_id}/coverage", json={**body, "high": 0.7}).json()
    assert not third["comparison"]["comparable"] and any("阈值" in r for r in third["comparison"]["reasons"])
    assert "不可直接比较" in third["markdown"]

    # 早于模型记录功能的旧报告：没有 models 字段 → 不可确认可比
    from app.db import Repository, SessionLocal

    with SessionLocal() as db:
        old = {k: v for k, v in third.items() if k not in ("models", "comparison")}
        Repository(db).set_setting(f"coverage:{kb_id}", old)
    fourth = client.post(f"/api/v1/eval/{kb_id}/coverage", json={**body, "high": 0.7}).json()
    assert not fourth["comparison"]["comparable"] and "没有记录所用模型" in fourth["comparison"]["reasons"][0]


def test_startup_check_recovers_interrupted_and_incomplete_indexes(client, no_reindex):
    """自动重建被打断（热重载 / 进程重启）不能让知识库停留在空索引；残缺索引在启动自检时被发现并重建。"""
    import time

    from app.db import Repository, SessionLocal
    from app.services.index_service import index_service
    from app.services.reindex_service import PENDING_KEY, reindex_service

    kb_id, docs = _kb_with_docs(client, ["员工手册_2026版.md", "售后政策.md"])
    assert index_service.indexed_docs(kb_id) == 2

    def check() -> list[str]:
        no_reindex.clear()
        reindex_service.request("测试")
        for _ in range(100):
            if not reindex_service.status()["checking"]:
                break
            time.sleep(0.05)
        return [reason for k, _name, reason in no_reindex if k == kb_id]

    # 1) 重建进行到一半进程被杀：正式索引一直完整；重启后清掉临时目录，重新排队
    index_service.begin_rebuild(kb_id)
    index_service.save(kb_id)                              # 重建线程写出了临时目录
    assert index_service._staging_dir(kb_id).exists() and index_service.indexed_docs(kb_id) == 2
    index_service._staging.pop(kb_id)                      # 模拟进程退出：内存里的重建状态丢失
    index_service._tls.rebuilding = None
    reasons = check()
    assert index_service.indexed_docs(kb_id) == 2 and not index_service._staging_dir(kb_id).exists()
    assert reasons and reasons[0].startswith("上次重建被中断")

    # 2) 进程在重建途中退出，留下"待重建"记录 → 启动后继续
    with SessionLocal() as db:
        Repository(db).set_setting(PENDING_KEY, {kb_id: "入库参数变更"})
    assert check()[0].startswith("上次重建被中断")

    # 3) 索引里的文档比数据库里"已完成"的少（不论什么原因）→ 判为不完整
    with SessionLocal() as db:
        Repository(db).set_setting(PENDING_KEY, {})
    idx = index_service.get(kb_id)
    idx.remove_document(docs[0]["id"])
    index_service.save(kb_id)
    assert check() == ["索引不完整（1/2 份文档）"]


def test_pending_rebuild_is_cleared_only_after_success(client, monkeypatch):
    import time

    import importlib

    from app.db import Repository, SessionLocal

    rs_mod = importlib.import_module("app.services.reindex_service")   # app.services 包里同名属性是实例，要拿模块
    PENDING_KEY, reindex_service = rs_mod.PENDING_KEY, rs_mod.reindex_service

    def pending() -> dict:
        with SessionLocal() as db:
            return Repository(db).get_setting(PENDING_KEY) or {}
    seen: list[dict] = []
    outcome = {"fail": True}

    def fake_reindex(kb_id):
        seen.append(dict(pending()))                     # 重建进行中：记录必须还在（进程此刻退出就能续上）
        if outcome["fail"]:
            raise RuntimeError("模拟失败")

    monkeypatch.setattr(rs_mod.document_service, "reindex_kb", fake_reindex)

    def run_once():
        reindex_service.enqueue("kb-pending", "测试库", "入库参数变更")
        for _ in range(100):
            if not reindex_service.is_active("kb-pending"):
                break
            time.sleep(0.05)

    run_once()
    assert "kb-pending" in seen[0] and "kb-pending" in pending()     # 失败：保留，下次启动再试
    outcome["fail"] = False
    run_once()
    assert "kb-pending" not in pending()                             # 成功：移除


def test_rebuild_keeps_live_index_and_absorbs_concurrent_changes(client, monkeypatch):
    """整库重建期间：检索一直读完整的旧索引；别的线程删除 / 上传 / 改元数据，替换时都被补进新索引。"""
    import importlib
    import threading

    ds_mod = importlib.import_module("app.services.document_service")
    from app.services.corpus_service import corpus_service
    from app.services.index_service import index_service

    # 三份文档：第一次入库时删掉一份，剩下那份的入库发生在重建主循环里（不持锁）——元数据修改要在那时发生，
    # 而不是在收尾补齐阶段（持锁）：那样修改线程会卡在锁上，测试只能靠超时和竞态碰运气
    kb_id, docs = _kb_with_docs(client, ["员工手册_2026版.md", "售后政策.md", "员工手册_2024版.md"])
    ids = [d["id"] for d in docs]
    service = ds_mod.document_service
    original = service.process
    calls: list[str] = []
    live_during: list[set[str]] = []
    new_doc: dict = {}

    def other_thread(fn):
        errors: list[BaseException] = []

        def run():
            try:
                fn()
            except BaseException as exc:   # noqa: BLE001 — 抛回测试线程
                errors.append(exc)

        t = threading.Thread(target=run)
        t.start()
        t.join(30)
        assert not t.is_alive(), "并发写入线程卡住了（多半在等重建线程持有的知识库锁）"
        if errors:
            raise errors[0]

    def hooked(doc_id):
        calls.append(doc_id)
        live_during.append(set(index_service.live(kb_id).store.docs))
        if len(calls) == 1:
            victim = next(i for i in ids if i != doc_id)

            def concurrent():                     # 另一个线程：删一份、上传一份（都写在旧索引上）
                service.delete(victim)
                with (ROOT / "data" / "raw" / "K8s运维手册.md").open("rb") as f:
                    d = service.save_upload(kb_id, "K8s运维手册.md", f)
                original(d.id)
                new_doc["id"], new_doc["victim"] = d.id, victim

            other_thread(concurrent)
        elif len(calls) == 2:                     # 第一份已进新索引之后，别处改了它的元数据（主循环里，重建线程未持锁）
            assert doc_id != new_doc["id"], "第二次入库应发生在重建主循环里，而不是收尾补齐阶段"
            other_thread(lambda: corpus_service.apply_metadata(kb_id, [{"docId": calls[0], "fields": {"department": "测试部"}}]))
        return original(doc_id)

    monkeypatch.setattr(service, "process", hooked)
    assert service.reindex_kb(kb_id) == []

    assert live_during[0] == set(ids)                                   # 重建开始后，正式索引仍是完整的旧索引
    final = index_service.live(kb_id)
    assert set(final.store.docs) == set(ids) - {new_doc["victim"]} | {new_doc["id"]}   # 删掉的不回来，新上传的补进来
    assert final.store.docs[calls[0]].department == "测试部"            # 重建期间的元数据修改没丢
    leftovers = [p.name for p in index_service.directory(kb_id).parent.iterdir() if p.name.startswith(f".{kb_id}")]
    assert leftovers == [] and index_service.target_dir(kb_id) == index_service.directory(kb_id)


def test_manual_metadata_survives_reparse(client, no_reindex):
    """语料治理里人工改的字段，重新解析（含整库重建）时不能被文件自带的 front matter（department: 人力资源部）改回去。"""
    from app.services.corpus_service import corpus_service
    from app.services.document_service import document_service
    from app.services.index_service import index_service

    kb_id, docs = _kb_with_docs(client, ["员工手册_2026版.md"])
    doc_id = docs[0]["id"]
    assert index_service.get(kb_id).store.docs[doc_id].department == "人力资源部"     # 初次入库：来自 front matter

    assert corpus_service.apply_metadata(kb_id, [{"docId": doc_id, "fields": {"department": "测试部"}}]) == 1
    document_service.process(doc_id)                                                   # 按原文件重新解析

    doc = client.get(f"/api/v1/documents/?kbId={kb_id}").json()[0]
    assert doc["status"] == "COMPLETED" and doc["department"] == "测试部"
    assert index_service.get(kb_id).store.docs[doc_id].department == "测试部"
    assert index_service.get(kb_id).store.docs[doc_id].version == "2026.1"            # 没人工改过的字段照常取 front matter

    document_service.update_markdown(doc_id, "---\ndepartment: 人力资源部\n---\n\n# 员工手册\n\n校对后的正文")
    assert index_service.get(kb_id).store.docs[doc_id].department == "测试部"          # 校对回灌同样保留人工值


def test_excluding_a_duplicate_keeps_the_other_copy(client, no_reindex):
    """体检里剔除重复副本：只剔除被选中的那一份；保留的那份在重新解析后依然在（剔除按「文档 + 指纹」记录）。"""
    kb_id = client.post("/api/v1/knowledge-bases/", json={"name": "重复测试", "description": "", "category": ""}).json()["id"]
    _CREATED.append(kb_id)
    for name in ("售后政策-A.md", "售后政策-B.md"):
        with (ROOT / "data" / "raw" / "售后政策.md").open("rb") as f:
            client.post(f"/api/v1/documents/upload?kbId={kb_id}", files={"file": (name, f, "text/markdown")})
    dup = next(c for c in client.get(f"/api/v1/corpus/{kb_id}/health").json()["chunks"] if "duplicate" in c["issues"])
    victim, keeper = dup["docId"], dup["duplicateOf"]["docId"]
    count = lambda d: client.get(f"/api/v1/documents/{d}").json()["chunkCount"]   # noqa: E731
    keeper_before, victim_before = count(keeper), count(victim)

    client.post(f"/api/v1/corpus/{kb_id}/health/exclude", json={"chunkIds": [dup["chunkId"]]})
    assert count(victim) < victim_before
    client.post(f"/api/v1/documents/{keeper}/reprocess")
    assert count(keeper) == keeper_before                     # 保留的副本没有被一起剔掉
    client.post(f"/api/v1/documents/{victim}/reprocess")
    assert count(victim) < victim_before                      # 被剔除的那份重新解析后仍然不入库


def test_applied_metadata_survives_reparse(client, no_reindex):
    """在语料治理里应用的元数据（即便字段原本已被自动抽取过）是人工确认值，重新解析不能被文件里的 front matter 改回去。"""
    kb_id, docs = _kb_with_docs(client, ["员工手册_2026版.md"])
    doc = docs[0]
    assert doc["version"] and doc["title"]                    # front matter 已经给了版本与标题
    client.post(f"/api/v1/corpus/{kb_id}/metadata/apply", json={"items": [{"docId": doc["id"], "fields": {"title": "人工标题", "version": "9.9"}}]})
    d = client.get(f"/api/v1/documents/{doc['id']}").json()
    assert d["autoMeta"]["version"]["source"] == "manual" and d["autoMeta"]["title"]["source"] == "manual"
    client.post(f"/api/v1/documents/{doc['id']}/reprocess")
    d = client.get(f"/api/v1/documents/{doc['id']}").json()
    assert (d["title"], d["version"]) == ("人工标题", "9.9")


def test_eval_reports_record_models_and_comparability(client, no_reindex):
    kb_id, _ = _kb_with_docs(client, ["员工手册_2026版.md", "售后政策.md"])
    golden = [{"question": "年假可以跨年使用吗", "answer_keywords": ["跨年"]}, {"question": "怎么退货", "relevant_sources": ["售后政策.md"]}]
    run = lambda **kw: client.post(f"/api/v1/eval/{kb_id}/run", json={"examples": golden, **kw}).json()   # noqa: E731

    first = run()
    assert first["models"]["embedderModel"] and first["models"]["questionSet"] and first["comparison"] is None
    assert "- 模型：嵌入" in first["markdown"]
    second = run()
    assert "comparison" in second, second
    assert second["comparison"]["comparable"] and second["comparison"]["previous"]["hit@1"] == first["stageMetrics"]["final"]["hit@1"]

    third = run(overrides={"rerank": {"top_k": 3}})
    assert not third["comparison"]["comparable"] and "按请求的检索参数覆盖不同" in third["comparison"]["reasons"]
    golden.append({"question": "年假需要提前几天申请", "answer_keywords": ["提前"]})
    fourth = run(overrides={"rerank": {"top_k": 3}})
    assert not fourth["comparison"]["comparable"] and fourth["comparison"]["reasons"] == ["评估集（问题或标注）不同"]
    assert fourth["comparison"]["notes"] == []

    # 全局检索策略变了：仍可比较（这正是策略 A/B），但要提示变化可能来自策略
    client.put("/api/v1/settings/", json={"overrides": {"retrieval": {"fusion": {"rrf_k": 61}}}})
    try:
        fifth = run(overrides={"rerank": {"top_k": 3}})
    finally:
        client.post("/api/v1/settings/reset")
    assert fifth["comparison"]["comparable"] and "检索策略配置与上次不同" in fifth["comparison"]["notes"][0]
