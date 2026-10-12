"""整库重建（换向量模型 / 改入库参数时自动触发）不能丢掉人工校对的结果。"""

from __future__ import annotations

import json
import os
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_reindex_prefers_proofread_markdown_over_original_file(monkeypatch, tmp_path: Path):
    os.environ.setdefault("SQLITE_URL", f"sqlite:///{tmp_path}/app.db")   # 单独运行时也不碰真实数据库
    import importlib

    mod = importlib.import_module("app.services.document_service")   # app.services 包里同名属性是实例，要拿模块

    original = tmp_path / "售后政策.md"
    original.write_text("# 售后服务政策\n\n原文", encoding="utf-8")
    docs = [
        SimpleNamespace(id="edited", file_name="售后政策.md", status="COMPLETED", file_path=str(original), markdown_content="# 售后服务政策\n\n校对后的版本", parser="markdown+edited"),
        SimpleNamespace(id="plain", file_name="售后政策.md", status="COMPLETED", file_path=str(original), markdown_content="# 售后服务政策\n\n原文", parser="mineru"),
        SimpleNamespace(id="orphan", file_name="已删除.pdf", status="COMPLETED", file_path=str(tmp_path / "已删除.pdf"), markdown_content="# 只剩正文", parser="pdfplumber"),
    ]

    class FakeRepo:
        def __init__(self, _db):
            pass

        def list_documents(self, _kb_id):
            return docs

        def get_document(self, doc_id):
            return next((d for d in docs if d.id == doc_id), None)

    calls: list[tuple] = []
    monkeypatch.setattr(mod, "SessionLocal", lambda: nullcontext(None))
    monkeypatch.setattr(mod, "Repository", FakeRepo)
    monkeypatch.setattr(mod.index_service, "begin_rebuild", lambda kb_id: calls.append(("begin", kb_id)))
    monkeypatch.setattr(mod.index_service, "finish_rebuild", lambda kb_id: calls.append(("finish", kb_id)))
    monkeypatch.setattr(mod.document_service, "_reconcile_rebuild", lambda *a: None)
    monkeypatch.setattr(mod.document_service, "process", lambda doc_id: calls.append(("parse-file", doc_id)))
    monkeypatch.setattr(mod.document_service, "update_markdown", lambda doc_id, md: calls.append(("markdown", doc_id, md)))

    assert mod.document_service.reindex_kb("kb-1") == []

    assert calls == [
        ("begin", "kb-1"),                                        # 旧索引挪到备份，而不是直接删除
        ("markdown", "edited", "# 售后服务政策\n\n校对后的版本"),   # 校对过：用保存的 Markdown，不回到原文件
        ("parse-file", "plain"),                                  # 没校对过：按当前解析配置重新解析原文件
        ("markdown", "orphan", "# 只剩正文"),                      # 原文件不在了：用 Markdown 兜底
        ("finish", "kb-1"),                                       # 全部完成才删除备份
    ]


def _index_service(monkeypatch, tmp_path: Path):
    os.environ.setdefault("SQLITE_URL", f"sqlite:///{tmp_path}/app.db")
    from app import config as app_config_mod
    from app.db import init_db
    from app.services.index_service import index_service

    init_db()                                 # begin_rebuild 要读策略配置（app_settings 表）
    monkeypatch.setattr(app_config_mod.app_config, "index_root", tmp_path / "indexes")
    (tmp_path / "indexes").mkdir()
    return index_service


def _write_index(directory: Path, docs: int, marker: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "chunks.jsonl").write_text(marker, encoding="utf-8")
    (directory / "index_meta.json").write_text(json.dumps({"docs": docs, "chunks": docs}), encoding="utf-8")


def _fresh_service():
    """模拟进程重启：新的 IndexService 实例（内存里的缓存 / 进行中的重建全没了，只剩磁盘）。"""
    from app.services.index_service import IndexService

    return IndexService()


def test_rebuild_builds_in_staging_and_swaps_atomically(monkeypatch, tmp_path: Path):
    """重建期间正式索引不动（检索看到的是完整旧索引）；重建线程写临时目录；完成后整体替换，不留临时文件。"""
    _index_service(monkeypatch, tmp_path)
    svc = _fresh_service()
    live = svc.directory("kb")
    _write_index(live, 7, "完整的旧索引")

    svc.begin_rebuild("kb")
    staging = svc.target_dir("kb")
    assert staging != live and (live / "chunks.jsonl").read_text(encoding="utf-8") == "完整的旧索引"
    _write_index(staging, 2, "新索引（进行中）")
    assert svc.indexed_docs("kb") == 7                        # 其他线程此刻读到的仍是旧索引

    _write_index(staging, 7, "新索引")
    svc.finish_rebuild("kb")
    assert (live / "chunks.jsonl").read_text(encoding="utf-8") == "新索引" and svc.target_dir("kb") == live
    assert sorted(p.name for p in live.parent.iterdir()) == ["kb"]


def test_interrupted_rebuild_keeps_previous_index(monkeypatch, tmp_path: Path):
    """重建途中进程被杀：正式索引从未被动过；重启后删掉临时目录，返回需要重新排队的知识库。"""
    _index_service(monkeypatch, tmp_path)
    before = _fresh_service()
    live = before.directory("kb")
    _write_index(live, 7, "完整的旧索引")
    before.begin_rebuild("kb")
    _write_index(before.target_dir("kb"), 1, "只有一份文档的半成品")   # ……进程在这里被杀掉

    after = _fresh_service()
    assert after.recover_interrupted(skip={"other"}) == ["kb"]
    assert (live / "chunks.jsonl").read_text(encoding="utf-8") == "完整的旧索引" and after.indexed_docs("kb") == 7
    assert sorted(p.name for p in live.parent.iterdir()) == ["kb"] and after.recover_interrupted() == []
    before._tls.rebuilding = None


def test_interrupted_swap_is_rolled_back_or_completed(monkeypatch, tmp_path: Path):
    """两次目录改名之间被杀：正式目录已挪走、新目录还没就位 → 挪回旧的；新目录已就位 → 只清理旧的。"""
    _index_service(monkeypatch, tmp_path)
    svc = _fresh_service()
    live, old, staging = svc.directory("kb"), svc._old_dir("kb"), svc._staging_dir("kb")
    _write_index(old, 7, "旧索引")
    _write_index(staging, 7, "新索引")
    assert _fresh_service().recover_interrupted() == ["kb"]
    assert (live / "chunks.jsonl").read_text(encoding="utf-8") == "旧索引" and not old.exists() and not staging.exists()

    _write_index(old, 7, "旧索引")                            # 新目录已就位，只差删旧目录
    _write_index(live, 7, "新索引")
    _fresh_service().recover_interrupted()
    assert (live / "chunks.jsonl").read_text(encoding="utf-8") == "新索引" and not old.exists()


def test_legacy_backup_from_in_place_rebuild_is_restored(monkeypatch, tmp_path: Path):
    """原地重建（Qdrant 后端，或旧版本留下的）的备份：用它覆盖残缺的正式目录。"""
    _index_service(monkeypatch, tmp_path)
    svc = _fresh_service()
    _write_index(svc._backup_dir("kb"), 7, "完整的旧索引")
    _write_index(svc.directory("kb"), 1, "残缺")
    assert svc.recover_interrupted() == ["kb"]
    assert (svc.directory("kb") / "chunks.jsonl").read_text(encoding="utf-8") == "完整的旧索引"


def test_recovery_skips_rebuild_in_progress(monkeypatch, tmp_path: Path):
    _index_service(monkeypatch, tmp_path)
    svc = _fresh_service()
    _write_index(svc.directory("kb"), 3, "旧")
    svc.begin_rebuild("kb")
    _write_index(svc.target_dir("kb"), 1, "进行中")
    assert svc.recover_interrupted(skip={"kb"}) == []         # 本进程正在重建：临时目录是有效的进行中状态
    assert svc._staging_dir("kb").exists() and svc.indexed_docs("kb") == 3
    svc.abort_rebuild("kb")
    assert not svc._staging_dir("kb").exists() and svc.indexed_docs("kb") == 3


# --- 服务层：重建排队状态持久化，进程重启后继续 ------------------------------------------------
class _FakeDB:
    """进程外的持久状态（SQLite 的替身）：两个 ReindexService 实例共享它，模拟"重启前 / 重启后"。"""

    def __init__(self, fingerprint: str | None, kbs: dict[str, list[str]]):
        self.settings: dict = {} if fingerprint is None else {"ingest_fingerprint": fingerprint}
        self.kbs = kbs                                           # kb_id -> 各文档状态

    def repo(self, _db):
        store = self

        class Repo:
            def get_setting(self, key, default=None):
                return json.loads(json.dumps(store.settings[key])) if key in store.settings else default

            def set_setting(self, key, value):
                store.settings[key] = json.loads(json.dumps(value))

            def list_kbs(self):
                return [SimpleNamespace(id=k, name=f"知识库 {k}") for k in store.kbs]

            def list_documents(self, kb_id):
                return [SimpleNamespace(status=s) for s in store.kbs[kb_id]]

        return Repo()


def _reindex_module(monkeypatch, tmp_path: Path, db: _FakeDB, indexed: dict[str, int]):
    os.environ.setdefault("SQLITE_URL", f"sqlite:///{tmp_path}/app.db")
    import importlib

    mod = importlib.import_module("app.services.reindex_service")
    monkeypatch.setattr(mod, "SessionLocal", lambda: nullcontext(None))
    monkeypatch.setattr(mod, "Repository", db.repo)
    monkeypatch.setattr(mod, "ingest_fingerprint", lambda: "new")
    monkeypatch.setattr(mod.index_service, "needs_reindex", lambda kb_id: False)
    monkeypatch.setattr(mod.index_service, "indexed_docs", lambda kb_id: indexed.get(kb_id, 0))
    monkeypatch.setattr(mod.index_service, "recover_interrupted", lambda skip=None: [])
    monkeypatch.setattr(mod.index_service, "_builds", {})       # 别的测试真实重建留下的记录不能串进来
    return mod


def _drain(svc) -> None:
    worker = svc._worker                  # 工作线程退出时会把 svc._worker 清空，先拿住引用
    if worker is not None:
        worker.join(timeout=5)
        assert not worker.is_alive()


def test_rebuild_interrupted_by_restart_resumes_on_startup(monkeypatch, tmp_path: Path):
    """配置变更触发重建，重建途中进程被杀（uvicorn --reload）：指纹已是新的，但重启后仍要继续重建。"""
    import threading

    db = _FakeDB("old", {"kb": ["COMPLETED"] * 7})
    indexed = {"kb": 7}
    mod = _reindex_module(monkeypatch, tmp_path, db, indexed)

    started, release = threading.Event(), threading.Event()

    def hang(kb_id):                      # 重建进行中……进程在这里"死掉"（线程永远不会正常结束）
        started.set()
        release.wait(5)

    monkeypatch.setattr(mod.document_service, "reindex_kb", hang)
    before = mod.ReindexService()
    try:
        before._check_and_enqueue("策略配置变更")
        assert started.wait(5)
        assert db.settings["ingest_fingerprint"] == "new"
        assert db.settings["reindex_pending"] == {"kb": "入库参数变更"}
        indexed["kb"] = 1                 # 被打断时索引里只剩一份文档

        rebuilt: list[str] = []
        monkeypatch.setattr(mod.document_service, "reindex_kb", rebuilt.append)
        after = mod.ReindexService()      # 重启：内存里的队列 / 任务状态全没了，只剩数据库
        after._check_and_enqueue("启动检查")
        _drain(after)

        assert rebuilt == ["kb"]
        assert after._jobs["kb"]["reason"] == "上次重建被中断，继续重建" and after._jobs["kb"]["status"] == "done"
        assert db.settings["reindex_pending"] == {}             # 跑完才清除

        rebuilt.clear()
        indexed["kb"] = 7
        again = mod.ReindexService()      # 再重启一次：已经完成，不再重复重建
        again._check_and_enqueue("启动检查")
        _drain(again)
        assert rebuilt == []
    finally:
        release.set()
        _drain(before)


def test_crash_before_enqueue_keeps_old_fingerprint(monkeypatch, tmp_path: Path):
    """检查过程中途退出（还没把受影响的知识库记入待重建）：新指纹不能先落库，否则重启后认为"配置没变"。"""
    db = _FakeDB("old", {"kb": ["COMPLETED"] * 3})
    mod = _reindex_module(monkeypatch, tmp_path, db, {"kb": 3})

    crashing = mod.ReindexService()
    monkeypatch.setattr(crashing, "enqueue", lambda *a: (_ for _ in ()).throw(RuntimeError("进程被杀")))
    crashing._check_and_enqueue("策略配置变更")
    assert db.settings["ingest_fingerprint"] == "old"
    assert "reindex_pending" not in db.settings

    rebuilt: list[str] = []
    monkeypatch.setattr(mod.document_service, "reindex_kb", rebuilt.append)
    restarted = mod.ReindexService()
    restarted._check_and_enqueue("启动检查")
    _drain(restarted)
    assert rebuilt == ["kb"] and restarted._jobs["kb"]["reason"] == "入库参数变更"
    assert db.settings["ingest_fingerprint"] == "new" and db.settings["reindex_pending"] == {}


def test_startup_rebuilds_kb_whose_index_lost_documents(monkeypatch, tmp_path: Path):
    """指纹一致、也没有待重建记录，但索引只有 1/7 份文档（旧版本留下的残缺索引）→ 重建；完整的不动。"""
    db = _FakeDB("new", {"broken": ["COMPLETED"] * 7 + ["FAILED"], "fine": ["COMPLETED"] * 3 + ["PROCESSING"]})
    mod = _reindex_module(monkeypatch, tmp_path, db, {"broken": 1, "fine": 3})

    rebuilt: list[str] = []
    monkeypatch.setattr(mod.document_service, "reindex_kb", rebuilt.append)
    svc = mod.ReindexService()
    svc._check_and_enqueue("启动检查")
    _drain(svc)

    assert rebuilt == ["broken"]                                 # 失败 / 处理中的文档不计入应有数量
    assert svc._jobs["broken"]["reason"] == "索引不完整（1/7 份文档）"


# --- 重建途中个别文档失败 ---------------------------------------------------------------------
def _rebuild_with_failures(monkeypatch, tmp_path: Path, fail: set[str]):
    """真实的 index_service 备份 / 恢复 + 假的文档入库：fail 里的文档入库失败（一个抛异常、其余像 process() 一样只标 FAILED）。"""
    import importlib

    svc = _index_service(monkeypatch, tmp_path)
    mod = importlib.import_module("app.services.document_service")
    _write_index(svc.directory("kb"), 3, "完整的旧索引")
    src = tmp_path / "a.md"
    src.write_text("# a", encoding="utf-8")
    docs = {i: SimpleNamespace(id=i, file_name=f"{i}.md", status="COMPLETED", file_path=str(src), markdown_content=f"# {i}", parser="markdown+edited" if i == "d1" else "markdown", updated_at=None) for i in ("d1", "d2", "d3")}

    class FakeRepo:
        def __init__(self, _db):
            pass

        def list_documents(self, _kb_id):
            return list(docs.values())

        def get_document(self, doc_id):
            return docs.get(doc_id)

    written: list[str] = []

    def ingest(doc_id, *_):
        if doc_id in fail:
            docs[doc_id].status = "FAILED"
            if doc_id == "d1":            # update_markdown 会把异常抛出来
                raise ValueError("坏的 Markdown")
            return
        written.append(doc_id)
        _write_index(svc.target_dir("kb"), len(written), "新索引")   # 重建线程写临时目录

    monkeypatch.setattr(mod, "SessionLocal", lambda: nullcontext(None))
    monkeypatch.setattr(mod, "Repository", FakeRepo)
    monkeypatch.setattr(mod.document_service, "process", ingest)
    monkeypatch.setattr(mod.document_service, "update_markdown", ingest)
    return svc, mod


def test_rebuild_continues_past_failed_documents(monkeypatch, tmp_path: Path):
    """一份文档抛异常、一份被标 FAILED：其余照常重建，重建正常收尾（不留备份，下次检查不会把旧索引恢复回来），返回失败清单。"""
    svc, mod = _rebuild_with_failures(monkeypatch, tmp_path, fail={"d1", "d2"})

    assert mod.document_service.reindex_kb("kb") == ["d1.md", "d2.md"]
    assert (svc.directory("kb") / "chunks.jsonl").read_text(encoding="utf-8") == "新索引" and svc.indexed_docs("kb") == 1
    assert svc.recover_interrupted() == []


def test_rebuild_where_every_document_fails_restores_previous_index(monkeypatch, tmp_path: Path):
    """全部失败（例如向量模型加载不了）：不能用空索引换掉完整的旧索引——立即恢复并报错。"""
    import pytest

    svc, mod = _rebuild_with_failures(monkeypatch, tmp_path, fail={"d1", "d2", "d3"})

    with pytest.raises(RuntimeError, match="全部 3 份文档重建失败，已保留重建前的索引"):
        mod.document_service.reindex_kb("kb")
    assert (svc.directory("kb") / "chunks.jsonl").read_text(encoding="utf-8") == "完整的旧索引" and svc.indexed_docs("kb") == 3
    assert svc.recover_interrupted() == []                    # 备份已用掉


def test_partial_rebuild_is_done_with_failed_docs_reported(monkeypatch, tmp_path: Path):
    """服务层：部分文档失败算完成（清掉待重建标记，不会每次启动重跑），失败清单挂在任务上给前端提示。"""
    db = _FakeDB("old", {"kb": ["COMPLETED"] * 3})
    mod = _reindex_module(monkeypatch, tmp_path, db, {"kb": 3})
    monkeypatch.setattr(mod.document_service, "reindex_kb", lambda kb_id: ["坏文件.pdf"])

    svc = mod.ReindexService()
    svc._check_and_enqueue("策略配置变更")
    _drain(svc)

    job = svc.status()["recent"][0]
    assert job["status"] == "done" and job["error"] is None and job["failedDocs"] == ["坏文件.pdf"]
    assert db.settings["reindex_pending"] == {}


def test_swap_waits_for_queries_without_deadlock(monkeypatch, tmp_path: Path):
    """检索持读锁时会在 get() 里短暂取知识库锁；重建收尾必须按 swap_guard → lock 的顺序加锁，否则两边互相等死。
    同时验证：替换等在途检索结束才发生，替换期间新检索被挡住。"""
    import threading
    import time

    _index_service(monkeypatch, tmp_path)
    svc = _fresh_service()
    events: list[str] = []
    in_query = threading.Event()

    def query():
        with svc.reading("kb"):
            in_query.set()
            time.sleep(0.3)
            with svc.lock("kb"):                  # 相当于 get() 取缓存
                events.append("query-done")

    def rebuild_wrap_up():
        in_query.wait(2)
        with svc.lock("kb"):                      # 第 1 步：只挡写入者
            events.append("reconcile")
        with svc.swap_guard("kb"), svc.lock("kb"):
            events.append("swap")
            with svc.reading("kb"):               # 持写锁的线程读自己的数据：不能自锁
                events.append("swap-read")

    threads = [threading.Thread(target=query), threading.Thread(target=rebuild_wrap_up)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert not any(t.is_alive() for t in threads), f"死锁：{events}"
    assert events.index("query-done") < events.index("swap")      # 替换等在途检索结束
    assert events[-2:] == ["swap", "swap-read"]


def test_running_rebuild_is_not_requeued_as_interrupted(monkeypatch, tmp_path: Path):
    """重建进行中（待重建记录还在）时又触发一次检查（保存任意配置都会）：不能把它当作"上次中断"再排一次完整重建。"""
    db = _FakeDB("new", {"kb": ["COMPLETED"] * 3})
    db.settings["reindex_pending"] = {"kb": "入库参数变更"}
    mod = _reindex_module(monkeypatch, tmp_path, db, {"kb": 3})
    svc = mod.ReindexService()
    svc._jobs["kb"] = {"kbId": "kb", "name": "知识库 kb", "status": "running", "reason": "入库参数变更", "error": None, "startedAt": 0, "finishedAt": None, "rerun": False}
    calls: list[tuple] = []
    monkeypatch.setattr(svc, "enqueue", lambda *a: calls.append(a))
    svc._check_and_enqueue("策略配置变更")
    assert calls == []


# --- 自动检查 vs 在途写入：保存配置时用户正在上传 / 删除 --------------------------------------------
def _in_flight_write(mod, kb_id: str) -> ExitStack:
    """本进程里一次进行中的写入（document_service.process / update_markdown / delete 整段登记为在途写入）。close() = 写入结束。"""
    stack = ExitStack()
    stack.enter_context(mod.document_service._writing(kb_id))
    return stack


def test_doc_finishing_ingestion_during_check_does_not_trigger_rebuild(monkeypatch, tmp_path: Path):
    """检查读到的两个数对不上：新文档在数据库里已是 COMPLETED，索引里却还只有 1 份（索引落后于数据库——修复前入库途中保存配置
    就会这样），而它的入库还没结束；接着它在检查进行中入库完毕。在途写入期间看到的缺口不能直接拿来整库重建（所有文档会回到处理中），
    要等写入结束再核对——这时索引是完整的。"""
    db = _FakeDB("new", {"kb-race": ["COMPLETED", "COMPLETED"]})
    indexed = {"kb-race": 1}
    mod = _reindex_module(monkeypatch, tmp_path, db, indexed)
    rebuilt: list[str] = []
    monkeypatch.setattr(mod.document_service, "reindex_kb", rebuilt.append)
    reads: list[int] = []
    monkeypatch.setattr(mod.index_service, "indexed_docs", lambda kb_id: reads.append(indexed[kb_id]) or indexed[kb_id])

    events: list[str] = []
    upload = _in_flight_write(mod, "kb-race")

    def finish_upload() -> None:
        indexed["kb-race"] = 2            # 索引保存可见
        upload.close()                    # 入库结束
        events.append("upload finished")

    fake_repo = mod.Repository

    def repo_with_timing_hook(session):   # 时间钩子：检查走到最后一步（写新指纹）时，那份文档入库完毕
        r = fake_repo(session)
        set_setting = r.set_setting

        def set_and_finish(key, value):
            set_setting(key, value)
            if key == mod.FINGERPRINT_KEY:
                finish_upload()

        r.set_setting = set_and_finish
        return r

    monkeypatch.setattr(mod, "Repository", repo_with_timing_hook)
    svc = mod.ReindexService()
    try:
        svc._check_and_enqueue("策略配置变更")
        events.append("check returned")
        _drain(svc)
    finally:
        upload.close()

    assert events == ["upload finished", "check returned"]    # 文档确实是在检查进行中入库完毕的
    assert rebuilt == [] and svc._jobs == {}
    assert reads == [1, 2]                               # 快照预筛读到错开的 1/2；在途期间不再读；入库结束后的核对读到 2/2


def test_real_shortfall_seen_during_ingestion_is_rebuilt_once_it_ends(monkeypatch, tmp_path: Path):
    """反过来：索引真的缺文档（3 份完成、索引只有 1 份），检查时恰好有人在上传——推迟到入库结束再核对，仍然少（2/4）就重建。
    在途入库只推迟结论，不会吞掉真正的"索引不完整"。"""
    db = _FakeDB("new", {"kb-short": ["COMPLETED"] * 3 + ["PROCESSING"]})
    indexed = {"kb-short": 1}
    mod = _reindex_module(monkeypatch, tmp_path, db, indexed)
    rebuilt: list[str] = []
    monkeypatch.setattr(mod.document_service, "reindex_kb", rebuilt.append)

    upload = _in_flight_write(mod, "kb-short")
    svc = mod.ReindexService()
    try:
        svc._check_and_enqueue("策略配置变更")
        assert svc._jobs == {}                           # 有在途入库：先不下结论
        db.kbs["kb-short"][-1] = "COMPLETED"             # 上传的文档入库完成：索引 +1、状态 COMPLETED
        indexed["kb-short"] = 2
    finally:
        upload.close()                                   # 入库结束 → 再核对
    _drain(svc)

    assert rebuilt == ["kb-short"] and svc._jobs["kb-short"]["reason"] == "索引不完整（2/4 份文档）"


def test_delete_straddling_the_recheck_does_not_trigger_rebuild(monkeypatch, tmp_path: Path):
    """删除是先删数据库记录、再改索引：核对时读完成数（还算着它）和读索引（已经没有它）之间正好删掉一份，两边同样会错开。
    走真实的 document_service.delete()（整段登记为在途写入）：两次读之间发生过写入就不下结论，写入结束后重新核对。"""
    import importlib

    db = _FakeDB("new", {"kb-del": ["COMPLETED"] * 3})
    indexed = {"kb-del": 3}
    mod = _reindex_module(monkeypatch, tmp_path, db, indexed)
    ds_mod = importlib.import_module("app.services.document_service")
    rows = {d: SimpleNamespace(id=d, kb_id="kb-del", file_name=f"{d}.md", file_path=None, parser="markdown", chunk_count=1, markdown_content=f"# {d}") for d in ("a", "d1", "d2")}

    class DocRepo:
        def __init__(self, _db):
            pass

        def get_document(self, doc_id):
            return rows.get(doc_id)

        def delete_document(self, doc):
            del rows[doc.id]
            db.kbs["kb-del"].pop()                       # 先删数据库记录……

    class FakeIndexService:
        def lock(self, kb_id):
            return nullcontext()

        def get(self, kb_id):
            return SimpleNamespace(remove_document=lambda doc_id: None)

        def save(self, kb_id, idx=None):
            indexed[kb_id] -= 1                          # ……再改索引

    monkeypatch.setattr(ds_mod, "SessionLocal", lambda: nullcontext(None))
    monkeypatch.setattr(ds_mod, "Repository", DocRepo)
    monkeypatch.setattr(ds_mod, "index_service", FakeIndexService())
    monkeypatch.setattr(ds_mod, "_lake_event", lambda *a, **k: None)
    rebuilt: list[str] = []
    monkeypatch.setattr(mod.document_service, "reindex_kb", rebuilt.append)

    deletes = ["d1", "d2"]
    reads: list[int] = []

    def indexed_docs(kb_id):              # 时间钩子：检查每次读索引之前，恰好有一份文档删完
        if deletes:
            ds_mod.document_service.delete(deletes.pop(0))
        reads.append(indexed[kb_id])
        return indexed[kb_id]

    monkeypatch.setattr(mod.index_service, "indexed_docs", indexed_docs)
    svc = mod.ReindexService()
    svc._check_and_enqueue("策略配置变更")
    _drain(svc)

    # 快照 3 份 → 预筛读到 2（删了 d1）→ 核对：完成数 2、索引读到 1（中间删了 d2）→ 有写入，重新核对：1/1
    assert reads == [2, 1, 1] and db.kbs["kb-del"] == ["COMPLETED"]
    assert rebuilt == [] and svc._jobs == {}


def test_recheck_gives_up_after_repeated_collisions(monkeypatch, tmp_path: Path):
    """每次核对都碰上写入（上传接连不断）：最多再核对 3 次就放手、留给下一次检查——不排队、不留回调、不无限递归。"""
    db = _FakeDB("new", {"kb-busy": ["COMPLETED"] * 3})
    indexed = {"kb-busy": 1}
    mod = _reindex_module(monkeypatch, tmp_path, db, indexed)
    rebuilt: list[str] = []
    monkeypatch.setattr(mod.document_service, "reindex_kb", rebuilt.append)
    reads: list[int] = []
    still_writing = ExitStack()

    def indexed_docs(kb_id):              # 时间钩子：前 5 次读索引都碰上写入，第 5 次（最后一次核对）时那次写入还没结束
        if len(reads) < 4:
            with mod.document_service._writing(kb_id):
                reads.append(indexed[kb_id])
        elif len(reads) == 4:
            still_writing.enter_context(mod.document_service._writing(kb_id))
            reads.append(indexed[kb_id])
        else:
            reads.append(indexed[kb_id])
        return indexed[kb_id]

    monkeypatch.setattr(mod.index_service, "indexed_docs", indexed_docs)
    svc = mod.ReindexService()
    try:
        svc._check_and_enqueue("策略配置变更")
        assert "kb-busy" not in mod.document_service._idle_hooks    # 已放手：不再挂回调
    finally:
        still_writing.close()                            # 那次写入结束：不会再触发核对
    _drain(svc)

    assert len(reads) == 1 + 4                           # 快照预筛 1 次 + 首次核对与 3 次重试
    assert rebuilt == [] and svc._jobs == {}


def test_processing_rows_left_by_a_dead_process_do_not_defer_the_check(monkeypatch, tmp_path: Path):
    """热重载 / 崩溃会留下永远停在 PROCESSING 的文档：它们不是本进程的在途写入，不能让"索引不完整"的检查一直推迟下去
    （所以"在途"看的是本进程登记的写入，而不是数据库里的状态）。"""
    db = _FakeDB("new", {"kb-zombie": ["COMPLETED"] * 3 + ["PROCESSING"]})
    mod = _reindex_module(monkeypatch, tmp_path, db, {"kb-zombie": 1})
    rebuilt: list[str] = []
    monkeypatch.setattr(mod.document_service, "reindex_kb", rebuilt.append)

    svc = mod.ReindexService()
    svc._check_and_enqueue("启动检查")
    _drain(svc)

    assert rebuilt == ["kb-zombie"] and svc._jobs["kb-zombie"]["reason"] == "索引不完整（1/3 份文档）"


# --- 入库途中清空了索引缓存（PUT /settings、切换模型都会 invalidate）：写入不能丢 -----------------------
def _ingest_with_cache_cleared_midway(monkeypatch, tmp_path: Path, *, switch_model: bool):
    """真实的 document_service.process / update_markdown + IndexService + IngestPipeline（离线 lite 配置），
    文档 b 入库途中缓存被清空；switch_model=True 时同时切换了向量模型。
    返回 (index_service, document_service 模块, kb, 文档, 每次清空时的在途写入数, 排队 / 记下的重建, 入库前的写入计数)。"""
    import importlib

    from prism.config import Settings
    from prism.index.embeddings import HashEmbedder, build_embedder
    from prism.ingest import IngestPipeline

    svc = _index_service(monkeypatch, tmp_path)
    is_mod = importlib.import_module("app.services.index_service")
    ds_mod = importlib.import_module("app.services.document_service")
    rs = importlib.import_module("app.services.reindex_service").reindex_service
    from app.services.model_service import model_service

    class NewModel(HashEmbedder):         # 切换后的向量模型：签名与原来的哈希向量不同
        name = "hash-v2"

    lite = Settings.lite(index_dir=tmp_path / "lite")    # 离线：哈希向量、无 LLM
    current = {"embedder": build_embedder(lite.embedding)}
    monkeypatch.setattr(ds_mod.settings_service, "get", lambda: lite)
    monkeypatch.setattr(model_service, "embedder", lambda: current["embedder"])
    monkeypatch.setattr(ds_mod, "_lake_event", lambda *a, **k: None)
    scheduled: list[tuple] = []
    monkeypatch.setattr(rs, "enqueue", lambda *a: scheduled.append(("enqueue", *a)))
    monkeypatch.setattr(rs, "defer", lambda *a: scheduled.append(("defer", *a)))

    kb = "kb-model-switch" if switch_model else "kb-settings-save"

    class IndexRepo:                      # index_service 读别名 / 剔除清单
        def __init__(self, _db):
            pass

        def get_setting(self, key, default=None):
            return default

    monkeypatch.setattr(is_mod, "SessionLocal", lambda: nullcontext(None))
    monkeypatch.setattr(is_mod, "Repository", IndexRepo)

    docs: dict[str, SimpleNamespace] = {}
    for name, text in (("a", "# A\n\n第一份文档。"), ("b", "# B\n\n第二份文档，入库途中缓存被清空。")):
        src = tmp_path / f"{name}.md"
        src.write_text(text, encoding="utf-8")
        docs[name] = SimpleNamespace(
            id=name, kb_id=kb, file_name=f"{name}.md", file_path=str(src), file_format="md", status="PENDING", error_message=None,
            markdown_content=None, auto_meta=None, parser=None, title=name, version=None, valid_from=None, valid_to=None,
            tags=None, department=None, access_level=0, allowed_departments=None,
        )

    class DocRepo:
        def __init__(self, _db):
            pass

        def get_document(self, doc_id):
            return docs.get(doc_id)

        def update_document(self, doc, **fields):
            for k, v in fields.items():
                setattr(doc, k, v)
            return doc

    monkeypatch.setattr(ds_mod, "SessionLocal", lambda: nullcontext(None))
    monkeypatch.setattr(ds_mod, "Repository", DocRepo)

    in_flight: list[int] = []
    ingest = IngestPipeline.ingest_document

    def ingest_then_clear_cache(self, parsed, *a, **k):
        out = ingest(self, parsed, *a, **k)
        if parsed.doc_id == "b":
            started, finished = ds_mod.document_service.index_writes(kb)
            in_flight.append(started - finished)
            if switch_model:
                current["embedder"] = NewModel(lite.embedding.dim)
            svc.invalidate()                             # PUT /api/v1/settings/ 或切换模型，落在入库途中
        return out

    monkeypatch.setattr(IngestPipeline, "ingest_document", ingest_then_clear_cache)
    svc.invalidate()
    base = ds_mod.document_service.index_writes(kb)
    ds_mod.document_service.process("a")
    ds_mod.document_service.process("b")
    return svc, ds_mod, kb, docs, in_flight, scheduled, base


def test_settings_save_during_ingest_keeps_the_document_in_the_index(monkeypatch, tmp_path: Path):
    """入库途中保存策略配置（清空索引缓存）：这份文档的索引保存曾被静默跳过——状态 COMPLETED、索引里却没有，
    之后的自动检查判"索引不完整"、整库重建。现在 process() / update_markdown() 保存它们实际写入的索引对象，整段登记为在途写入。"""
    svc, ds_mod, kb, docs, in_flight, scheduled, base = _ingest_with_cache_cleared_midway(monkeypatch, tmp_path, switch_model=False)
    try:
        assert [docs[x].status for x in "ab"] == ["COMPLETED", "COMPLETED"], [docs[x].error_message for x in "ab"]
        assert svc.indexed_docs(kb) == 2 and {"a", "b"} <= set(svc.get(kb).store.docs)

        ds_mod.document_service.update_markdown("b", "# B\n\n校对后的正文")   # 校对回灌途中同样清空一次缓存
        assert docs["b"].status == "COMPLETED"
        assert any("校对后的正文" in c.text for c in svc.get(kb).store.chunks.values() if c.doc_id == "b")

        assert in_flight == [1, 1] and scheduled == []   # 两次都登记为在途写入；向量模型没变，不必重建
        started, finished = ds_mod.document_service.index_writes(kb)
        assert (started - base[0], finished - base[1]) == (3, 3)    # 三次写入，结束后不再在途
    finally:
        svc.invalidate()


def test_model_switch_during_ingest_saves_the_document_and_records_a_pending_rebuild(monkeypatch, tmp_path: Path):
    """入库途中切换了向量模型：文档照样保存（不丢），但写进去的是旧模型的向量，得重建——光靠之后的检查不够：之后的入库会按新模型
    加载这份索引、把 index_meta 里的模型改写成新的，"向量模型变更"再也查不出来。所以记一笔持久化的待重建（下一次检查执行）；
    不当场排队，免得和切换模型触发的那次检查撞车、整库重建两遍。"""
    svc, _ds_mod, kb, docs, in_flight, scheduled, _base = _ingest_with_cache_cleared_midway(monkeypatch, tmp_path, switch_model=True)
    try:
        assert [docs[x].status for x in "ab"] == ["COMPLETED", "COMPLETED"], [docs[x].error_message for x in "ab"]
        assert svc.indexed_docs(kb) == 2 and in_flight == [1]
        assert scheduled == [("defer", kb, "向量模型变更")]
    finally:
        svc.invalidate()


# --- 重建途中又触发检查：只有会改变结果，才跑完再来一遍 ---------------------------------------------
def test_rebuild_is_marked_started_before_it_reads_settings_or_model(monkeypatch, tmp_path: Path):
    """"已开始"必须先于读配置 / 模型登记：检查看到"还没开始"就不让任务再来一遍，前提是这次重建之后才读配置 / 模型，
    读得到检查之前就落库的变更。替换完成后记录仍保留（任务此刻还没标记完成），forget_rebuild 才清掉。"""
    import importlib

    from prism.config import Settings
    from prism.index.embeddings import HashEmbedder

    _index_service(monkeypatch, tmp_path)
    is_mod = importlib.import_module("app.services.index_service")
    svc = _fresh_service()
    lite = Settings.lite(index_dir=tmp_path / "lite")
    emb = HashEmbedder(lite.embedding.dim)
    seen: list[tuple] = []
    monkeypatch.setattr(is_mod.settings_service, "get", lambda: seen.append(("settings", svc.rebuild_embedder("kb"))) or lite)
    monkeypatch.setattr(is_mod.model_service, "embedder", lambda: seen.append(("model", svc.rebuild_embedder("kb"))) or emb)

    assert svc.rebuild_embedder("kb") == (False, None)
    svc.begin_rebuild("kb")
    assert seen == [("settings", (True, None)), ("model", (True, None))]
    assert svc.rebuild_embedder("kb") == (True, ("hash", None))
    svc.finish_rebuild("kb")
    assert svc.rebuild_embedder("kb") == (True, ("hash", None))
    svc.forget_rebuild("kb")
    assert svc.rebuild_embedder("kb") == (False, None)


def _rebuild_with_hooks(monkeypatch, tmp_path: Path, *, qdrant: bool = False) -> SimpleNamespace:
    """真实的 ReindexService 工作线程 + 真实的 IndexService 整库重建（begin_rebuild / finish_rebuild 和它们留下的记录），
    假的文档入库：每遍重建写一份记着所用向量模型的索引。正式索引起初是模型 a 建的，run() 换到模型 b 触发第一遍重建。
    hooks[时刻] = 回调，只在第一遍重建的 before_begin / after_begin / after_finish 时刻执行：换模型、改入库参数、
    保存检索期配置，各自跑一次自动检查——检查在单独的线程里跑完再继续（相当于 request() 起的检查线程），不靠 sleep。"""
    import functools
    import importlib
    import threading

    from prism.config import Settings
    from prism.index.embeddings import HashEmbedder

    index_svc = _index_service(monkeypatch, tmp_path)
    is_mod = importlib.import_module("app.services.index_service")
    db = _FakeDB("v1", {"kb": ["COMPLETED"] * 3})
    mod = _reindex_module(monkeypatch, tmp_path, db, {"kb": 3})
    fingerprint = {"value": "v1"}
    monkeypatch.setattr(mod, "ingest_fingerprint", lambda: fingerprint["value"])
    # 真实判断：正式索引记着的模型 ≠ 当前模型
    monkeypatch.setattr(mod.index_service, "needs_reindex", functools.partial(is_mod.IndexService.needs_reindex, index_svc))

    lite = Settings.lite(index_dir=tmp_path / "lite")
    if qdrant:
        lite = lite.model_copy(update={"vector_store": lite.vector_store.model_copy(update={"backend": "qdrant"})})
    models = {n: type(f"Model{n.upper()}", (HashEmbedder,), {"name": f"hash-{n}"})(lite.embedding.dim) for n in "abc"}
    current = {"model": models["a"]}
    monkeypatch.setattr(is_mod.settings_service, "get", lambda: lite)
    monkeypatch.setattr(is_mod.model_service, "embedder", lambda: current["model"])
    # 上一次重建（模型 a）留下的记录：新任务开始时要作废，否则 begin_rebuild 之前到来的变更会被当成"重建已开始"
    monkeypatch.setattr(index_svc, "_builds", {"kb": is_mod.embedder_signature(models["a"])})

    def write(directory: Path, model) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        name, model_name = is_mod.embedder_signature(model)
        (directory / "index_meta.json").write_text(json.dumps({"docs": 3, "chunks": 3, "embedder": name, "model": model_name}), encoding="utf-8")

    write(index_svc.directory("kb"), models["a"])
    hooks: dict = {}
    built: list[str] = []

    def reindex_kb(kb_id):
        step = (lambda when: hooks.get(when, lambda: None)()) if not built else (lambda when: None)
        step("before_begin")
        index_svc.begin_rebuild(kb_id)
        try:
            in_place = index_svc.target_dir(kb_id) == index_svc.directory(kb_id)
            used = current["model"] if in_place else index_svc.get(kb_id).embedder    # 原地重建逐份按当时的模型写入
            built.append(used.name)
            write(index_svc.target_dir(kb_id), used)
            step("after_begin")
            index_svc.finish_rebuild(kb_id)
        except BaseException:
            index_svc.abort_rebuild(kb_id)
            raise
        step("after_finish")
        return []

    monkeypatch.setattr(mod.document_service, "reindex_kb", reindex_kb)
    svc = mod.ReindexService()

    def check(reason: str) -> None:
        t = threading.Thread(target=svc._check_and_enqueue, args=(reason,))
        t.start()
        t.join(5)
        assert not t.is_alive()

    def switch_model(name: str) -> None:
        current["model"] = models[name]
        check("模型配置变更")

    def change_ingest_settings() -> None:
        fingerprint["value"] = "v2"
        check("策略配置变更")

    def run() -> None:
        current["model"] = models["b"]
        svc.enqueue("kb", "知识库 kb", mod.MODEL_CHANGED)
        _drain(svc)

    return SimpleNamespace(
        svc=svc, db=db, index=index_svc, hooks=hooks, built=built, run=run, switch_model=switch_model, change_ingest_settings=change_ingest_settings,
        save_retrieval_settings=lambda: check("策略配置变更"),     # 检索期参数：指纹不变，但 PUT /settings 照样触发检查
    )


@pytest.mark.parametrize("when", ["before_begin", "after_begin", "after_finish"])
def test_unrelated_check_during_rebuild_does_not_rebuild_twice(monkeypatch, tmp_path: Path, when: str):
    """换模型触发的重建进行中，又保存了一次检索期配置。替换之前正式索引一直记着旧模型，检查据此判"向量模型变更"、给在跑的任务
    记上"跑完再来一遍"——同一个模型整库重建两遍（文档全部回到处理中、MinerU 重新解析、索引增强的 LLM 调用再付一次费）。
    检查落在重建的哪个阶段都只该重建一遍。"""
    env = _rebuild_with_hooks(monkeypatch, tmp_path)
    reruns: list[bool] = []

    def hook() -> None:
        env.save_retrieval_settings()
        reruns.append(env.svc._jobs["kb"]["rerun"])

    env.hooks[when] = hook
    env.run()
    assert reruns == [False] and env.built == ["hash-b"]
    assert env.svc._jobs["kb"]["status"] == "done" and env.db.settings["reindex_pending"] == {}


@pytest.mark.parametrize(
    ("when", "change", "built"),
    [
        ("before_begin", "model:c", ["hash-c"]),             # 还没读模型：这一遍就用上 c
        ("before_begin", "ingest", ["hash-b"]),              # 还没读配置：这一遍就用上新参数
        ("after_begin", "model:c", ["hash-b", "hash-c"]),    # 已按 b 在建：建完再按 c 来一遍
        ("after_begin", "model:a", ["hash-b", "hash-a"]),    # 换回 a：正式索引（a）看着不过期，可这一遍替换后就是 b 了
        ("after_begin", "ingest", ["hash-b", "hash-b"]),     # 已入库的文档用的是旧参数
        ("after_finish", "model:c", ["hash-b", "hash-c"]),   # 已替换、任务还没标记完成
        ("after_finish", "ingest", ["hash-b", "hash-b"]),
    ],
)
def test_change_the_running_rebuild_missed_rebuilds_once_more(monkeypatch, tmp_path: Path, when: str, change: str, built: list[str]):
    """反过来：重建已经读过的东西又变了，跑完要再来一遍（且只多一遍）；它还没读到的变更这一遍就会用上，不必再来。"""
    env = _rebuild_with_hooks(monkeypatch, tmp_path)
    env.hooks[when] = (lambda: env.switch_model(change[-1])) if change.startswith("model:") else env.change_ingest_settings
    env.run()
    assert env.built == built
    assert env.svc._jobs["kb"]["status"] == "done" and env.db.settings["reindex_pending"] == {}


def test_later_check_does_not_cancel_an_earlier_rerun(monkeypatch, tmp_path: Path):
    """先改了入库参数（已开始的重建得再来一遍），接着又保存一次检索期配置（单看模型不必再来）：后一次不能撤销前一次的决定。"""
    env = _rebuild_with_hooks(monkeypatch, tmp_path)

    def hook() -> None:
        env.change_ingest_settings()
        env.save_retrieval_settings()

    env.hooks["after_begin"] = hook
    env.run()
    assert env.built == ["hash-b", "hash-b"]


@pytest.mark.parametrize(("change", "built"), [("retrieval", ["hash-b"]), ("model:c", ["hash-b", "hash-c"])])
def test_in_place_rebuild_stays_conservative(monkeypatch, tmp_path: Path, change: str, built: list[str]):
    """Qdrant 原地重建没有临时索引、不记签名（逐份文档按当时的模型写进正式目录，可能新旧混杂）：看正式目录记着的模型，
    没变就不再来一遍，变了就再来一遍。"""
    env = _rebuild_with_hooks(monkeypatch, tmp_path, qdrant=True)
    env.hooks["after_begin"] = env.save_retrieval_settings if change == "retrieval" else (lambda: env.switch_model("c"))
    env.run()
    assert env.built == built
    assert env.index.rebuild_embedder("kb") == (True, None)
