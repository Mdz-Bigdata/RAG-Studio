"""单文档「重新解析」不能静默覆盖人工校对：默认用校对后的 Markdown 重建；显式放弃校对才回到源文件，并先备份校对版。"""

from __future__ import annotations

import io
import os

import pytest

SOURCE = "# 休假制度\n\n## 年假\n\n员工每年享有 5 天带薪年假。\n"
EDIT = "补充（校对）：当年未休完的年假可以跨年使用，最晚次年 3 月 31 日前休完。"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("reprocess")
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


def _doc(client, doc_id):
    d = client.get(f"/api/v1/documents/{doc_id}").json()
    md = client.get(f"/api/v1/documents/{doc_id}/markdown").json()["markdownContent"]
    chunks = " ".join(c["text"] for c in client.get(f"/api/v1/documents/{doc_id}/chunks").json())
    return d, md, chunks


def test_reprocess_keeps_manual_edits_unless_discarded(client):
    kb = client.post("/api/v1/knowledge-bases/", json={"name": "校对保护", "description": "重新解析回归"}).json()
    r = client.post(f"/api/v1/documents/upload?kbId={kb['id']}", files={"file": ("休假制度.md", io.BytesIO(SOURCE.encode()), "text/markdown")})
    doc_id = r.json()["id"]
    d, md, _ = _doc(client, doc_id)
    assert d["status"] == "COMPLETED" and "edited" not in (d["parser"] or "")

    r = client.put(f"/api/v1/documents/{doc_id}/markdown", json={"markdownContent": md.rstrip() + "\n\n" + EDIT + "\n"})
    assert r.status_code == 200 and "edited" in r.json()["parser"]

    # 默认重新解析：保留校对内容（索引里也还在）
    assert client.post(f"/api/v1/documents/{doc_id}/reprocess").status_code == 200
    d, md, chunks = _doc(client, doc_id)
    assert d["status"] == "COMPLETED" and "edited" in d["parser"]
    assert EDIT in md and "跨年使用" in chunks
    assert client.post(f"/api/v1/documents/{doc_id}/reprocess", json={"discardEdits": False}).status_code == 200
    assert EDIT in _doc(client, doc_id)[1]

    # 显式放弃校对：回到源文件，校对版先备份到上传目录
    from app.config import app_config

    assert client.post(f"/api/v1/documents/{doc_id}/reprocess", json={"discardEdits": True}).status_code == 200
    d, md, chunks = _doc(client, doc_id)
    assert d["status"] == "COMPLETED" and "edited" not in (d["parser"] or "")
    assert EDIT not in md and "跨年使用" not in chunks
    backups = list((app_config.upload_dir / kb["id"]).glob(f"{doc_id}.edited-*.md"))
    assert len(backups) == 1 and EDIT in backups[0].read_text(encoding="utf-8")


def test_restore_edit_backup(client):
    kb = client.post("/api/v1/knowledge-bases/", json={"name": "校对恢复", "description": "备份恢复回归"}).json()
    doc_id = client.post(f"/api/v1/documents/upload?kbId={kb['id']}", files={"file": ("休假制度.md", io.BytesIO(SOURCE.encode()), "text/markdown")}).json()["id"]
    md = _doc(client, doc_id)[1]
    client.put(f"/api/v1/documents/{doc_id}/markdown", json={"markdownContent": md.rstrip() + "\n\n" + EDIT + "\n"})
    assert client.get(f"/api/v1/documents/{doc_id}").json()["editBackups"] == 0
    client.post(f"/api/v1/documents/{doc_id}/reprocess", json={"discardEdits": True})        # 放弃校对 → 自动备份

    backups = client.get(f"/api/v1/documents/{doc_id}/edit-backups").json()
    assert len(backups) == 1 and EDIT in backups[0]["preview"] and backups[0]["identical"] is False
    assert client.get(f"/api/v1/documents/{doc_id}").json()["editBackups"] == 1

    # 恢复：文档回到校对版；当前是源文件解析结果（未校对），不需要再备份
    r = client.post(f"/api/v1/documents/{doc_id}/edit-backups/{backups[0]['name']}/restore")
    assert r.status_code == 200
    d, md, chunks = _doc(client, doc_id)
    assert d["status"] == "COMPLETED" and "edited" in d["parser"] and EDIT in md and "跨年使用" in chunks
    assert d["editBackups"] == 1 and client.get(f"/api/v1/documents/{doc_id}/edit-backups").json()[0]["identical"] is True

    # 改成另一个校对版后再恢复旧的：当前校对版先备份，恢复本身可撤销
    client.put(f"/api/v1/documents/{doc_id}/markdown", json={"markdownContent": SOURCE + "\n第二版校对。\n"})
    client.post(f"/api/v1/documents/{doc_id}/edit-backups/{backups[0]['name']}/restore")
    after = client.get(f"/api/v1/documents/{doc_id}/edit-backups").json()
    assert len(after) == 2 and any("第二版校对" in b["preview"] for b in after) and EDIT in _doc(client, doc_id)[1]

    assert client.post(f"/api/v1/documents/{doc_id}/edit-backups/other.edited-20260101-000000.md/restore").status_code == 400
    assert client.post(f"/api/v1/documents/{doc_id}/edit-backups/{doc_id}.edited-20000101-000000.md/restore").status_code == 404
    assert client.delete(f"/api/v1/documents/{doc_id}/edit-backups/{after[-1]['name']}").json()["ok"]
    assert len(client.get(f"/api/v1/documents/{doc_id}/edit-backups").json()) == 1


def test_document_changes_are_recorded_in_lake(client):
    """上传 / 校对 / 保留校对重建 / 删除都在 Agent 湖留下 doc_event（后台写入，轮询等待）；文档快照跟着最新内容更新。"""
    from app.services.lake_service import lake_service

    kb = client.post("/api/v1/knowledge-bases/", json={"name": "变更记录", "description": "doc_event 回归"}).json()
    doc_id = client.post(f"/api/v1/documents/upload?kbId={kb['id']}", files={"file": ("休假制度.md", io.BytesIO(SOURCE.encode()), "text/markdown")}).json()["id"]
    md = _doc(client, doc_id)[1]
    client.put(f"/api/v1/documents/{doc_id}/markdown", json={"markdownContent": md.rstrip() + "\n\n" + EDIT + "\n"})
    client.post(f"/api/v1/documents/{doc_id}/reprocess")
    client.delete(f"/api/v1/documents/{doc_id}")

    assert lake_service.flush()
    lake = lake_service.lake()
    events = lake.list(where=lake.where_clause(kinds=["doc_event"], parent_ids=[f"doc-{doc_id}"]), limit=20)[0]
    import json as _json

    by_event = {_json.loads(e["meta"])["event"]: e for e in events}
    assert set(by_event) == {"upload", "edit", "reprocess_keep_edits", "delete"}, set(by_event)
    edit = by_event["edit"]
    assert edit["kb_id"] == kb["id"] and "校对回灌，+1 行 / -0 行" in edit["content"] and "跨年使用" in edit["content"]
    assert "正文没有变化" in by_event["reprocess_keep_edits"]["content"]
    snap = lake.get(f"doc-{doc_id}")
    assert snap["status"] == "deleted" and "跨年使用" in snap["content"]          # 快照是最后一次的校对版内容，删除后标记 deleted
