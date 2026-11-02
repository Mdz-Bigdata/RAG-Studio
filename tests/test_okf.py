"""OKF（开放知识格式）：知识包扫描 → 条目入库（带 okf 元数据、OKF 标签）→ 问题点名条目时确定性置顶 → Git 同步（新增 / 更新 / 删除）。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from prism.config import Settings
from prism.ingest import IngestPipeline
from prism.okf import OKFCatalog, okf_markdown, scan_bundle
from prism.retrieval import HybridRetriever
from prism.schemas import DocMetadata

ROOT = Path(__file__).resolve().parents[1]

CHURN = """---
type: metric
title: "Monthly Churn Rate"
description: "Official formula for calculating monthly customer churn."
owner: "data-engineering"
tags: [revenue, kpi]
timestamp: 2026-06-20T10:00:00Z
aliases: [月流失率, 客户流失率]
---
# Monthly Churn Rate

The official churn rate formula used in all board reports:

Churn Rate = (Customers Lost During Month / Customers at Start of Month) × 100

### Rules
- **Do NOT** use trial accounts in the denominator.
- Source of truth: `analytics.monthly_churn_summary` table.

### Related
- [Monthly Active Users](mau.md)
"""
MAU = """---
type: metric
title: 月活跃用户
owner: data-engineering
---
# 月活跃用户

月活跃用户（MAU）= 当月至少登录一次的去重账号数，不含内部测试账号。
"""
PLAIN = "# 没有 front matter 的普通说明\n\n这不是条目。\n"


def write_bundle(root: Path) -> Path:
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "metrics" / "monthly_churn.md").write_text(CHURN, encoding="utf-8")
    (root / "metrics" / "mau.md").write_text(MAU, encoding="utf-8")
    (root / "README.md").write_text(PLAIN, encoding="utf-8")
    return root


# --------------------------------------------------------------------------------------
def test_scan_bundle_and_markdown_roundtrip(tmp_path: Path):
    entries = scan_bundle(write_bundle(tmp_path))
    assert [e.title for e in entries] == ["月活跃用户", "Monthly Churn Rate"]          # README 没有 front matter，不算条目
    churn = entries[1]
    assert churn.type == "metric" and churn.owner == "data-engineering" and churn.timestamp.startswith("2026-06-20")
    assert churn.aliases == ["月流失率", "客户流失率", "monthly churn"] and churn.related == ["mau"]   # 文件名也算一个别名
    assert churn.path == "metrics/monthly_churn.md" and churn.tags == ["revenue", "kpi"]
    # 入库用的 Markdown：front matter 补 okf 段与 OKF 标签，正文不变；现有 Markdown 解析器读得出来
    from prism.ingest.parsers.markdown_parser import MarkdownParser

    doc = MarkdownParser().parse_text(okf_markdown(churn, "repo"), tmp_path / "x.md")
    assert doc.metadata.title == "Monthly Churn Rate" and sorted(doc.metadata.tags) == ["OKF", "kpi", "revenue"]
    assert doc.metadata.extra["okf"]["owner"] == "data-engineering" and doc.metadata.extra["okf"]["aliases"] == churn.aliases
    assert doc.metadata.extra["okf"]["path"] == "metrics/monthly_churn.md" and "× 100" in doc.markdown
    assert scan_bundle(tmp_path, "metrics")[0].path == "mau.md"                   # 子目录


def test_catalog_matching():
    docs = {
        "a": DocMetadata(title="Monthly Churn Rate", extra={"okf": {"aliases": ["月流失率", "客户流失率", "churn"]}}),
        "b": DocMetadata(title="月活跃用户", extra={"okf": {"aliases": ["MAU"]}}),
        "c": DocMetadata(title="普通文档"),                                          # 不是条目
    }
    cat = OKFCatalog(docs)
    assert len(cat) == 2
    assert [m.doc_id for m in cat.match("我们的月流失率是怎么算的？")] == ["a"]
    assert cat.match("客户流失率公式")[0].matched == "客户流失率"
    assert [m.doc_id for m in cat.match("how is the monthly churn rate defined")] == ["a"]
    assert [m.doc_id for m in cat.match("MAU 口径")] == ["b"] and [m.doc_id for m in cat.match("mau 口径")] == ["b"]
    assert cat.match("churned users last week") == []                              # 拉丁名整词匹配：churned 不算
    assert cat.match("普通文档怎么写") == [] and cat.match("") == []
    both = cat.match("月流失率和月活跃用户的区别")
    assert [m.doc_id for m in both] == ["b", "a"] or [m.doc_id for m in both] == ["a", "b"]    # 两个都命中（名字一样长时按顺序）
    assert len(cat.match("月流失率和月活跃用户", limit=1)) == 1
    assert cat.match("月流失率", allowed_doc=lambda d: d != "a") == []              # 权限 / 时效过滤后的文档不置顶


@pytest.fixture(scope="module")
def okf_index(tmp_path_factory):
    """OKF 条目 + 普通语料（员工手册等）一起入库。"""
    root = tmp_path_factory.mktemp("okf")
    bundle = write_bundle(root / "bundle")
    upload = root / "upload"
    upload.mkdir()
    for e in scan_bundle(bundle):
        (upload / Path(e.path).name).write_text(okf_markdown(e, "test"), encoding="utf-8")
    st = Settings.lite(index_dir=root / "index")
    p = IngestPipeline(st)
    report = p.ingest_paths([upload, ROOT / "data" / "raw" / "员工手册_2026版.md"])
    assert not report.failed, report.failed
    return p.index


def test_retrieval_pins_named_entry(okf_index):
    r = HybridRetriever(okf_index.settings, okf_index)
    res = r.retrieve("月流失率怎么算？试用账号要算进去吗", top_k=3)
    top = res.contexts[0]
    assert top.authority == "okf" and top.score == 1.0 and "× 100" in top.chunk.text
    assert top.chunk.metadata["okf"]["owner"] == "data-engineering"
    assert res.trace.okf and res.trace.okf[0]["title"] == "Monthly Churn Rate" and res.trace.okf[0]["matched"] == "月流失率"
    assert any("OKF 权威条目命中" in n for n in res.trace.notes)
    assert len({c.chunk.chunk_id for c in res.contexts}) == len(res.contexts)       # 置顶的不和 RAG 结果重复
    # 交给模型的资料里标出权威条目
    from prism.generation.prompts import format_context

    text = format_context(top)
    assert "【权威条目（OKF）】" in text and "负责人：data-engineering" in text and "更新：2026-06-20" in text
    # 条目正文 Related 链到的《月活跃用户》一并置顶（占 max_pinned 名额），链路说明写明
    assert [x["title"] for x in res.trace.okf] == ["Monthly Churn Rate", "月活跃用户"] and res.trace.okf[1]["matched"] == "related:Monthly Churn Rate"
    assert any("一并带上它链接的 《月活跃用户》" in n for n in res.trace.notes)
    assert all(c.authority == "okf" for c in res.contexts if c.chunk.doc_id in {x["doc_id"] for x in res.trace.okf})
    only_one = r.retrieve("月流失率怎么算", top_k=3, overrides={"okf": {"max_pinned": 1}})
    assert [x["title"] for x in only_one.trace.okf] == ["Monthly Churn Rate"]
    # 没点名条目：不置顶，照常 RAG
    plain = r.retrieve("年假可以跨年使用吗", top_k=3)
    assert plain.contexts and all(c.authority is None for c in plain.contexts) and not plain.trace.okf
    # 没点名但 RAG 召回到了条目的段落：同样标为权威（权威是来源的属性），但不置顶、不进 trace.okf
    recalled = r.retrieve("trial accounts denominator board reports", top_k=3)
    hit = [c for c in recalled.contexts if "trial accounts" in c.chunk.text]
    assert hit and hit[0].authority == "okf" and hit[0].chunk.metadata["okf"]["owner"] == "data-engineering" and not recalled.trace.okf
    # 关闭开关
    off = r.retrieve("月流失率怎么算", top_k=3, overrides={"okf": {"enabled": False}})
    assert all(c.authority is None for c in off.contexts)


# --------------------------------------------------------------------------------------
def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="需要 git")
def test_import_from_git_repo_sync_and_remove(tmp_path: Path):
    from fastapi.testclient import TestClient

    from app import config as app_config_mod

    cfg = app_config_mod.app_config
    saved = (cfg.data_dir, cfg.upload_dir, cfg.index_root)
    cfg.data_dir, cfg.upload_dir, cfg.index_root = tmp_path / "data", tmp_path / "data" / "uploads", tmp_path / "data" / "indexes"
    for d in (cfg.upload_dir, cfg.index_root):
        d.mkdir(parents=True, exist_ok=True)
    repo = write_bundle(tmp_path / "repo")
    _git(["init", "-q", "-b", "main"], repo)
    _git(["-c", "user.email=t@t", "-c", "user.name=t", "add", "."], repo)
    _git(["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"], repo)
    from app.main import app

    try:
        with TestClient(app) as c:
            kb = c.post("/api/v1/knowledge-bases/", json={"name": "OKF 库"}).json()["id"]
            r = c.post(f"/api/v1/okf/{kb}/import", json={"source": str(repo) + "/.git", "ref": "main"})
            assert r.status_code == 200, r.text
            first = r.json()
            assert (first["created"], first["updated"], first["unchanged"], first["total"]) == (2, 0, 0, 2) and first["failed"] == [] and first["commit"]
            docs = c.get(f"/api/v1/documents/?kbId={kb}").json()
            assert len(docs) == 2 and all(d["status"] == "COMPLETED" and "OKF" in d["tags"] for d in docs)
            st = c.get(f"/api/v1/okf/{kb}").json()
            assert st["count"] == 2 and st["isGit"] and st["ref"] == "main" and {e["title"] for e in st["entries"]} == {"Monthly Churn Rate", "月活跃用户"}
            assert c.get(f"/api/v1/okf/{kb}/match?q=月流失率怎么算").json()[0]["title"] == "Monthly Churn Rate"
            # 检索实验室：条目置顶并标注
            res = c.post(f"/api/v1/retrieval/{kb}/playground", json={"query": "月流失率怎么算"}).json()
            assert res["contexts"][0]["authority"] == "okf" and res["okf"][0]["matched"] == "月流失率"
            # 再同步：什么都没变
            again = c.post(f"/api/v1/okf/{kb}/sync").json()
            assert (again["created"], again["updated"], again["unchanged"], again["removed"]) == (0, 0, 2, 0)
            # 仓库里改一条、删一条，提交后同步
            (repo / "metrics" / "mau.md").write_text(MAU.replace("不含内部测试账号", "不含内部测试账号与机器人账号"), encoding="utf-8")
            (repo / "metrics" / "monthly_churn.md").unlink()
            _git(["-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"], repo)
            _git(["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "update"], repo)
            third = c.post(f"/api/v1/okf/{kb}/sync").json()
            assert (third["created"], third["updated"], third["unchanged"], third["removed"]) == (0, 1, 0, 1)
            docs = c.get(f"/api/v1/documents/?kbId={kb}").json()
            assert len(docs) == 1 and docs[0]["title"] == "月活跃用户"
            assert c.get(f"/api/v1/okf/{kb}/match?q=月流失率").json() == []
            res = c.post(f"/api/v1/retrieval/{kb}/playground", json={"query": "MAU 口径是什么"}).json()
            assert res["contexts"][0]["authority"] == "okf" and "机器人账号" in res["contexts"][0]["text"]
            # 自动同步：远端提交没变就跳过；仓库再提交一次后一轮同步就重新导入
            from app.services.okf_service import okf_service

            assert c.get("/api/v1/okf/auto-sync").json()["minutes"] == 0
            assert c.put("/api/v1/okf/auto-sync", json={"minutes": 15}).json()["minutes"] == 15
            round1 = c.post("/api/v1/okf/auto-sync/run").json()
            assert (round1["kbs"], round1["checked"], round1["unchanged"], round1["updated"]) == (1, 1, 1, 0)
            # 登记表丢了（加登记表之前导入的知识库）：按磁盘上的清单照样轮询
            from app.db import Repository, SessionLocal

            with SessionLocal() as db:
                Repository(db).set_setting("okf:kbs", [])
            assert c.post("/api/v1/okf/auto-sync/run").json()["checked"] == 1
            (repo / "metrics" / "mau.md").write_text(MAU.replace("去重账号数", "去重账号数（自动同步）"), encoding="utf-8")
            _git(["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-am", "auto"], repo)
            assert okf_service.remote_commit(str(repo) + "/.git", "main") == _git(["rev-parse", "HEAD"], repo).strip()
            round2 = c.post("/api/v1/okf/auto-sync/run").json()
            assert (round2["checked"], round2["updated"], round2["errors"]) == (1, 1, [])
            assert "自动同步" in c.post(f"/api/v1/retrieval/{kb}/playground", json={"query": "MAU 口径是什么"}).json()["contexts"][0]["text"]
            c.put("/api/v1/okf/auto-sync", json={"minutes": 0})
            # 取消关联：条目文档删除
            assert c.delete(f"/api/v1/okf/{kb}").json()["removed"] == 1
            assert c.get(f"/api/v1/documents/?kbId={kb}").json() == [] and c.get(f"/api/v1/okf/{kb}").json()["count"] == 0
            assert c.post("/api/v1/okf/auto-sync/run").json()["kbs"] == 0            # 取消关联后不再轮询它
            # 坏来源
            assert c.post(f"/api/v1/okf/{kb}/import", json={"source": str(tmp_path / "nope")}).status_code == 400
    finally:
        cfg.data_dir, cfg.upload_dir, cfg.index_root = saved
