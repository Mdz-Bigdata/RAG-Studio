"""飞书文档接入：链接解析、块 ⇄ Markdown、客户端鉴权 / 重试、同步进知识库、自动同步 / 事件订阅、远程增删改查、文档智能体。

飞书开放平台用内存里的假服务（httpx.MockTransport）模拟，不碰网络。
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
import uuid
from typing import Any

import httpx
import pytest

from prism.feishu import FeishuAgent, FeishuClient, FeishuError, Tool, blocks_to_html, blocks_to_markdown, markdown_to_blocks, parse_ref
from prism.feishu.convert import md_inline
from prism.generation.llm import FakeLLM


# ======================================================================================
# 假的飞书开放平台
# ======================================================================================
class FakeFeishu:
    """docs: document_id → {title, revision, blocks: {block_id: block}}；根块 id = document_id。"""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.wiki: dict[str, tuple[str, str]] = {}
        self.calls: list[tuple[str, str]] = []
        self.tokens_issued = 0
        self.expire_next = False          # 下一次业务请求返回 token 过期
        self.rate_limit_next = 0          # 接下来 N 次业务请求返回限流
        self.convert_ok = False           # Markdown 转换接口是否可用
        self.forbidden: set[str] = set()
        self.subscribed: set[str] = set()
        self.media: dict[str, bytes] = {}
        self.missing_scopes: set[str] = set()   # 路径前缀：模拟应用没开通对应权限
        self.shared: dict[str, str] = {}         # document_id → 链接分享范围

    # ---- 造数据
    def add_doc(self, title: str, children: list[dict[str, Any]], document_id: str | None = None) -> str:
        did = document_id or ("doxcn" + uuid.uuid4().hex[:18])
        root = {"block_id": did, "block_type": 1, "page": {"elements": [{"text_run": {"content": title}}]}, "children": []}
        self.docs[did] = {"title": title, "revision": 1, "blocks": {did: root}}
        self._insert(did, did, children, -1)
        return did

    def _insert(self, did: str, parent: str, children: list[dict[str, Any]], index: int) -> list[str]:
        doc = self.docs[did]
        ids = []
        for ch in children:
            ch = json.loads(json.dumps(ch))
            nested = ch.pop("_children", [])
            bid = ch.get("block_id") or "blk" + uuid.uuid4().hex[:12]
            ch.update(block_id=bid, parent_id=parent, children=[])
            doc["blocks"][bid] = ch
            ids.append(bid)
            if nested:
                self._insert(did, bid, nested, -1)
        kids = doc["blocks"][parent]["children"]
        if index < 0 or index >= len(kids):
            kids.extend(ids)
        else:
            kids[index:index] = ids
        doc["revision"] += 1
        return ids

    def markdown(self, did: str) -> str:
        return blocks_to_markdown(list(self.docs[did]["blocks"].values()))

    # ---- 路由
    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/open-apis")
        method = request.method
        self.calls.append((method, path))
        body = json.loads(request.content or b"{}") if request.content else {}
        ok = lambda data=None: httpx.Response(200, json={"code": 0, "msg": "success", "data": data or {}})   # noqa: E731
        if path == "/auth/v3/tenant_access_token/internal":
            if body.get("app_secret") != "secret":
                return httpx.Response(200, json={"code": 10014, "msg": "app secret invalid"})
            self.tokens_issued += 1
            return httpx.Response(200, json={"code": 0, "msg": "ok", "tenant_access_token": f"t-{self.tokens_issued}", "expire": 7200})
        if not request.headers.get("Authorization", "").startswith("Bearer t-"):
            return httpx.Response(401, json={"code": 99991661, "msg": "missing token"})
        if self.expire_next:
            self.expire_next = False
            return httpx.Response(400, json={"code": 99991663, "msg": "token expired"})
        if self.rate_limit_next:
            self.rate_limit_next -= 1
            return httpx.Response(429, json={"code": 99991400, "msg": "too many requests"})
        if any(path.startswith(prefix) for prefix in self.missing_scopes):
            return httpx.Response(400, json={"code": 99991672, "msg": "Access denied. One of the following scopes is required"})

        if path == "/wiki/v2/spaces/get_node":
            obj = self.wiki.get(request.url.params["token"])
            return ok({"node": {"obj_token": obj[0], "obj_type": obj[1], "title": "wiki"}}) if obj else httpx.Response(400, json={"code": 131005, "msg": "not found"})
        if path == "/drive/v1/files" and method == "GET":
            return ok({"files": [{"token": d, "name": v["title"], "type": "docx", "url": f"https://x.feishu.cn/docx/{d}"} for d, v in self.docs.items()], "has_more": False})
        m = re.fullmatch(r"/drive/v1/permissions/(\w+)/public", path)
        if m and method == "PATCH":
            self.shared[m.group(1)] = body["link_share_entity"]
            return ok({"permission_public": body})
        m = re.fullmatch(r"/drive/v1/medias/(\w+)/download", path)
        if m and method == "GET":
            data = self.media.get(m.group(1))
            if data is None:
                return httpx.Response(400, json={"code": 1061045, "msg": "file not found"})
            return httpx.Response(200, content=data, headers={"content-type": "image/png"})
        m = re.fullmatch(r"/drive/v1/files/(\w+)/subscribe", path)
        if m and method == "POST":
            if m.group(1) not in self.docs:
                return httpx.Response(404, json={"code": 1061007, "msg": "file not found"})
            self.subscribed.add(m.group(1))
            return ok({})
        m = re.fullmatch(r"/drive/v1/files/(\w+)", path)
        if m and method == "DELETE":
            self.docs.pop(m.group(1), None)
            return ok({})
        if path == "/docx/v1/documents" and method == "POST":
            did = self.add_doc(body["title"], [])
            return ok({"document": {"document_id": did, "revision_id": 1, "title": body["title"]}})
        if path == "/docx/v1/documents/blocks/convert":
            if not self.convert_ok:
                return httpx.Response(400, json={"code": 99991672, "msg": "no permission"})
            blocks = markdown_to_blocks(body["content"])
            for i, b in enumerate(blocks):
                b["block_id"] = f"tmp{i}"
            return ok({"first_level_block_ids": [b["block_id"] for b in blocks], "blocks": blocks})
        m = re.fullmatch(r"/docx/v1/documents/(\w+)(/.*)?", path)
        if not m:
            return httpx.Response(404, json={"code": 404, "msg": "no route"})
        did, rest = m.group(1), m.group(2) or ""
        if did in self.forbidden:
            return httpx.Response(403, json={"code": 1770032, "msg": "forbidden"})
        if did not in self.docs:
            return httpx.Response(404, json={"code": 1770002, "msg": "not found"})
        doc = self.docs[did]
        if rest == "" and method == "GET":
            return ok({"document": {"document_id": did, "revision_id": doc["revision"], "title": doc["title"]}})
        if rest == "/blocks" and method == "GET":
            return ok({"items": list(doc["blocks"].values()), "has_more": False})
        m2 = re.fullmatch(r"/blocks/(\w+)/children", rest)
        if m2 and method == "POST":
            assert len(body["children"]) <= 50
            ids = self._insert(did, m2.group(1), body["children"], body.get("index", -1))
            return ok({"children": [doc["blocks"][i] for i in ids]})
        m2 = re.fullmatch(r"/blocks/(\w+)/descendant", rest)
        if m2 and method == "POST":
            by_id = {b["block_id"]: b for b in body["descendants"]}
            self._insert(did, m2.group(1), [{k: v for k, v in by_id[i].items() if k != "block_id"} for i in body["children_id"]], body.get("index", -1))
            return ok({})
        m2 = re.fullmatch(r"/blocks/(\w+)/children/batch_delete", rest)
        if m2 and method == "DELETE":
            kids = doc["blocks"][m2.group(1)]["children"]
            gone = kids[body["start_index"]:body["end_index"]]
            del kids[body["start_index"]:body["end_index"]]
            for g in gone:
                doc["blocks"].pop(g, None)
            doc["revision"] += 1
            return ok({})
        m2 = re.fullmatch(r"/blocks/(\w+)", rest)
        if m2 and method == "PATCH":
            b = doc["blocks"][m2.group(1)]
            field = next(k for k in b if isinstance(b[k], dict) and "elements" in b[k])
            b[field]["elements"] = body["update_text_elements"]["elements"]
            doc["revision"] += 1
            return ok({"block": b})
        return httpx.Response(404, json={"code": 404, "msg": f"no route {method} {path}"})


def _t(text: str, **style: Any) -> dict[str, Any]:
    return {"text_run": {"content": text, **({"text_element_style": style} if style else {})}}


SAMPLE = [
    {"block_type": 3, "heading1": {"elements": [_t("报销制度")]}},
    {"block_type": 2, "text": {"elements": [_t("差旅报销须在 "), _t("30 天", bold=True), _t(" 内提交，详见"), _t("财务系统", link={"url": "https%3A%2F%2Ffin.example.com%2Fa%3Fb%3D1"})]}},
    {"block_type": 4, "heading2": {"elements": [_t("标准")]}},
    {"block_type": 12, "bullet": {"elements": [_t("一线城市住宿 600 元/晚")]}, "_children": [{"block_type": 12, "bullet": {"elements": [_t("含早餐")]}}]},
    {"block_type": 12, "bullet": {"elements": [_t("其他城市住宿 400 元/晚")]}},
    {"block_type": 13, "ordered": {"elements": [_t("填写报销单")]}},
    {"block_type": 13, "ordered": {"elements": [_t("主管审批")]}},
    {"block_type": 17, "todo": {"elements": [_t("上传发票")], "style": {"done": True}}},
    {"block_type": 14, "code": {"elements": [_t("SELECT 1;")], "style": {"language": 56}}},
    {"block_type": 27, "image": {"token": "img1"}},
    {"block_type": 31, "table": {"property": {"row_size": 2, "column_size": 2}, "cells": ["c1", "c2", "c3", "c4"]},
     "_children": [{"block_id": f"c{i}", "block_type": 32, "table_cell": {}, "_children": [{"block_type": 2, "text": {"elements": [_t(v)]}}]}
                   for i, v in zip(range(1, 5), ["级别", "额度", "经理", "800"])]},
]


# ======================================================================================
# 纯函数
# ======================================================================================
def test_parse_ref():
    assert parse_ref("https://abc.feishu.cn/docx/doxcnAbCdEf123456?from=wiki").__dict__ == {"kind": "docx", "token": "doxcnAbCdEf123456"}
    assert parse_ref("https://abc.feishu.cn/wiki/wikcnXyZ1234567890").kind == "wiki"
    assert parse_ref("https://abc.larksuite.com/drive/folder/fldcnQwErTy12345").kind == "folder"
    assert parse_ref("https://abc.feishu.cn/docs/doccnOldOld123456").kind == "doc"
    assert parse_ref("doxcnAbCdEf123456").kind == "docx"
    with pytest.raises(ValueError):
        parse_ref("https://example.com/hello")


def test_blocks_to_markdown_keeps_structure():
    fake = FakeFeishu()
    did = fake.add_doc("差旅制度", SAMPLE)
    md = fake.markdown(did)
    assert md.startswith("# 差旅制度\n")
    assert "# 报销制度" in md and "## 标准" in md
    assert "**30 天**" in md and "[财务系统](https://fin.example.com/a?b=1)" in md
    assert "- 一线城市住宿 600 元/晚\n  - 含早餐" in md
    assert "1. 填写报销单\n" in md and "2. 主管审批" in md
    assert "- [x] 上传发票" in md
    assert "```sql\nSELECT 1;\n```" in md
    assert "[图片]" in md
    assert "| 级别 | 额度 |" in md and "| 经理 | 800 |" in md


def test_blocks_to_html_keeps_feishu_layout():
    fake = FakeFeishu()
    blocks = SAMPLE + [
        {"block_type": 12, "bullet": {"elements": [_t("跨页表格合并", bold=True), _t("："), _t("宽度差异<10%", bold=True, text_color=1), _t("底色", background_color=3)]}},
        {"block_type": 2, "text": {"elements": [_t("<script>alert(1)</script>"), _t("坏链接", link={"url": "javascript%3Aalert(1)"})], "style": {"align": 2}}},
        {"block_type": 13, "ordered": {"elements": [_t("从 5 开始")], "style": {"sequence": "5"}}},
        {"block_type": 13, "ordered": {"elements": [_t("续号")], "style": {"sequence": "auto"}}},
        {"block_type": 19, "callout": {"background_color": 5, "border_color": 5, "emoji_id": "bulb"}, "_children": [{"block_type": 2, "text": {"elements": [_t("提示")]}}]},
    ]
    did = fake.add_doc("差旅制度", blocks)
    html = blocks_to_html(list(fake.docs[did]["blocks"].values()), image_src=lambda t: f"/api/v1/feishu/media/{t}")
    assert html.startswith('<div class="fs-doc"><h1 class="fs-title">差旅制度</h1>')
    assert '<div class="fs-h fs-h1">报销制度</div>' in html and '<div class="fs-h fs-h2">标准</div>' in html
    # 粗体紧挨中文标点也照样是粗体（Markdown 会把 **粗体：**正文 显示成字面量）；颜色 / 底色保留
    assert "<strong>跨页表格合并</strong>：" in html
    assert '<span style="color:#D83931"><strong>宽度差异&lt;10%</strong></span>' in html and '<span style="background:#FFF6C2">底色</span>' in html
    # 列表：嵌套、编号（sequence 起始值与续号）、待办
    assert '<span class="fs-marker">•</span><div class="fs-li-body">一线城市住宿 600 元/晚</div></div><div class="fs-nested"><div class="fs-li"><span class="fs-marker">◦</span>' in html
    assert '<span class="fs-marker">1.</span><div class="fs-li-body">填写报销单' in html and '<span class="fs-marker">2.</span><div class="fs-li-body">主管审批' in html
    assert '<span class="fs-marker">5.</span><div class="fs-li-body">从 5 开始' in html and '<span class="fs-marker">6.</span><div class="fs-li-body">续号' in html
    assert "fs-todo-done" in html and "☑" in html
    # 图片走代理地址；表格、代码、高亮块
    assert '<img src="/api/v1/feishu/media/img1"' in html
    assert '<table class="fs-table"><tbody><tr><td><p class="fs-p">级别</p></td><td><p class="fs-p">额度</p></td></tr>' in html
    assert '<div class="fs-code-lang">sql</div><pre><code>SELECT 1;</code></pre>' in html
    assert '<div class="fs-callout" style="background:#E1EAFF;border-color:#82A7FC"><span class="fs-callout-emoji">💡</span>' in html
    # 安全：文字转义、只保留 http(s) 链接
    assert "<script>" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "javascript:" not in html and '<p class="fs-p" style="text-align:center">' in html
    assert '<a href="https://fin.example.com/a?b=1" target="_blank" rel="noopener noreferrer">财务系统</a>' in html


def test_blocks_to_html_table_widths_and_merges():
    cells = [{"block_id": f"m{i}", "block_type": 32, "table_cell": {}, "_children": [{"block_type": 2, "text": {"elements": [_t(f"格{i}")]}}]} for i in range(6)]
    fake = FakeFeishu()
    did = fake.add_doc("合并", [{"block_type": 31, "table": {"property": {"row_size": 2, "column_size": 3, "column_width": [100, 200, 300], "header_row": True,
                                                                          "merge_info": [{"row_span": 1, "col_span": 2}] + [{"row_span": 1, "col_span": 1}] * 5},
                                                             "cells": [f"m{i}" for i in range(6)]}, "_children": cells}])
    html = blocks_to_html(list(fake.docs[did]["blocks"].values()))
    assert '<table class="fs-table" style="width:600px"><colgroup><col style="width:100px"><col style="width:200px"><col style="width:300px"></colgroup>' in html
    assert '<th colspan="2"><p class="fs-p">格0</p></th><th><p class="fs-p">格2</p></th></tr>' in html     # 被合并的格1不再单独出现
    assert "格1" not in html and "<td><p class=\"fs-p\">格3</p></td>" in html


def test_markdown_to_blocks():
    md = "---\ntitle: x\n---\n# 标题\n\n第一段 **加粗** 与 `code` 和 [链接](https://a.com/x?y=1)\n续行\n\n- 甲\n- [ ] 待办\n1. 一\n> 引用\n\n```python\nprint(1)\n```\n\n---\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    blocks = markdown_to_blocks(md)
    types = [b["block_type"] for b in blocks]
    assert types == [3, 2, 12, 17, 13, 15, 14, 22, 2, 2]
    para = blocks[1]["text"]["elements"]
    assert [e["text_run"]["content"] for e in para] == ["第一段 ", "加粗", " 与 ", "code", " 和 ", "链接", "\n续行"]     # 段内换行保留
    assert para[1]["text_run"]["text_element_style"] == {"bold": True}
    assert para[5]["text_run"]["text_element_style"]["link"]["url"] == "https%3A%2F%2Fa.com%2Fx%3Fy%3D1"
    assert blocks[6]["code"]["style"]["language"] == 49
    assert md_inline("~~删~~ *斜*")[0]["text_run"]["text_element_style"] == {"strikethrough": True}


def test_markdown_round_trip_through_blocks():
    fake = FakeFeishu()
    src = "## 流程\n\n1. 提交\n2. 审批\n\n- **注意**：发票必须是原件\n\n```bash\nls -la\n```\n"
    did = fake.add_doc("rt", markdown_to_blocks(src))
    back = fake.markdown(did)
    for piece in ("## 流程", "1. 提交", "2. 审批", "- **注意**：发票必须是原件", "```bash\nls -la\n```"):
        assert piece in back


# ======================================================================================
# 客户端
# ======================================================================================
def _client(fake: FakeFeishu, secret: str = "secret") -> FeishuClient:
    return FeishuClient("cli_test", secret, transport=httpx.MockTransport(fake.handler))


def test_client_token_refresh_and_retry(monkeypatch):
    monkeypatch.setattr("prism.feishu.client.time.sleep", lambda s: None)
    fake = FakeFeishu()
    did = fake.add_doc("A", SAMPLE[:2])
    c = _client(fake)
    assert c.get_document(did)["title"] == "A"
    assert fake.tokens_issued == 1
    fake.expire_next = True
    assert c.get_document(did)["revision_id"] >= 1
    assert fake.tokens_issued == 2                    # 过期后刷新了一次
    fake.rate_limit_next = 2
    assert len(c.list_blocks(did)) == 3               # 限流退避后成功
    with pytest.raises(FeishuError) as e:
        c.get_document("doxcnNotExist0001")
    assert e.value.code == 1770002 and "不存在" in str(e.value)
    fake.forbidden.add(did)
    with pytest.raises(FeishuError, match="添加文档应用"):
        c.get_document(did)
    with pytest.raises(FeishuError, match="tenant_access_token"):
        _client(fake, "wrong").get_document(did)


def test_client_append_batches_over_50():
    fake = FakeFeishu()
    did = fake.add_doc("big", [])
    c = _client(fake)
    n = c.append_children(did, markdown_to_blocks("\n\n".join(f"第 {i} 段" for i in range(120))))
    assert n == 120
    assert len(fake.docs[did]["blocks"][did]["children"]) == 120
    assert sum(1 for m, p in fake.calls if m == "POST" and p.endswith("/children")) == 3


# ======================================================================================
# 服务与接口
# ======================================================================================
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("feishu-app")
    from fastapi.testclient import TestClient

    from app import config as app_config_mod

    cfg = app_config_mod.app_config
    saved = (cfg.data_dir, cfg.upload_dir, cfg.index_root)
    cfg.data_dir, cfg.upload_dir, cfg.index_root = data_dir, data_dir / "uploads", data_dir / "indexes"
    for d in (cfg.upload_dir, cfg.index_root):
        d.mkdir(parents=True, exist_ok=True)

    from app.main import app
    from app.services.feishu_service import feishu_service

    fake = FakeFeishu()
    feishu_service.transport = httpx.MockTransport(fake.handler)
    with TestClient(app) as c:
        r = c.put("/api/v1/feishu/config", json={"app_id": "cli_test", "app_secret": "secret", "web_base": "https://acme.feishu.cn"})
        assert r.status_code == 200 and r.json()["app_secret"] == "********" and r.json()["configured"]
        kb = c.post("/api/v1/knowledge-bases/", json={"name": "飞书库", "description": "飞书文档"}).json()
        yield c, fake, kb["id"]
    feishu_service.transport = None
    cfg.data_dir, cfg.upload_dir, cfg.index_root = saved


def test_config_keeps_secret_when_masked(env):
    c, fake, _ = env
    r = c.put("/api/v1/feishu/config", json={"app_secret": "********", "folder_token": ""})
    assert r.status_code == 200
    assert c.post("/api/v1/feishu/test").json()["ok"] is True


def test_import_sync_and_resync(env):
    c, fake, kb_id = env
    did = fake.add_doc("差旅制度", SAMPLE)
    r = c.post("/api/v1/feishu/import", json={"kbId": kb_id, "refs": [f"https://acme.feishu.cn/docx/{did}", "https://example.com/nope"]})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res[0]["ok"] and res[0]["status"] == "created"
    assert not res[1]["ok"] and "无法识别" in res[1]["error"]
    links = c.get(f"/api/v1/feishu/docs?kbId={kb_id}").json()
    link = next(lk for lk in links if lk["documentId"] == did)
    assert link["status"] == "SYNCED" and link["localStatus"] == "COMPLETED" and link["chunkCount"] > 0
    doc = c.get(f"/api/v1/documents/{link['localDocId']}").json()
    assert doc["fileName"] == "差旅制度.md" and "飞书" in doc["tags"]
    md = c.get(f"/api/v1/documents/{link['localDocId']}/markdown").json()["markdownContent"]
    assert "一线城市住宿 600 元/晚" in md

    # 检索得到飞书内容
    hits = c.post(f"/api/v1/retrieval/{kb_id}/playground", json={"query": "一线城市住宿标准", "topK": 3})
    assert hits.status_code == 200 and "600 元/晚" in json.dumps(hits.json(), ensure_ascii=False)

    # 没改过：跳过
    assert c.post(f"/api/v1/feishu/docs/{link['id']}/sync").json()["status"] == "unchanged"
    # 飞书上改了：重新入库
    fake._insert(did, did, [{"block_type": 2, "text": {"elements": [_t("新增：出租车费实报实销")]}}], -1)
    assert c.post(f"/api/v1/feishu/docs/{link['id']}/sync").json()["status"] == "updated"
    md = c.get(f"/api/v1/documents/{link['localDocId']}/markdown").json()["markdownContent"]
    assert "出租车费实报实销" in md
    # 整库同步：都没变
    assert all(x["status"] == "unchanged" for x in c.post(f"/api/v1/feishu/sync?kbId={kb_id}").json())


def test_connection_reports_missing_scopes(env, describer):
    c, fake, _ = env
    r = c.post("/api/v1/feishu/test").json()
    assert r["ok"] and r["scopes"] == {"drive": True, "docx": True, "wiki": True, "media": True}
    fake.missing_scopes = {"/wiki/", "/drive/v1/medias/"}
    try:
        r = c.post("/api/v1/feishu/test").json()
        assert not r["ok"] and r["token"] and r["scopes"] == {"drive": True, "docx": True, "wiki": False, "media": False}
        assert "wiki:wiki:readonly" in r["error"] and "docs:document.media:download" in r["error"] and "docx:document" not in r["error"]
        did = _image_doc(fake, "缺权限的图片", "imgNoScope01")
        md = c.get("/api/v1/feishu/remote", params={"ref": did}).json()["markdown"]
        assert "[图片（没能读取：应用缺少下载图片的权限（docs:document.media:download 或 drive:drive））]" in md
    finally:
        fake.missing_scopes = set()
    # 文档链接前缀贴成完整链接也只取域名
    assert c.put("/api/v1/feishu/config", json={"web_base": "https://acme.feishu.cn/wiki/"}).json()["web_base"] == "https://acme.feishu.cn"


def test_wiki_and_unsupported(env):
    c, fake, kb_id = env
    did = fake.add_doc("Wiki 页面", SAMPLE[:2])
    fake.wiki["wikcnNode1234567890"] = (did, "docx")
    fake.wiki["wikcnSheet123456789"] = ("shtcnAAAAAAAAAAAA", "sheet")
    r = c.get("/api/v1/feishu/remote", params={"ref": "https://acme.feishu.cn/wiki/wikcnNode1234567890"})
    assert r.status_code == 200 and r.json()["documentId"] == did and "30 天" in r.json()["markdown"]
    r = c.get("/api/v1/feishu/remote", params={"ref": "https://acme.feishu.cn/wiki/wikcnSheet123456789"})
    assert r.status_code == 400 and "电子表格" in r.json()["detail"]
    fake.forbidden.add(did)
    r = c.post("/api/v1/feishu/import", json={"kbId": kb_id, "refs": [did]})
    assert r.json()[0]["ok"] is False and "添加文档应用" in r.json()[0]["error"]
    failed = next(lk for lk in c.get(f"/api/v1/feishu/docs?kbId={kb_id}").json() if lk["documentId"] == did)
    assert failed["status"] == "FAILED"
    fake.forbidden.discard(did)


def test_remote_crud(env):
    c, fake, kb_id = env
    # 增：新建并直接入库
    r = c.post("/api/v1/feishu/remote", json={"title": "周报", "markdown": "# 周报\n\n## 本周\n\n- 完成飞书接入\n", "kbId": kb_id})
    assert r.status_code == 200, r.text
    did = r.json()["documentId"]
    assert r.json()["url"] == f"https://acme.feishu.cn/docx/{did}"
    assert r.json()["sync"]["status"] == "created"
    md = fake.markdown(did)
    assert md.count("周报") == 1 and "- 完成飞书接入" in md      # 正文里重复的标题被去掉

    # 改：追加（自动重新同步）
    r = c.put("/api/v1/feishu/remote", json={"ref": did, "markdown": "## 下周\n\n- 上线"})
    assert r.status_code == 200 and r.json()["resynced"] == [kb_id]
    assert "## 下周" in fake.markdown(did)
    link = next(lk for lk in c.get(f"/api/v1/feishu/docs?kbId={kb_id}").json() if lk["documentId"] == did)
    assert "上线" in c.get(f"/api/v1/documents/{link['localDocId']}/markdown").json()["markdownContent"]

    # 改：定点改字，保留样式
    fake._insert(did, did, [{"block_type": 2, "text": {"elements": [_t("预算 "), _t("100 万", bold=True)]}}], -1)
    r = c.post("/api/v1/feishu/remote/replace-text", json={"ref": "周报", "find": "100 万", "replace": "120 万", "kbId": kb_id})
    assert r.status_code == 200 and r.json()["replaced"] == 1
    assert "**120 万**" in fake.markdown(did)

    # 改：整篇替换（带表格：转换接口不可用时按行写成文本）
    r = c.put("/api/v1/feishu/remote", json={"ref": did, "mode": "replace", "markdown": "# 周报\n\n全新内容\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"})
    assert r.status_code == 200 and r.json()["removedBlocks"] > 0
    md = fake.markdown(did)
    assert "全新内容" in md and "下周" not in md and "1 ｜ 2" in md
    fake.convert_ok = True
    c.put("/api/v1/feishu/remote", json={"ref": did, "markdown": "| x | y |\n|---|---|\n| 3 | 4 |\n"})
    assert any(p.endswith("/descendant") for _, p in fake.calls)

    # 查：文件夹
    files = c.get("/api/v1/feishu/folder").json()["files"]
    assert any(f["token"] == did and f["linkedKbs"] == [kb_id] for f in files)

    # 删：远程删除并从知识库移除
    local_id = link["localDocId"]
    r = c.delete("/api/v1/feishu/remote", params={"ref": did})
    assert r.status_code == 200 and r.json()["unlinked"]
    assert did not in fake.docs
    assert c.get(f"/api/v1/documents/{local_id}").status_code == 404
    assert not any(lk["documentId"] == did for lk in c.get(f"/api/v1/feishu/docs?kbId={kb_id}").json())


def test_unlink_keeps_remote(env):
    c, fake, kb_id = env
    did = fake.add_doc("保留远程", SAMPLE[:2])
    res = c.post("/api/v1/feishu/import", json={"kbId": kb_id, "refs": [did]}).json()[0]
    r = c.delete(f"/api/v1/feishu/docs/{res['linkId']}")
    assert r.status_code == 200 and r.json()["localDeleted"]
    assert did in fake.docs
    assert c.get(f"/api/v1/documents/{res['localDocId']}").status_code == 404


# ======================================================================================
# 图片：下载 + OCR / 看图描述
# ======================================================================================
def _png(color: str = "white") -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (320, 200), color).save(buf, format="PNG")
    return buf.getvalue()


class FakeDescriber:
    def __init__(self) -> None:
        self.calls = 0
        self.capabilities = {"ocr": "fake-ocr", "vlm": "fake-vlm"}
        self.cfg = type("Cfg", (), {"min_image_side": 48})()

    def describe(self, image, sha1, *, hint=""):
        self.calls += 1
        return "Warp Scheduler SM有两个指令发送单元", "Fermi 架构示意图：两个 Warp 调度器并发发射指令\n每组 16 个 CUDA core"


@pytest.fixture
def describer(monkeypatch):
    from app.services.feishu_service import FeishuService

    d = FakeDescriber()
    monkeypatch.setattr(FeishuService, "_describer", staticmethod(lambda: d))
    # 同步入库时图片索引也会 OCR：不加载本机 rapidocr（onnxruntime 的后台线程会在进程退出时偶发崩溃，其它测试同样不依赖本机 OCR）
    monkeypatch.setattr("prism.multimodal.describe.OCREngine.read", lambda self, image: "")
    return d


def _image_doc(fake: FakeFeishu, title: str, token: str) -> str:
    fake.media[token] = _png()
    return fake.add_doc(title, [
        {"block_type": 2, "text": {"elements": [_t("GPU 基本知识")]}},
        {"block_type": 27, "image": {"token": token, "width": 320, "height": 200}},
        {"block_type": 27, "image": {"token": token + "Missing"}},
    ])


def test_images_read_with_ocr_and_description(env, describer):
    c, fake, kb_id = env
    did = _image_doc(fake, "GPU基本知识", "imgFermi01")
    r = c.get("/api/v1/feishu/remote", params={"ref": did})
    assert r.status_code == 200, r.text
    md = r.json()["markdown"]
    assert "[图片]\n> 图片内容：Fermi 架构示意图：两个 Warp 调度器并发发射指令\n> 每组 16 个 CUDA core\n> 图中文字：Warp Scheduler SM有两个指令发送单元" in md
    assert "[图片（没能读取：" in md                                      # 下载失败的图只留说明，不影响整篇
    # 按图片 token 缓存：再读一次不重新下载、不重新识别
    downloads = sum(1 for _, p in fake.calls if p.startswith("/drive/v1/medias/imgFermi01/"))
    c.get("/api/v1/feishu/remote", params={"ref": did})
    assert sum(1 for _, p in fake.calls if p.startswith("/drive/v1/medias/imgFermi01/")) == downloads == 1
    assert describer.calls == 1
    # 智能体读到的也是带图片文字的全文
    from app.services.feishu_service import feishu_service

    assert "指令发送单元" in feishu_service.tools(kb_id)["read_doc"].fn({"document_id": did})["markdown"]


def test_images_synced_into_kb(env, describer):
    c, fake, kb_id = env
    from app import config as app_config_mod

    did = _image_doc(fake, "图片同步", "imgSync0001")
    res = c.post("/api/v1/feishu/import", json={"kbId": kb_id, "refs": [did]}).json()[0]
    assert res["ok"], res
    md = _local_md(c, kb_id, did)
    assert f"![图片](feishu-assets/{did}/imgSync0001.png)" in md and "图中文字：Warp Scheduler" in md
    assets = app_config_mod.app_config.upload_dir / kb_id / "feishu-assets" / did
    assert (assets / "imgSync0001.png").is_file()
    # 检索得到图片里的文字
    hits = c.post(f"/api/v1/retrieval/{kb_id}/playground", json={"query": "SM 有几个指令发送单元", "topK": 3})
    assert "指令发送单元" in json.dumps(hits.json(), ensure_ascii=False)
    # 移出知识库时图片目录一并清理
    assert c.delete(f"/api/v1/feishu/docs/{res['linkId']}").status_code == 200
    assert not assets.exists()


def test_view_mode_html_and_media_proxy(env, describer):
    c, fake, kb_id = env
    did = _image_doc(fake, "页面展示", "imgView0001")
    r = c.get("/api/v1/feishu/remote", params={"ref": did, "view": "true"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert '<img src="/api/v1/feishu/media/imgView0001"' in d["html"] and "GPU 基本知识" in d["html"]
    assert "图中文字" not in d["markdown"] and describer.calls == 0           # 展示不跑 OCR；markdown 是可编辑的原文
    img = c.get("/api/v1/feishu/media/imgView0001")
    assert img.status_code == 200 and img.content == fake.media["imgView0001"] and img.headers["content-type"] == "image/png"
    fake.media.pop("imgView0001")
    assert c.get("/api/v1/feishu/media/imgView0001").status_code == 200       # 已缓存：不再找飞书要
    assert c.get("/api/v1/feishu/media/imgNotThere01").status_code == 400
    assert c.get("/api/v1/feishu/media/..%2Fapp.db").status_code in (400, 404)


def test_images_off(env, describer):
    c, fake, kb_id = env
    did = _image_doc(fake, "不读图片", "imgOff00001")
    try:
        c.put("/api/v1/feishu/config", json={"read_images": False})
        md = c.get("/api/v1/feishu/remote", params={"ref": did}).json()["markdown"]
        assert "[图片]" in md and "图中文字" not in md and describer.calls == 0
    finally:
        c.put("/api/v1/feishu/config", json={"read_images": True})


# ======================================================================================
# 自动同步：定时 / 事件订阅
# ======================================================================================
def _edit(fake: FakeFeishu, did: str, text: str) -> None:
    fake._insert(did, did, [{"block_type": 2, "text": {"elements": [_t(text)]}}], -1)


def _local_md(c, kb_id: str, did: str) -> str:
    link = next(lk for lk in c.get(f"/api/v1/feishu/docs?kbId={kb_id}").json() if lk["documentId"] == did)
    return c.get(f"/api/v1/documents/{link['localDocId']}/markdown").json()["markdownContent"]


def test_auto_sync_round_covers_all_kbs(env):
    c, fake, kb_id = env
    kb2 = c.post("/api/v1/knowledge-bases/", json={"name": "飞书库二", "description": "第二个"}).json()["id"]
    d1 = fake.add_doc("自动同步一", [{"block_type": 2, "text": {"elements": [_t("初版内容甲")]}}])
    d2 = fake.add_doc("自动同步二", [{"block_type": 2, "text": {"elements": [_t("初版内容乙")]}}])
    assert c.post("/api/v1/feishu/import", json={"kbId": kb_id, "refs": [d1]}).json()[0]["ok"]
    assert c.post("/api/v1/feishu/import", json={"kbId": kb2, "refs": [d2]}).json()[0]["ok"]

    _edit(fake, d1, "甲：自动同步后的新内容")
    _edit(fake, d2, "乙：自动同步后的新内容")
    r = c.post("/api/v1/feishu/sync/auto")
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["kbs"] >= 2 and res["updated"] >= 2 and res["checked"] >= 2     # 前面测试里失败过的文档也会顺带补上
    assert "甲：自动同步后的新内容" in _local_md(c, kb_id, d1)
    assert "乙：自动同步后的新内容" in _local_md(c, kb2, d2)
    assert c.get("/api/v1/feishu/config").json()["autoSync"]["last"]["updated"] == res["updated"]
    # 再跑一轮：都没改，只检查不更新
    assert c.post("/api/v1/feishu/sync/auto").json()["updated"] == 0


def test_auto_sync_round_skips_when_busy_and_logs_errors(env, monkeypatch):
    c, fake, kb_id = env
    from app.services.feishu_service import feishu_service

    with feishu_service._sync_lock:                      # 已有一轮在跑：跳过
        assert feishu_service.auto_sync_round() is None
        assert c.post("/api/v1/feishu/sync/auto").json() == {"skipped": True}

    def boom(kb, **kw):
        raise RuntimeError("索引坏了")

    monkeypatch.setattr(feishu_service, "sync_kb", boom)  # 知识库级别的异常只记录，不中断这一轮
    res = feishu_service.auto_sync_round()
    assert res is not None and res["errors"] and "索引坏了" in res["errors"][0]
    assert not feishu_service._sync_lock.locked()


def test_auto_sync_loop_follows_config(env, monkeypatch):
    c, fake, kb_id = env
    from app.services.feishu_service import feishu_service

    assert feishu_service.auto_sync_running()            # 应用启动时就起了后台线程
    rounds: list[float] = []
    ran = threading.Event()

    def fake_round():
        rounds.append(time.monotonic())
        ran.set()
        return {}

    monkeypatch.setattr(feishu_service, "auto_sync_round", fake_round)
    monkeypatch.setattr(feishu_service, "auto_sync_tick", 0.02)
    feishu_service.stop_auto_sync()                      # 用新的 tick 重启
    feishu_service.start_auto_sync()
    try:
        time.sleep(0.2)
        assert rounds == []                              # 默认 0 = 关闭
        assert c.put("/api/v1/feishu/config", json={"auto_sync_minutes": 30}).json()["auto_sync_minutes"] == 30
        assert ran.wait(2)                               # 打开后先补一轮
        time.sleep(0.2)
        assert len(rounds) == 1                          # 30 分钟内不会再跑
        assert c.put("/api/v1/feishu/config", json={"auto_sync_minutes": -1}).status_code == 422
    finally:
        c.put("/api/v1/feishu/config", json={"auto_sync_minutes": 0})
        feishu_service.stop_auto_sync()
        monkeypatch.undo()
        feishu_service.start_auto_sync()


def _encrypt(payload: dict[str, Any], key: str) -> str:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    plain = json.dumps(payload).encode()
    pad = 16 - len(plain) % 16
    iv = b"0123456789abcdef"
    enc = Cipher(algorithms.AES(hashlib.sha256(key.encode()).digest()), modes.CBC(iv)).encryptor()
    return base64.b64encode(iv + enc.update(plain + bytes([pad]) * pad) + enc.finalize()).decode()


def _edit_event(did: str, token: str = "vtok", event_id: str | None = None) -> dict[str, Any]:
    return {"schema": "2.0", "header": {"event_id": event_id or uuid.uuid4().hex, "token": token, "event_type": "drive.file.edit_v1"},
            "event": {"file_token": did, "file_type": "docx", "operator_id_list": [{"open_id": "ou_x"}]}}


def test_events_verification_and_edit(env):
    c, fake, kb_id = env
    ev = "/api/v1/feishu/events"
    # 没配置 Token：拒绝
    assert c.post(ev, json={"type": "url_verification", "challenge": "abc", "token": "vtok"}).status_code == 403

    r = c.put("/api/v1/feishu/config", json={"event_verification_token": "vtok"})
    assert r.json()["event_verification_token"] == "********"
    assert c.post(ev, json={"type": "url_verification", "challenge": "abc", "token": "vtok"}).json() == {"challenge": "abc"}
    assert c.post(ev, json={"type": "url_verification", "challenge": "abc", "token": "bad"}).status_code == 403
    assert c.put("/api/v1/feishu/config", json={"event_verification_token": "********"}).status_code == 200   # 回传脱敏值不覆盖
    assert c.post(ev, json={"type": "url_verification", "challenge": "x", "token": "vtok"}).json() == {"challenge": "x"}

    # 配置了 Token 后新接入的文档自动订阅变更；之前接入的可以一键补订阅
    did = fake.add_doc("事件文档", [{"block_type": 2, "text": {"elements": [_t("第一版")]}}])
    other = fake.add_doc("没变的文档", [{"block_type": 2, "text": {"elements": [_t("不动")]}}])
    assert c.post("/api/v1/feishu/import", json={"kbId": kb_id, "refs": [did, other]}).json()[0]["ok"]
    assert {did, other} <= fake.subscribed
    sub = c.post("/api/v1/feishu/events/subscribe").json()
    assert sub["total"] >= 2 and sub["subscribed"] == sub["total"] - len(sub["failed"])

    # 文档编辑事件：只同步那一篇
    _edit(fake, did, "事件推送后的第二版")
    _edit(fake, other, "这篇也改了，但没有事件")
    fake.calls.clear()
    eid = uuid.uuid4().hex
    r = c.post(ev, json=_edit_event(did, event_id=eid))
    assert r.status_code == 200 and r.json() == {"documentId": did}
    assert "事件推送后的第二版" in _local_md(c, kb_id, did)
    assert "这篇也改了" not in _local_md(c, kb_id, other)
    assert not any(other in p for _, p in fake.calls)
    # 飞书重推同一事件：去重
    assert "ignored" in c.post(ev, json=_edit_event(did, event_id=eid)).json()
    # 别的事件 / 没接入的文档 / Token 不对
    assert "ignored" in c.post(ev, json={**_edit_event(did), "header": {"token": "vtok", "event_type": "im.message.receive_v1"}}).json()
    assert "ignored" in c.post(ev, json=_edit_event("doxcnNotLinked000001")).json()
    assert c.post(ev, json=_edit_event(did, token="bad")).status_code == 403
    assert c.post(ev, content=b"not json", headers={"Content-Type": "application/json"}).status_code == 400


def test_events_encrypted(env):
    c, fake, kb_id = env
    ev = "/api/v1/feishu/events"
    c.put("/api/v1/feishu/config", json={"event_verification_token": "vtok", "event_encrypt_key": ""})
    challenge = {"encrypt": _encrypt({"type": "url_verification", "challenge": "enc", "token": "vtok"}, "ekey")}
    assert c.post(ev, json=challenge).status_code == 400                 # 没配置 Encrypt Key
    c.put("/api/v1/feishu/config", json={"event_encrypt_key": "ekey"})
    assert c.post(ev, json=challenge).json() == {"challenge": "enc"}
    assert c.post(ev, json={"encrypt": _encrypt({"type": "url_verification", "challenge": "x", "token": "vtok"}, "wrong")}).status_code == 400

    did = fake.add_doc("加密事件文档", [{"block_type": 2, "text": {"elements": [_t("旧")]}}])
    assert c.post("/api/v1/feishu/import", json={"kbId": kb_id, "refs": [did]}).json()[0]["ok"]
    _edit(fake, did, "加密推送带来的新内容")
    r = c.post(ev, json={"encrypt": _encrypt(_edit_event(did), "ekey")})
    assert r.json() == {"documentId": did}
    assert "加密推送带来的新内容" in _local_md(c, kb_id, did)
    c.put("/api/v1/feishu/config", json={"event_verification_token": "", "event_encrypt_key": ""})


# ======================================================================================
# 智能体
# ======================================================================================
def _events(text: str) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def test_agent_offline_reads_linked_url(env):
    c, fake, kb_id = env
    did = fake.add_doc("考勤制度", [{"block_type": 2, "text": {"elements": [_t("每天 9:30 前打卡")]}}])
    r = c.post("/api/v1/feishu/agent/run", json={"kbId": kb_id, "question": f"总结一下 https://acme.feishu.cn/docx/{did} 这篇文档"})
    evs = _events(r.text)
    assert evs[0]["type"] == "run" and evs[0]["data"]["offline"] is True
    tool = next(e for e in evs if e["type"] == "tool")
    assert tool["data"]["tool"] == "read_doc" and tool["data"]["status"] == "ok"
    assert "9:30" in evs[-1]["content"]


def test_agent_plans_edit_and_waits_for_confirmation(env):
    c, fake, kb_id = env
    from app.services.feishu_service import feishu_service

    did = fake.add_doc("值班表", [{"block_type": 2, "text": {"elements": [_t("周一值班：张三")]}}])

    def responder(prompt: str) -> str:
        if prompt.startswith("用户请求") and "已执行" not in prompt and "工具结果" not in prompt:
            return json.dumps({"steps": [{"tool": "read_doc", "args": {"document_id": did}, "why": "先看内容"}]})
        if "已执行" in prompt:
            if "replace_text" in prompt:
                return '{"done": true}'
            return json.dumps({"steps": [{"tool": "replace_text", "args": {"document_id": did, "find": "张三", "replace": "李四"}, "why": "改值班人"}]})
        return "已提交修改：把周一值班人从张三改为李四，确认后生效。"

    agent = FeishuAgent(feishu_service.tools(kb_id), llm=FakeLLM(responder), max_steps=4, allow_write=False)
    evs = list(agent.run("把值班表里的张三换成李四"))
    kinds = [e["type"] for e in evs]
    assert kinds == ["run", "plan", "tool", "reflect", "pending", "final"]
    final = evs[-1]
    assert final["data"]["actions"][0]["status"] == "pending"
    assert "张三" in fake.markdown(did)                     # 还没改

    action = final["data"]["actions"][0]
    r = c.post("/api/v1/feishu/agent/execute", json={"kbId": kb_id, "tool": action["tool"], "args": action["args"]})
    assert r.status_code == 200 and r.json()["result"]["replaced"] == 1
    assert "李四" in fake.markdown(did) and "张三" not in fake.markdown(did)
    # 只读工具不能走 execute
    assert c.post("/api/v1/feishu/agent/execute", json={"kbId": kb_id, "tool": "read_doc", "args": {"document_id": did}}).status_code == 400


def test_agent_allow_write_creates_doc(env):
    c, fake, kb_id = env
    from app.services.feishu_service import feishu_service

    def responder(prompt: str) -> str:
        if "已执行" in prompt:
            return '{"done": true}'
        if "工具结果" in prompt:
            return "已新建《会议纪要》并同步进知识库。"
        return json.dumps({"steps": [{"tool": "create_doc", "args": {"title": "会议纪要", "markdown": "## 结论\n\n- 下周上线"}}]})

    agent = FeishuAgent(feishu_service.tools(kb_id), llm=FakeLLM(responder), allow_write=True)
    evs = list(agent.run("新建一篇会议纪要"))
    tool = next(e for e in evs if e["type"] == "tool")
    assert tool["data"]["status"] == "ok", tool
    did = tool["data"]["result"]["documentId"]
    assert "下周上线" in fake.markdown(did)
    assert tool["data"]["result"]["sync"]["status"] == "created"
    assert evs[-1]["data"]["actions"][0]["status"] == "ok"


def test_agent_reads_the_kbs_only_doc_when_asked_about_this_article(env):
    """截图里的问题：「这篇文章最核心的内容是什么？」——知识库里只有一篇，模型却一个工具都没规划，反问用户要文章。"""
    c, fake, _ = env
    from app.services.feishu_service import feishu_service

    kb_id = c.post("/api/v1/knowledge-bases/", json={"name": "Agent 开发规范", "description": "单篇"}).json()["id"]
    did = fake.add_doc("Agent 开发规范", [{"block_type": 2, "text": {"elements": [_t("工具调用必须幂等，失败要可重试")]}}])
    assert c.post("/api/v1/feishu/import", json={"kbId": kb_id, "refs": [f"https://acme.feishu.cn/docx/{did}"]}).json()[0]["ok"]

    systems: list[str] = []

    class Recording(FakeLLM):
        def chat(self, messages, *, system=None, **kw):
            systems.append(system or "")
            return super().chat(messages, system=system, **kw)

    def responder(prompt: str) -> str:
        if "工具结果" in prompt:
            return "核心：工具调用必须幂等。" if "幂等" in prompt else "没读到原文"
        if "已执行" in prompt:
            return '{"done": true}'
        return '{"steps": []}'                                        # 模型没规划任何工具

    agent = feishu_service.agent(kb_id)
    agent.llm = Recording(responder)
    evs = list(agent.run("这篇文章最核心的内容是什么？"))
    plan = next(e for e in evs if e["type"] == "plan")
    assert plan["data"]["steps"][0]["tool"] == "read_doc" and "Agent 开发规范" in plan["data"]["steps"][0]["why"]
    tool = next(e for e in evs if e["type"] == "tool")
    assert tool["data"]["status"] == "ok" and "幂等" in tool["data"]["result"]["markdown"]
    assert evs[-1]["content"] == "核心：工具调用必须幂等。"
    # 规划时模型就看得到这篇文档，并被告知不要反问用户要链接
    assert "《Agent 开发规范》" in systems[0] and "不要反过来让用户提供链接" in systems[0]


def test_agent_doc_fallback_only_for_document_questions():
    tools = {"read_doc": Tool("读", lambda a: a), "search": Tool("检索", lambda a: [])}
    one = FeishuAgent(tools, llm=FakeLLM(lambda p: '{"steps": []}'), docs=[{"title": "甲", "documentId": "d1"}])
    assert one._doc_fallback("总结一下要点")[0]["args"] == {"document_id": "d1"}
    assert one._doc_fallback("你好") == []                          # 闲聊不去读文档
    many = FeishuAgent(tools, llm=None, docs=[{"title": "甲", "documentId": "d1"}, {"title": "乙", "documentId": "d2"}])
    assert many._doc_fallback("这篇文章讲了什么")[0]["tool"] == "search"
    assert FeishuAgent(tools, llm=None)._doc_fallback("这篇文章讲了什么") == []   # 没接入文档：不臆造


ARTICLE = "# Agent 开发铁律\n\n## 铁律一：先设计，再写 Prompt\n\n先产出规格说明书（Spec），避免需求模糊导致模型幻觉。\n\n## 铁律二：最小闭环\n\n- 第一版只做核心功能\n- 大任务拆成小步骤\n"


@pytest.fixture(scope="module")
def article_kb(env):
    """只有一篇上传文档（不是飞书文档）的知识库，复现「这篇文章最核心的内容是什么？」。"""
    import io

    c, fake, _ = env
    kb = c.post("/api/v1/knowledge-bases/", json={"name": "Agent 开发规范", "description": "上传的规范"}).json()["id"]
    r = c.post(f"/api/v1/documents/upload?kbId={kb}", files={"file": ("Agent开发铁律.md", io.BytesIO(ARTICLE.encode()), "text/markdown")})
    assert r.status_code == 201, r.text
    return kb, r.json()["id"]


def test_agent_reads_the_only_uploaded_doc(env, article_kb):
    c, fake, _ = env
    kb, doc_id = article_kb
    from app.services.feishu_service import feishu_service

    def responder(prompt: str) -> str:
        if prompt.startswith("用户请求") and "工具结果" not in prompt and "已执行" not in prompt:
            return '{"steps": []}'                                 # 模型没规划任何工具（截图里的情况）
        if "已执行" in prompt:
            return '{"done": true}'
        return "核心：先设计再写 Prompt，从最小闭环做起。（出自《Agent 开发铁律》）"

    agent = feishu_service.agent(kb)
    assert any(d.get("docId") == doc_id for d in agent.docs)       # 上传的文档也交给了智能体
    agent.llm = FakeLLM(responder)
    evs = list(agent.run("这篇文章最核心的内容是什么？"))
    tool = next(e for e in evs if e["type"] == "tool")
    assert tool["data"]["tool"] == "read_kb_doc" and tool["data"]["status"] == "ok"
    assert "最小闭环" in tool["data"]["result"]["markdown"] and not tool["data"]["result"]["markdown"].startswith("---")
    assert "最小闭环" in evs[-1]["content"]
    # 规划 / 反思的系统提示里列出了这篇文档
    assert f"《Agent 开发铁律》 知识库文档 doc_id={doc_id}（用 read_kb_doc 读）" in agent._docs_text()


def test_write_kb_article_into_feishu_verbatim(env, article_kb):
    c, fake, _ = env
    kb, doc_id = article_kb
    from app.services.feishu_service import feishu_service

    tools = feishu_service.tools(kb)
    # 新建：正文由服务端原样复制（不经模型转抄），标题默认用原标题，不再同步回知识库（已经有这篇了）
    r = tools["create_doc"].fn({"source_doc_id": doc_id})
    did = r["documentId"]
    md = fake.markdown(did)
    assert fake.docs[did]["title"] == "Agent 开发铁律" and md.count("Agent 开发铁律") == 1
    for piece in ("## 铁律一：先设计，再写 Prompt", "先产出规格说明书（Spec），避免需求模糊导致模型幻觉。", "- 第一版只做核心功能", "- 大任务拆成小步骤"):
        assert piece in md
    assert "sync" not in r and fake.shared[did] == "tenant_editable"   # 新文档打开组织内链接分享，用户才看得到
    # 追加到已有文档；按标题也能找到知识库文档
    target = fake.add_doc("周报", [{"block_type": 2, "text": {"elements": [_t("开头")]}}])
    tools["append_doc"].fn({"document_id": target, "source_doc_id": "Agent 开发铁律"})
    assert "## 铁律二：最小闭环" in fake.markdown(target) and fake.markdown(target).index("开头") < fake.markdown(target).index("铁律二")
    with pytest.raises(ValueError, match="没有这篇文档"):
        tools["append_doc"].fn({"document_id": target, "source_doc_id": "不存在的文档"})
    # 写操作待确认时，确认接口照样能执行带 source_doc_id 的调用
    r = c.post("/api/v1/feishu/agent/execute", json={"kbId": kb, "tool": "create_doc", "args": {"source_doc_id": doc_id, "title": "铁律副本"}})
    assert r.status_code == 200 and "铁律一" in fake.markdown(r.json()["result"]["documentId"])


def test_new_doc_share_scope_missing_is_reported(env):
    c, fake, kb_id = env
    fake.missing_scopes = {"/drive/v1/permissions/"}
    try:
        r = c.post("/api/v1/feishu/remote", json={"title": "看不到的文档", "markdown": "正文"})
        assert r.status_code == 200 and "docs:permission.setting:write_only" in r.json()["shareError"]
    finally:
        fake.missing_scopes = set()
    c.put("/api/v1/feishu/config", json={"new_doc_share": "closed"})
    try:
        r = c.post("/api/v1/feishu/remote", json={"title": "不改分享", "markdown": "正文"}).json()
        assert r["documentId"] not in fake.shared and "shareError" not in r
    finally:
        c.put("/api/v1/feishu/config", json={"new_doc_share": "tenant_editable"})


def test_agent_unknown_tool_and_dedupe():
    calls = []
    tools = {"echo": Tool("回显", lambda a: calls.append(a) or a)}

    def responder(prompt: str) -> str:
        if "已执行" in prompt:
            return json.dumps({"steps": [{"tool": "echo", "args": {"x": 1}}]})      # 重复调用：跳过
        if "工具结果" in prompt:
            return "完成"
        return json.dumps({"steps": [{"tool": "echo", "args": {"x": 1}}, {"tool": "nope", "args": {}}]})

    evs = list(FeishuAgent(tools, llm=FakeLLM(responder), max_steps=5).run("测试"))
    assert len(calls) == 1
    assert any(e["type"] == "tool" and e["data"]["status"] == "error" and "未知工具" in e["data"]["error"] for e in evs)
    assert evs[-1]["content"] == "完成"
