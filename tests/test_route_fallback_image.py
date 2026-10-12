"""带图提问 + 自动路由兜底：路由选中的库答不上、换到别的库时，图片那部分（以图搜文、相似图片）也必须换到新库，
不能把旧库的图搜文结果混进新库的上下文；换库后重做以图搜文失败时，宁可不要图片侧，也不能用旧库的结果。"""

from __future__ import annotations

import base64
import io
import json
import os

import pytest

LEAVE = "# 员工手册\n\n## 年假\n\n员工每年享有 5 天带薪年假，工作满 3 年后增加至 10 天。年假需提前 3 个工作日在 OA 系统提交申请。\n"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("fallback_image")
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


def _png_b64() -> str:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (320, 120), "white")
    ImageDraw.Draw(im).rectangle([20, 20, 300, 100], outline=(0, 0, 0), width=3)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _stream(client, body: dict) -> dict:
    events: dict = {}
    with client.stream("POST", "/api/v1/chat/stream", json=body) as r:
        ev = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:") and ev in ("meta", "plan", "sources", "images", "done"):
                events[ev] = json.loads(line[5:])
    return events


@pytest.fixture()
def kbs(client, monkeypatch):
    from app.services.retrieval_service import retrieval_service

    empty = client.post("/api/v1/knowledge-bases/", json={"name": "空库·图片兜底", "description": "还没有文档"}).json()
    hr = client.post("/api/v1/knowledge-bases/", json={"name": "人事库·图片兜底", "description": "年假、请假"}).json()
    client.post(f"/api/v1/documents/upload?kbId={hr['id']}", files={"file": ("员工手册.md", io.BytesIO(LEAVE.encode()), "text/markdown")})
    # 离线哈希向量没法按语义路由：固定成"空库排第一"，必然触发兜底
    monkeypatch.setattr(retrieval_service, "route", lambda q, top_n=1, **kw: [{"kb_id": empty["id"], "name": "空库·图片兜底", "score": 0.6},
                                                                             {"kb_id": hr["id"], "name": "人事库·图片兜底", "score": 0.5}][:top_n])
    return empty["id"], hr["id"]


def _spy(monkeypatch, fail_on_call: int | None = None):
    """记录以图搜文（_image_to_text）与合并图片侧（_merge_image_side）各用的是哪个知识库。"""
    from app.services.chat_service import ChatService, chat_service

    seen = {"i2t": [], "merge": []}

    def image_to_text(kb_id, message, image_b64, top_k, **kw):
        seen["i2t"].append(kb_id)
        if fail_on_call is not None and len(seen["i2t"]) == fail_on_call:
            raise RuntimeError("模拟：新库的以图搜文失败")
        return ChatService._image_to_text(chat_service, kb_id, message, image_b64, top_k, **kw)

    def merge(message, r, *args, **kwargs):
        seen["merge"].append(r.index.directory.name)
        return ChatService._merge_image_side(chat_service, message, r, *args, **kwargs)

    monkeypatch.setattr(chat_service, "_image_to_text", image_to_text)
    monkeypatch.setattr(chat_service, "_merge_image_side", merge)
    return seen


def test_image_side_follows_the_fallback_kb(client, kbs, monkeypatch):
    empty, hr = kbs
    seen = _spy(monkeypatch)
    ev = _stream(client, {"kbId": "auto", "message": "这张图里说的年假怎么休", "imageB64": _png_b64()})
    fb = ev["plan"]["routeFallback"]
    assert fb and fb["fromKbId"] == empty and fb["toKbId"] == hr
    assert seen["i2t"] == [empty, hr]                   # 先按路由选中的库认图，换库后按新库重做以图搜文
    assert seen["merge"] == [hr]                        # 并入上下文、找相似图片用的是新库
    assert ev["sources"] and all(s["fileName"] == "员工手册.md" for s in ev["sources"])
    assert all(hr in (im.get("url") or "") for im in ev.get("images") or [])


def test_failed_redo_never_mixes_the_old_kb(client, kbs, monkeypatch):
    empty, hr = kbs
    seen = _spy(monkeypatch, fail_on_call=2)            # 换库后重做以图搜文时出错
    ev = _stream(client, {"kbId": "auto", "message": "这张图里说的年假怎么休", "imageB64": _png_b64()})
    assert ev["plan"]["routeFallback"]["toKbId"] == hr   # 仍然换到能答的库
    assert seen["merge"] == []                          # 绝不拿旧库的图搜文结果去并入新库
    image_search = ev["plan"]["imageSearch"] or {}
    assert image_search.get("mode") == "error" and any("图片检索失败" in n for n in image_search.get("notes", []))
    assert ev["sources"] and all(s["fileName"] == "员工手册.md" for s in ev["sources"])


def test_redo_reuses_the_first_recognition(client, kbs, monkeypatch):
    """换库后重做以图搜文：图中文字、图片描述与知识库无关，沿用第一次的识别结果——整轮只 OCR 一次、只描述一次。"""
    from prism.multimodal.describe import ImageDescriber, OCREngine

    calls = {"ocr": 0, "caption": 0}
    real_read, real_caption = OCREngine.read, ImageDescriber.caption

    def read(self, image):
        calls["ocr"] += 1
        return real_read(self, image)

    def caption(self, image, sha1, *, hint=""):
        calls["caption"] += 1
        return real_caption(self, image, sha1, hint=hint)

    monkeypatch.setattr(OCREngine, "read", read)
    monkeypatch.setattr(ImageDescriber, "caption", caption)
    seen = _spy(monkeypatch)
    ev = _stream(client, {"kbId": "auto", "message": "这张图里说的年假怎么休", "imageB64": _png_b64()})
    empty, hr = kbs
    assert ev["plan"]["routeFallback"]["toKbId"] == hr and seen["i2t"] == [empty, hr] and seen["merge"] == [hr]
    assert calls == {"ocr": 1, "caption": 1}
