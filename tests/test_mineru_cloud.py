"""MinerU 远程解析：官方云端 API（mode: cloud）与自托管服务（mode: server）走同一套 v1 客户端。"""

from __future__ import annotations

import pytest

from prism.config import MINERU_CLOUD_API_URL, MinerUConfig, Settings
from prism.ingest.parsers import mineru_tiers as mt


def test_mode_decides_where_parsing_runs(monkeypatch):
    monkeypatch.delenv("MINERU_API_KEY", raising=False)
    assert MinerUConfig().remote_endpoint() is None and MinerUConfig().resolved_mode == "local"
    # 旧配置没有 mode：填了 api_url 就是自托管服务
    legacy = MinerUConfig(api_url="http://mineru:8000/", api_key="k")
    assert legacy.mode == "server" and legacy.remote_endpoint() == ("http://mineru:8000", "k")
    # 明确选了本机：即使留着服务地址也不连
    assert MinerUConfig(mode="local", api_url="http://mineru:8000").remote_endpoint() is None
    # 选了自托管但没填地址：无处可连，按本机处理
    assert MinerUConfig(mode="server").resolved_mode == "local"
    # 云端：默认官方地址；密钥先看配置，再看环境变量，都没有就匿名
    assert MinerUConfig(mode="cloud").remote_endpoint() == (MINERU_CLOUD_API_URL, "")
    monkeypatch.setenv("MINERU_API_KEY", "sk_env")
    assert MinerUConfig(mode="cloud").remote_endpoint() == (MINERU_CLOUD_API_URL, "sk_env")
    assert MinerUConfig(mode="cloud", cloud={"api_key": "sk_cfg"}).remote_endpoint() == (MINERU_CLOUD_API_URL, "sk_cfg")
    # 页面保存的覆盖值叠在 config.yaml 基线上
    s = Settings.lite().with_overrides({"parsing": {"mineru": {"mode": "cloud", "cloud": {"api_key": "sk_ui"}}}})
    assert s.parsing.mineru.resolved_mode == "cloud" and s.parsing.mineru.remote_endpoint()[1] == "sk_ui"


def test_cloud_tiers_come_from_the_service_and_missing_ones_fall_back(monkeypatch):
    # 云端目前只提供 standard：/v1/tiers 返回 {"object": "list", "data": [{"id": "standard", ...}]}
    assert mt._parse_remote_tiers({"object": "list", "data": [{"id": "standard", "object": "tier"}]}) == {"standard": True}
    assert mt.pick_remote_tier("flash", {"standard": True}) == "standard"
    assert mt.pick_remote_tier("standard", {"standard": True, "basic": True}) == "standard"
    assert mt.pick_remote_tier("basic", {"flash": True, "basic": True}) == "basic"
    assert mt.pick_remote_tier("flash", {}) == "flash"          # 对方没说提供什么（连不上）：照配置发
    assert mt.pick_remote_tier("flash", {"turbo": True}) == "turbo"

    calls = []
    monkeypatch.setattr(mt, "remote_tiers", lambda url, key="", timeout=3.0: (calls.append((url, key)), {"url": url, "reachable": True, "tiers": {"standard": True}, "error": None})[1])
    cfg = MinerUConfig(mode="cloud", tier="flash", cloud={"api_url": "https://cloud.test/api", "api_key": "sk_1"})
    ov = mt.tier_overview(cfg, available=True)
    status = {t["tier"]: (t["status"], t["statusLabel"]) for t in ov["tiers"]}
    assert status["standard"] == ("remote", "云端 API 提供") and status["flash"] == ("remote_missing", "云端 API 未提供")
    assert ov["mode"] == "cloud" and ov["effectiveTier"] == "standard" and ov["source"] == "remote"
    assert ov["remote"]["kind"] == "cloud" and ov["remote"]["hasKey"] is True
    assert ov["recommended"]["tier"] == "standard" and [t["tier"] for t in ov["tiers"] if t["recommended"]] == ["standard"]
    assert all(t["download"] is None for t in ov["tiers"])     # 远程解析不需要本机下载模型包
    # 连不上：无法确认提供哪些档位，如实标出来（而不是显示成"云端 API 提供"）；档位照配置发
    monkeypatch.setattr(mt, "remote_tiers", lambda url, key="", timeout=3.0: {"url": url, "reachable": False, "tiers": {}, "error": "timed out"})
    down = mt.tier_overview(MinerUConfig(mode="cloud", tier="flash", cloud={"api_url": "https://down.test/api"}), available=True)
    assert {t["status"] for t in down["tiers"]} == {"remote_unreachable"} and down["tiers"][0]["statusLabel"] == "云端 API 连接不上" and down["effectiveTier"] == "flash"
    monkeypatch.setattr(mt, "remote_tiers", lambda url, key="", timeout=3.0: (calls.append((url, key)), {"url": url, "reachable": True, "tiers": {"standard": True}, "error": None})[1])
    # 短期缓存：状态页和解析各问一次，只发一轮请求
    assert mt.effective_tier(cfg) == "standard" and calls == [("https://cloud.test/api", "sk_1")]


def test_check_connection_reports_identity_limits_and_bad_keys(monkeypatch):
    import httpx

    assert mt.check_connection(MinerUConfig())["ok"] is False
    assert "地址" in mt.check_connection(MinerUConfig(mode="server"))["error"]

    monkeypatch.setattr(mt, "remote_tiers", lambda url, key="", timeout=3.0: {"url": url, "reachable": True, "tiers": {"standard": True}, "error": None})
    seen = {}

    def fake_get(url, headers=None, timeout=None):
        seen["url"], seen["auth"] = url, (headers or {}).get("Authorization")
        if (headers or {}).get("Authorization") == "Bearer sk_bad":
            return httpx.Response(401, json={"msgCode": "A0202", "msg": "user authenticate failed"})
        return httpx.Response(200, json={"object": "usage", "access_level": "anonymous", "current": {"pages_processed": 3}, "limits": {"max_pages_per_file": 200, "max_file_size_bytes": 209715200}})

    monkeypatch.setattr(httpx, "get", fake_get)
    r = mt.check_connection(MinerUConfig(mode="cloud", cloud={"api_url": "https://cloud.test/api"}))
    assert r["ok"] and r["accessLevel"] == "anonymous" and r["limits"]["max_pages_per_file"] == 200 and r["tiers"] == ["standard"]
    assert seen == {"url": "https://cloud.test/api/v1/usage", "auth": None} and r["hasKey"] is False
    bad = mt.check_connection(MinerUConfig(mode="cloud", cloud={"api_url": "https://cloud.test/api", "api_key": "sk_bad"}))
    assert bad["ok"] is False and "密钥" in bad["error"] and "sk_bad" not in str(bad)

    monkeypatch.setattr(mt, "remote_tiers", lambda url, key="", timeout=3.0: {"url": url, "reachable": False, "tiers": {}, "error": "timed out"})
    down = mt.check_connection(MinerUConfig(mode="cloud"))
    assert down["ok"] is False and "连接不上" in down["error"]


def test_parser_sends_documents_to_the_configured_remote(monkeypatch, tmp_path):
    pytest.importorskip("mineru.parser")
    import mineru.parser
    import mineru.render

    from prism.ingest.parsers.mineru_parser import MinerUParser

    made = []

    class FakeResult:
        middle_json = {"fake": True}

        def markdown(self):
            return "# 标题\n\n正文"

        def to_json(self):
            return "{}"

    class FakeApiParser:
        def __init__(self, **kwargs):
            made.append(kwargs)

        def parse(self, path, page_range=""):
            return FakeResult()

    monkeypatch.setattr(mineru.parser, "MinerUApiParser", FakeApiParser)
    monkeypatch.setattr(mineru.parser, "parse", lambda *a, **k: pytest.fail("云端模式不应在本机解析"))
    monkeypatch.setattr(mineru.render, "render_content_list", lambda mid: [{"type": "text", "text": "标题", "text_level": 1, "page_idx": 0}, {"type": "text", "text": "正文", "page_idx": 0}])
    monkeypatch.setattr(mt, "remote_tiers", lambda url, key="", timeout=3.0: {"url": url, "reachable": True, "tiers": {"standard": True}, "error": None})
    monkeypatch.delenv("MINERU_API_KEY", raising=False)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    # 云端：配置的是 flash，云端只有 standard → 实际用 standard；没有密钥 → 交给 SDK（环境变量 / 匿名）；图片随结果取回
    doc = MinerUParser(MinerUConfig(mode="cloud", tier="flash", ocr_mode="ocr", cloud={"api_url": "https://cloud-a.test/api"}, save_outputs=False)).parse(pdf)
    assert made[-1] == {"api_url": "https://cloud-a.test/api", "api_key": None, "tier": "standard", "ocr_mode": "ocr", "include_images": True}
    assert doc.metadata.parser_tier == "standard" and doc.metadata.extra["mineru_mode"] == "cloud" and [b.text for b in doc.blocks] == ["标题", "正文"]

    # 自托管服务：行为与以前一致（不取图片包），密钥原样带上
    doc = MinerUParser(MinerUConfig(api_url="http://mineru-b:8000", api_key="k", tier="standard", save_outputs=False)).parse(pdf)
    assert made[-1] == {"api_url": "http://mineru-b:8000", "api_key": "k", "tier": "standard", "ocr_mode": None, "include_images": False}
    assert doc.metadata.extra["mineru_mode"] == "server"


def test_api_keys_are_masked_and_the_mask_is_never_saved_back():
    from app.api.settings import MASKED, _drop_masked, _mask_secrets

    shown = _mask_secrets({"parsing": {"mineru": {"api_key": "", "cloud": {"api_key": "sk_secret", "api_url": "u"}}}})
    assert shown["parsing"]["mineru"]["cloud"] == {"api_key": MASKED, "api_url": "u"} and shown["parsing"]["mineru"]["api_key"] == ""
    # 页面把脱敏值原样传回来：丢掉这一项，不能把真密钥覆盖成星号；清空（""）和新密钥照常保存
    assert _drop_masked({"parsing": {"mineru": {"mode": "cloud", "cloud": {"api_key": MASKED}}}}) == {"parsing": {"mineru": {"mode": "cloud", "cloud": {}}}}
    assert _drop_masked({"parsing": {"mineru": {"cloud": {"api_key": ""}, "api_key": "new"}}}) == {"parsing": {"mineru": {"cloud": {"api_key": ""}, "api_key": "new"}}}


def test_new_fields_do_not_invalidate_existing_indexes(monkeypatch):
    import importlib

    mod = importlib.import_module("app.services.reindex_service")   # app.services 包里同名属性是实例，要拿模块

    def fingerprint(mineru: dict | None = None) -> str:
        s = Settings.lite().with_overrides({"parsing": {"mineru": mineru}} if mineru else None)
        monkeypatch.setattr(mod.settings_service, "get", lambda: s)
        return mod.ingest_fingerprint()

    base = fingerprint()
    # 本机解析：顺手填了云端密钥不触发重建
    assert fingerprint({"cloud": {"api_key": "sk_1"}}) == base
    # 切到云端：解析结果会变 → 重建；之后换密钥不重建
    cloud = fingerprint({"mode": "cloud"})
    assert cloud != base and fingerprint({"mode": "cloud", "cloud": {"api_key": "sk_2"}}) == cloud
    # 指纹里没有新字段，也没有密钥：升级后已有索引不会被判为过期
    dumped = Settings.lite().parsing.model_dump(mode="json", exclude={"mineru": {"mode", "cloud"}})
    assert "mode" not in dumped["mineru"] and "cloud" not in dumped["mineru"]
