"""generation.llm.aux_model：辅助调用（查询改写 / HyDE / text-to-SQL / 路由）换成同一端点上更快的模型。

只有端点的模型列表里确实有它才换；列表里没有、取不到列表、或不是 OpenAI 兼容端点时一律沿用回答模型。
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import SimpleNamespace as NS


def _service(monkeypatch, tmp_path: Path, aux_model: str | None):
    os.environ.setdefault("SQLITE_URL", f"sqlite:///{tmp_path}/app.db")   # 单独运行时也不碰真实数据库
    mod = importlib.import_module("app.services.model_service")
    llm_cfg = NS(aux_model=aux_model, model_dump_json=lambda: f'{{"aux_model": "{aux_model}"}}')
    monkeypatch.setattr(mod, "settings_service", NS(get=lambda: NS(generation=NS(llm=llm_cfg))))
    return mod


def _openai_llm(mod, model: str, listed: list[str] | Exception):
    from prism.config import LLMConfig

    llm = mod.OpenAICompatibleLLM(LLMConfig(provider="openai_compatible", model=model), api_key="x", base_url="http://127.0.0.1:9/v1", aux=True)

    class Models:
        def list(self):
            if isinstance(listed, Exception):
                raise listed
            return NS(data=[NS(id=m) for m in listed])

    llm.client = NS(with_options=lambda **_kw: NS(models=Models()))
    return llm


def test_aux_model_only_when_endpoint_serves_it(monkeypatch, tmp_path: Path):
    mod = _service(monkeypatch, tmp_path, "deepseek-flash")
    svc = mod.ModelService()
    assert svc._aux_model(_openai_llm(mod, "deepseek-v4-pro", ["deepseek-flash", "deepseek-v4-pro"])) == "deepseek-flash"
    assert svc._aux_model(_openai_llm(mod, "glm-4.6", ["glm-4.6"])) is None                     # 换了供应商后留下的旧名字
    assert svc._aux_model(_openai_llm(mod, "deepseek-v4-pro", RuntimeError("404"))) is None      # 取不到模型列表
    assert svc._aux_model(_openai_llm(mod, "deepseek-flash", ["deepseek-flash"])) is None        # 本来就是它
    assert svc._aux_model(NS(model="claude-opus-5-5")) is None                                    # 非 OpenAI 兼容端点
    assert svc._aux_model(None) is None
    assert _service(monkeypatch, tmp_path, None).ModelService()._aux_model(_openai_llm(mod, "deepseek-v4-pro", ["deepseek-flash"])) is None


def test_llm_info_switches_only_aux_calls(monkeypatch, tmp_path: Path):
    mod = _service(monkeypatch, tmp_path, "deepseek-flash")
    svc = mod.ModelService()
    monkeypatch.setattr(svc, "_active", lambda category: None)
    built = []

    def build(cfg, *, aux):
        llm = _openai_llm(mod, "deepseek-v4-pro", ["deepseek-flash", "deepseek-v4-pro"])
        built.append(aux)
        return llm, {"source": "config", "model": "deepseek-v4-pro"}

    monkeypatch.setattr(svc, "_build_llm", build)
    aux_llm, aux_info = svc.llm_info(aux=True)
    main_llm, main_info = svc.llm_info(aux=False)
    assert aux_llm.model == "deepseek-flash" and aux_info["model"] == "deepseek-flash" and aux_info["mainModel"] == "deepseek-v4-pro"
    assert main_llm.model == "deepseek-v4-pro" and main_info["model"] == "deepseek-v4-pro"   # 回答仍用主模型
    assert svc.llm_info(aux=True)[0] is aux_llm and built == [True, False]                     # 按配置缓存，不重复取模型列表
