"""策略配置：config.yaml 基线 + 数据库中的覆盖（策略配置页保存）。"""

from __future__ import annotations

import threading
from typing import Any

from prism.config import Settings, deep_merge

from ..config import app_config
from ..db import Repository, SessionLocal

SETTING_KEY = "settings_overrides"

# 允许前端覆盖的顶层段落（避免把路径类配置改坏）
EDITABLE_SECTIONS = ("parsing", "cleaning", "chunking", "enrich", "corpus", "sparse", "vector_store", "retrieval", "multimodal", "generation", "memory", "recommend", "agents")


class SettingsService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cached: Settings | None = None
        self._overrides: dict[str, Any] = {}

    def base(self) -> Settings:
        s = Settings.load(app_config.config_yaml)
        s.paths.data_dir = app_config.data_dir
        s.paths.index_dir = app_config.index_root
        s.paths.upload_dir = app_config.upload_dir
        s.paths.mineru_out_dir = app_config.data_dir / "mineru_out"
        s.paths.popo_dir = app_config.data_dir / "popo"
        s.parsing.mineru.enabled = s.parsing.mineru.enabled and app_config.use_mineru
        return s

    def overrides(self) -> dict[str, Any]:
        with SessionLocal() as db:
            return Repository(db).get_setting(SETTING_KEY, {}) or {}

    def get(self) -> Settings:
        with self._lock:
            if self._cached is None:
                self._overrides = self.overrides()
                self._cached = self.base().with_overrides(self._overrides)
            return self._cached

    def update(self, overrides: dict[str, Any]) -> Settings:
        clean = {k: v for k, v in overrides.items() if k in EDITABLE_SECTIONS}
        with SessionLocal() as db:
            repo = Repository(db)
            merged = deep_merge(repo.get_setting(SETTING_KEY, {}) or {}, clean)
            Settings.load(app_config.config_yaml).with_overrides(merged)  # 校验
            repo.set_setting(SETTING_KEY, merged)
        self.invalidate()
        return self.get()

    def reset(self) -> Settings:
        with SessionLocal() as db:
            Repository(db).set_setting(SETTING_KEY, {})
        self.invalidate()
        return self.get()

    def invalidate(self) -> None:
        with self._lock:
            self._cached = None


settings_service = SettingsService()
