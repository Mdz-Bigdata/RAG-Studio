from __future__ import annotations

import os
from pathlib import Path

import pytest

# 测试保持离线、确定：默认不加载本地 2GB 重排模型，更不触发下载。
# `RAG_TEST_RERANKER=1 pytest tests/test_rerankers.py -k live` 才会用真实的 bge-reranker-v2-m3。
if os.environ.get("RAG_TEST_RERANKER") != "1":
    os.environ.setdefault("RAG_DISABLE_LOCAL_RERANKER", "1")
# 同理：向量模型默认用哈希（config.yaml 里是 BGE-M3）；`RAG_TEST_EMBEDDER=1` 才加载真实模型
if os.environ.get("RAG_TEST_EMBEDDER") != "1":
    os.environ.setdefault("RAG_DISABLE_LOCAL_EMBEDDER", "1")

from prism.config import Settings  # noqa: E402
from prism.index import Index  # noqa: E402
from prism.ingest import IngestPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


@pytest.fixture
def lite_settings(tmp_path: Path) -> Settings:
    return Settings.lite(index_dir=tmp_path / "index")


@pytest.fixture(scope="session")
def built_index(tmp_path_factory) -> Index:
    settings = Settings.lite(index_dir=tmp_path_factory.mktemp("index"))
    pipeline = IngestPipeline(settings)
    report = pipeline.ingest_paths([RAW], save=True)
    assert report.documents >= 4, report.summary()
    assert not report.failed, report.failed
    return pipeline.index


@pytest.fixture(scope="session")
def session_settings(built_index: Index) -> Settings:
    return built_index.settings
