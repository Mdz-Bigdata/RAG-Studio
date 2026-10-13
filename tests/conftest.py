from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# 测试保持离线、确定：默认不加载本地 2GB 重排模型，更不触发下载。
# `RAG_TEST_RERANKER=1 pytest tests/test_rerankers.py -k live` 才会用真实的 bge-reranker-v2-m3。
if os.environ.get("RAG_TEST_RERANKER") != "1":
    os.environ.setdefault("RAG_DISABLE_LOCAL_RERANKER", "1")
# 同理：向量模型默认用哈希（config.yaml 里是 BGE-M3）；`RAG_TEST_EMBEDDER=1` 才加载真实模型
if os.environ.get("RAG_TEST_EMBEDDER") != "1":
    os.environ.setdefault("RAG_DISABLE_LOCAL_EMBEDDER", "1")

# 应用测试绝不能碰真实的 data/app.db：app.config / app.db.session 在第一次 import 时就把数据库地址和 .env 里的
# 密钥定死，而测试文件在收集阶段就可能 import 它们（早于任何 fixture）——那样整套测试会在真实库里建库、改策略、
# 停用模型。所以在最早的时机（conftest 先于所有测试文件加载）指到临时库，并清空密钥保持离线。
_TEST_DB_DIR = Path(tempfile.mkdtemp(prefix="rag-studio-test-"))
atexit.register(shutil.rmtree, _TEST_DB_DIR, ignore_errors=True)
os.environ["SQLITE_URL"] = f"sqlite:///{_TEST_DB_DIR / 'app.db'}"
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""

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
