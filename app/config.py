"""应用级配置：路径、数据库、CORS，以及 .env 环境变量（兼容旧版 Antigravity 的 .env 字段）。"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[1]  # backend/
# DATA_DIR：把数据目录（SQLite / 上传 / 索引 / LanceDB）整体指到别处，用于临时环境和联调，不碰真实数据
DATA_DIR = Path(os.getenv("DATA_DIR") or BASE_DIR / "data").expanduser().resolve()


def load_dotenv(path: Path) -> None:
    """最小 .env 解析：KEY=VALUE，忽略注释与空行；不覆盖已存在的环境变量。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(BASE_DIR / ".env")


class AppConfig(BaseModel):
    project_name: str = "RAG Agent Studio"
    version: str = "0.2.0"
    api_prefix: str = "/api/v1"
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    upload_dir: Path = DATA_DIR / "uploads"
    index_root: Path = DATA_DIR / "indexes"
    config_yaml: Path = BASE_DIR / "config.yaml"
    sqlite_url: str = os.getenv("SQLITE_URL", f"sqlite:///{DATA_DIR / 'app.db'}")
    cors_origins: list[str] = ["*"]

    # 旧版 .env 兼容：无模型配置中心记录时的环境变量兜底
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_api_base: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    qdrant_url: str | None = os.getenv("QDRANT_URL") or None
    use_mineru: bool = os.getenv("USE_MINERU", "true").lower() == "true"


app_config = AppConfig()
for d in (app_config.data_dir, app_config.upload_dir, app_config.index_root):
    d.mkdir(parents=True, exist_ok=True)
