import os
from pydantic import BaseModel

class Settings(BaseModel):
    """
    RAG 系统全局配置类
    """
    # 基础配置
    PROJECT_NAME: str = "Corporate-Hybrid-RAG"
    API_V1_STR: str = "/api/v1"
    
    # 数据库配置
    SQLITE_URL: str = os.getenv("SQLITE_URL", "sqlite:///./rag.db")
    QDRANT_URL: str | None = os.getenv("QDRANT_URL", None)  # 若为空，则默认使用内存模式
    QDRANT_PATH: str | None = os.getenv("QDRANT_PATH", "./qdrant_db")  # 本地文件模式路径
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "hybrid_rag_collection")
    
    # 大模型与向量模型配置 (OpenAI 兼容接口)
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY", None)
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # 本地 Embedding 配置（如果不想调用在线 API）
    LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    USE_LOCAL_EMBEDDING: bool = os.getenv("USE_LOCAL_EMBEDDING", "false").lower() == "true"
    
    # 重排序配置
    RERANK_API_KEY: str | None = os.getenv("RERANK_API_KEY", None)
    RERANK_API_URL: str = os.getenv("RERANK_API_URL", "https://api.jina.ai/v1/rerank")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "jina-rerank-v2-base-multilingual")
    USE_LLM_RERANK: bool = os.getenv("USE_LLM_RERANK", "true").lower() == "true"  # 默认使用大模型重排，降低外部 API 依赖
    
    # RRF 融合参数
    RRF_K: int = int(os.getenv("RRF_K", "60"))
    
    # MinerU 配置
    MINERU_COMMAND: str = os.getenv("MINERU_COMMAND", "magic-pdf")
    USE_MINERU: bool = os.getenv("USE_MINERU", "false").lower() == "true"

# 实例化全局配置
settings = Settings()
