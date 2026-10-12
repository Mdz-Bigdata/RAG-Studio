from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.repository import init_db
from app.api import kb_router, document_router, chat_router, model_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Hybrid RAG (MinerU + Qdrant + Dense + BM25 + HyDE + Rerank + RRF + UnWeaver)",
    version="1.0.0"
)

# 配置跨域请求 (CORS) - 允许 React 前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 企业级开发中，生产环境请配置具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    """
    系统启动时，初始化 SQLite 数据库表
    """
    init_db()
    print("Database tables initialized successfully.")

# 注册 API 路由分发
app.include_router(kb_router, prefix=f"{settings.API_V1_STR}/knowledge-bases", tags=["KnowledgeBases"])
app.include_router(document_router, prefix=f"{settings.API_V1_STR}/documents", tags=["Documents"])
app.include_router(chat_router, prefix=f"{settings.API_V1_STR}/chat", tags=["Chat"])
app.include_router(model_router, prefix=f"{settings.API_V1_STR}/models", tags=["Models"])

@app.get("/")
def read_root():
    return {
        "status": "online",
        "project": settings.PROJECT_NAME,
        "api_docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
