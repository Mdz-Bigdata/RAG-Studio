"""RAG Agent Studio API。

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload      # 开发：前端由 vite 提供并代理 /api
    rag serve --port 8000                                          # 等价入口

若存在 frontend/dist（`npm run build` 的产物），同一进程直接托管前端页面，单端口即可访问。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import agents_router, chat_router, corpus_router, documents_router, eval_router, kb_router, lake_router, lance_router, memory_router, models_router, multimodal_router, recommend_router, retrieval_router, settings_router, system_router
from .config import app_config
from .db import init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("bm25s").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    # 启动时自查：config.yaml 被改过、或向量模型换了，就自动重建过期的索引
    from .services import reindex_service

    reindex_service.request("启动检查")
    # 模型预热放后台线程：不阻塞启动，第一个提问也不用再等模型加载
    import threading

    from .services import model_service

    threading.Thread(target=model_service.warmup, name="model-warmup", daemon=True).start()
    yield


app = FastAPI(
    title=app_config.project_name,
    version=app_config.version,
    description="MinerU 解析 + 语义感知切片 + Parent-Child + 向量/BM25/HyDE 多路召回 + RRF + Reranker + MMR + UnWeaver 织网 + 上下文压缩 + 多跳 / Agentic RAG + 多模态（文搜图 / 图搜图 / 图文混合 / 图搜文）+ 评测闭环",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=app_config.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

p = app_config.api_prefix
app.include_router(system_router, prefix=f"{p}/system", tags=["System"])
app.include_router(kb_router, prefix=f"{p}/knowledge-bases", tags=["KnowledgeBases"])
app.include_router(documents_router, prefix=f"{p}/documents", tags=["Documents"])
app.include_router(retrieval_router, prefix=f"{p}/retrieval", tags=["Retrieval"])
app.include_router(multimodal_router, prefix=f"{p}/multimodal", tags=["Multimodal"])
app.include_router(chat_router, prefix=f"{p}/chat", tags=["Chat"])
app.include_router(models_router, prefix=f"{p}/models", tags=["Models"])
app.include_router(settings_router, prefix=f"{p}/settings", tags=["Settings"])
app.include_router(eval_router, prefix=f"{p}/eval", tags=["Eval"])
app.include_router(corpus_router, prefix=f"{p}/corpus", tags=["Corpus"])
app.include_router(lance_router, prefix=f"{p}/lancedb", tags=["LanceDB"])
app.include_router(memory_router, prefix=f"{p}/memory", tags=["Memory"])
app.include_router(recommend_router, prefix=f"{p}/recommend", tags=["Recommend"])
app.include_router(agents_router, prefix=f"{p}/agents", tags=["Agents"])
app.include_router(lake_router, prefix=f"{p}/lake", tags=["AgentLake"])


@app.get(f"{p}/health")
def health():
    return {"status": "online", "project": app_config.project_name, "version": app_config.version, "docs": "/docs", "api": p}


DIST = app_config.base_dir / "frontend" / "dist"

if (DIST / "index.html").exists():
    # 生产形态：托管构建后的前端；非 /api 的未知路径回退到 index.html（SPA）
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "Not Found")
        target = (DIST / path).resolve()
        if path and target.is_file() and DIST.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
else:

    @app.get("/")
    def root():
        return health()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
