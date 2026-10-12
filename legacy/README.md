# legacy：Antigravity Hybrid RAG v1（存档，不参与运行）

原 `RAG-Agent` 项目第一版的源码，保留作对照。它的全部能力已经按新架构重写并并入
`prism/`（核心库）、`app/`（后端）、`frontend/`（前端）。

| v1 | 现在 |
| --- | --- |
| `service/parser-service.py`：pdfplumber 解析 + 固定 500 字切块 | `prism/ingest/parsers/`（MinerU 4.x / Markdown / Excel 行级 / docx / pptx / pdf 兜底）+ `ingest/cleaning.py` + `ingest/chunking/`（结构 / 语义切片 + Parent-Child） |
| `service/embedding-service.py`：OpenAI 兼容 + mock 向量 | `prism/index/embeddings.py`（sentence-transformers / OpenAI 兼容 / 哈希兜底） |
| `repository/qdrant-repository.py` | `prism/index/vector_store.py`（numpy / Qdrant / Chroma） |
| `service/retrieval-service.py`：向量 + BM25 + HyDE + RRF | `prism/retrieval/retriever.py`（漏斗式链路）+ `fusion.py` + `query_transform.py` |
| `service/rerank-service.py`：LLM 重排 / API / mock | `prism/retrieval/rerank.py`（Cross-Encoder / API / LLM / 词法） |
| `service/unweaver-service.py`：实体定义织入 | `prism/graph/unweaver.py` |
| `service/llm-service.py`：流式回答 + mock | `prism/generation/`（Anthropic SDK / OpenAI 兼容 / 回答协议 / 抽取式兜底） |
| `api/kb-router.py` | `app/api/knowledge_bases.py` |
| `api/document-router.py`（上传 / Markdown 校对 / parse-temp） | `app/api/documents.py`（另加切片预览、元数据、重新解析、Popo） |
| `api/chat-router.py`（`/{kb}/playground`、`/{kb}/chat`） | `app/api/retrieval.py`（`/retrieval/{kb}/playground`）、`app/api/chat.py`（`/chat/stream` + 会话） |
| `api/model-router.py`（配置 / 激活 / 拉取列表 / 连接测试） | `app/api/models.py` |
| `frontend-src-v1/components/*`（问答台、知识库、模型设置、检索游乐场） | `frontend/src/pages/*`（另加总览、语料覆盖、评测中心、策略配置） |

`.env` 里的旧字段（`OPENAI_API_KEY` / `OPENAI_API_BASE` / `LLM_MODEL` / `EMBEDDING_MODEL` / `USE_MINERU`）仍然有效，见 `app/config.py`。
v1 的运行数据（`rag.db`、`qdrant_db/`）表结构不同，没有迁移；重新上传文档即可。
