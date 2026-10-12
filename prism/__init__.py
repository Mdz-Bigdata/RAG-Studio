"""prism：生产级 RAG 检索优化参考实现（核心库，Web 后端见 app/）。

分层结构（对应 docs/RAG_OPTIMIZATION_SUMMARY.md 的优化框架）：

- ingest     文档解析（MinerU 4.x / Markdown / Excel / Docx / PPTX / 兜底）、MinerU-Popo 文档树桥接、清洗、语义感知切片、Parent-Child
- index      向量索引（numpy / Qdrant / Chroma）、BM25 关键词索引、块存储与元数据
- graph      UnWeaver 实体定义织网（GraphRAG-lite）
- retrieval  查询改写 / Multi-Query / HyDE、双路召回、RRF 融合、Reranker 精排、MMR、粒度扩展、压缩、多跳 / Agentic / 路由
- generation 回答协议（依据 / 拒答 / 冲突 / 引用）与 LLM 客户端（Anthropic / OpenAI 兼容）
- eval       黄金评估集、Recall@K / MRR / NDCG、分阶段消融、错误归因
"""

__version__ = "0.2.0"
