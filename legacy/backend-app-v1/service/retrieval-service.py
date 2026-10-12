import re
from sqlalchemy.orm import Session
from rank_bm25 import BM25Okapi
from app.config import settings
from app.model import ChunkModel
from app.repository import qdrant_repository
import importlib
_embedding_mod = importlib.import_module("app.service.embedding-service")
_llm_mod = importlib.import_module("app.service.llm-service")
embedding_service = _embedding_mod.EmbeddingService()
llm_service = _llm_mod.LLMService()

class RetrievalService:
    """
    混合检索服务。集成 Dense 向量检索、本地 BM25 稀疏检索、HyDE 生成以及多路召回融合 (RRF)。
    """
    
    def tokenize(self, text: str) -> list[str]:
        """
        中英文混合轻量分词器。
        将中文字符按字拆分，英文单词按词提取，实现零依赖的高精度倒排匹配。
        """
        pattern = re.compile(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+")
        return pattern.findall(text.lower())

    def hybrid_search(self, db: Session, kb_id: str, query: str, top_k: int = 5) -> list[dict]:
        """
        混合检索核心逻辑：Dense 召回 + BM25 召回 + HyDE 召回，通过 RRF 融合，并过滤去重。
        """
        # 从数据库中拉取当前知识库下的所有 Chunks (用于 BM25 索引和属性映射)
        db_chunks = db.query(ChunkModel).filter(ChunkModel.knowledgeBaseId == kb_id).all()
        if not db_chunks:
            return []
            
        # 建立 Chunk ID 到 Model 的快速映射
        chunk_map = {c.id: c for c in db_chunks}
        
        # --- 1. 第一路：密集向量检索 (Dense Retrieval) ---
        dense_results = []
        try:
            query_vector = embedding_service.get_embedding(query, db=db)
            collection_name = f"{settings.QDRANT_COLLECTION}_{kb_id}"
            
            # 确保 collection 存在
            qdrant_repository.init_collection(collection_name, len(query_vector))
            
            qdrant_points = qdrant_repository.search_vectors(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k * 2
            )
            
            for p in qdrant_points:
                chunk_id = p.id
                if chunk_id in chunk_map:
                    dense_results.append({
                        "id": chunk_id,
                        "documentId": chunk_map[chunk_id].documentId,
                        "knowledgeBaseId": kb_id,
                        "content": chunk_map[chunk_id].content,
                        "chunkIndex": chunk_map[chunk_id].chunkIndex,
                        "score": p.score,
                        "source": "dense"
                    })
        except Exception as e:
            print(f"Dense retrieval failed: {e}")
            
        # --- 2. 第二路：BM25 检索 (Sparse Retrieval) ---
        bm25_results = []
        try:
            # 构建分词语料库
            corpus = [self.tokenize(c.content) for c in db_chunks]
            bm25 = BM25Okapi(corpus)
            
            tokenized_query = self.tokenize(query)
            # 计算得分
            doc_scores = bm25.get_scores(tokenized_query)
            
            # 配对 Chunk 与得分，并按得分降序
            scored_chunks = list(zip(db_chunks, doc_scores))
            scored_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)
            
            # 只取前 top_k * 2 个相关文档
            for c, score in scored_chunks[:top_k * 2]:
                if score > 0:  # 排除完全不匹配的
                    bm25_results.append({
                        "id": c.id,
                        "documentId": c.documentId,
                        "knowledgeBaseId": kb_id,
                        "content": c.content,
                        "chunkIndex": c.chunkIndex,
                        "score": float(score),
                        "source": "bm25"
                    })
        except Exception as e:
            print(f"BM25 retrieval failed: {e}")
            
        # --- 3. 第三路：HyDE 检索 (Hypothetical Document Embeddings) ---
        hyde_results = []
        try:
            # 使用 LLM 生成假想文档
            hyde_doc = llm_service.generate_hyde_document(query, db=db)
            hyde_vector = embedding_service.get_embedding(hyde_doc, db=db)
            collection_name = f"{settings.QDRANT_COLLECTION}_{kb_id}"
            
            qdrant_points_hyde = qdrant_repository.search_vectors(
                collection_name=collection_name,
                query_vector=hyde_vector,
                limit=top_k * 2
            )
            
            for p in qdrant_points_hyde:
                chunk_id = p.id
                if chunk_id in chunk_map:
                    hyde_results.append({
                        "id": chunk_id,
                        "documentId": chunk_map[chunk_id].documentId,
                        "knowledgeBaseId": kb_id,
                        "content": chunk_map[chunk_id].content,
                        "chunkIndex": chunk_map[chunk_id].chunkIndex,
                        "score": p.score,
                        "source": "hyde"
                    })
        except Exception as e:
            print(f"HyDE retrieval failed: {e}")

        # --- 4. 多路召回融合 (RRF) ---
        rrf_results = self._reciprocal_rank_fusion(
            dense_results, bm25_results, hyde_results, 
            k=settings.RRF_K
        )
        
        # 返回前 top_k * 3 个作为重排序候选
        return rrf_results[:top_k * 3]

    def _reciprocal_rank_fusion(self, *lists, k: int = 60) -> list[dict]:
        """
        使用倒数排名融合算法 (RRF) 归一化并融合多路检索排序结果
        """
        rrf_scores = {}
        
        for lst in lists:
            for rank, item in enumerate(lst):
                doc_id = item["id"]
                if doc_id not in rrf_scores:
                    # 保留原有的属性，但重设得分为 0.0
                    rrf_scores[doc_id] = {
                        "item": dict(item), 
                        "rrf_score": 0.0,
                        "sources": []
                    }
                # RRF 公式累加
                rrf_scores[doc_id]["rrf_score"] += 1.0 / (k + rank + 1)
                rrf_scores[doc_id]["sources"].append(item["source"])
                
        # 降序排序
        sorted_rrf = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        
        final_results = []
        for x in sorted_rrf:
            item = x["item"]
            item["score"] = x["rrf_score"]
            item["sources"] = list(set(x["sources"]))
            final_results.append(item)
            
        return final_results
