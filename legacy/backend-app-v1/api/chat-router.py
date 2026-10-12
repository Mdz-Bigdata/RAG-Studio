import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.config import settings
from app.repository import get_db, SQLiteRepository, qdrant_repository
from app.schema import ChatRequest, SearchPlaygroundRequest, SearchPlaygroundResponse
from app.service import retrieval_service, rerank_service, llm_service, unweaver_service, embedding_service
from app.model import ChunkModel

router = APIRouter()

@router.post("/{kb_id}/playground", response_model=SearchPlaygroundResponse)
def search_playground(
    kb_id: str, 
    req: SearchPlaygroundRequest, 
    db: Session = Depends(get_db)
):
    """
    检索游乐场接口。
    针对调试环境，并行运行多路召回：Dense、BM25、HyDE，展示 RRF 融合、LLM Rerank 排序以及 UnWeaver 实体织入的完整中间链条。
    """
    repo = SQLiteRepository(db)
    
    # 1. 确保知识库存在
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )
        
    db_chunks = db.query(ChunkModel).filter(ChunkModel.knowledgeBaseId == kb_id).all()
    if not db_chunks:
        return SearchPlaygroundResponse(
            denseResults=[],
            bm25Results=[],
            hydeResults=[],
            rrfResults=[],
            rerankedResults=[],
            weavedRelations=[],
            hydeDocument=""
        )
        
    chunk_map = {c.id: c for c in db_chunks}
    
    # --- 1.1 Dense 检索 ---
    dense_results = []
    try:
        query_vector = embedding_service.get_embedding(req.query, db=db)
        collection_name = f"{settings.QDRANT_COLLECTION}_{kb_id}"
        qdrant_repository.init_collection(collection_name, len(query_vector))
        points = qdrant_repository.search_vectors(collection_name, query_vector, limit=req.topK)
        for p in points:
            if p.id in chunk_map:
                dense_results.append({
                    "id": p.id,
                    "content": chunk_map[p.id].content,
                    "score": float(p.score),
                    "chunkIndex": chunk_map[p.id].chunkIndex,
                    "source": "dense"
                })
    except Exception as e:
        print(f"Playground Dense search failed: {e}")
        
    # --- 1.2 BM25 检索 ---
    bm25_results = []
    try:
        corpus = [retrieval_service.tokenize(c.content) for c in db_chunks]
        bm25_engine = BM25Okapi(corpus)
        query_tokens = retrieval_service.tokenize(req.query)
        scores = bm25_engine.get_scores(query_tokens)
        scored_chunks = sorted(list(zip(db_chunks, scores)), key=lambda x: x[1], reverse=True)
        for c, score in scored_chunks[:req.topK]:
            if score > 0:
                bm25_results.append({
                    "id": c.id,
                    "content": c.content,
                    "score": float(score),
                    "chunkIndex": c.chunkIndex,
                    "source": "bm25"
                })
    except Exception as e:
        print(f"Playground BM25 search failed: {e}")
        
    # --- 1.3 HyDE 检索 ---
    hyde_doc = ""
    hyde_results = []
    try:
        hyde_doc = llm_service.generate_hyde_document(req.query, db=db)
        hyde_vector = embedding_service.get_embedding(hyde_doc, db=db)
        collection_name = f"{settings.QDRANT_COLLECTION}_{kb_id}"
        points_hyde = qdrant_repository.search_vectors(collection_name, hyde_vector, limit=req.topK)
        for p in points_hyde:
            if p.id in chunk_map:
                hyde_results.append({
                    "id": p.id,
                    "content": chunk_map[p.id].content,
                    "score": float(p.score),
                    "chunkIndex": chunk_map[p.id].chunkIndex,
                    "source": "hyde"
                })
    except Exception as e:
        print(f"Playground HyDE search failed: {e}")
        
    # --- 1.4 RRF 融合 ---
    rrf_results = retrieval_service._reciprocal_rank_fusion(
        dense_results, bm25_results, hyde_results, k=settings.RRF_K
    )[:req.topK * 2]
    
    # 格式化 RRF 输出
    rrf_out = []
    for r in rrf_results:
        rrf_out.append({
            "id": r["id"],
            "content": r["content"],
            "score": r["score"],
            "chunkIndex": r["chunkIndex"],
            "sources": r["sources"]
        })
        
    # --- 1.5 Rerank 重新排序 ---
    reranked_results = rerank_service.rerank(req.query, rrf_results, top_k=req.topK, db=db)
    rerank_out = []
    for r in reranked_results:
        rerank_out.append({
            "id": r["id"],
            "content": r["content"],
            "score": r.get("rerankScore", r["score"] * 100),
            "chunkIndex": r["chunkIndex"],
            "sources": r.get("sources", [])
        })
        
    # --- 1.6 UnWeaver 实体织入 ---
    _, weaved_relations = unweaver_service.weave_context(db, kb_id, rerank_out)
    
    return SearchPlaygroundResponse(
        denseResults=dense_results,
        bm25Results=bm25_results,
        hydeResults=hyde_results,
        rrfResults=rrf_out,
        rerankedResults=rerank_out,
        weavedRelations=weaved_relations,
        hydeDocument=hyde_doc
    )

@router.post("/{kb_id}/chat")
def chat_stream(
    kb_id: str, 
    req: ChatRequest, 
    db: Session = Depends(get_db)
):
    """
    智能对话问答 (流式推送)。
    通过 SSE (Server-Sent Events) 返回：引用来源 (sources)、UnWeaver 编织链 (relations) 及大模型回答流。
    """
    repo = SQLiteRepository(db)
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )
        
    # 1. 混合多路检索召回 (Dense + BM25 + HyDE -> RRF)
    candidates = retrieval_service.hybrid_search(db, kb_id, req.message, top_k=5)
    
    # 2. 精排序
    reranked_chunks = rerank_service.rerank(req.message, candidates, top_k=3, db=db)
    
    # 3. UnWeaver 关系编织扩展背景
    final_chunks, weaved_relations = unweaver_service.weave_context(db, kb_id, reranked_chunks)
    
    # 4. 组装参考 Context
    context_parts = []
    for idx, c in enumerate(final_chunks):
        source_type = "Weaved Context" if c.get("isWeaved") else f"Chunk {c['chunkIndex']}"
        context_parts.append(f"[{source_type}]: {c['content']}")
    context_str = "\n\n".join(context_parts)
    
    # 5. 准备推送给前端的引用源列表
    sources = []
    for c in final_chunks:
        # 获取源文档名字
        doc_name = "System Glossary"
        if "documentId" in c:
            doc = repo.get_document(c["documentId"])
            if doc:
                doc_name = doc.fileName
                
        sources.append({
            "id": c["id"],
            "fileName": doc_name,
            "content": c["content"],
            "score": c.get("rerankScore", c.get("score", 0.0)),
            "isWeaved": c.get("isWeaved", False),
            "weavedEntity": c.get("weavedEntity", "")
        })

    def event_generator():
        # 发送引用源事件
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        # 发送实体编织事件
        yield f"event: relations\ndata: {json.dumps(weaved_relations)}\n\n"
        
        # 发送流式大模型生成事件
        try:
            for token in llm_service.generate_answer_stream(
                req.message, 
                context_str, 
                db=db, 
                image_b64=req.imageB64, 
                temp_doc_content=req.tempDocContent
            ):
                yield f"event: message\ndata: {json.dumps({'content': token})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
