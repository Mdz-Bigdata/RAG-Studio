from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db import KnowledgeBase, Repository, get_db
from ..schemas import KBCreate, KBResponse, KBUpdate
from ..services import document_service, index_service, retrieval_service, reindex_service

router = APIRouter()


def kb_response(kb: KnowledgeBase, repo: Repository) -> KBResponse:
    stats = index_service.stats(kb.id)
    docs = repo.list_documents(kb.id)
    return KBResponse(
        id=kb.id, name=kb.name, description=kb.description, category=kb.category, language=kb.language,
        createdAt=kb.created_at, updatedAt=kb.updated_at, documentCount=len(docs),
        chunkCount=int(stats.get("children") or 0), entityCount=int(stats.get("entities") or 0),
        indexEmbedder=(f"{stats.get('embedder')}:{stats.get('model')}" if stats.get("embedder") else None),
        needsReindex=bool(stats.get("needsReindex")),
        reindexing=reindex_service.is_active(kb.id),
    )


@router.get("/", response_model=list[KBResponse])
def list_kbs(db: Session = Depends(get_db)):
    repo = Repository(db)
    return [kb_response(kb, repo) for kb in repo.list_kbs()]


@router.post("/", response_model=KBResponse, status_code=status.HTTP_201_CREATED)
def create_kb(body: KBCreate, db: Session = Depends(get_db)):
    repo = Repository(db)
    kb = repo.create_kb(name=body.name, description=body.description, category=body.category, language=body.language or "zh")
    return kb_response(kb, repo)


@router.get("/{kb_id}", response_model=KBResponse)
def get_kb(kb_id: str, db: Session = Depends(get_db)):
    repo = Repository(db)
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "知识库不存在")
    return kb_response(kb, repo)


@router.put("/{kb_id}", response_model=KBResponse)
def update_kb(kb_id: str, body: KBUpdate, db: Session = Depends(get_db)):
    repo = Repository(db)
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "知识库不存在")
    repo.update_kb(kb, **body.model_dump())
    return kb_response(kb, repo)


@router.delete("/{kb_id}")
def delete_kb(kb_id: str, db: Session = Depends(get_db)):
    repo = Repository(db)
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "知识库不存在")
    repo.delete_kb(kb)
    index_service.drop(kb_id)
    return {"message": "知识库及其索引已删除"}


@router.get("/{kb_id}/stats")
def kb_stats(kb_id: str, db: Session = Depends(get_db)):
    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    return index_service.stats(kb_id)


@router.get("/{kb_id}/graph")
def kb_graph(kb_id: str, depth: int = 2, maxEntities: int = 200, db: Session = Depends(get_db)):
    """结构拓扑：文档 → 章节 → 小节 → 实体，实体之间的三元组关系，以及提到实体的其他章节。"""
    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    return retrieval_service.graph(kb_id, section_depth=max(1, min(depth, 4)), max_entities=max(10, min(maxEntities, 500)))


@router.get("/{kb_id}/entities")
def kb_entities(kb_id: str, db: Session = Depends(get_db)):
    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    return retrieval_service.entities(kb_id)


@router.post("/{kb_id}/reindex")
def reindex_kb(kb_id: str, background: BackgroundTasks, db: Session = Depends(get_db)):
    if not Repository(db).get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    kb = Repository(db).get_kb(kb_id)
    reindex_service.enqueue(kb_id, kb.name, "手动重建")
    return {"message": "已开始重建索引（后台执行）"}
