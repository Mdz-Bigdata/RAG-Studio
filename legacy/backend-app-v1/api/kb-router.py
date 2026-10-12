import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.repository import get_db, SQLiteRepository
from app.model import KnowledgeBaseModel
from app.schema import KBCreate, KBResponse

router = APIRouter()

@router.post("/", response_model=KBResponse, status_code=status.HTTP_201_CREATED)
def create_knowledge_base(kb_in: KBCreate, db: Session = Depends(get_db)):
    """
    新建知识库
    """
    repo = SQLiteRepository(db)
    kb_id = str(uuid.uuid4())
    kb = KnowledgeBaseModel(
        id=kb_id,
        name=kb_in.name,
        description=kb_in.description
    )
    return repo.create_kb(kb)

@router.get("/", response_model=list[KBResponse])
def list_knowledge_bases(db: Session = Depends(get_db)):
    """
    获取全部知识库列表
    """
    repo = SQLiteRepository(db)
    return repo.list_kbs()

@router.get("/{kb_id}", response_model=KBResponse)
def get_knowledge_base(kb_id: str, db: Session = Depends(get_db)):
    """
    获取特定知识库信息
    """
    repo = SQLiteRepository(db)
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )
    return kb

@router.delete("/{kb_id}", status_code=status.HTTP_200_OK)
def delete_knowledge_base(kb_id: str, db: Session = Depends(get_db)):
    """
    删除知识库及其下属所有文档和向量
    """
    repo = SQLiteRepository(db)
    # TODO: 级联清理向量库
    # Qdrant 我们通常在 document-router 或者这里删除对应的 Qdrant Collection。
    # 既然是按 kb 隔离 collection，我们可以直接删除对应的 collection！
    from app.repository import qdrant_repository
    from app.config import settings
    
    kb = repo.get_kb(kb_id)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )
        
    # 清理 Qdrant 中的 Collection
    collection_name = f"{settings.QDRANT_COLLECTION}_{kb_id}"
    try:
        if qdrant_repository.client.collection_exists(collection_name):
            qdrant_repository.client.delete_collection(collection_name)
    except Exception as e:
        print(f"Error deleting Qdrant collection on KB deletion: {e}")
        
    success = repo.delete_kb(kb_id)
    if not success:
         raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete knowledge base"
        )
    return {"message": "Knowledge base and associated vectors deleted successfully"}
