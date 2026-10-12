from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.config import settings
from app.model import Base, KnowledgeBaseModel, DocumentModel, ChunkModel, EntityRelationModel, ModelConfigModel

# 创建数据库引擎
engine = create_engine(
    settings.SQLITE_URL, 
    connect_args={"check_same_thread": False}  # 仅针对 SQLite，支持多线程
)

# 绑定 Session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    初始化数据库，创建所有未存在的表
    """
    Base.metadata.create_all(bind=engine)

def get_db():
    """
    获取数据库会话生成器 (FastAPI Dependency)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class SQLiteRepository:
    """
    SQLite 数据访问仓储
    """
    def __init__(self, db: Session):
        self.db = db
        
    # --- 知识库 CRUD ---
    
    def create_kb(self, kb_model: KnowledgeBaseModel) -> KnowledgeBaseModel:
        self.db.add(kb_model)
        self.db.commit()
        self.db.refresh(kb_model)
        return kb_model
        
    def get_kb(self, kb_id: str) -> KnowledgeBaseModel | None:
        return self.db.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == kb_id).first()
        
    def list_kbs(self) -> list[KnowledgeBaseModel]:
        return self.db.query(KnowledgeBaseModel).all()
        
    def delete_kb(self, kb_id: str) -> bool:
        kb = self.get_kb(kb_id)
        if kb:
            self.db.delete(kb)
            self.db.commit()
            return True
        return False

    # --- 文档 CRUD ---
    
    def create_document(self, doc_model: DocumentModel) -> DocumentModel:
        self.db.add(doc_model)
        self.db.commit()
        self.db.refresh(doc_model)
        return doc_model
        
    def get_document(self, doc_id: str) -> DocumentModel | None:
        return self.db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
        
    def get_documents_by_kb(self, kb_id: str) -> list[DocumentModel]:
        return self.db.query(DocumentModel).filter(DocumentModel.knowledgeBaseId == kb_id).all()
        
    def update_document_status(self, doc_id: str, status: str, error_msg: str | None = None) -> DocumentModel | None:
        doc = self.get_document(doc_id)
        if doc:
            doc.status = status
            if error_msg:
                doc.errorMessage = error_msg
            self.db.commit()
            self.db.refresh(doc)
            return doc
        return None
        
    def update_document_markdown(self, doc_id: str, markdown: str, status: str = "COMPLETED") -> DocumentModel | None:
        doc = self.get_document(doc_id)
        if doc:
            doc.markdownContent = markdown
            doc.status = status
            self.db.commit()
            self.db.refresh(doc)
            return doc
        return None
        
    def delete_document(self, doc_id: str) -> bool:
        doc = self.get_document(doc_id)
        if doc:
            self.db.delete(doc)
            self.db.commit()
            return True
        return False

    # --- 切片 CRUD ---
    
    def bulk_create_chunks(self, chunks: list[ChunkModel]) -> list[ChunkModel]:
        self.db.add_all(chunks)
        self.db.commit()
        return chunks
        
    def get_chunks_by_document(self, doc_id: str) -> list[ChunkModel]:
        return self.db.query(ChunkModel).filter(ChunkModel.documentId == doc_id).all()
        
    def get_chunks_by_kb(self, kb_id: str) -> list[ChunkModel]:
        return self.db.query(ChunkModel).filter(ChunkModel.knowledgeBaseId == kb_id).all()

    # --- 实体关系 CRUD (UnWeaver) ---
    
    def create_entity_relation(self, relation: EntityRelationModel) -> EntityRelationModel:
        self.db.add(relation)
        self.db.commit()
        self.db.refresh(relation)
        return relation
        
    def bulk_create_entity_relations(self, relations: list[EntityRelationModel]) -> list[EntityRelationModel]:
        self.db.add_all(relations)
        self.db.commit()
        return relations
        
    def get_relations_by_kb(self, kb_id: str) -> list[EntityRelationModel]:
        return self.db.query(EntityRelationModel).filter(EntityRelationModel.knowledgeBaseId == kb_id).all()
        
    def get_relation_by_entity(self, kb_id: str, entity_name: str) -> list[EntityRelationModel]:
        return self.db.query(EntityRelationModel).filter(
            EntityRelationModel.knowledgeBaseId == kb_id,
            EntityRelationModel.entityName == entity_name
        ).all()

    # --- 大模型配置 CRUD ---
    
    def create_or_update_model_config(self, config_model: ModelConfigModel) -> ModelConfigModel:
        """
        创建或更新大模型配置
        """
        existing = self.db.query(ModelConfigModel).filter(ModelConfigModel.id == config_model.id).first()
        if existing:
            existing.providerName = config_model.providerName
            existing.baseUrl = config_model.baseUrl
            if config_model.apiKey:  # 如果提供了新 Key 则更新
                existing.apiKey = config_model.apiKey
            existing.category = config_model.category
            existing.modelName = config_model.modelName
            existing.isActive = config_model.isActive
            self.db.commit()
            self.db.refresh(existing)
            return existing
        else:
            self.db.add(config_model)
            self.db.commit()
            self.db.refresh(config_model)
            return config_model

    def list_model_configs(self) -> list[ModelConfigModel]:
        """
        获取所有大模型配置
        """
        return self.db.query(ModelConfigModel).all()

    def get_model_config(self, config_id: str) -> ModelConfigModel | None:
        """
        根据 ID 获取模型配置
        """
        return self.db.query(ModelConfigModel).filter(ModelConfigModel.id == config_id).first()

    def get_active_model_config(self, category: str) -> ModelConfigModel | None:
        """
        获取特定分类当前激活的配置
        """
        return self.db.query(ModelConfigModel).filter(
            ModelConfigModel.category == category,
            ModelConfigModel.isActive == 1
        ).first()

    def activate_model_config(self, config_id: str, category: str) -> bool:
        """
        激活指定的模型配置，并确保同分类下其他配置被置为未激活
        """
        # 1. 将同类别的所有配置设为未激活
        self.db.query(ModelConfigModel).filter(
            ModelConfigModel.category == category
        ).update({ModelConfigModel.isActive: 0})
        
        # 2. 将指定配置设为激活
        config = self.get_model_config(config_id)
        if config:
            config.isActive = 1
            self.db.commit()
            return True
        self.db.commit()
        return False

    def delete_model_config(self, config_id: str) -> bool:
        """
        删除模型配置
        """
        config = self.get_model_config(config_id)
        if config:
            self.db.delete(config)
            self.db.commit()
            return True
        return False

