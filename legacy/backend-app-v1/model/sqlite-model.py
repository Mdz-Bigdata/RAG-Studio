from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class KnowledgeBaseModel(Base):
    """
    知识库模型
    """
    __tablename__ = "knowledge_bases"
    
    id = Column(String(36), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500), nullable=True)
    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    documents = relationship("DocumentModel", back_populates="knowledgeBase", cascade="all, delete-orphan")

class DocumentModel(Base):
    """
    文档模型 (支持 MinerU 解析)
    """
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, index=True)
    knowledgeBaseId = Column(String(36), ForeignKey("knowledge_bases.id"), nullable=False)
    fileName = Column(String(255), nullable=False)
    filePath = Column(String(500), nullable=True)
    fileSize = Column(Integer, nullable=True)
    status = Column(String(50), default="PENDING")  # PENDING, PROCESSING, COMPLETED, FAILED
    markdownContent = Column(Text, nullable=True)     # MinerU 解析出的结构化 markdown
    errorMessage = Column(String(500), nullable=True)
    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    knowledgeBase = relationship("KnowledgeBaseModel", back_populates="documents")
    chunks = relationship("ChunkModel", back_populates="document", cascade="all, delete-orphan")

class ChunkModel(Base):
    """
    文档切片模型
    """
    __tablename__ = "chunks"
    
    id = Column(String(36), primary_key=True, index=True)
    documentId = Column(String(36), ForeignKey("documents.id"), nullable=False)
    knowledgeBaseId = Column(String(36), nullable=False, index=True)
    content = Column(Text, nullable=False)
    chunkIndex = Column(Integer, nullable=False)
    qdrantPointId = Column(String(36), nullable=True)
    createdAt = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    document = relationship("DocumentModel", back_populates="chunks")

class EntityRelationModel(Base):
    """
    实体关系模型 (UnWeaver)
    """
    __tablename__ = "entity_relations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    entityName = Column(String(100), nullable=False, index=True)
    relationType = Column(String(50), nullable=False)  # DEFINED_IN, MENTIONED_IN
    definition = Column(Text, nullable=True)          # 实体的定义/描述
    chunkId = Column(String(36), ForeignKey("chunks.id"), nullable=False)
    knowledgeBaseId = Column(String(36), nullable=False, index=True)
    createdAt = Column(DateTime, default=datetime.utcnow)

class ModelConfigModel(Base):
    """
    大模型配置模型
    """
    __tablename__ = "model_configs"
    
    id = Column(String(36), primary_key=True, index=True)
    providerName = Column(String(100), nullable=False)
    baseUrl = Column(String(500), nullable=False)
    apiKey = Column(String(500), nullable=True)
    category = Column(String(50), nullable=False, index=True)  # text, embedding, multimodal, rerank
    modelName = Column(String(255), nullable=False)
    isActive = Column(DateTime, nullable=True)  # 若不为空，表示该配置已激活且为激活时间（方便多模型按激活时间排序）
    # 或者用 Boolean
    # 为了简单明了，我们直接用 isActive = Column(Integer, default=0) 0不激活，1激活
    isActive = Column(Integer, default=0, index=True)
    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

