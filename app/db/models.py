"""SQLite 元数据模型：知识库 / 文档 / 模型配置 / 会话 / 设置 / 评测报告。切片与向量在 prism.Index（每库一目录）。"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _uuid() -> str:
    return str(uuid.uuid4())


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(100), nullable=False)
    description = Column(String(1000), nullable=True)
    category = Column(String(100), nullable=True)      # 业务域：制度 / 产品 / 售后 / 运维 ...（查询路由用）
    language = Column(String(20), default="zh")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    documents = relationship("Document", back_populates="knowledge_base", cascade="all, delete-orphan")
    sessions = relationship("ChatSession", back_populates="knowledge_base", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=_uuid)
    kb_id = Column(String(36), ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)
    file_format = Column(String(20), nullable=True)
    status = Column(String(20), default="PENDING")     # PENDING / PROCESSING / COMPLETED / FAILED
    parser = Column(String(50), nullable=True)
    parser_tier = Column(String(20), nullable=True)
    markdown_content = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    parent_count = Column(Integer, default=0)
    entity_count = Column(Integer, default=0)
    error_message = Column(String(1000), nullable=True)
    title = Column(String(255), nullable=True)
    version = Column(String(50), nullable=True)
    valid_from = Column(String(20), nullable=True)
    valid_to = Column(String(20), nullable=True)
    tags = Column(String(500), nullable=True)          # 逗号分隔
    department = Column(String(100), nullable=True)
    access_level = Column(Integer, default=0)            # 0 公开 / 1 内部 / 2 机密 / 3 绝密
    allowed_departments = Column(String(500), nullable=True)   # 逗号分隔；空 = 不限部门
    auto_meta = Column(Text, nullable=True)            # 自动抽取的元数据 JSON：{field: {value, source, evidence, applied}}
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")


class ModelConfig(Base):
    __tablename__ = "model_configs"

    id = Column(String(36), primary_key=True, default=_uuid)
    provider_name = Column(String(100), nullable=False)
    provider_type = Column(String(30), nullable=False, default="openai_compatible")  # anthropic / openai_compatible / local / api
    base_url = Column(String(500), nullable=True)
    api_key = Column(String(500), nullable=True)
    category = Column(String(30), nullable=False, index=True)  # text / embedding / multimodal / rerank
    model_name = Column(String(255), nullable=False)
    is_active = Column(Integer, default=0, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=_uuid)
    kb_id = Column(String(36), ForeignKey("knowledge_bases.id"), nullable=True, index=True)
    title = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    knowledge_base = relationship("KnowledgeBase", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=_uuid)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    sources_json = Column(Text, nullable=True)
    relations_json = Column(Text, nullable=True)
    trace_json = Column(Text, nullable=True)
    citations_json = Column(Text, nullable=True)
    model = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class AppSetting(Base):
    __tablename__ = "app_settings"

    key = Column(String(100), primary_key=True)
    value_json = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EvalReportRow(Base):
    __tablename__ = "eval_reports"

    id = Column(String(36), primary_key=True, default=_uuid)
    kb_id = Column(String(36), nullable=False, index=True)
    name = Column(String(200), nullable=True)
    report_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
