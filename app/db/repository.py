"""数据访问层。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from .models import AppSetting, ChatMessage, ChatSession, Document, EvalReportRow, KnowledgeBase, ModelConfig


class Repository:
    def __init__(self, db: Session) -> None:
        self.db = db

    # --- 知识库 ---------------------------------------------------------------------
    def create_kb(self, **fields: Any) -> KnowledgeBase:
        kb = KnowledgeBase(**fields)
        self.db.add(kb)
        self.db.commit()
        self.db.refresh(kb)
        return kb

    def get_kb(self, kb_id: str) -> KnowledgeBase | None:
        return self.db.get(KnowledgeBase, kb_id)

    def list_kbs(self) -> list[KnowledgeBase]:
        return self.db.query(KnowledgeBase).order_by(KnowledgeBase.created_at.asc()).all()

    def update_kb(self, kb: KnowledgeBase, **fields: Any) -> KnowledgeBase:
        for k, v in fields.items():
            if v is not None:
                setattr(kb, k, v)
        self.db.commit()
        self.db.refresh(kb)
        return kb

    def delete_kb(self, kb: KnowledgeBase) -> None:
        self.db.delete(kb)
        self.db.commit()

    # --- 文档 -----------------------------------------------------------------------
    def create_document(self, **fields: Any) -> Document:
        doc = Document(**fields)
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def get_document(self, doc_id: str) -> Document | None:
        return self.db.get(Document, doc_id)

    def list_documents(self, kb_id: str) -> list[Document]:
        return self.db.query(Document).filter(Document.kb_id == kb_id).order_by(Document.created_at.desc()).all()

    def update_document(self, doc: Document, **fields: Any) -> Document:
        for k, v in fields.items():
            setattr(doc, k, v)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def delete_document(self, doc: Document) -> None:
        self.db.delete(doc)
        self.db.commit()

    # --- 模型配置 -------------------------------------------------------------------
    def list_model_configs(self) -> list[ModelConfig]:
        return self.db.query(ModelConfig).order_by(ModelConfig.category, ModelConfig.created_at).all()

    def get_model_config(self, config_id: str) -> ModelConfig | None:
        return self.db.get(ModelConfig, config_id)

    def get_active_model_config(self, category: str) -> ModelConfig | None:
        return self.db.query(ModelConfig).filter(ModelConfig.category == category, ModelConfig.is_active == 1).first()

    def upsert_model_config(self, config_id: str | None, **fields: Any) -> ModelConfig:
        existing = self.get_model_config(config_id) if config_id else None
        if existing:
            for k, v in fields.items():
                if k == "api_key" and not v:
                    continue  # 空 / 脱敏值不覆盖旧 key
                setattr(existing, k, v)
            self.db.commit()
            self.db.refresh(existing)
            return existing
        cfg = ModelConfig(**({"id": config_id} if config_id else {}), **fields)
        self.db.add(cfg)
        self.db.commit()
        self.db.refresh(cfg)
        return cfg

    def activate_model_config(self, config: ModelConfig) -> None:
        self.db.query(ModelConfig).filter(ModelConfig.category == config.category).update({ModelConfig.is_active: 0})
        config.is_active = 1
        self.db.commit()

    def deactivate_model_config(self, config: ModelConfig) -> None:
        config.is_active = 0
        self.db.commit()

    def delete_model_config(self, config: ModelConfig) -> None:
        self.db.delete(config)
        self.db.commit()

    # --- 会话 -----------------------------------------------------------------------
    def create_session(self, kb_id: str | None, title: str | None) -> ChatSession:
        s = ChatSession(kb_id=kb_id, title=title)
        self.db.add(s)
        self.db.commit()
        self.db.refresh(s)
        return s

    def get_session(self, session_id: str) -> ChatSession | None:
        return self.db.get(ChatSession, session_id)

    def list_sessions(self, kb_id: str | None = None) -> list[ChatSession]:
        q = self.db.query(ChatSession)
        if kb_id:
            q = q.filter(ChatSession.kb_id == kb_id)
        return q.order_by(ChatSession.updated_at.desc()).all()

    def delete_session(self, s: ChatSession) -> None:
        self.db.delete(s)
        self.db.commit()

    def add_message(self, session: ChatSession, role: str, content: str, **extra: Any) -> ChatMessage:
        m = ChatMessage(session_id=session.id, role=role, content=content, **{k: (json.dumps(v, ensure_ascii=False) if k.endswith("_json") and not isinstance(v, str) else v) for k, v in extra.items()})
        self.db.add(m)
        session.updated_at = datetime.utcnow()
        if not session.title and role == "user":
            session.title = content[:40]
        self.db.commit()
        self.db.refresh(m)
        return m

    def list_messages(self, session_id: str) -> list[ChatMessage]:
        return self.db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc()).all()

    # --- 设置 -----------------------------------------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.db.get(AppSetting, key)
        return json.loads(row.value_json) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        row = self.db.get(AppSetting, key)
        if row:
            row.value_json = json.dumps(value, ensure_ascii=False)
        else:
            self.db.add(AppSetting(key=key, value_json=json.dumps(value, ensure_ascii=False)))
        self.db.commit()

    # --- 评测报告 -------------------------------------------------------------------
    def add_eval_report(self, kb_id: str, name: str | None, report: dict) -> EvalReportRow:
        row = EvalReportRow(kb_id=kb_id, name=name, report_json=json.dumps(report, ensure_ascii=False))
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    def list_eval_reports(self, kb_id: str, limit: int = 10) -> list[EvalReportRow]:
        return self.db.query(EvalReportRow).filter(EvalReportRow.kb_id == kb_id).order_by(EvalReportRow.created_at.desc()).limit(limit).all()
