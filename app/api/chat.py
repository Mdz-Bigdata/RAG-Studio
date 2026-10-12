from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..db import ChatMessage, ChatSession, Repository, get_db
from ..schemas import ChatMessageResponse, ChatSessionResponse, ChatStreamRequest, SessionCreate
from ..services import chat_service

router = APIRouter()


def _session(s: ChatSession, n: int) -> ChatSessionResponse:
    return ChatSessionResponse(id=s.id, kbId=s.kb_id, title=s.title, createdAt=s.created_at, updatedAt=s.updated_at, messageCount=n)


def _message(m: ChatMessage) -> ChatMessageResponse:
    load = lambda v, d: (json.loads(v) if v else d)
    return ChatMessageResponse(id=m.id, role=m.role, content=m.content, sources=load(m.sources_json, []), relations=load(m.relations_json, []), citations=load(m.citations_json, []), trace=load(m.trace_json, None), model=m.model, createdAt=m.created_at)


@router.get("/sessions", response_model=list[ChatSessionResponse])
def list_sessions(kbId: str | None = None, db: Session = Depends(get_db)):
    repo = Repository(db)
    return [_session(s, len(s.messages)) for s in repo.list_sessions(kbId)]


@router.post("/sessions", response_model=ChatSessionResponse)
def create_session(body: SessionCreate, db: Session = Depends(get_db)):
    s = Repository(db).create_session(body.kbId, body.title)
    return _session(s, 0)


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
def messages(session_id: str, db: Session = Depends(get_db)):
    repo = Repository(db)
    if not repo.get_session(session_id):
        raise HTTPException(404, "会话不存在")
    return [_message(m) for m in repo.list_messages(session_id)]


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    repo = Repository(db)
    s = repo.get_session(session_id)
    if not s:
        raise HTTPException(404, "会话不存在")
    repo.delete_session(s)
    return {"message": "会话已删除"}


@router.post("/stream")
def stream(body: ChatStreamRequest, db: Session = Depends(get_db)):
    """SSE：meta → plan → sources → relations → images → message* → done。"""
    repo = Repository(db)
    if body.kbId and body.kbId != "auto" and not repo.get_kb(body.kbId):
        raise HTTPException(404, "知识库不存在")
    if not body.message.strip() and not body.imageB64:
        raise HTTPException(400, "消息不能为空")
    gen = chat_service.stream(kb_id=body.kbId, session_id=body.sessionId, message=body.message, image_b64=body.imageB64, temp_doc_content=body.tempDocContent, mode=body.mode, overrides=body.overrides, top_k=body.topK, user=body.user.model_dump() if body.user else None)
    return StreamingResponse(gen, media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
