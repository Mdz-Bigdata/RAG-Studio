from __future__ import annotations

import json

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from prism.ingest.parsers import ParserRegistry

from ..db import Document, Repository, get_db
from ..schemas import ChunkView, DocumentEditRequest, DocumentMarkdownResponse, DocumentMetaUpdate, DocumentResponse
from ..services import document_service, settings_service

router = APIRouter()


def doc_response(d: Document) -> DocumentResponse:
    return DocumentResponse(
        id=d.id, kbId=d.kb_id, fileName=d.file_name, fileSize=d.file_size, fileFormat=d.file_format, status=d.status, parser=d.parser, parserTier=d.parser_tier,
        chunkCount=d.chunk_count or 0, parentCount=d.parent_count or 0, entityCount=d.entity_count or 0, errorMessage=d.error_message, title=d.title, version=d.version,
        validFrom=d.valid_from, validTo=d.valid_to, tags=[t for t in (d.tags or "").split(",") if t],
        department=d.department, accessLevel=int(d.access_level or 0), allowedDepartments=[t for t in (d.allowed_departments or "").split(",") if t],
        autoMeta=_json(d.auto_meta), editBackups=document_service.count_edit_backups(d), createdAt=d.created_at, updatedAt=d.updated_at,
    )


def _json(raw: str | None) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except ValueError:
        return {}


@router.get("/supported-formats")
def supported_formats():
    reg = ParserRegistry(settings_service.get())
    return {"extensions": sorted(reg.supported_extensions()), "mineru": reg.use_mineru}


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def upload(kbId: str, background: BackgroundTasks, file: UploadFile = File(...), db: Session = Depends(get_db)):
    repo = Repository(db)
    if not repo.get_kb(kbId):
        raise HTTPException(404, "知识库不存在")
    if not file.filename:
        raise HTTPException(400, "文件名无效")
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    supported = ParserRegistry(settings_service.get()).supported_extensions()
    if ext not in supported:
        raise HTTPException(400, f"不支持的格式 .{ext}；支持：{', '.join(sorted(supported))}")
    doc = document_service.save_upload(kbId, file.filename, file.file)
    background.add_task(document_service.process, doc.id, reason="upload")
    return doc_response(doc)


@router.get("/", response_model=list[DocumentResponse])
def list_documents(kbId: str, db: Session = Depends(get_db)):
    return [doc_response(d) for d in Repository(db).list_documents(kbId)]


@router.get("/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: str, db: Session = Depends(get_db)):
    d = Repository(db).get_document(doc_id)
    if not d:
        raise HTTPException(404, "文档不存在")
    return doc_response(d)


@router.put("/{doc_id}/meta", response_model=DocumentResponse)
def update_meta(doc_id: str, body: DocumentMetaUpdate, background: BackgroundTasks, db: Session = Depends(get_db)):
    repo = Repository(db)
    d = repo.get_document(doc_id)
    if not d:
        raise HTTPException(404, "文档不存在")
    fields = {"title": body.title, "version": body.version, "valid_from": body.validFrom, "valid_to": body.validTo}
    if body.tags is not None:
        fields["tags"] = ",".join(body.tags)
    if body.department is not None:
        fields["department"] = body.department.strip() or None
    if body.accessLevel is not None:
        fields["access_level"] = max(0, min(3, body.accessLevel))
    if body.allowedDepartments is not None:
        fields["allowed_departments"] = ",".join(x.strip() for x in body.allowedDepartments if x.strip())
    repo.update_document(d, **{k: v for k, v in fields.items() if v is not None or k in ("department",)})
    if d.markdown_content:
        background.add_task(document_service.update_markdown, doc_id, d.markdown_content)
    return doc_response(d)


@router.get("/{doc_id}/markdown", response_model=DocumentMarkdownResponse)
def get_markdown(doc_id: str, db: Session = Depends(get_db)):
    d = Repository(db).get_document(doc_id)
    if not d:
        raise HTTPException(404, "文档不存在")
    return DocumentMarkdownResponse(id=d.id, fileName=d.file_name, markdownContent=d.markdown_content or "")


@router.put("/{doc_id}/markdown", response_model=DocumentResponse)
def update_markdown(doc_id: str, body: DocumentEditRequest, db: Session = Depends(get_db)):
    if not Repository(db).get_document(doc_id):
        raise HTTPException(404, "文档不存在")
    try:
        return doc_response(document_service.update_markdown(doc_id, body.markdownContent, reason="edit"))
    except Exception as exc:
        raise HTTPException(500, f"重新索引失败：{exc}") from exc


@router.get("/{doc_id}/chunks", response_model=list[ChunkView])
def chunks(doc_id: str, db: Session = Depends(get_db)):
    if not Repository(db).get_document(doc_id):
        raise HTTPException(404, "文档不存在")
    out = []
    for c in document_service.chunks(doc_id):
        out.append(ChunkView(chunkId=c["chunk_id"], kind=c["kind"], text=c["text"], breadcrumbs=c["breadcrumbs"], tokenCount=c["token_count"], pageStart=c["page_start"], pageEnd=c["page_end"], blockTypes=c["block_types"], childIds=c["child_ids"], parentId=c["parent_id"], contextSummary=c.get("context_summary"), hypotheticalQuestions=c.get("hypothetical_questions") or [], keywords=c.get("keywords") or [], aliases=c.get("aliases") or []))
    return out


class ReprocessRequest(BaseModel):
    discardEdits: bool = False           # 人工校对过的文档：true = 放弃校对、回到源文件重新解析（先备份校对版）


@router.post("/{doc_id}/reprocess", response_model=DocumentResponse)
def reprocess(doc_id: str, background: BackgroundTasks, body: ReprocessRequest | None = None, db: Session = Depends(get_db)):
    """用当前解析 / 切片配置重新处理源文件（切换 MinerU 档位、启用 Popo 文档树、调整切片参数后使用）。

    人工校对过的文档（parser 带 +edited）默认保留校对内容：用校对后的 Markdown 按当前清洗 / 切片 / 索引配置重建，
    不回到源文件——否则校对成果会被原文件静默覆盖。确实要换解析档位 / 用 Popo 文档树时传 discardEdits=true，
    会先把校对版备份到上传目录，再重新解析源文件。"""
    repo = Repository(db)
    d = repo.get_document(doc_id)
    if not d:
        raise HTTPException(404, "文档不存在")
    edited = "edited" in (d.parser or "") and bool(d.markdown_content)
    if edited and not (body and body.discardEdits):
        repo.update_document(d, status="PENDING", error_message=None)
        background.add_task(document_service.update_markdown, doc_id, d.markdown_content, reason="reprocess_keep_edits")
        return doc_response(d)
    if not d.file_path:
        raise HTTPException(400, "该文档没有源文件，无法重新解析")
    if edited:
        document_service.backup_edited_markdown(doc_id)
    repo.update_document(d, status="PENDING", error_message=None)
    background.add_task(document_service.process, doc_id, reason="reprocess_source" if edited else "reprocess")
    return doc_response(d)


@router.get("/{doc_id}/edit-backups")
def edit_backups(doc_id: str):
    """校对版备份列表（放弃校对重新解析、恢复另一份备份之前自动生成），新的在前。"""
    try:
        return document_service.edit_backups(doc_id)
    except KeyError:
        raise HTTPException(404, "文档不存在") from None


@router.post("/{doc_id}/edit-backups/{name}/restore", response_model=DocumentResponse)
def restore_edit_backup(doc_id: str, name: str, background: BackgroundTasks, db: Session = Depends(get_db)):
    """用一份校对版备份重建文档（等同于把它作为校对内容回灌）；当前若也是校对版且内容不同，先备份当前版本。"""
    repo = Repository(db)
    d = repo.get_document(doc_id)
    if not d:
        raise HTTPException(404, "文档不存在")
    try:
        markdown = document_service.read_edit_backup(doc_id, name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KeyError:
        raise HTTPException(404, "备份不存在") from None
    db.refresh(d)
    repo.update_document(d, status="PENDING", error_message=None)
    background.add_task(document_service.update_markdown, doc_id, markdown, reason="restore_backup")
    return doc_response(d)


@router.delete("/{doc_id}/edit-backups/{name}")
def delete_edit_backup(doc_id: str, name: str):
    try:
        document_service.delete_edit_backup(doc_id, name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KeyError:
        raise HTTPException(404, "备份不存在") from None
    return {"ok": True}


@router.get("/{doc_id}/popo")
def popo_status(doc_id: str, db: Session = Depends(get_db)):
    """MinerU-Popo 状态：是否已导出输入 / 已推理 / 文档树就绪 / 当前索引是否在用文档树。"""
    if not Repository(db).get_document(doc_id):
        raise HTTPException(404, "文档不存在")
    return document_service.popo_status(doc_id)


@router.post("/{doc_id}/popo-export")
def popo_export(doc_id: str, db: Session = Depends(get_db)):
    """MinerU 页面级解析 → 标签归一化，写出 MinerU-Popo 推理所需输入。"""
    if not Repository(db).get_document(doc_id):
        raise HTTPException(404, "文档不存在")
    try:
        return document_service.popo_export(doc_id)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"导出失败：{exc}") from exc


@router.delete("/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    if not Repository(db).get_document(doc_id):
        raise HTTPException(404, "文档不存在")
    document_service.delete(doc_id)
    return {"message": "文档、切片、向量与实体已删除"}


@router.post("/parse-temp")
def parse_temp(file: UploadFile = File(...)):
    """临时提取上传文档的文本（不入库、不建向量），仅供单次会话分析。"""
    if not file.filename:
        raise HTTPException(400, "文件名无效")
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "txt"
    supported = ParserRegistry(settings_service.get()).supported_extensions() | {"txt", "md", "markdown", "log"}
    if ext not in supported:
        raise HTTPException(400, f"不支持的格式 .{ext}")
    try:
        text = document_service.parse_temp(file.filename, file.file.read())
    except Exception as exc:
        raise HTTPException(500, f"临时解析失败：{exc}") from exc
    return {"content": text, "fileName": file.filename, "chars": len(text)}
