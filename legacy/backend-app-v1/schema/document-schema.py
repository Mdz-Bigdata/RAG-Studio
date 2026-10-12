from datetime import datetime
from pydantic import BaseModel

class DocumentResponse(BaseModel):
    """
    文档详情响应 Schema
    """
    id: str
    knowledgeBaseId: str
    fileName: str
    fileSize: int | None
    status: str
    errorMessage: str | None
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True

class DocumentEditRequest(BaseModel):
    """
    文档 Markdown 手工编辑请求 Schema
    """
    markdownContent: str
