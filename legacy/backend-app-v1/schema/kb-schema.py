from datetime import datetime
from pydantic import BaseModel

class KBCreate(BaseModel):
    """
    创建知识库请求 Schema
    """
    name: str
    description: str | None = None

class KBResponse(BaseModel):
    """
    知识库详情响应 Schema
    """
    id: str
    name: str
    description: str | None
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True
