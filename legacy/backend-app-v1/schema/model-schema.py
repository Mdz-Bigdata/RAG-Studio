from datetime import datetime
from pydantic import BaseModel

class ModelConfigSaveRequest(BaseModel):
    """
    保存/更新模型配置请求 Schema
    """
    id: str | None = None
    providerName: str
    baseUrl: str
    apiKey: str | None = None
    category: str  # text, embedding, multimodal, rerank
    modelName: str

class ModelConfigResponse(BaseModel):
    """
    模型配置详情响应 Schema (对 API Key 进行脱敏)
    """
    id: str
    providerName: str
    baseUrl: str
    category: str
    modelName: str
    isActive: int
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True

class ModelFetchRequest(BaseModel):
    """
    拉取服务商可用大模型列表请求 Schema
    """
    baseUrl: str
    apiKey: str | None = None

class ModelTestConnectionRequest(BaseModel):
    """
    测试配置连接可用性请求 Schema
    """
    providerName: str
    baseUrl: str
    apiKey: str | None = None
    category: str
    modelName: str
