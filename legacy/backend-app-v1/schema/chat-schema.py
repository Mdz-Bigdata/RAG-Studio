from pydantic import BaseModel

class ChatHistoryItem(BaseModel):
    """
    历史聊天记录单元
    """
    role: str  # user 或 assistant
    content: str

class ChatRequest(BaseModel):
    """
    问答对话请求 Schema
    """
    message: str
    history: list[ChatHistoryItem] = []
    imageB64: str | None = None
    tempDocContent: str | None = None

class SearchPlaygroundRequest(BaseModel):
    """
    检索游乐场调试请求 Schema
    """
    query: str
    topK: int = 5

class SearchPlaygroundResponse(BaseModel):
    """
    检索游乐场调试响应 Schema
    """
    denseResults: list[dict]
    bm25Results: list[dict]
    hydeResults: list[dict]
    rrfResults: list[dict]
    rerankedResults: list[dict]
    weavedRelations: list[dict]
    hydeDocument: str
