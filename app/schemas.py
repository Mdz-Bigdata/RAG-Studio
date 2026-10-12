"""API 请求 / 响应模型（camelCase，供前端直接消费）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from prism.training import MiningConfig


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- 知识库 ---------------------------------------------------------------------------
class KBCreate(BaseModel):
    name: str
    description: str | None = None
    category: str | None = None
    language: str | None = "zh"


class KBUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    category: str | None = None
    language: str | None = None


class KBResponse(BaseModel):
    id: str
    name: str
    description: str | None
    category: str | None
    language: str | None
    createdAt: datetime
    updatedAt: datetime
    documentCount: int = 0
    chunkCount: int = 0
    entityCount: int = 0
    indexEmbedder: str | None = None
    needsReindex: bool = False
    reindexing: bool = False


# --- 文档 -----------------------------------------------------------------------------
class DocumentResponse(BaseModel):
    id: str
    kbId: str
    fileName: str
    fileSize: int | None
    fileFormat: str | None
    status: str
    parser: str | None
    parserTier: str | None
    chunkCount: int
    parentCount: int
    entityCount: int
    errorMessage: str | None
    title: str | None
    version: str | None
    validFrom: str | None
    validTo: str | None
    tags: list[str]
    department: str | None = None
    accessLevel: int = 0
    allowedDepartments: list[str] = Field(default_factory=list)
    autoMeta: dict[str, Any] = Field(default_factory=dict)   # 自动抽取的元数据：{field: {value, source, evidence, applied}}
    editBackups: int = 0                 # 校对版备份数（放弃校对 / 恢复前自动备份），> 0 时可「恢复校对版」
    createdAt: datetime
    updatedAt: datetime


class DocumentMarkdownResponse(BaseModel):
    id: str
    fileName: str
    markdownContent: str


class DocumentEditRequest(BaseModel):
    markdownContent: str


class DocumentMetaUpdate(BaseModel):
    title: str | None = None
    version: str | None = None
    validFrom: str | None = None
    validTo: str | None = None
    tags: list[str] | None = None
    department: str | None = None
    accessLevel: int | None = None
    allowedDepartments: list[str] | None = None


class ChunkView(BaseModel):
    chunkId: str
    kind: str
    text: str
    breadcrumbs: list[str]
    tokenCount: int
    pageStart: int | None
    pageEnd: int | None
    blockTypes: list[str]
    childIds: list[str] = Field(default_factory=list)
    parentId: str | None = None
    contextSummary: str | None = None
    hypotheticalQuestions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


# --- 检索 -----------------------------------------------------------------------------
RetrievalMode = Literal["hybrid", "multihop", "agentic"]


class UserIdentity(BaseModel):
    """检索时的用户身份（权限过滤用）。本项目没有登录体系，由前端"当前身份"切换器传入。"""

    level: int = 3
    department: str | None = None
    userId: str | None = None            # 长期记忆 / 个性化推荐按它区分用户（为空则用 "default"）


class RetrievalRequest(BaseModel):
    query: str
    mode: RetrievalMode = "hybrid"
    topK: int | None = None
    useTransform: bool = True
    overrides: dict[str, Any] | None = None    # prism.config.RetrievalConfig 的部分覆盖
    filters: dict[str, Any] | None = None
    user: UserIdentity | None = None


class ScoredItem(BaseModel):
    chunkId: str
    score: float
    rank: int
    route: str
    text: str
    breadcrumb: str
    docId: str
    fileName: str | None = None


class ContextView(BaseModel):
    chunkId: str
    docId: str
    fileName: str | None
    breadcrumb: str
    text: str
    compressedText: str | None
    score: float
    tokenCount: int
    pageStart: int | None
    pageEnd: int | None
    matchedChildIds: list[str]
    weaved: bool
    weavedEntity: str | None


class RetrievalResponse(BaseModel):
    query: str
    strategy: str
    queries: list[str]
    hydeDocument: str | None
    notes: list[str]
    routes: dict[str, list[ScoredItem]]
    fused: list[ScoredItem]
    reranked: list[ScoredItem]
    diversified: list[ScoredItem]
    contexts: list[ContextView]
    weaved: list[dict[str, Any]]
    hops: list[dict[str, Any]]
    compression: dict[str, Any]
    metadataFilter: dict[str, Any] | None = None   # 自查询识别出的元数据条件及是否生效
    timingsMs: dict[str, float]
    funnel: dict[str, int]


class MultimodalSearchRequest(BaseModel):
    mode: Literal["text2image", "image2image", "hybrid", "image2text"] = "text2image"
    query: str | None = None
    imageB64: str | None = None              # data URL 或裸 base64
    topK: int | None = None
    alpha: float | None = None               # hybrid：文本权重
    minScore: float | None = None            # image2image：余弦相似度阈值
    docIds: list[str] | None = None


class RouteRequest(BaseModel):
    query: str
    topN: int = 1
    mode: Literal["embedding", "llm"] = "embedding"


# --- 对话 -----------------------------------------------------------------------------
class ChatSessionResponse(BaseModel):
    id: str
    kbId: str | None
    title: str | None
    createdAt: datetime
    updatedAt: datetime
    messageCount: int = 0


class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[int] = Field(default_factory=list)
    trace: dict[str, Any] | None = None      # assistant：检索链路 + images / imageCitations；user：queryImage
    model: str | None = None
    createdAt: datetime


class ChatStreamRequest(BaseModel):
    kbId: str | None = None                  # None 或 "auto" 时启用查询路由
    sessionId: str | None = None
    message: str = ""                        # 只发图片（以图搜图 / 以图搜文）时可以为空
    imageB64: str | None = None
    tempDocContent: str | None = None
    mode: RetrievalMode = "hybrid"
    overrides: dict[str, Any] | None = None
    topK: int | None = None
    user: UserIdentity | None = None


class SessionCreate(BaseModel):
    kbId: str | None = None
    title: str | None = None


# --- 模型配置 -------------------------------------------------------------------------
class ModelConfigSave(BaseModel):
    id: str | None = None
    providerName: str
    providerType: Literal["anthropic", "openai_compatible", "local", "api"] = "openai_compatible"
    baseUrl: str | None = None
    apiKey: str | None = None
    category: Literal["text", "embedding", "multimodal", "rerank", "image_embedding"]
    modelName: str
    keyFromConfigId: str | None = None    # apiKey 为空时复用这条已保存配置的 Key（同一服务商配置多个用途不用重复填）


class ModelConfigResponse(BaseModel):
    id: str
    providerName: str
    providerType: str
    baseUrl: str | None
    apiKeyMasked: str
    category: str
    modelName: str
    isActive: int
    createdAt: datetime
    updatedAt: datetime


class ModelFetchRequest(BaseModel):
    baseUrl: str
    apiKey: str | None = None
    providerType: str = "openai_compatible"
    keyFromConfigId: str | None = None    # apiKey 为空时复用这条已保存配置的 Key


class ModelTestRequest(BaseModel):
    configId: str | None = None
    providerName: str | None = None
    providerType: str = "openai_compatible"
    baseUrl: str | None = None
    apiKey: str | None = None
    category: str
    modelName: str


# --- 设置 -----------------------------------------------------------------------------
class SettingsUpdate(BaseModel):
    overrides: dict[str, Any]


# --- 评测 -----------------------------------------------------------------------------
class EvalExampleIn(BaseModel):
    question: str
    relevant_doc_ids: list[str] = Field(default_factory=list)
    relevant_sources: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    answer_keywords: list[str] = Field(default_factory=list)
    category: str = "常见问题"
    expected_refusal: bool = False


class EvalRunRequest(BaseModel):
    examples: list[EvalExampleIn] | None = None
    useSample: bool = False
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    name: str | None = None
    overrides: dict[str, Any] | None = None


class HardNegativeMiningRequest(BaseModel):
    examples: list[EvalExampleIn] = Field(min_length=1, max_length=500)
    heldOut: list[EvalExampleIn] = Field(default_factory=list, max_length=500)
    config: MiningConfig = Field(default_factory=MiningConfig)


class MultimodalEvalRequest(BaseModel):
    """examples 为空或 auto=True：从该知识库的图片库自动生成样例（自检）。"""

    examples: list[dict[str, Any]] | None = None
    auto: bool = False
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5])
    name: str | None = None


class CoverageRequest(BaseModel):
    """语料准备：意图空间（用户会问什么）。每项是问题文本，或 {"question", "category"}。"""

    intents: list[str | dict[str, Any]] | None = None
    useSample: bool = False
    high: float = 0.6        # ≥ high → 高匹配
    partial: float = 0.35    # ≥ partial → 部分匹配；否则无匹配


class VectorBenchmarkRequest(BaseModel):
    pcaDim: int | None = None
    quantize: Literal["none", "int8"] | None = None
    rescore: bool | None = None
    oversample: float | None = None
    queries: int = 30
    topK: int = 10
