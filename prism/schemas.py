"""全链路共享的数据模型。

设计要点：
- Block 是解析器输出的最小结构单元（标题 / 段落 / 表格 / 代码 ...），保留页码、bbox 与标题层级，
  这样切片阶段可以"按文档结构切"而不是"按固定字符切"。
- Chunk 分 parent / child 两种粒度（Small-to-Big）：child 用于向量与 BM25 检索，parent 交给 LLM 阅读。
- 每个 Chunk 都携带 breadcrumbs（文档 > 章节 > 小节）——这是性价比最高的一次性投入，
  即使块本身没有关键词，标题也能补上语义信号。
- RetrievalTrace 记录每一层看到了什么（查询变换 / 每路召回 / 融合 / 精排 / MMR / 织网 / 压缩），
  是可解释检索链路与错误归因的基础。
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    TITLE = "title"
    TEXT = "text"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    EQUATION = "equation"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    HEADER = "header"          # 页眉
    FOOTER = "footer"          # 页脚
    PAGE_NUMBER = "page_number"
    OTHER = "other"


# 参与正文切片的块类型；页眉页脚页码等在清洗阶段剔除
CONTENT_BLOCK_TYPES = {
    BlockType.TITLE,
    BlockType.TEXT,
    BlockType.LIST,
    BlockType.TABLE,
    BlockType.CODE,
    BlockType.EQUATION,
    BlockType.CAPTION,
    BlockType.FOOTNOTE,
}

# 不可拆分的原子块：切片时整体保留，避免表格 / 代码被拦腰斩断
ATOMIC_BLOCK_TYPES = {BlockType.TABLE, BlockType.CODE, BlockType.EQUATION}


class Block(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: BlockType = BlockType.TEXT
    text: str = ""
    level: int | None = None            # 标题层级（1 = H1），非标题为 None
    page: int | None = None             # 0-based 页码（MinerU 约定）
    bbox: list[float] | None = None     # 归一化坐标 [x0, y0, x1, y1]，0-1 或 0-1000
    meta: dict[str, Any] = Field(default_factory=dict)

    def is_content(self) -> bool:
        return self.type in CONTENT_BLOCK_TYPES and bool(self.text.strip())


class DocMetadata(BaseModel):
    """语料治理需要的"身份证"：来源、版本、时效、权限标签。"""

    source: str = ""                     # 原始文件路径或 URL
    file_format: str = ""                # pdf / md / xlsx ...
    parser: str = ""                     # mineru / markdown / excel ...
    parser_tier: str | None = None       # MinerU tier：flash / basic / standard / advanced
    title: str | None = None
    version: str | None = None
    valid_from: date | None = None
    valid_to: date | None = None
    department: str | None = None
    tags: list[str] = Field(default_factory=list)
    # 权限：access_level 0 公开 / 1 内部 / 2 机密 / 3 绝密；allowed_departments 为空表示不限部门
    access_level: int = 0
    allowed_departments: list[str] = Field(default_factory=list)
    language: str = "zh"
    created_at: datetime = Field(default_factory=datetime.now)
    extra: dict[str, Any] = Field(default_factory=dict)

    def is_effective(self, on: date | None = None) -> bool:
        """按生效区间过滤过期制度——"2024 版 vs 2026 版反复横跳"的解法。"""
        on = on or date.today()
        if self.valid_from and on < self.valid_from:
            return False
        if self.valid_to and on > self.valid_to:
            return False
        return True

    def is_visible_to(self, user: dict[str, Any] | None) -> bool:
        """权限过滤：用户级别 ≥ 文档级别，且（文档不限部门 / 用户属于允许的部门 / 用户是最高级别）。user=None 表示不做权限过滤。"""
        if user is None:
            return True
        level = int(user.get("level", 0) or 0)
        if level < self.access_level:
            return False
        if self.allowed_departments and level < ACCESS_ADMIN_LEVEL:
            return str(user.get("department") or "") in self.allowed_departments
        return True


class ParsedDocument(BaseModel):
    doc_id: str
    blocks: list[Block] = Field(default_factory=list)
    metadata: DocMetadata = Field(default_factory=DocMetadata)
    markdown: str | None = None          # 解析器给出的全文 Markdown（MinerU 原生支持）

    @staticmethod
    def make_doc_id(source: str) -> str:
        return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]

    @property
    def title(self) -> str:
        if self.metadata.title:
            return self.metadata.title
        for b in self.blocks:
            if b.type == BlockType.TITLE and b.text.strip():
                return b.text.strip()
        return self.metadata.source.rsplit("/", 1)[-1]

    def to_markdown(self) -> str:
        """没有原生 Markdown 时，从块重建一份（用于校对 / 展示）。"""
        if self.markdown:
            return self.markdown
        lines: list[str] = []
        for b in self.blocks:
            if b.type == BlockType.TITLE:
                lines.append("#" * max(1, min(6, b.level or 1)) + " " + b.text)
            elif b.type == BlockType.CODE:
                lines.append("```" + str(b.meta.get("language") or "") + "\n" + b.text + "\n```")
            elif b.type in (BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER):
                continue
            else:
                lines.append(b.text)
            lines.append("")
        return "\n".join(lines).strip() + "\n"


ACCESS_LEVELS = {0: "公开", 1: "内部", 2: "机密", 3: "绝密"}
ACCESS_ADMIN_LEVEL = 3

# parent / child：Small-to-Big 的阅读 / 检索粒度；section / sentence：多粒度分层索引的章节级 / 句子级单元
ChunkKind = Literal["parent", "child", "section", "sentence"]


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    kind: ChunkKind
    text: str                                    # 原文内容（交给 LLM 阅读 / 展示）
    parent_id: str | None = None                 # child -> parent
    child_ids: list[str] = Field(default_factory=list)
    order: int = 0                               # 在文档内的顺序，用于相邻块自动合并
    breadcrumbs: list[str] = Field(default_factory=list)   # [文档标题, H1, H2, ...]
    page_start: int | None = None
    page_end: int | None = None
    token_count: int = 0
    block_types: list[str] = Field(default_factory=list)
    context_summary: str | None = None           # Contextual Retrieval：LLM 生成的上下文说明
    hypothetical_questions: list[str] = Field(default_factory=list)  # 假设性问题（QA 对索引）
    summary: str | None = None                   # 摘要索引：parent 的摘要（单独一路召回，命中后返回原文）
    keywords: list[str] = Field(default_factory=list)   # 关键词扩展（corpus.keywords）
    aliases: list[str] = Field(default_factory=list)    # 别名扩展：正文出现了词典里某个词，同组的其余词
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def breadcrumb_text(self) -> str:
        return " > ".join(x for x in self.breadcrumbs if x)

    def embed_text(self) -> str:
        """真正进入向量 / BM25 索引的文本：面包屑 + 上下文说明 + 假设性问题 + 关键词 / 别名 + 正文。"""
        parts: list[str] = []
        if self.breadcrumbs:
            parts.append(f"[来源：{self.breadcrumb_text}]")
        if self.keywords:
            parts.append("关键词：" + "、".join(self.keywords))
        if self.aliases:
            parts.append("别名：" + "、".join(self.aliases))
        if self.context_summary:
            parts.append(self.context_summary)
        if self.hypothetical_questions:
            parts.append("相关问题：" + "；".join(self.hypothetical_questions))
        parts.append(self.text)
        return "\n".join(parts)


class ScoredChunk(BaseModel):
    """某一路召回 / 融合 / 精排的输出条目。"""

    chunk_id: str
    score: float
    rank: int
    route: str = ""                    # dense / sparse / fused / rerank / mmr


class WeavedRelation(BaseModel):
    """UnWeaver 织入的实体关系：检索块提到了实体 E，但定义 E 的块没被召回 → 自动拉取。"""

    entity: str
    relation_type: str = "DEFINED_IN"
    definition: str = ""
    source_chunk_id: str
    triggered_by: str = ""             # 触发织入的检索块 id
    target: str | None = None          # 三元组的宾语（RELATED_TO 时）


class HopTrace(BaseModel):
    hop: int
    query: str
    found_chunk_ids: list[str] = Field(default_factory=list)
    note: str = ""


class RetrievalTrace(BaseModel):
    """可解释检索链路：每一层看到了什么，方便错误归因。"""

    original_query: str
    strategy: str = "hybrid"                                   # hybrid / precise / multihop / agentic / none
    queries: list[str] = Field(default_factory=list)          # 改写 / 扩展后的所有查询（含原始）
    hyde_document: str | None = None
    routes: dict[str, list[ScoredChunk]] = Field(default_factory=dict)  # 每路 top-N
    fused: list[ScoredChunk] = Field(default_factory=list)
    reranked: list[ScoredChunk] = Field(default_factory=list)
    diversified: list[ScoredChunk] = Field(default_factory=list)  # MMR 之后
    weaved: list[WeavedRelation] = Field(default_factory=list)
    hops: list[HopTrace] = Field(default_factory=list)
    compression: dict[str, Any] = Field(default_factory=dict)    # {"tokens_before":..., "tokens_after":...}
    kb_route: list[dict[str, Any]] = Field(default_factory=list) # 查询路由结果
    # 自查询识别出的元数据条件：{"conditions", "source", "applied", "docs_before", "docs_after", "reason"}
    metadata_filter: dict[str, Any] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class Context(BaseModel):
    """最终送入 LLM 的上下文单元（通常是 parent chunk）。"""

    chunk: Chunk
    score: float
    matched_child_ids: list[str] = Field(default_factory=list)
    weaved: bool = False
    weaved_entity: str | None = None
    compressed_text: str | None = None

    def text_for_llm(self) -> str:
        return self.compressed_text if self.compressed_text else self.chunk.text

    def citation_label(self) -> str:
        bc = self.chunk.breadcrumb_text or self.chunk.doc_id
        pages = ""
        if self.chunk.page_start is not None:
            end = self.chunk.page_end if self.chunk.page_end is not None else self.chunk.page_start
            pages = f" p.{self.chunk.page_start + 1}" if end == self.chunk.page_start else f" p.{self.chunk.page_start + 1}-{end + 1}"
        return f"{bc}{pages}"


class RetrievalResult(BaseModel):
    query: str
    contexts: list[Context]
    trace: RetrievalTrace


class Answer(BaseModel):
    query: str
    text: str
    refused: bool = False
    citations: list[int] = Field(default_factory=list)   # 引用的上下文序号（1-based）
    contexts: list[Context] = Field(default_factory=list)
    trace: RetrievalTrace | None = None
    model: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
