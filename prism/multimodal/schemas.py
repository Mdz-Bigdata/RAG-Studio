"""多模态数据模型：图片记录、检索命中、检索结果。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..schemas import Context, ScoredChunk

SearchMode = Literal["text2image", "image2image", "hybrid", "image2text", "text2video"]


def format_ts(seconds: float | None) -> str:
    """秒 → mm:ss（满一小时 h:mm:ss）。舍去小数，与播放器的读数一致。"""
    if seconds is None:
        return ""
    s = max(0, int(seconds))
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


class ImageRecord(BaseModel):
    """知识库里的一张图片：像素文件 + 位置 + 三种文本信号（图注 / OCR / VLM 描述）。"""

    image_id: str                        # <doc_id>_img<NNN>
    doc_id: str
    file: str                            # 相对索引目录的路径：images/<doc_id>/<NNN>.png
    sha1: str
    width: int
    height: int
    order: int = 0                       # 在文档内的出现顺序
    page: int | None = None              # 0-based 页码 / 幻灯片序号
    bbox: list[float] | None = None
    origin: str = ""                     # pdf / docx / pptx / markdown / mineru / file / video
    # 视频关键帧：帧所在时刻与它代表的片段（镜头）起止，单位秒；普通图片都是 None
    time: float | None = None
    time_start: float | None = None
    time_end: float | None = None
    shot: int | None = None              # 镜头序号：同一镜头的相邻片段在文搜视频里可以连成一段，跨镜头不连
    doc_title: str = ""
    caption: str = ""                    # 图注 / alt 文本
    ocr_text: str = ""                   # 图中文字（精确数值 / 标签以它为准）
    description: str = ""                # VLM 生成的语义描述
    context_text: str = ""               # 图片附近的正文
    breadcrumbs: list[str] = Field(default_factory=list)
    anchor_chunk_id: str | None = None   # 图片所在的 parent 块（图文关联）

    @property
    def breadcrumb_text(self) -> str:
        return " > ".join(x for x in self.breadcrumbs if x)

    @property
    def is_video(self) -> bool:
        return self.time is not None

    def span_text(self) -> str:
        """视频片段的时间范围，如 01:05–01:12。"""
        if not self.is_video:
            return ""
        return f"{format_ts(self.time_start if self.time_start is not None else self.time)}–{format_ts(self.time_end if self.time_end is not None else self.time)}"

    def label(self) -> str:
        if self.is_video:
            return f"{self.doc_title or self.doc_id} {self.span_text()}"
        page = f" p.{self.page + 1}" if self.page is not None else ""
        return f"{self.breadcrumb_text or self.doc_title or self.doc_id}{page}"

    def search_text(self) -> str:
        """进入描述文本索引（向量 + BM25）的"图片文本摘要"：位置 + 图注 + 语义描述 + 图中文字 + 上下文。"""
        parts: list[str] = []
        if self.is_video:
            parts.append(f"[视频画面：{self.label()}]")
        elif self.breadcrumbs or self.doc_title:
            parts.append(f"[图片来源：{self.label()}]")
        if self.caption:
            parts.append(f"图注：{self.caption}")
        if self.description:
            parts.append(f"图片内容描述：{self.description}")
        if self.ocr_text:
            parts.append(f"图片中文字：{self.ocr_text}")
        if self.context_text:
            parts.append(f"上下文：{self.context_text}")
        return "\n".join(parts)

    def brief(self, max_chars: int = 160) -> str:
        text = self.caption or self.description or self.ocr_text or self.context_text or self.doc_title
        text = " ".join(text.split())
        return text[:max_chars] + ("…" if len(text) > max_chars else "")


class RouteHit(BaseModel):
    rank: int
    score: float


class ImageHit(BaseModel):
    record: ImageRecord
    score: float                         # 融合后归一化到 (0, 1]；单路图搜图时为余弦相似度
    rank: int = 0
    routes: dict[str, RouteHit] = Field(default_factory=dict)
    linked: bool = False                 # 因为所在段落被文本检索命中而带出（图文关联）


class VideoSegment(BaseModel):
    """文搜视频的一条结果：同一视频里一段连续的命中片段（相邻命中镜头合并），附代表帧。"""

    doc_id: str
    start: float
    end: float
    time: float                          # 代表帧（得分最高的关键帧）的时刻：播放器从这里定位
    score: float
    rank: int = 0
    hits: list[ImageHit] = Field(default_factory=list)   # 片段里命中的关键帧，按时间顺序


class MultimodalResult(BaseModel):
    mode: SearchMode
    query: str = ""
    has_image: bool = False
    images: list[ImageHit] = Field(default_factory=list)
    contexts: list[Context] = Field(default_factory=list)          # 图搜文的结果
    segments: list[VideoSegment] = Field(default_factory=list)     # 文搜视频的结果
    routes: dict[str, list[ScoredChunk]] = Field(default_factory=dict)  # 每路排名（chunk_id 为 image_id 或 child id）
    fused: list[ScoredChunk] = Field(default_factory=list)
    query_description: str = ""          # 查询图片的 OCR / VLM 描述
    ocr_text: str = ""                   # 其中的图中文字（原样）与 VLM 描述分开留一份：同一张图要重做检索时（问答换库）直接沿用
    caption: str = ""
    notes: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    embedder: dict[str, Any] = Field(default_factory=dict)
