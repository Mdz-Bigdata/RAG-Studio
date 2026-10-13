"""配置模型：config.yaml + 环境变量 / 数据库覆盖。

所有数值默认值都来自实践经验（见 docs/RAG_OPTIMIZATION_SUMMARY.md 的"参数速查"）：
- child 150 token / parent 500-800 token / overlap 10%-15%
- 漏斗式管理：每路召回 top-N = 50 → RRF 候选 ≥ 50 → Reranker 保留 10 → 最终上下文 3-5
- RRF k = 60；MMR λ = 0.7
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    index_dir: Path = Path("data/index")
    mineru_out_dir: Path = Path("data/mineru_out")
    popo_dir: Path = Path("data/popo")
    upload_dir: Path = Path("data/uploads")
    lancedb_dir: Path = Path("data/lancedb")     # 全局 LanceDB：记忆 / 推荐 / 多智能体黑板 / 多模态表（知识库向量在各自索引目录）


MINERU_CLOUD_API_URL = "https://mineru.net/api"


class MinerUCloudConfig(BaseModel):
    """MinerU 官方云端 API（mineru.net，v1 协议，与自托管 `mineru-kit api-server` 同一套接口）。"""

    api_url: str = MINERU_CLOUD_API_URL
    api_key: str = ""                    # 留空读环境变量 MINERU_API_KEY；两者都没有时以匿名身份调用（有页数 / 频率限制）


class MinerUConfig(BaseModel):
    enabled: bool = True
    # 解析在哪里跑：local 本机 SDK（要下载模型包）| server 自托管 mineru-kit api-server（api_url）| cloud 官方云端 API（cloud.*）
    # 云端模式会把文档上传到 MinerU 服务器；本机不需要模型包和内存，适合没有 GPU 的小机器用 standard 档
    mode: Literal["local", "server", "cloud"] = "local"
    # flash: 原生文本无推理；basic: 小模型 OCR/公式/表格（CPU 可跑）；standard: 小模型 + VLM（默认）；advanced: 更多算力
    # 不用 Literal：自托管解析服务（/v1/tiers）可能提供目录之外的档位
    tier: str = "standard"
    ocr_mode: Literal["auto", "txt", "ocr"] = "auto"
    image_analysis: bool = True
    page_range: str = ""                 # 仅 PDF，"1-5,r3-r1"，空 = 全部
    api_url: str | None = None           # 连接自托管 `mineru-kit api-server` 时填写，否则本地 SDK
    api_key: str = ""
    save_outputs: bool = True            # 落盘 markdown / middle_json / structured_content 便于复用与 Popo
    model_source: Literal["huggingface", "modelscope"] | None = None   # 模型下载源；国内用 modelscope（MINERU_MODEL_SOURCE）
    cloud: MinerUCloudConfig = MinerUCloudConfig()

    @model_validator(mode="before")
    @classmethod
    def _infer_mode(cls, data: Any) -> Any:
        """旧配置没有 mode：只要填了 api_url 就是连自托管服务。"""
        if isinstance(data, dict) and "mode" not in data and data.get("api_url"):
            return {**data, "mode": "server"}
        return data

    def remote_endpoint(self) -> tuple[str, str] | None:
        """远程解析的（地址, 密钥）；本机解析返回 None。server 模式没填地址时无处可连，按本机处理。"""
        if self.mode == "cloud":
            return (self.cloud.api_url or MINERU_CLOUD_API_URL).rstrip("/"), self.cloud.api_key or os.environ.get("MINERU_API_KEY", "")
        if self.mode == "server" and self.api_url:
            return self.api_url.rstrip("/"), self.api_key
        return None

    @property
    def resolved_mode(self) -> Literal["local", "server", "cloud"]:
        """实际生效的模式（server 没填地址 → local）。"""
        return self.mode if self.remote_endpoint() is not None else "local"


class PopoConfig(BaseModel):
    """MinerU-Popo：页面级解析 → 文档级语义树（跨页表格 / 段落续接 / 标题层级 / 图文关联）。"""

    enabled: bool = False
    repo_dir: Path | None = None         # MinerU-Popo 仓库路径（git clone）
    model_path: str = "models/MinerU-Popo"
    backend: Literal["transformers", "vllm"] = "transformers"
    use_tree_hierarchy: bool = True      # 用 Popo 树重建标题层级替代 MinerU 的 text_level


class ParsingConfig(BaseModel):
    mineru: MinerUConfig = MinerUConfig()
    popo: PopoConfig = PopoConfig()
    markdown_first: bool = True          # .md 文件永远走 Markdown 解析器（必做）


class CleaningConfig(BaseModel):
    drop_headers_footers: bool = True
    header_footer_min_page_ratio: float = 0.4    # 同一行出现在 ≥40% 页面视为页眉/页脚
    fix_pdf_linebreaks: bool = True
    dedup_near_duplicates: bool = True
    dedup_jaccard_threshold: float = 0.9
    garbled_ratio_threshold: float = 0.3        # 不可打印/乱码字符占比超过则丢弃
    min_block_chars: int = 2


class ChunkingConfig(BaseModel):
    strategy: Literal["structural", "semantic", "recursive"] = "structural"
    parent_max_tokens: int = 600
    parent_min_tokens: int = 120
    child_tokens: int = 150
    child_overlap_ratio: float = 0.12
    respect_atomic_blocks: bool = True          # 表格 / 代码不拆
    table_row_split_tokens: int = 800           # 超长表格按行拆分，每块重复表头
    semantic_breakpoint_percentile: float = 80.0  # 语义切块：相邻句距离 > P80 视为话题转折
    semantic_min_sentences: int = 2
    # 多粒度分层索引：在 parent / child 之外再建"章节级"与"句子级"两层，检索时按问题类型选粒度
    multi_granularity: bool = False
    section_max_tokens: int = 1500              # 章节级单元的上限（超长章节截断，只影响章节级索引）
    section_depth: int = 1                      # 章节按第几级标题归并：1 = 一级标题（"假期管理"下的年假 / 调休 / 病假合成一章）
    sentence_min_chars: int = 8                 # 句子级：过短的句子并入前一句


class EnrichConfig(BaseModel):
    contextual_summary: bool = False     # Anthropic Contextual Retrieval（每块调用一次 LLM）
    hypothetical_questions: bool = False # 为每个 parent 生成 N 个假设性问题，构建 QA 对索引
    questions_per_chunk: int = 3
    # 摘要索引：为每个 parent 生成摘要，单独建一路索引；命中摘要返回原文。无 LLM 时退化为抽取式摘要（标题 + 首句）
    summary_index: bool = False
    summary_max_chars: int = 200
    max_concurrency: int = 4


class CorpusConfig(BaseModel):
    """语料优化与增强（prism/corpus）：关键词扩展、自动元数据、质量体检阈值。
    知识库级的别名词典 / 剔除清单 / FAQ 增强存在应用数据库里，不在这里。"""

    keywords: bool = False               # 入库时为每个段落抽取关键词（TF-IDF + 编号保护），写进向量与 BM25 文本；不需要 LLM
    keywords_per_chunk: int = 8
    auto_metadata: bool = True           # 上传解析后自动抽取标题 / 版本 / 生效日期 / 部门 / 标签 / 密级，只填空字段
    auto_metadata_llm: bool = False      # 规则没找到的字段再让文本模型补（结果需能在原文里找到依据）
    near_dup_threshold: float = 0.85     # 质量体检：近似重复的 Jaccard 阈值（字符 5-gram）
    garbled_threshold: float = 0.15      # 质量体检：异常字符占比阈值


class EmbeddingConfig(BaseModel):
    provider: Literal["hash", "sentence_transformers", "openai_compatible"] = "sentence_transformers"
    model: str = "BAAI/bge-m3"
    dim: int = 1024                      # hash provider 使用；真实模型以输出为准
    batch_size: int = 32
    normalize: bool = True
    query_instruction: str = ""          # bge-m3 无需 instruction；bge-large-zh 用 "为这个句子生成表示以用于检索相关文章："
    base_url: str | None = None          # openai_compatible
    api_key_env: str = "EMBEDDING_API_KEY"
    api_key: str | None = None           # 直接给出 key（来自模型配置中心）时优先
    device: str | None = None            # cpu / cuda / mps，None = 自动


class VectorSearchConfig(BaseModel):
    """查询期的向量检索调优（不需要重建索引）：降维 / 量化换速度与内存，精排回算保精度。"""

    pca_dim: int | None = None           # numpy 后端：PCA 降到该维度做粗筛（None = 不降维）
    quantize: Literal["none", "int8"] = "none"   # numpy 后端：粗筛矩阵 int8 量化（内存约 1/4）
    rescore: bool = True                 # 粗筛后用原始全精度向量重新打分（召回损失接近 0）
    oversample: float = 4.0              # 粗筛取 top_n × oversample 个候选再精算
    ef_search: int = 128                 # qdrant / lancedb(HNSW) 后端：HNSW 搜索宽度，越大越准越慢
    nprobes: int = 20                    # lancedb 后端：IVF 索引查询时探查的分区数，越大越准越慢
    refine_factor: int | None = 5        # lancedb 后端：量化索引多取 k×refine 条再用原始向量重排（None = 不重排）


class LanceDBConfig(BaseModel):
    """LanceDB 后端的建索引期参数（改了会触发重建）。查询期参数（nprobes / refine_factor / ef）在 vector_store.search。"""

    index_type: Literal["auto", "none", "IVF_HNSW_SQ", "IVF_PQ", "IVF_FLAT"] = "auto"
    index_min_rows: int = 20000          # auto：行数达到才建 ANN 索引（IVF_HNSW_SQ）；不到就精确扫描，召回 100%
    fts_tokenizer: Literal["icu", "ngram", "whitespace"] = "icu"   # 原生全文索引分词：icu 自带中文分词、无需下载模型
    optimize_every: int = 20             # 每 N 次写入做一次 optimize：合并小文件 + 把新数据并入索引
    keep_versions_days: int = 7          # optimize 时清理多少天以前的旧版本（时间旅行可回看的范围）


class VectorStoreConfig(BaseModel):
    backend: Literal["numpy", "chroma", "qdrant", "lancedb"] = "numpy"
    lancedb: LanceDBConfig = LanceDBConfig()
    qdrant_url: str | None = None        # 远程 Qdrant；为空则用本地文件模式
    qdrant_path: Path | None = None      # 本地持久化路径；两者都空 = 内存模式
    collection_prefix: str = "rag_children"
    # 以下为建索引期参数（改了需要重建）：qdrant 的 HNSW 图与量化
    hnsw_m: int = 16                     # 每个节点的邻居数：越大召回越高、内存越大
    hnsw_ef_construct: int = 100         # 建图时的搜索宽度
    quantization: Literal["none", "int8", "binary"] = "none"   # qdrant 标量 / 二值量化（查询时自动 rescore）
    search: VectorSearchConfig = VectorSearchConfig()


class SparseConfig(BaseModel):
    k1: float = 1.5                      # 长文档场景可调到 2.0
    b: float = 0.75                      # 文档长度均匀可调到 0.5
    use_jieba: bool = True
    user_dict: Path | None = None        # 领域词典，提升专有名词切分质量
    exact_token_boost: float = 0.0       # >0 时，对含精确 token（编号 / 错误码）的块加分兜底


class FusionConfig(BaseModel):
    method: Literal["rrf", "convex"] = "rrf"
    rrf_k: int = 60
    weight_dense: float = 1.0
    weight_sparse: float = 1.0
    convex_alpha: float = 0.5            # convex: alpha * dense + (1-alpha) * sparse（min-max 归一化后）


class RerankConfig(BaseModel):
    enabled: bool = True
    # auto：显式配置的 API（api_url）→ 本地交叉编码器（装了 torch + transformers）→ LLM 打分 → 词法。
    # cross_encoder：本地模型（model，默认 BAAI/bge-reranker-v2-m3）；cohere：Cohere Rerank v2；api：Jina / SiliconFlow 等 /rerank 端点
    provider: Literal["auto", "cross_encoder", "cohere", "api", "llm", "lexical", "none"] = "auto"
    model: str = "BAAI/bge-reranker-v2-m3"   # 本地交叉编码器（也是通用 api provider 的模型名）
    cohere_model: str = "rerank-v4.0-pro"    # cohere provider：rerank-v4.0-pro / rerank-v4.0-fast / rerank-v3.5 / rerank-multilingual-v3.0
    auto_download: bool = True           # 本地模型权重不在本机时后台下载（约 2.3GB），期间用 LLM / 词法精排顶上
    candidates: int = 50                 # 粗召回池 recall_top_k：传给 Reranker 的候选数，低于 50 效果断崖式下跌
    keep_n: int = 10                     # 重排保留 rerank_top_n
    top_k: int = 5                       # 最终上下文 context_top_n
    min_score: float | None = None       # 归一化分数阈值，低于则认为"库里没有答案"
    # 精排全部低于 min_score 时的交叉验证：原问题的向量路与 BM25 路第一名指向同一段落，就不判"库中无答案"，保留该段。
    # 交叉编码器对大幅改写的问法（"压缩库被篡改" ↔ "xz / liblzma 后门"）会给出极低分，单一阈值分不开"措辞差异大"和"库里没有"
    min_score_agreement: bool = True
    min_score_ratio: float | None = None # 相对阈值（动态 top-K）：分数低于 top-1 × ratio 的候选不进入上下文，避免凑数
    device: str | None = None
    api_url: str | None = None           # api provider：Jina / SiliconFlow 等 /rerank 端点的 base URL；cohere 留空即官方地址
    api_key: str | None = None           # api / cohere 的密钥；cohere 不填则读环境变量 COHERE_API_KEY
    llm_batch_size: int = 20             # llm provider：每次打分的文档数


class DiversifyConfig(BaseModel):
    """MMR 多样化：在相关性与冗余之间取舍，避免 top-K 都是同一段的不同切片。"""

    enabled: bool = False
    mmr_lambda: float = 0.7              # 1.0 = 只看相关性；0 = 只看多样性


class CompressionConfig(BaseModel):
    """上下文压缩（LLMLingua 思想的轻量实现）：规则过滤 / 去重，或轻量 LLM 抽取。"""

    enabled: bool = False
    mode: Literal["rule", "llm"] = "rule"
    ratio: float = 0.6                   # rule 模式保留的句子比例
    min_tokens: int = 120                # 短于该长度的上下文不压缩


class GranularityConfig(BaseModel):
    return_parents: bool = True          # Small-to-Big：命中 child 返回 parent
    auto_merge_adjacent: bool = True     # 同一 parent 内多个相邻 child 命中 → 合并
    sentence_window: int = 0             # >0：以命中 child 为中心，向前后各扩展 N 个 child
    max_context_tokens: int = 3000
    order: Literal["score", "lost_in_the_middle"] = "lost_in_the_middle"


class QueryTransformConfig(BaseModel):
    rewrite: bool = True                 # 口语 → 书面检索表达（Query 压缩）
    multi_query: int = 3                 # 0 = 关闭；扩展为 N 个不同角度问法
    hyde: bool = True
    hyde_max_query_chars: int = 30       # 仅对短 query 启用 HyDE
    decompose: bool = True               # 多跳问题拆分子问题
    step_back: bool = False
    keep_original: bool = True           # 原始问题必须保留在检索列表里
    skip_hyde_on_precise_tokens: bool = True
    condense_history: bool = True        # 多轮对话：结合历史把追问改写为独立问题


class GraphConfig(BaseModel):
    """UnWeaver 实体织网 / GraphRAG-lite：定义抽取 + 检索后按实体关系扩展上下文。"""

    unweaver_enabled: bool = True
    llm_extraction: bool = False         # 用 LLM 抽取 (实体, 关系, 实体) 三元组（索引阶段成本）
    max_weaved: int = 3                  # 每次最多织入的定义块数
    # 织入块也要过精排：与问题的精排分低于 weave_min_score（未设置时用 retrieval.rerank.min_score）就不织入。
    # LLM 抽取的实体很密（"年假""员工手册"都算），不设门槛时几乎每次都会织进与问题无关的段落
    weave_relevance: bool = True
    weave_min_score: float | None = None


class MultiHopConfig(BaseModel):
    enabled: bool = False
    max_hops: int = 3


class AgenticConfig(BaseModel):
    enabled: bool = False
    max_steps: int = 3


class RouterConfig(BaseModel):
    """查询路由：按知识库类别 / 语言把 query 分流到最合适的知识库。"""

    enabled: bool = False
    mode: Literal["embedding", "llm"] = "embedding"
    top_kbs: int = 1


class SQLRouteConfig(BaseModel):
    """混合检索的第三路：Excel / CSV 入库成 SQLite 表，按字段过滤 / 比较 / 排序（向量与 BM25 都够不到的结构化查询）。"""

    enabled: bool = True                 # 知识库里没有表格数据时这一路自动不出现
    text_to_sql: bool = True             # 有 LLM 时用 text-to-SQL；否则（或失败时）走规则兜底
    top_n: int = 20
    weight: float = 1.0                  # RRF 权重；结构化问答为主的库可调高


class SummaryRouteConfig(BaseModel):
    """摘要索引召回路（需入库时开启 enrich.summary_index）。"""

    enabled: bool = True
    weight: float = 1.0


class HierarchyConfig(BaseModel):
    """多粒度召回（需入库时开启 chunking.multi_granularity）。

    auto：按问题类型选层——概念 / 概述类问题加"章节级"路，细节 / 数值类问题加"句子级"路，其余只走默认段落级；
    all：章节级 + 句子级都参与融合。"""

    enabled: bool = True
    routing: Literal["auto", "all"] = "auto"
    weight: float = 1.0


class AccessConfig(BaseModel):
    """权限元数据过滤：检索前按用户身份（级别 + 部门）筛掉无权查看的文档。"""

    enabled: bool = True


class MetadataFilterConfig(BaseModel):
    """自查询：从问题里读出年份 / 部门 / 标签 / 文件类型等限定，召回前先按文档属性缩小范围。"""

    # off：不识别；rule：只认问题里的年份和完整出现的部门名；llm：让辅助模型在本库已有的取值里挑条件（没有模型时退到 rule）
    mode: Literal["off", "rule", "llm"] = "llm"
    fields: list[Literal["department", "tags", "year", "file_format"]] = ["department", "tags", "year", "file_format"]
    include_missing: bool = True         # 文档没填该属性时不排除：标签通常填不全，排除会把没打标的文档整批丢掉
    fallback: bool = True                # 按识别出的条件没有检索结果时，去掉条件重查一次


class RetrievalConfig(BaseModel):
    top_n_per_route: int = 50
    metadata_filter: MetadataFilterConfig = MetadataFilterConfig()
    summary: SummaryRouteConfig = SummaryRouteConfig()
    hierarchy: HierarchyConfig = HierarchyConfig()
    access: AccessConfig = AccessConfig()
    parallel_recall: bool = True         # 多路召回并发执行：各路耗时不叠加
    sparse_backend: Literal["bm25", "lancedb_fts"] = "bm25"   # 关键词路：bm25s + jieba，或 LanceDB 原生全文检索（需 lancedb 向量后端）
    sql: SQLRouteConfig = SQLRouteConfig()
    fusion: FusionConfig = FusionConfig()
    rerank: RerankConfig = RerankConfig()
    diversify: DiversifyConfig = DiversifyConfig()
    granularity: GranularityConfig = GranularityConfig()
    compression: CompressionConfig = CompressionConfig()
    query_transform: QueryTransformConfig = QueryTransformConfig()
    graph: GraphConfig = GraphConfig()
    multihop: MultiHopConfig = MultiHopConfig()
    agentic: AgenticConfig = AgenticConfig()
    router: RouterConfig = RouterConfig()
    filter_expired: bool = True          # 按 valid_from / valid_to 过滤过期语料
    # 问题明确问历史规定（"2024 版""旧版""去年"）时不过滤过期语料：资料里标注"已失效"，由回答说明时效
    history_intent: bool = True


class ImageEmbeddingConfig(BaseModel):
    """图文统一嵌入（CLIP 类双塔模型）：文本与图片映射到同一向量空间 → 文搜图 / 图搜图 / 图搜文。"""

    # auto：已安装 torch + transformers 用 clip，否则回退 perceptual（离线感知特征，只支持图搜图，文搜图走描述文本）
    provider: Literal["auto", "clip", "api", "perceptual"] = "auto"
    model: str = "OFA-Sys/chinese-clip-vit-base-patch16"   # 中文用 Chinese-CLIP；英文可用 openai/clip-vit-base-patch32
    device: str | None = None            # cpu / cuda / mps；None = 有 CUDA 用 CUDA，否则 CPU（MPS 在多线程服务里会崩，需显式开启）
    batch_size: int = 16
    max_text_tokens: int | None = None   # None = 按模型自动（CLIP 77 / Chinese-CLIP 64）
    base_url: str | None = None          # api：Jina 兼容多模态 /embeddings 端点
    api_key_env: str = "IMAGE_EMBEDDING_API_KEY"
    api_key: str | None = None


class MultimodalConfig(BaseModel):
    """多模态 RAG：图片抽取 → OCR / VLM 描述 → 图文向量 + 描述文本双索引 → 文搜图 / 图搜图 / 图文混合 / 图搜文 → 带图片引用的回答。"""

    enabled: bool = True
    embedding: ImageEmbeddingConfig = ImageEmbeddingConfig()
    # --- 抽取
    min_image_side: int = 48             # 短边小于该像素的图（图标 / 分隔线）不入库
    min_image_area: int = 6400           # 面积阈值（像素）
    max_images_per_doc: int = 300
    max_side: int = 1280                 # 入库图片最长边（等比缩小），控制存储与 VLM 成本
    drop_repeated_pages: int = 3         # 同一张图出现在 ≥N 页 → 视为 logo / 水印，丢弃
    context_chars: int = 300             # 图片上下文（同页 / 邻近段落）截取长度
    # --- 描述（路线一：文本描述法）
    ocr: Literal["auto", "none", "rapidocr", "paddleocr", "tesseract"] = "auto"
    ocr_min_chars: int = 2               # OCR 结果少于该字符数视为无意义
    vlm_caption: bool = True             # 用激活的多模态模型（VLM）生成图片语义描述；无模型时自动跳过
    caption_max_side: int = 768          # 送入 VLM 的图片最长边
    # --- 检索
    top_k: int = 6
    top_n_per_route: int = 30
    rrf_k: int = 60
    weight_clip: float = 1.0             # 图文向量路（CLIP 文本塔 → 图片向量）
    weight_desc_dense: float = 1.0       # 描述文本向量路
    weight_desc_sparse: float = 1.0      # 描述文本 BM25 路（OCR 文字 / 编号 / 数值的精确匹配）
    # LanceDB 多模态表路（图片 + 正文同一张表 mm）：打开后文搜图 / 问答附图多一路表内图片文字全文（ICU 分词），图搜文多一路
    # CLIP 向量 + 图中文字的原生混合检索，与上面各路一起 RRF 融合；表在保存索引时（以及查询前发现有变化时）增量同步。需要 lancedb
    lance_table: bool = False
    weight_lance_mm: float = 1.0
    hybrid_alpha: float = 0.5            # 图文混合检索的文本权重：score = α·z(text·I) + (1-α)·z(image·I)
    min_score: float | None = None       # 图搜图余弦相似度阈值（低于则过滤），例如 0.7
    # 图搜文：相似图片达到该相似度时，其所在段落作为候选。None = 按模型自动
    # （CLIP 图-图余弦普遍偏高：无关图也有 0.7~0.85，取 0.9；离线感知特征取 0.7）
    neighbor_min_score: float | None = None
    ocr_identify: bool = True            # 图搜图 / 图搜文：图中有文字时，用图中文字 + 视觉一起认定是库里哪张图（需要 OCR）
    ocr_identify_min_chars: int = 6      # 图中有效文字少于这么多就只看视觉
    weight_ocr_identify: float = 1.0     # 认定时图中文字路的 RRF 权重
    neighbor_floor: float = 0.6          # 图搜文："明显领先"认定的最低相似度
    neighbor_margin: float = 0.15        # 图搜文：第一名领先第二名至少这么多，才认定是同一张图（翻拍 / 截图）
    chunk_sync_per_query: int = 512      # 图搜文直达路：每次查询最多补建多少个正文块的图文向量（大库首查不卡死）
    # --- 问答（端到端）
    chat_images: int = 4                 # 回答最多携带的图片数
    chat_attach_images: int = 2          # 以像素形式交给 VLM 的检索图片数（其余只给描述文本）
    # 普通提问时附图的图文向量门槛：相似度要比"中性提示词"基线高出这么多（显式要图时不受限）。
    # 用相对幅度而不是绝对值：Chinese-CLIP 的文-图余弦挤在 0.32~0.49，无关问题也有 0.36，绝对阈值分不开
    chat_clip_margin: float = 0.065
    chat_term_coverage: float = 0.5      # 普通提问时，问题关键词在图片描述中的覆盖率达到该值才附图
    # 问题和图片文字不是同一种语言（中文问英文图表）、关键词路几乎匹配不上时，用辅助模型把问题译成图片文字的语言补充检索。
    # CLIP 论文（不开 VLM 描述）实测：文搜图 hit@1 0.65 → 0.77，回答附图精确率 / 召回率 0.05 / 0.10 → 0.67 / 0.75
    cross_lingual: bool = True


class LLMConfig(BaseModel):
    provider: Literal["anthropic", "openai_compatible", "none"] = "anthropic"
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = 4096
    temperature: float = 0.0             # openai_compatible 使用
    base_url: str | None = None          # openai_compatible：如 https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key_env: str = "OPENAI_API_KEY"  # openai_compatible 使用；anthropic 走 ANTHROPIC_API_KEY / ant auth
    api_key: str | None = None           # 直接给出 key（来自模型配置中心）时优先
    # 查询改写 / HyDE / 上下文摘要等"小任务"可用更快的模型
    aux_model: str | None = None
    aux_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    # 回答前是否让推理模型（OpenAI 兼容端点，如 deepseek-v4-pro）先思考：思考让首字晚 2.5~4.5 秒（实测）。
    # auto = 只在多跳 / Agentic 模式思考；on = 总是；off = 从不。Claude 的思考深度由 effort 控制，不受此项影响
    thinking: Literal["auto", "on", "off"] = "auto"


class GenerationConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    cite_sources: bool = True
    refuse_when_no_context: bool = True
    language: str = "zh"


class EvalConfig(BaseModel):
    golden_set: Path = Path("data/eval/golden_qa.jsonl")
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    ablation: bool = True                # 分别评测 dense / sparse / fused / reranked


class MemoryConfig(BaseModel):
    """长期记忆（LanceDB 表 memories）：对话里记住用户的身份、偏好、事实，回答时按相关度 × 新近度 × 重要度召回。"""

    enabled: bool = True
    recall_top_k: int = 4                # 每轮回答最多带几条记忆
    min_score: float = 0.35              # 综合分低于该值的记忆不进提示词
    extract: Literal["llm", "rule", "off"] = "llm"   # 回答后从用户发言抽取记忆：llm（辅助模型）/ rule（规则）/ off
    dedup_threshold: float = 0.9         # 与已有记忆余弦相似度 ≥ 该值视为同一条：合并更新，不重复写入
    half_life_days: float = 30.0         # 新近度半衰期
    w_similarity: float = 0.7
    w_recency: float = 0.15
    w_importance: float = 0.15


class RecommendConfig(BaseModel):
    """个性化推荐（LanceDB 表 rec_items / rec_events）：用户兴趣向量 = 近期交互条目向量的时间衰减加权和。"""

    enabled: bool = True
    half_life_days: float = 14.0         # 交互的兴趣权重半衰期
    mmr_lambda: float = 0.7              # 推荐列表多样化：1 只看相关，0 只看多样
    popularity_weight: float = 0.1       # 热度加分（冷启动 / 打破平局）
    exclude_seen: bool = True            # 不推荐已经看过 / 点过的条目


class AgentsConfig(BaseModel):
    """多智能体协作：规划 → 并行研究 → 分析 → 审校（证据不足打回重查）→ 撰稿；黑板与运行记录存 LanceDB。"""

    max_review_rounds: int = 2           # 审校判定证据不足时最多回到研究环节几轮
    research_top_k: int = 4              # 每个子问题检索带回的资料数
    max_subtasks: int = 3
    use_memory: bool = True              # 规划与撰稿时参考该用户的长期记忆


class LakeConfig(BaseModel):
    """Agent 湖（LanceDB 表 agent_lake）：问题 / 检索日志 / 回答 / 反馈 / 记忆 / 黑板 / 文档 / 分析记录放进同一张带版本的宽表，
    既给分析用（检索质量、知识缺口、主题、SQL、分析智能体），也给检索增强用（相似历史问答作为参考上下文）。"""

    enabled: bool = True
    record_chat: bool = True             # 智能问答每一轮自动沉淀：问题 + 检索日志 + 回答
    augment: bool = True                 # 回答时召回相似的历史问答作为参考上下文（只取未被点踩、非拒答的回答）
    augment_top_k: int = 2
    augment_min_score: float = 0.75      # 历史问题与当前问题的余弦相似度下限（BGE-M3 同义改写约 0.75~0.9）
    low_score: float = 0.05              # 检索最高分低于该值记为"低置信"（bge-reranker 分数，无关 < 0.01）
    sql_row_limit: int = 500             # SQL 分析返回的最多行数
    agent_max_steps: int = 4             # 分析智能体最多调用几次工具（含反思后追加的）
    agent_reflect: bool = True           # 执行完规划的工具后让模型看结果再决定要不要追加调用（ReAct），直到够用或达到上限
    catalog_examples: int = 3            # 规划时从湖里取几条"以前分析成功的 SQL"作为示例（按问题相似度）


class Settings(BaseModel):
    paths: PathsConfig = PathsConfig()
    parsing: ParsingConfig = ParsingConfig()
    cleaning: CleaningConfig = CleaningConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    enrich: EnrichConfig = EnrichConfig()
    corpus: CorpusConfig = CorpusConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    sparse: SparseConfig = SparseConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    multimodal: MultimodalConfig = MultimodalConfig()
    generation: GenerationConfig = GenerationConfig()
    eval: EvalConfig = EvalConfig()
    memory: MemoryConfig = MemoryConfig()
    recommend: RecommendConfig = RecommendConfig()
    agents: AgentsConfig = AgentsConfig()
    lake: LakeConfig = LakeConfig()

    @classmethod
    def load(cls, path: str | os.PathLike | None = None, overrides: dict[str, Any] | None = None) -> "Settings":
        """加载 YAML；`RAG_CONFIG` 环境变量可指定路径；overrides 用于 CLI / 测试 / 数据库临时覆盖。"""
        path = path or os.environ.get("RAG_CONFIG") or "config.yaml"
        data: dict[str, Any] = {}
        p = Path(path)
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if overrides:
            data = deep_merge(data, overrides)
        return cls.model_validate(data)

    def with_overrides(self, overrides: dict[str, Any] | None) -> "Settings":
        """返回应用了部分覆盖的新 Settings（不修改自身）。"""
        if not overrides:
            return self
        return Settings.model_validate(deep_merge(self.model_dump(mode="json"), overrides))

    @classmethod
    def lite(cls, index_dir: str | os.PathLike = "data/index") -> "Settings":
        """无模型下载、无网络的最小配置：哈希向量 + BM25 + 词法重排 + 无 LLM。用于测试与冒烟。"""
        return cls(
            paths=PathsConfig(index_dir=Path(index_dir)),
            parsing=ParsingConfig(mineru=MinerUConfig(enabled=False)),
            embedding=EmbeddingConfig(provider="hash", dim=512),
            multimodal=MultimodalConfig(embedding=ImageEmbeddingConfig(provider="perceptual"), ocr="none", vlm_caption=False),
            retrieval=RetrievalConfig(
                rerank=RerankConfig(enabled=False, provider="none"),
                query_transform=QueryTransformConfig(rewrite=False, multi_query=0, hyde=False, decompose=False),
            ),
            generation=GenerationConfig(llm=LLMConfig(provider="none")),
        )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


_deep_merge = deep_merge
