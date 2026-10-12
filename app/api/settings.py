from __future__ import annotations

from fastapi import APIRouter, HTTPException

from prism.config import Settings

from ..schemas import SettingsUpdate
from ..services import index_service, model_service, settings_service, reindex_service

router = APIRouter()

# 策略配置页的字段说明（推荐值来自 docs/RAG_OPTIMIZATION_SUMMARY.md）
FIELD_DOCS = {
    "chunking.strategy": "structural：按标题/段落切；semantic：相邻句向量距离断点；recursive：段落→句→字符",
    "chunking.parent_max_tokens": "parent 阅读粒度，推荐 500-800 token（中文 300-500 字）",
    "chunking.child_tokens": "child 检索粒度，推荐 100-150 token",
    "chunking.child_overlap_ratio": "重叠率 10%-15%，太大重复检索、浪费 token",
    "retrieval.top_n_per_route": "每路召回 top-N（recall_top_k），推荐 50-100",
    "retrieval.sql.enabled": "混合检索第三路：Excel / CSV 入库成表，按字段过滤 / 比较 / 排序；库里没有表格时不生效",
    "retrieval.sql.text_to_sql": "有文本模型时用 text-to-SQL（只读 SELECT）；关闭或失败时走规则兜底",
    "retrieval.parallel_recall": "向量 / BM25 / HyDE / SQL 各路并发执行，耗时不叠加",
    "retrieval.fusion.method": "rrf：只看排名，零参数最稳；convex：min-max 后加权，需有评估集调 α",
    "retrieval.fusion.rrf_k": "RRF 平滑常数，业界标准 60（可试 20/40）",
    "retrieval.fusion.weight_sparse": "金融/法律/IT 文档可调高 BM25 权重",
    "retrieval.rerank.provider": "auto：显式 API → 本地 bge-reranker-v2-m3（torch + transformers）→ LLM → 词法；cohere 需在模型配置里添加 Cohere 或设置 COHERE_API_KEY",
    "retrieval.rerank.model": "本地交叉编码器，默认 BAAI/bge-reranker-v2-m3（中英双语；Apple Silicon 上 50 个候选约 0.2s）",
    "retrieval.rerank.cohere_model": "rerank-v4.0-pro 质量最好；rerank-v4.0-fast 延迟低；rerank-v3.5 多语言",
    "retrieval.rerank.auto_download": "本地重排模型不在本机时后台下载（约 2.3GB），期间由 LLM / 词法精排顶上",
    "retrieval.rerank.candidates": "传给 Reranker 的候选数，必须 ≥ 50（20 候选 R@5 仅 0.458）",
    "retrieval.rerank.keep_n": "重排保留 rerank_top_n，推荐 10",
    "retrieval.rerank.top_k": "最终上下文 context_top_n，推荐 3-5（Lost in the Middle）",
    "retrieval.rerank.min_score": "低于阈值判定库中无答案 → 拒答。单靠阈值分不开两类情况：大幅改写的相关问题可能只有 0.001 级，库里没有答案的问题也可能到 0.06，而措辞模糊的相关问题（「图2 说明了什么」）可能只有 0.1——取偏低的阈值（如 0.03）保证能答，再由「两路交叉验证」救回大幅改写的问题",
    "retrieval.rerank.min_score_agreement": "两路交叉验证：精排全部低于 min_score 时，若原问题的向量路与 BM25 路第一名是同一段落，就不判无答案（救回措辞差异大的相关问题）；库里没有答案的问题两路通常指向不同段落",
    "retrieval.rerank.min_score_ratio": "相对阈值：分数低于 top-1 × ratio 的候选不进入上下文，避免用弱相关内容凑满 top-K",
    "retrieval.diversify.mmr_lambda": "MMR λ：1 只看相关性，0 只看多样性，推荐 0.7",
    "retrieval.compression.mode": "rule：零成本规则压缩；llm：轻量模型逐字抽取",
    "retrieval.query_transform.hyde": "短 query 有效，含精确编号/数字时自动跳过",
    "retrieval.query_transform.multi_query": "扩展 3-5 个角度分别检索，原始问题始终保留",
    "generation.llm.thinking": "推理模型先思考会让首字晚 3~5 秒；auto 只在多跳 / Agentic 模式思考（Claude 看「推理强度」）",
    "generation.llm.aux_model": "查询改写 / HyDE / text-to-SQL 用的快速模型，须在同一端点；留空 = 与回答模型相同",
    "retrieval.graph.unweaver_enabled": "实体定义织入：检索块提到术语但定义块未召回时自动补齐；织入块同样经过过期 / 权限过滤",
    "retrieval.graph.weave_relevance": "织入块也过一遍精排，与问题的相关度低于门槛就不织入（LLM 抽取的实体很密，不设门槛时常织进无关段落）",
    "retrieval.graph.weave_min_score": "织入门槛；留空 = 沿用「拒答阈值」retrieval.rerank.min_score，两者都没设时不过滤",
    "retrieval.history_intent": "问题明确问旧规定（“2024 版”“旧版”“去年”）时检索已失效文档，资料中标注“已失效”，由回答说明时效；关闭则一律过滤",
    "retrieval.multihop.max_hops": "多跳最大跳数，推荐 2-3",
    "multimodal.embedding.provider": "auto：装了 torch + transformers 用 CLIP，否则离线感知特征（只支持图搜图，文搜图走描述文本）",
    "multimodal.embedding.model": "中文用 OFA-Sys/chinese-clip-vit-base-patch16；原版 CLIP 以英文训练，中文查询效果差",
    "multimodal.vlm_caption": "入库时用多模态模型给每张图生成语义描述（每图一次调用，按内容哈希缓存）",
    "multimodal.ocr": "图中文字：rapidocr（推荐，CPU 可跑）/ paddleocr / tesseract；精确数值以 OCR 为准",
    "multimodal.cross_lingual": "问题和图片文字语言不同（中文问英文图表）时，用辅助模型把问题译成图片文字的语言补充检索；关闭后跨语言资料主要依赖 VLM 中文描述",
    "multimodal.hybrid_alpha": "图文混合检索的文本权重 α：score = α·z(text·I) + (1-α)·z(image·I)",
    "multimodal.min_score": "图搜图余弦相似度阈值，低于则过滤（例如 0.7）；不设则总返回 top-K",
    "multimodal.rrf_k": "图片多路融合的 RRF 常数：OCR 文本很短、BM25 区分度低时可调到 100",
    "multimodal.chat_images": "一轮回答最多携带的图片数；0 = 回答不带图",
    "multimodal.chat_attach_images": "以像素形式交给 VLM 的检索图片数（其余只给描述文本），越多越贵",
    "multimodal.lance_table": "把 LanceDB 多模态表（图片 + 正文同一张表）接入线上图文检索：文搜图 / 问答附图多一路表内图片文字的全文检索（ICU 分词），图搜文多一路 CLIP 向量 + 图中文字的原生混合检索；表在保存索引时增量同步，需要安装 lancedb",
    "multimodal.weight_lance_mm": "LanceDB 多模态表路在 RRF 融合里的权重；0 = 只同步表、不参与检索",
    "parsing.mineru.tier": "四档按「质量—速度—成本」取舍；PDF / 图片支持全部四档，Office 等在 flash 层原生解析",
    "parsing.mineru.model_source": "模型包下载源；国内网络选 modelscope",
    "parsing.mineru.api_url": "填写后连接自托管 mineru-kit api-server，档位由服务端 /v1/tiers 决定",
}


# 场景预设：检索 / 后检索手段不是全上最好——按数据特征和瓶颈组合。每个预设只改 retrieval 段，保存前可再微调。
PRESETS = [
    {
        "key": "enterprise", "name": "企业综合知识库", "combo": "混合检索 + RAG-Fusion + 重排序",
        "reason": "数据异构（制度 / 手册 / 表格）、问法多样：多条查询铺开召回面，RRF 把跨查询都靠前的顶上来，再用重排保准确。",
        "bottleneck": "查不全",
        "retrieval": {
            "query_transform": {"rewrite": True, "multi_query": 3, "hyde": True, "decompose": True},
            "fusion": {"method": "rrf", "rrf_k": 60, "weight_dense": 1.0, "weight_sparse": 1.0},
            "sql": {"enabled": True},
            "rerank": {"enabled": True, "candidates": 50, "keep_n": 10, "top_k": 5, "min_score_ratio": None},
            "diversify": {"enabled": False}, "compression": {"enabled": False},
        },
    },
    {
        "key": "regulated", "name": "医疗 / 法律 / 合规", "combo": "混合检索 + 重排序 + 上下文过滤",
        "reason": "出错代价大：不做 HyDE / 多查询这类可能引入臆测的扩展，抬高关键词权重保条款编号与剂量规格，重排后丢弃弱相关、压缩去噪，过期版本一律过滤。",
        "bottleneck": "查不准",
        "retrieval": {
            "query_transform": {"rewrite": True, "multi_query": 0, "hyde": False, "decompose": False},
            "fusion": {"method": "rrf", "rrf_k": 60, "weight_dense": 1.0, "weight_sparse": 1.5},
            "sql": {"enabled": True},
            "rerank": {"enabled": True, "candidates": 50, "keep_n": 8, "top_k": 4, "min_score_ratio": 0.5},
            "diversify": {"enabled": False},
            "compression": {"enabled": True, "mode": "rule", "ratio": 0.6},
            "filter_expired": True,
        },
    },
    {
        "key": "faq", "name": "客服 FAQ", "combo": "混合检索 + 重排序",
        "reason": "问法口语、答案短：改写把口语转成检索表达，向量懂同义问法、BM25 兜产品名与错误码，重排后只给 3 条，响应快。",
        "bottleneck": "查不准",
        "retrieval": {
            "query_transform": {"rewrite": True, "multi_query": 0, "hyde": False, "decompose": False},
            "fusion": {"method": "rrf", "rrf_k": 60, "weight_dense": 1.0, "weight_sparse": 1.0},
            "rerank": {"enabled": True, "candidates": 50, "keep_n": 6, "top_k": 3, "min_score_ratio": None},
            "diversify": {"enabled": False}, "compression": {"enabled": False},
        },
    },
    {
        "key": "longdoc", "name": "长文档问答", "combo": "父子索引 + 上下文压缩",
        "reason": "小块检索、大块阅读保住上下文；父块又长又吵，按问题压缩去噪省 token；MMR 避免 top-K 都是同一段的切片。",
        "bottleneck": "答不对 / token 浪费",
        "retrieval": {
            "query_transform": {"rewrite": True, "multi_query": 0, "hyde": True, "decompose": True},
            "granularity": {"return_parents": True, "auto_merge_adjacent": True, "order": "lost_in_the_middle"},
            "rerank": {"enabled": True, "candidates": 50, "keep_n": 10, "top_k": 4, "min_score_ratio": None},
            "diversify": {"enabled": True, "mmr_lambda": 0.7},
            "compression": {"enabled": True, "mode": "rule", "ratio": 0.6},
        },
    },
]

# 决策思路：先判断瓶颈在哪一环，再决定加哪一招
DECISION_GUIDE = [
    {"symptom": "查不全（相关内容没被召回）", "action": "混合检索（向量 + BM25 + SQL）/ Multi-Query 多路召回", "fields": ["retrieval.query_transform.multi_query", "retrieval.sql.enabled", "retrieval.top_n_per_route"]},
    {"symptom": "查不准（召回了但排在后面、噪声在前）", "action": "RRF 融合 + 重排序；重排后丢弃弱相关", "fields": ["retrieval.rerank.enabled", "retrieval.rerank.candidates", "retrieval.rerank.min_score_ratio"]},
    {"symptom": "块又长又吵（token 浪费、被无关内容带偏）", "action": "上下文压缩 / 过滤", "fields": ["retrieval.compression.enabled", "retrieval.compression.mode"]},
    {"symptom": "答不对（上下文不完整）", "action": "Small-to-Big 返回父块；再往前查切片与解析", "fields": ["retrieval.granularity.return_parents", "chunking.parent_max_tokens"]},
]


def _mask_secrets(obj):
    """配置里的 api_key 不回传明文（模型配置中心的密钥同样只返回脱敏值）。"""
    if isinstance(obj, dict):
        return {k: ("********" if k == "api_key" and v else _mask_secrets(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask_secrets(x) for x in obj]
    return obj


@router.get("/presets")
def presets():
    """场景预设与决策思路：一键把一组检索 / 后检索参数填入策略配置或检索实验室。"""
    return {"presets": PRESETS, "guide": DECISION_GUIDE}


@router.get("/")
def get_settings():
    s = settings_service.get()
    return {"effective": _mask_secrets(s.model_dump(mode="json")), "overrides": _mask_secrets(settings_service.overrides()), "defaults": _mask_secrets(settings_service.base().model_dump(mode="json")), "fieldDocs": FIELD_DOCS}


@router.put("/")
def update_settings(body: SettingsUpdate):
    try:
        s = settings_service.update(body.overrides)
    except Exception as exc:
        raise HTTPException(400, f"配置无效：{exc}") from exc
    model_service.invalidate()
    index_service.invalidate()
    # 入库参数（解析 / 清洗 / 切片 / 索引增强）变了 → 后台自动重建受影响的知识库
    scheduled = reindex_service.fingerprint_changed()
    reindex_service.request("策略配置变更")
    return {"effective": _mask_secrets(s.model_dump(mode="json")), "overrides": _mask_secrets(settings_service.overrides()), "reindexScheduled": scheduled}


@router.post("/reset")
def reset_settings():
    s = settings_service.reset()
    model_service.invalidate()
    index_service.invalidate()
    scheduled = reindex_service.fingerprint_changed()
    reindex_service.request("恢复默认配置")
    return {"effective": _mask_secrets(s.model_dump(mode="json")), "overrides": {}, "reindexScheduled": scheduled}


@router.get("/reindex-status")
def reindex_status():
    """自动重建索引的进度：前端保存配置 / 切换向量模型后轮询。"""
    return reindex_service.status()


@router.get("/schema")
def settings_schema():
    return Settings.model_json_schema()
