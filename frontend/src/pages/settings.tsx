import { useEffect, useMemo, useState } from 'react';
import { Boxes, CheckCircle2, FileText, GitBranch, Layers, Network, RotateCcw, Save, Scissors, ShieldCheck, Shuffle, Sparkles, Zap, RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import type { MineruConnectionTest, MineruTiersInfo, PresetsPayload, SettingsPayload, StrategyPreset, ReindexStatus } from '../api/types';
import { Badge, Button, Callout, Spinner, Toggle } from '../components/ui';
import { useApp } from '../store/app-store';
import { VectorBenchmarkCard } from '../components/vector-benchmark';

// when：按其他字段的当前值决定是否显示；secret：密钥，后端只回传脱敏值，留空 = 不修改
type FieldSpec = { path: string; label: string; hint?: string; when?: (val: (path: string) => any) => boolean } & (
  | { type: 'bool' }
  | { type: 'num'; min: number; max: number; step?: number; nullable?: boolean }
  | { type: 'select'; options: [string, string][] }
  | { type: 'text'; placeholder?: string }
  | { type: 'secret'; placeholder?: string }
);
const MASKED = '********';
const mineruMode = (val: (path: string) => any) => String(val('parsing.mineru.mode') ?? 'local');

const GROUPS: { title: string; icon: React.ReactNode; tone: string; note?: string; reindex?: boolean; fields: FieldSpec[] }[] = [
  {
    title: '文档解析', icon: <FileText size={14} />, tone: 'info', reindex: true, note: 'Markdown 永远直解析；Excel / CSV 走行级结构化；其余格式优先 MinerU。',
    fields: [
      { path: 'parsing.mineru.enabled', label: '启用 MinerU', type: 'bool', hint: '未安装 mineru 包时自动使用内置解析器' },
      { path: 'parsing.mineru.mode', label: '解析方式', type: 'select', options: [['local', '本机（下载模型包）'], ['cloud', '云端 API（mineru.net）'], ['server', '自托管服务（mineru-kit api-server）']] },
      { path: 'parsing.mineru.cloud.api_key', label: '云端 API 密钥', type: 'secret', placeholder: 'sk_…（可留空，匿名调用）', when: (v) => mineruMode(v) === 'cloud' },
      { path: 'parsing.mineru.api_url', label: '解析服务地址', type: 'text', placeholder: 'http://127.0.0.1:8000', when: (v) => mineruMode(v) === 'server' },
      { path: 'parsing.mineru.api_key', label: '解析服务密钥', type: 'secret', placeholder: '未启用鉴权可留空', when: (v) => mineruMode(v) === 'server' },
      // 选项由后端 /system/mineru/tiers 动态加载（本机安装情况 / 云端 API 或自托管服务的 /v1/tiers）；这里的静态列表只是接口不可用时的兜底
      { path: 'parsing.mineru.tier', label: 'MinerU 档位', type: 'select', options: [['flash', 'flash · 原生文本，无模型'], ['basic', 'basic · 小模型 OCR / 公式 / 表格'], ['standard', 'standard · 小模型 + VLM'], ['advanced', 'advanced · 最高质量']] },
      { path: 'parsing.mineru.model_source', label: '模型下载源', type: 'select', options: [['', '默认（HuggingFace）'], ['huggingface', 'HuggingFace'], ['modelscope', 'ModelScope（国内）']], when: (v) => mineruMode(v) === 'local' },
      { path: 'parsing.mineru.ocr_mode', label: 'OCR 模式', type: 'select', options: [['auto', 'auto'], ['txt', 'txt'], ['ocr', 'ocr']] },
      { path: 'parsing.mineru.image_analysis', label: '图片 / 图表内容分析', type: 'bool' },
      { path: 'parsing.popo.enabled', label: '使用 MinerU-Popo 文档树', type: 'bool', hint: '需先跑完 Popo 推理；跨页表格合并、段落续接、标题层级重建' },
    ],
  },
  {
    title: '清洗', icon: <Sparkles size={14} />, tone: 'info', reindex: true,
    fields: [
      { path: 'cleaning.drop_headers_footers', label: '去除页眉页脚', type: 'bool' },
      { path: 'cleaning.fix_pdf_linebreaks', label: '修复 PDF 断行', type: 'bool' },
      { path: 'cleaning.dedup_near_duplicates', label: '近似去重', type: 'bool' },
      { path: 'cleaning.dedup_jaccard_threshold', label: '去重阈值（Jaccard）', type: 'num', min: 0.5, max: 1, step: 0.01 },
      { path: 'cleaning.garbled_ratio_threshold', label: '乱码占比阈值', type: 'num', min: 0.05, max: 1, step: 0.05 },
    ],
  },
  {
    title: '切片（Parent-Child）', icon: <Scissors size={14} />, tone: 'accent', reindex: true, note: '最值得花时间的一层：语义切块比固定长度 Top-5 命中率高约 20 个点。',
    fields: [
      { path: 'chunking.strategy', label: '切片策略', type: 'select', options: [['structural', '结构切片（标题 / 段落）'], ['semantic', '语义切片（相邻句向量距离）'], ['recursive', '递归切片']] },
      { path: 'chunking.parent_max_tokens', label: 'parent 最大 token', type: 'num', min: 200, max: 2000, step: 50 },
      { path: 'chunking.parent_min_tokens', label: 'parent 最小 token', type: 'num', min: 0, max: 600, step: 10, hint: '过小的块并入相邻块' },
      { path: 'chunking.child_tokens', label: 'child token', type: 'num', min: 50, max: 600, step: 10 },
      { path: 'chunking.child_overlap_ratio', label: 'child 重叠率', type: 'num', min: 0, max: 0.5, step: 0.01 },
      { path: 'chunking.respect_atomic_blocks', label: '表格 / 代码不拆分', type: 'bool' },
      { path: 'chunking.semantic_breakpoint_percentile', label: '语义断点分位数', type: 'num', min: 50, max: 99, step: 1 },
      { path: 'chunking.multi_granularity', label: '多粒度分层索引', type: 'bool', hint: '额外建章节级与句子级两层索引，检索时按问题类型选粒度（入库计算量约翻倍，不需要 LLM）' },
      { path: 'chunking.section_max_tokens', label: '章节级单元上限 token', type: 'num', min: 300, max: 6000, step: 100 },
    ],
  },
  {
    title: '索引增强', icon: <Layers size={14} />, tone: 'accent', reindex: true, note: '索引阶段的一次性 LLM 成本，需要已配置文本模型。',
    fields: [
      { path: 'enrich.contextual_summary', label: 'Contextual Retrieval（上下文摘要）', type: 'bool', hint: '每块一次 LLM 调用；Recall@5 约 +2.8pp' },
      { path: 'enrich.hypothetical_questions', label: '假设性问题（QA 对索引）', type: 'bool' },
      { path: 'enrich.questions_per_chunk', label: '每块问题数', type: 'num', min: 1, max: 8 },
      { path: 'enrich.summary_index', label: '摘要索引', type: 'bool', hint: '为每个段落生成摘要，单独建一路召回，命中摘要返回原文；无文本模型时用抽取式摘要（标题 + 首句）' },
      { path: 'enrich.summary_max_chars', label: '摘要最大字数', type: 'num', min: 50, max: 600, step: 10 },
      { path: 'retrieval.graph.llm_extraction', label: 'LLM 三元组抽取（GraphRAG-lite）', type: 'bool' },
      { path: 'sparse.k1', label: 'BM25 k1', type: 'num', min: 0.5, max: 3, step: 0.1, hint: '长文档可调到 2.0' },
      { path: 'sparse.b', label: 'BM25 b', type: 'num', min: 0, max: 1, step: 0.05, hint: '文档长度均匀可调到 0.5' },
      { path: 'sparse.exact_token_boost', label: '精确 token 加分', type: 'num', min: 0, max: 20, step: 0.5, hint: '对含编号 / 错误码的块兜底加分' },
    ],
  },
  {
    title: '语料增强', icon: <Sparkles size={14} />, tone: 'accent', reindex: true, note: '不需要 LLM 的文档扩展与元数据补全。别名词典、剔除清单、FAQ 增强在「语料治理」页按知识库维护。',
    fields: [
      { path: 'corpus.keywords', label: '关键词扩展', type: 'bool', hint: '为每个段落抽取 TF-IDF 主题词与编号 / 错误码，写进向量与 BM25 文本（短段落、专有名词多的库收益最大）' },
      { path: 'corpus.keywords_per_chunk', label: '每段关键词数', type: 'num', min: 2, max: 20 },
      { path: 'corpus.auto_metadata', label: '上传时自动抽取元数据', type: 'bool', hint: '版本 / 生效与失效日期 / 部门 / 标签 / 密级，只填空字段' },
      { path: 'corpus.auto_metadata_llm', label: '元数据用文本模型补全', type: 'bool', hint: '规则没找到的字段再调一次模型；结果须能在原文里找到依据' },
      { path: 'corpus.near_dup_threshold', label: '近似重复阈值（体检）', type: 'num', min: 0.5, max: 1, step: 0.01 },
      { path: 'corpus.garbled_threshold', label: '乱码占比阈值（体检）', type: 'num', min: 0.02, max: 1, step: 0.01 },
    ],
  },
  {
    title: '查询层', icon: <Zap size={14} />, tone: 'hyde',
    fields: [
      { path: 'retrieval.query_transform.rewrite', label: '查询改写 / 压缩', type: 'bool' },
      { path: 'retrieval.query_transform.multi_query', label: 'Multi-Query 数量', type: 'num', min: 0, max: 8 },
      { path: 'retrieval.query_transform.hyde', label: 'HyDE 假设文档', type: 'bool' },
      { path: 'retrieval.query_transform.hyde_max_query_chars', label: 'HyDE 仅用于短 query（字符数）', type: 'num', min: 5, max: 200 },
      { path: 'retrieval.query_transform.skip_hyde_on_precise_tokens', label: '精确查询跳过 HyDE', type: 'bool', hint: '含编号 / 数字 / 日期时，假设答案里的数字是编的' },
      { path: 'retrieval.query_transform.decompose', label: '多跳问题分解', type: 'bool' },
      { path: 'retrieval.query_transform.step_back', label: 'Step-back 抽象', type: 'bool' },
      { path: 'retrieval.query_transform.condense_history', label: '多轮对话指代补全', type: 'bool' },
    ],
  },
  {
    title: '向量索引（建库参数）', icon: <Layers size={14} />, tone: 'accent', reindex: true, note: '向量库后端与 Qdrant 的 HNSW 图、量化方式在建索引时确定，改了会自动重建。numpy 后端是精确暴力检索，数万条以内足够快。',
    fields: [
      { path: 'vector_store.backend', label: '向量库后端', type: 'select', options: [['numpy', 'numpy（精确，零依赖）'], ['lancedb', 'LanceDB（嵌入式：向量 + 全文 + 版本）'], ['qdrant', 'Qdrant（HNSW 近似检索）'], ['chroma', 'Chroma']] },
      { path: 'vector_store.lancedb.index_type', label: 'LanceDB ANN 索引', type: 'select', options: [['auto', 'auto（够量自动建 IVF_HNSW_SQ）'], ['none', '不建（始终精确扫描）'], ['IVF_HNSW_SQ', 'IVF_HNSW_SQ'], ['IVF_PQ', 'IVF_PQ（最省内存）'], ['IVF_FLAT', 'IVF_FLAT']] },
      { path: 'vector_store.lancedb.index_min_rows', label: 'LanceDB 建索引门槛（行）', type: 'num', min: 256, max: 10000000, step: 1000, hint: 'auto 时行数达到才建 ANN；之下精确扫描，召回 100%' },
      { path: 'vector_store.hnsw_m', label: 'HNSW m（邻居数）', type: 'num', min: 4, max: 64, hint: '越大召回越高、内存越大；常用 16' },
      { path: 'vector_store.hnsw_ef_construct', label: 'HNSW ef_construct', type: 'num', min: 16, max: 512, step: 8, hint: '建图时的搜索宽度，常用 100-200' },
      { path: 'vector_store.quantization', label: 'Qdrant 量化', type: 'select', options: [['none', '不量化'], ['int8', 'int8 标量量化（内存约 1/4）'], ['binary', '二值量化（内存约 1/32，需 rescore）']] },
    ],
  },
  {
    title: '向量检索调优', icon: <Zap size={14} />, tone: 'dense', note: '查询期参数，保存即生效、不需要重建：先在低维 / 量化的矩阵上粗筛，再用原始向量精算回排，用少量精度换内存和速度。',
    fields: [
      { path: 'vector_store.search.pca_dim', label: 'PCA 降维到', type: 'num', min: 8, max: 2048, step: 8, nullable: true, hint: 'numpy 后端；不设 = 不降维。常用 128-256' },
      { path: 'vector_store.search.quantize', label: '粗筛量化', type: 'select', options: [['none', '不量化（float32）'], ['int8', 'int8（内存约 1/4）']] },
      { path: 'vector_store.search.rescore', label: '原始向量精算回排', type: 'bool', hint: '关闭后速度略快，但降维 / 量化带来的误差会直接体现在排序里' },
      { path: 'vector_store.search.oversample', label: '粗筛放大倍数', type: 'num', min: 1, max: 20, step: 0.5, hint: '粗筛取 top_n × 倍数个候选再精算' },
      { path: 'vector_store.search.ef_search', label: 'HNSW ef_search', type: 'num', min: 16, max: 1024, step: 16, hint: 'Qdrant / LanceDB HNSW 搜索宽度，越大越准越慢' },
      { path: 'vector_store.search.nprobes', label: 'LanceDB nprobes', type: 'num', min: 1, max: 500, hint: 'IVF 索引查询时探查的分区数，越大越准越慢' },
      { path: 'vector_store.search.refine_factor', label: 'LanceDB refine_factor', type: 'num', min: 1, max: 50, nullable: true, hint: '量化索引多取 k×refine 条再用原始向量重排；不设 = 不重排' },
    ],
  },
  {
    title: '召回与融合', icon: <Layers size={14} />, tone: 'fused',
    fields: [
      { path: 'retrieval.top_n_per_route', label: '每路召回 top-N', type: 'num', min: 5, max: 300 },
      { path: 'retrieval.fusion.method', label: '融合方法', type: 'select', options: [['rrf', 'RRF（倒数排名融合）'], ['convex', '凸组合（min-max 加权）']] },
      { path: 'retrieval.fusion.rrf_k', label: 'RRF k', type: 'num', min: 1, max: 200 },
      { path: 'retrieval.summary.enabled', label: '摘要索引召回', type: 'bool', hint: '需先在"索引增强"开启摘要索引' },
      { path: 'retrieval.summary.weight', label: '摘要路权重', type: 'num', min: 0, max: 5, step: 0.1 },
      { path: 'retrieval.hierarchy.enabled', label: '分层召回', type: 'bool', hint: '需先在"切片"开启多粒度分层索引' },
      { path: 'retrieval.hierarchy.routing', label: '分层路由', type: 'select', options: [['auto', '按问题类型（概念 → 章节，细节 → 句子）'], ['all', '章节级 + 句子级都参与']] },
      { path: 'retrieval.hierarchy.weight', label: '分层路权重', type: 'num', min: 0, max: 5, step: 0.1 },
      { path: 'retrieval.access.enabled', label: '权限元数据过滤', type: 'bool', hint: '按顶栏"当前身份"的级别与部门，过滤无权查看的文档' },
      { path: 'retrieval.fusion.weight_dense', label: '向量权重', type: 'num', min: 0, max: 5, step: 0.1 },
      { path: 'retrieval.fusion.weight_sparse', label: 'BM25 权重', type: 'num', min: 0, max: 5, step: 0.1 },
      { path: 'retrieval.fusion.convex_alpha', label: '凸组合 α', type: 'num', min: 0, max: 1, step: 0.05, hint: '技术文档 0.6-0.7；FAQ / 合同 0.4-0.5' },
      { path: 'retrieval.parallel_recall', label: '多路并发召回', type: 'bool' },
      { path: 'retrieval.sparse_backend', label: '关键词路实现', type: 'select', options: [['bm25', 'bm25s + jieba'], ['lancedb_fts', 'LanceDB 原生全文（ICU 分词）']], hint: 'LanceDB 全文需要向量库后端为 LanceDB' },
      { path: 'retrieval.sql.enabled', label: 'SQL 结构化检索（第三路）', type: 'bool' },
      { path: 'retrieval.sql.text_to_sql', label: 'text-to-SQL（LLM）', type: 'bool' },
      { path: 'retrieval.sql.weight', label: 'SQL 路权重', type: 'num', min: 0, max: 5, step: 0.1 },
      { path: 'retrieval.filter_expired', label: '过滤过期文档', type: 'bool' },
      { path: 'retrieval.history_intent', label: '问历史规定时放行过期文档', type: 'bool', hint: '“2024 版”“旧版”“去年”这类问题检索已失效文档，资料中标注失效' },
      { path: 'retrieval.metadata_filter.mode', label: '识别问题里的范围限定（自查询）', type: 'select', options: [['off', '关闭'], ['rule', '规则：年份、完整出现的部门名'], ['llm', '模型：在本库已有的部门 / 标签 / 年份 / 文件类型里挑条件']], hint: '识别出条件后，召回前先按文档属性缩小范围；模型方式每次检索多一次辅助模型调用' },
      { path: 'retrieval.metadata_filter.include_missing', label: '识别出的条件不排除未填属性的文档', type: 'bool', hint: '标签通常填不全；关闭后没打标的文档会被整批排除' },
      { path: 'retrieval.metadata_filter.fallback', label: '按条件查不到时去掉条件重查', type: 'bool' },
    ],
  },
  {
    title: '精排与多样化', icon: <Shuffle size={14} />, tone: 'rerank',
    fields: [
      { path: 'retrieval.rerank.enabled', label: '启用 Reranker', type: 'bool' },
      { path: 'retrieval.rerank.provider', label: '精排方式', type: 'select', options: [['auto', 'auto（API → 本地 bge → LLM → 词法）'], ['cross_encoder', '本地交叉编码器'], ['cohere', 'Cohere Rerank'], ['api', '通用 /rerank 接口'], ['llm', 'LLM 打分（慢）'], ['lexical', '词法精排'], ['none', '不重排']], hint: '模型配置中心激活的重排模型优先；Cohere 的密钥在模型配置里添加，或设置环境变量 COHERE_API_KEY' },
      { path: 'retrieval.rerank.model', label: '本地重排模型', type: 'select', options: [['BAAI/bge-reranker-v2-m3', 'BAAI/bge-reranker-v2-m3（默认）'], ['BAAI/bge-reranker-large', 'BAAI/bge-reranker-large'], ['BAAI/bge-reranker-base', 'BAAI/bge-reranker-base（轻量）']] },
      { path: 'retrieval.rerank.cohere_model', label: 'Cohere 模型', type: 'select', options: [['rerank-v4.0-pro', 'rerank-v4.0-pro（质量优先）'], ['rerank-v4.0-fast', 'rerank-v4.0-fast（低延迟）'], ['rerank-v3.5', 'rerank-v3.5'], ['rerank-multilingual-v3.0', 'rerank-multilingual-v3.0'], ['rerank-english-v3.0', 'rerank-english-v3.0']] },
      { path: 'retrieval.rerank.auto_download', label: '本地模型自动下载', type: 'bool' },
      { path: 'retrieval.rerank.candidates', label: '候选数', type: 'num', min: 5, max: 300 },
      { path: 'retrieval.rerank.keep_n', label: '重排保留 rerank_top_n', type: 'num', min: 1, max: 50 },
      { path: 'retrieval.rerank.top_k', label: '最终上下文 context_top_n', type: 'num', min: 1, max: 20 },
      { path: 'retrieval.rerank.min_score', label: '拒答阈值 min_score', type: 'num', min: 0, max: 1, step: 0.01, nullable: true },
      { path: 'retrieval.rerank.min_score_agreement', label: '拒答前两路交叉验证', type: 'bool', hint: '精排低于阈值但向量与 BM25 第一名是同一段落时不拒答' },
      { path: 'retrieval.rerank.min_score_ratio', label: '相对阈值 min_score_ratio', type: 'num', min: 0, max: 1, step: 0.05, nullable: true },
      { path: 'retrieval.diversify.enabled', label: 'MMR 多样化', type: 'bool' },
      { path: 'retrieval.diversify.mmr_lambda', label: 'MMR λ', type: 'num', min: 0, max: 1, step: 0.05 },
    ],
  },
  {
    title: '上下文工程', icon: <Scissors size={14} />, tone: 'mmr',
    fields: [
      { path: 'retrieval.granularity.return_parents', label: 'Small-to-Big（返回父块）', type: 'bool' },
      { path: 'retrieval.granularity.auto_merge_adjacent', label: '相邻命中自动合并', type: 'bool' },
      { path: 'retrieval.granularity.sentence_window', label: '句子窗口', type: 'num', min: 0, max: 5, hint: '关闭父块返回时，向前后扩展的子块数' },
      { path: 'retrieval.granularity.max_context_tokens', label: '上下文 token 预算', type: 'num', min: 500, max: 20000, step: 100 },
      { path: 'retrieval.granularity.order', label: '排列方式', type: 'select', options: [['lost_in_the_middle', '首尾优先（Lost in the Middle）'], ['score', '按分数降序']] },
      { path: 'retrieval.compression.enabled', label: '上下文压缩', type: 'bool' },
      { path: 'retrieval.compression.mode', label: '压缩方式', type: 'select', options: [['rule', '规则（零成本）'], ['llm', 'LLM 逐字抽取']] },
      { path: 'retrieval.compression.ratio', label: '保留比例', type: 'num', min: 0.1, max: 1, step: 0.05 },
    ],
  },
  {
    title: '织网 · 多跳 · Agentic · 路由', icon: <Network size={14} />, tone: 'weaved',
    fields: [
      { path: 'retrieval.graph.unweaver_enabled', label: 'UnWeaver 实体织入', type: 'bool' },
      { path: 'retrieval.graph.max_weaved', label: '最多织入块数', type: 'num', min: 0, max: 10 },
      { path: 'retrieval.graph.weave_relevance', label: '织入块须与问题相关', type: 'bool', hint: '织入块过一遍精排，低于门槛不织入' },
      { path: 'retrieval.graph.weave_min_score', label: '织入门槛', type: 'num', min: 0, max: 1, step: 0.01, nullable: true, hint: '不设 = 沿用拒答阈值 min_score' },
      { path: 'retrieval.multihop.max_hops', label: '多跳最大跳数', type: 'num', min: 1, max: 5 },
      { path: 'retrieval.agentic.max_steps', label: 'Agentic 最大步数', type: 'num', min: 1, max: 6 },
      { path: 'retrieval.router.mode', label: '查询路由方式', type: 'select', options: [['embedding', '描述向量相似度'], ['llm', 'LLM 选择']] },
    ],
  },
  {
    title: '生成', icon: <GitBranch size={14} />, tone: 'ok',
    fields: [
      { path: 'generation.refuse_when_no_context', label: '无资料时拒答', type: 'bool', hint: '敢说"不知道"是 RAG 成熟的标志' },
      { path: 'generation.llm.effort', label: '推理强度（Claude）', type: 'select', options: [['low', 'low'], ['medium', 'medium'], ['high', 'high'], ['xhigh', 'xhigh'], ['max', 'max']] },
      { path: 'generation.llm.max_tokens', label: '最大输出 token', type: 'num', min: 256, max: 64000, step: 256 },
      { path: 'generation.llm.thinking', label: '回答前思考（推理模型）', type: 'select', options: [['auto', 'auto · 仅多跳 / Agentic 模式'], ['off', 'off · 从不（最快）'], ['on', 'on · 总是（首字慢 3~5 秒）']] },
      { path: 'generation.llm.aux_model', label: '辅助模型（查询改写等）', type: 'text', placeholder: '与回答模型相同' },
    ],
  },
  {
    title: 'LanceDB 应用：记忆 · 推荐 · 多智能体 · 多模态表', icon: <Boxes size={14} />, tone: 'accent', note: '长期记忆（memories）、推荐（rec_items / rec_events）、多智能体黑板（agent_board）在 data/lancedb；多模态表（mm）每个知识库一张，在该库索引目录的 LanceDB 库里。',
    fields: [
      { path: 'multimodal.lance_table', label: '多模态表接入线上图文检索', type: 'bool', hint: '文搜图 / 问答附图多一路表内图片文字全文（ICU 分词），图搜文多一路 CLIP + 图中文字混合检索；保存索引时自动增量同步' },
      { path: 'multimodal.weight_lance_mm', label: '多模态表路权重', type: 'num', min: 0, max: 3, step: 0.1, hint: '在图片多路 RRF 融合里的权重；0 = 只同步表、不参与检索' },
      { path: 'memory.enabled', label: '长期记忆', type: 'bool', hint: '问答时按用户召回记忆用于个性化，回答后在后台抽取新记忆' },
      { path: 'memory.extract', label: '记忆抽取方式', type: 'select', options: [['llm', 'LLM（辅助模型）'], ['rule', '规则'], ['off', '不抽取']] },
      { path: 'memory.recall_top_k', label: '每轮最多带几条记忆', type: 'num', min: 0, max: 20 },
      { path: 'memory.min_score', label: '记忆召回阈值', type: 'num', min: 0, max: 1, step: 0.05, hint: '综合分 = 0.7·相似度 + 0.15·新近度 + 0.15·重要度' },
      { path: 'memory.half_life_days', label: '记忆新近度半衰期（天）', type: 'num', min: 1, max: 3650 },
      { path: 'recommend.enabled', label: '个性化推荐记录交互', type: 'bool', hint: '问答里的提问、被引用的章节记为推荐交互' },
      { path: 'recommend.mmr_lambda', label: '推荐多样化 λ', type: 'num', min: 0, max: 1, step: 0.05, hint: '1 只看相关，0 只看多样' },
      { path: 'agents.max_review_rounds', label: '审校最多打回几轮', type: 'num', min: 0, max: 4 },
      { path: 'agents.research_top_k', label: '研究员每个子问题带回资料数', type: 'num', min: 1, max: 12 },
      { path: 'agents.use_memory', label: '智能体参考用户记忆', type: 'bool' },
    ],
  },
];

const getPath = (o: any, path: string) => path.split('.').reduce((a, k) => a?.[k], o);
function setPath(o: any, path: string, v: any) {
  const ks = path.split('.');
  let cur = o;
  for (const k of ks.slice(0, -1)) cur = cur[k] ??= {};
  cur[ks[ks.length - 1]] = v;
}

export function SettingsPage() {
  const { toast, refreshStatus, refreshKbs } = useApp();
  const [data, setData] = useState<SettingsPayload | null>(null);
  const [edits, setEdits] = useState<Record<string, any>>({});
  const [saving, setSaving] = useState(false);
  const [mineru, setMineru] = useState<MineruTiersInfo | null>(null);
  const [verify, setVerify] = useState<{ loading: boolean; ok?: boolean; output?: string }>({ loading: false });
  const [conn, setConn] = useState<{ loading: boolean; result?: MineruConnectionTest }>({ loading: false });
  const [presets, setPresets] = useState<PresetsPayload | null>(null);
  const [preset, setPreset] = useState<string | null>(null);
  const loadTiers = () => api.system.mineruTiers().then(setMineru).catch(() => setMineru(null));

  const load = () => api.settings.get().then((d) => { setData(d); setEdits({}); }).catch((e) => toast(e instanceof Error ? e.message : String(e), 'bad'));
  useEffect(() => { load(); loadTiers(); api.settings.presets().then(setPresets).catch(() => setPresets(null)); }, []); // eslint-disable-line

  // 自动重建索引的进度：保存了入库类参数后后台自动重建，这里轮询直到完成
  const [reindex, setReindex] = useState<ReindexStatus | null>(null);
  const [watch, setWatch] = useState(true);
  useEffect(() => {
    if (!watch) return;
    let stop = false;
    const tick = async () => {
      try {
        const st = await api.settings.reindexStatus();
        if (stop) return;
        setReindex((prev) => {
          if (prev?.busy && !st.busy) {
            const failed = st.recent.filter((j) => j.status === 'failed');
            const partial = st.recent.filter((j) => j.status === 'done' && j.failedDocs?.length);
            if (failed.length) toast(`索引重建失败：${failed.map((j) => `${j.name}（${j.error ?? '未知错误'}）`).join('、')}`, 'bad');
            else if (partial.length) toast(`索引已重建，但有文档入库失败：${partial.map((j) => `${j.name}：${j.failedDocs!.join('、')}`).join('；')}`, 'bad');
            else toast('索引已按新配置自动重建完成', 'ok');
            refreshStatus();
            refreshKbs().catch(() => undefined);
          }
          return st;
        });
        if (!st.busy) setWatch(false);
      } catch { /* 后端重启中，下一轮再试 */ }
    };
    tick();
    const t = window.setInterval(tick, 2000);
    return () => { stop = true; window.clearInterval(t); };
  }, [watch]); // eslint-disable-line

  const dirty = Object.keys(edits).length;
  const needsReindex = useMemo(() => GROUPS.some((g) => g.reindex && g.fields.some((f) => f.type !== 'secret' && f.path in edits)), [edits]);
  const val = (path: string) => (path in edits ? edits[path] : getPath(data?.effective, path));
  const change = (path: string, v: any) => setEdits((e) => {
    const next = { ...e };
    const put = (p: string, x: any) => { if (JSON.stringify(x) === JSON.stringify(getPath(data?.effective, p))) delete next[p]; else next[p] = x; };
    put(path, v);
    // 云端 API 不提供 flash / basic：切过去时把档位一并换成 standard，免得界面显示的档位和实际用的不一致
    if (path === 'parsing.mineru.mode' && v === 'cloud' && !['standard', 'advanced'].includes(String(next['parsing.mineru.tier'] ?? getPath(data?.effective, 'parsing.mineru.tier')))) put('parsing.mineru.tier', 'standard');
    return next;
  });
  // 密钥：输入框清空 = 不修改（撤销这项改动）；要删除已保存的密钥用旁边的「清除」
  const changeSecret = (path: string, v: string) => setEdits((e) => {
    const next = { ...e };
    if (v === '') delete next[path]; else next[path] = v;
    return next;
  });

  // 把预设展开成"路径 → 值"，只保留与当前生效值不同的项；不立即保存，可以在下面继续微调
  const applyPreset = (p: StrategyPreset) => {
    const flat: Record<string, any> = {};
    const walk = (o: any, prefix: string) => { for (const [k, v] of Object.entries(o)) { if (v && typeof v === 'object' && !Array.isArray(v)) walk(v, `${prefix}.${k}`); else flat[`${prefix}.${k}`] = v; } };
    walk(p.retrieval, 'retrieval');
    const next: Record<string, any> = {};
    for (const [path, v] of Object.entries(flat)) if (JSON.stringify(v) !== JSON.stringify(getPath(data?.effective, path))) next[path] = v;
    setEdits(next);
    setPreset(p.key);
    toast(Object.keys(next).length ? `已填入「${p.name}」的 ${Object.keys(next).length} 项改动，检查后点保存` : `当前配置已经与「${p.name}」一致`, 'ok');
  };

  const save = async () => {
    const overrides: Record<string, any> = {};
    for (const [p, v] of Object.entries(edits)) setPath(overrides, p, v);
    setSaving(true);
    try {
      const r = await api.settings.update(overrides);
      toast(r.reindexScheduled ? '已保存，正在后台自动重建索引' : '策略已保存并立即生效', 'ok');
      setWatch(true);
      await load();
      loadTiers();
      refreshStatus();
      refreshKbs().catch(() => undefined);
    } catch (e) { toast(e instanceof Error ? e.message : String(e), 'bad'); } finally { setSaving(false); }
  };
  const reset = async () => {
    if (!window.confirm('恢复为 config.yaml 默认值？')) return;
    const r = await api.settings.reset();
    toast(r.reindexScheduled ? '已恢复默认，正在后台自动重建索引' : '已恢复默认', 'ok');
    setWatch(true);
    await load();
    refreshStatus();
  };

  if (!data) return <Spinner label="加载配置…" />;
  // MinerU 档位：全部档位都列出、都可选；标注状态（可用 / 需下载模型 / 未安装 / 由解析服务提供）与按硬件的推荐
  const tierOptions = (fallback: [string, string][], current: string): [string, string][] => {
    if (!mineru?.tiers.length) return fallback;
    const opts = mineru.tiers.map((t): [string, string] => [t.tier, `${t.tier} · ${t.label}${t.recommended ? ' · 推荐' : ''}${t.status === 'ready' || t.status === 'remote' ? '' : `（${t.statusLabel}）`}`]);
    return current && !opts.some(([v]) => v === current) ? [...opts, [current, current]] : opts;
  };
  const runVerify = async (tier: string) => {
    setVerify({ loading: true });
    try { const r = await api.system.mineruVerify(tier); loadTiers(); setVerify({ loading: false, ok: r.ok, output: r.output }); } catch (e) { setVerify({ loading: false, ok: false, output: e instanceof Error ? e.message : String(e) }); }
  };
  const tierDetail = (selected: string) => {
    const t = mineru?.tiers.find((x) => x.tier === selected);
    if (!mineru || !t) return null;
    const tone = t.status === 'ready' || t.status === 'remote' ? 'ok' : 'warn';
    const remoteName = mineru.remote?.kind === 'cloud' ? '云端 API' : '解析服务';
    return (
      <div className="chunk col gap-6" style={{ margin: '2px 0 10px' }}>
        <div className="row gap-6 wrap">
          <Badge mono tone="accent">{t.tier}</Badge>
          <Badge tone={tone}>{t.statusLabel}</Badge>
          {t.recommended ? <Badge tone="info"><CheckCircle2 size={10} />{mineru.remote ? '推荐' : '本机推荐'}</Badge> : null}
          <Badge mono>质量 {t.quality}</Badge><Badge mono>速度 {t.speed}</Badge>
        </div>
        <div className="small">{t.use}</div>
        {mineru.remote ? <div className="xs muted">由{remoteName}运行，本机不需要模型包和显存</div> : <div className="xs muted">模型：{t.models} · 运行要求：{t.requires}</div>}
        <div className="xs muted">支持格式：{t.formats}</div>
        {t.status === 'not_installed' ? <div className="xs text-warning">本机未安装 MinerU：档位可以先选好，安装后对文档点「重新解析」即生效；在此之前由内置解析器处理。<code style={{ marginLeft: 4 }}>uv pip install -U "mineru[torch]&gt;=4.0,&lt;5"</code></div> : null}
        {t.status === 'needs_models' && t.download ? (
          <div className="row gap-8 wrap xs">
            <span className="muted">首次使用需下载模型包（约 {t.modelGb}GB）：</span><code>{t.download}</code>
            <Button size="xs" loading={verify.loading} onClick={() => runVerify(t.tier)}><ShieldCheck size={11} />校验模型</Button>
          </div>
        ) : null}
        {t.modelRepos.length ? <div className="row gap-4 wrap">{t.modelRepos.map((r) => <Badge key={r.name} mono tone={r.ready ? 'ok' : 'warn'} title={r.root}>{r.name} · {r.ready ? '已就绪' : `缺 ${r.missing} 项`}</Badge>)}</div> : null}
        {t.status === 'missing_deps' ? <div className="xs text-warning">缺少运行依赖：<code>{t.missingModules.join(', ')}</code>。安装 <code>mineru[torch]</code>（Torch 小模型）或 <code>mineru[full]</code>（vLLM / LMDeploy）后重试。</div> : null}
        {t.status === 'remote_missing' && mineru.remote?.kind === 'cloud' ? <div className="xs text-warning">云端 API 不提供这个档位，解析时会自动改用 <code>{mineru.effectiveTier}</code>。</div> : null}
        {t.status === 'remote_missing' && mineru.remote?.kind !== 'cloud' ? <div className="xs text-warning">解析服务 {mineru.remote?.url} 的 /v1/tiers 没有列出这个档位，解析时会自动改用 <code>{mineru.effectiveTier}</code>；要用这一档，用 <code>mineru-kit api-server --tier {t.tier}</code> 启动服务。</div> : null}
        {mineru.remote && !mineru.remote.reachable ? <div className="xs text-warning">{remoteName} {mineru.remote.url} 暂时无法访问（{mineru.remote.error ?? '无响应'}）{mineru.remote.kind === 'cloud' ? '，文档会改用内置解析器处理。' : '，档位以服务端启动参数为准。'}</div> : null}
        {verify.output !== undefined ? <pre className={`pre xs ${verify.ok ? 'text-success' : 'text-warning'}`} style={{ margin: 0, maxHeight: 140, overflow: 'auto' }}>{verify.ok ? '模型包完整。\n' : ''}{verify.output}</pre> : null}
        {!t.recommended ? <div className="xs faint">{mineru.remote ? '建议' : '按本机硬件的建议'}：{mineru.recommended.reason}。<button className="btn ghost xs" onClick={() => change('parsing.mineru.tier', mineru.recommended.tier)}>改用 {mineru.recommended.tier}</button></div> : null}
      </div>
    );
  };
  const runConnTest = async () => {
    setConn({ loading: true });
    try { const r = await api.system.mineruTest(); loadTiers(); setConn({ loading: false, result: r }); } catch (e) { setConn({ loading: false, result: { ok: false, mode: 'cloud', error: e instanceof Error ? e.message : String(e) } }); }
  };
  const mb = (bytes?: number) => (bytes ? `${Math.round(bytes / 1048576)}MB` : '—');
  const modeDetail = (mode: string) => {
    if (mode !== 'cloud' && mode !== 'server') return null;
    const unsaved = ['parsing.mineru.mode', 'parsing.mineru.cloud.api_key', 'parsing.mineru.api_url', 'parsing.mineru.api_key'].some((p) => p in edits);
    const r = conn.result;
    return (
      <div className="chunk col gap-6" style={{ margin: '2px 0 10px' }}>
        {mode === 'cloud' ? (
          <div className="xs muted">文档会<b>上传到 MinerU 官方服务器</b>（mineru.net）解析，结果取回后在本机切片入库；本机不需要模型包和显存，适合内存小、没有 GPU 的机器用高质量档位。不填密钥时以匿名身份调用，有页数 / 频率限制；涉密文档请用本机或自托管方式。</div>
        ) : (
          <div className="xs muted">连接你自己部署的 <code>mineru-kit api-server</code>（与云端 API 同一套 v1 接口），文档只发给这台服务。</div>
        )}
        <div className="row gap-8 wrap xs">
          <Button size="xs" loading={conn.loading} disabled={unsaved} onClick={runConnTest}><ShieldCheck size={11} />测试连接</Button>
          {unsaved ? <span className="faint">先保存，再测试连接</span> : null}
          {!unsaved && r ? <Badge tone={r.ok ? 'ok' : 'bad'}>{r.ok ? '连接正常' : '连接失败'}</Badge> : null}
          {!unsaved && r?.ok && r.accessLevel ? <Badge mono>身份 {r.accessLevel}</Badge> : null}
          {!unsaved && r?.ok && r.tiers?.length ? <Badge mono>档位 {r.tiers.join(' / ')}</Badge> : null}
        </div>
        {!unsaved && r && !r.ok ? <div className="xs text-warning">{r.error}</div> : null}
        {!unsaved && r?.ok && r.limits ? <div className="xs faint">限额：单文件 ≤ {r.limits.max_pages_per_file ?? '—'} 页 / {mb(r.limits.max_file_size_bytes)}{r.usage ? `；本周期已处理 ${r.usage.pages_processed ?? 0} 页` : ''}{mode === 'cloud' && !r.hasKey ? '（匿名额度，填入密钥后按账号额度计）' : ''}</div> : null}
      </div>
    );
  };
  const overridden = (path: string) => getPath(data.overrides, path) !== undefined;

  return (
    <div className="page col gap-16 fade-in">
      <div className="card pad row between wrap sticky-bar">
        <div className="small muted flex-1" style={{ minWidth: 260 }}>基线来自 <code>config.yaml</code>，此处修改保存在本地数据库并覆盖基线。
          {reindex?.busy ? (
            <span className="row gap-6" style={{ display: 'inline-flex', marginLeft: 8 }}>
              <Badge tone="info"><RefreshCw size={11} className="spin" />正在自动重建索引{reindex.running[0] ? `：${reindex.running[0].name}` : ''}{reindex.queued.length ? `（另有 ${reindex.queued.length} 个排队）` : ''}</Badge>
            </span>
          ) : ' 检索类参数保存即生效；解析 / 切片类参数保存后自动重建索引。'}</div>
        <div className="row gap-8">
          {dirty ? <Badge tone="info">{dirty} 项未保存</Badge> : null}
          <Button size="sm" onClick={reset}><RotateCcw size={13} />恢复默认</Button>
          <Button size="sm" variant="primary" loading={saving} disabled={!dirty} onClick={save}><Save size={13} />保存</Button>
        </div>
      </div>
      {presets ? (
        <div className="card pad col gap-12">
          <div className="row between wrap">
            <div className="card-title" style={{ marginBottom: 0 }}><Sparkles size={14} />场景预设</div>
            <span className="xs faint">手段不是全上最好：按数据特征和瓶颈组合，每次只改一组并跑一次评测确认。</span>
          </div>
          <div className="grid grid-4">
            {presets.presets.map((p) => (
              <div key={p.key} className={`chunk col gap-6 ${preset === p.key ? 'hl' : ''}`} style={{ cursor: 'pointer' }} onClick={() => applyPreset(p)} role="button">
                <div className="row between"><span className="strong small">{p.name}</span><Badge tone="info">{p.bottleneck}</Badge></div>
                <div className="xs mono" style={{ color: 'var(--c-fused)' }}>{p.combo}</div>
                <div className="xs muted">{p.reason}</div>
              </div>
            ))}
          </div>
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>瓶颈在哪</th><th>对症的手段</th><th>对应参数</th></tr></thead>
              <tbody>{presets.guide.map((g) => <tr key={g.symptom}><td>{g.symptom}</td><td className="muted">{g.action}</td><td><div className="row gap-4 wrap">{g.fields.map((f) => <Badge key={f} mono>{f}</Badge>)}</div></td></tr>)}</tbody>
            </table>
          </div>
        </div>
      ) : null}
      {needsReindex ? <Callout>你修改了解析 / 清洗 / 切片 / 索引增强参数：保存后系统会<b>自动在后台重建</b>所有知识库的索引，已入库文档按新参数重新处理，无需手动操作。重建期间该知识库的检索结果可能暂时不完整。</Callout> : null}

      <div className="grid grid-2" style={{ alignItems: 'start' }}>
        {GROUPS.map((g) => (
          <div key={g.title} className="card">
            <div className="card-header"><h3><span className={`badge ${g.tone}`} style={{ padding: '3px 6px' }}>{g.icon}</span>{g.title}</h3>{g.reindex && g.fields.some((f) => f.path in edits) ? <Badge tone="info">保存后自动重建索引</Badge> : null}</div>
            <div className="card-body" style={{ paddingTop: 6, paddingBottom: 6 }}>
              {g.note ? <div className="xs faint" style={{ padding: '6px 0 2px' }}>{g.note}</div> : null}
              {g.fields.filter((f) => !f.when || f.when(val)).map((f) => {
                const v = val(f.path);
                const isMode = f.path === 'parsing.mineru.mode';
                const hint = f.hint ?? data.fieldDocs[f.path];
                const isTier = f.path === 'parsing.mineru.tier';
                const options = f.type === 'select' ? (isTier ? tierOptions(f.options, String(v ?? '')) : f.options) : [];
                return (
                  <div key={f.path}>
                  <div className="field inline setting-row" style={isTier || (isMode && v !== 'local') ? { borderBottom: 'none' } : undefined}>
                    <label style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 1 }}>
                      <span className="row gap-6">{f.label}{f.path in edits ? <span className="dot busy" /> : overridden(f.path) ? <span className="dot ok" title="已覆盖默认值" /> : null}</span>
                      {hint ? <span className="hint">{hint}</span> : null}
                    </label>
                    {f.type === 'bool' ? <Toggle on={!!v} onChange={(x) => change(f.path, x)} /> : f.type === 'select' ? (
                      <select style={{ width: isTier ? 300 : 220 }} value={v ?? ''} onChange={(e) => change(f.path, e.target.value === '' && f.path.endsWith('model_source') ? null : e.target.value)}>{options.map(([val2, label]) => <option key={val2} value={val2}>{label}</option>)}</select>
                    ) : f.type === 'text' ? (
                      <input type="text" style={{ width: 220 }} value={v ?? ''} placeholder={f.placeholder} onChange={(e) => change(f.path, e.target.value.trim() === '' ? null : e.target.value.trim())} />
                    ) : f.type === 'secret' ? (
                      <span className="row gap-6">
                        <input type="password" autoComplete="new-password" style={{ width: 220 }} value={f.path in edits ? edits[f.path] : ''} placeholder={getPath(data.effective, f.path) === MASKED ? '已保存，留空则不修改' : f.placeholder} onChange={(e) => changeSecret(f.path, e.target.value.trim())} />
                        {getPath(data.effective, f.path) === MASKED && !(f.path in edits) ? <button className="btn ghost xs" title="删除已保存的密钥（保存后生效）" onClick={() => setEdits((e) => ({ ...e, [f.path]: '' }))}>清除</button> : null}
                        {edits[f.path] === '' ? <span className="xs text-warning">保存后清除</span> : null}
                      </span>
                    ) : (
                      <input type="number" style={{ width: 100 }} min={f.min} max={f.max} step={f.step ?? 1} value={v ?? ''} placeholder={f.nullable ? '不设' : ''} onChange={(e) => change(f.path, e.target.value === '' ? (f.nullable ? null : getPath(data.effective, f.path)) : Number(e.target.value))} />
                    )}
                  </div>
                  {isTier ? tierDetail(String(v ?? '')) : null}
                  {isMode ? modeDetail(String(v ?? 'local')) : null}
                  </div>
                );
              })}
              {g.title === '向量检索调优' ? <VectorBenchmarkCard pcaDim={val('vector_store.search.pca_dim') ?? null} quantize={String(val('vector_store.search.quantize') ?? 'none')} rescore={!!val('vector_store.search.rescore')} oversample={Number(val('vector_store.search.oversample') ?? 4)} /> : null}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
