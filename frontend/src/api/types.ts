// 与 app/schemas.py 一一对应（camelCase）；评测 / 覆盖分析报告沿用 prism 的 snake_case 字段。

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string | null;
  category: string | null;
  language: string | null;
  createdAt: string;
  updatedAt: string;
  documentCount: number;
  chunkCount: number;
  entityCount: number;
  indexEmbedder: string | null;
  needsReindex: boolean;
  reindexing: boolean;
}

export type DocStatus = 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export interface Document {
  id: string;
  kbId: string;
  fileName: string;
  fileSize: number | null;
  fileFormat: string | null;
  status: DocStatus;
  parser: string | null;
  parserTier: string | null;
  chunkCount: number;
  parentCount: number;
  entityCount: number;
  errorMessage: string | null;
  title: string | null;
  version: string | null;
  validFrom: string | null;
  validTo: string | null;
  tags: string[];
  department: string | null;
  accessLevel: number;
  allowedDepartments: string[];
  autoMeta?: Partial<Record<MetaField, MetaSuggestion & { applied?: boolean }>>;
  editBackups?: number;            // 校对版备份数（放弃校对 / 恢复前自动备份）
  createdAt: string;
  updatedAt: string;
}

/** 校对版备份：放弃校对重新解析、或恢复另一份备份之前自动保存的校对版 Markdown。 */
export interface EditBackup { name: string; createdAt: string; chars: number; preview: string; identical: boolean }

export interface ChunkView {
  chunkId: string;
  kind: 'parent' | 'child';
  text: string;
  breadcrumbs: string[];
  tokenCount: number;
  pageStart: number | null;
  pageEnd: number | null;
  blockTypes: string[];
  childIds: string[];
  parentId: string | null;
  contextSummary: string | null;
  hypotheticalQuestions: string[];
  keywords?: string[];
  aliases?: string[];
}

// --- 语料治理 ------------------------------------------------------------------------
export type ChunkIssue = 'garbled' | 'duplicate' | 'near_duplicate' | 'low_info' | 'too_short' | 'oversized';
export type DocIssue = 'expired' | 'version_conflict' | 'empty' | 'poor_parse' | 'missing_meta';
export type Severity = 'high' | 'medium' | 'low';
export interface HealthChunk {
  chunkId: string; docId: string; docTitle: string; fileName?: string | null; breadcrumb: string; issues: ChunkIssue[]; details: string[];
  preview: string; tokens: number; fingerprint: string; page: number | null; severity: Severity;
  duplicateOf: { chunkId: string; docId: string; docTitle: string; breadcrumb: string } | null;
}
export interface HealthDoc {
  docId: string; title: string; fileName?: string | null; synthetic?: boolean; version: string | null; validFrom: string | null; validTo: string | null;
  parents: number; children: number; flagged: number; score: number; issues: { type: DocIssue; label: string; severity: Severity; detail: string }[];
}
export interface HealthConflict { key: string; keepDocId: string; docs: { docId: string; title: string; version: string | null; validFrom: string | null; validTo: string | null; keep: boolean }[] }
export interface HealthReport {
  score: number; grade: string;
  totals: { docs: number; parents: number; children: number; flagged: number; excluded: number };
  issueCounts: Record<ChunkIssue, number>; docIssueCounts: Record<DocIssue, number>;
  issueLabels: Record<ChunkIssue, { label: string; severity: Severity; action: string }>;
  chunks: HealthChunk[]; docs: HealthDoc[]; conflicts: HealthConflict[]; recommendations: string[];
  excluded: { fingerprint: string; reason: string; docId: string; fileName?: string | null; breadcrumb: string; preview: string; at: string }[];
}
export interface AliasSuggestion { terms: string[]; count: number; evidence: string }
export type FaqStatus = 'pending' | 'approved' | 'rejected' | 'gap';
export interface FaqItem {
  id: string; intent: string; question: string; answer: string; status: FaqStatus; method: 'llm' | 'extractive' | 'none' | 'manual';
  bucket: 'none' | 'partial' | 'manual'; coverageScore: number | null; note?: string; published: boolean; createdAt: string;
  sources: { chunkId: string; docId: string; breadcrumb: string; snippet: string }[];
}
export interface AugmentState {
  items: FaqItem[]; docId: string | null; publishedAt: string | null; llm: boolean;
  coverage: { available: boolean; n: number; gaps: number; pendingGaps: { intent: string; bucket: 'none' | 'partial'; score: number }[] };
}
export type MetaField = 'title' | 'version' | 'valid_from' | 'valid_to' | 'department' | 'tags' | 'access_level';
export interface MetaSuggestion { value: string | number | string[]; source: 'rule' | 'llm' | 'tfidf' | 'manual'; evidence: string }
export interface MetaRow {
  docId: string; fileName: string; current: Record<MetaField, string | number | string[] | null>;
  suggestions: Partial<Record<MetaField, MetaSuggestion & { empty: boolean; same: boolean }>>;
}

export type RetrievalMode = 'hybrid' | 'multihop' | 'agentic';

export interface ScoredItem {
  chunkId: string;
  score: number;
  rank: number;
  route: string;
  text: string;
  breadcrumb: string;
  docId: string;
  fileName: string | null;
}

export interface ContextView {
  chunkId: string;
  docId: string;
  fileName: string | null;
  breadcrumb: string;
  text: string;
  compressedText: string | null;
  score: number;
  tokenCount: number;
  pageStart: number | null;
  pageEnd: number | null;
  matchedChildIds: string[];
  weaved: boolean;
  weavedEntity: string | null;
}

export interface WeavedRelation {
  entity: string;
  relation_type: string;
  definition: string;
  source_chunk_id: string;
  triggered_by: string;
  target: string | null;
}

export interface HopTrace {
  hop: number;
  query: string;
  found_chunk_ids: string[];
  note: string;
}

export interface RetrievalResponse {
  query: string;
  strategy: string;
  queries: string[];
  hydeDocument: string | null;
  notes: string[];
  routes: Record<string, ScoredItem[]>;
  fused: ScoredItem[];
  reranked: ScoredItem[];
  diversified: ScoredItem[];
  contexts: ContextView[];
  weaved: WeavedRelation[];
  hops: HopTrace[];
  compression: { mode?: string; tokens_before?: number; tokens_after?: number };
  metadataFilter?: MetadataFilterInfo | null;
  timingsMs: Record<string, number>;
  funnel: { recall: number; fused: number; reranked: number; contexts: number };
}

// 元数据过滤：字段与 prism.retrieval.metadata_filter.FIELDS 一致
export type MetadataField = 'department' | 'tags' | 'year' | 'file_format';
export type MetadataConditions = Partial<Record<MetadataField, (string | number)[]>>;
export interface MetadataFilterInfo {
  conditions: MetadataConditions;
  source: 'rule' | 'llm' | 'upstream';
  applied: boolean;
  docs_before: number;
  docs_after: number;
  reason?: 'no_match' | 'no_narrowing' | 'no_results';
}
export type MetadataFacets = { docs: number } & Record<MetadataField, { value: string | number; count: number }[]>;

// 检索策略覆盖：与 prism.config.RetrievalConfig 字段一致（snake_case，部分即可）
export interface RetrievalOverrides {
  top_n_per_route?: number;
  parallel_recall?: boolean;
  sql?: { enabled?: boolean; text_to_sql?: boolean; top_n?: number; weight?: number };
  fusion?: { method?: 'rrf' | 'convex'; rrf_k?: number; weight_dense?: number; weight_sparse?: number; convex_alpha?: number };
  rerank?: { enabled?: boolean; provider?: string; candidates?: number; keep_n?: number; top_k?: number; min_score?: number | null; min_score_ratio?: number | null };
  diversify?: { enabled?: boolean; mmr_lambda?: number };
  granularity?: { return_parents?: boolean; auto_merge_adjacent?: boolean; sentence_window?: number; max_context_tokens?: number; order?: 'score' | 'lost_in_the_middle' };
  compression?: { enabled?: boolean; mode?: 'rule' | 'llm'; ratio?: number; min_tokens?: number };
  query_transform?: { rewrite?: boolean; multi_query?: number; hyde?: boolean; decompose?: boolean; step_back?: boolean; condense_history?: boolean };
  graph?: { unweaver_enabled?: boolean; max_weaved?: number };
  multihop?: { enabled?: boolean; max_hops?: number };
  agentic?: { enabled?: boolean; max_steps?: number };
  filter_expired?: boolean;
  summary?: { enabled?: boolean; weight?: number };
  hierarchy?: { enabled?: boolean; routing?: 'auto' | 'all'; weight?: number };
  access?: { enabled?: boolean };
  metadata_filter?: { mode?: 'off' | 'rule' | 'llm'; include_missing?: boolean; fallback?: boolean };
}

export interface UserIdentity {
  level: number;
  department: string | null;
  userId?: string | null;   // 长期记忆 / 个性化推荐按它区分用户（空 = default）
}

export interface VectorBenchmark {
  ok: boolean;
  message?: string;
  backend?: string;
  vectors: number;
  dim: number;
  queries: number;
  topK: number;
  tuned: boolean;
  exactMs: number;
  tunedMs: number;
  recall: number;
  coarseDim: number | null;
  coarseBytes: number | null;
  fullBytes: number | null;
}

export interface RouteResult {
  kb_id: string;
  name: string;
  score: number;
  reason: string;
}

export interface ChatSession {
  id: string;
  kbId: string | null;
  title: string | null;
  createdAt: string;
  updatedAt: string;
  messageCount: number;
}

export interface ChatSource extends ContextView {
  index: number;
}

// ---- 多模态：图片、检索命中、检索响应
export type MultimodalMode = 'text2image' | 'image2image' | 'hybrid' | 'image2text';

export interface ImageView {
  imageId: string;
  docId: string;
  fileName: string | null;
  url: string;
  page: number | null;
  width: number;
  height: number;
  origin: string;
  caption: string;
  description: string;
  ocrText: string;
  contextText: string;
  breadcrumb: string;
  anchorChunkId: string | null;
  label: string;
}

export interface ImageHitView extends ImageView {
  score: number;
  rank: number;
  linked: boolean;
  routes: Record<string, { rank: number; score: number }>;
  index?: number;          // 问答里的 [图n] 编号
}

export interface MultimodalRouteItem {
  id: string;
  kind: 'image' | 'chunk';
  rank: number;
  score: number;
  url: string | null;
  text: string;
  label: string;
}

export interface MultimodalResponse {
  mode: MultimodalMode;
  query: string;
  hasImage: boolean;
  images: ImageHitView[];
  contexts: ContextView[];
  routes: Record<string, MultimodalRouteItem[]>;
  routeLabels: Record<string, string>;
  queryDescription: string;
  notes: string[];
  timingsMs: Record<string, number>;
  embedder: Record<string, unknown>;
}

export interface ImageEmbeddingInfo extends RuntimeInfo {
  crossModal: boolean;
  loaded: boolean;
}

export interface MultimodalStatus {
  enabled: boolean;
  images: number;
  documentsWithImages: number;
  described: number;
  withOcr: number;
  embedder: ImageEmbeddingInfo;
  indexEmbedder: { images: number; image_embedder: string | null; image_model: string | null; cross_modal: boolean | null; chunk_clip: number };
  vision: boolean;
  ocr: string | null;
  defaults: { topK: number; alpha: number; minScore: number | null; rrfK: number };
  routes?: Partial<Record<MultimodalMode, string[]>>;   // 各检索方式实际参与融合的召回路（随开关 / 模型变化）
}

export interface ImageSearchInfo {
  mode: string;
  routes: string[];
  notes: string[];
  queryDescription?: string;
  timingsMs: Record<string, number>;
}

/** 回答的引用校验：[n] 所在的句子在本轮第 n 段资料里找不到词面依据（串号 / 沿用了上一轮的编号）；
 *  number：句子里的数值在本轮资料和问题里都找不到（index 为句子引用的第一个编号，没有引用时为 0）。 */
export interface CitationIssue {
  index: number;
  reason: 'mismatch' | 'unsupported' | 'out_of_range' | 'number';
  sentence: string;
  support: number;
  suggest: number | null;            // mismatch 时更吻合的资料编号
  value?: string;                    // number 时的数值
}

export interface ChatPlan {
  citationCheck?: CitationIssue[];   // 回答生成后才有：随 done 事件下发，并存进历史消息的 trace
  kbSuggestion?: { kbId: string; name: string; routeScore: number; score: number; fileName: string | null } | null;
  // 自动路由兜底：路由选中的库答不上，换成了排名后面、分数达标的库
  routeFallback?: { fromKbId: string; fromName: string; fromScore: number | null; toKbId: string; toName: string; toScore: number; reason: string } | null;   // 选的库里没检索到资料时，可能有答案的其它知识库
  imageSearch?: ImageSearchInfo | null;
  images?: ImageHitView[];          // 仅历史消息：随 trace 持久化
  imageCitations?: number[];
  queryImage?: string | null;       // 仅历史 user 消息
  strategy: string;
  queries: string[];
  hydeDocument: string | null;
  notes: string[];
  hops: HopTrace[];
  timingsMs: Record<string, number>;
  funnel: RetrievalResponse['funnel'] | Record<string, never>;
  compression: RetrievalResponse['compression'];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources: ChatSource[];
  relations: WeavedRelation[];
  citations: number[];
  trace: ChatPlan | null;
  model: string | null;
  createdAt: string;
  images?: ImageHitView[];          // 回答携带的图片（[图n]）
  imageCitations?: number[];
  queryImage?: string | null;       // 用户随问题上传的图片
  // 仅前端流式过程中使用
  streaming?: boolean;
  meta?: { sessionId: string; kbId: string | null; mode: string; language: string; route: RouteResult[]; queryImage?: string | null };
  error?: string;
}

export type ModelCategory = 'text' | 'embedding' | 'multimodal' | 'rerank' | 'image_embedding';
export type ProviderType = 'anthropic' | 'openai_compatible' | 'local' | 'api';

export interface ModelConfig {
  id: string;
  providerName: string;
  providerType: ProviderType;
  baseUrl: string | null;
  apiKeyMasked: string;
  category: ModelCategory;
  modelName: string;
  isActive: number;
  createdAt: string;
  updatedAt: string;
}

export interface ProviderPreset {
  name: string;
  type: ProviderType;
  baseUrl: string | null;
  categories: ModelCategory[];
  models: Partial<Record<ModelCategory, string[]>>;
}

export interface RuntimeInfo {
  source: string;
  provider: string | null;
  providerType?: string | null;
  model: string | null;
  offline: boolean;
  // 重排：实际实现（cross_encoder / cohere / api / llm / lexical）与本地模型的实时状态
  kind?: string;
  state?: 'idle' | 'loading' | 'ready' | 'failed';
  cached?: boolean;
  device?: string | null;
  error?: string | null;
  fallback?: string | null;
}

export interface SystemStatus {
  project: string;
  version: string;
  coreVersion: string;
  runtime: {
    embedding: RuntimeInfo;
    llm: RuntimeInfo;
    multimodal: RuntimeInfo;
    rerank: RuntimeInfo;
    imageEmbedding: ImageEmbeddingInfo;
    vision: boolean;
    ocr: string | null;
    multimodalEnabled: boolean;
    mineruAvailable: boolean;
    mineruEnabled: boolean;
    mineruTier: string;
    mineruMode?: 'local' | 'server' | 'cloud';
    popoEnabled: boolean;
    crossEncoderAvailable: boolean;
    vectorBackend: string;
    offlineMode: boolean;
  };
  counts: { knowledgeBases: number; documents: number; chunks: number; modelConfigs: number };
  pipeline: {
    chunking: Record<string, unknown>;
    funnel: { recallTopK: number; candidates: number; rerankTopN: number; contextTopN: number };
    fusion: Record<string, unknown>;
    queryTransform: Record<string, unknown>;
    diversify: Record<string, unknown>;
    compression: Record<string, unknown>;
    graph: Record<string, unknown>;
    granularity: Record<string, unknown>;
    rerank: Record<string, unknown>;
    sparse: Record<string, unknown>;
  };
}

export interface SettingsPayload {
  effective: Record<string, any>;
  overrides: Record<string, any>;
  defaults: Record<string, any>;
  fieldDocs: Record<string, string>;
}

export interface StrategyPreset {
  key: string;
  name: string;
  combo: string;
  reason: string;
  bottleneck: string;
  retrieval: RetrievalOverrides;
}

export interface PresetsPayload {
  presets: StrategyPreset[];
  guide: { symptom: string; action: string; fields: string[] }[];
}

export interface MultimodalEvalExampleResult {
  task: string;
  question: string;
  category: string;
  ok: boolean;
  rank: number | null;
  metrics: Record<string, number>;
  top: string[];
  note: string;
}

export interface MultimodalEvalReport {
  n: number;
  ks: number[];
  task_metrics: Record<string, Record<string, number>>;
  route_metrics: Record<string, Record<string, number>>;
  examples: MultimodalEvalExampleResult[];
  seconds: number;
  embedder: Record<string, unknown>;
  markdown: string;
  name: string | null;
  kb_id: string;
  createdAt: string;
}

export interface EvalExample {
  question: string;
  relevant_doc_ids?: string[];
  relevant_sources?: string[];
  relevant_chunk_ids?: string[];
  answer_keywords?: string[];
  category?: string;
  expected_refusal?: boolean;
}

export interface EvalExampleResult {
  question: string;
  category: string;
  per_stage: Record<string, Record<string, number>>;
  error_type: string | null;
  final_top: string[];
}

export interface HardNegativeMiningConfig {
  candidate_pool: number;
  num_negatives: number;
  rank_min: number;
  rank_max: number;
  max_per_doc: number;
  exclude_same_document: boolean;
  use_teacher: boolean;
  min_positive_score: number;
  max_negative_score: number;
  min_negative_score: number;
  synthetic_negatives: number;
  absolute_margin: number;
  relative_margin: number;
  dev_fraction: number;
  seed: number;
}

export interface SuspectedPositive {
  chunk_id: string;
  doc_id: string;
  score: number;
  rank: number;
  text: string;
}

export interface SyntheticNegative {
  text: string;
  rerank_text: string;
  body: string;
  score: number | null;
  source_chunk_id: string;
}

export interface HardNegativeItem {
  query: string;
  category: string;
  pos: string[];
  neg: string[];
  rerank_pos: string[];
  rerank_neg: string[];
  positive_chunk_ids: string[];
  negative_chunk_ids: string[];
  positive_doc_ids: string[];
  negative_doc_ids: string[];
  pos_scores: number[];
  neg_scores: number[];
  negative_routes: string[][];
  negative_ranks: number[];
  split: 'train' | 'dev';
  requires_review: boolean;
  shortage?: number;
  suspected_positives?: SuspectedPositive[];
  suspected_positive_count?: number;
  easy_filtered?: number;
  synthetic_neg?: SyntheticNegative[];
  synthetic_rejected?: Record<string, number>;
}

export interface HardNegativeReport {
  kb_id: string;
  config: HardNegativeMiningConfig;
  stats: { input_examples: number; queries: number; items: number; negatives: number; train: number; dev: number; review_required: number; shortages: number; suspected_positives?: number; easy_filtered?: number; synthetic_negatives?: number; synthetic_rejected?: number };
  skipped: { question: string; reason: string; suspected_positives?: SuspectedPositive[]; easy_filtered?: number }[];
  warnings: string[];
  models: Record<string, unknown>;
  runtimeModels?: Record<string, unknown>;
  items: HardNegativeItem[];
}

export interface EvalReport {
  id: string;
  name: string | null;
  createdAt: string;
  n: number;
  ks: number[];
  stageMetrics: Record<string, Record<string, number>>;
  categoryMetrics: Record<string, Record<string, number>>;
  errorCounts: Record<string, number>;
  refusalTotal: number;
  refusalCorrect: number;
  seconds: number;
  examples: EvalExampleResult[];
  markdown: string;
  models?: CoverageModels & { overrides?: Record<string, unknown>; questionSet?: string };
  comparison?: EvalComparison | null;
}

export interface EvalComparison {
  comparable: boolean;
  reasons: string[];
  notes?: string[];
  previous: { createdAt?: string | null; name?: string | null; n: number; 'hit@1'?: number; mrr?: number; final?: Record<string, number>; refusalCorrect?: number; refusalTotal?: number };
}

// ---- 实体织网：结构拓扑（文档 → 章节 → 小节 → 实体 + 实体关系）
export type GraphNodeType = 'doc' | 'section' | 'entity';
export interface GraphNode {
  id: string;
  type: GraphNodeType;
  label: string;
  docId?: string;
  level?: number;
  path?: string;
  firstChunkId?: string;
  definitions?: { text: string; chunkId: string; docId: string; source: string }[];
  sources?: string[];
  degree: number;
  chunks?: number;
  entityType?: string;                  // policy / process / product / system / role / metric / concept / other
  entityTypeSource?: 'llm' | 'rule';    // LLM 抽取时给出，或按名称规则推断
}
export type GraphEdgeType = 'contains' | 'defines' | 'mentions' | 'relation';
export interface GraphEdge {
  source: string;
  target: string;
  type: GraphEdgeType;
  label: string;
  group?: string;                       // 关系类别（仅 relation 边）：structure / require / constraint / cause / process / replace / usage / other
  groupSource?: 'llm' | 'rule';
  docIds?: string[];                    // 关系边：从哪些文档抽出来的（按出现顺序）
}
export interface KbGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: { docs: number; sections: number; entities: number; relations: number; mentions: number; defines: number; totalEntities: number; truncated: boolean };
  config: { unweaver: boolean; llmExtraction: boolean };
  entityTypes?: { id: string; label: string; count: number }[];
  relationGroups?: { id: string; label: string; count: number; examples: string[] }[];
  relationGroupSource?: { llm?: number; rule?: number };   // 关系类别来源：LLM 抽取时给出 / 关键词规则兜底
}

export interface EntityRelation {
  entity: string;
  relation_type: string;
  definition: string;
  target: string | null;
  chunk_id: string;
  doc_id: string;
  source: string;
}

// ---- 解析能力（MinerU 档位 / 解析器矩阵 / MinerU-Popo）
export type MineruTierStatus = 'ready' | 'needs_models' | 'missing_deps' | 'not_installed' | 'remote' | 'remote_missing';

export interface MineruTier {
  tier: string;
  label: string;
  quality: string;
  speed: string;
  use: string;
  models: string;
  requires: string;
  needsModels: boolean;
  modelGb: number;
  minMemoryGb: number;
  formats: string;
  status: MineruTierStatus;
  statusLabel: string;
  selectable: boolean;
  current: boolean;
  recommended: boolean;
  download: string | null;
  modelRepos: { name: string; ready: boolean; root: string; missing: number }[];
  missingModules: string[];
}

export interface MineruTiersInfo {
  available: boolean;
  enabled: boolean;
  tier: string;
  apiUrl: string | null;
  tiers: MineruTier[];
  hardware: { system: string; machine: string; accelerator: 'apple_silicon' | 'nvidia' | 'none'; gpuMemoryGb: number | null };
  recommended: { tier: string; extra: string; reason: string };
  remote: { url: string; reachable: boolean; tiers: Record<string, boolean>; error: string | null; kind: 'cloud' | 'server'; hasKey: boolean } | null;
  source: 'remote' | 'local' | 'catalog';
  mode: 'local' | 'server' | 'cloud';          // 实际生效的解析方式
  effectiveTier: string;                        // 实际使用的档位（远程不提供所选档位时会改用别的）
}

// 远程解析（云端 API / 自托管服务）的连接测试结果
export interface MineruConnectionTest {
  ok: boolean;
  mode: 'local' | 'server' | 'cloud';
  url?: string;
  hasKey?: boolean;
  tiers?: string[];
  accessLevel?: string | null;
  usage?: { pages_processed?: number; files_processed?: number; jobs_created?: number } | null;
  limits?: { max_pages_per_file?: number; max_file_size_bytes?: number } | null;
  error?: string | null;
}

export interface ParsingInfo {
  matrix: { ext: string; parser: string | null; viaMineru: boolean; needsMineru: boolean }[];
  mineru: {
    available: boolean;
    enabled: boolean;
    tier: string;
    ocrMode: string;
    imageAnalysis: boolean;
    apiUrl: string | null;
    tiers: MineruTier[];
    recommended?: MineruTiersInfo['recommended'];
    install: string[];
  };
  popo: { enabled: boolean; dir: string; trees: number; subtasks: string[]; steps: string[] };
}

export interface PopoStatus {
  applicable: boolean;
  mineruAvailable: boolean;
  popoEnabled: boolean;
  exported: boolean;
  inferred: boolean;
  treeReady: boolean;
  usingTree: boolean;
  paths: Record<string, string>;
  commands: string[];
}

// ---- 语料准备：意图 × 知识覆盖
export type CoverageBucket = 'high' | 'partial' | 'none';

export interface CoverageIntent {
  intent: string;
  category: string;
  score: number;
  bucket: CoverageBucket;
  bucket_label: string;
  term_coverage: number;
  dense_sim: number;
  rerank_score: number | null;
  missing_terms: string[];
  top: { chunk_id: string; doc_id: string; breadcrumb: string; score: number; snippet: string; file_name?: string | null }[];
  action: string;
  note: string;
}

export interface CoverageDoc {
  doc_id: string;
  title: string;
  source: string;
  file_name?: string | null;
  version: string | null;
  valid_to: string | null;
  expired: boolean;
  chunks: number;
  hits: number;
  hit_intents: string[];
  status: 'core' | 'supporting' | 'unused' | 'expired';
  action: string;
}

export interface CoverageReport {
  kb_id: string;
  n: number;
  thresholds: { high: number; partial: number };
  buckets: Record<CoverageBucket, number>;
  coverage_rate: number;
  answerable_rate: number;
  by_category: Record<string, Record<CoverageBucket, number>>;
  intents: CoverageIntent[];
  docs: CoverageDoc[];
  recommendations: string[];
  scoring: string;
  seconds: number;
  markdown: string;
  createdAt?: string;
  models?: CoverageModels;
  comparison?: CoverageComparison | null;
}

export interface CoverageModels {
  embedder: string; embedderModel: string; embedderOffline: boolean;
  reranker: string; rerankerModel: string | null; rerankerState: string | null; rerankerFallback: boolean; rerankerEffective: string; scoring: string;
}
export interface CoverageComparison {
  comparable: boolean;
  reasons: string[];
  notes?: string[];
  sameIntents: boolean;
  previous: { createdAt?: string | null; n: number; buckets: Record<CoverageBucket, number>; coverage_rate: number; answerable_rate: number; scoring: string; models?: CoverageModels | null };
}

export interface ReindexJob {
  kbId: string;
  name: string;
  status: 'queued' | 'running' | 'done' | 'failed';
  reason: string;
  error: string | null;
  failedDocs?: string[];     // 整库重建完成，但这些文档入库失败（已标 FAILED）
  startedAt: number | null;
  finishedAt: number | null;
}

export interface ReindexStatus {
  busy: boolean;
  checking: boolean;
  running: ReindexJob[];
  queued: ReindexJob[];
  recent: ReindexJob[];
}
