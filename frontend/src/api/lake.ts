// Agent 湖 API（/api/v1/lake）：概况 / 记录 / 混合检索 / 检索质量与缺口 / 主题 / SQL / 分析智能体 / 版本 · Tag · 分支 / 派生列 / 回填 / 反馈。
import { API_BASE, ApiError } from './client';

export type LakeKind = 'question' | 'retrieval' | 'answer' | 'analysis' | 'tool_call' | 'memory' | 'board' | 'document' | 'doc_event' | 'retest' | 'note';

export const KIND_LABEL: Record<LakeKind, string> = {
  question: '问题', retrieval: '检索日志', answer: '回答', analysis: '湖分析', tool_call: '工具调用',
  memory: '长期记忆', board: '智能体黑板', document: '文档', doc_event: '文档变更', retest: '缺口复测', note: '笔记',
};

/** 查看哪个快照：主表最新 / 某个版本号或 Tag / 某个分支。 */
export interface LakeRef { version?: string | null; branch?: string | null }

export interface LakeRecord {
  id: string; kind: LakeKind; session_id: string; user_id: string; agent: string; kb_id: string; title: string; content: string;
  meta: string; refs: string; parent_id: string; status: string; score: number | null; latency_ms: number | null; feedback: number;
  ts: number; day: string; embed_model: string; [derived: string]: unknown;
}

export interface LakeHit extends LakeRecord { vectorScore: number | null; ftsScore: number | null; rrf: number; routes: string[] }

export interface LakeInfo {
  exists: boolean; rows: number; version?: number; versions?: number; bytes?: number | null; fragments?: number | null; dim: number; embedModel: string;
  derivedColumns?: { name: string; type: string }[]; tags?: number; branches?: number; indices?: { name: string; type: string; columns: string[] }[];
}

export interface LakeOverview {
  info: LakeInfo;
  stats: { rows: number; kinds: Partial<Record<LakeKind, number>>; sessions: number; users: number; agents: number; daily: ({ day: string } & Partial<Record<LakeKind, number>>)[]; firstTs: number | null; lastTs: number | null };
  cfg: { enabled: boolean; record_chat: boolean; augment: boolean; augment_top_k: number; augment_min_score: number; low_score: number };
}

export type RetestStatus = 'resolved' | 'improved' | 'answerable' | 'still_gap' | 'wrong_kb' | 'error';
export const RETEST_LABEL: Record<RetestStatus, string> = { resolved: '已解决', improved: '有改善', answerable: '能回答（检索分偏低）', still_gap: '仍是缺口', wrong_kb: '答案在其它知识库', error: '复测失败' };
export interface AltKb { kbId: string; name: string; score: number; fileName: string | null; answer: string | null }

export interface LakeGap {
  questionId: string; question: string; reasons: string[]; ts: number; sessionId: string; kbId: string; score: number | null; answer: string | null; answerId: string | null;
  retest?: { status: RetestStatus; score: number | null; ts: number; altKb?: AltKb | null };   // 缺口之后最近一次复测（altKb：能回答它的其它知识库）
  docChangesAfter?: string[];                                          // 缺口之后同一知识库里变更过的文档
}

export interface RetestResult {
  questionId: string; question: string; kbId: string; type: 'gap' | 'weak'; reasons: string[]; oldScore: number | null; newScore: number | null;
  status: RetestStatus; statusLabel: string; refused: boolean | null; answer: string; contexts: { docId: string; fileName: string | null; score: number }[]; error: string | null; ms: number;
  altKb: AltKb | null;
}
export interface RetestResponse { tested: number; counts: Record<RetestStatus, number>; results: RetestResult[]; elapsedMs: number; checkAnswer: boolean }

export interface LakeQuality {
  questions: number; answers: number; retrievals: number;
  latency: { avg: number | null; p50: number | null; p95: number | null };
  score: { avg: number | null; lowCount: number; lowThreshold: number; buckets: { label: string; count: number }[] };
  refused: number; refusalRate: number | null; feedback: { up: number; down: number };
  gapCount: number; gapRate: number | null; gaps: LakeGap[];
  historicalGapCount: number; resolvedCount: number; resolvedGaps: LakeGap[]; lastRetestTs: number | null;
  ignoredCount?: number; ignoredGaps?: LakeGap[];      // 用户标记「忽略」的问题：不算缺口，可恢复
  misroutedCount: number; misroutedGaps: LakeGap[];      // 答案在其它知识库（选错库 / 路由问题，不算缺语料）
  topDocs: { docId: string; fileName: string; retrieved: number; cited: number }[];
  byStrategy: { strategy: string; count: number; avgScore: number | null; p50LatencyMs: number | null }[];
}

export interface LakeTopics {
  k: number; questions: number;
  clusters: { label: string; keywords: string[]; size: number; gapRate: number; avgScore: number | null; samples: { id: string; question: string }[]; questionIds?: string[] }[];
  written?: { column: string; branch: string; filled: number; version: number };
}

export interface SqlResult { columns: string[]; rows: unknown[][]; rowCount: number; truncated: boolean; elapsedMs: number }

export interface AnalysisStep { tool: string; args: Record<string, unknown>; why: string; result: unknown; error: string | null; empty?: boolean; rows?: number | null; ms: number }
// 湖目录：列 / meta 键 / 指标口径 / 示例 SQL，按意图检索（分析智能体规划前先查，SQL 页右侧可点选）
export interface CatalogColumn { kind: 'column'; name: string; type: string; description: string; samples: string[]; range: string | null; nulls: number | null; notes?: string[]; doc: string; relevance: number }
export interface CatalogMeta { kind: 'meta'; name: string; recordKind: string; count: number; samples: string[]; access: string; doc: string; relevance: number }
export interface CatalogMetric { kind: 'metric'; name: string; unit: string; definition: string; sql: string; source: 'builtin' | 'custom'; id?: string; doc: string; relevance: number }
export interface CatalogExample { kind: 'example'; name: string; sql: string; source: 'builtin' | 'history' | 'custom'; id?: string; doc: string; relevance: number }
// 人工维护的条目（湖里 agent = catalog 的 note 记录）：指标口径 / 示例 / 列备注
export interface CatalogCustom { id: string; type: 'metric' | 'example' | 'column'; name: string; content: string; meta: Record<string, string>; day: string; user_id: string }
export interface CatalogEntryInput { type: 'metric' | 'example' | 'column'; name?: string; sql?: string; definition?: string; unit?: string; column?: string; note?: string }
export interface LakeCatalog { columns: CatalogColumn[]; metaKeys: CatalogMeta[]; metrics: CatalogMetric[]; examples: CatalogExample[]; totals: Record<string, number>; custom?: CatalogCustom[] }
export interface AnalysisResult { question: string; planner: 'llm' | 'rule'; summarizer: 'llm' | 'template'; rounds?: number; steps: AnalysisStep[]; answer: string; elapsedMs: number; recordId?: string | null; catalog?: LakeCatalog | null }

export interface LakeVersion { version: number; timestamp: string | null; metadata: Record<string, string>; tags: string[] }
export interface LakeBranch { name: string; parentBranch: string; parentVersion: number | null; version: number; rows: number; derivedColumns: string[] }
export interface LakeVersions { versions: LakeVersion[]; tags: { name: string; version: number }[]; branches: LakeBranch[]; derivedColumns?: string[] }   // derivedColumns：主表上的派生列
export interface BranchDiff { branch: string; addedColumns: string[]; removedColumns: string[]; rowsOnlyInBranch: number; rowsOnlyInMain: number; mainRows: number; branchRows: number; columnValues: Record<string, { value: string; count: number }[]> }

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}/lake${path}`, { ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) } });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body);
    } catch { /* ignore */ }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

const post = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) });
const qs = (params: Record<string, string | number | null | undefined>) => {
  const p = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => { if (v !== null && v !== undefined && v !== '') p.set(k, String(v)); });
  const s = p.toString();
  return s ? `?${s}` : '';
};

export const lakeApi = {
  overview: (ref: LakeRef = {}) => req<LakeOverview>(`/overview${qs({ version: ref.version, branch: ref.branch })}`),
  quality: (ref: LakeRef = {}, days?: number | null) => req<LakeQuality>(`/quality${qs({ version: ref.version, branch: ref.branch, days })}`),
  records: (p: { kind?: string; sessionId?: string; parentId?: string; limit?: number; offset?: number } & LakeRef = {}) =>
    req<{ items: LakeRecord[]; total: number }>(`/records${qs(p as Record<string, string | number | null | undefined>)}`),
  record: (id: string) => req<{ record: LakeRecord; parent: LakeRecord | null; children: LakeRecord[] }>(`/records/${encodeURIComponent(id)}`),
  addNote: (content: string, title = '') => req<{ inserted: number }>('/records', post({ content, title })),
  search: (body: { query: string; kinds?: string[] | null; mode?: 'hybrid' | 'vector' | 'fts'; topK?: number; days?: number | null; where?: string | null } & LakeRef) => req<LakeHit[]>('/search', post(body)),
  topics: (body: { k?: number | null; writeColumn?: string | null } & LakeRef) => req<LakeTopics>('/topics', post(body)),
  schema: (ref: LakeRef = {}) => req<{ columns: { name: string; type: string }[]; doc: string }>(`/schema${qs({ version: ref.version, branch: ref.branch })}`),
  sql: (sql: string, ref: LakeRef = {}) => req<SqlResult>('/sql', post({ sql, ...ref })),
  analyze: (question: string, ref: LakeRef = {}) => req<AnalysisResult>('/analyze', post({ question, ...ref })),
  catalog: (q: string, ref: LakeRef = {}) => req<LakeCatalog>(`/catalog${qs({ q: q || undefined, version: ref.version, branch: ref.branch })}`),
  catalogAdd: (entry: CatalogEntryInput) => req<{ inserted: number }>('/catalog', post(entry)),
  retest: (body: { limit?: number; includeWeak?: boolean; checkAnswer?: boolean }) => req<RetestResponse>('/retest', post(body)),
  feedback: (answerId: string, value: -1 | 0 | 1) => req<{ ok: boolean }>('/feedback', post({ answerId, value })),
  sync: () => req<Record<string, { inserted: number; skipped: number }>>('/sync', post({})),
  optimize: () => req<LakeInfo>('/optimize', post({})),
  versions: () => req<LakeVersions>('/versions'),
  /** 删除主表 v{version} 及更早的全部版本（Lance 只能按时间整段清理）。kept：被保留的分支起点。 */
  /** 删除一条记录（默认连同 parent_id 指向它的关联记录）；dryRun 只返回将被删除的 id。删掉的记录回填历史数据时不会再写回来。 */
  deleteRecord: (id: string, opts: { cascade?: boolean; dryRun?: boolean } = {}) => req<{ deleted: string[]; dryRun: boolean }>(`/records/${encodeURIComponent(id)}${qs({ cascade: opts.cascade === false ? 'false' : null, dryRun: opts.dryRun ? 'true' : null })}`, { method: 'DELETE' }),
  /** 批量删除记录（如删除某个主题下的全部问题），默认连同关联记录。 */
  deleteRecords: (ids: string[], opts: { cascade?: boolean; dryRun?: boolean } = {}) => req<{ deleted: string[]; dryRun: boolean }>('/records/delete', post({ ids, cascade: opts.cascade ?? true, dryRun: !!opts.dryRun })),
  /** 忽略 / 恢复一个知识缺口（问题记录保留，只是不再算缺口）。 */
  ignoreGap: (questionId: string, ignored = true) => req<{ ok: boolean; ignored: boolean }>(`/gaps/${encodeURIComponent(questionId)}/ignore`, post({ ignored })),
  deleteVersions: (version: number) => req<{ deleted: number[]; kept: number[]; version: number }>(`/versions/${version}`, { method: 'DELETE' }),
  createTag: (name: string, version?: number | null) => req<{ name: string; version: number }>('/tags', post({ name, version: version ?? null })),
  deleteTag: (name: string) => req<{ ok: boolean }>(`/tags/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  createBranch: (name: string, from: { fromTag?: string | null; fromVersion?: number | null }) => req<{ name: string; version: number; rows: number }>('/branches', post({ name, ...from })),
  deleteBranch: (name: string) => req<{ ok: boolean }>(`/branches/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  diffBranch: (name: string) => req<BranchDiff>(`/branches/${encodeURIComponent(name)}/diff`),
  promoteBranch: (name: string, columns?: string[] | null) => req<{ columns: string[]; updatedRows: number; insertedRows: number; version: number }>(`/branches/${encodeURIComponent(name)}/promote`, post({ columns: columns ?? null })),
  addColumn: (body: { name: string; expr?: string | null; branch?: string | null }) => req<{ column: string; branch: string; filled: number; version: number }>('/columns', post(body)),
  dropColumn: (name: string, branch?: string | null) => req<{ ok: boolean }>(`/columns/${encodeURIComponent(name)}${qs({ branch })}`, { method: 'DELETE' }),
};
