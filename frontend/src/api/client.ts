// 后端 API 客户端。开发态通过 Vite 代理 `/api` → FastAPI（VITE_API_TARGET，默认 http://localhost:8000）；生产态由 FastAPI 静态托管。
import type { AliasSuggestion, AugmentState, FaqStatus, HealthReport, MetaField, MetaRow, ChatMessage, ChatPlan, ChatSession, ChatSource, ChunkView, CoverageReport, Document, EntityRelation, EvalExample, EvalReport, ImageHitView, ImageView, KnowledgeBase, MultimodalEvalReport, MineruTiersInfo, ModelConfig, MultimodalMode, MultimodalResponse, MultimodalStatus, ParsingInfo, PopoStatus, EditBackup, PresetsPayload, ProviderPreset, RetrievalMode, RetrievalOverrides, RetrievalResponse, RouteResult, RuntimeInfo, SettingsPayload, SystemStatus, WeavedRelation, ReindexStatus, UserIdentity, VectorBenchmark, KbGraph } from './types';
import type { HardNegativeMiningConfig, HardNegativeReport, MetadataFacets } from './types';

export const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '/api/v1';

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: init?.body instanceof FormData ? init?.headers : { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const json = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) });
const put = (body: unknown): RequestInit => ({ method: 'PUT', body: JSON.stringify(body) });

export const api = {
  system: {
    status: () => request<SystemStatus>('/system/status'),
    parsing: () => request<ParsingInfo>('/system/parsing'),
    mineruTiers: () => request<MineruTiersInfo>('/system/mineru/tiers'),
    mineruVerify: (tier?: string) => request<{ ok: boolean; ran: boolean; output: string }>(`/system/mineru/verify${tier ? `?tier=${encodeURIComponent(tier)}` : ''}`, { method: 'POST' }),
  },
  kb: {
    list: () => request<KnowledgeBase[]>('/knowledge-bases/'),
    create: (body: { name: string; description?: string; category?: string; language?: string }) => request<KnowledgeBase>('/knowledge-bases/', json(body)),
    get: (id: string) => request<KnowledgeBase>(`/knowledge-bases/${id}`),
    update: (id: string, body: { name?: string; description?: string; category?: string; language?: string }) => request<KnowledgeBase>(`/knowledge-bases/${id}`, put(body)),
    remove: (id: string) => request<{ message: string }>(`/knowledge-bases/${id}`, { method: 'DELETE' }),
    stats: (id: string) => request<Record<string, unknown>>(`/knowledge-bases/${id}/stats`),
    entities: (id: string) => request<EntityRelation[]>(`/knowledge-bases/${id}/entities`),
    graph: (id: string, depth = 2) => request<KbGraph>(`/knowledge-bases/${id}/graph?depth=${depth}`),
    reindex: (id: string) => request<{ message: string }>(`/knowledge-bases/${id}/reindex`, { method: 'POST' }),
  },
  documents: {
    supportedFormats: () => request<{ extensions: string[]; mineru: boolean }>('/documents/supported-formats'),
    list: (kbId: string) => request<Document[]>(`/documents/?kbId=${encodeURIComponent(kbId)}`),
    get: (id: string) => request<Document>(`/documents/${id}`),
    upload: (kbId: string, file: File) => {
      const fd = new FormData();
      fd.append('file', file);
      return request<Document>(`/documents/upload?kbId=${encodeURIComponent(kbId)}`, { method: 'POST', body: fd });
    },
    remove: (id: string) => request<{ message: string }>(`/documents/${id}`, { method: 'DELETE' }),
    markdown: (id: string) => request<{ id: string; fileName: string; markdownContent: string }>(`/documents/${id}/markdown`),
    updateMarkdown: (id: string, markdownContent: string) => request<Document>(`/documents/${id}/markdown`, put({ markdownContent })),
    updateMeta: (id: string, body: { title?: string; version?: string; validFrom?: string; validTo?: string; tags?: string[]; department?: string; accessLevel?: number; allowedDepartments?: string[] }) => request<Document>(`/documents/${id}/meta`, put(body)),
    chunks: (id: string) => request<ChunkView[]>(`/documents/${id}/chunks`),
    // 人工校对过的文档默认保留校对内容（用校对后的 Markdown 重建）；discardEdits = 放弃校对、回到源文件重新解析（后端先备份校对版）
    reprocess: (id: string, discardEdits = false) => request<Document>(`/documents/${id}/reprocess`, json({ discardEdits })),
    editBackups: (id: string) => request<EditBackup[]>(`/documents/${id}/edit-backups`),
    restoreEditBackup: (id: string, name: string) => request<Document>(`/documents/${id}/edit-backups/${encodeURIComponent(name)}/restore`, { method: 'POST' }),
    deleteEditBackup: (id: string, name: string) => request<{ ok: boolean }>(`/documents/${id}/edit-backups/${encodeURIComponent(name)}`, { method: 'DELETE' }),
    popo: (id: string) => request<PopoStatus>(`/documents/${id}/popo`),
    popoExport: (id: string) => request<PopoStatus & { written: Record<string, string> }>(`/documents/${id}/popo-export`, { method: 'POST' }),
    parseTemp: (file: File) => {
      const fd = new FormData();
      fd.append('file', file);
      return request<{ content: string; fileName: string; chars: number }>('/documents/parse-temp', { method: 'POST', body: fd });
    },
  },
  retrieval: {
    playground: (kbId: string, body: { query: string; mode?: RetrievalMode; topK?: number; useTransform?: boolean; overrides?: RetrievalOverrides; filters?: Record<string, unknown>; user?: UserIdentity | null }) =>
      request<RetrievalResponse>(`/retrieval/${kbId}/playground`, json(body)),
    vectorBenchmark: (kbId: string, body: { pcaDim?: number | null; quantize?: 'none' | 'int8'; rescore?: boolean; oversample?: number; queries?: number; topK?: number }) =>
      request<VectorBenchmark>(`/retrieval/${kbId}/vector-benchmark`, json(body)),
    facets: (kbId: string, user?: UserIdentity | null) =>
      request<MetadataFacets>(`/retrieval/${kbId}/facets${user ? `?level=${user.level}${user.department ? `&department=${encodeURIComponent(user.department)}` : ''}` : ''}`),
    route: (query: string, topN = 1, mode: 'embedding' | 'llm' = 'embedding') => request<RouteResult[]>('/retrieval/route', json({ query, topN, mode })),
  },
  multimodal: {
    status: (kbId: string) => request<MultimodalStatus>(`/multimodal/${kbId}/status`),
    images: (kbId: string, docId?: string) => request<ImageView[]>(`/multimodal/${kbId}/images${docId ? `?docId=${encodeURIComponent(docId)}` : ''}`),
    search: (kbId: string, body: { mode: MultimodalMode; query?: string; imageB64?: string | null; topK?: number; alpha?: number; minScore?: number | null; docIds?: string[] }) =>
      request<MultimodalResponse>(`/multimodal/${kbId}/search`, json(body)),
  },
  chat: {
    sessions: (kbId?: string | null) => request<ChatSession[]>(`/chat/sessions${kbId ? `?kbId=${encodeURIComponent(kbId)}` : ''}`),
    createSession: (kbId: string | null, title?: string) => request<ChatSession>('/chat/sessions', json({ kbId, title })),
    messages: (sessionId: string) => request<ChatMessage[]>(`/chat/sessions/${sessionId}/messages`),
    deleteSession: (sessionId: string) => request<{ message: string }>(`/chat/sessions/${sessionId}`, { method: 'DELETE' }),
  },
  models: {
    rerankerStatus: () => request<RuntimeInfo>('/models/reranker/status'),
    rerankerDownload: () => request<RuntimeInfo>('/models/reranker/download', { method: 'POST' }),
    providers: () => request<ProviderPreset[]>('/models/providers'),
    list: () => request<ModelConfig[]>('/models/configs'),
    save: (body: { id?: string; providerName: string; providerType: string; baseUrl?: string | null; apiKey?: string | null; category: string; modelName: string; keyFromConfigId?: string | null }) => request<ModelConfig>('/models/configs', json(body)),
    activate: (id: string) => request<{ message: string }>(`/models/configs/${id}/activate`, { method: 'POST' }),
    deactivate: (id: string) => request<{ message: string }>(`/models/configs/${id}/deactivate`, { method: 'POST' }),
    remove: (id: string) => request<{ message: string }>(`/models/configs/${id}`, { method: 'DELETE' }),
    fetchList: (body: { baseUrl: string; apiKey?: string | null; providerType?: string; keyFromConfigId?: string | null }) => request<{ source: string; error?: string; total?: number; models: Record<string, string[]>; cached?: string[] }>('/models/fetch-list', json(body)),
    test: (body: { configId?: string; providerName?: string; providerType?: string; baseUrl?: string | null; apiKey?: string | null; category: string; modelName: string }) =>
      request<{ success: boolean; latency: number; message: string }>('/models/test-connection', json(body)),
  },
  settings: {
    get: () => request<SettingsPayload>('/settings/'),
    presets: () => request<PresetsPayload>('/settings/presets'),
    update: (overrides: Record<string, unknown>) => request<{ effective: Record<string, any>; overrides: Record<string, any>; reindexScheduled?: boolean }>('/settings/', put({ overrides })),
    reindexStatus: () => request<ReindexStatus>('/settings/reindex-status'),
    reset: () => request<{ effective: Record<string, any>; overrides: Record<string, any>; reindexScheduled?: boolean }>('/settings/reset', { method: 'POST' }),
  },
  corpus: {
    health: (kbId: string) => request<HealthReport>(`/corpus/${kbId}/health`),
    exclude: (kbId: string, chunkIds: string[], reason = '') => request<{ parents: number; chunks: number }>(`/corpus/${kbId}/health/exclude`, json({ chunkIds, reason })),
    restore: (kbId: string, fingerprints: string[]) => request<{ reingesting: string[] }>(`/corpus/${kbId}/health/restore`, json({ fingerprints })),
    retire: (kbId: string, docIds: string[], validTo?: string) => request<{ validTo: string }>(`/corpus/${kbId}/health/retire`, json({ docIds, validTo })),
    aliases: (kbId: string) => request<{ groups: string[][]; updatedAt: string | null }>(`/corpus/${kbId}/aliases`),
    saveAliases: (kbId: string, groups: string[][]) => request<{ groups: string[][]; updatedAt: string; reindexing: boolean }>(`/corpus/${kbId}/aliases`, put({ groups })),
    discoverAliases: (kbId: string) => request<AliasSuggestion[]>(`/corpus/${kbId}/aliases/discover`, { method: 'POST' }),
    augment: (kbId: string) => request<AugmentState>(`/corpus/${kbId}/augment`),
    generate: (kbId: string, intents?: string[]) => request<AugmentState>(`/corpus/${kbId}/augment/generate`, json({ intents: intents ?? null })),
    addItem: (kbId: string, question: string, answer: string) => request<AugmentState>(`/corpus/${kbId}/augment/items`, json({ question, answer })),
    updateItem: (kbId: string, id: string, body: { question?: string; answer?: string; status?: FaqStatus }) => request<AugmentState>(`/corpus/${kbId}/augment/items/${id}`, put(body)),
    deleteItem: (kbId: string, id: string) => request<AugmentState>(`/corpus/${kbId}/augment/items/${id}`, { method: 'DELETE' }),
    publish: (kbId: string) => request<AugmentState>(`/corpus/${kbId}/augment/publish`, { method: 'POST' }),
    extractMeta: (kbId: string, useLlm = false, docIds?: string[]) => request<MetaRow[]>(`/corpus/${kbId}/metadata/extract`, json({ useLlm, docIds })),
    applyMeta: (kbId: string, items: { docId: string; fields: Partial<Record<MetaField, unknown>> }[]) => request<{ updated: number }>(`/corpus/${kbId}/metadata/apply`, json({ items })),
  },
  eval: {
    sample: () => request<EvalExample[]>('/eval/sample'),
    run: (kbId: string, body: { examples?: EvalExample[]; useSample?: boolean; ks?: number[]; name?: string; overrides?: RetrievalOverrides }) => request<EvalReport>(`/eval/${kbId}/run`, json(body)),
    reports: (kbId: string) => request<EvalReport[]>(`/eval/${kbId}/reports`),
    hardNegatives: (kbId: string, body: { examples: EvalExample[]; heldOut?: EvalExample[]; config?: Partial<HardNegativeMiningConfig> }) => request<HardNegativeReport>(`/eval/${kbId}/hard-negatives`, json(body)),
    multimodalSample: (kbId: string) => request<Record<string, unknown>[]>(`/eval/${kbId}/multimodal/sample`),
    multimodalRun: (kbId: string, body: { examples?: Record<string, unknown>[]; auto?: boolean; ks?: number[]; name?: string }) => request<MultimodalEvalReport>(`/eval/${kbId}/multimodal/run`, json(body)),
    multimodalReports: (kbId: string) => request<MultimodalEvalReport[]>(`/eval/${kbId}/multimodal/reports`),
    coverage: (kbId: string, body: { intents?: (string | { question: string; category?: string })[]; useSample?: boolean; high?: number; partial?: number }) => request<CoverageReport>(`/eval/${kbId}/coverage`, json(body)),
    lastCoverage: (kbId: string) => request<CoverageReport | null>(`/eval/${kbId}/coverage`),
  },
};

// ---------------------------------------------------------------------------------
// SSE 对话流：event: meta | plan | sources | relations | images | message | error | done
export interface ChatStreamHandlers {
  onMeta?: (meta: NonNullable<ChatMessage['meta']>) => void;
  onPlan?: (plan: ChatPlan) => void;
  onSources?: (sources: ChatSource[]) => void;
  onRelations?: (relations: WeavedRelation[]) => void;
  onImages?: (images: ImageHitView[]) => void;
  onToken?: (token: string) => void;
  onError?: (error: string) => void;
  onDone?: (done: { citations: number[]; imageCitations: number[]; model: string; offline: boolean; vision: boolean }) => void;
}

export async function streamChat(
  body: { kbId: string | null; sessionId?: string | null; message: string; imageB64?: string | null; tempDocContent?: string | null; mode?: RetrievalMode; overrides?: RetrievalOverrides; topK?: number; user?: UserIdentity | null },
  handlers: ChatStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/chat/stream`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal });
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const dispatch = (block: string) => {
    const eventMatch = block.match(/^event:\s*(.+)$/m);
    const dataLine = block.split('\n').find((l) => l.startsWith('data:'));
    if (!eventMatch || !dataLine) return;
    const name = eventMatch[1].trim();
    let data: any;
    try {
      data = JSON.parse(dataLine.slice(5).trim());
    } catch {
      return;
    }
    switch (name) {
      case 'meta': handlers.onMeta?.(data); break;
      case 'plan': handlers.onPlan?.(data); break;
      case 'sources': handlers.onSources?.(data); break;
      case 'relations': handlers.onRelations?.(data); break;
      case 'images': handlers.onImages?.(data); break;
      case 'message': handlers.onToken?.(data.content ?? ''); break;
      case 'error': handlers.onError?.(data.error ?? 'unknown'); break;
      case 'done': handlers.onDone?.(data); break;
    }
  };
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() ?? '';
    for (const p of parts) if (p.trim()) dispatch(p);
  }
  if (buffer.trim()) dispatch(buffer);
}
