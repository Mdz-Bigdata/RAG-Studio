// LanceDB 应用接口：控制台（/lancedb）、长期记忆（/memory）、个性化推荐（/recommend）、多智能体协作（/agents）
import { API_BASE, ApiError } from './client';

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) } });
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
  return (await res.json()) as T;
}
const post = (body?: unknown): RequestInit => ({ method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) });
const qs = (params: Record<string, string | number | boolean | null | undefined>) => {
  const p = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '').map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
  return p.length ? `?${p.join('&')}` : '';
};

// --- LanceDB 控制台 ---------------------------------------------------------------------------
export interface LanceIndexInfo { name: string; type: string; columns: string[] }
export interface LanceTableInfo {
  name: string; rows: number; version: number; versions: number; bytes?: number | null; fragments?: number | null; smallFragments?: number | null;
  indices: LanceIndexInfo[]; schema: { name: string; type: string }[]; purpose?: string;
}
export interface LanceKbRow { kbId: string; name: string; backend?: string; vectors?: number; images?: number; mirror?: boolean; multimodal?: boolean; error?: string }
export interface LanceOverview {
  version: string; dir: string; backend: string; sparseBackend: string; lancedb: Record<string, unknown>;
  search: { nprobes: number; refineFactor: number | null; ef: number }; tables: LanceTableInfo[]; kbs: LanceKbRow[];
}
export interface LanceVersion { version: number; timestamp: string; metadata: Record<string, unknown> }
export interface LanceKbInfo extends Partial<LanceTableInfo> {
  kbId: string; mode: 'backend' | 'mirror' | 'none'; note?: string; exists?: boolean; path?: string; annIndexed?: boolean;
  indexMinRows?: number; writesSinceOptimize?: number; versionsList?: LanceVersion[];
  aux?: Record<string, { backend: string; rows: number }>;   // 摘要 / 章节 / 句子辅助索引（后端为 LanceDB 时）
}
export type LanceSearchMode = 'vector' | 'exact' | 'fts' | 'hybrid';
export interface LanceHit { chunkId: string; score: number; breadcrumb: string; text: string; docId: string; fileName?: string | null }
export interface LanceSearchResult {
  kbId: string; source: string; mode: LanceSearchMode; version?: number | null; hits: LanceHit[];
  timingsMs: { embed: number; search: number }; annIndexed: boolean; plan?: string | null;
}
export interface MMHit { id: string; modality: 'image' | 'text'; docId: string; docTitle: string; refId: string; title: string; text: string; page?: number | null; hasImage: boolean; score: number; fileName?: string | null }
export interface MMResult { hits: MMHit[]; how?: string; note?: string; version?: number; ms?: number }
export interface MMSync { images: number; texts: number; written: number; removed: number; table: string; path?: string; crossModal: boolean | null; embedder: string; version?: number | null; ms?: number }
export interface MMInfo extends Partial<LanceTableInfo> { exists?: boolean; byModality?: { image: number; text: number }; path?: string; online?: boolean }

export const lanceApi = {
  overview: () => req<LanceOverview>('/lancedb/overview'),
  tableVersions: (name: string) => req<LanceVersion[]>(`/lancedb/tables/${encodeURIComponent(name)}/versions`),
  kbInfo: (kbId: string) => req<LanceKbInfo>(`/lancedb/kb/${kbId}`),
  mirror: (kbId: string) => req<{ rows: number; mode: string; table?: string; ms?: number; note?: string }>(`/lancedb/kb/${kbId}/mirror`, post()),
  search: (kbId: string, body: { query: string; mode: LanceSearchMode; k?: number; version?: number | null; nprobes?: number | null; refineFactor?: number | null; explain?: boolean }) =>
    req<LanceSearchResult>(`/lancedb/kb/${kbId}/search`, post(body)),
  buildIndex: (kbId: string, indexType: string) => req<LanceKbInfo & { ms: number }>(`/lancedb/kb/${kbId}/index`, post({ indexType })),
  optimize: (kbId: string) => req<LanceKbInfo & { ms: number }>(`/lancedb/kb/${kbId}/optimize`, post()),
  mmInfo: (kbId: string) => req<MMInfo>(`/lancedb/mm/${kbId}`),
  mmSync: (kbId: string) => req<MMSync>(`/lancedb/mm/${kbId}/sync`, post()),
  mmSearch: (kbId: string, body: { text?: string; imageB64?: string | null; target: 'image' | 'text' | 'all'; mode: 'vector' | 'fts' | 'hybrid'; k?: number }) =>
    req<MMResult>(`/lancedb/mm/${kbId}/search`, post(body)),
  thumbUrl: (kbId: string, rowId: string) => `${API_BASE}/lancedb/mm/${kbId}/thumb/${encodeURIComponent(rowId)}`,
};

// --- 长期记忆 ---------------------------------------------------------------------------------
export type MemoryKind = 'profile' | 'preference' | 'fact' | 'episode' | 'task';
export interface MemoryItem {
  id: string; content: string; userId: string; agentId: string; kind: MemoryKind; importance: number; source: string; sessionId: string; tags: string[];
  createdAt: number; updatedAt: number; lastAccess: number; accessCount: number; expiresAt: number;
  score?: number; similarity?: number; recency?: number; action?: 'created' | 'updated';
}
export interface MemoryStats {
  rows: number; users: number; byKind: Record<string, number>; byUser: Record<string, number>; byAgent?: Record<string, number>;
  table?: LanceTableInfo; embedder?: string; model?: string | null; dim?: number;
}

export const memoryApi = {
  list: (params: { userId?: string; agentId?: string; kind?: string; keyword?: string; limit?: number }) => req<MemoryItem[]>(`/memory${qs(params)}`),
  add: (body: { userId: string; content: string; kind: MemoryKind; importance: number; ttlDays?: number | null; tags?: string[] }) => req<MemoryItem>('/memory', post(body)),
  update: (id: string, body: { content?: string; kind?: MemoryKind; importance?: number }) => req<MemoryItem>(`/memory/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  remove: (id: string) => req<{ ok: boolean }>(`/memory/${id}`, { method: 'DELETE' }),
  search: (body: { userId: string; query: string; topK?: number; minScore?: number; kinds?: string[] | null }) => req<MemoryItem[]>('/memory/search', post(body)),
  extract: (body: { userId: string; text: string; answer?: string }) => req<MemoryItem[]>('/memory/extract', post(body)),
  forget: (body: { userId?: string | null; minKeepScore?: number }) => req<{ expired: number; decayed: number }>('/memory/forget', post(body)),
  clearUser: (userId: string) => req<{ deleted: number }>(`/memory/user/${encodeURIComponent(userId)}`, { method: 'DELETE' }),
  stats: () => req<MemoryStats>('/memory/stats'),
  versions: () => req<LanceVersion[]>('/memory/versions'),
};

// --- 个性化推荐 -------------------------------------------------------------------------------
export type RecKind = 'doc' | 'section' | 'question' | 'image';
export type RecEventKind = 'view' | 'click' | 'like' | 'dislike' | 'ask' | 'cite';
export interface RecItem { id: string; kbId: string; kind: RecKind; refId: string; title: string; text: string; tags: string[]; popularity: number; score?: number; similarity?: number; reason?: string; myVote?: RecVote | null }
export type RecVote = 'like' | 'dislike';
export interface RecProfile { events: number; byKind: Record<string, number>; recent: { kind: RecEventKind; text: string; itemId: string; ts: number }[] }
export interface RecResult { mode: 'empty' | 'popular' | 'personal'; items: RecItem[]; profile: RecProfile | null; memoryInterests?: number }
export interface RecStats { items: number; itemsByKind: Record<string, number>; events: number; users: number; tables: LanceTableInfo[]; embedder?: string; model?: string | null; dim?: number }

export const recommendApi = {
  sync: (kbId: string) => req<{ kbId: string; items: number; removed: number; byKind: Record<string, number> }>(`/recommend/sync/${kbId}`, post()),
  get: (params: { userId: string; kbId?: string | null; kinds?: string; k?: number; useMemory?: boolean }) => req<RecResult>(`/recommend${qs(params)}`),
  event: (body: { userId: string; kind: RecEventKind; itemId?: string; kbId?: string; text?: string }) => req<Record<string, unknown>>('/recommend/events', post(body)),
  similar: (itemId: string, k = 6) => req<RecItem[]>(`/recommend/similar/${itemId}${qs({ k })}`),
  stats: () => req<RecStats>('/recommend/stats'),
};

// --- 多智能体协作 -----------------------------------------------------------------------------
export interface AgentRole { key: string; name: string; desc: string }
export interface AgentEvidence { n: number; chunkId: string; docId: string; label: string; text: string; task: string }
export interface AgentEvent { type: string; agent: string; content: string; data: Record<string, any>; ms: number }
export interface AgentRunSummary { runId: string; question: string; answer: string; ts: number; kbId?: string; userId?: string }
export interface BoardRow { id: string; runId: string; step: number; agent: string; kind: string; content: string; refs: any; ts: number }

export const agentsApi = {
  roles: () => req<AgentRole[]>('/agents/roles'),
  runs: () => req<AgentRunSummary[]>('/agents/runs'),
  history: (runId: string) => req<BoardRow[]>(`/agents/runs/${runId}`),
  deleteRun: (runId: string) => req<{ runs: number; rows: number }>(`/agents/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' }),
  deleteRuns: (runIds: string[]) => req<{ rows: number }>('/agents/runs/delete', post({ runIds })),
  clearRuns: () => req<{ rows: number }>('/agents/runs', { method: 'DELETE' }),
  board: () => req<LanceTableInfo>('/agents/board'),
  async run(body: { kbId: string; question: string; userId: string }, onEvent: (e: AgentEvent) => void, signal?: AbortSignal): Promise<void> {
    const res = await fetch(`${API_BASE}/agents/run`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal });
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
      const dataLine = block.split('\n').find((l) => l.startsWith('data:'));
      if (!dataLine) return;
      try {
        onEvent(JSON.parse(dataLine.slice(5).trim()) as AgentEvent);
      } catch {
        /* ignore malformed block */
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
  },
};

// --- 共享小工具 -------------------------------------------------------------------------------
export const fmtBytes = (n?: number | null) => (n == null ? '—' : n < 1024 ? `${n} B` : n < 1024 ** 2 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1024 ** 2).toFixed(1)} MB`);
export const fmtTime = (ts?: number | string | null) => {
  if (!ts) return '—';
  const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
  return Number.isNaN(d.getTime()) ? String(ts) : d.toLocaleString('zh-CN', { hour12: false });
};
