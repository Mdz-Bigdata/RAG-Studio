// 飞书文档接口：应用配置、同步进知识库、自动同步 / 事件订阅、远程增删改查、文档智能体（SSE）
import { API_BASE, ApiError } from './client';

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}/feishu${path}`, { ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) } });
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
const send = (method: string, body?: unknown): RequestInit => ({ method, body: body === undefined ? undefined : JSON.stringify(body) });
const qs = (params: Record<string, string | number | boolean | null | undefined>) => {
  const p = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '').map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
  return p.length ? `?${p.join('&')}` : '';
};

export interface FeishuConfig {
  app_id: string;
  app_secret: string;            // 已保存时为 ********
  domain: 'feishu' | 'lark';
  web_base: string;
  folder_token: string;
  default_kb_id: string;
  use_convert_api: boolean;
  agent_allow_write: boolean;
  new_doc_share: 'tenant_editable' | 'tenant_readable' | 'closed';
  agent_max_steps: number;
  read_images: boolean;
  auto_sync_minutes: number;     // 0 = 关闭
  event_verification_token: string;   // 已保存时为 ********
  event_encrypt_key: string;          // 已保存时为 ********
  configured: boolean;
  fromEnv: boolean;
  autoSync: { running: boolean; last: FeishuAutoSyncRound | null };
}
export interface FeishuAutoSyncRound { at: string; kbs: number; checked: number; updated: number; failed: number; errors: string[] }
export interface FeishuLink {
  id: string; kbId: string; documentId: string; title: string | null; url: string | null; revisionId: number | null;
  status: 'SYNCED' | 'FAILED' | 'MISSING' | 'PENDING'; errorMessage: string | null; localDocId: string | null; localStatus: string | null;
  chunkCount: number; syncedAt: string | null; createdAt: string | null;
}
export interface FeishuSyncResult { ok?: boolean; ref?: string; status?: 'created' | 'updated' | 'unchanged'; linkId?: string; documentId?: string; title?: string; localDocId?: string; error?: string }
export interface FeishuRemoteDoc { documentId: string; title: string; revisionId: number | null; url: string; markdown: string; chars: number; blocks: number; html?: string }
export interface FeishuFile { token: string; name: string; type: string; url: string; modifiedTime?: string; linkedKbs: string[] }
export interface FeishuAgentEvent { type: string; agent: string; content: string; data: Record<string, any>; ms: number }
export interface FeishuAction { id: string; tool: string; args: Record<string, any>; status: 'ok' | 'pending' | 'error'; error?: string }

export const feishuApi = {
  config: () => req<FeishuConfig>('/config'),
  saveConfig: (body: Partial<FeishuConfig>) => req<FeishuConfig>('/config', send('PUT', body)),
  test: () => req<{ ok: boolean; token: boolean; folder: { token: string | null; files: number } | null; error?: string }>('/test', send('POST')),

  links: (kbId: string) => req<FeishuLink[]>(`/docs${qs({ kbId })}`),
  import: (kbId: string, refs: string[]) => req<FeishuSyncResult[]>('/import', send('POST', { kbId, refs })),
  syncLink: (linkId: string, force = false) => req<FeishuSyncResult>(`/docs/${linkId}/sync${qs({ force })}`, send('POST')),
  syncKb: (kbId: string, force = false) => req<FeishuSyncResult[]>(`/sync${qs({ kbId, force })}`, send('POST')),
  syncAutoNow: () => req<FeishuAutoSyncRound | { skipped: true }>('/sync/auto', send('POST')),
  subscribeEvents: () => req<{ total: number; subscribed: number; failed: Record<string, string> }>('/events/subscribe', send('POST')),
  unlink: (linkId: string, deleteLocal = true) => req<{ linkId: string; localDeleted: boolean }>(`/docs/${linkId}${qs({ deleteLocal })}`, send('DELETE')),

  folder: (folderToken?: string) => req<{ folderToken: string | null; files: FeishuFile[] }>(`/folder${qs({ folderToken })}`),
  // view=true：多返回按飞书原样排版的 html（服务端由块渲染、文字已转义），图片走 /feishu/media 代理
  read: (ref: string, kbId?: string, view = false) => req<FeishuRemoteDoc>(`/remote${qs({ ref, kbId, view: view || undefined })}`),
  create: (body: { title: string; markdown: string; folderToken?: string; kbId?: string }) => req<{ documentId: string; title: string; url: string; blocks: number; shareError?: string }>('/remote', send('POST', body)),
  write: (body: { ref: string; markdown: string; mode: 'append' | 'replace'; kbId?: string }) => req<{ documentId: string; url: string; blocks: number; removedBlocks: number }>('/remote', send('PUT', body)),
  replaceText: (body: { ref: string; find: string; replace: string; kbId?: string }) => req<{ replaced: number; blocks: number; skippedStyledMatches: number }>('/remote/replace-text', send('POST', body)),
  remove: (ref: string, kbId?: string) => req<{ documentId: string; deleted: boolean }>(`/remote${qs({ ref, kbId })}`, send('DELETE')),

  execute: (body: { kbId?: string; tool: string; args: Record<string, any> }) => req<{ tool: string; result: any }>('/agent/execute', send('POST', body)),
  async run(body: { kbId?: string; question: string; allowWrite?: boolean }, onEvent: (e: FeishuAgentEvent) => void, signal?: AbortSignal): Promise<void> {
    const res = await fetch(`${API_BASE}/feishu/agent/run`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal });
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
        onEvent(JSON.parse(dataLine.slice(5).trim()) as FeishuAgentEvent);
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
