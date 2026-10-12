export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—';
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : iso + 'Z');
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString('zh-CN', { hour12: false, month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
}

export function formatScore(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return v.toFixed(digits);
}

export function pct(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return `${(v * 100).toFixed(1)}%`;
}

export function truncate(s: string, n = 120): string {
  if (!s) return '';
  return s.length > n ? s.slice(0, n) + '…' : s;
}

export const STATUS_LABEL: Record<string, string> = {
  PENDING: '排队中',
  PROCESSING: '解析索引中',
  COMPLETED: '已完成',
  FAILED: '失败',
};

export const ROUTE_LABEL: Record<string, string> = {
  dense: '向量召回',
  sparse: 'BM25 召回',
  hyde: 'HyDE 召回',
  sql: 'SQL 结构化',
  summary: '摘要索引',
  section: '章节级',
  sentence: '句子级',
  fused: 'RRF 融合',
  rerank: '精排',
  mmr: 'MMR',
};

export function routeKind(route: string): 'dense' | 'sparse' | 'hyde' | 'other' {
  if (route.includes('hyde')) return 'hyde';
  if (route.startsWith('dense') || route.includes(':dense')) return 'dense';
  if (route.startsWith('sparse') || route.includes(':sparse')) return 'sparse';
  return 'other';
}

export function routeLabel(route: string): string {
  const [head, tail] = route.split(':');
  const hop = head.startsWith('hop') ? `${head} · ` : '';
  const kind = head.startsWith('hop') ? tail : head;
  const q = route.split(':').pop();
  const qLabel = q && q.startsWith('q') ? `查询 ${Number(q.slice(1)) + 1}` : q === 'hyde' ? '假设文档' : '';
  return `${hop}${ROUTE_LABEL[kind] ?? kind}${qLabel ? ` · ${qLabel}` : ''}`;
}
