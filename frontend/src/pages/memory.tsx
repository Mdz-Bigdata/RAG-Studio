// 长期记忆（LanceDB 表 memories）：按用户记住身份、偏好、事实；召回 = 相似度 × 新近度 × 重要度；可抽取、编辑、遗忘。
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Brain, Eraser, MessageSquareQuote, Pencil, Plus, Search, Sparkles, Trash2 } from 'lucide-react';
import { ApiError } from '../api/client';
import { fmtTime, memoryApi, type LanceVersion, type MemoryItem, type MemoryKind, type MemoryStats } from '../api/lance';
import { Badge, Button, Callout, Empty, Modal, ScoreBar, Spinner, Tabs, confirmDialog } from '../components/ui';
import { useApp } from '../store/app-store';

const KINDS: { value: MemoryKind; label: string; tone: string }[] = [
  { value: 'profile', label: '身份', tone: 'info' },
  { value: 'preference', label: '偏好', tone: 'accent' },
  { value: 'fact', label: '事实', tone: '' },
  { value: 'task', label: '任务', tone: 'warn' },
  { value: 'episode', label: '经历', tone: 'mmr' },
];
const kindOf = (k: string) => KINDS.find((x) => x.value === k) ?? { value: k as MemoryKind, label: k, tone: '' };
const errText = (e: unknown) => (e instanceof ApiError ? e.message : String(e));

export function MemoryPage() {
  const { identity, toast } = useApp();
  const userId = identity.userId || 'default';
  const [items, setItems] = useState<MemoryItem[] | null>(null);
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [versions, setVersions] = useState<LanceVersion[]>([]);
  const [kind, setKind] = useState<'all' | MemoryKind>('all');
  const [keyword, setKeyword] = useState('');
  const [editing, setEditing] = useState<MemoryItem | null>(null);

  const load = useCallback(async () => {
    try {
      const [list, st, vs] = await Promise.all([memoryApi.list({ userId, kind: kind === 'all' ? undefined : kind, keyword: keyword || undefined }), memoryApi.stats(), memoryApi.versions()]);
      setItems(list);
      setStats(st);
      setVersions(vs);
    } catch (e) {
      toast(`读取记忆失败：${errText(e)}`, 'bad');
    }
  }, [userId, kind, keyword, toast]);
  useEffect(() => { load(); }, [load]);

  const remove = async (m: MemoryItem) => {
    if (!confirmDialog(`删除这条记忆？\n${m.content}`)) return;
    await memoryApi.remove(m.id);
    toast('已删除', 'ok');
    load();
  };
  const forget = async () => {
    const r = await memoryApi.forget({ userId });
    toast(`遗忘完成：过期 ${r.expired} 条，衰减 ${r.decayed} 条`, 'ok');
    load();
  };
  const clearAll = async () => {
    if (!confirmDialog(`清空用户「${userId}」的全部记忆？此操作会写入新版本，可在版本历史里看到。`)) return;
    const r = await memoryApi.clearUser(userId);
    toast(`已清空 ${r.deleted} 条`, 'ok');
    load();
  };

  const mine = stats?.byUser?.[userId] ?? 0;
  return (
    <div className="page col gap-16 fade-in">
      <div className="card stat-strip">
        <div className="stat"><div className="label"><Brain size={12} />当前用户</div><div className="value">{userId}</div><div className="sub">在右上角「用户」里切换</div></div>
        <div className="stat"><div className="label">该用户的记忆</div><div className="value">{mine}</div><div className="sub">全库 {stats?.rows ?? 0} 条 · {stats?.users ?? 0} 个用户</div></div>
        <div className="stat"><div className="label">按类型</div><div className="value" style={{ fontSize: 13, paddingTop: 8, lineHeight: 1.5 }}>{KINDS.map((k) => `${k.label} ${stats?.byKind?.[k.value] ?? 0}`).join(' · ')}</div><div className="sub">身份 / 偏好 / 事实 / 任务 / 经历</div></div>
        <div className="stat"><div className="label">向量模型</div><div className="value" style={{ fontSize: 15, paddingTop: 6 }}>{stats?.model ?? stats?.embedder ?? '—'}</div><div className="sub">{stats?.dim ? `${stats.dim} 维 · 换模型自动重算` : ''}</div></div>
      </div>

      <Callout icon={<Sparkles size={14} />}>
        智能问答每轮会按<b>相似度 × 新近度 × 重要度</b>召回该用户的记忆，只用于个性化（称呼、部门、表达偏好），事实仍以知识库资料为准；
        回答结束后在后台从用户的话里抽取新记忆（如「我在财务部」「以后先给结论」），与已有记忆近似重复时合并更新，不重复堆积。
      </Callout>

      <div className="grid grid-2" style={{ alignItems: 'start' }}>
        <AddCard userId={userId} onDone={load} />
        <RecallCard userId={userId} />
      </div>

      <div className="card">
        <div className="card-header">
          <h3><Brain size={14} />记忆列表</h3>
          <div className="row gap-8 wrap">
            <Tabs value={kind} onChange={setKind} items={[{ value: 'all', label: '全部' }, ...KINDS.map((k) => ({ value: k.value, label: k.label }))]} />
            <input value={keyword} onChange={(e) => setKeyword(e.target.value)} placeholder="关键词过滤" style={{ width: 140 }} />
            <Button size="sm" onClick={forget} title="删除过期的，以及长期没用到且重要度低的"><Eraser size={13} />遗忘清理</Button>
            <Button size="sm" variant="danger" onClick={clearAll}><Trash2 size={13} />清空该用户</Button>
          </div>
        </div>
        {items === null ? <div className="card-body"><Spinner label="加载中…" /></div> : items.length === 0 ? (
          <Empty icon={<Brain size={32} />} title="还没有记忆" desc="在智能问答里说说自己（例如「我在财务部，回答请先给结论」），或在上面手动添加 / 从一段话里抽取。" />
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>内容</th><th>类型</th><th style={{ width: 120 }}>重要度</th><th className="num">被用到</th><th>来源</th><th>更新时间</th><th /></tr></thead>
              <tbody>
                {items.map((m) => (
                  <tr key={m.id}>
                    <td style={{ maxWidth: 420 }}><div className="small">{m.content}</div>{m.expiresAt ? <div className="xs faint">到期：{fmtTime(m.expiresAt)}</div> : null}</td>
                    <td><Badge tone={kindOf(m.kind).tone}>{kindOf(m.kind).label}</Badge></td>
                    <td><div className="row gap-6"><div className="flex-1"><ScoreBar value={m.importance} /></div><span className="mono xs">{m.importance.toFixed(2)}</span></div></td>
                    <td className="num">{m.accessCount}</td>
                    <td className="xs">{m.source === 'chat' ? '对话抽取' : m.source === 'manual' ? '手动' : m.source}</td>
                    <td className="xs">{fmtTime(m.updatedAt)}</td>
                    <td><div className="row gap-4"><Button size="xs" variant="ghost" icon onClick={() => setEditing(m)} aria-label="编辑"><Pencil size={12} /></Button><Button size="xs" variant="ghost" icon onClick={() => remove(m)} aria-label="删除"><Trash2 size={12} /></Button></div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {versions.length ? <div className="card-body xs faint">LanceDB 版本：当前 v{versions[0]?.version}，共 {versions.length} 个可回看版本（每次写入 / 删除都会产生新版本）。</div> : null}
      </div>

      {editing ? <EditModal item={editing} onClose={() => setEditing(null)} onSaved={() => { setEditing(null); load(); }} /> : null}
    </div>
  );
}

function AddCard({ userId, onDone }: { userId: string; onDone: () => void }) {
  const { toast } = useApp();
  const [tab, setTab] = useState<'manual' | 'extract'>('manual');
  const [content, setContent] = useState('');
  const [kind, setKind] = useState<MemoryKind>('preference');
  const [importance, setImportance] = useState(0.7);
  const [ttl, setTtl] = useState<number | ''>('');
  const [text, setText] = useState('我叫李明，在财务部做报销审核。以后回答请先给结论，再讲细节。');
  const [extracted, setExtracted] = useState<MemoryItem[] | null>(null);
  const [busy, setBusy] = useState(false);

  const add = async () => {
    if (!content.trim()) return;
    setBusy(true);
    try {
      const r = await memoryApi.add({ userId, content, kind, importance, ttlDays: ttl || null });
      toast(r.action === 'updated' ? '与已有记忆近似：已合并更新' : '已记住', 'ok');
      setContent('');
      onDone();
    } catch (e) {
      toast(errText(e), 'bad');
    } finally {
      setBusy(false);
    }
  };
  const extract = async () => {
    setBusy(true);
    try {
      const r = await memoryApi.extract({ userId, text });
      setExtracted(r);
      toast(r.length ? `抽取到 ${r.length} 条记忆` : '这段话里没有值得长期记住的信息', r.length ? 'ok' : '');
      onDone();
    } catch (e) {
      toast(errText(e), 'bad');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <div className="card-header"><h3><Plus size={14} />写入记忆</h3><Tabs value={tab} onChange={setTab} items={[{ value: 'manual', label: '手动添加' }, { value: 'extract', label: '从一段话抽取' }]} /></div>
      <div className="card-body col gap-12">
        {tab === 'manual' ? (
          <>
            <textarea rows={2} value={content} onChange={(e) => setContent(e.target.value)} placeholder="写成一句可以独立理解的陈述，例如：用户在财务部，负责报销审核" />
            <div className="row gap-12 wrap xs muted">
              <label className="row gap-6" style={{ whiteSpace: 'nowrap' }}>类型<select value={kind} onChange={(e) => setKind(e.target.value as MemoryKind)} style={{ width: 'auto', padding: '3px 24px 3px 8px', fontSize: 12 }}>{KINDS.map((k) => <option key={k.value} value={k.value}>{k.label}</option>)}</select></label>
              <label className="row gap-6" style={{ whiteSpace: 'nowrap' }}>重要度<input type="range" min={0} max={1} step={0.05} value={importance} onChange={(e) => setImportance(Number(e.target.value))} /><span className="mono">{importance.toFixed(2)}</span></label>
              <label className="row gap-6" style={{ whiteSpace: 'nowrap' }} title="到期后不再召回，遗忘清理时删除">有效期<input type="number" min={1} value={ttl} placeholder="永久" onChange={(e) => setTtl(e.target.value === '' ? '' : Number(e.target.value))} style={{ width: 70 }} />天</label>
            </div>
            <div><Button variant="primary" onClick={add} loading={busy}><Plus size={13} />记住</Button></div>
          </>
        ) : (
          <>
            <textarea rows={3} value={text} onChange={(e) => setText(e.target.value)} placeholder="粘贴用户说过的话" />
            <div className="row gap-8"><Button variant="primary" onClick={extract} loading={busy}><MessageSquareQuote size={13} />抽取并记住</Button><span className="xs faint">有文本模型时用辅助模型抽取，否则用规则；一次性的提问不会被记住</span></div>
            {extracted?.length ? (
              <div className="col gap-6">{extracted.map((m) => <div key={m.id} className="chunk row between gap-8"><span className="small">{m.content}</span><span className="row gap-6"><Badge tone={kindOf(m.kind).tone}>{kindOf(m.kind).label}</Badge><Badge tone={m.action === 'updated' ? 'info' : 'ok'}>{m.action === 'updated' ? '合并更新' : '新记忆'}</Badge></span></div>)}</div>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}

function RecallCard({ userId }: { userId: string }) {
  const { toast } = useApp();
  const [query, setQuery] = useState('报销单应该怎么提交');
  const [hits, setHits] = useState<MemoryItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const run = async () => {
    setLoading(true);
    try {
      setHits(await memoryApi.search({ userId, query, topK: 5 }));
    } catch (e) {
      toast(errText(e), 'bad');
    } finally {
      setLoading(false);
    }
  };
  const best = useMemo(() => Math.max(0.0001, ...(hits ?? []).map((h) => h.score ?? 0)), [hits]);
  return (
    <div className="card">
      <div className="card-header"><h3><Search size={14} />召回测试</h3><span className="xs faint">score = 0.7·相似度 + 0.15·新近度 + 0.15·重要度</span></div>
      <div className="card-body col gap-12">
        <div className="row gap-8"><input className="flex-1" value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && run()} placeholder="用户的新问题" /><Button variant="primary" onClick={run} loading={loading}><Search size={13} />召回</Button></div>
        {hits === null ? <div className="xs faint">问答时就是这样为当前问题挑选记忆的（测试不计入"被用到"次数）。</div> : hits.length === 0 ? <Empty title="没有相关记忆" /> : (
          <div className="col gap-6">
            {hits.map((h) => (
              <div key={h.id} className="chunk">
                <div className="chunk-head"><span className="ellipsis">{h.content}</span><span className="mono">{(h.score ?? 0).toFixed(3)}</span></div>
                <ScoreBar value={h.score ?? 0} max={best} />
                <div className="row gap-12 xs faint mt-4"><span>相似度 {(h.similarity ?? 0).toFixed(3)}</span><span>新近度 {(h.recency ?? 0).toFixed(3)}</span><span>重要度 {h.importance.toFixed(2)}</span><Badge tone={kindOf(h.kind).tone}>{kindOf(h.kind).label}</Badge></div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function EditModal({ item, onClose, onSaved }: { item: MemoryItem; onClose: () => void; onSaved: () => void }) {
  const { toast } = useApp();
  const [content, setContent] = useState(item.content);
  const [kind, setKind] = useState<MemoryKind>(item.kind);
  const [importance, setImportance] = useState(item.importance);
  const save = async () => {
    try {
      await memoryApi.update(item.id, { content, kind, importance });
      toast('已更新', 'ok');
      onSaved();
    } catch (e) {
      toast(errText(e), 'bad');
    }
  };
  return (
    <Modal title="编辑记忆" onClose={onClose} footer={<><Button variant="ghost" onClick={onClose}>取消</Button><Button variant="primary" onClick={save}>保存</Button></>}>
      <div className="col gap-12">
        <textarea rows={3} value={content} onChange={(e) => setContent(e.target.value)} />
        <div className="row gap-12 wrap xs muted">
          <label className="row gap-6" style={{ whiteSpace: 'nowrap' }}>类型<select value={kind} onChange={(e) => setKind(e.target.value as MemoryKind)} style={{ width: 'auto', padding: '3px 24px 3px 8px', fontSize: 12 }}>{KINDS.map((k) => <option key={k.value} value={k.value}>{k.label}</option>)}</select></label>
          <label className="row gap-6" style={{ whiteSpace: 'nowrap' }}>重要度<input type="range" min={0} max={1} step={0.05} value={importance} onChange={(e) => setImportance(Number(e.target.value))} /><span className="mono">{importance.toFixed(2)}</span></label>
        </div>
      </div>
    </Modal>
  );
}
