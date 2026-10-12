// 语料治理：在调检索参数之前，先把语料本身做干净、做完整。
// 质量体检（剔除噪音 / 重复 / 旧版本）→ 覆盖增强（补缺口 FAQ）→ 别名词典（补词汇鸿沟）→ 元数据（补过滤依据）。
import { useCallback, useEffect, useMemo, useState } from 'react';
import { BookPlus, Check, CheckCircle2, Eraser, FileCheck2, HeartPulse, History, Plus, RefreshCw, Replace, RotateCcw, Search, Send, Sparkles, Tags, Trash2, Undo2, X } from 'lucide-react';
import type { PageKey } from '../app';
import { api } from '../api/client';
import type { AliasSuggestion, AugmentState, ChunkIssue, FaqItem, FaqStatus, HealthReport, MetaField, MetaRow, Severity } from '../api/types';
import { KbSelect } from '../components/kb-select';
import { Badge, Button, Callout, Empty, Spinner, Tabs, confirmDialog } from '../components/ui';
import { formatDate } from '../lib/format';
import { ACCESS_LEVELS, useApp } from '../store/app-store';

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));
const SEV_TONE: Record<Severity, string> = { high: 'bad', medium: 'warn', low: '' };
type Tab = 'health' | 'augment' | 'aliases' | 'meta';

export function CorpusPage({ go }: { go: (key: PageKey) => void }) {
  const { currentKb } = useApp();
  const [tab, setTab] = useState<Tab>('health');
  return (
    <div className="page col gap-16 fade-in">
      <div className="card">
        <div className="card-header">
          <Tabs value={tab} onChange={setTab} items={[
            { value: 'health', label: '质量体检', icon: <HeartPulse size={13} /> },
            { value: 'augment', label: '覆盖增强', icon: <BookPlus size={13} /> },
            { value: 'aliases', label: '别名词典', icon: <Replace size={13} /> },
            { value: 'meta', label: '元数据', icon: <Tags size={13} /> },
          ]} />
          <KbSelect />
        </div>
      </div>
      {!currentKb ? (
        <div className="card"><Empty icon={<HeartPulse size={36} />} title="请先选择知识库" desc="语料治理针对单个知识库进行。" action={<Button size="sm" onClick={() => go('kb')}>去知识库</Button>} /></div>
      ) : tab === 'health' ? <HealthTab kbId={currentKb.id} go={go} />
        : tab === 'augment' ? <AugmentTab kbId={currentKb.id} go={go} />
        : tab === 'aliases' ? <AliasesTab kbId={currentKb.id} go={go} />
        : <MetaTab kbId={currentKb.id} />}
    </div>
  );
}

// ---------------------------------------------------------------------------------------
function HealthTab({ kbId, go }: { kbId: string; go: (key: PageKey) => void }) {
  const { toast, refreshKbs } = useApp();
  const [report, setReport] = useState<HealthReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<ChunkIssue | 'all'>('all');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try { setReport(await api.corpus.health(kbId)); setSelected(new Set()); } catch (e) { toast(errMsg(e), 'bad'); } finally { setLoading(false); }
  }, [kbId, toast]);
  useEffect(() => { setReport(null); load(); }, [load]);

  const rows = useMemo(() => (report?.chunks ?? []).filter((c) => filter === 'all' || c.issues.includes(filter)), [report, filter]);
  const toggle = (id: string) => setSelected((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; });

  const exclude = async (ids: string[]) => {
    if (!ids.length || !confirmDialog(`把 ${ids.length} 个段落移出检索？可以在下方「已剔除」里随时恢复。`)) return;
    setBusy(true);
    try {
      const r = await api.corpus.exclude(kbId, ids, filter === 'all' ? '' : report?.issueLabels[filter]?.label ?? '');
      toast(`已剔除 ${r.parents} 个段落（${r.chunks} 个切片）`, 'ok');
      await load();
      refreshKbs().catch(() => undefined);
    } catch (e) { toast(errMsg(e), 'bad'); } finally { setBusy(false); }
  };
  const restore = async (fps: string[]) => {
    try {
      const r = await api.corpus.restore(kbId, fps);
      toast(`已恢复，${r.reingesting.length} 份文档正在后台重新入库`, 'ok');
      window.setTimeout(load, 1500);
    } catch (e) { toast(errMsg(e), 'bad'); }
  };
  const retire = async (docIds: string[]) => {
    if (!confirmDialog(`把 ${docIds.length} 份旧版本的失效日期设为昨天？检索时会自动过滤，文档不会被删除。`)) return;
    try {
      const r = await api.corpus.retire(kbId, docIds);
      toast(`已设置失效日期 ${r.validTo}`, 'ok');
      await load();
    } catch (e) { toast(errMsg(e), 'bad'); }
  };

  if (!report) return <div className="card"><div className="card-body"><Spinner label="体检中…" /></div></div>;
  const { totals } = report;
  const tone = report.score >= 90 ? 'var(--success)' : report.score >= 75 ? 'var(--accent)' : report.score >= 60 ? 'var(--warning)' : 'var(--danger)';
  const issueKeys = (Object.keys(report.issueCounts) as ChunkIssue[]).filter((k) => report.issueCounts[k] > 0);

  return (
    <>
      <div className="card">
        <div className="stat-strip">
          <div className="stat"><div className="label">健康分</div><div className="value" style={{ color: tone }}>{report.score}</div><div className="sub">{report.grade}</div></div>
          <div className="stat"><div className="label">文档 / 段落</div><div className="value">{totals.docs} / {totals.parents}</div><div className="sub">{totals.children} 个检索切片</div></div>
          <div className="stat"><div className="label">问题段落</div><div className="value">{totals.flagged}</div><div className="sub">{totals.parents ? Math.round((totals.flagged / totals.parents) * 100) : 0}%</div></div>
          <div className="stat"><div className="label">多版本冲突</div><div className="value">{report.conflicts.length}</div><div className="sub">组</div></div>
          <div className="stat"><div className="label">已剔除</div><div className="value">{totals.excluded}</div><div className="sub">段落</div></div>
        </div>
        <div className="card-body row between wrap gap-8" style={{ borderTop: '1px solid var(--border)' }}>
          <div className="xs faint">按内容指纹剔除：重新解析、换向量模型重建索引后依然生效，可随时恢复。</div>
          <Button size="sm" loading={loading} onClick={load}><RefreshCw size={13} />重新体检</Button>
        </div>
      </div>

      <Callout icon={<HeartPulse size={15} />}>
        <div className="col gap-4"><b>建议</b>{report.recommendations.map((r, i) => <div key={i}>· {r}</div>)}</div>
      </Callout>

      {report.conflicts.length ? (
        <div className="card">
          <div className="card-header"><h3><History size={15} />多版本同时生效</h3><span className="xs faint">同名文档的多个版本都在生效期内，新旧规定会一起被召回</span></div>
          <div className="card-body col gap-12">
            {report.conflicts.map((g) => (
              <div key={g.key} className="chunk">
                <div className="col gap-6">
                  {g.docs.map((d) => (
                    <div key={d.docId} className="row gap-8 wrap">
                      {d.keep ? <Badge tone="ok">保留</Badge> : <Badge tone="warn">旧版本</Badge>}
                      <span className="strong">{d.title}</span>
                      <span className="xs muted mono">{d.version ? `v${d.version}` : '无版本号'} · {d.validFrom ?? '…'} ~ {d.validTo ?? '…'}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-8"><Button size="xs" onClick={() => retire(g.docs.filter((d) => !d.keep).map((d) => d.docId))}><History size={11} />让旧版本失效</Button></div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="card">
        <div className="card-header">
          <h3><Eraser size={15} />问题段落</h3>
          <div className="row gap-6 wrap">
            <button className={`btn xs chip ${filter === 'all' ? 'primary' : ''}`} onClick={() => setFilter('all')}>全部 {report.chunks.length}</button>
            {issueKeys.map((k) => <button key={k} className={`btn xs chip ${filter === k ? 'primary' : ''}`} onClick={() => setFilter(k)}>{report.issueLabels[k].label} {report.issueCounts[k]}</button>)}
          </div>
        </div>
        {rows.length === 0 ? <Empty icon={<CheckCircle2 size={32} />} title="没有问题段落" desc="乱码、目录页、过短、超长、重复与近似重复都没有发现。" /> : (
          <>
            <div className="card-body row between wrap gap-8" style={{ borderBottom: '1px solid var(--border)' }}>
              <div className="row gap-8 wrap">
                <Button size="xs" onClick={() => setSelected(new Set(rows.map((r) => r.chunkId)))}>全选当前 {rows.length} 条</Button>
                {selected.size ? <Button size="xs" variant="ghost" onClick={() => setSelected(new Set())}>清空选择</Button> : null}
                {filter !== 'all' ? <span className="xs muted">处置：{report.issueLabels[filter].action}</span> : null}
              </div>
              <Button size="sm" variant="primary" loading={busy} disabled={!selected.size} onClick={() => exclude([...selected])}><Eraser size={13} />剔除所选（{selected.size}）</Button>
            </div>
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th style={{ width: 30 }} /><th>问题</th><th>位置</th><th>内容</th><th /></tr></thead>
                <tbody>
                  {rows.map((c) => (
                    <tr key={c.chunkId}>
                      <td><input type="checkbox" checked={selected.has(c.chunkId)} onChange={() => toggle(c.chunkId)} aria-label="选择" /></td>
                      <td style={{ minWidth: 150 }}>
                        <div className="row gap-4 wrap">{c.issues.map((i) => <Badge key={i} tone={SEV_TONE[report.issueLabels[i].severity]}>{report.issueLabels[i].label}</Badge>)}</div>
                        <div className="xs faint mt-4">{c.details.join('；')}</div>
                      </td>
                      <td style={{ maxWidth: 220 }}>
                        <div className="ellipsis strong" title={c.fileName ?? c.docTitle}>{c.fileName ?? c.docTitle}</div>
                        <div className="xs faint ellipsis" title={c.breadcrumb}>{c.breadcrumb}{c.page ? ` · p.${c.page}` : ''}</div>
                      </td>
                      <td style={{ maxWidth: 420 }}>
                        <div className="small clamp-3 pre">{c.preview}</div>
                        {c.duplicateOf ? <div className="xs muted mt-4">保留的副本：《{c.duplicateOf.docTitle}》{c.duplicateOf.breadcrumb ? ` › ${c.duplicateOf.breadcrumb}` : ''}</div> : null}
                      </td>
                      <td><Button size="xs" onClick={() => exclude([c.chunkId])}>剔除</Button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>

      <div className="card">
        <div className="card-header"><h3><FileCheck2 size={15} />文档体检</h3><Button size="xs" onClick={() => go('kb')}>去知识库校对 / 重新解析</Button></div>
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>文档</th><th className="num">段落</th><th className="num">问题段落</th><th className="num">得分</th><th>文档级问题</th></tr></thead>
            <tbody>
              {report.docs.map((d) => (
                <tr key={d.docId}>
                  <td style={{ maxWidth: 260 }}><div className="strong ellipsis">{d.fileName ?? d.title}</div><div className="xs faint">{d.version ? `v${d.version} · ` : ''}{d.validFrom ?? '…'} ~ {d.validTo ?? '…'}{d.synthetic ? ' · 增强语料' : ''}</div></td>
                  <td className="num">{d.parents}</td>
                  <td className="num">{d.flagged}</td>
                  <td className="num">{d.score}</td>
                  <td>{d.issues.length ? <div className="col gap-4">{d.issues.map((i) => <div key={i.type} className="row gap-6 top"><Badge tone={SEV_TONE[i.severity]}>{i.label}</Badge><span className="xs muted">{i.detail}</span></div>)}</div> : <span className="faint">—</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {report.excluded.length ? (
        <div className="card">
          <div className="card-header"><h3><Undo2 size={15} />已剔除（{report.excluded.length}）</h3><Button size="xs" onClick={() => confirmDialog(`恢复全部 ${report.excluded.length} 个段落？相关文档会重新入库。`) && restore(report.excluded.map((x) => x.fingerprint))}>全部恢复</Button></div>
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>来源</th><th>内容</th><th>原因</th><th>时间</th><th /></tr></thead>
              <tbody>
                {report.excluded.map((x) => (
                  <tr key={x.fingerprint}>
                    <td style={{ maxWidth: 200 }}><div className="ellipsis">{x.fileName ?? '（文档已删除）'}</div><div className="xs faint ellipsis">{x.breadcrumb}</div></td>
                    <td style={{ maxWidth: 420 }}><div className="small clamp-2 muted">{x.preview}</div></td>
                    <td className="xs">{x.reason || '—'}</td>
                    <td className="xs faint">{formatDate(x.at)}</td>
                    <td><Button size="xs" onClick={() => restore([x.fingerprint])}><RotateCcw size={11} />恢复</Button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </>
  );
}

// ---------------------------------------------------------------------------------------
const ITEM_STATUS: Record<FaqStatus, { label: string; tone: string }> = {
  pending: { label: '待审核', tone: 'info' },
  gap: { label: '知识缺口', tone: 'bad' },
  approved: { label: '已通过', tone: 'ok' },
  rejected: { label: '已驳回', tone: '' },
};
const METHOD_LABEL: Record<FaqItem['method'], string> = { llm: '模型整理', extractive: '抽取式草稿', none: '无候选语料', manual: '人工录入' };
const BUCKET_LABEL: Record<FaqItem['bucket'], { label: string; tone: string }> = { none: { label: '无匹配', tone: 'bad' }, partial: { label: '部分匹配', tone: 'warn' }, manual: { label: '手动', tone: '' } };

function AugmentTab({ kbId, go }: { kbId: string; go: (key: PageKey) => void }) {
  const { toast, refreshKbs } = useApp();
  const [state, setState] = useState<AugmentState | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [filter, setFilter] = useState<FaqStatus | 'all'>('all');
  const [questions, setQuestions] = useState('');
  const [draft, setDraft] = useState({ question: '', answer: '' });

  const load = useCallback(() => api.corpus.augment(kbId).then(setState).catch((e) => toast(errMsg(e), 'bad')), [kbId, toast]);
  useEffect(() => { setState(null); load(); }, [load]);

  const run = async (key: string, fn: () => Promise<AugmentState>, ok?: (s: AugmentState) => string) => {
    setBusy(key);
    try { const s = await fn(); setState(s); if (ok) toast(ok(s), 'ok'); } catch (e) { toast(errMsg(e), 'bad'); } finally { setBusy(null); }
  };

  if (!state) return <div className="card"><div className="card-body"><Spinner label="加载中…" /></div></div>;
  const items = state.items.filter((it) => filter === 'all' || it.status === filter);
  const approved = state.items.filter((it) => it.status === 'approved');
  const dirty = state.items.some((it) => (it.status === 'approved') !== it.published);
  const count = (s: FaqStatus) => state.items.filter((it) => it.status === s).length;
  const pending = state.coverage.pendingGaps;

  return (
    <>
      <div className="grid grid-2" style={{ alignItems: 'start' }}>
        <div className="card">
          <div className="card-header"><h3><Sparkles size={15} />从覆盖缺口生成 FAQ 草稿</h3></div>
          <div className="card-body col gap-12">
            {!state.coverage.available ? (
              <Callout tone="warn">还没有覆盖分析结果。先在「语料覆盖」里分析一批真实用户问题，这里会列出"部分匹配 / 无匹配"的缺口。<div className="mt-8"><Button size="sm" onClick={() => go('coverage')}>去语料覆盖</Button></div></Callout>
            ) : (
              <div className="col gap-8">
                <div className="small">最近一次覆盖分析：<b>{state.coverage.n}</b> 条意图，缺口 <b>{state.coverage.gaps}</b> 条，尚未处理 <b>{pending.length}</b> 条。</div>
                {pending.length ? <div className="row gap-4 wrap">{pending.slice(0, 12).map((g) => <Badge key={g.intent} tone={BUCKET_LABEL[g.bucket].tone}>{g.intent}</Badge>)}{pending.length > 12 ? <span className="xs faint">…</span> : null}</div> : null}
                <div><Button variant="primary" size="sm" loading={busy === 'gen'} disabled={!pending.length} onClick={() => run('gen', () => api.corpus.generate(kbId), (s) => `已生成草稿，共 ${s.items.length} 条`)}><Sparkles size={13} />为 {Math.min(pending.length, 30)} 条缺口生成草稿</Button></div>
              </div>
            )}
            <div className="divider" />
            <textarea rows={4} value={questions} onChange={(e) => setQuestions(e.target.value)} placeholder={'也可以直接粘贴问题（每行一个），例如：\n年假能不能留到明年用\n出差住酒店超标了怎么办'} />
            <div><Button size="sm" loading={busy === 'q'} disabled={!questions.trim()} onClick={() => run('q', () => api.corpus.generate(kbId, questions.split('\n').map((q) => q.trim()).filter(Boolean)), () => { setQuestions(''); return '已生成草稿'; })}><Search size={13} />检索语料并生成草稿</Button></div>
            {!state.llm ? <div className="xs text-warning">未配置文本模型：生成的是抽取式草稿（最相关原文的前几句），需要人工改写后再通过。</div> : <div className="xs faint">模型只根据检索到的原文整理答案；资料不足时标为"知识缺口"，由人工补写或补充源文档。</div>}
          </div>
        </div>

        <div className="card">
          <div className="card-header"><h3><Plus size={15} />人工补充问答</h3></div>
          <div className="card-body col gap-12">
            <input value={draft.question} onChange={(e) => setDraft({ ...draft, question: e.target.value })} placeholder="问题（用用户的说法）" />
            <textarea rows={5} value={draft.answer} onChange={(e) => setDraft({ ...draft, answer: e.target.value })} placeholder="答案（留空则记为知识缺口，稍后补写）" />
            <div><Button size="sm" loading={busy === 'add'} disabled={!draft.question.trim()} onClick={() => run('add', () => api.corpus.addItem(kbId, draft.question, draft.answer), () => { setDraft({ question: '', answer: '' }); return '已添加'; })}><Plus size={13} />添加</Button></div>
            <div className="xs faint">知识库里确实没有的内容（覆盖分析的"无匹配"），在这里由业务人员补写，比等待源文档更新更快。</div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="row gap-6 wrap">
            <button className={`btn xs chip ${filter === 'all' ? 'primary' : ''}`} onClick={() => setFilter('all')}>全部 {state.items.length}</button>
            {(Object.keys(ITEM_STATUS) as FaqStatus[]).map((s) => <button key={s} className={`btn xs chip ${filter === s ? 'primary' : ''}`} onClick={() => setFilter(s)}>{ITEM_STATUS[s].label} {count(s)}</button>)}
          </div>
          <div className="row gap-8">
            <span className="xs muted">{state.docId ? `已发布 ${state.items.filter((i) => i.published).length} 条${state.publishedAt ? ` · ${formatDate(state.publishedAt)}` : ''}` : '尚未发布'}{dirty ? ' · 有未发布的更改' : ''}</span>
            <Button size="sm" variant={dirty ? 'primary' : 'default'} loading={busy === 'pub'} disabled={!dirty} onClick={() => run('pub', () => api.corpus.publish(kbId), (s) => { refreshKbs().catch(() => undefined); return s.docId ? `已发布 ${approved.length} 条 FAQ 到知识库` : '已撤下增强语料文档'; })}><Send size={13} />{approved.length || !state.docId ? `发布 ${approved.length} 条到知识库` : '撤下增强语料'}</Button>
          </div>
        </div>
        {items.length === 0 ? <Empty icon={<BookPlus size={32} />} title="暂无条目" desc="从覆盖缺口生成草稿，或人工补充问答。通过审核的条目发布后成为一篇「增强语料 · FAQ」文档，像普通文档一样被检索与引用。" /> : (
          <div className="card-body col gap-12">
            {items.map((it) => <FaqCard key={it.id + it.status + (it.published ? 1 : 0)} item={it} busy={busy === it.id} onSave={(body) => run(it.id, () => api.corpus.updateItem(kbId, it.id, body))} onDelete={() => confirmDialog('删除该条目？') && run(it.id, () => api.corpus.deleteItem(kbId, it.id))} />)}
          </div>
        )}
      </div>
    </>
  );
}

function FaqCard({ item, busy, onSave, onDelete }: { item: FaqItem; busy: boolean; onSave: (body: { question?: string; answer?: string; status?: FaqStatus }) => void; onDelete: () => void }) {
  const [q, setQ] = useState(item.question);
  const [a, setA] = useState(item.answer);
  const edited = q !== item.question || a !== item.answer;
  return (
    <div className={`chunk ${item.status === 'approved' ? 'hl' : ''}`}>
      <div className="row between wrap gap-8 mb-12">
        <div className="row gap-6 wrap">
          <Badge tone={ITEM_STATUS[item.status].tone}>{ITEM_STATUS[item.status].label}</Badge>
          <Badge tone={BUCKET_LABEL[item.bucket].tone}>{BUCKET_LABEL[item.bucket].label}{item.coverageScore != null ? ` · ${item.coverageScore.toFixed(2)}` : ''}</Badge>
          <Badge>{METHOD_LABEL[item.method]}</Badge>
          {item.published ? <Badge tone="accent"><Check size={10} />已发布</Badge> : null}
        </div>
        <div className="row gap-6">
          {edited ? <Button size="xs" loading={busy} onClick={() => onSave({ question: q, answer: a })}>保存修改</Button> : null}
          {item.status !== 'approved' ? <Button size="xs" variant="primary" loading={busy} disabled={!a.trim()} title={!a.trim() ? '先补写答案' : ''} onClick={() => onSave({ question: q, answer: a, status: 'approved' })}><Check size={11} />通过</Button> : null}
          {item.status !== 'rejected' ? <Button size="xs" loading={busy} onClick={() => onSave({ status: 'rejected' })}><X size={11} />驳回</Button> : <Button size="xs" loading={busy} onClick={() => onSave({ status: a.trim() ? 'pending' : 'gap' })}><Undo2 size={11} />撤销驳回</Button>}
          <Button size="xs" variant="ghost" icon onClick={onDelete} aria-label="删除"><Trash2 size={12} /></Button>
        </div>
      </div>
      <input value={q} onChange={(e) => setQ(e.target.value)} className="strong" />
      {item.intent !== q ? <div className="xs faint mt-4">原始意图：{item.intent}</div> : null}
      <textarea className="mt-8" rows={Math.min(8, Math.max(3, Math.ceil(a.length / 60)))} value={a} onChange={(e) => setA(e.target.value)} placeholder="补写答案…" />
      {item.note ? <div className={`xs mt-4 ${item.status === 'gap' ? 'text-danger' : 'text-warning'}`}>{item.note}</div> : null}
      {item.sources.length ? (
        <div className="col gap-4 mt-8">
          {item.sources.map((s) => <div key={s.chunkId} className="xs muted clamp-2"><Badge mono>来源</Badge> {s.breadcrumb}：{s.snippet}</div>)}
        </div>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------------------
const splitTerms = (s: string) => s.split(/[,，、;；|/\n]+/).map((t) => t.trim()).filter(Boolean);

function AliasesTab({ kbId, go }: { kbId: string; go: (key: PageKey) => void }) {
  const { toast } = useApp();
  const [rows, setRows] = useState<string[] | null>(null);
  const [saved, setSaved] = useState<string[]>([]);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [sugg, setSugg] = useState<AliasSuggestion[] | null>(null);
  const [finding, setFinding] = useState(false);

  useEffect(() => {
    setRows(null);
    setSugg(null);
    api.corpus.aliases(kbId).then((r) => { const lines = r.groups.map((g) => g.join('，')); setRows(lines.length ? lines : ['']); setSaved(lines); setUpdatedAt(r.updatedAt); }).catch((e) => toast(errMsg(e), 'bad'));
  }, [kbId, toast]);

  const groups = (rows ?? []).map(splitTerms).filter((g) => g.length >= 2);
  const dirty = JSON.stringify(groups.map((g) => g.join('，'))) !== JSON.stringify(saved);
  const save = async () => {
    setSaving(true);
    try {
      const r = await api.corpus.saveAliases(kbId, groups);
      const lines = r.groups.map((g) => g.join('，'));
      setRows([...lines, '']);
      setSaved(lines);
      setUpdatedAt(r.updatedAt);
      toast(r.reindexing ? `已保存 ${r.groups.length} 组：查询扩展已生效，正在后台重建索引写入文档扩展` : `已保存 ${r.groups.length} 组`, 'ok');
    } catch (e) { toast(errMsg(e), 'bad'); } finally { setSaving(false); }
  };
  const discover = async () => {
    setFinding(true);
    try { const s = await api.corpus.discoverAliases(kbId); setSugg(s); if (!s.length) toast('没有在语料里发现新的"全称（简称）"写法', 'warn'); } catch (e) { toast(errMsg(e), 'bad'); } finally { setFinding(false); }
  };
  const adopt = (terms: string[]) => {
    setRows((prev) => {
      const list = (prev ?? []).filter((r) => r.trim());
      const low = terms.map((t) => t.toLowerCase());
      const i = list.findIndex((r) => splitTerms(r).some((t) => low.includes(t.toLowerCase())));
      if (i >= 0) {
        const merged = Array.from(new Set([...splitTerms(list[i]), ...terms]));
        return [...list.slice(0, i), merged.join('，'), ...list.slice(i + 1), ''];
      }
      return [...list, terms.join('，'), ''];
    });
    setSugg((s) => s?.filter((x) => x.terms.join() !== terms.join()) ?? null);
  };

  if (!rows) return <div className="card"><div className="card-body"><Spinner label="加载中…" /></div></div>;
  return (
    <div className="grid grid-2" style={{ alignItems: 'start' }}>
      <div className="card">
        <div className="card-header"><h3><Replace size={15} />别名 / 同义词组</h3><span className="xs faint">{updatedAt ? `更新于 ${formatDate(updatedAt)}` : '尚未保存'}</span></div>
        <div className="card-body col gap-8">
          <div className="xs muted">每行一组，逗号或顿号分隔：缩写 ↔ 全称、产品代号 ↔ 正式名、口语 ↔ 书面语。</div>
          {rows.map((r, i) => (
            <div key={i} className="row gap-6">
              <input value={r} onChange={(e) => setRows(rows.map((x, j) => (j === i ? e.target.value : x)))} placeholder="K8s，Kubernetes" />
              {splitTerms(r).length === 1 ? <span className="xs text-warning" style={{ whiteSpace: 'nowrap' }}>至少两个词</span> : null}
              <Button size="xs" variant="ghost" icon aria-label="删除" onClick={() => setRows(rows.length > 1 ? rows.filter((_, j) => j !== i) : [''])}><Trash2 size={12} /></Button>
            </div>
          ))}
          <div className="row between mt-4">
            <Button size="xs" onClick={() => setRows([...rows, ''])}><Plus size={11} />添加一组</Button>
            <Button size="sm" variant="primary" loading={saving} disabled={!dirty} onClick={save}>保存（{groups.length} 组）</Button>
          </div>
          <div className="divider" />
          <div className="xs faint">
            双向生效：<b>查询扩展</b>（BM25 查询里出现组内任一词，补上同组其余词）保存即生效；<b>文档扩展</b>（切片里出现组内词，把别名写进向量与 BM25 文本）需要重建索引，保存后自动在后台排队。
            另外可以在「策略配置 → 语料增强」打开<b>关键词扩展</b>，为每个段落补充 TF-IDF 主题词与编号。
            <span className="mt-4" style={{ display: 'block' }}><Button size="xs" onClick={() => go('settings')}>去策略配置</Button></span>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header"><h3><Search size={15} />从语料发现</h3><Button size="sm" loading={finding} onClick={discover}><Sparkles size={13} />扫描语料</Button></div>
        {sugg === null ? <Empty icon={<Replace size={32} />} title="扫描语料里的别名写法" desc={<>识别"服务等级协议（SLA）""SLA（服务等级协议）""员工带薪年休假（以下简称“年假”）""X，又称 Y"等文档自己写明的对应关系。</>} /> : sugg.length === 0 ? <Empty title="没有新的候选" /> : (
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>别名组</th><th className="num">次数</th><th>出处</th><th /></tr></thead>
              <tbody>
                {sugg.map((s) => (
                  <tr key={s.terms.join('|')}>
                    <td><div className="row gap-4 wrap">{s.terms.map((t) => <Badge key={t} mono>{t}</Badge>)}</div></td>
                    <td className="num">{s.count}</td>
                    <td style={{ maxWidth: 280 }}><div className="xs muted clamp-2">{s.evidence}</div></td>
                    <td><Button size="xs" onClick={() => adopt(s.terms)}><Plus size={11} />加入</Button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------
const META_LABEL: Record<MetaField, string> = { title: '标题', version: '版本', valid_from: '生效日期', valid_to: '失效日期', department: '部门', tags: '标签', access_level: '密级' };
const SOURCE_LABEL: Record<string, string> = { rule: '规则', llm: '模型', tfidf: '主题词', manual: '人工' };
const fmt = (k: MetaField, v: unknown) => (v === null || v === undefined || v === '' || (Array.isArray(v) && !v.length) ? '—' : k === 'access_level' ? ACCESS_LEVELS[Number(v)]?.doc ?? String(v) : Array.isArray(v) ? v.join('、') : String(v));

function MetaTab({ kbId }: { kbId: string }) {
  const { toast, refreshKbs } = useApp();
  const [rows, setRows] = useState<MetaRow[] | null>(null);
  const [useLlm, setUseLlm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const [picked, setPicked] = useState<Record<string, Set<MetaField>>>({});
  const [applying, setApplying] = useState(false);

  const extract = async () => {
    setLoading(true);
    try {
      const r = await api.corpus.extractMeta(kbId, useLlm);
      setRows(r);
      // 默认勾选：当前为空、且建议值不同的字段（人工填过的不动）
      setPicked(Object.fromEntries(r.map((row) => [row.docId, new Set((Object.keys(row.suggestions) as MetaField[]).filter((k) => row.suggestions[k]!.empty && !row.suggestions[k]!.same))])));
    } catch (e) { toast(errMsg(e), 'bad'); } finally { setLoading(false); }
  };
  useEffect(() => { setRows(null); setPicked({}); }, [kbId]);

  const shown = (rows ?? []).filter((r) => showAll || Object.values(r.suggestions).some((s) => s && !s.same));
  const total = Object.values(picked).reduce((n, s) => n + s.size, 0);
  const toggle = (docId: string, k: MetaField) => setPicked((p) => { const s = new Set(p[docId] ?? []); if (s.has(k)) s.delete(k); else s.add(k); return { ...p, [docId]: s }; });
  const apply = async () => {
    const items = (rows ?? []).filter((r) => picked[r.docId]?.size).map((r) => ({ docId: r.docId, fields: Object.fromEntries([...picked[r.docId]].map((k) => [k, r.suggestions[k]!.value])) }));
    setApplying(true);
    try {
      const r = await api.corpus.applyMeta(kbId, items);
      toast(`已更新 ${r.updated} 份文档的元数据（检索过滤立即生效）`, 'ok');
      refreshKbs().catch(() => undefined);
      await extract();
    } catch (e) { toast(errMsg(e), 'bad'); } finally { setApplying(false); }
  };

  return (
    <div className="card">
      <div className="card-header">
        <h3><Tags size={15} />自动元数据</h3>
        <div className="row gap-12">
          <label className="row gap-6 small muted"><input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />规则没找到的字段用文本模型补</label>
          <Button size="sm" loading={loading} onClick={extract}><Sparkles size={13} />{rows ? '重新抽取' : '抽取元数据'}</Button>
        </div>
      </div>
      <div className="card-body xs faint" style={{ borderBottom: '1px solid var(--border)' }}>
        只认文档自己写明的信息："自 2026 年 1 月 1 日起施行""有效期至……""密级：机密""版本：V2.1""发布部门：财务部"。模型给出的值必须能在原文里找到依据，否则丢弃。
        新上传的文档会自动抽取并只填空字段（「策略配置 → 语料增强」可关闭）；这里用于批量复核已有文档。
      </div>
      {!rows ? <Empty icon={<Tags size={32} />} title="从正文抽取元数据" desc="版本、生效 / 失效日期、部门、密级是检索前硬过滤的依据：过期制度不召回、无权限文档不可见。" action={<Button size="sm" variant="primary" loading={loading} onClick={extract}>开始抽取</Button>} /> : (
        <>
          <div className="card-body row between wrap gap-8" style={{ borderBottom: '1px solid var(--border)' }}>
            <label className="row gap-6 small muted"><input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />显示没有新建议的文档（{rows.length - shown.length}）</label>
            <Button size="sm" variant="primary" loading={applying} disabled={!total} onClick={apply}><Check size={13} />应用所选（{total} 个字段）</Button>
          </div>
          {shown.length === 0 ? <Empty icon={<CheckCircle2 size={32} />} title="元数据已是最新" desc="抽取结果与当前值一致。" /> : (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>文档</th><th>字段</th><th>当前</th><th>建议</th><th>依据</th></tr></thead>
                <tbody>
                  {shown.flatMap((row) => {
                    const keys = (Object.keys(row.suggestions) as MetaField[]).filter((k) => showAll || !row.suggestions[k]!.same);
                    return keys.map((k, i) => {
                      const s = row.suggestions[k]!;
                      return (
                        <tr key={row.docId + k}>
                          {i === 0 ? <td rowSpan={keys.length} style={{ maxWidth: 220, verticalAlign: 'top' }}><div className="strong ellipsis" title={row.fileName}>{row.fileName}</div></td> : null}
                          <td style={{ whiteSpace: 'nowrap' }}><label className="row gap-6"><input type="checkbox" disabled={s.same} checked={!!picked[row.docId]?.has(k)} onChange={() => toggle(row.docId, k)} />{META_LABEL[k]}</label></td>
                          <td className={s.empty ? 'faint' : ''}>{fmt(k, row.current[k])}</td>
                          <td><span className={s.same ? 'faint' : 'strong'}>{fmt(k, s.value)}</span> <Badge tone={s.source === 'rule' ? 'ok' : s.source === 'llm' ? 'accent' : ''}>{SOURCE_LABEL[s.source] ?? s.source}</Badge>{!s.empty && !s.same ? <span className="xs text-warning" style={{ marginLeft: 6 }}>将覆盖人工值</span> : null}</td>
                          <td style={{ maxWidth: 320 }}><div className="xs muted clamp-2" title={s.evidence}>{s.evidence}</div></td>
                        </tr>
                      );
                    });
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
