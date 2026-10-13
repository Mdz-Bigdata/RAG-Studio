// Agent 湖：问答链路沉淀出的上下文资产（问题 / 检索日志 / 回答 / 反馈 / 工具调用 / 记忆 / 黑板 / 文档）放在一张 LanceDB 宽表里，
// 这里做湖分析（检索质量、知识缺口、主题、SQL、分析智能体），并管理版本 / Tag / 分支 / 派生列。
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { BookOpen, Bot, Check, ChevronDown, ChevronLeft, ChevronRight, Database, Eye, EyeOff, GitBranch, GitMerge, History, Layers, Play, RefreshCw, Search, Sparkles, Tag, Terminal, Trash2, TriangleAlert, Waves, X } from 'lucide-react';
import { KIND_LABEL, RETEST_LABEL, lakeApi, type AnalysisResult, type CatalogEntryInput, type LakeCatalog, type LakeGap, type RetestResponse, type RetestStatus, type BranchDiff, type LakeHit, type LakeKind, type LakeOverview, type LakeQuality, type LakeRecord, type LakeRef, type LakeTopics, type LakeVersions, type SqlResult } from '../api/lake';
import { Markdown } from '../components/markdown';
import { Badge, Button, Callout, Drawer, Empty, ScoreBar, Spinner, Tabs, confirmDialog } from '../components/ui';
import { formatBytes } from '../lib/format';
import { useApp } from '../store/app-store';

type TabKey = 'overview' | 'gaps' | 'topics' | 'search' | 'sql' | 'agent' | 'versions';
const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));
const fmtTs = (ts: number | null | undefined) => (ts ? new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false }) : '—');
const pctText = (v: number | null | undefined) => (v === null || v === undefined ? '—' : `${Math.round(v * 100)}%`);
const KIND_TONE: Partial<Record<LakeKind, string>> = { question: 'accent', retrieval: 'info', answer: 'ok', analysis: 'weaved', tool_call: 'mmr', memory: 'hyde', board: 'fused', document: 'sparse', doc_event: 'rerank', retest: 'ok' };
const KINDS = Object.keys(KIND_LABEL) as LakeKind[];

const SQL_SAMPLES: { label: string; sql: string }[] = [
  { label: '各类型记录数', sql: "SELECT kind, count(*) AS n FROM lake GROUP BY kind ORDER BY n DESC" },
  { label: '每天的问题数与平均检索分', sql: "SELECT q.day, count(*) AS questions, round(avg(r.score), 4) AS avg_score\nFROM lake q LEFT JOIN lake r ON r.parent_id = q.id AND r.kind = 'retrieval'\nWHERE q.kind = 'question'\nGROUP BY q.day ORDER BY q.day" },
  { label: '检索策略对比', sql: "SELECT json_extract_string(meta, '$.strategy') AS strategy, count(*) AS n,\n       round(avg(score), 4) AS avg_score, round(median(latency_ms)) AS p50_ms\nFROM lake WHERE kind = 'retrieval' GROUP BY 1 ORDER BY n DESC" },
  { label: '拒答 / 被点踩的回答', sql: "SELECT a.day, q.content AS question, a.status, a.feedback\nFROM lake a JOIN lake q ON q.id = a.parent_id\nWHERE a.kind = 'answer' AND (a.status <> 'ok' OR a.feedback < 0)\nORDER BY a.ts DESC" },
  { label: '最慢的 10 次检索', sql: "SELECT day, title AS question, latency_ms, score FROM lake\nWHERE kind = 'retrieval' ORDER BY latency_ms DESC NULLS LAST LIMIT 10" },
];
const AGENT_SAMPLES = ['最近哪些问题没答上来？该补什么语料？', '用户最关心哪些主题？', '检索延迟和置信度怎么样，哪种策略效果更好？', '每天有多少提问？趋势如何？', '拒答率和点踩率各是多少，口径是什么？', '哪些文档被检索得最多？'];

// ------------------------------------------------------------------ 小部件
function KindBadge({ kind }: { kind: string }) {
  return <Badge tone={KIND_TONE[kind as LakeKind] ?? ''}>{KIND_LABEL[kind as LakeKind] ?? kind}</Badge>;
}

/** 单系列竖条图（每天的记录数）：无图例，标题即系列名；悬停显示数值。 */
function DailyBars({ data, field }: { data: { day: string; value: number }[]; field: string }) {
  const max = Math.max(1, ...data.map((d) => d.value));
  if (!data.some((d) => d.value)) return <div className="xs faint">近 30 天没有{field}</div>;
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 120, borderBottom: '1px solid var(--border)' }} role="img" aria-label={`每天的${field}`}>
        {data.map((d) => (
          <div key={d.day} title={`${d.day}：${d.value}`} style={{ flex: 1, minWidth: 6, height: '100%', display: 'flex', alignItems: 'flex-end', cursor: 'default' }}>
            <div style={{ width: '100%', height: `${(d.value / max) * 100}%`, minHeight: d.value ? 2 : 0, background: 'var(--accent)', borderRadius: '4px 4px 0 0' }} />
          </div>
        ))}
      </div>
      <div className="row between xs faint mt-4"><span>{data[0].day}</span><span>最多 {max}</span><span>{data[data.length - 1].day}</span></div>
    </div>
  );
}

/** 横向条形：标签 + 条 + 数值（文字用正文色，条用强调色）。 */
function HBars({ items }: { items: { label: string; value: number; hint?: string }[] }) {
  const max = Math.max(1, ...items.map((i) => i.value));
  return (
    <div className="col gap-6">
      {items.map((i) => (
        <div key={i.label} className="row gap-8" title={i.hint ?? `${i.label}：${i.value}`}>
          <span className="small ellipsis" style={{ width: 110, flexShrink: 0 }}>{i.label}</span>
          <div style={{ flex: 1 }}><ScoreBar value={i.value} max={max} /></div>
          <span className="small tnum" style={{ width: 44, textAlign: 'right' }}>{i.value}</span>
        </div>
      ))}
    </div>
  );
}

/** SQL 结果里有 day 列和一个数值列、且不止几行时，先画成每天的条形图，再给表。 */
function SqlResultView({ columns, rows }: { columns: string[]; rows: unknown[][] }) {
  const di = columns.findIndex((c) => c === 'day' || c === 'q.day' || c.endsWith('_day'));
  const vi = di < 0 ? -1 : columns.findIndex((_, i) => i !== di && rows.every((r) => typeof r[i] === 'number' || r[i] === null));
  const series = di >= 0 && vi >= 0 && rows.length >= 3 ? rows.map((r) => ({ day: String(r[di]), value: Number(r[vi] ?? 0) })).sort((a, b) => a.day.localeCompare(b.day)) : null;
  return (
    <div className="col gap-8">
      {series ? <div className="card-body" style={{ paddingTop: 4, paddingBottom: 0 }}><DailyBars data={series} field={columns[vi]} /></div> : null}
      <ResultTable columns={columns} rows={rows} />
    </div>
  );
}

function ResultTable({ columns, rows }: { columns: string[]; rows: unknown[][] }) {
  const cell = (v: unknown) => (v === null || v === undefined ? <span className="faint">NULL</span> : typeof v === 'object' ? JSON.stringify(v) : String(v));
  return (
    <div className="table-wrap" style={{ maxHeight: 420, overflow: 'auto' }}>
      <table className="table">
        <thead><tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j} className={typeof v === 'number' ? 'num' : ''} style={{ maxWidth: 420 }}><div className="clamp-3 pre small">{cell(v)}</div></td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

/** 记录详情：本条 + 父记录 + 子记录（问题 → 检索日志 → 回答 / 分析 → 工具调用）。 */
function RecordDrawer({ id, onClose, onOpen, onDelete }: { id: string; onClose: () => void; onOpen: (id: string) => void; onDelete: (id: string, what?: string) => Promise<boolean> }) {
  const [data, setData] = useState<{ record: LakeRecord; parent: LakeRecord | null; children: LakeRecord[] } | null>(null);
  const [error, setError] = useState('');
  const [rev, setRev] = useState(0);                 // 删掉一条关联记录（检索日志 / 回答）后重新加载
  useEffect(() => {
    setData(null);
    lakeApi.record(id).then(setData).catch((e) => setError(errMsg(e)));
  }, [id, rev]);
  const kindName = (kind: string) => KIND_LABEL[kind as LakeKind] ?? '记录';
  const deleteLinked = (c: LakeRecord) => onDelete(c.id, `这条${kindName(c.kind)}`).then((ok) => { if (ok) setRev((v) => v + 1); return ok; });
  const r = data?.record;
  const meta = useMemo(() => { try { return JSON.stringify(JSON.parse(r?.meta || '{}'), null, 2); } catch { return r?.meta ?? ''; } }, [r]);
  const derived = r ? Object.keys(r).filter((k) => !['id', 'kind', 'session_id', 'user_id', 'agent', 'kb_id', 'title', 'content', 'meta', 'refs', 'parent_id', 'status', 'score', 'latency_ms', 'feedback', 'ts', 'day', 'embed_model'].includes(k)) : [];
  return (
    <Drawer title={r ? <span className="row gap-6"><KindBadge kind={r.kind} />记录详情</span> : '记录详情'} subtitle={r ? `${fmtTs(r.ts)} · ${r.id}` : undefined} onClose={onClose}
      footer={r ? <Button size="sm" variant="ghost" onClick={() => onDelete(r.id, `这条${kindName(r.kind)}`)}><Trash2 size={13} />删除这条记录{data?.children.length ? '（含关联记录）' : ''}</Button> : undefined}>
      {error ? <Callout tone="bad">{error}</Callout> : !r ? <Spinner label="加载中" /> : (
        <div className="col gap-12">
          <div className="row gap-6 wrap">
            <Badge>状态 {r.status}</Badge>
            {r.score !== null ? <Badge mono>score {r.score.toFixed(4)}</Badge> : null}
            {r.latency_ms !== null ? <Badge mono>{Math.round(r.latency_ms)} ms</Badge> : null}
            {r.feedback ? <Badge tone={r.feedback > 0 ? 'ok' : 'bad'}>{r.feedback > 0 ? '👍 点赞' : '👎 点踩'}</Badge> : null}
            <Badge>智能体 {r.agent}</Badge><Badge>用户 {r.user_id}</Badge>
            {derived.map((k) => <Badge key={k} tone="accent">{k} = {String(r[k] ?? '—')}</Badge>)}
          </div>
          {r.kind === 'answer' || r.kind === 'analysis' ? <div className="card pad"><Markdown text={r.content} /></div> : <div className="card pad small pre">{r.content}</div>}
          {data.parent ? (
            <div><div className="section-label">父记录</div>
              <RecordRow id={data.parent.id} onOpen={onOpen} onDelete={(pid) => onDelete(pid, `父记录（${kindName(data.parent!.kind)}）`)} delTitle={`删除父记录（${kindName(data.parent.kind)}），连同本条`}><KindBadge kind={data.parent.kind} /><span className="small ellipsis flex-1">{data.parent.content}</span></RecordRow>
            </div>
          ) : null}
          {data.children.length ? (
            <div><div className="section-label">关联记录（{data.children.length}）</div>
              {data.children.map((c) => (
                <RecordRow key={c.id} id={c.id} onOpen={onOpen} onDelete={() => deleteLinked(c)} delTitle={`删除这条${kindName(c.kind)}`}><KindBadge kind={c.kind} /><span className="small ellipsis flex-1">{c.title || c.content}</span></RecordRow>
              ))}
            </div>
          ) : null}
          <div><div className="section-label">元数据 meta</div><pre className="card pad xs pre mono" style={{ maxHeight: 320, overflow: 'auto', margin: 0 }}>{meta}</pre></div>
        </div>
      )}
    </Drawer>
  );
}

// 记录列表行：整行点开详情，悬停出现 🗑 直接删除（行本身不能是 <button>，否则里面放不了删除按钮）
function RecordRow({ id, onOpen, onDelete, top, delTitle = '删除这条记录', selected, children }: { id: string; onOpen: (id: string) => void; onDelete?: (id: string) => Promise<boolean>; top?: boolean; delTitle?: string; selected?: boolean; children: React.ReactNode }) {
  return (
    <div role="button" tabIndex={0} className={`list-item record-row ${selected ? 'is-selected' : ''}`} style={{ alignItems: top ? 'flex-start' : undefined }} onClick={() => onOpen(id)}
      onKeyDown={(e) => { if (e.target === e.currentTarget && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); onOpen(id); } }}>
      {children}
      {onDelete ? <button type="button" className="ref-del" title={delTitle} aria-label={delTitle} onClick={(e) => { e.stopPropagation(); onDelete(id); }}><Trash2 size={13} /></button> : null}
    </div>
  );
}

const RECENT_PAGE_SIZE = 50;

/** 最近写入：每页 50 条，列表区域内滚动，上一页 / 下一页翻页；可按类型多选筛选 */
function RecentWrites({ refv, reloadKey, onOpen, onDelete, onDeleteMany, kindCounts }: { refv: LakeRef; reloadKey: number; onOpen: (id: string) => void; onDelete?: (id: string) => Promise<boolean>; onDeleteMany?: (ids: string[], what?: string) => Promise<boolean>; kindCounts: Partial<Record<LakeKind, number>> }) {
  const [page, setPage] = useState(0);
  const [kinds, setKinds] = useState<LakeKind[]>([]);                // 空 = 全部类型
  const kindKey = kinds.join(',');
  const kindOptions = (Object.entries(kindCounts) as [LakeKind, number][]).filter(([, n]) => n > 0).sort((a, b) => b[1] - a[1]);
  const allCount = kindOptions.reduce((n, [, c]) => n + c, 0);
  const toggleKind = (k: LakeKind) => { setKinds((cur) => (cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k])); setPage(0); };

  // ---- 批量选择：跨页保留；换类型筛选 / 版本时清空（避免删到看不见的记录）
  const selectable = !!onDeleteMany;
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [selectingAll, setSelectingAll] = useState(false);
  const [deleting, setDeleting] = useState(false);
  useEffect(() => { setSelected(new Set()); }, [refv, kindKey]);
  const toggleOne = (id: string) => setSelected((cur) => { const next = new Set(cur); if (next.has(id)) next.delete(id); else next.add(id); return next; });
  const [items, setItems] = useState<LakeRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const seq = useRef(0);
  const pages = Math.max(1, Math.ceil(total / RECENT_PAGE_SIZE));

  // 切换版本 / 分支：回到第一页；去掉当前快照里已经没有的类型
  useEffect(() => { setPage(0); }, [refv]);
  useEffect(() => { setKinds((cur) => cur.filter((k) => (kindCounts[k] ?? 0) > 0)); }, [kindCounts]);
  useEffect(() => {
    const n = ++seq.current;
    setLoading(true);
    setError('');
    lakeApi.records({ ...refv, kind: kindKey || undefined, limit: RECENT_PAGE_SIZE, offset: page * RECENT_PAGE_SIZE })
      .then((r) => {
        if (n !== seq.current) return;                       // 翻页太快：丢弃过期结果
        // 删除后当前页可能空了（最后一页只剩被删的那条）：退回上一页
        if (!r.items.length && page > 0 && r.total > 0) { setPage(Math.max(0, Math.ceil(r.total / RECENT_PAGE_SIZE) - 1)); return; }
        setItems(r.items);
        setTotal(r.total);
        scrollRef.current?.scrollTo({ top: 0 });
      })
      .catch((e) => { if (n === seq.current) setError(errMsg(e)); })
      .finally(() => { if (n === seq.current) setLoading(false); });
  }, [refv, page, reloadKey, kindKey]);

  const pageIds = items.map((r) => r.id);
  const pageSelected = pageIds.filter((id) => selected.has(id)).length;
  const pageAll = pageIds.length > 0 && pageSelected === pageIds.length;
  const togglePage = () => setSelected((cur) => {
    const next = new Set(cur);
    if (pageAll) pageIds.forEach((id) => next.delete(id)); else pageIds.forEach((id) => next.add(id));
    return next;
  });
  // 选中当前筛选下的全部记录（跨页）：分批取 id
  const selectAllMatching = async () => {
    setSelectingAll(true);
    try {
      const ids: string[] = [];
      for (let offset = 0; offset < total; offset += 500) {
        const r = await lakeApi.records({ ...refv, kind: kindKey || undefined, limit: 500, offset });
        ids.push(...r.items.map((x) => x.id));
        if (r.items.length < 500) break;
      }
      setSelected(new Set(ids));
    } catch (e) { setError(errMsg(e)); } finally { setSelectingAll(false); }
  };
  const deleteSelected = async () => {
    if (!onDeleteMany || !selected.size) return;
    setDeleting(true);
    const ok = await onDeleteMany([...selected], `所选的 ${selected.size} 条记录`);
    setDeleting(false);
    if (ok) setSelected(new Set());
  };
  const headCheckRef = useRef<HTMLInputElement>(null);
  useEffect(() => { if (headCheckRef.current) headCheckRef.current.indeterminate = pageSelected > 0 && !pageAll; }, [pageSelected, pageAll]);

  const from = total ? page * RECENT_PAGE_SIZE + 1 : 0;
  const to = Math.min(total, (page + 1) * RECENT_PAGE_SIZE);
  const pager = (
    <div className="row gap-6">
      <span className="xs faint">{total ? `第 ${from}–${to} 条 · 共 ${total} 条 · 第 ${page + 1} / ${pages} 页` : '共 0 条'}</span>
      <Button size="xs" disabled={page === 0 || loading} onClick={() => setPage((p) => Math.max(0, p - 1))}><ChevronLeft size={12} />上一页</Button>
      <Button size="xs" disabled={page >= pages - 1 || loading} onClick={() => setPage((p) => Math.min(pages - 1, p + 1))}>下一页<ChevronRight size={12} /></Button>
    </div>
  );
  return (
    <div className="card">
      <div className="card-header">
        <h3>最近写入{loading ? <Spinner size={12} /> : null}</h3>
        {pager}
      </div>
      {kindOptions.length > 1 ? (
        <div className="kind-chips" role="group" aria-label="按类型筛选">
          <button type="button" className={`kind-chip ${kinds.length ? '' : 'on'}`} aria-pressed={!kinds.length} onClick={() => { setKinds([]); setPage(0); }}>全部<span>{allCount}</span></button>
          {kindOptions.map(([k, n]) => (
            <button key={k} type="button" className={`kind-chip tone-${KIND_TONE[k] ?? 'none'} ${kinds.includes(k) ? 'on' : ''}`} aria-pressed={kinds.includes(k)} onClick={() => toggleKind(k)} title={kinds.includes(k) ? `取消「${KIND_LABEL[k] ?? k}」` : `筛选「${KIND_LABEL[k] ?? k}」，可多选`}>
              {KIND_LABEL[k] ?? k}<span>{n}</span>
            </button>
          ))}
          {kinds.length ? <button type="button" className="link-btn xs" onClick={() => { setKinds([]); setPage(0); }}>清除筛选</button> : null}
        </div>
      ) : null}
      {selectable && items.length ? (
        <div className={`bulk-bar ${selected.size ? 'active' : ''}`}>
          <label className="row gap-6 small" style={{ cursor: 'pointer' }}>
            <input ref={headCheckRef} type="checkbox" checked={pageAll} onChange={togglePage} aria-label="全选本页" />
            全选本页
          </label>
          {selected.size ? (
            <>
              <span className="small strong">已选 {selected.size} 条</span>
              {pageAll && selected.size < total ? (
                <button type="button" className="link-btn small" disabled={selectingAll} onClick={selectAllMatching}>{selectingAll ? '正在选择…' : `选择${kinds.length ? '当前筛选下的' : ''}全部 ${total} 条`}</button>
              ) : selected.size === total && total > pageIds.length ? <span className="xs faint">已选中{kinds.length ? '当前筛选下的' : ''}全部记录（含其他页）</span> : null}
              <span className="flex-1" />
              <Button size="xs" onClick={() => setSelected(new Set())}>取消选择</Button>
              <Button size="xs" variant="danger" loading={deleting} onClick={deleteSelected}><Trash2 size={12} />删除所选（{selected.size}）</Button>
            </>
          ) : <span className="xs faint">勾选记录后可批量删除</span>}
        </div>
      ) : null}
      {error ? <div className="card-body"><Callout tone="bad">{error}</Callout></div> : null}
      <div ref={scrollRef} className="recent-writes scroll" aria-busy={loading}>
        {items.length ? items.map((r) => (
          <RecordRow key={r.id} id={r.id} onOpen={onOpen} onDelete={onDelete} selected={selected.has(r.id)}>
            {selectable ? (
              <input type="checkbox" className="row-check" checked={selected.has(r.id)} aria-label="选择这条记录"
                onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()} onChange={() => toggleOne(r.id)} />
            ) : null}
            <KindBadge kind={r.kind} /><span className="small ellipsis flex-1">{r.title || r.content}</span><span className="xs faint">{fmtTs(r.ts)}</span>
          </RecordRow>
        )) : !loading ? <div className="card-body xs faint">{kinds.length ? '当前类型下暂无记录' : '暂无记录'}</div> : null}
        {items.length && pages > 1 ? <div className="recent-writes-foot">{pager}</div> : null}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ 各标签页
function OverviewTab({ refv, onOpen, onDelete, onDeleteMany, reloadKey }: { refv: LakeRef; onOpen: (id: string) => void; onDelete?: (id: string) => Promise<boolean>; onDeleteMany?: (ids: string[], what?: string) => Promise<boolean>; reloadKey: number }) {
  const [ov, setOv] = useState<LakeOverview | null>(null);
  const [q, setQ] = useState<LakeQuality | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    setError('');
    Promise.all([lakeApi.overview(refv), lakeApi.quality(refv)])
      .then(([o, qq]) => { setOv(o); setQ(qq); })
      .catch((e) => setError(errMsg(e)));
  }, [refv, reloadKey]);
  if (error) return <Callout tone="bad">{error}</Callout>;
  if (!ov || !q) return <Spinner label="读取 Agent 湖" />;
  const s = ov.stats;
  // 连续 30 天的坐标轴（没有提问的日子补 0），柱子宽度不随数据天数变化
  const byDay = new Map(s.daily.map((d) => [d.day, d.question ?? 0]));
  const daily = Array.from({ length: 30 }, (_, i) => {
    const d = new Date(Date.now() - (29 - i) * 86400000);
    const day = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    return { day, value: byDay.get(day) ?? 0 };
  });
  return (
    <div className="col gap-16">
      <div className="card">
        <div className="stat-strip">
          <div className="stat"><div className="label">记录</div><div className="value">{s.rows}</div><div className="sub">{s.sessions} 个会话 · {s.users} 个用户</div></div>
          <div className="stat"><div className="label">问题</div><div className="value">{q.questions}</div><div className="sub">{q.answers} 个回答</div></div>
          <div className="stat"><div className="label">知识缺口</div><div className="value" style={{ color: q.gapCount ? 'var(--warning)' : undefined }}>{q.gapCount}</div><div className="sub">占问题 {pctText(q.gapRate)}</div></div>
          <div className="stat"><div className="label">拒答率</div><div className="value">{pctText(q.refusalRate)}</div><div className="sub">👍 {q.feedback.up} · 👎 {q.feedback.down}</div></div>
          <div className="stat"><div className="label">检索耗时 p50</div><div className="value">{q.latency.p50 ?? '—'}<span className="xs faint"> ms</span></div><div className="sub">p95 {q.latency.p95 ?? '—'} ms</div></div>
          <div className="stat"><div className="label">平均最高分</div><div className="value">{q.score.avg?.toFixed(3) ?? '—'}</div><div className="sub">低于 {q.score.lowThreshold} 的 {q.score.lowCount} 次</div></div>
        </div>
      </div>
      <div className="grid grid-2">
        <div className="card"><div className="card-header"><h3><History size={15} />每天的提问数（近 30 天）</h3></div><div className="card-body"><DailyBars data={daily} field="提问" /></div></div>
        <div className="card"><div className="card-header"><h3><Layers size={15} />湖里有什么</h3><span className="xs faint">表 agent_lake · v{ov.info.version ?? '—'} · {formatBytes(ov.info.bytes ?? 0)}</span></div>
          <div className="card-body"><HBars items={KINDS.filter((k) => s.kinds[k]).map((k) => ({ label: KIND_LABEL[k], value: s.kinds[k] ?? 0 }))} />{!s.rows ? <div className="xs faint">还是空的：在智能问答里提问会自动沉淀，或点右上角「回填历史数据」。</div> : null}</div></div>
      </div>
      <div className="grid grid-2">
        <div className="card"><div className="card-header"><h3><Sparkles size={15} />检索置信度分布</h3><span className="xs faint">最终上下文的最高重排分</span></div>
          <div className="card-body"><HBars items={q.score.buckets.map((b) => ({ label: b.label, value: b.count }))} /></div></div>
        <div className="card"><div className="card-header"><h3><Database size={15} />被检索最多的文档</h3></div>
          {q.topDocs.length ? (
            <div className="table-wrap"><table className="table"><thead><tr><th>文档</th><th className="num">进入上下文</th><th className="num">被引用</th></tr></thead>
              <tbody>{q.topDocs.map((d) => <tr key={d.docId}><td className="ellipsis" style={{ maxWidth: 280 }}>{d.fileName}</td><td className="num">{d.retrieved}</td><td className="num">{d.cited}</td></tr>)}</tbody></table></div>
          ) : <div className="card-body xs faint">暂无检索日志</div>}
        </div>
      </div>
      {q.byStrategy.length ? (
        <div className="card"><div className="card-header"><h3>检索策略对比</h3></div>
          <div className="table-wrap"><table className="table"><thead><tr><th>策略</th><th className="num">次数</th><th className="num">平均最高分</th><th className="num">耗时 p50（ms）</th></tr></thead>
            <tbody>{q.byStrategy.map((b) => <tr key={b.strategy}><td>{b.strategy}</td><td className="num">{b.count}</td><td className="num">{b.avgScore?.toFixed(4) ?? '—'}</td><td className="num">{b.p50LatencyMs ?? '—'}</td></tr>)}</tbody></table></div></div>
      ) : null}
      <RecentWrites refv={refv} reloadKey={reloadKey} onOpen={onOpen} onDelete={onDelete} onDeleteMany={onDeleteMany} kindCounts={s.kinds} />
    </div>
  );
}

const RETEST_TONE: Record<RetestStatus, string> = { resolved: 'ok', improved: 'info', answerable: 'ok', still_gap: 'bad', wrong_kb: 'accent', error: 'warn' };

// 缺口行的操作：忽略（不再算缺口，可恢复）/ 恢复 / 删除（连同检索日志、回答）。看快照时不提供。
type GapActions = { ignore: (g: LakeGap, ignored: boolean) => void; remove: (g: LakeGap) => void };

function GapRow({ g, onOpen, resolved, actions, ignored }: { g: LakeGap; onOpen: (id: string) => void; resolved?: boolean; actions?: GapActions; ignored?: boolean }) {
  return (
    <tr style={{ cursor: 'pointer', opacity: resolved ? 0.75 : 1 }} onClick={() => onOpen(g.questionId)}>
      <td style={{ maxWidth: 380 }}><div className="strong small">{g.question}</div>{g.answer ? <div className="xs faint clamp-2">{g.answer}</div> : null}</td>
      <td><div className="row gap-4 wrap">{g.reasons.map((r) => <Badge key={r} tone={resolved ? '' : r.includes('点踩') || r.includes('失败') ? 'bad' : 'warn'}>{r}</Badge>)}</div></td>
      <td className="num">{g.score?.toFixed(3) ?? '—'}</td>
      <td>
        {g.retest ? <Badge tone={RETEST_TONE[g.retest.status]} title={`复测于 ${fmtTs(g.retest.ts)}`}>{RETEST_LABEL[g.retest.status]}{g.retest.score !== null ? ` · ${g.retest.score.toFixed(3)}` : ''}</Badge> : <span className="xs faint">未复测</span>}
        {g.retest?.altKb ? <div className="xs mt-4">能回答的库：<b>{g.retest.altKb.name}</b>（{g.retest.altKb.score.toFixed(2)}）</div> : null}
        {g.docChangesAfter?.length ? <div className="xs faint mt-4" title={g.docChangesAfter.join('、')}>之后变更：{g.docChangesAfter.join('、')}</div> : null}
      </td>
      <td className="xs faint">{fmtTs(g.ts)}</td>
      {actions ? (
        <td className="num" onClick={(e) => e.stopPropagation()} style={{ whiteSpace: 'nowrap' }}>
          <Button size="sm" variant="ghost" title={ignored ? '恢复为知识缺口' : '忽略：不再算作知识缺口（记录保留，可恢复）'} onClick={() => actions.ignore(g, !ignored)}>{ignored ? <><Eye size={13} />恢复</> : <><EyeOff size={13} />忽略</>}</Button>
          <Button size="sm" variant="ghost" title="删除这个问题及其检索日志、回答" aria-label="删除这个问题" onClick={() => actions.remove(g)}><Trash2 size={13} /></Button>
        </td>
      ) : null}
    </tr>
  );
}

function GapsTab({ refv, onOpen, go, onDelete }: { refv: LakeRef; onOpen: (id: string) => void; go: (t: TabKey) => void; onDelete?: (id: string, question: string) => Promise<boolean> }) {
  const { toast } = useApp();
  const [q, setQ] = useState<LakeQuality | null>(null);
  const [days, setDays] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [includeWeak, setIncludeWeak] = useState(false);
  const [checkAnswer, setCheckAnswer] = useState(true);
  const [retesting, setRetesting] = useState(false);
  const [retest, setRetest] = useState<RetestResponse | null>(null);
  const [showResolved, setShowResolved] = useState(false);
  const [showMisrouted, setShowMisrouted] = useState(true);
  const [showIgnored, setShowIgnored] = useState(false);
  const load = useCallback(() => { setQ(null); lakeApi.quality(refv, days).then(setQ).catch((e) => setError(errMsg(e))); }, [refv, days]);
  useEffect(load, [load]);
  const isSnapshot = !!(refv.version || refv.branch);
  const actions: GapActions | undefined = onDelete && !isSnapshot ? {
    ignore: (g, ignored) => {
      lakeApi.ignoreGap(g.questionId, ignored)
        .then(() => { toast(ignored ? '已忽略，不再算作知识缺口（可在「已忽略」里恢复）' : '已恢复为知识缺口', 'ok'); load(); })
        .catch((e) => toast(errMsg(e), 'bad'));
    },
    remove: (g) => { onDelete(g.questionId, g.question).then((ok) => { if (ok) load(); }); },
  } : undefined;
  const runRetest = async () => {
    setRetesting(true);
    try {
      const r = await lakeApi.retest({ limit: 20, includeWeak, checkAnswer });
      setRetest(r);
      toast(`复测 ${r.tested} 个问题：已解决 ${r.counts.resolved}、有改善 ${r.counts.improved}、能回答（检索分偏低）${r.counts.answerable}、仍是缺口 ${r.counts.still_gap}、答案在其它知识库 ${r.counts.wrong_kb}`, r.counts.still_gap ? 'warn' : 'ok');
      load();
    } catch (e) { toast(errMsg(e), 'bad'); } finally { setRetesting(false); }
  };
  if (error) return <Callout tone="bad">{error}</Callout>;
  return (
    <div className="col gap-16">
      <div className="card">
        <div className="card-header">
          <h3><TriangleAlert size={15} />知识缺口{q ? `（未解决 ${q.gapCount} / 问题 ${q.questions}）` : ''}</h3>
          <div className="row gap-8">
            <select value={days ?? ''} onChange={(e) => setDays(e.target.value ? Number(e.target.value) : null)} style={{ width: 120 }}>
              <option value="">全部时间</option><option value="1">近 1 天</option><option value="7">近 7 天</option><option value="30">近 30 天</option>
            </select>
            <Button size="sm" onClick={() => go('agent')}><Bot size={13} />让智能体分析</Button>
          </div>
        </div>
        <div className="card-body row between wrap gap-8" style={{ borderBottom: '1px solid var(--border)' }}>
          <div className="xs muted" style={{ maxWidth: 620 }}>
            判定规则：没检索到资料、检索最高分低于阈值、回答拒答、用户点踩、生成失败。湖里记录的是历史——补了语料之后点「复测缺口」，
            用现在的知识库重新检索并生成回答，已解决的会移到下方「复测已解决」；本库答不上、但别的知识库能答的，归为「答案在其它知识库」（选错库 / 路由问题，不用补语料）。{q ? <> 历史缺口 {q.historicalGapCount} 个，复测已解决 {q.resolvedCount} 个，答案在其它知识库 {q.misroutedCount} 个{q.lastRetestTs ? `，上次复测 ${fmtTs(q.lastRetestTs)}` : ''}。</> : null}
          </div>
          <div className="row gap-12 wrap">
            <label className="row gap-6 xs" title="连同当时答了、但检索分低于 0.3 的问题一起复测"><input type="checkbox" checked={includeWeak} onChange={(e) => setIncludeWeak(e.target.checked)} style={{ width: 'auto' }} />含低分问题</label>
            <label className="row gap-6 xs" title="每个问题再生成一次回答，判断是否拒答（更准，但更慢）"><input type="checkbox" checked={checkAnswer} onChange={(e) => setCheckAnswer(e.target.checked)} style={{ width: 'auto' }} />生成回答检查拒答</label>
            <Button size="sm" variant="primary" loading={retesting} disabled={isSnapshot} title={isSnapshot ? '复测要和现在的知识库比，请切回「主表 · 最新」' : undefined} onClick={runRetest}><RefreshCw size={13} />复测缺口</Button>
          </div>
        </div>
        {!q ? <div className="card-body"><Spinner /></div> : q.gaps.length ? (
          <div className="table-wrap"><table className="table">
            <thead><tr><th>问题</th><th>原因</th><th className="num">当时检索分</th><th>复测 / 之后的文档变更</th><th>时间</th>{actions ? <th /> : null}</tr></thead>
            <tbody>{q.gaps.map((g) => <GapRow key={g.questionId} g={g} onOpen={onOpen} actions={actions} />)}</tbody>
          </table></div>
        ) : <Empty icon={<Sparkles size={22} />} title="没有未解决的知识缺口" desc={q.resolvedCount ? `历史上的 ${q.resolvedCount} 个缺口复测后都已解决。` : '所有问题都检索到了资料、没有拒答或点踩。'} />}
        {q && q.misroutedGaps.length ? (
          <>
            <button className="list-item" style={{ width: '100%', borderTop: '1px solid var(--border)', borderRadius: 0 }} onClick={() => setShowMisrouted((v) => !v)}>
              <Badge tone="accent">答案在其它知识库 {q.misroutedCount}</Badge><span className="xs faint">提问时选错了库或路由不准：建议用自动路由、或完善知识库描述 · {showMisrouted ? '收起' : '展开'}</span>
            </button>
            {showMisrouted ? (
              <div className="table-wrap"><table className="table">
                <tbody>{q.misroutedGaps.map((g) => <GapRow key={g.questionId} g={g} onOpen={onOpen} resolved actions={actions} />)}</tbody>
              </table></div>
            ) : null}
          </>
        ) : null}
        {q && q.resolvedGaps.length ? (
          <>
            <button className="list-item" style={{ width: '100%', borderTop: '1px solid var(--border)', borderRadius: 0 }} onClick={() => setShowResolved((v) => !v)}>
              <Badge tone="ok">复测已解决 {q.resolvedCount}</Badge><span className="xs faint">{showResolved ? '收起' : '展开'}</span>
            </button>
            {showResolved ? (
              <div className="table-wrap"><table className="table">
                <tbody>{q.resolvedGaps.map((g) => <GapRow key={g.questionId} g={g} onOpen={onOpen} resolved actions={actions} />)}</tbody>
              </table></div>
            ) : null}
          </>
        ) : null}
        {q && q.ignoredGaps?.length ? (
          <>
            <button className="list-item" style={{ width: '100%', borderTop: '1px solid var(--border)', borderRadius: 0 }} onClick={() => setShowIgnored((v) => !v)}>
              <Badge>已忽略 {q.ignoredCount}</Badge><span className="xs faint">不计入缺口和缺口率，记录还在 · {showIgnored ? '收起' : '展开'}</span>
            </button>
            {showIgnored ? (
              <div className="table-wrap"><table className="table">
                <tbody>{q.ignoredGaps.map((g) => <GapRow key={g.questionId} g={g} onOpen={onOpen} resolved ignored actions={actions} />)}</tbody>
              </table></div>
            ) : null}
          </>
        ) : null}
      </div>

      {retest ? (
        <div className="card">
          <div className="card-header">
            <h3><RefreshCw size={15} />本次复测（{retest.tested} 个问题）</h3>
            <div className="row gap-6">
              {(Object.keys(RETEST_LABEL) as RetestStatus[]).filter((k) => retest.counts[k]).map((k) => <Badge key={k} tone={RETEST_TONE[k]}>{RETEST_LABEL[k]} {retest.counts[k]}</Badge>)}
              <span className="xs faint">{Math.round(retest.elapsedMs / 100) / 10} 秒{retest.checkAnswer ? ' · 已生成回答检查' : ' · 只检索'}</span>
            </div>
          </div>
          {retest.results.length ? (
            <div className="table-wrap"><table className="table">
              <thead><tr><th>问题</th><th>结果</th><th className="num">检索分</th><th>现在的回答 / 命中文档</th></tr></thead>
              <tbody>{retest.results.map((r) => (
                <tr key={r.questionId} style={{ cursor: 'pointer' }} onClick={() => onOpen(r.questionId)}>
                  <td style={{ maxWidth: 300 }}><div className="small strong">{r.question}</div><div className="xs faint">{r.type === 'weak' ? '低分问题' : '缺口'} · {r.reasons.join('、')}</div></td>
                  <td><Badge tone={RETEST_TONE[r.status]}>{r.statusLabel}</Badge>{r.refused ? <div className="xs text-warning mt-4">回答拒答</div> : null}</td>
                  <td className="num">{r.oldScore?.toFixed(3) ?? '—'} → {r.newScore?.toFixed(3) ?? '—'}</td>
                  <td style={{ maxWidth: 420 }}>
                    {r.error ? <span className="xs text-danger">{r.error}</span> : r.answer ? <div className="xs clamp-3">{r.answer}</div> : null}
                    {r.contexts.length ? <div className="xs faint mt-4">命中：{Array.from(new Set(r.contexts.map((c) => c.fileName ?? c.docId))).join('、')}</div> : <div className="xs faint mt-4">没有检索到资料</div>}
                    {r.altKb ? <div className="xs mt-4">「<b>{r.altKb.name}</b>」能回答（{r.altKb.fileName ?? '—'}，检索分 {r.altKb.score.toFixed(2)}）{r.altKb.answer ? `：${r.altKb.answer.slice(0, 80)}…` : ''}</div> : null}
                  </td>
                </tr>
              ))}</tbody>
            </table></div>
          ) : <div className="card-body xs faint">没有需要复测的问题。</div>}
        </div>
      ) : null}
    </div>
  );
}

function TopicsTab({ refv, branches, onOpen, onDeleteTopic }: { refv: LakeRef; branches: string[]; onOpen: (id: string) => void; onDeleteTopic?: (ids: string[], topic: string) => Promise<boolean> }) {
  const { toast } = useApp();
  const [k, setK] = useState<number | ''>('');
  const [write, setWrite] = useState(false);
  const [target, setTarget] = useState('');
  const [res, setRes] = useState<LakeTopics | null>(null);
  const [loading, setLoading] = useState(false);
  const run = async (opts: { noWrite?: boolean } = {}) => {
    const w = write && !opts.noWrite;          // 删除主题后重新聚类：只刷新结果，不重写派生列
    setLoading(true);
    try {
      const r = await lakeApi.topics({ k: k || null, writeColumn: w ? 'topic' : null, version: refv.version, branch: w ? (target || null) : refv.branch });
      setRes(r);
      if (r.written) toast(`已写入派生列 topic（${r.written.branch}，${r.written.filled} 行）`, 'ok');
    } catch (e) { toast(errMsg(e), 'bad'); } finally { setLoading(false); }
  };
  return (
    <div className="col gap-16">
      <div className="card"><div className="card-body row gap-12 wrap" style={{ alignItems: 'flex-end' }}>
        <div className="field" style={{ width: 140 }}><label>主题数 k</label><input type="number" min={1} max={20} value={k} placeholder="自动" onChange={(e) => setK(e.target.value ? Number(e.target.value) : '')} /></div>
        <label className="row gap-6 small"><input type="checkbox" checked={write} onChange={(e) => setWrite(e.target.checked)} style={{ width: 'auto' }} />把主题写成派生列 topic</label>
        {write ? (
          <div className="field" style={{ width: 180 }}><label>写到</label>
            <select value={target} onChange={(e) => setTarget(e.target.value)}><option value="">主表 main</option>{branches.map((b) => <option key={b} value={b}>分支 {b}</option>)}</select></div>
        ) : null}
        <Button variant="primary" loading={loading} onClick={() => run()}><Play size={13} />聚类</Button>
        <div className="xs faint flex-1">问题向量做余弦 k-means，主题词用 c-TF-IDF；缺口率 = 该主题里没答好的问题占比。建议先在分支上写列，验证后再合并回主表。</div>
      </div></div>
      {res ? (res.clusters.length ? (
        <div className="grid grid-2">
          {res.clusters.map((c) => (
            <div key={c.label} className="card">
              <div className="card-header"><h3>{c.label}</h3><div className="row gap-6"><Badge>{c.size} 个问题</Badge><Badge tone={c.gapRate >= 0.3 ? 'bad' : c.gapRate > 0 ? 'warn' : 'ok'}>缺口 {pctText(c.gapRate)}</Badge>{c.avgScore !== null ? <Badge mono>均分 {c.avgScore.toFixed(3)}</Badge> : null}
                {onDeleteTopic && c.questionIds?.length && !(refv.version || refv.branch) ? <Button size="sm" variant="ghost" title={`删除主题「${c.label}」下的全部 ${c.size} 个问题`} aria-label={`删除主题 ${c.label}`} onClick={() => onDeleteTopic(c.questionIds!, c.label).then((ok) => { if (ok) run({ noWrite: true }); })}><Trash2 size={13} /></Button> : null}</div></div>
              <div className="card-body col gap-6">
                <div className="row gap-4 wrap">{c.keywords.map((w) => <Badge key={w} tone="accent">{w}</Badge>)}</div>
                {c.samples.map((s) => <button key={s.id} className="list-item" style={{ width: '100%', textAlign: 'left' }} onClick={() => onOpen(s.id)}><span className="small ellipsis">{s.question}</span></button>)}
              </div>
            </div>
          ))}
        </div>
      ) : <Empty title="问题太少" desc={`至少需要 2 个问题，当前 ${res.questions} 个。`} />) : null}
    </div>
  );
}

function SearchTab({ refv, onOpen, onDelete }: { refv: LakeRef; onOpen: (id: string) => void; onDelete?: (id: string) => Promise<boolean> }) {
  const { toast } = useApp();
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'hybrid' | 'vector' | 'fts'>('hybrid');
  const [kinds, setKinds] = useState<LakeKind[]>([]);
  const [days, setDays] = useState('');
  const [where, setWhere] = useState('');
  const [hits, setHits] = useState<LakeHit[] | null>(null);
  const [loading, setLoading] = useState(false);
  const run = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try { setHits(await lakeApi.search({ query, mode, kinds: kinds.length ? kinds : null, topK: 20, days: days ? Number(days) : null, where: where.trim() || null, ...refv })); } catch (e) { toast(errMsg(e), 'bad'); } finally { setLoading(false); }
  };
  const toggle = (k: LakeKind) => setKinds((ks) => (ks.includes(k) ? ks.filter((x) => x !== k) : [...ks, k]));
  return (
    <div className="col gap-16">
      <div className="card"><div className="card-body col gap-12">
        <div className="row gap-8">
          <input value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && run()} placeholder="在湖里检索：历史问题、回答、检索日志、记忆、黑板、文档……" />
          <select value={mode} onChange={(e) => setMode(e.target.value as typeof mode)} style={{ width: 150 }}><option value="hybrid">混合（RRF）</option><option value="vector">仅向量</option><option value="fts">仅全文</option></select>
          <Button variant="primary" loading={loading} onClick={run}><Search size={13} />检索</Button>
        </div>
        <div className="row gap-6 wrap">{KINDS.map((k) => <button key={k} className={`badge ${kinds.includes(k) ? 'accent' : ''}`} style={{ cursor: 'pointer' }} onClick={() => toggle(k)}>{KIND_LABEL[k]}</button>)}<span className="xs faint">不选 = 全部类型</span></div>
        <div className="row gap-8 wrap">
          <label className="row gap-6 xs">最近<input type="number" min={1} value={days} placeholder="全部" style={{ width: 70 }} onChange={(e) => setDays(e.target.value)} />天</label>
          <input className="mono flex-1" value={where} placeholder="结构化过滤，例如 status = 'refused' AND feedback < 0（只认列的比较 / AND / OR / LIKE / IN）" style={{ minWidth: 320 }} onChange={(e) => setWhere(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && run()} />
        </div>
        <div className="xs faint">结构化条件在 LanceDB 里先过滤再算相似度（混合检索 = 结构化过滤 + 向量 / 全文）。智能问答里的「历史问答参考」就是用同一张表召回的：只取与当前问题相似度 ≥ 阈值、状态正常且没被点踩的历史回答，作为参考信息（不可引用）交给回答模型。</div>
      </div></div>
      {hits ? (hits.length ? (
        <div className="card">{hits.map((h) => (
          <RecordRow key={h.id} id={h.id} onOpen={onOpen} onDelete={onDelete && ((id) => onDelete(id).then((ok) => { if (ok) setHits((xs) => xs && xs.filter((x) => x.id !== id)); return ok; }))} top>
            <KindBadge kind={h.kind} />
            <div className="flex-1" style={{ minWidth: 0 }}><div className="small clamp-2">{h.title && h.kind !== 'question' ? <b>{h.title}　</b> : null}{h.content}</div><div className="xs faint">{fmtTs(h.ts)} · {h.session_id ? `会话 ${h.session_id.slice(0, 8)}` : h.agent}</div></div>
            <div className="col gap-4" style={{ alignItems: 'flex-end', flexShrink: 0 }}>
              <Badge mono>RRF {h.rrf.toFixed(4)}</Badge>
              <div className="row gap-4">{h.vectorScore !== null ? <Badge tone="dense" mono>向量 {h.vectorScore.toFixed(3)}</Badge> : null}{h.ftsScore !== null ? <Badge tone="sparse" mono>全文 {h.ftsScore.toFixed(2)}</Badge> : null}</div>
            </div>
          </RecordRow>
        ))}</div>
      ) : <Empty title="没有命中" />) : null}
    </div>
  );
}

/** 湖目录：按意图找相关的列 / meta 键 / 指标口径 / 示例 SQL。onPick：把一条 SQL 填进编辑器。 */
function CatalogPanel({ catalog, onPick, compact, onChanged }: { catalog: LakeCatalog | null; onPick?: (sql: string) => void; compact?: boolean; onChanged?: () => void }) {
  const { toast } = useApp();
  const [tab, setTab] = useState<'columns' | 'metaKeys' | 'metrics' | 'examples'>('columns');
  const [adding, setAdding] = useState(false);
  const [form, setForm] = useState<CatalogEntryInput>({ type: 'metric', name: '', sql: '', definition: '', unit: '', column: '', note: '' });
  const [saving, setSaving] = useState(false);
  const removeCustom = async (id: string, label: string) => {
    if (!confirmDialog(`删除目录条目「${label}」？`)) return;
    try { await lakeApi.deleteRecord(id, { cascade: false }); toast('已删除', 'ok'); onChanged?.(); } catch (e) { toast(errMsg(e), 'bad'); }
  };
  const save = async () => {
    setSaving(true);
    try { await lakeApi.catalogAdd(form); toast('已加入目录（带 SQL 的已在当前湖上试跑通过）', 'ok'); setAdding(false); setForm({ type: form.type, name: '', sql: '', definition: '', unit: '', column: '', note: '' }); onChanged?.(); } catch (e) { toast(errMsg(e), 'bad'); } finally { setSaving(false); }
  };
  if (!catalog) return <Spinner />;
  const customBadge = (id?: string, source?: string, label = '') => source === 'custom' && id && onChanged
    ? <span className="row gap-4"><Badge tone="accent">人工</Badge><button type="button" className="btn ghost xs" title="删除" onClick={() => removeCustom(id, label)}><X size={11} /></button></span>
    : null;
  const sections: { key: typeof tab; label: string; n: number }[] = [
    { key: 'columns', label: '列', n: catalog.columns.length }, { key: 'metaKeys', label: 'meta 键', n: catalog.metaKeys.length },
    { key: 'metrics', label: '指标口径', n: catalog.metrics.length }, { key: 'examples', label: '示例', n: catalog.examples.length },
  ];
  const pick = (sql: string) => onPick?.(sql);
  return (
    <div className="col gap-8">
      <div className="row gap-6 wrap">{sections.map((sec) => <button key={sec.key} className={`badge ${tab === sec.key ? 'accent' : ''}`} style={{ cursor: 'pointer' }} onClick={() => setTab(sec.key)}>{sec.label} {sec.n}</button>)}</div>
      {tab === 'columns' ? catalog.columns.map((c) => (
        <div key={c.name} className="col gap-2 xs" style={{ borderBottom: '1px solid var(--border)', paddingBottom: 6 }}>
          <div className="row between"><span className="mono">{c.name}</span><span className="faint">{c.type}</span></div>
          <div className="muted">{c.description}</div>
          {c.samples.length ? <div className="faint clamp-2">取值：{c.samples.join(' / ')}</div> : null}
          {c.range ? <div className="faint">范围：{c.range}</div> : null}
          {c.notes?.length && onChanged ? c.notes.map((n, i) => { const rec = catalog.custom?.find((x) => x.type === 'column' && x.meta.column === c.name && x.content === n); return <div key={i} className="row between gap-6"><span className="muted">备注：{n}</span>{rec ? <button type="button" className="btn ghost xs" title="删除备注" onClick={() => removeCustom(rec.id, `${c.name} 的备注`)}><X size={11} /></button> : null}</div>; }) : null}
        </div>
      )) : null}
      {onChanged ? (adding ? (
        <div className="col gap-6 xs" style={{ borderTop: '1px solid var(--border)', paddingTop: 8 }}>
          <select value={form.type} onChange={(e) => setForm({ ...form, type: e.target.value as CatalogEntryInput['type'] })}><option value="metric">指标口径</option><option value="example">问题 → SQL 示例</option><option value="column">列备注</option></select>
          {form.type === 'column' ? (
            <>
              <select value={form.column} onChange={(e) => setForm({ ...form, column: e.target.value })}><option value="">选择列…</option>{catalog.columns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}</select>
              <textarea rows={2} value={form.note} placeholder="这列的业务含义、取值注意事项…" onChange={(e) => setForm({ ...form, note: e.target.value })} />
            </>
          ) : (
            <>
              <input value={form.name} placeholder={form.type === 'metric' ? '指标名，例如：年假问题占比' : '问题，例如：被点踩的回答有哪些'} onChange={(e) => setForm({ ...form, name: e.target.value })} />
              {form.type === 'metric' ? <div className="row gap-6"><input className="flex-1" value={form.definition} placeholder="口径说明（怎么算）" onChange={(e) => setForm({ ...form, definition: e.target.value })} /><input style={{ width: 60 }} value={form.unit} placeholder="单位" onChange={(e) => setForm({ ...form, unit: e.target.value })} /></div> : null}
              <textarea className="mono" rows={3} value={form.sql} placeholder="SELECT … FROM lake WHERE kind = '…'" spellCheck={false} onChange={(e) => setForm({ ...form, sql: e.target.value })} />
            </>
          )}
          <div className="row gap-6"><Button size="xs" variant="primary" loading={saving} onClick={save}>保存</Button><Button size="xs" onClick={() => setAdding(false)}>取消</Button><span className="faint">带 SQL 的条目保存前会在当前湖上试跑一遍</span></div>
        </div>
      ) : <div className="row between xs" style={{ borderTop: '1px solid var(--border)', paddingTop: 8 }}><span className="faint">人工条目存在湖里，跟着版本走，分析智能体优先采用</span><Button size="xs" onClick={() => setAdding(true)}>添加</Button></div>) : null}
      {tab === 'metaKeys' ? (catalog.metaKeys.length ? catalog.metaKeys.map((m) => (
        <div key={`${m.recordKind}:${m.name}`} className="col gap-2 xs" style={{ borderBottom: '1px solid var(--border)', paddingBottom: 6 }}>
          <div className="row between gap-6"><span className="mono ellipsis">{m.name}</span><Badge tone="info">{KIND_LABEL[m.recordKind as LakeKind] ?? m.recordKind}</Badge></div>
          {m.samples.length ? <div className="faint clamp-2">示例：{m.samples.join('、')}</div> : null}
          <div className="mono faint clamp-2" title={m.access}>{m.access}</div>
        </div>
      )) : <Empty title="还没有 meta 数据" />) : null}
      {tab === 'metrics' ? catalog.metrics.map((m) => (
        <div key={m.name} className="col gap-2 xs" style={{ borderBottom: '1px solid var(--border)', paddingBottom: 6 }}>
          <div className="row between gap-6"><b>{m.name}{m.unit ? `（${m.unit}）` : ''}</b><span className="row gap-4">{customBadge(m.id, m.source, m.name)}{onPick ? <button className="btn ghost xs" onClick={() => pick(m.sql)}>填入 SQL</button> : null}</span></div>
          <div className="muted">{m.definition}</div>
          {!compact ? <pre className="mono faint pre" style={{ margin: 0, maxHeight: 90, overflow: 'auto' }}>{m.sql}</pre> : null}
        </div>
      )) : null}
      {tab === 'examples' ? catalog.examples.map((e, i) => (
        <div key={i} className="col gap-2 xs" style={{ borderBottom: '1px solid var(--border)', paddingBottom: 6 }}>
          <div className="row between gap-6"><span className="ellipsis">{e.name}</span><span className="row gap-4">{e.source === 'history' ? <Badge tone="ok">本库历史</Badge> : e.source === 'custom' ? customBadge(e.id, e.source, e.name) : <Badge>内置</Badge>}{onPick ? <button className="btn ghost xs" onClick={() => pick(e.sql)}>填入</button> : null}</span></div>
          {!compact ? <pre className="mono faint pre" style={{ margin: 0, maxHeight: 90, overflow: 'auto' }}>{e.sql}</pre> : null}
        </div>
      )) : null}
    </div>
  );
}

function SqlTab({ refv }: { refv: LakeRef }) {
  const [sql, setSql] = useState(SQL_SAMPLES[0].sql);
  const [res, setRes] = useState<SqlResult | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [intent, setIntent] = useState('');
  const [catalog, setCatalog] = useState<LakeCatalog | null>(null);
  // 湖目录：不填意图时列出每类的前几条；填了按相关度挑（含湖里以前分析成功的 SQL）
  useEffect(() => {
    const t = window.setTimeout(() => { lakeApi.catalog(intent, refv).then(setCatalog).catch(() => setCatalog(null)); }, intent ? 300 : 0);
    return () => window.clearTimeout(t);
  }, [refv, intent]);
  const run = async () => {
    setLoading(true); setError('');
    try { setRes(await lakeApi.sql(sql, refv)); } catch (e) { setError(errMsg(e)); setRes(null); } finally { setLoading(false); }
  };
  return (
    <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) 320px', gap: 16 }}>
      <div className="col gap-12">
        <div className="card"><div className="card-body col gap-8">
          <div className="row gap-6 wrap">{SQL_SAMPLES.map((s) => <button key={s.label} className="badge" style={{ cursor: 'pointer' }} onClick={() => setSql(s.sql)}>{s.label}</button>)}</div>
          <textarea className="mono" rows={7} value={sql} onChange={(e) => setSql(e.target.value)} onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') run(); }} spellCheck={false} />
          <div className="row between"><span className="xs faint">DuckDB 只读查询：表名 lake；禁止写操作与读取本机文件。⌘/Ctrl + Enter 执行</span><Button variant="primary" loading={loading} onClick={run}><Terminal size={13} />执行</Button></div>
        </div></div>
        {error ? <Callout tone="bad">{error}</Callout> : null}
        {res ? <div className="card"><div className="card-header"><h3>结果</h3><span className="xs faint">{res.rowCount} 行{res.truncated ? '（已截断）' : ''} · {res.elapsedMs} ms</span></div><SqlResultView columns={res.columns} rows={res.rows} /></div> : null}
      </div>
      <div className="card" style={{ alignSelf: 'flex-start' }}><div className="card-header"><h3><BookOpen size={15} />湖目录</h3><span className="xs faint">分析智能体写 SQL 前先查这里</span></div>
        <div className="card-body col gap-8">
          <input value={intent} onChange={(e) => setIntent(e.target.value)} placeholder="按意图找列 / 口径，例如：延迟、拒答、策略" />
          <CatalogPanel catalog={catalog} onPick={setSql} onChanged={() => lakeApi.catalog(intent, refv).then(setCatalog).catch(() => undefined)} />
        </div></div>
    </div>
  );
}

function AgentTab({ refv }: { refv: LakeRef }) {
  const [question, setQuestion] = useState('');
  const [res, setRes] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState<number | null>(null);
  const run = async (q = question) => {
    if (!q.trim()) return;
    setQuestion(q); setLoading(true); setError(''); setRes(null);
    try { setRes(await lakeApi.analyze(q, refv)); } catch (e) { setError(errMsg(e)); } finally { setLoading(false); }
  };
  return (
    <div className="col gap-16">
      <div className="card"><div className="card-body col gap-8">
        <div className="row gap-8">
          <input value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && run()} placeholder="用自然语言问湖里的事，例如：最近一周哪些问题没答上来？" />
          <Button variant="primary" loading={loading} onClick={() => run()}><Bot size={13} />分析</Button>
        </div>
        <div className="row gap-6 wrap">{AGENT_SAMPLES.map((s) => <button key={s} className="badge" style={{ cursor: 'pointer' }} onClick={() => run(s)}>{s}</button>)}</div>
        <div className="xs faint">智能体先在湖目录里找相关的列 / meta 键 / 指标口径 / 示例 SQL，按决策罗盘规划工具调用（目录 / 概况 / 检索质量 / 缺口 / 复测 / 路由 / 主题 / SQL / 检索），执行后看结果反思、不够再查，最后写结论。每次分析和工具调用也写回湖里，成为下次的示例。</div>
      </div></div>
      {loading ? <Spinner label="规划 → 调用工具 → 汇总结论" /> : null}
      {error ? <Callout tone="bad">{error}</Callout> : null}
      {res ? (
        <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: 16 }}>
          <div className="col gap-16">
            <div className="card"><div className="card-header"><h3><Sparkles size={15} />结论</h3><span className="xs faint">规划：{res.planner === 'llm' ? '模型' : '规则'} · 汇总：{res.summarizer === 'llm' ? '模型' : '模板'}{res.rounds && res.rounds > 1 ? ` · ${res.rounds} 轮` : ''} · {Math.round(res.elapsedMs)} ms</span></div>
              <div className="card-body"><Markdown text={res.answer} /></div></div>
            {res.catalog ? <div className="card"><div className="card-header"><h3><BookOpen size={15} />规划时参考的目录</h3><span className="xs faint">按问题挑出的列 / meta 键 / 口径 / 示例</span></div>
              <div className="card-body"><CatalogPanel catalog={res.catalog} compact /></div></div> : null}
          </div>
          <div className="card"><div className="card-header"><h3>工具调用（{res.steps.length}）</h3></div>
            {res.steps.map((s, i) => (
              <div key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                <button className="list-item" style={{ width: '100%', textAlign: 'left' }} onClick={() => setOpen(open === i ? null : i)}>
                  <Badge tone={s.error ? 'bad' : s.empty ? 'warn' : 'info'} mono>{i + 1}. {s.tool}</Badge><span className="small ellipsis flex-1">{s.why}</span><span className="xs faint">{s.empty ? '结果为空 · ' : typeof s.rows === 'number' ? `${s.rows} 行 · ` : ''}{s.ms} ms</span>
                </button>
                {open === i ? (
                  <div className="card-body col gap-8">
                    <pre className="xs mono pre" style={{ margin: 0 }}>{JSON.stringify(s.args, null, 2)}</pre>
                    {s.error ? <Callout tone="bad">{s.error}</Callout> : s.tool === 'sql' && s.result && typeof s.result === 'object' && 'columns' in (s.result as object)
                      ? <SqlResultView columns={(s.result as SqlResult).columns} rows={(s.result as SqlResult).rows} />
                      : <pre className="xs mono pre" style={{ margin: 0, maxHeight: 300, overflow: 'auto' }}>{JSON.stringify(s.result, null, 2)}</pre>}
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function VersionsTab({ data, reload, onDelete }: { data: LakeVersions | null; reload: () => void; onDelete: (ref: string) => Promise<boolean> }) {
  const { toast } = useApp();
  const [tagName, setTagName] = useState('');
  const [branchName, setBranchName] = useState('');
  const [branchFrom, setBranchFrom] = useState('');
  const [diff, setDiff] = useState<BranchDiff | null>(null);
  const [col, setCol] = useState({ name: 'confidence', expr: "CASE WHEN kind <> 'retrieval' THEN NULL WHEN score < 0.05 THEN 'low' WHEN score < 0.2 THEN 'medium' ELSE 'high' END", branch: '' });
  const act = async (fn: () => Promise<unknown>, ok: string) => {
    try { await fn(); toast(ok, 'ok'); reload(); } catch (e) { toast(errMsg(e), 'bad'); }
  };
  // 派生列：点 × 直接删除（删列是一次新的写入，之前的版本里这一列还在，可以时间旅行回去看）
  const dropCol = (name: string, branch: string | null) => {
    const where = branch ? `分支 ${branch}` : '主表';
    if (!confirmDialog(`删除${where} 上的派生列 ${name}？\n之前的版本里这一列还在（可切换到旧版本查看），当前${where} 不再有这一列。`)) return;
    act(() => lakeApi.dropColumn(name, branch), `已删除${where} 的派生列 ${name}`);
  };
  const colChips = (cols: string[], branch: string | null) => cols.map((c) => (
    <span key={c} className="col-chip mono">{c}<button type="button" title={`删除派生列 ${c}`} aria-label={`删除派生列 ${c}`} onClick={() => dropCol(c, branch)}><X size={11} /></button></span>
  ));
  if (!data) return <Spinner />;
  return (
    <div className="col gap-16">
      <Callout icon={<GitBranch size={15} />}>
        对应 Lance 的数据处理时间线：<b>主表</b>持续写入 → 打 <b>Tag</b> 冻结一个快照 → 从 Tag 拉<b>分支</b>，在分支上追加派生列（主题、打分、意图……，按主键附加新数据文件，不重写历史）→ 对比验证 → <b>合并</b>回主表。页面顶部可以切换到任意版本 / Tag / 分支做分析。
      </Callout>
      <div className="grid grid-2">
        <div className="card">
          <div className="card-header"><h3><GitBranch size={15} />分支</h3></div>
          <div className="card-body row gap-8">
            <input value={branchName} onChange={(e) => setBranchName(e.target.value)} placeholder="分支名，如 topic-exp" />
            <select value={branchFrom} onChange={(e) => setBranchFrom(e.target.value)} style={{ width: 170 }}>
              <option value="">从主表最新版本</option>{data.tags.map((t) => <option key={t.name} value={`tag:${t.name}`}>从 Tag {t.name}</option>)}
            </select>
            <Button disabled={!branchName.trim()} onClick={() => act(() => lakeApi.createBranch(branchName.trim(), branchFrom.startsWith('tag:') ? { fromTag: branchFrom.slice(4) } : {}), `已创建分支 ${branchName}`).then(() => setBranchName(''))}>创建</Button>
          </div>
          {data.branches.length ? data.branches.map((b) => (
            <div key={b.name} className="card-body row between wrap gap-8" style={{ borderTop: '1px solid var(--border)' }}>
              <div className="col gap-4"><b className="small">{b.name}</b><span className="xs faint">从 {b.parentBranch} v{b.parentVersion} 拉出 · 当前 v{b.version} · {b.rows} 行</span>{b.derivedColumns.length ? <div className="row gap-4 wrap"><span className="xs faint">派生列</span>{colChips(b.derivedColumns, b.name)}</div> : null}</div>
              <div className="row gap-6">
                <Button size="sm" onClick={() => lakeApi.diffBranch(b.name).then(setDiff).catch((e) => toast(errMsg(e), 'bad'))}>对比</Button>
                <Button size="sm" disabled={!b.derivedColumns.length} onClick={() => confirmDialog(`把分支 ${b.name} 的派生列（${b.derivedColumns.join('、')}）合并回主表？`) && act(() => lakeApi.promoteBranch(b.name), '已合并回主表')}><GitMerge size={13} />合并</Button>
                <Button size="sm" variant="ghost" title={`删除分支 ${b.name}`} aria-label={`删除分支 ${b.name}`} onClick={() => onDelete(`branch:${b.name}`).then((ok) => { if (ok && diff?.branch === b.name) setDiff(null); })}><Trash2 size={13} /></Button>
              </div>
            </div>
          )) : <div className="card-body xs faint" style={{ borderTop: '1px solid var(--border)' }}>还没有分支</div>}
          {diff ? (
            <div className="card-body col gap-6" style={{ borderTop: '1px solid var(--border)' }}>
              <div className="section-label">分支 {diff.branch} vs 主表</div>
              <div className="small">新增列：{diff.addedColumns.join('、') || '无'} · 删除列：{diff.removedColumns.join('、') || '无'} · 只在分支：{diff.rowsOnlyInBranch} 行 · 只在主表：{diff.rowsOnlyInMain} 行</div>
              {Object.entries(diff.columnValues).map(([c, vals]) => <div key={c} className="row gap-4 wrap"><span className="xs mono">{c}</span>{vals.map((v) => <Badge key={v.value}>{v.value} × {v.count}</Badge>)}</div>)}
            </div>
          ) : null}
        </div>
        <div className="card">
          <div className="card-header"><h3><Tag size={15} />Tag</h3></div>
          <div className="card-body row gap-8">
            <input value={tagName} onChange={(e) => setTagName(e.target.value)} placeholder="Tag 名，如 baseline-0930" />
            <Button disabled={!tagName.trim()} onClick={() => act(() => lakeApi.createTag(tagName.trim()), `已为当前版本打 Tag ${tagName}`).then(() => setTagName(''))}>给当前版本打 Tag</Button>
          </div>
          {data.tags.map((t) => (
            <div key={t.name} className="card-body row between" style={{ borderTop: '1px solid var(--border)' }}>
              <span className="small"><Badge tone="accent">{t.name}</Badge> → v{t.version}</span>
              <Button size="sm" variant="ghost" title={`删除 Tag ${t.name}`} aria-label={`删除 Tag ${t.name}`} onClick={() => onDelete(`tag:${t.name}`)}><Trash2 size={13} /></Button>
            </div>
          ))}
          <div className="card-header" style={{ borderTop: '1px solid var(--border)' }}><h3><Layers size={15} />追加派生列</h3></div>
          <div className="card-body col gap-8">
            <div className="row gap-4 wrap"><span className="xs faint">主表现有派生列</span>{data.derivedColumns?.length ? colChips(data.derivedColumns, null) : <span className="xs faint">无</span>}</div>
            <div className="row gap-8">
              <input value={col.name} onChange={(e) => setCol({ ...col, name: e.target.value })} placeholder="列名（snake_case）" style={{ width: 160 }} />
              <select value={col.branch} onChange={(e) => setCol({ ...col, branch: e.target.value })}><option value="">写到主表 main</option>{data.branches.map((b) => <option key={b.name} value={b.name}>写到分支 {b.name}</option>)}</select>
            </div>
            <textarea className="mono" rows={3} value={col.expr} onChange={(e) => setCol({ ...col, expr: e.target.value })} spellCheck={false} />
            <div className="row between"><span className="xs faint">SQL 表达式（DuckDB），按已有列计算</span><Button size="sm" onClick={() => act(() => lakeApi.addColumn({ name: col.name, expr: col.expr, branch: col.branch || null }), `已追加列 ${col.name}`)}>追加</Button></div>
          </div>
        </div>
      </div>
      <div className="card"><div className="card-header"><h3><History size={15} />主表版本（{data.versions.length}）</h3><span className="xs faint">每次写入都是一个版本，可在顶部切换查看；删除某个版本会连同更早的版本一起清理</span></div>
        <div className="table-wrap" style={{ maxHeight: 360, overflow: 'auto' }}><table className="table">
          <thead><tr><th>版本</th><th>时间</th><th className="num">行数</th><th className="num">分片</th><th>Tag</th><th /></tr></thead>
          <tbody>{data.versions.map((v, i) => <tr key={v.version}><td className="mono">v{v.version}</td><td className="xs">{v.timestamp ? new Date(v.timestamp).toLocaleString('zh-CN', { hour12: false }) : '—'}</td><td className="num">{v.metadata.total_rows ?? '—'}</td><td className="num">{v.metadata.total_fragments ?? '—'}</td><td>{v.tags.map((t) => <Badge key={t} tone="accent">{t}</Badge>)}</td><td className="num">{i > 0 ? <Button size="sm" variant="ghost" title={`删除 v${v.version} 及更早的版本`} aria-label={`删除 v${v.version} 及更早的版本`} onClick={() => onDelete(`v:${v.version}`)}><Trash2 size={13} /></Button> : <span className="xs faint">最新</span>}</td></tr>)}</tbody>
        </table></div></div>
    </div>
  );
}

// ------------------------------------------------------------------ 快照选择（原生 select 放不了删除按钮，自己画一个）
function refLabel(value: string): string {
  if (value.startsWith('branch:')) return `分支 ${value.slice(7)}`;
  if (value.startsWith('tag:')) return `Tag ${value.slice(4)}`;
  if (value.startsWith('v:')) return `v${value.slice(2)}`;
  return '主表 · 最新';
}

function RefPicker({ value, onChange, versions, onDelete }: { value: string; onChange: (v: string) => void; versions: LakeVersions | null; onDelete: (ref: string) => void }) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => { if (!wrap.current?.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', away);
    document.addEventListener('keydown', esc);
    return () => { document.removeEventListener('mousedown', away); document.removeEventListener('keydown', esc); };
  }, [open]);
  const pick = (v: string) => { onChange(v); setOpen(false); };
  const item = (v: string, label: React.ReactNode, extra?: React.ReactNode) => (
    <div key={v || 'latest'} role="option" aria-selected={value === v} tabIndex={0} className={`ref-item${value === v ? ' on' : ''}`} onClick={() => pick(v)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(v); } }}>
      <span className="ref-check">{value === v ? <Check size={13} /> : null}</span>
      <span className="ref-label">{label}</span>
      {extra}
    </div>
  );
  const del = (ref: string, title: string) => (
    <button type="button" className="ref-del" title={title} aria-label={title} onClick={(e) => { e.stopPropagation(); setOpen(false); onDelete(ref); }} onKeyDown={(e) => e.stopPropagation()}><Trash2 size={13} /></button>
  );
  const time = (ts: string | null) => (ts ? new Date(ts).toLocaleString('zh-CN', { hour12: false, month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '');
  return (
    <div className="ref-picker" ref={wrap}>
      <button type="button" className="ref-trigger" aria-haspopup="listbox" aria-expanded={open} title="在哪个快照上分析" onClick={() => setOpen((o) => !o)}>
        <span className="ref-label">{refLabel(value)}</span><ChevronDown size={13} className="faint" />
      </button>
      {open ? (
        <div className="ref-menu" role="listbox">
          {item('', '主表 · 最新')}
          {versions?.branches.length ? <><div className="ref-group">分支</div>{versions.branches.map((b) => item(`branch:${b.name}`, `分支 ${b.name}`, del(`branch:${b.name}`, `删除分支 ${b.name}`)))}</> : null}
          {versions?.tags.length ? <><div className="ref-group">Tag</div>{versions.tags.map((t) => item(`tag:${t.name}`, `Tag ${t.name}（v${t.version}）`, del(`tag:${t.name}`, `删除 Tag ${t.name}`)))}</> : null}
          {versions && versions.versions.length > 1 ? <><div className="ref-group">历史版本<span className="ref-hint">删除会连同更早的版本</span></div>{versions.versions.slice(1, 30).map((v) => item(`v:${v.version}`, `v${v.version}`, <>
            <span className="ref-meta">{time(v.timestamp)}</span>
            {del(`v:${v.version}`, `删除 v${v.version} 及更早的版本`)}
          </>))}</> : null}
        </div>
      ) : null}
    </div>
  );
}

// ------------------------------------------------------------------ 页面
export function LakePage() {
  const { toast } = useApp();
  const [tab, setTab] = useState<TabKey>('overview');
  const [refSel, setRefSel] = useState('');           // '' 最新 / 'v:12' / 'tag:x' / 'branch:y'
  const [versions, setVersions] = useState<LakeVersions | null>(null);
  const [drawer, setDrawer] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const loadVersions = useCallback(() => { lakeApi.versions().then(setVersions).catch(() => undefined); }, []);
  useEffect(loadVersions, [loadVersions]);

  // 删除一条记录（连同关联的检索日志 / 回答等后代记录）。先预览一下会删几条，确认框里说清楚。
  // what：确认框里怎么称呼要删的东西（主题 / 缺口问题），默认"这条记录"
  const deleteRecords = async (ids: string[], what?: string): Promise<boolean> => {
    try {
      const preview = await lakeApi.deleteRecords(ids, { dryRun: true });
      const extra = preview.deleted.length - ids.length;
      what = what ?? (ids.length > 1 ? `这 ${ids.length} 条记录` : '这条记录');
      if (!confirmDialog(`删除${what}${extra > 0 ? `，连同 ${extra} 条关联记录（检索日志 / 回答等）` : ''}？\n删除后「回填历史数据」不会再把它写回来；之前的版本里还能看到。`)) return false;
      const r = await lakeApi.deleteRecords(ids);
      toast(`已删除 ${r.deleted.length} 条记录`, 'ok');
      if (drawer && r.deleted.includes(drawer)) setDrawer(null);
      setReloadKey((k) => k + 1); loadVersions();
      return true;
    } catch (e) { toast(errMsg(e), 'bad'); return false; }
  };
  const deleteRecord = (id: string) => deleteRecords([id]);

  // 删除分支 / Tag / 历史版本（ref 同 refSel 的格式）。历史版本会连同更早的版本一起清理（Lance 只能按时间整段清理）。
  // 正在查看的快照被删掉时切回主表最新。返回是否删除成功。
  const deleteRef = async (ref: string): Promise<boolean> => {
    let message: string;
    let run: () => Promise<string>;
    if (ref.startsWith('branch:')) {
      const name = ref.slice(7);
      const b = versions?.branches.find((x) => x.name === name);
      message = `删除分支 ${name}？${b?.derivedColumns.length ? `\n分支上的派生列（${b.derivedColumns.join('、')}）没合并回主表的部分会一起丢失。` : ''}\n不可恢复。`;
      run = () => lakeApi.deleteBranch(name).then(() => `已删除分支 ${name}`);
    } else if (ref.startsWith('tag:')) {
      const name = ref.slice(4);
      message = `删除 Tag ${name}？\n只删除这个标记，对应版本的数据还在（之后可以被清理）。`;
      run = () => lakeApi.deleteTag(name).then(() => `已删除 Tag ${name}`);
    } else {
      const v = Number(ref.slice(2));
      const older = (versions?.versions ?? []).filter((x) => x.version <= v).length;
      message = `删除 v${v} 及更早的所有历史版本？（当前列表里有 ${older} 个）\n当前数据不受影响，但这些版本之后无法再查看，且不可恢复。`;
      run = () => lakeApi.deleteVersions(v).then((r) => {
        if (refSel.startsWith('v:') && r.deleted.includes(Number(refSel.slice(2)))) setRefSel('');
        return `已删除 ${r.deleted.length} 个历史版本${r.kept.length ? `，保留分支起点 ${r.kept.map((x) => `v${x}`).join('、')}` : ''}`;
      });
    }
    if (!confirmDialog(message)) return false;
    try {
      toast(await run(), 'ok');
      if (refSel === ref) setRefSel('');
      return true;
    } catch (e) {
      toast(errMsg(e), 'bad');
      return false;
    } finally {
      loadVersions();
    }
  };

  const refv: LakeRef = useMemo(() => {
    if (refSel.startsWith('branch:')) return { branch: refSel.slice(7) };
    if (refSel.startsWith('tag:')) return { version: refSel.slice(4) };
    if (refSel.startsWith('v:')) return { version: refSel.slice(2) };
    return {};
  }, [refSel]);
  const isSnapshotRef = !!refSel;                     // 看旧版本 / Tag / 分支时，列表里的记录是快照，不提供删除

  const sync = async () => {
    setSyncing(true);
    try {
      const r = await lakeApi.sync();
      const total = Object.values(r).reduce((n, x) => n + (x.inserted ?? 0), 0);
      toast(`回填完成：新增 ${total} 条（${Object.entries(r).map(([k, v]) => `${k} ${v.inserted}`).join(' · ')}）`, 'ok');
      setReloadKey((k) => k + 1); loadVersions();
    } catch (e) { toast(errMsg(e), 'bad'); } finally { setSyncing(false); }
  };

  return (
    <div className="page col gap-16">
      <div className="row between wrap gap-8">
        <Tabs value={tab} onChange={setTab} items={[
          { value: 'overview', label: '概览', icon: <Waves size={13} /> },
          { value: 'gaps', label: '知识缺口', icon: <TriangleAlert size={13} /> },
          { value: 'topics', label: '问题主题', icon: <Layers size={13} /> },
          { value: 'search', label: '湖检索', icon: <Search size={13} /> },
          { value: 'sql', label: 'SQL', icon: <Terminal size={13} /> },
          { value: 'agent', label: '分析智能体', icon: <Bot size={13} /> },
          { value: 'versions', label: '版本与分支', icon: <GitBranch size={13} /> },
        ]} />
        <div className="row gap-8">
          <RefPicker value={refSel} onChange={setRefSel} versions={versions} onDelete={deleteRef} />
          <Button size="sm" loading={syncing} onClick={sync} title="把历史对话、知识库文档、长期记忆、智能体黑板回填进湖（按 id 去重，可重复执行）"><Database size={13} />回填历史数据</Button>
          <Button size="sm" variant="ghost" onClick={() => { setReloadKey((k) => k + 1); loadVersions(); }} title="刷新"><RefreshCw size={13} /></Button>
        </div>
      </div>
      {refSel ? <Callout icon={<History size={15} />}>正在查看 <b>{refSel.replace('branch:', '分支 ').replace('tag:', 'Tag ').replace('v:', '版本 v')}</b> 的快照，分析结果不包含之后写入的数据。</Callout> : null}

      {tab === 'overview' && <OverviewTab refv={refv} onOpen={setDrawer} onDelete={isSnapshotRef ? undefined : deleteRecord} onDeleteMany={isSnapshotRef ? undefined : deleteRecords} reloadKey={reloadKey} />}
      {tab === 'gaps' && <GapsTab refv={refv} onOpen={setDrawer} go={setTab} onDelete={isSnapshotRef ? undefined : (id, question) => deleteRecords([id], `问题「${question.slice(0, 40)}」`)} />}
      {tab === 'topics' && <TopicsTab refv={refv} branches={versions?.branches.map((b) => b.name) ?? []} onOpen={setDrawer} onDeleteTopic={isSnapshotRef ? undefined : (ids, topic) => deleteRecords(ids, `主题「${topic}」下的 ${ids.length} 个问题`)} />}
      {tab === 'search' && <SearchTab refv={refv} onOpen={setDrawer} onDelete={isSnapshotRef ? undefined : deleteRecord} />}
      {tab === 'sql' && <SqlTab refv={refv} />}
      {tab === 'agent' && <AgentTab refv={refv} />}
      {tab === 'versions' && <VersionsTab data={versions} reload={loadVersions} onDelete={deleteRef} />}
      {drawer ? <RecordDrawer id={drawer} onClose={() => setDrawer(null)} onOpen={setDrawer} onDelete={(id, what) => deleteRecords([id], what)} /> : null}
    </div>
  );
}
