// 多智能体协作：规划者 → 研究员（并行检索，证据贴 LanceDB 黑板）→ 分析师 → 审校（证据不足改写重查）→ 撰稿人，实时时间线。
import { useCallback, useEffect, useRef, useState } from 'react';
import { BookOpenCheck, CheckSquare, ClipboardList, History, Microscope, PenLine, Play, Search, ShieldCheck, Square, Trash2, Users } from 'lucide-react';
import { ApiError } from '../api/client';
import { agentsApi, fmtTime, type AgentEvent, type AgentEvidence, type AgentRunSummary, type BoardRow } from '../api/lance';
import { KbSelect } from '../components/kb-select';
import { Markdown } from '../components/markdown';
import { Badge, Button, Callout, Empty, Spinner, confirmDialog } from '../components/ui';
import { useApp } from '../store/app-store';

const AGENTS: Record<string, { name: string; icon: React.ReactNode; tone: string; desc: string }> = {
  planner: { name: '规划者', icon: <ClipboardList size={14} />, tone: 'info', desc: '拆解问题、分派子任务' },
  researcher: { name: '研究员', icon: <Search size={14} />, tone: 'dense', desc: '并行检索，把证据贴上黑板' },
  analyst: { name: '分析师', icon: <Microscope size={14} />, tone: 'fused', desc: '按子任务从黑板取回证据并整理' },
  reviewer: { name: '审校', icon: <ShieldCheck size={14} />, tone: 'warn', desc: '核对证据，不够就改写重查' },
  writer: { name: '撰稿人', icon: <PenLine size={14} />, tone: 'ok', desc: '带引用写出最终回答' },
  system: { name: '系统', icon: <Users size={14} />, tone: '', desc: '' },
  user: { name: '用户', icon: <Users size={14} />, tone: '', desc: '' },
};
const TYPE_LABEL: Record<string, string> = { run: '开始', memory: '用户记忆', plan: '任务拆解', evidence: '检索证据', recall: '黑板取证', analysis: '分析', review: '审校意见', requery: '改写重查', final: '最终回答', error: '出错' };
const errText = (e: unknown) => (e instanceof ApiError ? e.message : String(e));

export function AgentsPage() {
  const { currentKb, identity, toast } = useApp();
  const [question, setQuestion] = useState('远程员工的通讯补贴和本地通勤员工的交通补贴有什么不同？分别怎么发放？');
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [runs, setRuns] = useState<AgentRunSummary[]>([]);
  const [replay, setReplay] = useState<{ run: AgentRunSummary; rows: BoardRow[] } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  const loadRuns = useCallback(() => { agentsApi.runs().then(setRuns).catch(() => undefined); }, []);
  useEffect(() => { loadRuns(); }, [loadRuns]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [events.length]);

  const run = async () => {
    if (!currentKb || !question.trim()) return;
    setEvents([]);
    setReplay(null);
    setRunning(true);
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      await agentsApi.run({ kbId: currentKb.id, question, userId: identity.userId || 'default' }, (e) => setEvents((xs) => [...xs, e]), ac.signal);
    } catch (e) {
      if (!ac.signal.aborted) toast(`协作失败：${errText(e)}`, 'bad');
    } finally {
      setRunning(false);
      loadRuns();
    }
  };
  const stop = () => abortRef.current?.abort();
  const openRun = async (r: AgentRunSummary) => {
    try {
      setReplay({ run: r, rows: await agentsApi.history(r.runId) });
      setEvents([]);
    } catch (e) {
      toast(errText(e), 'bad');
    }
  };

  const removeRuns = async (ids: string[] | 'all') => {
    const all = ids === 'all';
    if (!all && !ids.length) return false;
    if (!confirmDialog(all ? '清空全部运行记录？黑板上所有运行的问题、证据、分析与回答都会删除，不可恢复。' : `删除 ${ids.length} 条运行记录？对应的黑板记录会一并删除，不可恢复。`)) return false;
    try {
      if (all) await agentsApi.clearRuns();
      else if (ids.length === 1) await agentsApi.deleteRun(ids[0]);
      else await agentsApi.deleteRuns(ids);
      if (replay && (all || ids.includes(replay.run.runId))) setReplay(null);
      toast(all ? '已清空运行记录' : `已删除 ${ids.length} 条运行记录`, 'ok');
      return true;
    } catch (e) {
      toast(`删除失败：${errText(e)}`, 'bad');
      return false;
    } finally {
      loadRuns();
    }
  };

  const final = events.find((e) => e.type === 'final');
  const sources: AgentEvidence[] = final?.data.sources ?? [];
  return (
    <div className="page col gap-16 fade-in">
      <div className="card">
        <div className="card-body col gap-12">
          <div className="row gap-8 wrap">
            <KbSelect />
            <input className="flex-1" style={{ minWidth: 280 }} value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && !running && run()} placeholder="输入一个需要多步检索、对比或核对的问题" />
            {running ? <Button variant="danger" onClick={stop}><Square size={13} />停止</Button> : <Button variant="primary" onClick={run} disabled={!currentKb}><Play size={13} />开始协作</Button>}
          </div>
          <div className="row gap-6 wrap">
            {['planner', 'researcher', 'analyst', 'reviewer', 'writer'].map((k, i) => (
              <span key={k} className="row gap-6 xs muted">
                {i > 0 ? <span className="faint">→</span> : null}
                <Badge tone={AGENTS[k].tone}>{AGENTS[k].icon}{AGENTS[k].name}</Badge><span className="faint">{AGENTS[k].desc}</span>
              </span>
            ))}
          </div>
          <div className="xs faint">智能体之间不直接传话，而是读写同一块 <b>LanceDB 黑板</b>（表 agent_board）：研究员把证据贴上去，分析师 / 撰稿人按自己的子任务做语义检索取回；整次运行可回放，也能检索以前做过的相似任务。</div>
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) 300px', alignItems: 'start', gap: 16 }}>
        <div className="col gap-12">
          {replay ? <ReplayTimeline run={replay.run} rows={replay.rows} /> : events.length === 0 && !running ? (
            <div className="card"><Empty icon={<Users size={32} />} title="还没有开始" desc="选择知识库、输入问题后点「开始协作」，这里会实时显示每个智能体的工作。" /></div>
          ) : (
            <>
              {events.map((e, i) => <EventCard key={i} e={e} />)}
              {running ? <div className="card"><div className="card-body"><Spinner label="智能体协作中…" /></div></div> : null}
              <div ref={endRef} />
            </>
          )}
        </div>
        <div className="col gap-12">
          {sources.length ? (
            <div className="card">
              <div className="card-header"><h3><BookOpenCheck size={14} />证据（{sources.length}）</h3></div>
              <div className="card-body col gap-6">
                {sources.map((s) => (
                  <div key={s.chunkId} className="chunk">
                    <div className="chunk-head"><span className="ellipsis"><b className="mono">[{s.n}]</b> {s.label}</span></div>
                    <div className="xs muted clamp-2">{s.text}</div>
                    <div className="xs faint mt-4 ellipsis">子任务：{s.task}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
          <RunList runs={runs} activeId={replay?.run.runId} running={running} onOpen={openRun} onDelete={removeRuns} />
        </div>
      </div>
    </div>
  );
}

function RunList({ runs, activeId, running, onOpen, onDelete }: {
  runs: AgentRunSummary[];
  activeId?: string;
  running: boolean;
  onOpen: (r: AgentRunSummary) => void;
  onDelete: (ids: string[] | 'all') => Promise<boolean>;
}) {
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  // 列表刷新后，去掉已不存在的勾选
  useEffect(() => { setSelected((sel) => new Set(runs.filter((r) => sel.has(r.runId)).map((r) => r.runId))); }, [runs]);
  useEffect(() => { if (!runs.length) setSelecting(false); }, [runs.length]);

  const toggle = (id: string) => setSelected((sel) => {
    const next = new Set(sel);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  });
  const del = async (ids: string[] | 'all') => {
    setBusy(true);
    try {
      if (await onDelete(ids)) setSelected(new Set());
    } finally {
      setBusy(false);
    }
  };
  const allChecked = runs.length > 0 && selected.size === runs.length;

  return (
    <div className="card">
      <div className="card-header">
        <h3><History size={14} />运行记录<Badge mono>{runs.length}</Badge></h3>
        {runs.length ? (
          selecting
            ? <Button size="xs" variant="ghost" onClick={() => { setSelecting(false); setSelected(new Set()); }}>完成</Button>
            : <Button size="xs" variant="ghost" onClick={() => setSelecting(true)}><CheckSquare size={12} />批量</Button>
        ) : null}
      </div>
      {selecting ? (
        <div className="card-body row between wrap gap-6" style={{ borderBottom: '1px solid var(--border)' }}>
          <label className="row gap-6 xs muted" style={{ cursor: 'pointer' }}>
            <input type="checkbox" style={{ width: 'auto' }} checked={allChecked} onChange={() => setSelected(allChecked ? new Set() : new Set(runs.map((r) => r.runId)))} />全选
          </label>
          <div className="row gap-6">
            <Button size="xs" variant="danger" loading={busy} disabled={!selected.size} onClick={() => del([...selected])}><Trash2 size={11} />删除所选（{selected.size}）</Button>
            <Button size="xs" variant="danger" disabled={busy || running} title={running ? '协作进行中，结束后再清空' : '删除黑板上的全部运行记录'} onClick={() => del('all')}>清空全部</Button>
          </div>
        </div>
      ) : null}
      {runs.length === 0 ? <div className="card-body xs faint">还没有运行记录</div> : (
        <div className="card-body col gap-6">
          {runs.map((r) => {
            const checked = selected.has(r.runId);
            return (
              <div key={r.runId} className={`chunk run-item row gap-8 ${activeId === r.runId || checked ? 'hl' : ''}`} style={{ alignItems: 'flex-start' }}>
                {selecting ? <input type="checkbox" style={{ width: 'auto', marginTop: 3 }} checked={checked} onChange={() => toggle(r.runId)} aria-label="选择" /> : null}
                <button className="flex-1" style={{ textAlign: 'left', cursor: 'pointer', background: 'none', border: 0, padding: 0, color: 'inherit', font: 'inherit' }} onClick={() => (selecting ? toggle(r.runId) : onOpen(r))}>
                  <div className="small strong clamp-2">{r.question}</div>
                  <div className="xs faint mt-4">{fmtTime(r.ts)}{r.userId ? ` · ${r.userId}` : ''}</div>
                </button>
                {selecting ? null : (
                  <Button className="run-del" variant="ghost" icon size="xs" disabled={busy} title="删除这条运行记录" aria-label="删除" onClick={() => del([r.runId])}><Trash2 size={12} /></Button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function AgentHead({ agent, type, ms }: { agent: string; type: string; ms?: number }) {
  const a = AGENTS[agent] ?? AGENTS.system;
  return (
    <div className="card-header">
      <h3><Badge tone={a.tone}>{a.icon}{a.name}</Badge><span className="small muted">{TYPE_LABEL[type] ?? type}</span></h3>
      {ms != null ? <span className="mono xs faint">+{(ms / 1000).toFixed(1)}s</span> : null}
    </div>
  );
}

function EventCard({ e }: { e: AgentEvent }) {
  const d = e.data || {};
  if (e.type === 'run') {
    return (
      <div className="card">
        <AgentHead agent="system" type="run" ms={e.ms} />
        <div className="card-body col gap-8">
          <div className="small strong">{e.content}</div>
          <div className="row gap-6 wrap xs faint"><span className="mono">run {d.runId}</span>{d.offline ? <Badge tone="warn">未配置文本模型：抽取式协作</Badge> : null}</div>
          {d.similarRuns?.length ? <div className="xs muted">黑板上相似的历史任务：{d.similarRuns.map((r: any) => `「${r.question}」(${r.similarity})`).join('、')}</div> : null}
        </div>
      </div>
    );
  }
  return (
    <div className="card">
      <AgentHead agent={e.agent} type={e.type} ms={e.ms} />
      <div className="card-body col gap-8">
        {e.type === 'memory' ? <div className="row gap-6 wrap">{(d.memories as string[]).map((m) => <Badge key={m} tone="accent">{m}</Badge>)}</div> : null}
        {e.type === 'plan' ? (
          <>
            {e.content ? <div className="xs muted">{e.content}</div> : null}
            <div className="col gap-4">{(d.subtasks as string[]).map((t, i) => <div key={t} className="small"><b className="mono">{i + 1}.</b> {t}</div>)}</div>
          </>
        ) : null}
        {e.type === 'evidence' ? (
          <>
            <div className="small">检索：<b>{e.content}</b>{d.round > 1 ? <Badge tone="warn">第 {d.round} 轮</Badge> : null}</div>
            {(d.items as any[]).length ? <div className="col gap-4">{(d.items as any[]).map((it) => <div key={it.chunkId} className="row between xs"><span className="ellipsis muted">{it.label}</span><span className="mono faint">{Number(it.score).toFixed(3)}</span></div>)}</div> : <div className="xs text-warning">没有检索到资料</div>}
          </>
        ) : null}
        {e.type === 'recall' ? (
          <>
            <div className="xs muted">{e.content}</div>
            <div className="col gap-4">{(d.perTask as any[]).map((p) => <div key={p.task} className="row between xs"><span className="ellipsis">{p.task}</span><span className="mono faint">{p.evidence.length ? p.evidence.map((n: number) => `[${n}]`).join('') : '无'}</span></div>)}</div>
          </>
        ) : null}
        {e.type === 'analysis' || e.type === 'final' ? <Markdown text={e.content} /> : null}
        {e.type === 'review' ? (
          <div className="col gap-6">
            <div className="row gap-6 wrap"><Badge tone={d.sufficient ? 'ok' : 'warn'}>{d.sufficient ? '证据充分' : '证据不足'}</Badge><span className="small">{e.content}</span></div>
            {d.missing?.length ? <div className="xs muted">还缺：{d.missing.join('；')}</div> : null}
          </div>
        ) : null}
        {e.type === 'requery' ? <div className="row gap-6 wrap"><span className="xs muted">{e.content}：</span>{(d.queries as string[]).map((q) => <Badge key={q} tone="warn">{q}</Badge>)}</div> : null}
        {e.type === 'final' && d.sufficient === false ? <Callout tone="warn">审校认为证据可能不充分，回答中已标出不确定的部分。</Callout> : null}
        {e.type === 'error' ? <Callout tone="bad">{e.content}</Callout> : null}
      </div>
    </div>
  );
}

function ReplayTimeline({ run, rows }: { run: AgentRunSummary; rows: BoardRow[] }) {
  return (
    <div className="col gap-12">
      <div className="card"><div className="card-body row between wrap gap-8"><span className="small strong">回放：{run.question}</span><span className="xs faint">{fmtTime(run.ts)} · 黑板上 {rows.length} 条记录</span></div></div>
      {rows.filter((r) => r.kind !== 'question').map((r) => (
        <div key={r.id} className="card">
          <AgentHead agent={r.agent} type={r.kind === 'answer' ? 'final' : r.kind} />
          <div className="card-body">
            {r.kind === 'analysis' || r.kind === 'answer' ? <Markdown text={r.content} /> : <div className="small pre">{r.kind === 'review' ? reviewText(r.content) : r.content}</div>}
          </div>
        </div>
      ))}
    </div>
  );
}

function reviewText(raw: string): string {
  try {
    const v = JSON.parse(raw);
    return `${v.sufficient ? '证据充分' : '证据不足'}：${v.comment ?? ''}${v.followups?.length ? `\n改写重查：${v.followups.join('；')}` : ''}`;
  } catch {
    return raw;
  }
}
