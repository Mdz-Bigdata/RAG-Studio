// 语料准备：意图空间 × 知识空间。把每条意图与现有语料的匹配程度分到三个区间，
// 决定"继续准备知识"还是"拿掉与目标无关的材料"。
import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Download, FileQuestion, GitCompare, Library, Play, Target } from 'lucide-react';
import type { PageKey } from '../app';
import { api } from '../api/client';
import type { CoverageBucket, CoverageDoc, CoverageReport } from '../api/types';
import { KbSelect } from '../components/kb-select';
import { Badge, Button, Callout, Empty, ScoreBar, Tabs } from '../components/ui';
import { formatDate, pct } from '../lib/format';
import { useApp } from '../store/app-store';

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));
const BUCKET: Record<CoverageBucket, { label: string; tone: string; color: string; meaning: string; action: string }> = {
  high: { label: '高匹配', tone: 'ok', color: 'var(--success)', meaning: '意图有明确对应的语料', action: '纳入黄金评估集，作为回归基线' },
  partial: { label: '部分匹配', tone: 'warn', color: 'var(--warning)', meaning: '语料存在，但缺章节 / 版本旧 / 只命中部分关键词', action: '继续准备知识：补章节、更新版本、把表格导出为结构化数据' },
  none: { label: '无匹配', tone: 'bad', color: 'var(--danger)', meaning: '意图在库里没有对应语料', action: '列入知识补充计划，补齐前配置拒答与转人工话术' },
};
const DOC_STATUS: Record<CoverageDoc['status'], { label: string; tone: string }> = {
  core: { label: '核心语料', tone: 'ok' },
  supporting: { label: '辅助语料', tone: 'info' },
  unused: { label: '未被命中', tone: 'warn' },
  expired: { label: '已过期', tone: 'bad' },
};
const EXTRA_SAMPLES = ['年终奖怎么发', '笔记本电脑保修多久', 'Kafka 消费积压怎么处理', '2024 年年假最多能休几天'];

/** 每行一个问题；也接受黄金集 JSONL（{"question": ..., "category": ...}）。 */
function parseIntents(text: string): (string | { question: string; category?: string })[] {
  return text.split('\n').map((l) => l.trim()).filter(Boolean).map((l) => {
    if (l.startsWith('{')) {
      try {
        const o = JSON.parse(l);
        if (o && typeof o.question === 'string') return { question: o.question, category: o.category };
      } catch { /* 当作普通文本 */ }
    }
    return l;
  });
}

/** 与上次分析的对比：模型 / 打分公式 / 阈值都相同才给出变化量，否则明确提示不可比较。 */
function ComparisonCallout({ report }: { report: CoverageReport }) {
  const c = report.comparison!;
  const prev = c.previous;
  const when = prev.createdAt ? formatDate(prev.createdAt) : '上次';
  if (!c.comparable) {
    return (
      <Callout tone="warn" icon={<AlertTriangle size={15} />}>
        <div className="col gap-4">
          <b>与上次分析（{when}）不可直接比较</b>
          {c.reasons.map((r) => <div key={r}>· {r}</div>)}
          <div className="xs muted">上次：高 {prev.buckets.high} · 部分 {prev.buckets.partial} · 无 {prev.buckets.none}（可回答率 {pct(prev.answerable_rate)}）。分数变化可能来自模型或打分方式，而不是语料本身；要评估语料改动的效果，请在相同模型下各跑一次。</div>
        </div>
      </Callout>
    );
  }
  const delta = (a: number, b: number) => (a === b ? '持平' : `${a > b ? '+' : ''}${a - b}`);
  return (
    <Callout icon={<GitCompare size={15} />}>
      <div className="col gap-4">
        <b>与上次分析（{when}）相比</b>
        <div>高匹配 {prev.buckets.high} → {report.buckets.high}（{delta(report.buckets.high, prev.buckets.high)}）· 部分匹配 {prev.buckets.partial} → {report.buckets.partial} · 无匹配 {prev.buckets.none} → {report.buckets.none} · 可回答率 {pct(prev.answerable_rate)} → {pct(report.answerable_rate)}</div>
        {c.notes?.map((n) => <div key={n} className="xs text-warning">{n}</div>)}
        {!c.sameIntents ? <div className="xs text-warning">两次的意图集合不同，总体比例仅供参考。</div> : !c.notes?.length ? <div className="xs muted">模型、打分公式、检索策略、阈值与意图集合均相同，变化来自语料本身。</div> : null}
      </div>
    </Callout>
  );
}

export function CoveragePage({ go }: { go: (key: PageKey) => void }) {
  const { currentKb, toast } = useApp();
  const kbId = currentKb?.id;
  const [text, setText] = useState('');
  const [high, setHigh] = useState(0.6);
  const [partial, setPartial] = useState(0.35);
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<CoverageReport | null>(null);
  const [tab, setTab] = useState<'intents' | 'docs'>('intents');
  const [filter, setFilter] = useState<CoverageBucket | 'all'>('all');

  useEffect(() => {
    setReport(null);
    if (!kbId) return;
    api.eval.lastCoverage(kbId).then((r) => {
      if (!r) return;
      setReport(r);
      setText((t) => t || r.intents.map((i) => i.intent).join('\n'));
    }).catch(() => undefined);
  }, [kbId]);

  const intents = useMemo(() => parseIntents(text), [text]);
  const thresholdsOk = partial > 0 && partial < high && high <= 1;

  const loadSample = async () => {
    try {
      const s = await api.eval.sample();
      setText([...s.map((e) => e.question), ...EXTRA_SAMPLES].join('\n'));
      toast(`已载入 ${s.length + EXTRA_SAMPLES.length} 条示例意图（含库里没有答案的问题）`, 'ok');
    } catch (e) { toast(errMsg(e), 'bad'); }
  };
  const run = async () => {
    if (!kbId || intents.length === 0) return;
    setRunning(true);
    try {
      const r = await api.eval.coverage(kbId, { intents, high, partial });
      setReport(r);
      setFilter('all');
      toast(`分析完成：${r.n} 条意图，高匹配 ${r.buckets.high} · 部分 ${r.buckets.partial} · 无 ${r.buckets.none}`, 'ok');
    } catch (e) { toast(errMsg(e), 'bad'); } finally { setRunning(false); }
  };
  const download = () => {
    if (!report) return;
    const url = URL.createObjectURL(new Blob([report.markdown], { type: 'text/markdown;charset=utf-8' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `coverage-${currentKb?.name ?? 'kb'}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const rows = useMemo(() => (report ? [...report.intents].sort((a, b) => a.score - b.score).filter((r) => filter === 'all' || r.bucket === filter) : []), [report, filter]);

  return (
    <div className="page col gap-16 fade-in">
      <div className="grid grid-2">
        <div className="card">
          <div className="card-header">
            <h3><FileQuestion size={15} />意图空间：用户会问什么</h3>
            <div className="row gap-8"><KbSelect /><Button size="sm" onClick={loadSample}>载入示例</Button></div>
          </div>
          <div className="card-body col gap-12">
            <textarea rows={9} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} placeholder={'每行一个真实的用户问题（来自工单、搜索日志、群聊、销售 FAQ）\n例如：\n年假可以跨年使用吗\n订单 SO-8817 的状态\n笔记本电脑保修多久'} />
            <div className="row between wrap gap-12">
              <div className="row gap-12 wrap">
                <label className="row gap-6 small muted">高匹配 ≥<input type="number" style={{ width: 72 }} min={0.1} max={1} step={0.05} value={high} onChange={(e) => setHigh(Number(e.target.value))} /></label>
                <label className="row gap-6 small muted">部分匹配 ≥<input type="number" style={{ width: 72 }} min={0.05} max={0.95} step={0.05} value={partial} onChange={(e) => setPartial(Number(e.target.value))} /></label>
                <span className="xs faint">{thresholdsOk ? `${intents.length} 条意图` : <span className="text-danger">阈值需满足 0 &lt; 部分 &lt; 高 ≤ 1</span>}</span>
              </div>
              <Button variant="primary" loading={running} disabled={!kbId || intents.length === 0 || !thresholdsOk} onClick={run}><Play size={13} />分析覆盖</Button>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header"><h3><Target size={15} />匹配区间与处置</h3></div>
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>区间</th><th>含义</th><th>处置</th></tr></thead>
              <tbody>
                {(Object.keys(BUCKET) as CoverageBucket[]).map((b) => (
                  <tr key={b}><td><Badge tone={BUCKET[b].tone}>{BUCKET[b].label}</Badge></td><td className="muted">{BUCKET[b].meaning}</td><td>{BUCKET[b].action}</td></tr>
                ))}
                <tr><td><Badge tone="warn">未被命中的文档</Badge></td><td className="muted">没有任何意图用到它</td><td>确认与 RAG 目的无关后移出语料，避免噪音与权重虚高</td></tr>
              </tbody>
            </table>
          </div>
          <div className="card-body xs faint" style={{ borderTop: '1px solid var(--border)' }}>
            每条意图都跑一次真实检索（不做查询改写），匹配分由关键词覆盖率（按语料 IDF 加权，库里从未出现的词权重最高）、向量相似度与精排相关度组成，所以结论与线上行为一致。离线模式下没有真实语义，分数偏保守。
          </div>
        </div>
      </div>

      {!currentKb ? <div className="card"><Empty icon={<Target size={36} />} title="请先选择知识库" desc="覆盖分析需要一个已入库文档的知识库。" action={<Button size="sm" onClick={() => go('kb')}>去知识库</Button>} /></div> : !report ? (
        <div className="card"><Empty icon={<Target size={36} />} title="还没有覆盖分析" desc="在调任何检索参数之前，先确认知识库能回答用户会问的问题。载入示例意图或粘贴你自己的问题列表，然后点「分析覆盖」。" /></div>
      ) : (
        <>
          <div className="card">
            <div className="stat-strip">
              <div className="stat"><div className="label">意图数</div><div className="value">{report.n}</div><div className="sub">{report.seconds.toFixed(2)}s</div></div>
              {(Object.keys(BUCKET) as CoverageBucket[]).map((b) => (
                <div className="stat" key={b}><div className="label"><span className="dot" style={{ background: BUCKET[b].color }} />{BUCKET[b].label}</div><div className="value">{report.buckets[b]}</div><div className="sub">{pct(report.n ? report.buckets[b] / report.n : 0)}</div></div>
              ))}
              <div className="stat"><div className="label">可回答率（高 + 部分）</div><div className="value">{pct(report.answerable_rate)}</div><div className="sub">高匹配覆盖率 {pct(report.coverage_rate)}</div></div>
            </div>
            <div className="card-body" style={{ borderTop: '1px solid var(--border)' }}>
              <div className="seg-bar" role="img" aria-label={`高匹配 ${report.buckets.high}，部分匹配 ${report.buckets.partial}，无匹配 ${report.buckets.none}`}>
                {report.buckets.high ? <i className="seg-high" style={{ flex: report.buckets.high }} /> : null}
                {report.buckets.partial ? <i className="seg-partial" style={{ flex: report.buckets.partial }} /> : null}
                {report.buckets.none ? <i className="seg-none" style={{ flex: report.buckets.none }} /> : null}
              </div>
              <div className="xs faint mt-8">打分方式：{report.scoring}</div>
              {report.models ? (
                <div className="row gap-6 wrap mt-8 xs">
                  <span className="faint">模型：</span>
                  <Badge mono tone={report.models.embedderOffline ? 'warn' : ''} title="嵌入模型">嵌入 {report.models.embedderModel}{report.models.embedderOffline ? '（离线哈希，无真实语义）' : ''}</Badge>
                  <Badge mono tone={report.models.rerankerFallback ? 'warn' : ''} title="本次实际生效的精排">精排 {report.models.rerankerEffective}</Badge>
                  {report.createdAt ? <span className="faint">· {formatDate(report.createdAt)}</span> : null}
                </div>
              ) : <div className="xs text-warning mt-8">该报告未记录所用模型（早于模型记录功能），重新分析一次即可补上。</div>}
            </div>
          </div>

          {report.comparison ? <ComparisonCallout report={report} /> : null}

          <Callout>
            <div className="col gap-4">
              <b>建议</b>
              {report.recommendations.map((r, i) => <div key={i}>· {r}</div>)}
            </div>
          </Callout>

          <div className="card">
            <div className="card-header">
              <Tabs value={tab} onChange={setTab} items={[{ value: 'intents', label: `意图空间（${report.n}）`, icon: <FileQuestion size={13} /> }, { value: 'docs', label: `知识空间（${report.docs.length}）`, icon: <Library size={13} /> }]} />
              <div className="row gap-8 wrap">
                {tab === 'intents' ? (['all', 'none', 'partial', 'high'] as const).map((f) => (
                  <button key={f} className={`btn xs chip ${filter === f ? 'primary' : ''}`} onClick={() => setFilter(f)}>{f === 'all' ? '全部' : BUCKET[f].label}{f !== 'all' ? ` ${report.buckets[f]}` : ''}</button>
                )) : null}
                <Button size="sm" onClick={download}><Download size={13} />导出 Markdown</Button>
              </div>
            </div>
            {tab === 'intents' ? (
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th>意图</th><th>区间</th><th className="num">匹配分</th><th className="num">关键词覆盖</th><th>缺失关键词</th><th>最优语料</th></tr></thead>
                  <tbody>
                    {rows.map((r) => (
                      <tr key={r.intent}>
                        <td style={{ maxWidth: 340 }}>
                          <div>{r.intent}</div>
                          <div className="row gap-6 mt-4">{r.category !== '未分类' ? <Badge>{r.category}</Badge> : null}{r.note ? <span className="xs text-warning">{r.note}</span> : null}</div>
                        </td>
                        <td><Badge tone={BUCKET[r.bucket].tone}>{BUCKET[r.bucket].label}</Badge></td>
                        <td className="num" style={{ minWidth: 86 }}>{r.score.toFixed(2)}<div style={{ marginTop: 4 }}><ScoreBar value={r.score} color={BUCKET[r.bucket].color} /></div></td>
                        <td className="num">{pct(r.term_coverage)}</td>
                        <td><div className="row gap-4 wrap">{r.missing_terms.length ? r.missing_terms.map((t) => <Badge key={t} mono tone={r.bucket === 'none' ? 'bad' : ''}>{t}</Badge>) : <span className="faint">—</span>}</div></td>
                        <td style={{ maxWidth: 320 }}>
                          {r.top[0] ? <><div className="ellipsis" title={r.top[0].breadcrumb}>{r.bucket === 'none' ? <span className="faint">（最接近）</span> : null}{r.top[0].breadcrumb || r.top[0].file_name}</div><div className="xs faint ellipsis">{r.top[0].file_name}</div></> : <span className="faint">无候选</span>}
                        </td>
                      </tr>
                    ))}
                    {rows.length === 0 ? <tr><td colSpan={6} className="faint" style={{ textAlign: 'center', padding: 24 }}>该区间没有意图</td></tr> : null}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th>文档</th><th>版本 / 失效</th><th className="num">切片</th><th className="num">命中意图</th><th>状态</th><th>处置</th></tr></thead>
                  <tbody>
                    {report.docs.map((d) => (
                      <tr key={d.doc_id}>
                        <td style={{ maxWidth: 280 }}><div className="strong ellipsis">{d.title}</div><div className="xs faint ellipsis">{d.file_name ?? d.source}</div></td>
                        <td className="muted xs">{d.version ?? '—'}{d.valid_to ? <><br />至 {d.valid_to}</> : null}</td>
                        <td className="num">{d.chunks}</td>
                        <td className="num">{d.hits}</td>
                        <td><Badge tone={DOC_STATUS[d.status].tone}>{DOC_STATUS[d.status].label}</Badge></td>
                        <td style={{ maxWidth: 380 }}><div>{d.action}</div>{d.hit_intents.length ? <div className="xs faint mt-4 clamp-2">命中：{d.hit_intents.join('；')}</div> : null}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="row gap-8 wrap">
            <Button size="sm" onClick={() => go('kb')}>去知识库补充 / 下线文档</Button>
            <Button size="sm" onClick={() => go('eval')}>把已覆盖的意图标注成评估集</Button>
          </div>
        </>
      )}
    </div>
  );
}
