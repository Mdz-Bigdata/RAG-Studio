import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, BarChart3, CheckCircle2, FileJson, Play, XCircle } from 'lucide-react';
import { api } from '../api/client';
import type { EvalExample, EvalReport } from '../api/types';
import { KbSelect } from '../components/kb-select';
import { MultimodalEval } from '../components/multimodal-eval';
import { HardNegativeMining } from '../components/hard-negative-mining';
import { Badge, Button, Callout, Empty, ScoreBar, Tabs } from '../components/ui';
import { formatDate, pct } from '../lib/format';
import { useApp } from '../store/app-store';

const STAGES: { key: string; label: string; color: string }[] = [
  { key: 'dense', label: '仅向量', color: 'var(--c-dense)' },
  { key: 'sparse', label: '仅 BM25', color: 'var(--c-sparse)' },
  { key: 'fused', label: '融合', color: 'var(--c-fused)' },
  { key: 'final', label: '完整链路', color: 'var(--c-rerank)' },
];
const ERROR_HINT: Record<string, string> = {
  '知识缺失/切块失败': '任何一路都没召回 → 先查语料是否存在、切片是否把答案切断',
  '语义漂移(向量未召回)': 'BM25 找到了向量没找到 → 换 embedding / 加面包屑 / 上下文摘要',
  '关键词缺失(BM25未召回)': '向量找到了 BM25 没找到 → 领域词典 / 精确 token 保护',
  '融合失败': '某路找到了但融合后掉出候选 → 调 RRF 权重 / 候选数',
  '排序失败(rerank)': '候选里有但精排后掉出 top-K → 换 reranker / 候选 ≥ 50',
  '不该答却答': '无答案问题仍返回了上下文 → 设置 rerank.min_score 阈值 + 兜底话术',
};

function parseJsonl(text: string): EvalExample[] {
  return text.split('\n').map((l) => l.trim()).filter(Boolean).map((l, i) => {
    try { return JSON.parse(l) as EvalExample; } catch { throw new Error(`第 ${i + 1} 行不是合法 JSON`); }
  });
}

function TextEval() {
  const { currentKb, toast } = useApp();
  const [text, setText] = useState('');
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<EvalReport | null>(null);
  const [history, setHistory] = useState<EvalReport[]>([]);
  const [name, setName] = useState('');
  const kbId = currentKb?.id;

  useEffect(() => {
    setReport(null);
    if (kbId) api.eval.reports(kbId).then((r) => { setHistory(r); setReport(r[0] ?? null); }).catch(() => setHistory([]));
  }, [kbId]);

  const loadSample = async () => {
    try {
      const s = await api.eval.sample();
      setText(s.map((e) => JSON.stringify(e)).join('\n'));
      toast(`已载入 ${s.length} 条示例`, 'ok');
    } catch (e) { toast(e instanceof Error ? e.message : String(e), 'bad'); }
  };
  const run = async () => {
    if (!kbId) return;
    let examples: EvalExample[];
    try { examples = parseJsonl(text); } catch (e) { toast((e as Error).message, 'bad'); return; }
    if (examples.length === 0) { toast('请先填写或载入黄金评估集', 'warn'); return; }
    setRunning(true);
    try {
      const r = await api.eval.run(kbId, { examples, name: name || undefined });
      setReport(r);
      setHistory((h) => [r, ...h]);
      toast(`评测完成：${r.n} 条样例，用时 ${r.seconds}s`, 'ok');
    } catch (e) { toast(e instanceof Error ? e.message : String(e), 'bad'); } finally { setRunning(false); }
  };

  const metrics = useMemo(() => {
    if (!report) return [] as string[];
    const ks = report.ks;
    return [...ks.map((k) => `hit@${k}`), 'mrr', ...ks.map((k) => `ndcg@${k}`)];
  }, [report]);
  const final = report?.stageMetrics.final ?? {};
  const mainK = report ? (report.ks.includes(5) ? 5 : report.ks[report.ks.length - 1]) : 5;
  const failed = report?.examples.filter((e) => e.error_type) ?? [];
  const count = (() => { try { return parseJsonl(text).length; } catch { return -1; } })();

  return (
    <div className="page col gap-16 fade-in">
      <div className="grid grid-2">
        <div className="card">
          <div className="card-header"><h3><FileJson size={15} />黄金评估集（JSONL）</h3><div className="row gap-8"><KbSelect /><Button size="sm" onClick={loadSample}>载入示例</Button></div></div>
          <div className="card-body col gap-12">
            <textarea className="mono" rows={11} style={{ fontSize: 12 }} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} placeholder={'{"question": "年假可以跨年使用吗", "answer_keywords": ["次年 3 月 31 日"], "category": "常见问题"}\n{"question": "公司食堂几点开饭", "category": "无答案", "expected_refusal": true}'} />
            <div className="row between wrap gap-12">
              <span className="xs faint">{count < 0 ? <span className="text-danger">存在非法 JSON 行</span> : `${count} 条样例`} · 字段：question / answer_keywords / relevant_sources / relevant_chunk_ids / category / expected_refusal</span>
              <div className="row gap-8"><input style={{ width: 160 }} placeholder="本次评测备注（可选）" value={name} onChange={(e) => setName(e.target.value)} /><Button variant="primary" loading={running} disabled={!kbId || count <= 0} onClick={run}><Play size={13} />运行评测</Button></div>
            </div>
          </div>
        </div>
        <div className="card pad col gap-12">
          <div className="card-title" style={{ marginBottom: 0 }}><BarChart3 size={15} />为什么要评测</div>
          <Callout>没有度量，其他调优都是盲修。每改一个变量（切片 / 融合 / 重排 / 改写）都用同一份评估集跑一次，<b>指标不动就不合入</b>。</Callout>
          <ul className="small muted" style={{ paddingLeft: 18, lineHeight: 1.9 }}>
            <li>建议 ≥ 100 条：常见问题 50% · 长尾 30% · 多跳 / 模糊 20%，另加无答案与时效冲突样例</li>
            <li><b className="strong">分阶段消融</b>：仅向量 / 仅 BM25 / 融合 / 完整链路，定位是哪一层拖后腿</li>
            <li><b className="strong">错误归因</b>比指标更能指导优化方向（切块失败 / 语义漂移 / 关键词缺失 / 排序失败）</li>
            <li>评测默认关闭 LLM 查询变换，保证结果可复现</li>
          </ul>
          {history.length ? (
            <div>
              <div className="xs faint mb-8">历史报告</div>
              <div className="col gap-4">{history.slice(0, 5).map((h) => (
                <div key={h.id} className={`list-item ${report?.id === h.id ? 'active' : ''}`} style={{ padding: '6px 10px' }} onClick={() => setReport(h)}>
                  <span className="small flex-1 ellipsis">{h.name || '未命名'} · {formatDate(h.createdAt)}</span>
                  {h.models?.embedderOffline || h.models?.rerankerFallback ? <Badge tone="warn" title="评测时嵌入或精排处于降级状态，指标偏低，不宜与正常报告比较">降级</Badge> : null}
                  <Badge mono tone="rerank">hit@5 {pct(h.stageMetrics.final?.['hit@5'] ?? h.stageMetrics.final?.[`hit@${h.ks.at(-1)}`])}</Badge>
                </div>
              ))}</div>
            </div>
          ) : null}
        </div>
      </div>

      {!report ? <div className="card"><Empty icon={<BarChart3 size={40} />} title="尚无评测报告" desc="载入示例集（对应示例语料）或粘贴你自己的黄金集，然后运行评测。" /></div> : (
        <>
          <div className="card stat-strip">
            <div className="stat"><div className="label">Hit@{mainK}（完整链路）</div><div className="value">{pct(final[`hit@${mainK}`])}</div><div className="sub">Top-{mainK} 至少命中一条相关资料</div></div>
            <div className="stat"><div className="label">MRR</div><div className="value">{(final.mrr ?? 0).toFixed(3)}</div><div className="sub">正确答案排得越靠前越高</div></div>
            <div className="stat"><div className="label">NDCG@{mainK}</div><div className="value">{(final[`ndcg@${mainK}`] ?? 0).toFixed(3)}</div><div className="sub">综合排名与相关性</div></div>
            <div className="stat"><div className="label">拒答正确率</div><div className="value">{report.refusalTotal ? `${report.refusalCorrect}/${report.refusalTotal}` : '—'}</div><div className="sub">{report.refusalTotal ? '无答案样例中未返回上下文的比例' : '评估集中没有无答案样例'}</div></div>
          </div>

          <EvalProvenance report={report} mainK={mainK} />

          <div className="card">
            <div className="card-header"><h3>分阶段消融</h3><span className="xs faint">n = {report.n} · {report.seconds}s</span></div>
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>阶段</th>{metrics.map((m) => <th key={m} className="num">{m}</th>)}</tr></thead>
                <tbody>
                  {STAGES.filter((s) => report.stageMetrics[s.key]).map((s) => (
                    <tr key={s.key}>
                      <td><span className="row gap-6"><span className="dot" style={{ background: s.color }} />{s.label}</span></td>
                      {metrics.map((m) => {
                        const v = report.stageMetrics[s.key][m] ?? 0;
                        const best = Math.max(...STAGES.map((x) => report.stageMetrics[x.key]?.[m] ?? 0));
                        return <td key={m} className="num" style={{ minWidth: 70 }}><span style={v === best ? { color: s.color, fontWeight: 600 } : undefined}>{v.toFixed(3)}</span><div style={{ marginTop: 3 }}><ScoreBar value={v} color={s.color} /></div></td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="grid grid-2">
            <div className="card">
              <div className="card-header"><h3>按问题类型（完整链路）</h3></div>
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th>类型</th><th className="num">hit@1</th><th className="num">hit@{mainK}</th><th className="num">mrr</th></tr></thead>
                  <tbody>{Object.entries(report.categoryMetrics).map(([c, m]) => (
                    <tr key={c}><td>{c}</td><td className="num">{pct(m['hit@1'])}</td><td className="num">{pct(m[`hit@${mainK}`])}</td><td className="num">{(m.mrr ?? 0).toFixed(3)}</td></tr>
                  ))}</tbody>
                </table>
              </div>
            </div>
            <div className="card">
              <div className="card-header"><h3>错误归因</h3>{failed.length === 0 ? <Badge tone="ok"><CheckCircle2 size={11} />无失败样例</Badge> : <Badge tone="bad"><XCircle size={11} />{failed.length} 条失败</Badge>}</div>
              <div className="card-body col gap-8">
                {Object.keys(report.errorCounts).length === 0 ? <div className="small faint">全部样例在 Top-{report.ks.at(-1)} 内命中。</div> : Object.entries(report.errorCounts).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                  <div key={k} className="chunk"><div className="row between"><span className="strong small"><AlertTriangle size={12} className="text-warning" style={{ verticalAlign: -2 }} /> {k}</span><Badge tone="bad" mono>{v}</Badge></div><div className="xs muted mt-8">{ERROR_HINT[k] ?? ''}</div></div>
                ))}
                {failed.slice(0, 8).map((e, i) => <div key={i} className="xs muted">· [{e.error_type}] {e.question}</div>)}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header"><h3>逐条结果</h3></div>
            <div className="table-wrap scroll" style={{ maxHeight: 420 }}>
              <table className="table">
                <thead><tr><th>问题</th><th>类型</th><th className="num">向量</th><th className="num">BM25</th><th className="num">融合</th><th className="num">完整</th><th>结果</th></tr></thead>
                <tbody>{report.examples.map((e, i) => {
                  const rr = (s: string) => (e.per_stage[s] ? (e.per_stage[s].mrr ?? 0).toFixed(2) : '—');
                  return <tr key={i}><td style={{ maxWidth: 360 }}>{e.question}</td><td><Badge>{e.category}</Badge></td><td className="num">{rr('dense')}</td><td className="num">{rr('sparse')}</td><td className="num">{rr('fused')}</td><td className="num">{rr('final')}</td><td>{e.error_type ? <Badge tone="bad">{e.error_type}</Badge> : <Badge tone="ok">通过</Badge>}</td></tr>;
                })}</tbody>
              </table>
            </div>
            <div className="card-body xs faint" style={{ borderTop: '1px solid var(--border)' }}>数值为各阶段的倒数排名（1.00 = 排第一，0.50 = 排第二，0 = 未命中）。</div>
          </div>
        </>
      )}
    </div>
  );
}

/** 报告用了哪些模型，以及能否与上一份报告直接比较（模型 / 精排 / 参数覆盖 / 评估集 / K 值都相同才给出变化量）。 */
function EvalProvenance({ report, mainK }: { report: EvalReport; mainK: number }) {
  const m = report.models;
  const c = report.comparison;
  const final = report.stageMetrics.final ?? {};
  const prev = c?.previous;
  const when = prev?.createdAt ? formatDate(prev.createdAt) : '上次';
  const delta = (cur?: number, old?: number, digits = 3) => (cur == null || old == null ? '—' : `${old.toFixed(digits)} → ${cur.toFixed(digits)}`);
  return (
    <>
      <div className="row gap-6 wrap xs">
        <span className="faint">模型：</span>
        {m ? (
          <>
            <Badge mono tone={m.embedderOffline ? 'warn' : ''}>嵌入 {m.embedderModel}{m.embedderOffline ? '（离线哈希，无真实语义）' : ''}</Badge>
            <Badge mono tone={m.rerankerFallback ? 'warn' : ''}>精排 {m.rerankerEffective}</Badge>
            {m.overrides && Object.keys(m.overrides).length ? <Badge tone="info">含按请求的参数覆盖</Badge> : null}
          </>
        ) : <span className="text-warning">该报告未记录所用模型（早于模型记录功能），无法判断评测时是否处于降级状态。</span>}
      </div>
      {c && prev ? (
        c.comparable ? (
          <Callout icon={<CheckCircle2 size={15} />}>
            <b>与上次评测（{when}）相比</b>：Hit@1 {delta(final['hit@1'], prev['hit@1'])} · MRR {delta(final.mrr, prev.mrr)} · NDCG@{mainK} {delta(final[`ndcg@${mainK}`], prev.final?.[`ndcg@${mainK}`])} · 拒答 {prev.refusalCorrect ?? '—'}/{prev.refusalTotal ?? '—'} → {report.refusalCorrect}/{report.refusalTotal}
            {c.notes?.length ? c.notes.map((n) => <div key={n} className="xs text-warning mt-4">{n}</div>) : <div className="xs muted mt-4">模型、精排、检索策略、参数覆盖、评估集与 K 值均相同，变化来自语料或入库配置本身。</div>}
          </Callout>
        ) : (
          <Callout tone="warn" icon={<AlertTriangle size={15} />}>
            <div className="col gap-4">
              <b>与上次评测（{when}）不可直接比较</b>
              {c.reasons.map((r) => <div key={r}>· {r}</div>)}
              <div className="xs muted">上次：Hit@1 {prev['hit@1'] != null ? prev['hit@1'].toFixed(3) : '—'} · MRR {prev.mrr != null ? prev.mrr.toFixed(3) : '—'}。指标差异可能来自模型或配置，而不是语料本身。</div>
            </div>
          </Callout>
        )
      ) : null}
    </>
  );
}

export function EvalPage() {
  const [tab, setTab] = useState<'text' | 'multimodal' | 'training'>('text');
  return (
    <div className="page col gap-16 fade-in">
      <Tabs value={tab} onChange={setTab} items={[{ value: 'text', label: '文本检索评测' }, { value: 'multimodal', label: '多模态评测（图片）' }, { value: 'training', label: '难负样本与微调' }]} />
      {tab === 'text' ? <TextEval /> : tab === 'multimodal' ? <MultimodalEval /> : <HardNegativeMining />}
    </div>
  );
}
