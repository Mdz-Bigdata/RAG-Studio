// 多模态评测：文搜图 / 图搜图 / 图搜文 Hit@K·MRR、文搜图按召回路消融、回答附图的精确率 / 召回率 / 负例不附图率。
import { useEffect, useState } from 'react';
import { CheckCircle2, FileJson, Images, Play, Sparkles, XCircle } from 'lucide-react';
import { api } from '../api/client';
import type { MultimodalEvalReport } from '../api/types';
import { IMAGE_ROUTE } from './image-card';
import { KbSelect } from './kb-select';
import { Badge, Button, Callout, Empty, ScoreBar } from './ui';
import { formatDate, pct } from '../lib/format';
import { useApp } from '../store/app-store';

const TASK_LABEL: Record<string, string> = { text2image: '文搜图', image2image: '图搜图', image2text: '图搜文', answer: '回答附图' };
const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));

function parseJsonl(text: string): Record<string, unknown>[] {
  return text.split('\n').map((l) => l.trim()).filter(Boolean).map((l, i) => {
    try { return JSON.parse(l); } catch { throw new Error(`第 ${i + 1} 行不是合法 JSON`); }
  });
}

export function MultimodalEval() {
  const { currentKb, toast } = useApp();
  const kbId = currentKb?.id;
  const [text, setText] = useState('');
  const [name, setName] = useState('');
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<MultimodalEvalReport | null>(null);
  const [history, setHistory] = useState<MultimodalEvalReport[]>([]);

  useEffect(() => {
    setReport(null);
    if (kbId) api.eval.multimodalReports(kbId).then((r) => { setHistory(r); setReport(r[0] ?? null); }).catch(() => setHistory([]));
  }, [kbId]);

  const loadSample = async () => {
    if (!kbId) return;
    try {
      const s = await api.eval.multimodalSample(kbId);
      if (!s.length) { toast('这个知识库还没有图片，先上传含图片的文档', 'warn'); return; }
      setText(s.map((e) => JSON.stringify(e)).join('\n'));
      toast(`已从图片库生成 ${s.length} 条样例`, 'ok');
    } catch (e) { toast(errMsg(e), 'bad'); }
  };
  const run = async () => {
    if (!kbId) return;
    let examples: Record<string, unknown>[];
    try { examples = parseJsonl(text); } catch (e) { toast(errMsg(e), 'bad'); return; }
    setRunning(true);
    try {
      const r = await api.eval.multimodalRun(kbId, { examples: examples.length ? examples : undefined, auto: examples.length === 0, name: name || undefined });
      setReport(r);
      setHistory((h) => [r, ...h]);
      toast(`评测完成：${r.n} 条样例，用时 ${r.seconds.toFixed(1)}s`, 'ok');
    } catch (e) { toast(errMsg(e), 'bad'); } finally { setRunning(false); }
  };
  const count = (() => { try { return parseJsonl(text).length; } catch { return -1; } })();
  const failed = report?.examples.filter((e) => !e.ok) ?? [];

  return (
    <div className="col gap-16">
      <div className="grid grid-2">
        <div className="card">
          <div className="card-header"><h3><FileJson size={15} />多模态样例（JSONL）</h3><div className="row gap-8"><KbSelect /><Button size="sm" onClick={loadSample} disabled={!kbId}><Sparkles size={12} />从图片库生成</Button></div></div>
          <div className="card-body col gap-12">
            <textarea className="mono" rows={11} style={{ fontSize: 12 }} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false}
              placeholder={'{"task": "text2image", "question": "各区域销售额柱状图", "relevant_images": ["图1 2025年Q3各区域销售额"]}\n{"task": "image2text", "query_image": "kb:报告.pdf#p3", "relevant_keywords": ["PUMP-07"]}\n{"task": "answer", "question": "年假怎么休", "expect_no_image": true}'} />
            <div className="row between wrap gap-12">
              <span className="xs faint">{count < 0 ? <span className="text-danger">存在非法 JSON 行</span> : count ? `${count} 条样例` : '留空直接运行 = 用图片库自动生成的样例'} · task：text2image / image2image / image2text / answer</span>
              <div className="row gap-8"><input style={{ width: 160 }} placeholder="本次评测备注（可选）" value={name} onChange={(e) => setName(e.target.value)} /><Button variant="primary" loading={running} disabled={!kbId || count < 0} onClick={run}><Play size={13} />运行评测</Button></div>
            </div>
          </div>
        </div>
        <div className="card pad col gap-12">
          <div className="card-title" style={{ marginBottom: 0 }}><Images size={15} />多模态评测看什么</div>
          <Callout>传统 Recall@K 只看文本块，量不出"图找得对不对""回答带的图对不对"。这里分四类任务单独计分。</Callout>
          <ul className="small muted" style={{ paddingLeft: 18, lineHeight: 1.9 }}>
            <li><b className="strong">文搜图 / 图搜图</b>：Hit@K、MRR；文搜图另按召回路消融（图文向量 / 描述向量 / 描述 BM25 / 融合）</li>
            <li><b className="strong">图搜文</b>：期望关键词是否出现在前 K 个正文段落里</li>
            <li><b className="strong">回答附图</b>：附图精确率 / 召回率，以及与图无关的问题<b>不附图</b>的比例</li>
            <li>指认图片：image_id、图注片段，或 <code>文件名#p页码</code>；查询图片可写 <code>kb:指认方式</code>（库里的图缩小 + 压缩后作为查询）</li>
            <li>自动生成的样例是"自检"：能发现解析、索引、阈值上的明显问题，真实效果仍要靠人工整理的问答对</li>
          </ul>
          {history.length ? (
            <div>
              <div className="xs faint mb-8">历史报告</div>
              <div className="col gap-4">{history.slice(0, 5).map((h) => (
                <div key={h.createdAt} className={`list-item ${report?.createdAt === h.createdAt ? 'active' : ''}`} style={{ padding: '6px 10px' }} onClick={() => setReport(h)}>
                  <span className="small flex-1 ellipsis">{h.name || '未命名'} · {formatDate(h.createdAt)}</span>
                  <Badge mono tone="rerank">{h.n} 条 · 通过 {pct(h.examples.filter((e) => e.ok).length / Math.max(1, h.n))}</Badge>
                </div>
              ))}</div>
            </div>
          ) : null}
        </div>
      </div>

      {!report ? (
        <div className="card"><Empty icon={<Images size={40} />} title="还没有多模态评测报告" desc="点「从图片库生成」载入样例（或直接运行），即可得到四类任务的指标与未通过样例。" /></div>
      ) : (
        <>
          <div className="grid grid-4">
            {Object.entries(report.task_metrics).map(([task, m]) => (
              <div key={task} className="card pad col gap-8">
                <div className="row between"><span className="strong small">{TASK_LABEL[task] ?? task}</span><Badge mono>{m.n} 条</Badge></div>
                {Object.entries(m).filter(([k]) => k !== 'n').map(([k, v]) => (
                  <div key={k} className="col gap-4">
                    <div className="row between xs"><span className="muted">{k}</span><span className="mono">{pct(v)}</span></div>
                    <ScoreBar value={v} color={v >= 0.8 ? 'var(--c-fused)' : v >= 0.5 ? 'var(--c-rerank)' : 'var(--danger)'} />
                  </div>
                ))}
              </div>
            ))}
          </div>

          {Object.keys(report.route_metrics).length ? (
            <div className="card">
              <div className="card-header"><h3>文搜图消融：各召回路单独排序</h3><span className="xs faint">哪一路在拖后腿，就先调哪一路</span></div>
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th>召回路</th>{report.ks.map((k) => <th key={k} className="num">hit@{k}</th>)}</tr></thead>
                  <tbody>{Object.entries(report.route_metrics).map(([route, m]) => (
                    <tr key={route}><td><Badge tone={IMAGE_ROUTE[route]?.tone ?? (route === 'fused' ? 'fused' : '')}>{IMAGE_ROUTE[route]?.label ?? (route === 'fused' ? '融合（最终）' : route)}</Badge></td>{report.ks.map((k) => <td key={k} className="num">{pct(m[`hit@${k}`])}</td>)}</tr>
                  ))}</tbody>
                </table>
              </div>
            </div>
          ) : null}

          <div className="card">
            <div className="card-header"><h3>逐条结果</h3><span className="xs faint">{report.embedder?.model ? `图文向量：${String(report.embedder.model)}` : ''} · {report.seconds.toFixed(1)}s</span></div>
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th></th><th>任务</th><th>问题 / 查询</th><th className="num">名次</th><th>前几名</th><th>说明</th></tr></thead>
                <tbody>{[...failed, ...report.examples.filter((e) => e.ok)].map((e, i) => (
                  <tr key={i}>
                    <td>{e.ok ? <CheckCircle2 size={14} className="text-success" /> : <XCircle size={14} className="text-danger" />}</td>
                    <td><Badge>{TASK_LABEL[e.task] ?? e.task}</Badge>{e.category ? <div className="xs faint mt-4">{e.category}</div> : null}</td>
                    <td className="small">{e.question || <span className="faint">（图片）</span>}</td>
                    <td className="num">{e.rank ?? '—'}</td>
                    <td className="xs muted">{e.top.length ? e.top.join('；') : '—'}</td>
                    <td className="xs">{e.note || <span className="faint">通过</span>}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
