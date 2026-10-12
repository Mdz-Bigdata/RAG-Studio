import { useEffect, useRef, useState } from 'react';
import { Download, FileJson, Play } from 'lucide-react';
import { api } from '../api/client';
import type { EvalExample, HardNegativeItem, HardNegativeMiningConfig, HardNegativeReport, SuspectedPositive } from '../api/types';
import { useApp } from '../store/app-store';
import { KbSelect } from './kb-select';
import { Badge, Button, Callout, Empty } from './ui';

const DEFAULT_CONFIG: HardNegativeMiningConfig = {
  candidate_pool: 100, num_negatives: 4, rank_min: 1, rank_max: 100, max_per_doc: 2,
  exclude_same_document: true, use_teacher: true, min_positive_score: 0.5,
  max_negative_score: 0.8, min_negative_score: 0, synthetic_negatives: 0, absolute_margin: 0.1, relative_margin: 0.05,
  dev_fraction: 0.2, seed: 42,
};

const SKIP_REASON: Record<string, string> = {
  empty_question: '问题为空或没有可用文字',
  held_out_question: '与保留评测问题重合，已排除',
  expected_refusal: '预期拒答的问题不用于正负例训练',
  precise_labels_required: '需要明确块 ID 或答案关键词，单独来源标注不够精确',
  no_positive_in_active_index: '现行索引中没有匹配正例，请检查标注或文档时效',
  low_positive_score: '正例教师分低于门槛，请检查正例与分数阈值',
  no_safe_negatives: '排除相关内容或教师过滤后，没有可用负例',
};

const REJECT_REASON: Record<string, string> = {
  contains_answer_keyword: '含答案关键词',
  near_copy: '与已有正负例近似重复',
  teacher_too_close: '教师分贴近正例',
  easy: '教师分过低',
  teacher_unavailable: '教师不可用，无法打分',
  generation_failed: '生成失败',
};

function parseAnnotations(text: string): EvalExample[] {
  return text.split('\n').map((line) => line.trim()).filter(Boolean).map((line, i) => {
    let value: EvalExample;
    try { value = JSON.parse(line) as EvalExample; } catch { throw new Error(`第 ${i + 1} 行不是合法 JSON`); }
    if (!value || typeof value.question !== 'string' || !value.question.trim()) throw new Error(`第 ${i + 1} 行缺少非空 question`);
    return value;
  });
}

function download(name: string, content: string, type = 'application/x-ndjson') {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function jsonl(items: unknown[]): string {
  return items.map((item) => JSON.stringify(item)).join('\n') + (items.length ? '\n' : '');
}

function pairs(items: HardNegativeItem[]) {
  return items.flatMap((item) => [
    ...item.rerank_pos.map((passage) => ({ query: item.query, passage, label: 1 })),
    ...item.rerank_neg.map((passage) => ({ query: item.query, passage, label: 0 })),
  ]);
}

function Suspected({ items, total }: { items?: SuspectedPositive[]; total?: number }) {
  if (!items?.length) return null;
  return (
    <div className="col gap-8">
      <b className="small">疑似漏标正例（{total ?? items.length} 条，教师分贴近或超过正例，未作负例）</b>
      {items.map((s) => <div key={s.chunk_id} className="small"><div className="xs faint mono">{s.chunk_id} · 排名 {s.rank} · 教师 {s.score.toFixed(3)}</div><p style={{ whiteSpace: 'pre-wrap', marginTop: 4 }}>{s.text}</p></div>)}
      <span className="xs muted">确实能回答问题的，把块 ID 补进这条标注的 relevant_chunk_ids 后重新挖掘。</span>
    </div>
  );
}

export function HardNegativeMining() {
  const { currentKb, toast } = useApp();
  const kbId = currentKb?.id;
  const [text, setText] = useState('');
  const [heldOut, setHeldOut] = useState('');
  const [config, setConfig] = useState<HardNegativeMiningConfig>(DEFAULT_CONFIG);
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<HardNegativeReport | null>(null);
  const [approved, setApproved] = useState<Set<number>>(new Set());
  const [excluded, setExcluded] = useState<Set<number>>(new Set());
  // 逐条确认过的改写负例，键为“样本组序号:改写负例序号”
  const [syntheticOk, setSyntheticOk] = useState<Set<string>>(new Set());
  const requestSeq = useRef(0);

  useEffect(() => {
    requestSeq.current += 1;
    setRunning(false);
    setReport(null);
    setApproved(new Set());
    setExcluded(new Set());
    setSyntheticOk(new Set());
    return () => { requestSeq.current += 1; };
  }, [kbId]);

  const invalidate = () => { setReport(null); setApproved(new Set()); setExcluded(new Set()); setSyntheticOk(new Set()); };
  const updateConfig = <K extends keyof HardNegativeMiningConfig>(key: K, value: HardNegativeMiningConfig[K]) => {
    setConfig((prev) => {
      const next = { ...prev, [key]: value };
      if (key === 'candidate_pool' && typeof value === 'number' && Number.isFinite(value)) {
        next.rank_max = Math.min(prev.rank_max, value);
        next.rank_min = Math.min(prev.rank_min, next.rank_max);
      }
      return next;
    });
    invalidate();
  };
  const count = (() => { try { return parseAnnotations(text).length; } catch { return -1; } })();
  const run = async () => {
    if (!kbId) return;
    let examples: EvalExample[], reserved: EvalExample[];
    try {
      examples = parseAnnotations(text);
      reserved = parseAnnotations(heldOut);
      if (!examples.length || examples.length > 500 || reserved.length > 500) throw new Error('训练标注须为 1–500 条；保留评测问题最多 500 条');
    } catch (e) { toast(e instanceof Error ? e.message : String(e), 'bad'); return; }
    const seq = ++requestSeq.current;
    setRunning(true);
    invalidate();
    try {
      const result = await api.eval.hardNegatives(kbId, { examples, heldOut: reserved, config });
      if (requestSeq.current !== seq) return;
      setReport(result);
      toast(`挖掘完成：${result.stats.items} 组样本，${result.stats.review_required} 组待复核`, 'ok');
    } catch (e) {
      if (requestSeq.current === seq) toast(e instanceof Error ? e.message : String(e), 'bad');
    } finally { if (requestSeq.current === seq) setRunning(false); }
  };

  // 确认过的改写负例并入该组负例；缺教师分时整组不带分数，避免与文本错位
  const withSynthetic = (item: HardNegativeItem, i: number): HardNegativeItem => {
    const extra = (item.synthetic_neg ?? []).filter((_, j) => syntheticOk.has(`${i}:${j}`));
    if (!extra.length) return item;
    const scored = item.pos_scores.length > 0 && extra.every((s) => s.score != null);
    return {
      ...item, neg: [...item.neg, ...extra.map((s) => s.text)], rerank_neg: [...item.rerank_neg, ...extra.map((s) => s.rerank_text)],
      pos_scores: scored ? item.pos_scores : [], neg_scores: scored ? [...item.neg_scores, ...extra.map((s) => s.score as number)] : [],
    };
  };
  const eligible = report?.items.map(withSynthetic).filter((item, i) => item.neg.length > 0 && !excluded.has(i) && (!item.requires_review || approved.has(i))) ?? [];
  const forSplit = (split: 'train' | 'dev') => eligible.filter((item) => item.split === split);
  const exportEmbedding = (split: 'train' | 'dev') => {
    // 教师分通过过滤的样本带上 pos_scores / neg_scores，训练时可用 --distill-weight 蒸馏
    download(`${split}.jsonl`, jsonl(forSplit(split).map(({ query, pos, neg, pos_scores, neg_scores }) => (
      pos_scores.length ? { query, pos, neg, pos_scores, neg_scores } : { query, pos, neg }))));
  };
  const exportReranker = (split: 'train' | 'dev') => download(`reranker.${split}.jsonl`, jsonl(pairs(forSplit(split))));
  const exportAudit = () => {
    if (!report) return;
    download('hard-negative-review.json', JSON.stringify({ ...report, approved_item_indices: [...approved].sort((a, b) => a - b), excluded_item_indices: [...excluded].sort((a, b) => a - b), approved_synthetic: [...syntheticOk].sort(), export_eligible_queries: eligible.map((item) => item.query) }, null, 2), 'application/json');
  };

  const numberField = (key: keyof HardNegativeMiningConfig, label: string, min: number, max: number, step = 1) => (
    <label className="col gap-4 small" key={key}>
      <span className="muted">{label}</span>
      <input type="number" min={min} max={max} step={step} value={Number(config[key])} disabled={running}
        onChange={(e) => updateConfig(key, e.target.valueAsNumber)} />
    </label>
  );

  return (
    <div className="col gap-16">
      <Callout>从已标注的正例找出“很像答案、实际不回答问题”的候选，用于嵌入模型或重排模型微调。教师模型过滤只提供相关性判断，<b>不能替代人工黄金标注</b>；没有可靠教师时逐组复核后才能导出。</Callout>
      <div className="grid grid-2">
        <div className="card">
          <div className="card-header"><h3><FileJson size={15} />训练标注（JSONL）</h3><KbSelect /></div>
          <div className="card-body col gap-12">
            <textarea aria-label="训练标注 JSONL" className="mono" rows={9} style={{ fontSize: 12 }} value={text} disabled={running}
              onChange={(e) => { setText(e.target.value); invalidate(); }} spellCheck={false}
              placeholder={'{"question":"年假可以跨年使用吗","relevant_chunk_ids":["文档ID:p0003"],"answer_keywords":["次年 3 月 31 日"],"category":"制度"}\n推荐标注包含答案的 child 块 ID；父块标注可补充答案关键词定位子块。'} />
            <span className="xs faint">{count < 0 ? '存在格式错误' : `${count} 条训练标注`} · 支持 question / relevant_chunk_ids / relevant_doc_ids / relevant_sources / answer_keywords；来源或关键词标注需要复核。</span>
            <label className="col gap-6 small"><span className="muted">保留评测问题（JSONL，可选）</span>
              <textarea className="mono" rows={4} style={{ fontSize: 12 }} value={heldOut} disabled={running}
                onChange={(e) => { setHeldOut(e.target.value); invalidate(); }} spellCheck={false} placeholder={'{"question":"独立评测集里的问题"}'} />
            </label>
            <span className="xs faint">与保留问题重合的训练问题会被排除。此保护只覆盖问题文本，需另行留出同文档、同主题或改写问题组成的独立测试集。</span>
          </div>
        </div>
        <div className="card">
          <div className="card-header"><h3>挖掘参数</h3><Badge>当前知识库</Badge></div>
          <div className="card-body col gap-12">
            <div className="grid grid-2">
              {numberField('candidate_pool', '每路候选数', 10, 1000)}
              {numberField('num_negatives', '每组负例数', 1, 16)}
              {numberField('dev_fraction', '开发集比例', 0, 0.5, 0.05)}
              {numberField('seed', '随机种子', 0, 2147483647)}
              {numberField('synthetic_negatives', '大模型改写负例数（0 关闭）', 0, 4)}
            </div>
            {config.synthetic_negatives > 0 ? <span className="xs faint">开启后每个问题调用一次辅助文本模型，问题和正例原文会发送给它；生成结果逐条确认后才导出。</span> : null}
            <label className="row gap-8 small"><input type="checkbox" checked={config.exclude_same_document} disabled={running} onChange={(e) => updateConfig('exclude_same_document', e.target.checked)} />排除正例所在文档的候选</label>
            <label className="row gap-8 small"><input type="checkbox" checked={config.use_teacher} disabled={running} onChange={(e) => updateConfig('use_teacher', e.target.checked)} />使用当前重排模型过滤伪负例</label>
            <details>
              <summary className="small muted">教师阈值与候选范围</summary>
              <div className="grid grid-2 mt-8">
                {numberField('min_positive_score', '正例最低教师分', 0, 1, 0.05)}
                {numberField('max_negative_score', '负例最高教师分', 0, 1, 0.05)}
                {numberField('min_negative_score', '负例最低教师分（滤掉简单负例）', 0, 1, 0.05)}
                {numberField('absolute_margin', '正负分数绝对间隔', 0, 1, 0.05)}
                {numberField('relative_margin', '正负分数相对间隔', 0, 1, 0.05)}
                {numberField('rank_min', '候选起始排名', 1, config.rank_max)}
                {numberField('rank_max', '候选结束排名', config.rank_min, config.candidate_pool)}
                {numberField('max_per_doc', '同文档最多负例数', 1, 16)}
              </div>
            </details>
            <span className="xs faint">阈值随教师模型校准；候选不足会如实报告，避免用弱候选凑数。</span>
            <Button variant="primary" loading={running} disabled={!kbId || count <= 0 || count > 500} onClick={run}><Play size={13} />挖掘并预览</Button>
          </div>
        </div>
      </div>

      {!report ? <div className="card"><Empty icon={<FileJson size={36} />} title="还没有难负样本" desc="填写训练标注并运行挖掘，预览正负例后导出微调数据。" /></div> : (
        <>
          <div className="card stat-strip">
            <div className="stat"><div className="label">样本组</div><div className="value">{report.stats.items}</div><div className="sub">{report.stats.queries} 个问题 · {report.stats.negatives} 个负例{report.stats.synthetic_negatives ? ` · 改写 ${report.stats.synthetic_negatives}` : ''}</div></div>
            <div className="stat"><div className="label">训练 / 开发</div><div className="value">{report.stats.train} / {report.stats.dev}</div><div className="sub">按问题分组切分</div></div>
            <div className="stat"><div className="label">需人工复核</div><div className="value">{report.stats.review_required}</div><div className="sub">已确认 {approved.size} 组</div></div>
            <div className="stat"><div className="label">负例不足 / 跳过</div><div className="value">{report.stats.shortages} / {report.skipped.length}</div><div className="sub">疑似正例 {report.stats.suspected_positives ?? 0} · 简单负例 {report.stats.easy_filtered ?? 0}</div></div>
          </div>
          {report.warnings.length ? <Callout tone="warn"><div className="col gap-4">{report.warnings.map((warning, i) => <span key={i}>{warning}</span>)}</div></Callout> : null}
          <details className="card pad"><summary className="small muted">所用模型与挖掘参数</summary><pre className="mono xs" style={{ overflowX: 'auto' }}>{JSON.stringify({ models: report.models, runtimeModels: report.runtimeModels, config: report.config }, null, 2)}</pre></details>
          <div className="card">
            <div className="card-header"><h3>逐组复核</h3><span className="xs faint">可导出 {eligible.length} / {report.items.length} 组 · 展开查看全部正负例</span></div>
            <div className="card-body col gap-12">
              {report.items.map((item, i) => (
                <details key={`${item.query}-${i}`} className="card pad">
                  <summary className="small"><b>{item.query}</b> <Badge>{item.split}</Badge> <Badge tone={excluded.has(i) || (item.requires_review && !approved.has(i)) ? 'warn' : 'ok'}>{excluded.has(i) ? '已排除' : item.requires_review ? approved.has(i) ? '已复核' : '待复核' : '教师过滤通过'}</Badge></summary>
                  <div className="col gap-12 mt-8">
                    <div className="xs muted">{item.category} · 正例 {item.pos.length} 条 · 负例 {item.neg.length} 条{item.shortage ? ` · 缺少 ${item.shortage} 条负例` : ''}</div>
                    <div className="grid grid-2">
                      <div className="col gap-8"><b className="small">正例</b>{item.rerank_pos.map((passage, j) => <div key={j} className="small"><div className="xs faint mono">{item.positive_chunk_ids[j]}{item.pos_scores[j] != null ? ` · 教师 ${item.pos_scores[j].toFixed(3)}` : ''}</div><p style={{ whiteSpace: 'pre-wrap', marginTop: 4 }}>{passage}</p></div>)}</div>
                      <div className="col gap-8"><b className="small">难负例</b>{item.rerank_neg.map((passage, j) => <div key={j} className="small"><div className="xs faint mono">{item.negative_chunk_ids[j]} · 排名 {item.negative_ranks[j]} · {item.negative_routes[j]?.join(' / ')}{item.neg_scores[j] != null ? ` · 教师 ${item.neg_scores[j].toFixed(3)}` : ''}</div><p style={{ whiteSpace: 'pre-wrap', marginTop: 4 }}>{passage}</p></div>)}</div>
                    </div>
                    {item.synthetic_neg?.length ? (
                      <div className="col gap-8">
                        <b className="small">大模型改写的负例（非知识库原文，逐条确认后才导出）</b>
                        {item.synthetic_neg.map((s, j) => (
                          <label key={j} className="row gap-8 small" style={{ alignItems: 'flex-start' }}>
                            <input type="checkbox" aria-label={`确认第 ${j + 1} 条改写负例`} checked={syntheticOk.has(`${i}:${j}`)} onChange={(e) => {
                              const checked = e.target.checked;
                              setSyntheticOk((prev) => { const next = new Set(prev); if (checked) next.add(`${i}:${j}`); else next.delete(`${i}:${j}`); return next; });
                            }} />
                            <span><span className="xs faint mono">改写自 {s.source_chunk_id}{s.score != null ? ` · 教师 ${s.score.toFixed(3)}` : ' · 未打分'}</span><span style={{ display: 'block', whiteSpace: 'pre-wrap', marginTop: 4 }}>{s.rerank_text}</span></span>
                          </label>
                        ))}
                        <span className="xs muted">勾选表示已确认这段不回答问题，也不是同一事实的错误版本。</span>
                      </div>
                    ) : null}
                    {Object.keys(item.synthetic_rejected ?? {}).length ? <span className="xs muted">被过滤的改写结果：{Object.entries(item.synthetic_rejected ?? {}).map(([reason, n]) => `${REJECT_REASON[reason] ?? reason} ${n}`).join(' · ')}</span> : null}
                    <Suspected items={item.suspected_positives} total={item.suspected_positive_count} />
                    <label className="row gap-8 small"><input type="checkbox" checked={!excluded.has(i) && (!item.requires_review || approved.has(i))} onChange={(e) => {
                      const checked = e.target.checked;
                      if (item.requires_review) setApproved((prev) => { const next = new Set(prev); if (checked) next.add(i); else next.delete(i); return next; });
                      setExcluded((prev) => { const next = new Set(prev); if (checked) next.delete(i); else next.add(i); return next; });
                    }} />{item.requires_review ? '已确认正例回答问题、全部负例不回答问题，允许导出这一组' : '保留这一组；发现误标时取消勾选，从训练和开发集导出中排除'}</label>
                  </div>
                </details>
              ))}
              {!report.items.length ? <span className="small muted">本次没有可用样本，请检查正例标注、候选范围与教师阈值。</span> : null}
            </div>
          </div>
          {report.skipped.length ? <details className="card pad"><summary className="small muted">跳过的 {report.skipped.length} 个问题与原因</summary><div className="col gap-8 mt-8">{report.skipped.map((item, i) => <div className="small" key={i}><b>{item.question}</b><div className="xs muted">{SKIP_REASON[item.reason] ?? item.reason}{item.easy_filtered ? ` · 滤掉 ${item.easy_filtered} 条简单负例` : ''}</div><Suspected items={item.suspected_positives} /></div>)}</div></details> : null}
          <div className="card pad col gap-12">
            <b className="small">导出微调数据</b>
            <div className="row gap-8 wrap">
              {(['train', 'dev'] as const).map((split) => <Button key={`emb-${split}`} size="sm" disabled={!forSplit(split).length} onClick={() => exportEmbedding(split)}><Download size={12} />嵌入 {split}.jsonl（{forSplit(split).length}）</Button>)}
              {(['train', 'dev'] as const).map((split) => <Button key={`rr-${split}`} size="sm" disabled={!forSplit(split).length} onClick={() => exportReranker(split)}><Download size={12} />重排 {split}.jsonl（{pairs(forSplit(split)).length} 对）</Button>)}
              <Button size="sm" onClick={exportAudit}><Download size={12} />复核报告 JSON</Button>
            </div>
            <span className="xs muted">嵌入文件使用 FlagEmbedding 的 query / pos / neg 格式，教师过滤通过的样本附带 pos_scores / neg_scores；重排文件为 query / passage / label。待复核样本只有逐组确认后才会进入训练文件，改写负例只有逐条勾选后才会并入。</span>
            <Callout>微调完成后，用独立测试集对比 Recall@K、MRR、NDCG 和拒答率。更换嵌入模型需重建向量索引；更换重排模型需重新校准分数阈值。这里生成并导出数据，训练由单独的训练命令执行。</Callout>
          </div>
        </>
      )}
    </div>
  );
}
