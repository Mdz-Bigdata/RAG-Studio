// 检索策略面板：检索模式 + RetrievalConfig 的按请求覆盖（检索实验室与智能问答共用）。
// 未覆盖的项显示服务端生效值（从 /settings 读取）。
import React, { useEffect, useState } from 'react';
import { GitBranch, Layers, Network, Route, Scissors, Shuffle, Sparkles, Zap } from 'lucide-react';
import { api } from '../api/client';
import type { RetrievalMode, RetrievalOverrides, StrategyPreset } from '../api/types';
import { Badge, Toggle } from './ui';

interface Props {
  mode: RetrievalMode;
  onModeChange: (m: RetrievalMode) => void;
  value: RetrievalOverrides;
  onChange: (v: RetrievalOverrides) => void;
  topK?: number;
  onTopKChange?: (k: number) => void;
  compact?: boolean;
}

export const MODE_INFO: Record<RetrievalMode, { label: string; desc: string; icon: React.ReactNode }> = {
  hybrid: { label: '混合检索', desc: '向量 + BM25（+HyDE）→ RRF → 精排 → Small-to-Big', icon: <Layers size={13} /> },
  multihop: { label: '多跳检索', desc: '迭代检索串联信息：检索 → 判断缺什么 → 再检索', icon: <GitBranch size={13} /> },
  agentic: { label: 'Agentic RAG', desc: '模型自主选择策略（精确 / 混合 / 多跳 / 不检索）并反思补检', icon: <Sparkles size={13} /> },
};

const NUM_STYLE: React.CSSProperties = { width: 76, padding: '4px 8px', fontSize: 12 };
const SEL_STYLE: React.CSSProperties = { width: 110, padding: '4px 26px 4px 8px', fontSize: 12 };

export function StrategyPanel({ mode, onModeChange, value, onChange, topK, onTopKChange, compact }: Props) {
  const [eff, setEff] = useState<Record<string, any> | null>(null);
  const [presets, setPresets] = useState<StrategyPreset[]>([]);
  useEffect(() => {
    api.settings.get().then((s) => setEff(s.effective.retrieval)).catch(() => setEff(null));
    api.settings.presets().then((p) => setPresets(p.presets)).catch(() => setPresets([]));
  }, []);

  const get = (path: string[]): any => {
    let o: any = value;
    for (const p of path) o = o?.[p];
    if (o !== undefined) return o;
    let e: any = eff;
    for (const p of path) e = e?.[p];
    return e;
  };
  const set = (path: string[], v: any) => {
    const next: any = structuredClone(value ?? {});
    let o = next;
    for (const p of path.slice(0, -1)) o = o[p] ??= {};
    o[path[path.length - 1]] = v;
    onChange(next);
  };
  const changed = Object.keys(value ?? {}).length > 0;

  // 注意：以下是"渲染函数"而不是内部组件——内部组件每次渲染都会重新挂载，导致数字输入框失焦。
  const num = (path: string[], min: number, max: number, step = 1) => (
    <input type="number" style={NUM_STYLE} value={get(path) ?? ''} min={min} max={max} step={step} onChange={(e) => set(path, e.target.value === '' ? undefined : Number(e.target.value))} />
  );
  const sw = (path: string[]) => <Toggle on={!!get(path)} onChange={(v) => set(path, v)} />;
  const select = (path: string[], options: [string, string][], fallback: string) => (
    <select style={SEL_STYLE} value={get(path) ?? fallback} onChange={(e) => set(path, e.target.value)}>
      {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
    </select>
  );
  const row = (label: string, control: React.ReactNode, hint?: string) => (
    <div key={label} className="field inline setting-row" style={{ padding: '6px 0' }}>
      <label style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 1 }}>
        <span>{label}</span>
        {hint && !compact ? <span className="hint">{hint}</span> : null}
      </label>
      {control}
    </div>
  );
  const section = (icon: React.ReactNode, title: string, tone: string, rows: React.ReactNode[]) => (
    <div className="card" style={{ padding: '10px 12px' }}>
      <div className="row gap-6 mb-8"><Badge tone={tone}>{icon}{title}</Badge></div>
      {rows}
    </div>
  );

  return (
    <div className="col gap-8">
      <div className="card" style={{ padding: '10px 12px' }}>
        <div className="row between mb-8">
          <span className="strong small">检索模式</span>
          {changed ? <button className="btn ghost xs" onClick={() => onChange({})}>重置覆盖</button> : null}
        </div>
        <div className="col gap-4">
          {(Object.keys(MODE_INFO) as RetrievalMode[]).map((m) => (
            <div key={m} className={`list-item ${mode === m ? 'active' : ''}`} style={{ padding: '7px 10px' }} onClick={() => onModeChange(m)}>
              <span style={{ color: mode === m ? 'var(--text-1)' : 'var(--text-3)' }}>{MODE_INFO[m].icon}</span>
              <div className="flex-1">
                <div className="small strong">{MODE_INFO[m].label}</div>
                {!compact ? <div className="xs faint">{MODE_INFO[m].desc}</div> : null}
              </div>
            </div>
          ))}
        </div>
        {presets.length ? (
          <div className="mt-8">
            <div className="xs faint mb-8">场景预设（一键填入下面各项，仅本次请求）</div>
            <div className="row gap-4 wrap">{presets.map((p) => <button key={p.key} className="btn xs chip" title={`${p.combo}\n${p.reason}`} onClick={() => onChange(structuredClone(p.retrieval))}>{p.name}</button>)}</div>
          </div>
        ) : null}
        {onTopKChange ? row('最终上下文数 context_top_n', <input type="number" style={NUM_STYLE} min={1} max={20} value={topK ?? get(['rerank', 'top_k']) ?? 5} onChange={(e) => onTopKChange(Number(e.target.value) || 1)} />, '推荐 3-5，越多越容易 Lost in the Middle') : null}
      </div>

      {section(<Zap size={11} />, '查询层', 'hyde', [
        row('改写 / 压缩 rewrite', sw(['query_transform', 'rewrite']), '口语 → 书面检索表达'),
        row('Multi-Query 数量', num(['query_transform', 'multi_query'], 0, 8), '0 关闭；3-5 个角度分别检索'),
        row('HyDE 假设文档', sw(['query_transform', 'hyde']), '短 query 有效；含精确编号自动跳过'),
        row('多跳分解 decompose', sw(['query_transform', 'decompose'])),
        row('Step-back 抽象', sw(['query_transform', 'step_back'])),
      ])}

      {section(<Layers size={11} />, '召回与融合', 'fused', [
        row('每路召回 top-N', num(['top_n_per_route'], 5, 300), 'recall_top_k，推荐 50-100'),
        row('融合方法', select(['fusion', 'method'], [['rrf', 'RRF'], ['convex', '凸组合']], 'rrf'), 'RRF 只看排名最稳；凸组合需调 α'),
        row('RRF k', num(['fusion', 'rrf_k'], 1, 200), '业界标准 60'),
        row('摘要索引召回', sw(['summary', 'enabled']), '入库时开了摘要索引才有这一路'),
        row('分层召回', sw(['hierarchy', 'enabled']), '入库时开了多粒度分层才有这一路'),
        row('分层路由', select(['hierarchy', 'routing'], [['auto', '按问题类型'], ['all', '章节 + 句子都用']], 'auto'), '概念型 → 章节级；细节型 → 句子级'),
        row('权限过滤', sw(['access', 'enabled']), '按顶栏"当前身份"过滤文档'),
        row('BM25 权重', num(['fusion', 'weight_sparse'], 0, 5, 0.1), '术语密集领域可调高'),
        row('凸组合 α（向量权重）', num(['fusion', 'convex_alpha'], 0, 1, 0.05)),
        row('SQL 结构化检索', sw(['sql', 'enabled']), '第三路：表格按字段过滤 / 比较 / 排序'),
        row('SQL 路权重', num(['sql', 'weight'], 0, 5, 0.1)),
        row('过滤过期文档', sw(['filter_expired']), '按生效 / 失效日期'),
        row('识别问题里的范围限定', select(['metadata_filter', 'mode'], [['off', '关闭'], ['rule', '规则'], ['llm', '模型']], 'llm'), '年份 / 部门 / 标签 / 文件类型 → 召回前先缩小文档范围；规则只认年份和部门名'),
        row('保留未填属性的文档', sw(['metadata_filter', 'include_missing']), '关闭后，没填该属性的文档会被识别出的条件排除'),
      ])}

      {section(<Shuffle size={11} />, '精排与多样化', 'rerank', [
        row('启用 Reranker', sw(['rerank', 'enabled'])),
        row('候选数 candidates', num(['rerank', 'candidates'], 5, 300), '必须 ≥ 50，否则断崖式下跌'),
        row('保留 rerank_top_n', num(['rerank', 'keep_n'], 1, 50), '推荐 10'),
        row('拒答阈值 min_score', <input type="number" style={NUM_STYLE} min={0} max={1} step={0.05} value={get(['rerank', 'min_score']) ?? ''} placeholder="不设" onChange={(e) => set(['rerank', 'min_score'], e.target.value === '' ? null : Number(e.target.value))} />, '低于阈值判定库中无答案'),
        row('相对阈值 min_score_ratio', <input type="number" style={NUM_STYLE} min={0} max={1} step={0.05} value={get(['rerank', 'min_score_ratio']) ?? ''} placeholder="不设" onChange={(e) => set(['rerank', 'min_score_ratio'], e.target.value === '' ? null : Number(e.target.value))} />, '低于 top-1 × ratio 的候选不凑数'),
    row('MMR 多样化', sw(['diversify', 'enabled']), '避免 top-K 都是同一段的切片'),
        row('MMR λ', num(['diversify', 'mmr_lambda'], 0, 1, 0.05), '0.7：相关性为主'),
      ])}

      {section(<Scissors size={11} />, '上下文工程', 'mmr', [
        row('返回父块 Small-to-Big', sw(['granularity', 'return_parents'])),
        row('上下文压缩', sw(['compression', 'enabled'])),
        row('压缩方式', select(['compression', 'mode'], [['rule', '规则'], ['llm', 'LLM 抽取']], 'rule')),
        row('保留比例 ratio', num(['compression', 'ratio'], 0.1, 1, 0.1)),
      ])}

      {section(<Network size={11} />, '织网 · 多跳 · Agentic', 'weaved', [
        row('UnWeaver 实体织入', sw(['graph', 'unweaver_enabled']), '术语定义块自动补齐'),
        row('最多织入块数', num(['graph', 'max_weaved'], 0, 10)),
        row('多跳最大跳数', num(['multihop', 'max_hops'], 1, 5)),
        row('Agentic 最大步数', num(['agentic', 'max_steps'], 1, 6)),
      ])}

      {!compact ? <div className="xs faint row gap-6"><Route size={11} />未修改的项使用"策略配置"页保存的全局值；此处覆盖仅对本次请求生效。</div> : null}
    </div>
  );
}
