import { useEffect, useState } from 'react';
import { Clock, Filter, FlaskConical, GitBranch, Layers, Network, Search, Sparkles, Zap } from 'lucide-react';
import { api } from '../api/client';
import type { MetadataFacets, MetadataField, RetrievalMode, RetrievalOverrides, RetrievalResponse, ScoredItem } from '../api/types';
import { ContextCard } from '../components/context-card';
import { KbSelect } from '../components/kb-select';
import { StrategyPanel } from '../components/strategy-panel';
import { Badge, Button, Callout, Empty, ScoreBar, Tabs } from '../components/ui';
import { formatScore, routeKind, routeLabel } from '../lib/format';
import { useApp } from '../store/app-store';

const EXAMPLES = ['K8s Ingress 502 怎么排查', '订单 SO-8817 的状态', '出差住宿标准能不能超额', '远程员工的通讯补贴和本地通勤员工的交通补贴有什么不同', '怎么退货'];
const META_FIELDS: [MetadataField, string][] = [['department', '部门'], ['tags', '标签'], ['year', '年份'], ['file_format', '文件类型']];
const KIND_COLOR: Record<string, string> = { dense: 'var(--c-dense)', sparse: 'var(--c-sparse)', hyde: 'var(--c-hyde)', other: 'var(--c-fused)' };

function ItemList({ items, color, highlight }: { items: ScoredItem[]; color: string; highlight?: Set<string> }) {
  const max = Math.max(...items.map((i) => i.score), 1e-9);
  if (items.length === 0) return <div className="faint small" style={{ padding: 16, textAlign: 'center' }}>无结果</div>;
  return (
    <div className="col gap-6">
      {items.map((it) => (
        <div key={`${it.route}-${it.chunkId}`} className={`chunk ${highlight?.has(it.chunkId) ? 'hl' : ''}`} style={{ padding: '8px 10px' }}>
          <div className="chunk-head" style={{ marginBottom: 4 }}>
            <span className="row gap-6" style={{ minWidth: 0 }}><span className="rank" style={{ color }}>#{it.rank}</span><span className="ellipsis" title={it.breadcrumb}>{it.breadcrumb || it.fileName}</span></span>
            <span className="mono" style={{ color }}>{formatScore(it.score, it.score > 1 ? 2 : 4)}</span>
          </div>
          <ScoreBar value={it.score} max={max} color={color} />
          <div className="chunk-text clamp-3 small" style={{ marginTop: 5 }}>{it.text}</div>
        </div>
      ))}
    </div>
  );
}

export function PlaygroundPage() {
  const { currentKb, toast, identity } = useApp();
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<RetrievalMode>('hybrid');
  const [overrides, setOverrides] = useState<RetrievalOverrides>({});
  const [topK, setTopK] = useState(5);
  const [useTransform, setUseTransform] = useState(true);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<RetrievalResponse | null>(null);
  const [tab, setTab] = useState<'routes' | 'rank' | 'contexts' | 'extra'>('contexts');
  // 手动指定的元数据条件：按字面执行，没填该属性的文档不算匹配
  const [facets, setFacets] = useState<MetadataFacets | null>(null);
  const [picked, setPicked] = useState<Partial<Record<MetadataField, string>>>({});
  const kbId = currentKb?.id;
  const level = identity?.level;
  const department = identity?.department;

  useEffect(() => {
    setPicked({});
    setFacets(null);
    if (!kbId) return;
    let stale = false;
    api.retrieval.facets(kbId, level === undefined ? null : { level, department: department ?? null })
      .then((f) => { if (!stale) setFacets(f); }).catch(() => { if (!stale) setFacets(null); });
    return () => { stale = true; };
  }, [kbId, level, department]);

  const run = async (q = query) => {
    if (!currentKb || !q.trim()) return;
    setQuery(q);
    setLoading(true);
    try {
      const filters = Object.fromEntries(Object.entries(picked).filter(([, v]) => v).map(([k, v]) => [k, [k === 'year' ? Number(v) : v]]));
      setData(await api.retrieval.playground(currentKb.id, { query: q, mode, topK, useTransform, overrides, filters: Object.keys(filters).length ? filters : undefined, user: identity }));
    } catch (e) {
      toast(e instanceof Error ? e.message : String(e), 'bad');
    } finally {
      setLoading(false);
    }
  };

  const finalIds = new Set(data?.contexts.flatMap((c) => [c.chunkId, ...c.matchedChildIds]) ?? []);
  const rerankIds = new Set(data?.reranked.map((r) => r.chunkId) ?? []);
  const funnelMax = Math.max(data?.funnel.recall ?? 1, 1);
  const funnel = data ? [
    { label: '多路召回（合计）', v: data.funnel.recall, c: 'var(--c-dense)' },
    { label: '融合候选', v: data.funnel.fused, c: 'var(--c-fused)' },
    { label: '精排保留', v: data.funnel.reranked, c: 'var(--c-rerank)' },
    { label: '最终上下文', v: data.funnel.contexts, c: 'var(--c-mmr)' },
  ] : [];

  return (
    <div className="layout-playground">
      <div className="col gap-12" style={{ minHeight: 0 }}>
        <div className="card pad">
          <form className="row gap-8" onSubmit={(e) => { e.preventDefault(); run(); }}>
            <div className="flex-1" style={{ position: 'relative' }}>
              <Search size={15} className="faint" style={{ position: 'absolute', left: 11, top: 10 }} />
              <input style={{ paddingLeft: 34 }} placeholder={currentKb ? `在"${currentKb.name}"中调试检索链路…` : '请先选择知识库'} value={query} onChange={(e) => setQuery(e.target.value)} disabled={!currentKb} aria-label="检索问题" />
            </div>
            <Button variant="primary" type="submit" loading={loading} disabled={!currentKb || !query.trim()}><Zap size={14} />检索</Button>
          </form>
          <div className="row gap-8 wrap mt-12">
            <KbSelect />
            <label className="row gap-6 small muted" style={{ whiteSpace: 'nowrap', cursor: 'pointer' }} title="关闭后只用原始问题检索（不改写 / 不扩展 / 不做 HyDE）"><input type="checkbox" checked={useTransform} onChange={(e) => setUseTransform(e.target.checked)} />查询变换</label>
          </div>
          {facets && META_FIELDS.some(([f]) => facets[f].length > 0) ? (
            <div className="row gap-8 wrap mt-8">
              <span className="row gap-4 xs faint" title="手动限定文档范围：先按属性筛出候选，再做检索。不选则由“识别问题里的范围限定”自动判断。"><Filter size={12} />限定范围：</span>
              {META_FIELDS.filter(([f]) => facets[f].length > 0).map(([f, label]) => (
                <select key={f} aria-label={`按${label}过滤`} style={{ width: 'auto', padding: '4px 26px 4px 8px', fontSize: 12 }} value={picked[f] ?? ''} onChange={(e) => setPicked((prev) => ({ ...prev, [f]: e.target.value }))}>
                  <option value="">{label}：不限</option>
                  {facets[f].map((o) => <option key={String(o.value)} value={String(o.value)}>{o.value}（{o.count} 篇）</option>)}
                </select>
              ))}
            </div>
          ) : null}
          <div className="row gap-6 wrap mt-8">
            <span className="xs faint">示例：</span>
            {EXAMPLES.map((q) => <button key={q} className="btn xs chip" disabled={!currentKb} onClick={() => run(q)}>{q}</button>)}
          </div>
        </div>

        <div className="scroll flex-1 col gap-12" style={{ paddingRight: 4 }}>
          {!currentKb ? <div className="card"><Empty icon={<FlaskConical size={40} />} title="请先选择知识库" desc="在「知识库」页面创建并上传文档后，回到这里调试检索链路。" /></div> : !data ? (
            <div className="card"><Empty icon={<FlaskConical size={40} />} title="检索实验室" desc="输入问题后，这里会展示完整的可解释链路：查询层（改写 / Multi-Query / HyDE）→ 各路召回 → 融合 → 精排 → MMR → 最终上下文 → 实体织网 → 压缩。右侧可按请求覆盖任意参数做 A/B。" /></div>
          ) : (
            <>
              <div className="grid grid-2">
                <div className="card pad">
                  <div className="card-title"><Sparkles size={14} />查询层 <Badge mono>{data.strategy}</Badge></div>
                  <div className="col gap-4">
                    {data.queries.map((q, i) => <div key={i} className="row gap-6 small"><Badge tone={i === 0 ? 'ok' : 'hyde'}>{i === 0 ? '原始' : `变换 ${i}`}</Badge><span>{q}</span></div>)}
                  </div>
                  {data.hydeDocument ? <div className="small muted mt-12 pre" style={{ borderLeft: '2px solid var(--c-hyde)', paddingLeft: 10 }}><b style={{ color: 'var(--c-hyde)' }}>HyDE 假设文档：</b>{data.hydeDocument}</div> : null}
                  {data.metadataFilter ? (
                    <div className="row gap-6 wrap small mt-12">
                      <Badge tone={data.metadataFilter.applied ? 'ok' : ''}><Filter size={11} /> 元数据过滤{data.metadataFilter.applied ? '' : '（未生效）'}</Badge>
                      {META_FIELDS.filter(([f]) => data.metadataFilter?.conditions[f]?.length).map(([f, label]) => <Badge key={f} mono>{label} = {data.metadataFilter?.conditions[f]?.join(' / ')}</Badge>)}
                      <span className="xs faint">{data.metadataFilter.source === 'llm' ? '模型识别' : data.metadataFilter.source === 'rule' ? '规则识别' : '沿用首次检索'} · 文档 {data.metadataFilter.docs_before} → {data.metadataFilter.docs_after} 篇</span>
                    </div>
                  ) : null}
                  {data.notes.length ? <div className="col gap-4 mt-12">{data.notes.map((n, i) => <div key={i} className="xs faint">· {n}</div>)}</div> : null}
                </div>
                <div className="card pad">
                  <div className="card-title"><Layers size={14} />漏斗与耗时</div>
                  <div className="funnel">
                    {funnel.map((f) => (
                      <div className="funnel-row" key={f.label}><span className="muted">{f.label}</span><div className="bar" style={{ width: `${Math.max(3, (f.v / funnelMax) * 100)}%`, ['--p' as any]: f.c }} /><span className="mono" style={{ textAlign: 'right' }}>{f.v}</span></div>
                    ))}
                  </div>
                  <div className="row gap-6 wrap mt-12">
                    <Clock size={12} className="faint" />
                    {Object.entries(data.timingsMs).map(([k, v]) => <Badge key={k} mono tone={k === 'total' ? 'info' : ''}>{k} {v}ms</Badge>)}
                  </div>
                  {data.compression.tokens_before ? <div className="xs muted mt-8">上下文压缩（{data.compression.mode}）：{data.compression.tokens_before} → {data.compression.tokens_after} tok</div> : null}
                </div>
              </div>

              <div className="row between wrap">
                <Tabs value={tab} onChange={setTab} items={[
                  { value: 'contexts', label: `最终上下文（${data.contexts.length}）` },
                  { value: 'routes', label: `各路召回（${Object.keys(data.routes).length} 路）` },
                  { value: 'rank', label: '融合与精排' },
                  { value: 'extra', label: `织网 / 多跳（${data.weaved.length + data.hops.length}）` },
                ]} />
                <span className="xs faint">高亮 = 进入下一阶段</span>
              </div>

              {tab === 'contexts' ? (
                data.contexts.length === 0 ? <Callout tone="warn">没有上下文进入 LLM：{data.notes.at(-1) ?? '库中可能没有相关内容'}。这种情况下回答层会按协议拒答。</Callout> : (
                  <div className="col gap-8">
                    <div className="xs faint">以下为送入 LLM 的顺序（Lost-in-the-Middle 排列：最相关的放首尾）。</div>
                    {data.contexts.map((c, i) => <ContextCard key={c.chunkId} ctx={c} index={i + 1} />)}
                  </div>
                )
              ) : null}

              {tab === 'routes' ? (
                <div className="grid" style={{ gridTemplateColumns: `repeat(${Math.min(3, Math.max(1, Object.keys(data.routes).length))}, minmax(0, 1fr))` }}>
                  {Object.entries(data.routes).map(([route, items]) => {
                    const kind = routeKind(route);
                    return (
                      <div key={route} className="card pad">
                        <div className="row between mb-8"><Badge tone={kind === 'other' ? 'fused' : kind}>{routeLabel(route)}</Badge><span className="xs faint">{items.length} 条</span></div>
                        <ItemList items={items.slice(0, 10)} color={KIND_COLOR[kind]} highlight={rerankIds} />
                      </div>
                    );
                  })}
                </div>
              ) : null}

              {tab === 'rank' ? (
                <div className="grid" style={{ gridTemplateColumns: `repeat(${data.diversified.length ? 3 : 2}, minmax(0, 1fr))` }}>
                  <div className="card pad"><div className="row between mb-8"><Badge tone="fused">融合候选</Badge><span className="xs faint">{data.fused.length} 条</span></div><ItemList items={data.fused.slice(0, 15)} color="var(--c-fused)" highlight={rerankIds} /></div>
                  <div className="card pad"><div className="row between mb-8"><Badge tone="rerank">精排</Badge><span className="xs faint">{data.reranked.length} 条</span></div><ItemList items={data.reranked} color="var(--c-rerank)" highlight={finalIds} /></div>
                  {data.diversified.length ? <div className="card pad"><div className="row between mb-8"><Badge tone="mmr">MMR 多样化</Badge><span className="xs faint">{data.diversified.length} 条</span></div><ItemList items={data.diversified} color="var(--c-mmr)" highlight={finalIds} /></div> : null}
                </div>
              ) : null}

              {tab === 'extra' ? (
                <div className="grid grid-2">
                  <div className="card pad">
                    <div className="card-title"><Network size={14} />UnWeaver 实体织入</div>
                    {data.weaved.length === 0 ? <div className="small faint">本次未触发织入。当检索到的块提到某个术语、而定义该术语的块没有被召回时，定义块会被自动补进上下文。</div> : data.weaved.map((w, i) => (
                      <div key={i} className="chunk mb-8"><div className="strong" style={{ color: 'var(--c-weaved)' }}>{w.entity}</div><div className="small muted">{w.definition}</div><div className="xs faint mt-8">由 {w.triggered_by} 触发 → 织入 {w.source_chunk_id}</div></div>
                    ))}
                  </div>
                  <div className="card pad">
                    <div className="card-title"><GitBranch size={14} />多跳 / Agentic 轨迹</div>
                    {data.hops.length === 0 ? <div className="small faint">单次检索。切换到"多跳检索"或"Agentic RAG"模式可看到逐跳查询与补检过程。</div> : data.hops.map((h) => (
                      <div key={h.hop} className="chunk mb-8"><div className="row gap-6"><Badge tone="info" mono>第 {h.hop} 跳</Badge><span className="small">{h.query}</span></div><div className="xs faint mt-8">{h.note} · 命中 {h.found_chunk_ids.length} 条</div></div>
                    ))}
                  </div>
                </div>
              ) : null}
            </>
          )}
        </div>
      </div>

      <div className="scroll" style={{ minHeight: 0, paddingRight: 4 }}>
        <StrategyPanel mode={mode} onModeChange={setMode} value={overrides} onChange={setOverrides} topK={topK} onTopKChange={setTopK} />
      </div>
    </div>
  );
}
