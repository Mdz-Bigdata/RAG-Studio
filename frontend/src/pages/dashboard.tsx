import { AlertTriangle, ArrowRight, BarChart3, Cpu, Database, FlaskConical, MessageSquare, RefreshCw, Target, Upload } from 'lucide-react';
import type { PageKey } from '../app';
import type { RuntimeInfo } from '../api/types';
import { Badge, Button, Callout, Empty } from '../components/ui';
import { useApp } from '../store/app-store';

const SOURCE_LABEL: Record<string, string> = { config: '模型配置中心', env: '.env', 'config.yaml': 'config.yaml', fallback: '离线兜底', settings: '策略配置' };

function RuntimeRow({ title, info, fallback }: { title: string; info?: RuntimeInfo; fallback: string }) {
  return (
    <tr>
      <td className="muted" style={{ whiteSpace: 'nowrap' }}>{title}</td>
      <td className="mono" style={{ wordBreak: 'break-all' }}>{info?.model ?? info?.provider ?? '—'}</td>
      <td className="faint xs">{info?.offline ? fallback : `${info?.provider ?? ''} · ${info ? SOURCE_LABEL[info.source] ?? info.source : ''}`}</td>
      <td style={{ textAlign: 'right' }}>{info ? <Badge tone={info.offline ? 'warn' : 'ok'}>{info.offline ? '降级' : '就绪'}</Badge> : null}</td>
    </tr>
  );
}

export function DashboardPage({ go }: { go: (key: PageKey) => void }) {
  const { status, backendOnline, refreshStatus, kbs, setCurrentKbId } = useApp();

  if (!backendOnline) {
    return (
      <div className="page">
        <Callout tone="bad" icon={<AlertTriangle size={16} />}>
          <b>无法连接后端 API。</b>在项目根目录执行 <code>./start.sh</code>，或手动启动 <code>uvicorn app.main:app --port 8000</code> 后重试。
          <div className="mt-8"><Button size="sm" onClick={() => refreshStatus()}><RefreshCw size={13} />重试</Button></div>
        </Callout>
      </div>
    );
  }
  if (!status) return <div className="page grid grid-4">{[0, 1, 2, 3].map((i) => <div key={i} className="skeleton" style={{ height: 88 }} />)}</div>;

  const { runtime: rt, counts, pipeline: p } = status;
  const qt = p.queryTransform as Record<string, any>;
  const ch = p.chunking as Record<string, any>;
  const fu = p.fusion as Record<string, any>;
  const gr = (p.granularity ?? {}) as Record<string, any>;
  const rr = (p.rerank ?? {}) as Record<string, any>;
  const dv = p.diversify as Record<string, any>;
  const cp = p.compression as Record<string, any>;
  const gph = p.graph as Record<string, any>;
  const onOff = (b: unknown) => (b ? '开' : '关');
  const llmNote = rt.llm.offline ? '已开启 · 待文本模型' : null;

  // 六项核心检索能力：名称、它解决什么、当前生效值
  const caps: { name: string; color: string; on: boolean; why: string; value: string; note?: string | null }[] = [
    { name: '向量 + 关键词混合检索', color: 'var(--c-dense)', on: true, why: '语义灵活性与字面精确性互补：同义表达靠向量，编号 / 错误码 / 型号靠 BM25。', value: `dense(${rt.embedding.model}) + BM25(jieba) · 每路 top-${p.funnel.recallTopK}` },
    { name: '查询重写 / 扩展', color: 'var(--c-hyde)', on: !!qt.rewrite || qt.multi_query > 0, why: '把口语提问改写成多个角度、不同措辞的子问题检索同一个库；原始问题始终保留。', value: `rewrite ${onOff(qt.rewrite)} · multi-query ${qt.multi_query} · 分解 ${onOff(qt.decompose)}`, note: llmNote },
    { name: '假设文档嵌入 HyDE', color: 'var(--c-hyde)', on: !!qt.hyde, why: '先生成一段假设答案，用它的向量去检索；短问题有效，含编号 / 数字时自动跳过。', value: `hyde ${onOff(qt.hyde)} · 仅 ≤ ${qt.hyde_max_query_chars} 字 · 精确查询跳过 ${onOff(qt.skip_hyde_on_precise_tokens)}`, note: llmNote },
    { name: '不同粒度检索', color: 'var(--c-mmr)', on: !!gr.return_parents, why: '小块检索保证精度，命中后返回所属大块保证上下文完整（Small-to-Big）。', value: `child ${ch.child_tokens} tok → parent ${ch.parent_max_tokens} tok · 相邻合并 ${onOff(gr.auto_merge_adjacent)}` },
    { name: 'RRF 融合', color: 'var(--c-fused)', on: true, why: '不比较分数只比较排名，绕开向量与 BM25 量纲不同的问题。', value: fu.method === 'rrf' ? `rrf k=${fu.rrf_k} · dense ${fu.weight_dense} : sparse ${fu.weight_sparse}` : `凸组合 α=${fu.convex_alpha}` },
    { name: 'ReRanker 精排', color: 'var(--c-rerank)', on: !!rr.enabled && rt.rerank.provider !== 'none', why: '对融合候选逐对打分，把真正相关的浮上来；候选必须 ≥ 50。', value: `${rt.rerank.provider}${rt.rerank.model ? `(${rt.rerank.model})` : ''} · ${p.funnel.candidates} → ${p.funnel.rerankTopN} → ${p.funnel.contextTopN}`, note: rt.rerank.offline ? '词法兜底 · 建议配置重排模型' : null },
  ];

  const offline: { t: string; d: string; c: string }[] = [
    { t: '解析', d: rt.mineruAvailable ? `MinerU 4.x${rt.mineruMode === 'cloud' ? ' 云端 API' : ''} · ${rt.mineruTier}${rt.popoEnabled ? ' + Popo 文档树' : ''}` : 'Markdown / Excel 行级 / pdfplumber / docx / pptx', c: 'var(--info)' },
    { t: '清洗', d: '页眉页脚 · PDF 断行 · 乱码 · 近重复', c: 'var(--info)' },
    { t: '语义感知切片', d: `${ch.strategy} · parent ${ch.parent_max_tokens} / child ${ch.child_tokens} tok · 重叠 ${Math.round((ch.child_overlap_ratio ?? 0) * 100)}%`, c: 'var(--c-mmr)' },
    { t: '索引', d: `向量(${rt.vectorBackend}) + BM25 + 实体图`, c: 'var(--c-dense)' },
  ];
  const online: { t: string; d: string; c: string }[] = [
    { t: '查询层', d: `改写 ${onOff(qt.rewrite)} · Multi-Query ${qt.multi_query} · HyDE ${onOff(qt.hyde)}`, c: 'var(--c-hyde)' },
    { t: '多路召回', d: `向量 + BM25 + HyDE，每路 top-${p.funnel.recallTopK}`, c: 'var(--c-dense)' },
    { t: '融合', d: fu.method === 'rrf' ? `RRF k=${fu.rrf_k}` : `凸组合 α=${fu.convex_alpha}`, c: 'var(--c-fused)' },
    { t: '精排', d: `${rt.rerank.provider} · ${p.funnel.candidates} → ${p.funnel.rerankTopN}`, c: 'var(--c-rerank)' },
    { t: 'MMR', d: dv.enabled ? `λ=${dv.mmr_lambda}` : '未启用', c: 'var(--c-mmr)' },
    { t: 'Small-to-Big', d: `child → parent，取 top-${p.funnel.contextTopN}`, c: 'var(--c-mmr)' },
    { t: '织网 / 压缩', d: `UnWeaver ${onOff(gph.unweaver_enabled)} · 压缩 ${cp.enabled ? cp.mode : '关'}`, c: 'var(--c-weaved)' },
    { t: '生成', d: '回答协议：依据 · 拒答 · 冲突 · 引用', c: 'var(--success)' },
  ];
  const funnel = [
    { label: '每路召回 recall_top_k', v: p.funnel.recallTopK, c: 'var(--c-dense)' },
    { label: '精排候选 candidates', v: p.funnel.candidates, c: 'var(--c-fused)' },
    { label: '重排保留 rerank_top_n', v: p.funnel.rerankTopN, c: 'var(--c-rerank)' },
    { label: '最终上下文 context_top_n', v: p.funnel.contextTopN, c: 'var(--c-mmr)' },
  ];
  const maxF = Math.max(...funnel.map((f) => f.v), 1);

  return (
    <div className="page col gap-16 fade-in">
      {rt.offlineMode ? (
        <Callout tone="warn" icon={<AlertTriangle size={16} />}>
          <div className="row between wrap gap-12">
            <span><b>离线体验模式</b>：未配置模型，当前使用哈希向量 + BM25 + 词法精排 + 抽取式回答。所有页面都可操作，但向量路只反映词面重叠，没有真实语义。</span>
            <Button size="sm" variant="primary" onClick={() => go('models')}><Cpu size={13} />配置模型</Button>
          </div>
        </Callout>
      ) : null}

      <div className="card stat-strip">
        <div className="stat"><div className="label">知识库</div><div className="value">{counts.knowledgeBases}</div><div className="sub">按业务域拆库，支持查询路由</div></div>
        <div className="stat"><div className="label">文档</div><div className="value">{counts.documents}</div><div className="sub">PDF / Office / Markdown / Excel</div></div>
        <div className="stat"><div className="label">检索切片（child）</div><div className="value">{counts.chunks}</div><div className="sub">检索粒度；parent 为阅读粒度</div></div>
        <div className="stat"><div className="label">已保存的模型配置</div><div className="value">{counts.modelConfigs}</div><div className="sub">文本 / 嵌入 / 重排 / 多模态</div></div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>六项核心检索能力</h3>
          <div className="row gap-8">
            <Button size="sm" onClick={() => go('playground')}><FlaskConical size={13} />在实验室里逐层查看</Button>
            <Button size="sm" onClick={() => go('settings')}>调整策略</Button>
          </div>
        </div>
        <div className="card-body grid grid-3">
          {caps.map((c) => (
            <div key={c.name} className="cap" style={{ ['--p' as any]: c.on ? c.color : 'var(--border-strong)' }} title={c.note ?? undefined}>
              <span className="cap-mark" />
              <div style={{ minWidth: 0 }}>
                <h4>{c.name}<Badge tone={!c.on ? '' : c.note ? 'warn' : 'ok'}>{!c.on ? '未启用' : c.note ?? '已启用'}</Badge></h4>
                <p>{c.why}</p>
                <div className="v">{c.value}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>当前生效的管线</h3>
          <div className="row gap-6 wrap">
            <Badge tone={rt.mineruAvailable ? 'ok' : 'warn'}>{rt.mineruAvailable ? `MinerU${rt.mineruMode === 'cloud' ? ' 云端 API' : rt.mineruMode === 'server' ? ' 自托管' : ''} · ${rt.mineruTier}` : 'MinerU 未安装（内置解析器）'}</Badge>
            <Badge tone={rt.popoEnabled ? 'ok' : ''}>{rt.popoEnabled ? 'Popo 文档树 已启用' : 'Popo 文档树 未启用'}</Badge>
            <Badge tone={rt.crossEncoderAvailable ? 'ok' : ''}>{rt.crossEncoderAvailable ? 'Cross-Encoder 可用' : '无本地 Cross-Encoder'}</Badge>
          </div>
        </div>
        <div className="card-body col gap-12">
          <div>
            <div className="section-label">离线建库</div>
            <div className="pipeline">{offline.map((s, i) => <div key={s.t} className="pipeline-step" style={{ ['--p' as any]: s.c }}><div className="t"><span className="n">{i + 1}</span>{s.t}</div><div className="d">{s.d}</div></div>)}</div>
          </div>
          <div>
            <div className="section-label">在线检索</div>
            <div className="pipeline">{online.map((s, i) => <div key={s.t} className="pipeline-step" style={{ ['--p' as any]: s.c }}><div className="t"><span className="n">{i + 5}</span>{s.t}</div><div className="d">{s.d}</div></div>)}</div>
          </div>
        </div>
      </div>

      <div className="grid grid-2">
        <div className="card">
          <div className="card-header"><h3>运行时模型</h3><Button size="sm" onClick={() => go('models')}><Cpu size={13} />模型配置</Button></div>
          <div className="table-wrap">
            <table className="table">
              <tbody>
                <RuntimeRow title="文本 LLM" info={rt.llm} fallback="抽取式回答（返回检索原文）" />
                <RuntimeRow title="向量 Embedding" info={rt.embedding} fallback="哈希向量（只反映词面重叠）" />
                <RuntimeRow title="重排 Reranker" info={rt.rerank} fallback="词法精排" />
                <RuntimeRow title="多模态" info={rt.multimodal} fallback="回退文本模型" />
                <RuntimeRow title="图文向量" info={rt.imageEmbedding} fallback="离线感知特征（仅图搜图）" />
              </tbody>
            </table>
          </div>
          <div className="card-body" style={{ borderTop: '1px solid var(--border)' }}>
            <div className="section-label">漏斗式管理</div>
            <div className="funnel">
              {funnel.map((f) => (
                <div className="funnel-row" key={f.label}>
                  <span className="muted">{f.label}</span>
                  <div className="bar" style={{ width: `${Math.max(4, (f.v / maxF) * 100)}%`, ['--p' as any]: f.c }} />
                  <span className="mono" style={{ textAlign: 'right' }}>{f.v}</span>
                </div>
              ))}
            </div>
            <div className="xs faint mt-12">传给 Reranker 的候选低于 50 时效果断崖式下跌（20 候选 Recall@5 = 0.458，50 候选 = 0.826）；最终上下文保持 3-5 条，避免 Lost in the Middle。</div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>知识库</h3>
            <Button size="sm" variant="primary" onClick={() => go('kb')}><Upload size={13} />上传文档</Button>
          </div>
          <div className="card-body col gap-4">
            {kbs.length === 0 ? <Empty icon={<Database size={32} />} title="还没有知识库" desc="创建知识库并上传文档后即可检索与问答。" action={<Button size="sm" onClick={() => go('kb')}>新建知识库</Button>} /> : null}
            {kbs.map((kb) => (
              <div key={kb.id} className="list-item" onClick={() => { setCurrentKbId(kb.id); go('kb'); }}>
                <Database size={15} className="faint" />
                <div className="flex-1">
                  <div className="strong ellipsis">{kb.name}</div>
                  <div className="xs faint ellipsis">{kb.description || '暂无描述'}</div>
                </div>
                {kb.reindexing || kb.needsReindex ? <Badge tone="info">索引重建中</Badge> : null}
                <span className="xs muted mono" style={{ whiteSpace: 'nowrap' }}>{kb.documentCount} 文档 · {kb.chunkCount} 切片 · {kb.entityCount} 实体</span>
                <ArrowRight size={14} className="faint" />
              </div>
            ))}
          </div>
          <div className="card-body" style={{ borderTop: '1px solid var(--border)' }}>
            <div className="section-label">下一步</div>
            <div className="row gap-8 wrap">
              <Button size="sm" onClick={() => go('coverage')}><Target size={13} />检查语料覆盖</Button>
              <Button size="sm" onClick={() => go('playground')}><FlaskConical size={13} />调试检索</Button>
              <Button size="sm" onClick={() => go('chat')}><MessageSquare size={13} />开始问答</Button>
              <Button size="sm" onClick={() => go('eval')}><BarChart3 size={13} />跑评测</Button>
            </div>
            <div className="xs faint mt-12">调优顺序：<b className="strong">切片</b>（约一半时间）→ 混合检索 → 重排 → Query 改写；每次只改一个变量并跑一次评测。</div>
          </div>
        </div>
      </div>
    </div>
  );
}
