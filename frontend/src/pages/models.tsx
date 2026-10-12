import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CheckCircle2, Cpu, HardDrive, Images, Layers, MessageSquare, Plus, Power, RefreshCw, Search, Settings2, Shuffle, Trash2, Wifi, WifiOff, X, Zap } from 'lucide-react';
import { api } from '../api/client';
import type { ModelCategory, ModelConfig, ProviderPreset, RuntimeInfo } from '../api/types';
import { Badge, Button, Callout, Field, confirmDialog } from '../components/ui';
import { useApp } from '../store/app-store';

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));
const CATS: { key: ModelCategory; label: string; short: string; icon: React.ReactNode; desc: string; fallback: string }[] = [
  { key: 'text', label: '文本模型', short: '文本', icon: <MessageSquare size={14} />, desc: '回答生成、查询改写、HyDE、LLM 精排、多跳规划', fallback: '抽取式回答（直接返回检索原文）' },
  { key: 'embedding', label: '文本嵌入（Embedding）模型', short: '嵌入', icon: <Layers size={14} />, desc: '把文本映射成向量，支撑语义检索。更换后自动重建各知识库索引', fallback: '哈希向量（只反映词面重叠）' },
  { key: 'rerank', label: '重排模型', short: '重排', icon: <Shuffle size={14} />, desc: 'Cross-Encoder 精排，单次改动提升最大的一层', fallback: 'LLM 打分 → 词法精排' },
  { key: 'multimodal', label: '多模态模型', short: '多模态', icon: <Zap size={14} />, desc: '看图的模型（VLM）：入库时给图片生成描述，问答时直接看用户图片与检索到的图片', fallback: '使用文本模型（不发送图片像素，只给 OCR / 描述文本）' },
  { key: 'image_embedding', label: '图文向量模型', short: '图文', icon: <Images size={14} />, desc: 'CLIP 类双塔模型：图片与文字映射到同一向量空间，支撑文搜图 / 图搜图 / 图搜文。更换后图片向量自动重算', fallback: '离线感知特征（只支持图搜图；文搜图走图片描述文本）' },
];
const TYPE_LABEL: Record<string, string> = { anthropic: 'Anthropic SDK', openai_compatible: 'OpenAI 兼容', local: '本地模型', api: 'API' };

/** 两组卡片：生成类（2 列）与检索类（3 列），每行都排满，不留孤卡 */
const GROUPS: { title: string; hint: string; cols: 'grid-2' | 'grid-3'; keys: ModelCategory[] }[] = [
  { title: '生成模型', hint: '负责理解问题、组织回答与看图', cols: 'grid-2', keys: ['text', 'multimodal'] },
  { title: '检索模型', hint: '负责把文本 / 图片变成向量并给候选排序，决定召回质量', cols: 'grid-3', keys: ['embedding', 'rerank', 'image_embedding'] },
];

export function ModelsPage() {
  const { status, refreshStatus, refreshKbs, toast } = useApp();
  const [providers, setProviders] = useState<ProviderPreset[]>([]);
  const [configs, setConfigs] = useState<ModelConfig[]>([]);
  const [tests, setTests] = useState<Record<string, { loading: boolean; success?: boolean; latency?: number; message?: string }>>({});
  const [cfgOpen, setCfgOpen] = useState<ModelCategory | null>(null);

  const load = useCallback(async () => {
    try { setConfigs(await api.models.list()); } catch (e) { toast(errMsg(e), 'bad'); }
  }, [toast]);
  useEffect(() => {
    load();
    api.models.providers().then(setProviders).catch(() => undefined);
  }, [load]);
  const after = async () => { await load(); await refreshStatus(); refreshKbs().catch(() => undefined); };

  const test = async (key: string, body: Parameters<typeof api.models.test>[0]) => {
    setTests((t) => ({ ...t, [key]: { loading: true } }));
    try {
      const r = await api.models.test(body);
      setTests((t) => ({ ...t, [key]: { loading: false, ...r } }));
    } catch (e) { setTests((t) => ({ ...t, [key]: { loading: false, success: false, message: errMsg(e) } })); }
  };
  const toggleActive = async (c: ModelConfig) => {
    try {
      if (c.isActive) await api.models.deactivate(c.id); else await api.models.activate(c.id);
      toast(c.isActive ? `已停用 ${c.modelName}` : `已激活 ${c.modelName}；${ACTIVATE_NOTE[c.category]}`, 'ok');
      await after();
    } catch (e) { toast(errMsg(e), 'bad'); }
  };
  const remove = async (c: ModelConfig) => {
    if (!confirmDialog(`删除配置 ${c.providerName} / ${c.modelName}？`)) return;
    try { await api.models.remove(c.id); await after(); } catch (e) { toast(errMsg(e), 'bad'); }
  };

  const TestResult = ({ k }: { k: string }) => {
    const t = tests[k];
    if (!t || t.loading || t.success === undefined) return null;
    return <div className={`xs row gap-6 mt-8 ${t.success ? 'text-success' : 'text-danger'}`}>{t.success ? <Wifi size={11} /> : <WifiOff size={11} />}<span className="clamp-2">{t.success ? `连接成功 · ${t.latency}ms` : t.message}</span></div>;
  };
  const rt = status?.runtime;
  const rtOf = (c: ModelCategory): RuntimeInfo | undefined => (c === 'text' ? rt?.llm : c === 'embedding' ? rt?.embedding : c === 'rerank' ? rt?.rerank : c === 'image_embedding' ? rt?.imageEmbedding : rt?.multimodal);

  const renderCard = (cat: (typeof CATS)[number]) => {
    const list = configs.filter((c) => c.category === cat.key);
    const info = rtOf(cat.key);
    const editing = cfgOpen === cat.key;
    return (
      <div key={cat.key} className="card">
        <div className="card-header">
          <h3>{cat.icon}{cat.label}</h3>
          <div className="row gap-6">
            <Badge tone={info?.offline ? 'warn' : 'ok'}>{info?.offline ? '降级运行' : '已就绪'}</Badge>
            <Button size="xs" variant={editing ? 'default' : 'primary'} onClick={() => setCfgOpen(editing ? null : cat.key)}>{editing ? <><X size={11} />收起</> : <><Settings2 size={11} />配置</>}</Button>
          </div>
        </div>
        <div className="card-body col gap-8">
          <div className="xs faint">{cat.desc}</div>
          <div className="chunk">
            <div className="xs faint">当前生效</div>
            <div className="small strong ellipsis mono" title={info?.model ?? undefined}>{info?.model ?? info?.provider ?? '—'}</div>
            <div className="xs muted">{info?.offline ? `降级：${cat.fallback}` : `${info?.provider ?? ''} · 来源 ${info?.source ?? ''}`}</div>
          </div>
          {editing ? <ModelConfigurator key={cat.key} category={cat.key} providers={providers} configs={configs} current={info?.offline ? undefined : info?.model ?? undefined} onDone={async () => { setCfgOpen(null); await after(); }} /> : null}
          {cat.key === 'rerank' && info?.kind === 'cross_encoder' ? <LocalRerankerState info={info} onChanged={refreshStatus} /> : null}
          {list.length ? <div className="xs faint mt-8">已保存的配置（{list.length}）</div> : null}
          {list.length === 0 ? (
            <div className="xs faint" style={{ padding: '6px 0' }}>尚未添加配置，点右上角「配置」添加。</div>
          ) : list.map((c) => (
            <div key={c.id} className={`chunk ${c.isActive ? 'hl' : ''}`}>
              <div className="row between">
                <div style={{ minWidth: 0 }}>
                  <div className="small strong ellipsis mono" title={c.modelName}>{c.modelName}</div>
                  <div className="xs faint ellipsis">{c.providerName} · {TYPE_LABEL[c.providerType]}{c.apiKeyMasked ? ` · ${c.apiKeyMasked}` : ''}</div>
                </div>
                {c.isActive ? <Badge tone="ok"><CheckCircle2 size={10} />已激活</Badge> : null}
              </div>
              <div className="row gap-6 mt-8">
                <Button size="xs" variant={c.isActive ? 'default' : 'primary'} onClick={() => toggleActive(c)}><Power size={11} />{c.isActive ? '停用' : '激活'}</Button>
                <Button size="xs" loading={tests[c.id]?.loading} onClick={() => test(c.id, { configId: c.id, providerType: c.providerType, baseUrl: c.baseUrl, category: c.category, modelName: c.modelName })}><Wifi size={11} />测试</Button>
                <Button size="xs" variant="ghost" icon onClick={() => remove(c)} title="删除配置" aria-label="删除配置"><Trash2 size={12} /></Button>
              </div>
              <TestResult k={c.id} />
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="page col gap-16 fade-in">
      <Callout icon={<Cpu size={15} />}>
        <b>自由组合</b>：每类用途各激活一个模型即可，例如 Claude 负责回答、BGE-M3 负责文本嵌入、bge-reranker-v2-m3 负责精排。点卡片右上角「配置」选供应商与模型；同一服务商的 API Key 只需填一次。未配置的用途会自动降级（见各卡片），系统始终可用。解析优先级：此处激活的配置 → <code>.env</code> → <code>config.yaml</code> → 离线兜底。
      </Callout>
      {GROUPS.map((g) => (
        <section key={g.title} className="col gap-8">
          <div className="row gap-8" style={{ alignItems: 'baseline' }}>
            <span className="small strong">{g.title}</span>
            <span className="xs faint">{g.hint}</span>
          </div>
          <div className={`grid ${g.cols}`} style={{ alignItems: 'start' }}>
            {g.keys.map((k) => renderCard(CATS.find((c) => c.key === k)!))}
          </div>
        </section>
      ))}
    </div>
  );
}

/** 激活后的影响：嵌入 / 图文向量换模型要重算向量，其余用途立即生效 */
const ACTIVATE_NOTE: Record<ModelCategory, string> = {
  text: '立即生效',
  embedding: '正在后台自动重建索引',
  rerank: '立即生效',
  multimodal: '立即生效',
  image_embedding: '图片向量将在后台自动重算',
};
const NOUN: Record<ModelCategory, string> = { text: '文本模型', embedding: '嵌入模型', rerank: '重排模型', multimodal: '多模态模型', image_embedding: '图文向量模型' };
const sameUrl = (a?: string | null, b?: string | null) => (a ?? '').trim().replace(/\/+$/, '') === (b ?? '').trim().replace(/\/+$/, '');

/** 任一用途的就地配置：选供应商 → 自动加载模型列表（API 拉 /models，本地扫描 HuggingFace 缓存）→ 点选或手动输入 → 测试 / 保存并激活。
 *  同一服务商已保存过 Key 的（例如文本模型已配好 DeepSeek），API Key 留空即可复用，不必重复填写。 */
function ModelConfigurator({ category, providers, configs, current, onDone }: { category: ModelCategory; providers: ProviderPreset[]; configs: ModelConfig[]; current?: string; onDone: () => Promise<void> }) {
  const { toast } = useApp();
  const presets = useMemo(() => providers.filter((p) => p.categories.includes(category)), [providers, category]);
  const [provider, setProvider] = useState<ProviderPreset | null>(null);
  const [baseUrl, setBaseUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [models, setModels] = useState<string[]>([]);
  const [cached, setCached] = useState<string[]>([]);
  const [note, setNote] = useState('');
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [model, setModel] = useState('');
  const [test, setTest] = useState<{ loading: boolean; success?: boolean; latency?: number; message?: string }>({ loading: false });
  const [saving, setSaving] = useState(false);
  const seq = useRef(0);
  const noun = NOUN[category];
  const isLocal = provider?.type === 'local';

  // 可复用 Key 的已保存配置：同供应商、同端点、存过 Key；优先当前用途里激活的那条
  const donor = useMemo(() => {
    if (!provider || isLocal) return undefined;
    const hits = configs.filter((c) => c.apiKeyMasked && c.providerName === provider.name && sameUrl(c.baseUrl, baseUrl));
    return hits.find((c) => c.category === category && c.isActive) ?? hits.find((c) => c.isActive) ?? hits[0];
  }, [configs, provider, baseUrl, category, isLocal]);
  const keyFrom = !apiKey.trim() && donor ? donor.id : null;

  const pick = (p: ProviderPreset) => {
    seq.current++;
    setProvider(p);
    setBaseUrl(p.baseUrl ?? '');
    setApiKey('');
    setModels(p.models[category] ?? []);
    setCached([]);
    setNote('');
    setModel('');
    setQuery('');
    setTest({ loading: false });
  };
  // 默认供应商：该用途当前激活配置的供应商 → 本地模型 → 第一个预设
  useEffect(() => {
    if (provider || !presets.length) return;
    const active = configs.find((c) => c.category === category && c.isActive);
    pick(presets.find((p) => p.name === active?.providerName) ?? presets.find((p) => p.type === 'local') ?? presets[0]);
  }, [presets]); // eslint-disable-line

  const load = async (manual = false) => {
    if (!provider) return;
    const n = ++seq.current;
    setLoading(true);
    try {
      const r = await api.models.fetchList({ baseUrl, apiKey, providerType: provider.type, keyFromConfigId: keyFrom });
      if (n !== seq.current) return;                    // 供应商 / 输入又变了，丢弃过期结果
      const inCache = r.cached ?? [];
      let list = r.models[category] ?? [];
      let msg: string;
      if (r.source === 'local') {
        list = [...list].sort((a, b) => Number(inCache.includes(b)) - Number(inCache.includes(a)));
        msg = `本机已下载 ${list.filter((m) => inCache.includes(m)).length} 个${noun}（标记「本机」）；其余为常用预设，首次使用时自动下载`;
      } else if (r.source === 'api') {
        if (!list.length) list = Array.from(new Set(Object.values(r.models).flat()));
        msg = (r.models[category] ?? []).length ? `已从服务商拉取 ${r.total} 个模型，其中${noun} ${list.length} 个` : `已拉取 ${r.total} 个模型，未能按名称识别出${noun}，显示全部`;
      } else {
        list = list.length ? list : provider.models[category] ?? [];
        msg = `拉取失败${r.error ? `（${r.error.slice(0, 80)}）` : ''}，显示预设列表，也可手动输入`;
      }
      setModels(list);
      setCached(inCache);
      setNote(msg);
      setModel((cur) => cur || (current && list.includes(current) ? current : list[0]) || '');
      if (manual) toast(msg, r.source === 'presets' ? 'warn' : 'ok');
    } catch (e) {
      if (n === seq.current) setNote(errMsg(e));
    } finally {
      if (n === seq.current) setLoading(false);
    }
  };
  // 本地：选中即扫描；API：填好 Key（或可复用已保存的 Key、或改了 Base URL）停止输入 600ms 后自动拉取
  useEffect(() => {
    if (!provider) return;
    if (provider.type !== 'local') {
      if (!baseUrl.trim()) return;
      if (!/localhost|127\.0\.0\.1/.test(baseUrl) && apiKey.trim().length < 8 && !keyFrom) return;
    }
    const t = window.setTimeout(() => load(), provider.type === 'local' ? 0 : 600);
    return () => window.clearTimeout(t);
  }, [provider, baseUrl, apiKey, keyFrom]); // eslint-disable-line

  const q = query.trim();
  const shown = models.filter((m) => m.toLowerCase().includes(q.toLowerCase()));
  const custom = q && !models.includes(q) ? q : '';

  const runTest = async () => {
    if (!provider || !model) return;
    setTest({ loading: true });
    try {
      setTest({ loading: false, ...(await api.models.test({ configId: keyFrom ?? undefined, providerName: provider.name, providerType: provider.type, baseUrl, apiKey, category, modelName: model })) });
    } catch (e) { setTest({ loading: false, success: false, message: errMsg(e) }); }
  };
  const save = async () => {
    if (!provider || !model) return;
    setSaving(true);
    try {
      // 同供应商 + 同模型已存在：更新那一条，不重复新增
      const existing = configs.find((c) => c.category === category && c.providerName === provider.name && c.modelName === model && sameUrl(c.baseUrl, baseUrl));
      const cfg = await api.models.save({ id: existing?.id, providerName: provider.name, providerType: provider.type, baseUrl: baseUrl || null, apiKey: apiKey || null, keyFromConfigId: keyFrom, category, modelName: model });
      await api.models.activate(cfg.id);
      toast(`已激活 ${cfg.modelName}；${ACTIVATE_NOTE[category]}`, 'ok');
      await onDone();
    } catch (e) { toast(errMsg(e), 'bad'); } finally { setSaving(false); }
  };

  if (!presets.length) return <div className="chunk xs faint">暂无支持「{noun}」的供应商预设，可选择支持该用途的「自定义」供应商。</div>;
  return (
    <div className="chunk col gap-12" style={{ background: 'var(--bg-2)' }}>
      <Field label="供应商">
        <select value={provider?.name ?? ''} onChange={(e) => { const p = presets.find((x) => x.name === e.target.value); if (p) pick(p); }}>
          {presets.map((p) => <option key={p.name} value={p.name}>{p.name}（{TYPE_LABEL[p.type]}）</option>)}
        </select>
      </Field>
      {isLocal ? (
        <div className="xs faint">{category === 'image_embedding' ? <>本地 CLIP 通过 transformers 加载（<code>pip install torch transformers</code>），首次使用会下载权重。</> : <>本地模型通过 sentence-transformers 加载（<code>pip install sentence-transformers</code>），首次使用会下载权重。</>}</div>
      ) : (
        <>
          <Field label="Base URL" hint={provider?.type === 'anthropic' ? '留空使用官方地址' : 'OpenAI 兼容端点，通常以 /v1 结尾'}><input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="https://…/v1" /></Field>
          <Field label="API Key" hint={donor ? `留空将复用已保存的 Key（${donor.apiKeyMasked}）；填写新 Key 则使用新的` : '填好后自动拉取模型列表；仅保存在本机 SQLite'}>
            <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder={donor ? `${donor.apiKeyMasked}（已保存）` : 'sk-…（本地端点可留空）'} autoComplete="off" />
          </Field>
        </>
      )}
      <Field
        label={<span className="row between" style={{ width: '100%' }}><span>选择模型</span>{model ? <span className="xs muted">已选 <span className="mono strong">{model}</span></span> : null}</span>}
        hint={loading ? (isLocal ? '正在扫描本机 HuggingFace 缓存…' : '正在从服务商拉取模型列表…') : note || (isLocal ? '' : '填好 Base URL 与 API Key 后自动加载；也可直接输入模型名')}
      >
        <div className="col gap-6">
          <div style={{ position: 'relative' }}>
            <Search size={13} className="faint" style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)' }} />
            <input value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && q) setModel(shown.includes(q) || !shown.length ? q : shown[0]); }} placeholder="筛选列表，或输入列表里没有的模型名" autoComplete="off" style={{ paddingLeft: 30 }} />
          </div>
          <div className="scroll" style={{ maxHeight: 240, border: '1px solid var(--border)', borderRadius: 8, padding: 4 }}>
            {custom ? (
              <div className={`list-item small ${model === custom ? 'active' : ''}`} style={{ padding: '7px 10px' }} onClick={() => setModel(custom)}>
                <Plus size={12} /><span className="flex-1 ellipsis">使用自定义名称 <span className="mono strong">{custom}</span></span>
                {model === custom ? <CheckCircle2 size={12} /> : null}
              </div>
            ) : null}
            {shown.map((m) => (
              <div key={m} className={`list-item mono small ${m === model ? 'active' : ''}`} style={{ padding: '7px 10px' }} onClick={() => setModel(m)}>
                <span className="flex-1 ellipsis" title={m}>{m}</span>
                {m === current ? <Badge tone="info">当前生效</Badge> : null}
                {cached.includes(m) ? <Badge tone="ok"><HardDrive size={10} />本机</Badge> : null}
                {m === model ? <CheckCircle2 size={12} /> : null}
              </div>
            ))}
            {!shown.length && !custom ? <div className="xs faint" style={{ padding: '8px 10px' }}>{loading ? '加载中…' : '暂无可选模型，可在上方输入模型名'}</div> : null}
          </div>
        </div>
      </Field>
      <div className="row gap-8 wrap">
        <Button size="sm" loading={loading} disabled={!provider || (!isLocal && !baseUrl)} onClick={() => load(true)}><RefreshCw size={12} />{isLocal ? '重新扫描' : '重新拉取'}</Button>
        <Button size="sm" loading={test.loading} disabled={!model} onClick={runTest}><Wifi size={12} />测试</Button>
        <div className="flex-1" />
        <Button size="sm" variant="primary" loading={saving} disabled={!model} onClick={save}><Power size={12} />保存并激活</Button>
      </div>
      {!test.loading && test.success !== undefined ? <div className={`xs row gap-6 ${test.success ? 'text-success' : 'text-danger'}`}>{test.success ? <Wifi size={11} /> : <WifiOff size={11} />}<span className="clamp-2">{test.success ? `连接成功 · ${test.latency}ms` : test.message}</span></div> : null}
    </div>
  );
}

const RERANK_STATE: Record<string, string> = { idle: '未加载（首次精排时加载）', loading: '下载 / 加载中', ready: '已加载', failed: '不可用' };

/** 本地重排模型（默认 BAAI/bge-reranker-v2-m3）：权重是否在本机、是否已加载；没下载时可以一键后台下载。 */
function LocalRerankerState({ info, onChanged }: { info: RuntimeInfo; onChanged: () => Promise<void> | void }) {
  const { toast } = useApp();
  const [busy, setBusy] = useState(false);
  const download = async () => {
    setBusy(true);
    try {
      await api.models.rerankerDownload();
      toast('已开始后台下载重排模型（约 2.3GB），完成前由兜底精排顶上', 'ok');
      await onChanged();
    } catch (e) { toast(e instanceof Error ? e.message : String(e), 'bad'); } finally { setBusy(false); }
  };
  const state = info.state ?? 'idle';
  return (
    <div className="xs muted col gap-4">
      <div className="row gap-6 wrap">
        <Badge tone={info.cached ? 'ok' : 'warn'}>{info.cached ? '权重已在本机' : '权重未下载'}</Badge>
        <Badge tone={state === 'ready' ? 'ok' : state === 'failed' ? 'bad' : state === 'loading' ? 'info' : ''}>{RERANK_STATE[state] ?? state}{state === 'ready' && info.device ? ` · ${info.device}` : ''}</Badge>
        {info.fallback ? <span className="faint">兜底：{info.fallback}</span> : null}
      </div>
      {info.error ? <div className="text-danger">{info.error}</div> : null}
      {!info.cached && state !== 'loading' ? <div><Button size="xs" loading={busy} onClick={download}>下载权重</Button> <span className="faint">或命令行 <code>rag download-reranker</code></span></div> : null}
    </div>
  );
}
