// LanceDB 控制台：全局库的表 / 版本 / 索引，知识库向量（后端或镜像）的检索实验与索引维护，知识库的多模态表。
import { useCallback, useEffect, useState } from 'react';
import { Boxes, Database, FileSearch, GitBranch, Image as ImageIcon, Layers, Play, RefreshCw, Search, Upload, Wrench, Zap } from 'lucide-react';
import { ApiError } from '../api/client';
import {
  fmtBytes, fmtTime, lanceApi,
  type LanceKbInfo, type LanceOverview, type LanceSearchMode, type LanceSearchResult, type LanceVersion, type MMInfo, type MMResult,
} from '../api/lance';
import { KbSelect } from '../components/kb-select';
import { Badge, Button, Callout, Drawer, Empty, ScoreBar, Spinner, Tabs } from '../components/ui';
import { useApp } from '../store/app-store';

const MODE_LABEL: Record<LanceSearchMode, string> = { vector: '向量（ANN）', exact: '精确扫描', fts: '全文（BM25）', hybrid: '混合（RRF）' };
const errText = (e: unknown) => (e instanceof ApiError ? e.message : String(e));

function readFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

export function LanceDBPage() {
  const { currentKb, toast } = useApp();
  const [ov, setOv] = useState<LanceOverview | null>(null);
  const [kbInfo, setKbInfo] = useState<LanceKbInfo | null>(null);
  const [versions, setVersions] = useState<{ name: string; rows: LanceVersion[] } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setOv(await lanceApi.overview());
      if (currentKb) setKbInfo(await lanceApi.kbInfo(currentKb.id));
    } catch (e) {
      toast(`读取 LanceDB 状态失败：${errText(e)}`, 'bad');
    }
  }, [currentKb, toast]);
  useEffect(() => { load(); }, [load]);

  const act = async (key: string, fn: () => Promise<unknown>, ok: string) => {
    setBusy(key);
    try {
      await fn();
      toast(ok, 'ok');
      await load();
    } catch (e) {
      toast(errText(e), 'bad');
    } finally {
      setBusy(null);
    }
  };

  if (!ov) return <div className="page"><Spinner label="正在读取 LanceDB…" /></div>;
  const kbRow = ov.kbs.find((k) => k.kbId === currentKb?.id);

  return (
    <div className="page col gap-16 fade-in">
      <div className="card stat-strip">
        <div className="stat"><div className="label"><Database size={12} />LanceDB</div><div className="value">{ov.version}</div><div className="sub mono" title={ov.dir}>{ov.dir}</div></div>
        <div className="stat"><div className="label"><Boxes size={12} />全局库的表</div><div className="value">{ov.tables.length}</div><div className="sub">记忆 / 推荐 / 黑板 / Agent 湖 / 镜像</div></div>
        <div className="stat"><div className="label"><Layers size={12} />知识库向量后端</div><div className="value">{ov.backend}</div><div className="sub">{ov.backend === 'lancedb' ? '线上检索直接走 LanceDB' : '可在策略配置切换为 lancedb'}</div></div>
        <div className="stat"><div className="label"><FileSearch size={12} />关键词路</div><div className="value">{ov.sparseBackend === 'lancedb_fts' ? 'FTS' : 'BM25'}</div><div className="sub">{ov.sparseBackend === 'lancedb_fts' ? 'LanceDB 原生全文（ICU 分词）' : 'bm25s + jieba'}</div></div>
      </div>

      {ov.backend !== 'lancedb' ? (
        <Callout icon={<Zap size={14} />}>
          线上知识库目前用 <b>{ov.backend}</b> 向量后端。在「策略配置 → 向量库后端」选 <b>LanceDB</b> 保存后，系统会自动用 LanceDB 重建所有知识库（校对过的文档保留校对版本），
          之后检索的向量路、SQL 预过滤、全文 / 混合检索、版本回看都直接在 LanceDB 上进行。切换前，可以先在下方把当前知识库<b>镜像</b>进 LanceDB 体验，不影响线上。
        </Callout>
      ) : null}

      <div className="grid grid-2" style={{ alignItems: 'start' }}>
        <KbVectorCard kbInfo={kbInfo} kbRowBackend={kbRow?.backend} busy={busy} onAct={act} />
        <TablesCard ov={ov} onVersions={async (name) => setVersions({ name, rows: await lanceApi.tableVersions(name) })} />
      </div>

      {currentKb ? <SearchLab kbId={currentKb.id} info={kbInfo} /> : null}
      {currentKb ? <MultimodalCard kbId={currentKb.id} hasMirror={kbRow?.multimodal ?? false} images={kbRow?.images ?? 0} onChanged={load} /> : null}

      {versions ? (
        <Drawer title={`版本历史 · ${versions.name}`} subtitle="每次写入（追加 / 更新 / 删除 / 建索引）都是一个新版本，可以回看" onClose={() => setVersions(null)}>
          <VersionList rows={versions.rows} />
        </Drawer>
      ) : null}
    </div>
  );
}

function VersionList({ rows }: { rows: LanceVersion[] }) {
  if (!rows.length) return <Empty title="还没有版本" />;
  return (
    <div className="table-wrap">
      <table className="table">
        <thead><tr><th>版本</th><th>时间</th><th>说明</th></tr></thead>
        <tbody>
          {rows.map((v) => (
            <tr key={v.version}><td className="mono">v{v.version}</td><td className="xs">{fmtTime(v.timestamp)}</td><td className="xs faint mono">{Object.keys(v.metadata || {}).length ? JSON.stringify(v.metadata) : '—'}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function KbVectorCard({ kbInfo, kbRowBackend, busy, onAct }: { kbInfo: LanceKbInfo | null; kbRowBackend?: string; busy: string | null; onAct: (key: string, fn: () => Promise<unknown>, ok: string) => void }) {
  const { currentKb } = useApp();
  const [indexType, setIndexType] = useState('IVF_HNSW_SQ');
  if (!currentKb) return <div className="card"><Empty title="先选择一个知识库" /></div>;
  const mode = kbInfo?.mode ?? 'none';
  return (
    <div className="card">
      <div className="card-header">
        <h3><Layers size={14} />知识库向量 · RAG</h3>
        <div className="row gap-6"><KbSelect /></div>
      </div>
      <div className="card-body col gap-12">
        <div className="row gap-6 wrap">
          <Badge tone={mode === 'backend' ? 'ok' : mode === 'mirror' ? 'info' : 'warn'}>{mode === 'backend' ? '线上后端：LanceDB' : mode === 'mirror' ? `镜像（线上后端：${kbRowBackend ?? '—'}）` : `未接入（线上后端：${kbRowBackend ?? '—'}）`}</Badge>
          {kbInfo?.exists ? <Badge mono>v{kbInfo.version} · {kbInfo.versions} 个版本</Badge> : null}
          {kbInfo?.exists ? <Badge tone={kbInfo.annIndexed ? 'accent' : ''}>{kbInfo.annIndexed ? 'ANN 索引已建' : `精确扫描（≥ ${kbInfo.indexMinRows} 行自动建 ANN）`}</Badge> : null}
        </div>
        {mode === 'none' ? <div className="xs muted">{kbInfo?.note}</div> : null}
        {kbInfo?.exists ? (
          <div className="grid grid-3">
            <div className="chunk"><div className="xs faint">向量行数</div><div className="strong mono">{kbInfo.rows}</div></div>
            <div className="chunk"><div className="xs faint">存储（含索引）</div><div className="strong mono">{fmtBytes(kbInfo.bytes)}</div></div>
            <div className="chunk"><div className="xs faint">碎片 / 小碎片</div><div className="strong mono">{kbInfo.fragments ?? '—'} / {kbInfo.smallFragments ?? '—'}</div></div>
          </div>
        ) : null}
        {kbInfo?.indices?.length ? (
          <div className="row gap-6 wrap">{kbInfo.indices.map((ix) => <Badge key={ix.name} tone={ix.columns.includes('vector') ? 'dense' : ix.columns.includes('text') ? 'sparse' : ''} title={ix.name}>{ix.type} · {ix.columns.join(',')}</Badge>)}</div>
        ) : null}
        {kbInfo?.aux && Object.keys(kbInfo.aux).length ? (
          <div className="row gap-6 wrap xs muted">
            <span>辅助索引</span>
            {Object.entries(kbInfo.aux).map(([name, a]) => (
              <Badge key={name} tone={a.backend === 'lancedb' ? 'accent' : 'warn'} title={a.backend === 'lancedb' ? `LanceDB 表 aux_${name}：向量 + 全文，原生混合检索` : '还是 numpy 文件：重建该知识库后切到 LanceDB'}>
                {({ summary: '摘要', section: '章节', sentence: '句子' } as Record<string, string>)[name] ?? name} {a.rows} · {a.backend === 'lancedb' ? 'LanceDB' : 'numpy'}
              </Badge>
            ))}
          </div>
        ) : null}
        <div className="row gap-8 wrap">
          {mode !== 'backend' ? (
            <Button size="sm" variant="primary" loading={busy === 'mirror'} onClick={() => onAct('mirror', () => lanceApi.mirror(currentKb.id), '已把当前知识库的向量与正文镜像进 LanceDB')}>
              <Upload size={13} />{mode === 'mirror' ? '重新镜像' : '镜像到 LanceDB'}
            </Button>
          ) : null}
          {kbInfo?.exists ? (
            <>
              <select value={indexType} onChange={(e) => setIndexType(e.target.value)} style={{ width: 'auto', padding: '4px 26px 4px 8px', fontSize: 12 }} aria-label="ANN 索引类型">
                <option value="IVF_HNSW_SQ">IVF_HNSW_SQ（HNSW + 标量量化）</option>
                <option value="IVF_PQ">IVF_PQ（乘积量化，最省内存）</option>
                <option value="IVF_FLAT">IVF_FLAT（不量化）</option>
              </select>
              <Button size="sm" loading={busy === 'index'} onClick={() => onAct('index', () => lanceApi.buildIndex(currentKb.id, indexType), `已建 ${indexType} 索引`)}><Wrench size={13} />建 ANN 索引</Button>
              <Button size="sm" loading={busy === 'optimize'} onClick={() => onAct('optimize', () => lanceApi.optimize(currentKb.id), '已 optimize：合并小文件、新数据并入索引、清理过期版本')}><RefreshCw size={13} />Optimize</Button>
            </>
          ) : null}
        </div>
        <div className="xs faint">小表精确扫描就最快、召回 100%；行数上万后再建 ANN 索引，用 nprobes / refine_factor 在下方实验里权衡速度与召回。</div>
      </div>
    </div>
  );
}

function TablesCard({ ov, onVersions }: { ov: LanceOverview; onVersions: (name: string) => void }) {
  return (
    <div className="card">
      <div className="card-header"><h3><Boxes size={14} />全局库 · data/lancedb</h3><Badge mono>{ov.tables.length} 张表</Badge></div>
      {ov.tables.length === 0 ? (
        <Empty title="还没有表" desc="使用长期记忆、推荐、多智能体或 Agent 湖后，这里会出现对应的 LanceDB 表。" />
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>表</th><th>用途</th><th className="num">行数</th><th>版本</th><th>索引</th><th className="num">大小</th></tr></thead>
            <tbody>
              {ov.tables.map((t) => (
                <tr key={t.name}>
                  <td className="mono xs">{t.name}</td>
                  <td className="xs">{t.purpose || '—'}</td>
                  <td className="num">{t.rows}</td>
                  <td><button className="btn xs ghost" onClick={() => onVersions(t.name)} title="查看版本历史"><GitBranch size={11} />v{t.version}</button></td>
                  <td><div className="row gap-4 wrap">{t.indices.length ? t.indices.map((ix) => <Badge key={ix.name} mono>{ix.type}</Badge>) : <span className="xs faint">—</span>}</div></td>
                  <td className="num xs">{fmtBytes(t.bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function SearchLab({ kbId, info }: { kbId: string; info: LanceKbInfo | null }) {
  const { toast } = useApp();
  const [query, setQuery] = useState('出差能不能住贵一点的酒店');
  const [mode, setMode] = useState<LanceSearchMode>('hybrid');
  const [k, setK] = useState(5);
  const [nprobes, setNprobes] = useState<number | ''>('');
  const [refine, setRefine] = useState<number | ''>('');
  const [version, setVersion] = useState<number | ''>('');
  const [explain, setExplain] = useState(false);
  const [res, setRes] = useState<LanceSearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const ready = info?.exists;

  const run = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      setRes(await lanceApi.search(kbId, { query, mode: version ? 'vector' : mode, k, version: version || null, nprobes: nprobes || null, refineFactor: refine === '' ? null : refine, explain }));
    } catch (e) {
      toast(errText(e), 'bad');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-header"><h3><Search size={14} />检索实验 · 同一张表上的向量 / 全文 / 混合 / 时间旅行</h3>{res ? <Badge mono>embed {res.timingsMs.embed}ms · search {res.timingsMs.search}ms</Badge> : null}</div>
      <div className="card-body col gap-12">
        {!ready ? <Callout tone="warn">当前知识库还没有 LanceDB 数据：向量后端切到 LanceDB，或先点上方「镜像到 LanceDB」。</Callout> : null}
        <div className="row gap-8 wrap">
          <input className="flex-1" style={{ minWidth: 240 }} value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && run()} placeholder="输入问题" />
          <Tabs value={mode} onChange={setMode} items={(Object.keys(MODE_LABEL) as LanceSearchMode[]).map((m) => ({ value: m, label: MODE_LABEL[m] }))} />
          <Button variant="primary" onClick={run} loading={loading} disabled={!ready}><Play size={13} />检索</Button>
        </div>
        <div className="row gap-12 wrap xs muted">
          <label className="row gap-6">top-k <input type="number" min={1} max={50} value={k} onChange={(e) => setK(Number(e.target.value) || 5)} style={{ width: 64 }} /></label>
          <label className="row gap-6" title="IVF 索引查询时探查的分区数（有 ANN 索引时生效）">nprobes <input type="number" min={1} max={500} value={nprobes} placeholder="默认" onChange={(e) => setNprobes(e.target.value === '' ? '' : Number(e.target.value))} style={{ width: 72 }} /></label>
          <label className="row gap-6" title="量化索引多取 k×refine 条再用原始向量重排；0 = 不重排">refine_factor <input type="number" min={0} max={50} value={refine} placeholder="默认" onChange={(e) => setRefine(e.target.value === '' ? '' : Number(e.target.value))} style={{ width: 72 }} /></label>
          <label className="row gap-6" title="时间旅行：在历史版本上做向量检索（只读）">版本
            <select value={version} onChange={(e) => setVersion(e.target.value === '' ? '' : Number(e.target.value))} style={{ width: 'auto', padding: '3px 24px 3px 8px', fontSize: 12 }}>
              <option value="">当前</option>
              {(info?.versionsList ?? []).map((v) => <option key={v.version} value={v.version}>v{v.version} · {fmtTime(v.timestamp)}</option>)}
            </select>
          </label>
          <label className="row gap-6"><input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} />查询计划</label>
        </div>
        {res ? (
          res.hits.length === 0 ? <Empty title="没有命中" /> : (
            <div className="col gap-8">
              <div className="row gap-6 wrap xs muted"><Badge tone="info">{res.version ? `时间旅行 v${res.version}` : MODE_LABEL[res.mode]}</Badge><Badge>{res.source === 'backend' ? '线上后端' : '镜像表'}</Badge>{res.annIndexed ? <Badge tone="accent">ANN</Badge> : <Badge>精确扫描</Badge>}<span>分数：向量 = 余弦相似度，全文 = BM25，混合 = RRF 相关度</span></div>
              {res.hits.map((h, i) => (
                <div key={h.chunkId} className="chunk">
                  <div className="chunk-head"><span className="ellipsis"><b className="mono">#{i + 1}</b> {h.fileName ?? h.docId.slice(0, 8)} · {h.breadcrumb}</span><span className="mono">{h.score.toFixed(4)}</span></div>
                  <ScoreBar value={h.score} max={res.hits[0].score || 1} />
                  <div className="xs muted clamp-2 mt-4">{h.text}</div>
                </div>
              ))}
            </div>
          )
        ) : null}
        {res?.plan ? <pre className="chunk mono xs pre" style={{ maxHeight: 220, overflow: 'auto' }}>{res.plan}</pre> : null}
      </div>
    </div>
  );
}

function MultimodalCard({ kbId, hasMirror, images, onChanged }: { kbId: string; hasMirror: boolean; images: number; onChanged: () => void }) {
  const { toast } = useApp();
  const [info, setInfo] = useState<MMInfo | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [text, setText] = useState('销售额柱状图');
  const [image, setImage] = useState<string | null>(null);
  const [target, setTarget] = useState<'image' | 'text' | 'all'>('image');
  const [mode, setMode] = useState<'vector' | 'fts' | 'hybrid'>('vector');
  const [res, setRes] = useState<MMResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => { lanceApi.mmInfo(kbId).then(setInfo).catch(() => setInfo(null)); }, [kbId, hasMirror]);

  const sync = async () => {
    setSyncing(true);
    try {
      const r = await lanceApi.mmSync(kbId);
      toast(`已同步 ${r.images} 张图片${r.texts ? ` + ${r.texts} 段正文（CLIP 空间）` : ''}：写入 ${r.written} 行、删除 ${r.removed} 行，${r.ms} ms`, 'ok');
      setInfo(await lanceApi.mmInfo(kbId));
      onChanged();
    } catch (e) {
      toast(errText(e), 'bad');
    } finally {
      setSyncing(false);
    }
  };
  const run = async () => {
    setLoading(true);
    try {
      setRes(await lanceApi.mmSearch(kbId, { text: image ? undefined : text, imageB64: image, target, mode, k: 8 }));
    } catch (e) {
      toast(errText(e), 'bad');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <h3><ImageIcon size={14} />多模态表 · 图片与正文同表（CLIP 向量 + 缩略图 + 图注 / OCR）</h3>
        <div className="row gap-6">
          {info?.online ? <Badge tone="ok" title="策略配置里已打开「多模态表接入线上图文检索」：保存索引时自动增量同步，文搜图 / 问答附图 / 图搜文多这一路">已接入线上检索</Badge> : null}
          {info?.exists ? <Badge mono>{info.byModality?.image ?? 0} 图 · {info.byModality?.text ?? 0} 文 · v{info.version}</Badge> : null}
          <Button size="sm" onClick={sync} loading={syncing}><RefreshCw size={13} />{info?.exists ? '重新同步' : '同步到 LanceDB'}</Button>
        </div>
      </div>
      <div className="card-body col gap-12">
        <div className="xs muted">每个知识库一张表（索引目录的 LanceDB 库里的 mm）。{info?.online ? '已接入线上图文检索：保存索引时自动增量同步，这里手动同步还会补齐缺的正文图文向量。' : '在策略配置里打开「多模态表接入线上图文检索」后，保存索引时自动增量同步，并作为文搜图 / 图搜文的一路参与线上检索。'}</div>
        {images === 0 ? <div className="xs muted">当前知识库没有图片：表里只有正文行（CLIP 文本向量），用于以图搜文 / 找正文；接入线上检索后，保存索引（包括整库重建）时自动建好并维护。</div> : null}
        <div className="row gap-8 wrap">
          {image ? (
            <span className="row gap-6"><img src={image} alt="查询图片" style={{ height: 34, borderRadius: 6, border: '1px solid var(--border)' }} /><Button size="xs" variant="ghost" onClick={() => setImage(null)}>移除图片</Button></span>
          ) : (
            <input className="flex-1" style={{ minWidth: 200 }} value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && run()} placeholder="文字描述（文搜图 / 文搜文）" />
          )}
          <label className="btn sm" style={{ cursor: 'pointer' }}><Upload size={13} />以图搜<input type="file" accept="image/*" hidden onChange={async (e) => { const f = e.target.files?.[0]; if (f) setImage(await readFile(f)); e.target.value = ''; }} /></label>
          <Tabs value={target} onChange={setTarget} items={[{ value: 'image', label: '找图' }, { value: 'text', label: '找正文' }, { value: 'all', label: '全部' }]} />
          {!image ? <Tabs value={mode} onChange={setMode} items={[{ value: 'vector', label: 'CLIP 向量' }, { value: 'fts', label: '全文' }, { value: 'hybrid', label: '混合' }]} /> : null}
          <Button variant="primary" onClick={run} loading={loading} disabled={!info?.exists}><Play size={13} />检索</Button>
        </div>
        {res ? (
          <div className="col gap-8">
            <div className="row gap-6 wrap xs muted">{res.how ? <Badge tone="info">{res.how}</Badge> : null}{res.note ? <span className="text-warning">{res.note}</span> : null}<span>{res.ms} ms</span></div>
            {res.hits.length === 0 ? <Empty title="没有命中" /> : (
              <div className="grid grid-4">
                {res.hits.map((h) => (
                  <div key={h.id} className="chunk col gap-6">
                    {h.hasImage ? <img src={lanceApi.thumbUrl(kbId, h.id)} alt={h.title} style={{ width: '100%', aspectRatio: '4 / 3', objectFit: 'contain', background: 'var(--bg-3)', borderRadius: 6 }} /> : null}
                    <div className="row between gap-6"><Badge tone={h.modality === 'image' ? 'hyde' : 'sparse'}>{h.modality === 'image' ? '图片' : '正文'}</Badge><span className="mono xs">{h.score.toFixed(3)}</span></div>
                    <div className="xs strong clamp-2">{h.title}</div>
                    <div className="xs faint clamp-2">{h.fileName ?? h.docTitle}{h.page != null ? ` · 第 ${h.page + 1} 页` : ''}</div>
                    {h.modality === 'text' ? <div className="xs muted clamp-2">{h.text}</div> : null}
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
}
