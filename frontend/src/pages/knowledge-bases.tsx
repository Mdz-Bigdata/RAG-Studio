import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, Check, Database, History, FileText, Layers, List as ListIcon, Loader2, Network, Share2, Pencil, Plus, RefreshCw, RotateCw, ScanText, Search, Tags, Trash2, TreePine, Upload, X } from 'lucide-react';
import { api } from '../api/client';
import type { ChunkView, Document, EditBackup, EntityRelation, KbGraph, KnowledgeBase, ParsingInfo, PopoStatus } from '../api/types';
import { Badge, Button, Callout, Drawer, Empty, Field, Modal, Spinner, Tabs, confirmDialog } from '../components/ui';
import { EntityGraph } from '../components/entity-graph';
import { STATUS_LABEL, formatBytes, formatDate } from '../lib/format';
import { ACCESS_LEVELS, useApp } from '../store/app-store';

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));

function KbForm({ initial, onClose, onSaved }: { initial?: KnowledgeBase; onClose: () => void; onSaved: (kb: KnowledgeBase) => void }) {
  const { toast } = useApp();
  const [name, setName] = useState(initial?.name ?? '');
  const [description, setDescription] = useState(initial?.description ?? '');
  const [category, setCategory] = useState(initial?.category ?? '');
  const [language, setLanguage] = useState(initial?.language ?? 'zh');
  const [saving, setSaving] = useState(false);
  const save = async () => {
    if (!name.trim()) return;
    setSaving(true);
    try {
      const body = { name: name.trim(), description, category, language };
      const kb = initial ? await api.kb.update(initial.id, body) : await api.kb.create(body);
      toast(initial ? '知识库已更新' : '知识库已创建', 'ok');
      onSaved(kb);
    } catch (e) {
      toast(errMsg(e), 'bad');
    } finally {
      setSaving(false);
    }
  };
  return (
    <Modal title={initial ? '编辑知识库' : '新建知识库'} onClose={onClose} footer={<><Button onClick={onClose}>取消</Button><Button variant="primary" loading={saving} disabled={!name.trim()} onClick={save}>保存</Button></>}>
      <div className="col gap-12">
        <Field label="名称 *"><input value={name} onChange={(e) => setName(e.target.value)} placeholder="例如：员工制度" autoFocus /></Field>
        <Field label="描述" hint="查询路由会用名称 + 描述 + 类别判断问题该查哪个库，写清楚库里有什么。"><textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={3} placeholder="人事制度、差旅报销、假期管理…" /></Field>
        <div className="grid grid-2">
          <Field label="业务类别"><input value={category} onChange={(e) => setCategory(e.target.value)} placeholder="制度 / 产品 / 售后 / 运维" /></Field>
          <Field label="语言"><select value={language} onChange={(e) => setLanguage(e.target.value)}><option value="zh">中文</option><option value="en">English</option><option value="mixed">中英混合</option></select></Field>
        </div>
      </div>
    </Modal>
  );
}

function MarkdownDrawer({ doc, onClose, onSaved }: { doc: Document; onClose: () => void; onSaved: (d: Document) => void }) {
  const { toast } = useApp();
  const [text, setText] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  useEffect(() => {
    api.documents.markdown(doc.id).then((r) => setText(r.markdownContent)).catch((e) => { toast(errMsg(e), 'bad'); setText(''); });
  }, [doc.id, toast]);
  const save = async () => {
    if (text === null) return;
    setSaving(true);
    try {
      onSaved(await api.documents.updateMarkdown(doc.id, text));
      toast('已保存并重新切片 / 索引', 'ok');
      onClose();
    } catch (e) {
      toast(errMsg(e), 'bad');
    } finally {
      setSaving(false);
    }
  };
  return (
    <Drawer title="校对 Markdown 并重新索引" subtitle={doc.fileName} onClose={onClose} footer={<><Button onClick={onClose}>取消</Button><Button variant="primary" loading={saving} onClick={save}>保存并重新索引</Button></>}>
      <Callout icon={<AlertCircle size={15} />}>
        <b>Human-in-the-loop 数据治理</b>：解析结果可能存在表格错位、标题层级错误或术语漏识别。修正后保存，系统会级联清理旧切片并重新生成向量、BM25 与实体关系。用 <code>**术语** 是指 …</code> 的写法可让 UnWeaver 抽取定义。
      </Callout>
      {text === null ? <div className="mt-16"><Spinner label="加载中…" /></div> : <textarea className="mono flex-1 mt-12" style={{ minHeight: 400, resize: 'none', lineHeight: 1.6 }} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />}
    </Drawer>
  );
}

function ChunksDrawer({ doc, onClose }: { doc: Document; onClose: () => void }) {
  const { toast } = useApp();
  const [chunks, setChunks] = useState<ChunkView[] | null>(null);
  const [showChildren, setShowChildren] = useState(true);
  useEffect(() => {
    api.documents.chunks(doc.id).then(setChunks).catch((e) => { toast(errMsg(e), 'bad'); setChunks([]); });
  }, [doc.id, toast]);
  const parents = useMemo(() => (chunks ?? []).filter((c) => c.kind === 'parent'), [chunks]);
  const byId = useMemo(() => new Map((chunks ?? []).map((c) => [c.chunkId, c])), [chunks]);
  return (
    <Drawer title="切片预览（Parent-Child）" subtitle={doc.fileName} onClose={onClose}>
      {chunks === null ? <Spinner label="加载中…" /> : (
        <>
          <div className="row between mb-12">
            <div className="row gap-6"><Badge tone="mmr">{parents.length} parents（阅读粒度）</Badge><Badge tone="dense">{chunks.length - parents.length} children（检索粒度）</Badge></div>
            <Button size="sm" onClick={() => setShowChildren((v) => !v)}>{showChildren ? '隐藏子块' : '显示子块'}</Button>
          </div>
          <div className="col gap-12">
            {parents.map((p, i) => (
              <div key={p.chunkId} className="chunk">
                <div className="chunk-head">
                  <span className="row gap-6"><Badge tone="mmr" mono>P{i + 1}</Badge><span className="chunk-crumb">{p.breadcrumbs.join(' › ')}</span></span>
                  <span className="row gap-6">{p.blockTypes.map((t) => <Badge key={t}>{t}</Badge>)}<span className="mono">{p.tokenCount} tok</span></span>
                </div>
                <div className="chunk-text pre">{p.text}</div>
                {p.contextSummary ? <div className="xs muted mt-8">上下文摘要：{p.contextSummary}</div> : null}
                {p.hypotheticalQuestions.length ? <div className="xs muted mt-8">假设性问题：{p.hypotheticalQuestions.join('；')}</div> : null}
                {p.keywords?.length ? <div className="xs muted mt-8 row gap-4 wrap">关键词：{p.keywords.map((k) => <Badge key={k} mono>{k}</Badge>)}</div> : null}
                {p.aliases?.length ? <div className="xs muted mt-8 row gap-4 wrap">别名扩展：{p.aliases.map((k) => <Badge key={k} tone="info" mono>{k}</Badge>)}</div> : null}
                {showChildren && p.childIds.length > 1 ? (
                  <div className="col gap-6 mt-12" style={{ paddingLeft: 14, borderLeft: '2px solid var(--border-strong)' }}>
                    {p.childIds.map((cid, j) => {
                      const c = byId.get(cid);
                      return c ? (
                        <div key={cid} className="small">
                          <span className="row gap-6 xs faint"><Badge tone="dense" mono>C{j + 1}</Badge>{c.tokenCount} tok</span>
                          <div className="pre muted" style={{ marginTop: 2 }}>{c.text}</div>
                        </div>
                      ) : null;
                    })}
                  </div>
                ) : null}
              </div>
            ))}
            {parents.length === 0 ? <Empty title="暂无切片" /> : null}
          </div>
        </>
      )}
    </Drawer>
  );
}

const AUTO_LABEL: Record<string, string> = { title: '标题', version: '版本', valid_from: '生效日期', valid_to: '失效日期', department: '部门', tags: '标签', access_level: '密级' };

function MetaModal({ doc, onClose, onSaved }: { doc: Document; onClose: () => void; onSaved: (d: Document) => void }) {
  const { toast } = useApp();
  const [title, setTitle] = useState(doc.title ?? '');
  const [version, setVersion] = useState(doc.version ?? '');
  const [validFrom, setValidFrom] = useState(doc.validFrom ?? '');
  const [validTo, setValidTo] = useState(doc.validTo ?? '');
  const [tags, setTags] = useState(doc.tags.join(', '));
  const [department, setDepartment] = useState(doc.department ?? '');
  const [accessLevel, setAccessLevel] = useState(doc.accessLevel ?? 0);
  const [allowed, setAllowed] = useState((doc.allowedDepartments ?? []).join(', '));
  const [saving, setSaving] = useState(false);
  const autoApplied = Object.entries(doc.autoMeta ?? {}).filter(([, v]) => v?.applied && v.source !== 'manual') as [string, { evidence: string }][];
  const save = async () => {
    setSaving(true);
    try {
      onSaved(await api.documents.updateMeta(doc.id, { title, version, validFrom, validTo, tags: tags.split(/[,，]/).map((t) => t.trim()).filter(Boolean), department, accessLevel, allowedDepartments: allowed.split(/[,，]/).map((t) => t.trim()).filter(Boolean) }));
      toast('元数据已更新，正在后台重新索引', 'ok');
      onClose();
    } catch (e) {
      toast(errMsg(e), 'bad');
    } finally {
      setSaving(false);
    }
  };
  return (
    <Modal title="文档元数据" onClose={onClose} footer={<><Button onClick={onClose}>取消</Button><Button variant="primary" loading={saving} onClick={save}>保存</Button></>}>
      <div className="col gap-12">
        <Callout>元数据是 RAG 的<b>结构化控制面</b>：生效区间用于检索时过滤过期制度（避免新旧版本同时被召回），标签 / 版本可用于精准过滤与引用展示。</Callout>
        {autoApplied.length ? <div className="xs muted row gap-4 wrap"><Badge tone="accent">自动抽取</Badge>{autoApplied.map(([k, v]) => <span key={k} title={v.evidence}>{AUTO_LABEL[k] ?? k}</span>)}<span className="faint">· 悬停查看依据；可直接修改</span></div> : null}
        <Field label="标题"><input value={title} onChange={(e) => setTitle(e.target.value)} /></Field>
        <div className="grid grid-3">
          <Field label="版本"><input value={version} onChange={(e) => setVersion(e.target.value)} placeholder="2026.1" /></Field>
          <Field label="生效日期"><input type="date" value={validFrom} onChange={(e) => setValidFrom(e.target.value)} /></Field>
          <Field label="失效日期"><input type="date" value={validTo} onChange={(e) => setValidTo(e.target.value)} /></Field>
        </div>
        <Field label="标签（逗号分隔）"><input value={tags} onChange={(e) => setTags(e.target.value)} placeholder="制度, 差旅, 报销" /></Field>
        <div className="divider" />
        <div className="small strong">权限</div>
        <div className="grid grid-3">
          <Field label="所属部门"><input value={department} onChange={(e) => setDepartment(e.target.value)} placeholder="人力资源部" /></Field>
          <Field label="密级（最低可见级别）">
            <select value={accessLevel} onChange={(e) => setAccessLevel(Number(e.target.value))}>{ACCESS_LEVELS.map((l) => <option key={l.value} value={l.value}>{l.doc}</option>)}</select>
          </Field>
          <Field label="仅限部门（逗号分隔，空 = 不限）"><input value={allowed} onChange={(e) => setAllowed(e.target.value)} placeholder="客服部, 法务部" /></Field>
        </div>
        <div className="xs faint">检索与问答时按顶栏"当前身份"过滤：身份级别低于密级、或不在限定部门（管理员除外）的用户看不到这篇文档的任何内容。</div>
      </div>
    </Modal>
  );
}

function Step({ done, children }: { done: boolean; children: React.ReactNode }) {
  return (
    <div className="row gap-8 top small">
      <span className={`badge ${done ? 'ok' : ''}`} style={{ marginTop: 1 }}>{done ? <Check size={10} /> : <X size={10} />}{done ? '完成' : '未完成'}</span>
      <div className="flex-1">{children}</div>
    </div>
  );
}

const isEdited = (d: Document) => (d.parser ?? '').includes('edited');

/** 人工校对过的文档重新解析：默认保留校对（用校对后的 Markdown 按当前配置重建），或明确放弃校对、回到源文件。 */
function ReprocessEditedModal({ doc, onClose, onRun }: { doc: Document; onClose: () => void; onRun: (discardEdits: boolean) => Promise<void> }) {
  const [busy, setBusy] = useState<'keep' | 'discard' | null>(null);
  const run = async (discard: boolean) => { setBusy(discard ? 'discard' : 'keep'); try { await onRun(discard); onClose(); } finally { setBusy(null); } };
  return (
    <Modal title="重新解析校对过的文档" onClose={onClose} footer={<><Button onClick={onClose}>取消</Button>
      <Button variant="danger" loading={busy === 'discard'} disabled={!!busy} onClick={() => run(true)}>放弃校对，从源文件重新解析</Button>
      <Button variant="primary" loading={busy === 'keep'} disabled={!!busy} onClick={() => run(false)}>保留校对，按当前配置重建</Button></>}>
      <div className="col gap-12">
        <Callout icon={<Pencil size={15} />}>「{doc.fileName}」有人工校对过的内容（Markdown 校对回灌），直接从源文件重新解析会把校对内容覆盖掉。</Callout>
        <div className="small"><b>保留校对</b>：用校对后的 Markdown 按当前清洗 / 切片 / 索引配置重建，图片沿用已有的。切换 MinerU 档位、Popo 文档树这类解析层设置不会生效。</div>
        <div className="small"><b>放弃校对</b>：回到源文件重新解析（新的解析档位 / 文档树会生效），校对内容被源文件内容替换；替换前会自动备份校对版，之后可以在文档行的「恢复校对版」里找回。</div>
      </div>
    </Modal>
  );
}

/** 校对版备份：查看、恢复（按该版本重建，当前校对版会先备份）、删除。 */
function EditBackupsModal({ doc, onClose, onRestored }: { doc: Document; onClose: () => void; onRestored: (d: Document) => void }) {
  const { toast } = useApp();
  const [items, setItems] = useState<EditBackup[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const load = useCallback(() => { api.documents.editBackups(doc.id).then((b) => { setItems(b); setOpen((o) => o ?? b[0]?.name ?? null); }).catch((e) => toast(errMsg(e), 'bad')); }, [doc.id, toast]);
  useEffect(load, [load]);
  const restore = async (b: EditBackup) => {
    const msg = isEdited(doc) ? `用 ${formatDate(b.createdAt)} 的校对版重建「${doc.fileName}」？当前的校对内容会先自动备份，可以再恢复回来。` : `用 ${formatDate(b.createdAt)} 的校对版重建「${doc.fileName}」？当前从源文件解析的内容会被替换（之后仍可重新解析源文件）。`;
    if (!confirmDialog(msg)) return;
    setBusy(b.name);
    try { onRestored(await api.documents.restoreEditBackup(doc.id, b.name)); toast('已开始按校对版重建', 'ok'); onClose(); } catch (e) { toast(errMsg(e), 'bad'); } finally { setBusy(null); }
  };
  const remove = async (b: EditBackup) => {
    if (!confirmDialog(`永久删除 ${formatDate(b.createdAt)} 的这份校对版备份？删除后无法找回。`)) return;
    try { await api.documents.deleteEditBackup(doc.id, b.name); toast('已删除备份', 'ok'); load(); } catch (e) { toast(errMsg(e), 'bad'); }
  };
  return (
    <Modal title={`校对版备份 · ${doc.fileName}`} wide onClose={onClose} footer={<Button onClick={onClose}>关闭</Button>}>
      <div className="col gap-12">
        <Callout icon={<History size={15} />}>放弃校对、从源文件重新解析之前，以及恢复另一份备份之前，系统会自动把当时的校对版存一份。恢复 = 把这份内容作为校对结果回灌并重建索引。</Callout>
        {!items ? <Spinner label="读取备份…" /> : !items.length ? <Empty title="没有备份" /> : items.map((b) => (
          <div key={b.name} className="card">
            <div className="card-header" style={{ cursor: 'pointer' }} onClick={() => setOpen(open === b.name ? null : b.name)}>
              <h3><History size={14} />{formatDate(b.createdAt)}</h3>
              <div className="row gap-6">
                <Badge mono>{b.chars} 字</Badge>{b.identical ? <Badge tone="ok">与当前内容相同</Badge> : null}
                <Button size="xs" variant="primary" loading={busy === b.name} disabled={b.identical || !!busy || doc.status === 'PENDING' || doc.status === 'PROCESSING'} onClick={(e) => { e.stopPropagation(); restore(b); }}>恢复此版本</Button>
                <Button size="xs" variant="ghost" icon disabled={!!busy} onClick={(e) => { e.stopPropagation(); remove(b); }} title="删除这份备份" aria-label="删除备份"><Trash2 size={12} /></Button>
              </div>
            </div>
            {open === b.name ? <pre className="card-body xs pre mono" style={{ margin: 0, maxHeight: 260, overflow: 'auto' }}>{b.preview}{b.chars > b.preview.length ? '\n…' : ''}</pre> : null}
          </div>
        ))}
      </div>
    </Modal>
  );
}

function PopoModal({ doc, onClose, onReprocess }: { doc: Document; onClose: () => void; onReprocess: (d: Document) => void }) {
  const { toast } = useApp();
  const [st, setSt] = useState<PopoStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const load = useCallback(() => api.documents.popo(doc.id).then(setSt).catch((e) => toast(errMsg(e), 'bad')), [doc.id, toast]);
  useEffect(() => { load(); }, [load]);
  const exportInputs = async () => {
    setBusy(true);
    try {
      setSt(await api.documents.popoExport(doc.id));
      toast('已写出 MinerU-Popo 输入（Content List / PDF / 归一化块）', 'ok');
    } catch (e) { toast(errMsg(e), 'bad'); } finally { setBusy(false); }
  };
  return (
    <Modal title="MinerU-Popo 文档树" onClose={onClose} footer={<><Button onClick={load}><RefreshCw size={13} />刷新状态</Button><Button onClick={onClose}>关闭</Button></>}>
      <div className="col gap-12">
        <Callout icon={<TreePine size={15} />}>
          MinerU 读懂<b>每一页</b>，Popo 把多页<b>缝成一篇</b>：跨页段落续接、跨页表格合并、全文标题层级重建、图文关联。它只做后处理，不重新解析。
        </Callout>
        {!st ? <Spinner label="读取状态…" /> : !st.applicable ? (
          <Callout tone="warn">Popo 推理需要源 PDF 来渲染页面拼接图，当前文档是 <code>.{doc.fileFormat}</code>。Markdown / Excel 等原生结构化格式本身不存在跨页断裂问题。</Callout>
        ) : (
          <>
            <Step done={st.exported}>
              <div><b className="strong">1. MinerU 解析 + 标签归一化</b></div>
              <div className="xs muted">把 Content List 归一化为 Popo 的块格式（bbox → 0~1 xyxy，页码 1-based），并复制源 PDF。</div>
              <div className="mt-8"><Button size="sm" variant="primary" loading={busy} disabled={!st.mineruAvailable} onClick={exportInputs}>导出 Popo 输入</Button>{!st.mineruAvailable ? <span className="xs text-warning" style={{ marginLeft: 8 }}>需先安装 MinerU：<code>uv pip install -U "mineru[torch]&gt;=4.0,&lt;5"</code></span> : null}</div>
            </Step>
            <Step done={st.treeReady}>
              <div><b className="strong">2. Popo 推理 + 建树</b>（在 MinerU-Popo 仓库内，需 GPU）</div>
              <pre className="chunk pre xs mt-8" style={{ overflowX: 'auto' }}>{st.commands[0]}</pre>
              <div className="xs faint mt-4">{st.inferred ? '已检测到推理结果。' : '四个子任务：文本截断、表格截断、标题层级、图文关联。'}</div>
            </Step>
            <Step done={st.usingTree}>
              <div><b className="strong">3. 用文档树重建索引</b></div>
              <div className="xs muted">在「策略配置 → 文档解析」打开"使用 MinerU-Popo 文档树"，然后重新解析本文档。{st.popoEnabled ? '（开关已打开）' : '（开关当前关闭）'}</div>
              <div className="mt-8"><Button size="sm" disabled={!st.treeReady || !st.popoEnabled} onClick={async () => { const edited = isEdited(doc); if (edited && !confirmDialog('这份文档有人工校对内容。用文档树重新解析需要回到源 PDF，校对内容会被覆盖（会先把校对版备份到上传目录）。继续？')) return; try { onReprocess(await api.documents.reprocess(doc.id, edited)); toast(edited ? '已备份校对版，开始用文档树重新解析' : '已开始用文档树重新解析', 'ok'); onClose(); } catch (e) { toast(errMsg(e), 'bad'); } }}><RotateCw size={13} />重新解析</Button></div>
            </Step>
          </>
        )}
      </div>
    </Modal>
  );
}

function ParsingPanel() {
  const [info, setInfo] = useState<ParsingInfo | null>(null);
  useEffect(() => { api.system.parsing().then(setInfo).catch(() => setInfo(null)); }, []);
  if (!info) return <div className="card-body"><Spinner label="读取解析能力…" /></div>;
  const NEED = '需要 MinerU';
  const groups = new Map<string, string[]>();
  for (const m of [...info.matrix].sort((a, b) => Number(a.parser === null) - Number(b.parser === null))) {
    const key = m.parser ?? NEED;
    groups.set(key, [...(groups.get(key) ?? []), m.ext]);
  }
  const PARSER_NOTE: Record<string, string> = {
    markdown: 'Markdown 直解析（必做）：标题层级 = 章节结构，表格 / 代码是原子块，支持 front matter 元数据',
    excel: 'Excel / CSV 行级结构化：每行一个块并重复表头，避免整表压扁',
    text: '纯文本：按空行分段',
    mineru: 'MinerU 4.x：版面检测、OCR、公式、表格、图片描述，保留页码与 bbox',
    docx: 'python-docx 兜底：标题样式 → 层级，表格转 Markdown',
    pptx: 'python-pptx 兜底：每页一个章节',
    pdfplumber: 'pdfplumber / pypdf 兜底：文字层 + 简单表格；扫描件与复杂版面请用 MinerU',
    '需要 MinerU': '图片 / 扫描件 / 旧版 Office / EPUB / OFD / HTML：安装 MinerU 后可解析',
  };
  return (
    <div className="card-body col gap-16">
      <div>
        <div className="section-label">格式 → 解析器（当前环境）</div>
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>解析器</th><th>扩展名</th><th>说明</th></tr></thead>
            <tbody>
              {[...groups.entries()].map(([parser, exts]) => (
                <tr key={parser}>
                  <td><Badge tone={parser === 'mineru' ? 'ok' : parser === '需要 MinerU' ? 'warn' : 'info'} mono>{parser}</Badge></td>
                  <td><div className="row gap-4 wrap">{exts.map((x) => <Badge key={x} mono>.{x}</Badge>)}</div></td>
                  <td className="muted">{PARSER_NOTE[parser] ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-2" style={{ alignItems: 'start' }}>
        <div>
          <div className="section-label">MinerU 4.x 四档解析 <Badge tone={info.mineru.available ? 'ok' : 'warn'}>{info.mineru.available ? '已安装' : '未安装'}</Badge></div>
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>档位</th><th>用途</th><th>运行要求</th></tr></thead>
              <tbody>
                {info.mineru.tiers.map((t) => (
                  <tr key={t.tier}>
                    <td><Badge mono tone={t.tier === info.mineru.tier ? 'accent' : ''}>{t.tier}</Badge>{t.tier === info.mineru.tier ? <div className="xs faint mt-4">当前</div> : null}</td>
                    <td><div>{t.use}</div><div className="xs faint">{t.models}</div></td>
                    <td className="muted">{t.requires}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!info.mineru.available ? <pre className="chunk pre xs mt-8" style={{ overflowX: 'auto' }}>{info.mineru.install.join('\n')}</pre> : null}
          <div className="xs faint mt-8">档位、OCR 模式与图片分析在「策略配置 → 文档解析」里切换；切换后对文档点「重新解析」生效。</div>
        </div>
        <div>
          <div className="section-label">MinerU-Popo 文档树 <Badge tone={info.popo.enabled ? 'ok' : ''}>{info.popo.enabled ? '已启用' : '未启用'}</Badge>{info.popo.trees ? <Badge mono>{info.popo.trees} 棵树</Badge> : null}</div>
          <div className="col gap-6">
            {info.popo.subtasks.map((t) => <div key={t} className="small">· {t}</div>)}
          </div>
          <div className="divider" />
          <ol className="small muted" style={{ paddingLeft: 18, lineHeight: 1.8 }}>{info.popo.steps.map((t) => <li key={t}>{t}</li>)}</ol>
          <div className="xs faint mt-8">轻量需求（只要标题分级 / 跨页表格续接）可直接用 MinerU 4.0 内置的 LLM 辅助后处理；需要完整文档树、图文关联或多 OCR 通用时再接 Popo。</div>
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ doc }: { doc: Document }) {
  const tone = doc.status === 'COMPLETED' ? 'ok' : doc.status === 'FAILED' ? 'bad' : doc.status === 'PROCESSING' ? 'info' : 'warn';
  return <Badge tone={tone}>{doc.status === 'PROCESSING' || doc.status === 'PENDING' ? <Loader2 size={10} className="spin" /> : null}{STATUS_LABEL[doc.status]}</Badge>;
}

export function KnowledgeBasesPage() {
  const { kbs, currentKb, setCurrentKbId, refreshKbs, refreshStatus, toast } = useApp();
  const [kbModal, setKbModal] = useState<{ kb?: KnowledgeBase } | null>(null);
  const [reprocessDoc, setReprocessDoc] = useState<Document | null>(null);
  const [docs, setDocs] = useState<Document[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [formats, setFormats] = useState<{ extensions: string[]; mineru: boolean } | null>(null);
  const [uploading, setUploading] = useState(0);
  const [over, setOver] = useState(false);
  const [tab, setTab] = useState<'docs' | 'entities' | 'parsing'>('docs');
  const [entities, setEntities] = useState<EntityRelation[]>([]);
  const [entityQuery, setEntityQuery] = useState('');
  const [entityView, setEntityView] = useState<'graph' | 'list'>('graph');
  const [graph, setGraph] = useState<KbGraph | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [mdDoc, setMdDoc] = useState<Document | null>(null);
  const [chunkDoc, setChunkDoc] = useState<Document | null>(null);
  const [metaDoc, setMetaDoc] = useState<Document | null>(null);
  const [popoDoc, setPopoDoc] = useState<Document | null>(null);
  const [backupDoc, setBackupDoc] = useState<Document | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const kbId = currentKb?.id ?? null;

  const loadDocs = useCallback(async (id: string, silent = false) => {
    if (!silent) setLoadingDocs(true);
    try {
      setDocs(await api.documents.list(id));
    } catch (e) {
      if (!silent) toast(errMsg(e), 'bad');
    } finally {
      setLoadingDocs(false);
    }
  }, [toast]);

  useEffect(() => { api.documents.supportedFormats().then(setFormats).catch(() => undefined); }, []);
  useEffect(() => {
    setDocs([]);
    setEntities([]);
    setGraph(null);
    if (kbId) loadDocs(kbId);
  }, [kbId, loadDocs]);
  useEffect(() => {
    if (!kbId || tab !== 'entities') return;
    api.kb.entities(kbId).then(setEntities).catch(() => setEntities([]));
    setGraphLoading((v) => v || graph === null);
    // 轮询时内容没变就保留原对象：避免拓扑图重新布局、把用户的缩放 / 拖动复位
    api.kb.graph(kbId)
      .then((g) => setGraph((prev) => (prev && JSON.stringify(prev) === JSON.stringify(g) ? prev : g)))
      .catch(() => setGraph(null))
      .finally(() => setGraphLoading(false));
  }, [kbId, tab, docs]); // eslint-disable-line

  // 轮询处理中的文档
  const busy = docs.some((d) => d.status === 'PENDING' || d.status === 'PROCESSING') || !!currentKb?.needsReindex || !!currentKb?.reindexing;
  useEffect(() => {
    if (!kbId || !busy) return;
    const t = window.setInterval(async () => {
      const list = await api.documents.list(kbId).catch(() => null);
      if (!list) return;
      setDocs(list);
      if (!list.some((d) => d.status === 'PENDING' || d.status === 'PROCESSING')) {
        refreshKbs().catch(() => undefined);
        refreshStatus();
      }
    }, 2000);
    return () => window.clearInterval(t);
  }, [kbId, busy, refreshKbs, refreshStatus]);

  const upload = async (files: FileList | File[]) => {
    if (!kbId) return;
    const list = Array.from(files);
    setUploading(list.length);
    for (const f of list) {
      try {
        const d = await api.documents.upload(kbId, f);
        setDocs((prev) => [d, ...prev]);
      } catch (e) {
        toast(`${f.name}：${errMsg(e)}`, 'bad');
      }
      setUploading((n) => n - 1);
    }
    toast(`已提交 ${list.length} 个文件，后台解析与索引中`, 'ok');
  };
  const removeDoc = async (d: Document) => {
    if (!confirmDialog(`确定删除文档"${d.fileName}"？对应的切片、向量与实体会一并清除。`)) return;
    try {
      await api.documents.remove(d.id);
      setDocs((prev) => prev.filter((x) => x.id !== d.id));
      refreshKbs();
      toast('文档已删除', 'ok');
    } catch (e) {
      toast(errMsg(e), 'bad');
    }
  };
  const removeKb = async (kb: KnowledgeBase) => {
    if (!confirmDialog(`确定删除知识库"${kb.name}"？其下所有文档与索引都会被删除。`)) return;
    try {
      await api.kb.remove(kb.id);
      await refreshKbs();
      refreshStatus();
      toast('知识库已删除', 'ok');
    } catch (e) {
      toast(errMsg(e), 'bad');
    }
  };
  const reindex = async () => {
    if (!kbId) return;
    try {
      await api.kb.reindex(kbId);
      toast('已开始重建索引', 'ok');
      setTimeout(() => loadDocs(kbId, true), 800);
    } catch (e) {
      toast(errMsg(e), 'bad');
    }
  };
  const runReprocess = async (d: Document, discardEdits: boolean) => {
    try {
      replaceDoc(await api.documents.reprocess(d.id, discardEdits));
      toast(isEdited(d) && !discardEdits ? `已开始用校对后的内容重建"${d.fileName}"` : `已开始按当前解析 / 切片配置重新处理"${d.fileName}"${discardEdits ? '（校对版已备份，可点「恢复校对版」找回）' : ''}`, 'ok');
    } catch (e) {
      toast(errMsg(e), 'bad');
    }
  };
  // 校对过的文档先让用户选：保留校对重建，还是放弃校对回到源文件
  const reprocess = (d: Document) => (isEdited(d) ? setReprocessDoc(d) : runReprocess(d, false));
  const replaceDoc = (d: Document) => { setDocs((prev) => prev.map((x) => (x.id === d.id ? d : x))); refreshKbs(); };
  const filteredEntities = entities.filter((e) => !entityQuery || e.entity.toLowerCase().includes(entityQuery.toLowerCase()) || e.definition.toLowerCase().includes(entityQuery.toLowerCase()));

  return (
    <div className="page layout-split fade-in">
      <div className="card side">
        <div className="card-header">
          <h3><Database size={15} />知识库</h3>
          <Button size="sm" variant="primary" onClick={() => setKbModal({})}><Plus size={13} />新建</Button>
        </div>
        <div className="card-body col gap-4" style={{ padding: 10 }}>
          {kbs.length === 0 ? <Empty title="暂无知识库" desc="按业务域拆库，避免不同类型的内容互相干扰。" /> : null}
          {kbs.map((kb) => (
            <div key={kb.id} className={`list-item ${kb.id === kbId ? 'active' : ''}`} onClick={() => setCurrentKbId(kb.id)}>
              <div className="flex-1">
                <div className="strong ellipsis" title={kb.name}>{kb.name}</div>
                <div className="xs faint clamp-2">{kb.description || '暂无描述'}</div>
                <div className="xs muted mt-8 row gap-6 wrap"><span className="mono">{kb.documentCount} 文档 · {kb.chunkCount} 切片 · {kb.entityCount} 实体</span>{kb.category ? <Badge>{kb.category}</Badge> : null}</div>
              </div>
              <div className="col gap-4">
                <button className="btn ghost icon xs" title="编辑" onClick={(e) => { e.stopPropagation(); setKbModal({ kb }); }}><Pencil size={12} /></button>
                <button className="btn ghost icon xs" title="删除" onClick={(e) => { e.stopPropagation(); removeKb(kb); }}><Trash2 size={12} /></button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {!currentKb ? (
        <div className="card"><Empty icon={<Database size={40} />} title="请选择或新建一个知识库" desc="知识库承载文档、切片、向量、BM25 索引与实体图；建议按业务域拆库，避免不同类型内容互相干扰。" action={<Button variant="primary" onClick={() => setKbModal({})}><Plus size={13} />新建知识库</Button>} /></div>
      ) : (
        <div className="col gap-16">
          {currentKb.reindexing ? (
            <Callout icon={<RefreshCw size={15} className="spin" />}><b>正在自动重建索引</b>：配置或向量模型发生了变化，系统正按新参数重新处理本知识库的文档，完成后自动恢复，无需操作。</Callout>
          ) : currentKb.needsReindex ? (
            <Callout tone="warn" icon={<AlertCircle size={15} />}>
              <b>索引与当前向量模型不一致</b>（索引构建于 {currentKb.indexEmbedder}）。向量空间不同会得到无意义的检索结果，请重建索引。
              <div className="mt-8"><Button size="sm" onClick={reindex}><RefreshCw size={13} />重建索引</Button></div>
            </Callout>
          ) : null}

          <div className={`dropzone ${over ? 'over' : ''}`} onClick={() => fileRef.current?.click()} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={(e) => { e.preventDefault(); setOver(false); if (e.dataTransfer.files.length) upload(e.dataTransfer.files); }}>
            <input ref={fileRef} type="file" multiple hidden accept={formats?.extensions.map((x) => `.${x}`).join(',')} onChange={(e) => { if (e.target.files?.length) upload(e.target.files); e.target.value = ''; }} />
            <div className="row gap-8" style={{ justifyContent: 'center' }}>{uploading > 0 ? <Loader2 size={18} className="spin" /> : <Upload size={18} />}<span className="strong">{uploading > 0 ? `上传中（剩余 ${uploading}）…` : '点击或拖拽文件到这里上传'}</span></div>
            <div className="small faint mt-8">{formats?.mineru ? 'PDF / 图片 / Office 由 MinerU 4.x 解析' : 'PDF / Word / PPT 由内置解析器处理（安装 MinerU 后支持扫描件、图片与复杂版面）'} · Markdown 直解析 · Excel / CSV 行级结构化</div>
            <div className="xs faint mt-4 mono">{formats?.extensions.map((x) => `.${x}`).join('  ')}</div>
          </div>

          <div className="card">
            <div className="card-header">
              <Tabs value={tab} onChange={setTab} items={[{ value: 'docs', label: `文档（${docs.length}）`, icon: <FileText size={13} /> }, { value: 'entities', label: '实体织网', icon: <Network size={13} /> }, { value: 'parsing', label: '解析能力', icon: <ScanText size={13} /> }]} />
              <div className="row gap-8">
                <Button size="sm" onClick={reindex} title="按当前解析 / 切片 / 向量配置重建整库索引"><RefreshCw size={13} />重建索引</Button>
                <Button size="sm" variant="ghost" icon onClick={() => kbId && loadDocs(kbId)}><RefreshCw size={13} /></Button>
              </div>
            </div>
            {tab === 'docs' ? (
              <div>
                {loadingDocs ? <div className="card-body"><Spinner label="加载中…" /></div> : docs.length === 0 ? <Empty icon={<FileText size={36} />} title="知识库是空的" desc="上传 PDF / Word / PPT / Excel / Markdown 文档，系统会自动解析、清洗、切片并建立向量 + BM25 索引。" /> : (
                  <div>
                    {docs.map((d) => {
                      const expired = !!d.validTo && d.validTo < new Date().toISOString().slice(0, 10);
                      return (
                        <div key={d.id} className="doc-row">
                          <div className="doc-main">
                            <div className="row gap-8" style={{ minWidth: 0 }}>
                              <span className="strong ellipsis" title={d.fileName}>{d.fileName}</span>
                              <StatusBadge doc={d} />
                              {expired ? <Badge tone="bad" title="已过失效日期：检索时自动过滤">已过期</Badge> : null}
                            </div>
                            <div className="doc-meta">
                              <span className="mono">{d.fileFormat}</span>
                              <span>{formatBytes(d.fileSize)}</span>
                              <span title="解析器" className="mono">{d.parser ?? '—'}{d.parserTier ? ` · ${d.parserTier}` : ''}</span>
                              <span title="parent（阅读粒度） / child（检索粒度）"><b className="mono">{d.parentCount}</b> 父块 / <b className="mono">{d.chunkCount}</b> 子块</span>
                              {d.entityCount ? <span><b className="mono">{d.entityCount}</b> 实体</span> : null}
                              {d.version ? <span>v{d.version}</span> : null}
                              {d.accessLevel ? <Badge tone={d.accessLevel >= 2 ? 'warn' : 'info'}>{ACCESS_LEVELS[d.accessLevel]?.doc ?? `级别 ${d.accessLevel}`}</Badge> : null}
                              {d.allowedDepartments?.length ? <Badge title="仅这些部门可见">限 {d.allowedDepartments.join('、')}</Badge> : null}
                              {d.validFrom || d.validTo ? <span className="mono">{d.validFrom ?? '…'} ~ {d.validTo ?? '…'}</span> : null}
                              <span>更新于 {formatDate(d.updatedAt)}</span>
                              {d.tags.filter((t) => t !== '已过期').map((t) => <Badge key={t}>{t}</Badge>)}
                            </div>
                            {d.errorMessage ? <div className="xs text-danger mt-4">{d.errorMessage}</div> : null}
                          </div>
                          <div className="doc-actions">
                            <Button size="xs" disabled={d.status !== 'COMPLETED'} onClick={() => setChunkDoc(d)} title="查看 Parent-Child 切片"><Layers size={11} />切片</Button>
                            <Button size="xs" disabled={d.status !== 'COMPLETED'} onClick={() => setMdDoc(d)} title="校对 Markdown 并重新索引"><Pencil size={11} />校对</Button>
                            <Button size="xs" onClick={() => setMetaDoc(d)} title="版本 / 生效区间 / 标签"><Tags size={11} />元数据</Button>
                            {d.fileFormat === 'pdf' ? <Button size="xs" onClick={() => setPopoDoc(d)} title="MinerU-Popo 文档树"><TreePine size={11} />Popo</Button> : null}
                            {d.editBackups ? <Button size="xs" onClick={() => setBackupDoc(d)} title="查看 / 恢复校对版备份"><History size={11} />恢复校对版 · {d.editBackups}</Button> : null}
                            <Button size="xs" variant="ghost" icon disabled={d.status === 'PENDING' || d.status === 'PROCESSING'} onClick={() => reprocess(d)} title="按当前解析 / 切片配置重新解析" aria-label="重新解析"><RotateCw size={13} /></Button>
                            <Button size="xs" variant="ghost" icon onClick={() => removeDoc(d)} title="删除文档" aria-label="删除文档"><Trash2 size={13} /></Button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            ) : tab === 'parsing' ? <ParsingPanel /> : (
              <div className="card-body col gap-12">
                <div className="row between wrap gap-8">
                  <Tabs value={entityView} onChange={setEntityView} items={[{ value: 'graph', label: '拓扑图', icon: <Share2 size={13} /> }, { value: 'list', label: `实体列表（${entities.length}）`, icon: <ListIcon size={13} /> }]} />
                  {graph ? (
                    <div className="row gap-6 wrap">
                      <Badge>{graph.stats.docs} 文档</Badge>
                      <Badge>{graph.stats.sections} 章节</Badge>
                      <Badge tone="weaved">{graph.stats.entities} 实体</Badge>
                      <Badge tone="hyde">{graph.stats.relations} 关系</Badge>
                      <Badge>{graph.stats.mentions} 提及</Badge>
                      {graph.stats.truncated ? <Badge tone="warn" title="按出现次数保留最常见的实体">共 {graph.stats.totalEntities} 个，仅显示前 {graph.stats.entities}</Badge> : null}
                    </div>
                  ) : null}
                </div>
                {currentKb?.reindexing || currentKb?.needsReindex ? <Callout icon={<Loader2 size={14} className="spin" />}>正在重建索引（含实体抽取），完成后拓扑图自动刷新。</Callout> : null}
                {graph && !graph.config.unweaver ? <Callout tone="warn">UnWeaver 实体织网已关闭（策略配置 → 检索 → 图谱），只显示文档结构。</Callout> : graph && !graph.config.llmExtraction && graph.stats.entities === 0 ? (
                  <Callout tone="warn">当前只用正则抽取 <code>**术语** 是指 …</code> 这类显式定义，本知识库没有命中。到「策略配置」开启 <b>LLM 三元组抽取</b>，保存后会自动重建索引并抽取实体与关系。</Callout>
                ) : null}
                {entityView === 'graph' ? (
                  graphLoading && !graph ? <Spinner label="正在生成拓扑…" />
                    : !graph || graph.nodes.length === 0 ? <Empty icon={<Network size={36} />} title="还没有可展示的结构" desc="上传文档后，这里会按标题路径展示「文档 → 章节 → 小节」的结构，并挂上 UnWeaver 抽取到的实体与关系。" />
                    : <EntityGraph graph={graph} title={currentKb?.name} onOpenList={(name) => { setEntityQuery(name); setEntityView('list'); }} onExported={(name, err) => toast(err ? `导出失败：${err}` : `已导出 ${name}`, err ? 'bad' : 'ok')} />
                ) : (
                  <>
                    <div className="row gap-8"><Search size={14} className="faint" /><input placeholder="搜索实体或定义…" value={entityQuery} onChange={(e) => setEntityQuery(e.target.value)} /></div>
                    {filteredEntities.length === 0 ? <Empty icon={<Network size={36} />} title={entities.length ? '没有匹配的实体' : '暂无实体'} desc={<>UnWeaver 会从文档中抽取 <code>**术语** 是指 …</code>、<code>「术语」是 …</code>、<code>X is defined as …</code> 等定义；开启"LLM 三元组抽取"后还会抽取实体之间的关系。检索时若上下文提到术语而定义块未被召回，会自动织入。</>} /> : (
                      <table className="table">
                        <thead><tr><th>实体</th><th>关系</th><th>定义 / 宾语</th><th>来源</th></tr></thead>
                        <tbody>{filteredEntities.map((e, i) => (
                          <tr key={i}><td className="strong">{e.entity}</td><td><Badge tone={e.relation_type === 'DEFINED_IN' ? 'weaved' : 'info'}>{e.relation_type === 'DEFINED_IN' ? '定义' : '关系'}</Badge></td><td className="muted">{e.relation_type === 'RELATED_TO' ? `${e.definition} → ${e.target}` : e.definition}</td><td><Badge>{e.source === 'llm' ? 'LLM' : '正则'}</Badge></td></tr>
                        ))}</tbody>
                      </table>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {kbModal ? <KbForm initial={kbModal.kb} onClose={() => setKbModal(null)} onSaved={async (kb) => { setKbModal(null); await refreshKbs(); setCurrentKbId(kb.id); refreshStatus(); }} /> : null}
      {mdDoc ? <MarkdownDrawer doc={mdDoc} onClose={() => setMdDoc(null)} onSaved={replaceDoc} /> : null}
      {chunkDoc ? <ChunksDrawer doc={chunkDoc} onClose={() => setChunkDoc(null)} /> : null}
      {popoDoc ? <PopoModal doc={popoDoc} onClose={() => setPopoDoc(null)} onReprocess={replaceDoc} /> : null}
      {backupDoc ? <EditBackupsModal doc={backupDoc} onClose={() => setBackupDoc(null)} onRestored={replaceDoc} /> : null}
      {reprocessDoc ? <ReprocessEditedModal doc={reprocessDoc} onClose={() => setReprocessDoc(null)} onRun={(discard) => runReprocess(reprocessDoc, discard)} /> : null}
      {metaDoc ? <MetaModal doc={metaDoc} onClose={() => setMetaDoc(null)} onSaved={(d) => { replaceDoc(d); if (kbId) setTimeout(() => loadDocs(kbId, true), 600); }} /> : null}
    </div>
  );
}
