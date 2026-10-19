// 飞书文档：接入 / 同步进知识库、远程增删改查、文档智能体（写操作默认需确认）、应用配置（含自动同步 / 事件订阅）。
import { useCallback, useEffect, useRef, useState } from 'react';
import { BellRing, Bot, CheckCircle2, ExternalLink, FilePlus2, FolderOpen, Link2, Pencil, Play, PlugZap, RefreshCw, Save, Settings2, Square, Trash2, Unlink, Wrench } from 'lucide-react';
import { API_BASE, ApiError } from '../api/client';
import { feishuApi, type FeishuAction, type FeishuAgentEvent, type FeishuConfig, type FeishuFile, type FeishuLink, type FeishuRemoteDoc } from '../api/feishu';
import { fmtTime } from '../api/lance';
import { KbSelect } from '../components/kb-select';
import { Markdown } from '../components/markdown';
import { Badge, Button, Callout, Drawer, Empty, Field, Modal, Spinner, Tabs, Toggle, confirmDialog } from '../components/ui';
import { useApp } from '../store/app-store';

type Tab = 'docs' | 'agent' | 'config';
const errText = (e: unknown) => (e instanceof ApiError ? e.message : String(e));
const STATUS: Record<string, { label: string; tone: string }> = {
  SYNCED: { label: '已同步', tone: 'ok' }, FAILED: { label: '失败', tone: 'bad' }, MISSING: { label: '本地已删除', tone: 'warn' }, PENDING: { label: '待同步', tone: '' },
};
const AUTO_SYNC_OPTIONS: [number, string][] = [[0, '关闭'], [5, '每 5 分钟'], [15, '每 15 分钟'], [30, '每 30 分钟'], [60, '每小时'], [180, '每 3 小时'], [720, '每 12 小时'], [1440, '每天']];
const TOOL_LABEL: Record<string, string> = {
  list_docs: '列出文档', search: '检索知识库', read_doc: '读取文档', read_kb_doc: '读取知识库文档', create_doc: '新建文档', append_doc: '追加内容', replace_doc: '整篇替换',
  replace_text: '查找替换', delete_doc: '删除文档', sync_doc: '同步入库',
};

export function FeishuPage() {
  const { currentKb, toast } = useApp();
  const [tab, setTab] = useState<Tab>('docs');
  const [config, setConfig] = useState<FeishuConfig | null>(null);
  const loadConfig = useCallback(() => feishuApi.config().then(setConfig).catch((e) => toast(errText(e), 'bad')), [toast]);
  useEffect(() => { loadConfig(); }, [loadConfig]);
  useEffect(() => { if (config && !config.configured) setTab('config'); }, [config]);

  return (
    <div className="page col gap-16 fade-in">
      <div className="card">
        <div className="card-body row between wrap gap-8">
          <div className="row gap-8 wrap">
            <KbSelect />
            {config ? <Badge tone={config.configured ? 'ok' : 'warn'}>{config.configured ? `应用 ${config.app_id}${config.fromEnv ? '（环境变量）' : ''}` : '未配置飞书应用'}</Badge> : null}
          </div>
          <Tabs<Tab> value={tab} onChange={setTab} items={[
            { value: 'docs', label: '文档', icon: <Link2 size={13} /> },
            { value: 'agent', label: '文档智能体', icon: <Bot size={13} /> },
            { value: 'config', label: '应用配置', icon: <Settings2 size={13} /> },
          ]} />
        </div>
      </div>
      {tab === 'docs' ? (currentKb ? <DocsTab kbId={currentKb.id} /> : <div className="card"><Empty title="先选择一个知识库" desc="飞书文档会同步进当前知识库，之后检索、问答、评测都能用到。" /></div>) : null}
      {tab === 'agent' ? <AgentTab kbId={currentKb?.id} config={config} /> : null}
      {tab === 'config' ? <ConfigTab config={config} onSaved={setConfig} /> : null}
    </div>
  );
}

// ======================================================================================
// 文档：接入 / 同步 / 查看编辑 / 删除
// ======================================================================================
function DocsTab({ kbId }: { kbId: string }) {
  const { toast, refreshKbs } = useApp();
  const [links, setLinks] = useState<FeishuLink[]>([]);
  const [loading, setLoading] = useState(true);
  const [refs, setRefs] = useState('');
  const [busy, setBusy] = useState<string | null>(null);
  const [editing, setEditing] = useState<FeishuLink | { documentId: string; title: string; url: string | null } | null>(null);
  const [creating, setCreating] = useState(false);
  const [browsing, setBrowsing] = useState(false);

  const load = useCallback(() => feishuApi.links(kbId).then(setLinks).catch((e) => toast(errText(e), 'bad')).finally(() => setLoading(false)), [kbId, toast]);
  useEffect(() => { setLoading(true); load(); }, [load]);
  // 后台入库期间轮询状态
  useEffect(() => {
    if (!links.some((l) => l.localStatus === 'PENDING' || l.localStatus === 'PROCESSING')) return;
    const t = window.setTimeout(load, 2000);
    return () => window.clearTimeout(t);
  }, [links, load]);

  const run = async (key: string, fn: () => Promise<unknown>, ok?: string) => {
    setBusy(key);
    try {
      await fn();
      if (ok) toast(ok, 'ok');
    } catch (e) {
      toast(errText(e), 'bad');
    } finally {
      setBusy(null);
      load();
      refreshKbs().catch(() => undefined);
    }
  };
  const importRefs = (list: string[]) => run('import', async () => {
    const res = await feishuApi.import(kbId, list);
    const bad = res.filter((r) => !r.ok);
    const good = res.length - bad.length;
    if (good) toast(`已接入 ${good} 篇，后台解析入库中`, 'ok');
    bad.forEach((r) => toast(`${r.ref}：${r.error}`, 'bad'));
    if (!bad.length) setRefs('');
  });
  const syncAll = () => run('syncAll', async () => {
    const res = await feishuApi.syncKb(kbId);
    const changed = res.filter((r) => r.ok && r.status !== 'unchanged').length;
    const failed = res.filter((r) => !r.ok).length;
    toast(`检查了 ${res.length} 篇：${changed} 篇有更新${failed ? `，${failed} 篇失败` : ''}`, failed ? 'warn' : 'ok');
  });
  const deleteRemote = (l: FeishuLink) => {
    if (!confirmDialog(`删除飞书文档「${l.title}」？文档会进入飞书回收站，同时从知识库移除。`)) return;
    run(`del-${l.id}`, () => feishuApi.remove(l.documentId, kbId), '已删除飞书文档');
  };
  const unlink = (l: FeishuLink) => {
    if (!confirmDialog(`从知识库移除「${l.title}」？飞书上的文档不受影响。`)) return;
    run(`unlink-${l.id}`, () => feishuApi.unlink(l.id), '已从知识库移除');
  };

  return (
    <>
      <div className="card">
        <div className="card-header">
          <h3><Link2 size={14} />接入飞书文档</h3>
          <div className="row gap-6">
            <Button size="sm" onClick={() => setBrowsing(true)}><FolderOpen size={13} />浏览文件夹</Button>
            <Button size="sm" onClick={() => setCreating(true)}><FilePlus2 size={13} />新建文档</Button>
          </div>
        </div>
        <div className="card-body col gap-8">
          <textarea rows={3} value={refs} onChange={(e) => setRefs(e.target.value)} placeholder={'每行一个飞书文档链接（docx / wiki）或 document_id，例如\nhttps://xxx.feishu.cn/docx/doxcnAbCdEf…\nhttps://xxx.feishu.cn/wiki/wikcnXyZ…'} />
          <div className="row between wrap gap-8">
            <span className="xs faint">文档要对应用可见：在飞书文档右上角「…→ 添加文档应用」里加上本应用。同步时转成 Markdown（保留标题层级、列表、表格、代码），按版本号判断是否需要重新入库。</span>
            <Button variant="primary" size="sm" loading={busy === 'import'} disabled={!refs.trim()} onClick={() => importRefs(refs.split('\n').map((s) => s.trim()).filter(Boolean))}>接入并同步</Button>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>已接入的飞书文档<Badge mono>{links.length}</Badge></h3>
          <Button size="sm" loading={busy === 'syncAll'} disabled={!links.length} onClick={syncAll}><RefreshCw size={13} />检查更新</Button>
        </div>
        {loading ? <div className="card-body"><Spinner label="加载中…" /></div> : links.length === 0 ? (
          <div className="card-body"><Empty title="还没有接入飞书文档" desc="粘贴链接接入，或从飞书文件夹里挑选。" /></div>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>文档</th><th>状态</th><th>切片</th><th>上次同步</th><th style={{ width: 230 }}>操作</th></tr></thead>
              <tbody>
                {links.map((l) => (
                  <tr key={l.id}>
                    <td>
                      <div className="row gap-6"><span className="strong">{l.title || l.documentId}</span>{l.url ? <a href={l.url} target="_blank" rel="noreferrer" title="在飞书中打开"><ExternalLink size={12} /></a> : null}</div>
                      <div className="xs faint mono">{l.documentId}{l.revisionId ? ` · v${l.revisionId}` : ''}</div>
                      {l.errorMessage ? <div className="xs text-danger clamp-2" title={l.errorMessage}>{l.errorMessage}</div> : null}
                    </td>
                    <td>
                      <Badge tone={STATUS[l.status]?.tone}>{STATUS[l.status]?.label ?? l.status}</Badge>
                      {l.localStatus && l.localStatus !== 'COMPLETED' ? <Badge tone={l.localStatus === 'FAILED' ? 'bad' : 'warn'}>{l.localStatus === 'FAILED' ? '解析失败' : '入库中'}</Badge> : null}
                    </td>
                    <td className="mono">{l.chunkCount || '—'}</td>
                    <td className="xs muted">{fmtTime(l.syncedAt)}</td>
                    <td>
                      <div className="row gap-4">
                        <Button size="xs" variant="ghost" onClick={() => setEditing(l)} title="查看 / 编辑飞书上的内容"><Pencil size={12} />编辑</Button>
                        <Button size="xs" variant="ghost" loading={busy === `sync-${l.id}`} onClick={() => run(`sync-${l.id}`, async () => {
                          const r = await feishuApi.syncLink(l.id, l.status !== 'SYNCED');
                          toast(r.status === 'unchanged' ? '飞书上没有改动' : '已重新同步，后台解析入库中', 'ok');
                        })} title="从飞书拉取最新内容"><RefreshCw size={12} />同步</Button>
                        <Button size="xs" variant="ghost" icon loading={busy === `unlink-${l.id}`} onClick={() => unlink(l)} title="从知识库移除（不动飞书）"><Unlink size={12} /></Button>
                        <Button size="xs" variant="ghost" icon loading={busy === `del-${l.id}`} onClick={() => deleteRemote(l)} title="删除飞书文档"><Trash2 size={12} /></Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {editing ? <EditorDrawer kbId={kbId} target={editing} onClose={() => setEditing(null)} onChanged={load} /> : null}
      {creating ? <CreateModal kbId={kbId} onClose={() => setCreating(false)} onCreated={() => { setCreating(false); load(); refreshKbs().catch(() => undefined); }} /> : null}
      {browsing ? <FolderModal kbId={kbId} onClose={() => setBrowsing(false)} onImport={(ids) => { setBrowsing(false); importRefs(ids); }} /> : null}
    </>
  );
}

function EditorDrawer({ kbId, target, onClose, onChanged }: { kbId: string; target: { documentId: string; title: string | null; url: string | null }; onClose: () => void; onChanged: () => void }) {
  const { toast } = useApp();
  const [doc, setDoc] = useState<FeishuRemoteDoc | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<'view' | 'replace' | 'append' | 'find'>('view');
  const [draft, setDraft] = useState('');
  const [find, setFind] = useState('');
  const [repl, setRepl] = useState('');
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    setError(null);
    feishuApi.read(target.documentId, kbId, true).then((d) => { setDoc(d); setDraft(d.markdown.replace(/^# .*\n+/, '')); }).catch((e) => setError(errText(e)));
  }, [target.documentId, kbId]);
  useEffect(() => { load(); }, [load]);

  const act = async (fn: () => Promise<string>) => {
    setBusy(true);
    try {
      toast(await fn(), 'ok');
      setMode('view');
      load();
      onChanged();
    } catch (e) {
      toast(errText(e), 'bad');
    } finally {
      setBusy(false);
    }
  };
  const save = () => {
    if (mode === 'replace') {
      if (!confirmDialog('用编辑后的内容整篇替换飞书文档正文？图片、画板等非文字块会被删除。')) return;
      act(async () => { const r = await feishuApi.write({ ref: target.documentId, markdown: draft, mode: 'replace', kbId }); return `已替换：删除 ${r.removedBlocks} 块，写入 ${r.blocks} 块`; });
    } else if (mode === 'append') {
      act(async () => { const r = await feishuApi.write({ ref: target.documentId, markdown: draft, mode: 'append', kbId }); return `已追加 ${r.blocks} 块`; });
    } else if (mode === 'find') {
      act(async () => {
        const r = await feishuApi.replaceText({ ref: target.documentId, find, replace: repl, kbId });
        return `替换了 ${r.replaced} 处（${r.blocks} 个段落）${r.skippedStyledMatches ? `；${r.skippedStyledMatches} 处跨了样式边界，未处理` : ''}`;
      });
    }
  };

  return (
    <Drawer title={doc?.title ?? target.title ?? target.documentId} subtitle={doc ? `${doc.chars} 字 · ${doc.blocks} 块 · v${doc.revisionId ?? '—'}` : '读取飞书上的最新内容'} onClose={onClose}
      footer={mode === 'view' ? (
        <div className="row gap-6">
          {target.url ? <a className="btn sm ghost" href={target.url} target="_blank" rel="noreferrer"><ExternalLink size={12} />在飞书中打开</a> : null}
          <Button size="sm" onClick={() => { setDraft(''); setMode('append'); }} disabled={!doc}>追加内容</Button>
          <Button size="sm" onClick={() => setMode('find')} disabled={!doc}>查找替换</Button>
          <Button size="sm" variant="primary" onClick={() => { setDraft(doc?.markdown.replace(/^# .*\n+/, '') ?? ''); setMode('replace'); }} disabled={!doc}><Pencil size={12} />编辑全文</Button>
        </div>
      ) : (
        <div className="row gap-6">
          <Button size="sm" variant="ghost" onClick={() => setMode('view')} disabled={busy}>取消</Button>
          <Button size="sm" variant="primary" loading={busy} onClick={save} disabled={mode === 'find' ? !find : !draft.trim()}><Save size={12} />写回飞书</Button>
        </div>
      )}>
      {error ? <Callout tone="bad">{error}</Callout> : !doc ? <Spinner label="读取中…" /> : mode === 'view' ? (doc.html ? <div className="fs-paper" dangerouslySetInnerHTML={{ __html: doc.html }} /> : <Markdown text={doc.markdown} />) : mode === 'find' ? (
        <div className="col gap-12">
          <Field label="查找" hint="只替换同一样式内的文字（保留加粗 / 链接等样式）"><input value={find} onChange={(e) => setFind(e.target.value)} /></Field>
          <Field label="替换为"><input value={repl} onChange={(e) => setRepl(e.target.value)} /></Field>
        </div>
      ) : (
        <div className="col gap-8 flex-1">
          {mode === 'replace' ? <Callout tone="warn">整篇替换会删除原正文（含图片等非文字块）再写入；只改几个字请用「查找替换」。</Callout> : <div className="xs faint">内容会追加到文档末尾。</div>}
          <textarea className="mono flex-1" style={{ minHeight: 360 }} value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Markdown：标题 / 列表 / 表格 / 代码块 / 链接" />
        </div>
      )}
    </Drawer>
  );
}

function CreateModal({ kbId, onClose, onCreated }: { kbId: string; onClose: () => void; onCreated: () => void }) {
  const { toast } = useApp();
  const [title, setTitle] = useState('');
  const [markdown, setMarkdown] = useState('');
  const [sync, setSync] = useState(true);
  const [busy, setBusy] = useState(false);
  const create = async () => {
    setBusy(true);
    try {
      const r = await feishuApi.create({ title, markdown, kbId: sync ? kbId : undefined });
      toast(`已新建「${r.title}」${sync ? '，并同步进知识库' : ''}`, 'ok');
      onCreated();
    } catch (e) {
      toast(errText(e), 'bad');
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal title="新建飞书文档" onClose={onClose} wide footer={<>
      <Button variant="ghost" onClick={onClose}>取消</Button>
      <Button variant="primary" loading={busy} disabled={!title.trim()} onClick={create}><FilePlus2 size={13} />新建</Button>
    </>}>
      <div className="col gap-12">
        <Field label="标题"><input value={title} onChange={(e) => setTitle(e.target.value)} autoFocus /></Field>
        <Field label="正文（Markdown）"><textarea className="mono" rows={12} value={markdown} onChange={(e) => setMarkdown(e.target.value)} /></Field>
        <Field label="同步进当前知识库" inline><Toggle on={sync} onChange={setSync} /></Field>
        <div className="xs faint">文档建在「应用配置」里的默认文件夹；没配置时建在应用自己的云空间根目录（只有应用可见，需要在飞书里把文档分享给自己）。</div>
      </div>
    </Modal>
  );
}

function FolderModal({ kbId, onClose, onImport }: { kbId: string; onClose: () => void; onImport: (ids: string[]) => void }) {
  const [folder, setFolder] = useState('');
  const [files, setFiles] = useState<FeishuFile[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const load = useCallback((token?: string) => {
    setFiles(null);
    setError(null);
    feishuApi.folder(token || undefined).then((r) => setFiles(r.files)).catch((e) => setError(errText(e)));
  }, []);
  useEffect(() => { load(); }, [load]);
  const docs = (files ?? []).filter((f) => f.type === 'docx');
  const toggle = (t: string) => setPicked((s) => { const n = new Set(s); if (n.has(t)) n.delete(t); else n.add(t); return n; });
  return (
    <Modal title="从飞书文件夹接入" onClose={onClose} wide footer={<>
      <Button variant="ghost" onClick={onClose}>取消</Button>
      <Button variant="primary" disabled={!picked.size} onClick={() => onImport([...picked])}>接入所选（{picked.size}）</Button>
    </>}>
      <div className="col gap-12">
        <div className="row gap-6">
          <input className="flex-1" value={folder} onChange={(e) => setFolder(e.target.value)} placeholder="文件夹链接或 token；留空 = 应用配置里的默认文件夹" />
          <Button size="sm" onClick={() => load(folder)}>打开</Button>
        </div>
        {error ? <Callout tone="bad">{error}</Callout> : files === null ? <Spinner label="读取文件夹…" /> : docs.length === 0 ? <Empty title="这个文件夹里没有新版文档" desc={files.length ? `有 ${files.length} 个其他类型的文件（只支持 docx）` : undefined} /> : (
          <div className="col gap-4">
            {docs.map((f) => (
              <label key={f.token} className="chunk row gap-8" style={{ cursor: 'pointer' }}>
                <input type="checkbox" style={{ width: 'auto' }} checked={picked.has(f.token)} onChange={() => toggle(f.token)} disabled={f.linkedKbs.includes(kbId)} />
                <span className="flex-1 ellipsis">{f.name}</span>
                {f.linkedKbs.includes(kbId) ? <Badge tone="ok">已接入</Badge> : null}
              </label>
            ))}
          </div>
        )}
      </div>
    </Modal>
  );
}

// ======================================================================================
// 文档智能体
// ======================================================================================
const EXAMPLES = [
  '列出已接入的飞书文档，并概括每篇讲了什么',
  '对比知识库里几篇制度文档，找出相互矛盾的规定',
  '根据知识库内容写一篇「新员工入职须知」飞书文档',
  '把《值班表》里的「张三」改成「李四」',
];

function AgentTab({ kbId, config }: { kbId?: string; config: FeishuConfig | null }) {
  const { toast } = useApp();
  const [question, setQuestion] = useState(EXAMPLES[0]);
  const [allowWrite, setAllowWrite] = useState(false);
  const [events, setEvents] = useState<FeishuAgentEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState<Record<string, 'ok' | 'error' | 'running'>>({});
  const [writeOut, setWriteOut] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => { if (config) setAllowWrite(config.agent_allow_write); }, [config]);

  const run = async () => {
    if (!question.trim()) return;
    setEvents([]);
    setDone({});
    setRunning(true);
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      await feishuApi.run({ kbId, question, allowWrite }, (e) => setEvents((xs) => [...xs, e]), ac.signal);
    } catch (e) {
      if (!ac.signal.aborted) toast(`运行失败：${errText(e)}`, 'bad');
    } finally {
      setRunning(false);
    }
  };
  const execute = async (a: FeishuAction) => {
    setDone((d) => ({ ...d, [a.id]: 'running' }));
    try {
      await feishuApi.execute({ kbId, tool: a.tool, args: a.args });
      setDone((d) => ({ ...d, [a.id]: 'ok' }));
      toast(`已执行：${TOOL_LABEL[a.tool] ?? a.tool}`, 'ok');
    } catch (e) {
      setDone((d) => ({ ...d, [a.id]: 'error' }));
      toast(errText(e), 'bad');
    }
  };

  return (
    <>
      <div className="card">
        <div className="card-body col gap-12">
          <div className="row gap-8 wrap">
            <input className="flex-1" style={{ minWidth: 280 }} value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && !running && run()} placeholder="对飞书文档做什么？查询、总结、对比、新建、修改、删除…" />
            <label className="row gap-6 xs muted" title="关闭时，智能体的新建 / 修改 / 删除只会列出来，等你确认后再执行"><Toggle on={allowWrite} onChange={setAllowWrite} />直接写飞书</label>
            {running ? <Button variant="danger" onClick={() => abortRef.current?.abort()}><Square size={13} />停止</Button> : <Button variant="primary" onClick={run} disabled={!config?.configured}><Play size={13} />运行</Button>}
          </div>
          <div className="row gap-6 wrap">{EXAMPLES.map((q) => <button key={q} className="badge" style={{ cursor: 'pointer' }} onClick={() => setQuestion(q)}>{q}</button>)}</div>
          <div className="xs faint">工具：列出 / 检索（当前知识库的 RAG 检索）/ 读取 / 新建 / 追加 / 整篇替换 / 查找替换 / 删除 / 同步入库。分析（总结、对比、找矛盾）由模型基于读到的原文完成；改过的文档会自动重新同步进知识库。{kbId ? '' : '未选择知识库：检索与同步不可用。'}</div>
        </div>
      </div>
      {events.length === 0 && !running ? <div className="card"><Empty icon={<Bot size={32} />} title="还没有运行" desc="输入要求后点「运行」，这里会显示智能体的每一步。" /></div> : (
        <div className="col gap-12">
          {events.map((e, i) => <AgentEventCard key={i} e={e} done={done} onExecute={execute} onWriteOut={(md) => setWriteOut(md)} />)}
          {running ? <div className="card"><div className="card-body"><Spinner label="智能体工作中…" /></div></div> : null}
        </div>
      )}
      {writeOut !== null ? <WriteOutModal kbId={kbId} markdown={writeOut} defaultTitle={question.slice(0, 40)} onClose={() => setWriteOut(null)} /> : null}
    </>
  );
}

// 把智能体的结论写进飞书：新建文档，或追加到已有文档（填链接）
function WriteOutModal({ kbId, markdown, defaultTitle, onClose }: { kbId?: string; markdown: string; defaultTitle: string; onClose: () => void }) {
  const { toast } = useApp();
  const [mode, setMode] = useState<'create' | 'append'>('create');
  const [title, setTitle] = useState(defaultTitle);
  const [ref, setRef] = useState('');
  const [body, setBody] = useState(markdown);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<{ url?: string; warn?: string } | null>(null);
  const submit = async () => {
    setBusy(true);
    try {
      if (mode === 'create') {
        const r = await feishuApi.create({ title, markdown: body });
        setDone({ url: r.url, warn: r.shareError });
      } else {
        const r = await feishuApi.write({ ref, markdown: body, mode: 'append', kbId });
        setDone({ url: r.url });
      }
      toast(mode === 'create' ? '已新建飞书文档' : '已追加到飞书文档', 'ok');
    } catch (e) {
      toast(errText(e), 'bad');
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal title="写进飞书文档" onClose={onClose} wide footer={done ? <Button variant="primary" onClick={onClose}>完成</Button> : <>
      <Button variant="ghost" onClick={onClose}>取消</Button>
      <Button variant="primary" loading={busy} disabled={!body.trim() || (mode === 'create' ? !title.trim() : !ref.trim())} onClick={submit}><Save size={13} />写入</Button>
    </>}>
      {done ? (
        <div className="col gap-8">
          <Callout>已写入。{done.url ? <a href={done.url} target="_blank" rel="noreferrer">在飞书中打开 <ExternalLink size={11} /></a> : null}</Callout>
          {done.warn ? <Callout tone="warn">{done.warn}</Callout> : null}
        </div>
      ) : (
        <div className="col gap-12">
          <div className="row gap-6">
            <Button size="sm" variant={mode === 'create' ? 'primary' : 'ghost'} onClick={() => setMode('create')}>新建文档</Button>
            <Button size="sm" variant={mode === 'append' ? 'primary' : 'ghost'} onClick={() => setMode('append')}>追加到已有文档</Button>
          </div>
          {mode === 'create'
            ? <Field label="标题"><input value={title} onChange={(e) => setTitle(e.target.value)} /></Field>
            : <Field label="飞书文档链接" hint="文档要对应用可见且可编辑（文档右上角「…→ 添加文档应用」）"><input value={ref} onChange={(e) => setRef(e.target.value)} placeholder="https://xxx.feishu.cn/wiki/… 或 /docx/…" /></Field>}
          <Field label="正文（Markdown）"><textarea className="mono" rows={14} value={body} onChange={(e) => setBody(e.target.value)} /></Field>
        </div>
      )}
    </Modal>
  );
}

function AgentEventCard({ e, done, onExecute, onWriteOut }: { e: FeishuAgentEvent; done: Record<string, string>; onExecute: (a: FeishuAction) => void; onWriteOut: (markdown: string) => void }) {
  const d = e.data || {};
  const head = (label: React.ReactNode, tone = '') => (
    <div className="card-header"><h3><Badge tone={tone}>{label}</Badge>{e.content && e.type !== 'final' ? <span className="small muted">{e.content}</span> : null}</h3><span className="mono xs faint">+{(e.ms / 1000).toFixed(1)}s</span></div>
  );
  if (e.type === 'run') {
    return <div className="card">{head('开始', 'info')}<div className="card-body row gap-6 wrap xs">{d.offline ? <Badge tone="warn">未配置文本模型：只能按规则查询</Badge> : null}<Badge tone={d.allowWrite ? 'warn' : ''}>{d.allowWrite ? '写操作直接执行' : '写操作需确认'}</Badge></div></div>;
  }
  if (e.type === 'plan' || e.type === 'reflect') {
    return (
      <div className="card">{head(e.type === 'plan' ? '规划' : '反思', 'fused')}
        <div className="card-body col gap-4">{(d.steps as any[]).length ? (d.steps as any[]).map((s, i) => <div key={i} className="small"><b className="mono">{i + 1}.</b> {TOOL_LABEL[s.tool] ?? s.tool}<span className="faint"> — {s.why}</span></div>) : <span className="xs muted">不需要调用工具</span>}</div>
      </div>
    );
  }
  if (e.type === 'tool' || e.type === 'pending') {
    const a = d as FeishuAction & { result?: any; elapsedMs?: number };
    const state = done[a.id];
    return (
      <div className="card">{head(<><Wrench size={11} />{TOOL_LABEL[a.tool] ?? a.tool}</>, a.status === 'error' ? 'bad' : a.status === 'pending' ? 'warn' : 'dense')}
        <div className="card-body col gap-8">
          <ArgsView args={a.args} />
          {a.status === 'error' ? <Callout tone="bad">{a.error}</Callout> : null}
          {a.status === 'ok' ? <ResultView tool={a.tool} result={a.result} /> : null}
          {a.status === 'pending' ? (
            <div className="row gap-8">
              {state === 'ok' ? <Badge tone="ok"><CheckCircle2 size={11} />已执行</Badge> : <Button size="sm" variant="primary" loading={state === 'running'} onClick={() => onExecute(a)}>确认执行</Button>}
              <span className="xs faint">智能体提出的写操作，确认后才会改动飞书</span>
            </div>
          ) : null}
        </div>
      </div>
    );
  }
  if (e.type === 'final') {
    return (
      <div className="card">{head('结论', 'ok')}
        <div className="card-body col gap-12">
          <Markdown text={e.content} />
          <div className="row gap-8"><Button size="sm" onClick={() => onWriteOut(e.content)}><FilePlus2 size={12} />写进飞书文档</Button><span className="xs faint">把这段结论新建成飞书文档，或追加到已有文档</span></div>
        </div>
      </div>
    );
  }
  if (e.type === 'error') return <div className="card">{head('出错', 'bad')}<div className="card-body"><Callout tone="bad">{e.content}</Callout></div></div>;
  return null;
}

function ArgsView({ args }: { args: Record<string, any> }) {
  const entries = Object.entries(args || {});
  if (!entries.length) return null;
  return (
    <div className="col gap-4">
      {entries.map(([k, v]) => {
        const text = typeof v === 'string' ? v : JSON.stringify(v);
        return k === 'markdown' ? (
          <details key={k}><summary className="xs muted">markdown（{text.length} 字）</summary><div className="mt-4"><Markdown text={text} /></div></details>
        ) : <div key={k} className="xs"><span className="faint mono">{k}</span> <span className="pre">{text}</span></div>;
      })}
    </div>
  );
}

function ResultView({ tool, result }: { tool: string; result: any }) {
  if (tool === 'search' && Array.isArray(result)) {
    return <div className="col gap-4">{result.map((h: any, i: number) => <div key={i} className="chunk"><div className="chunk-head"><span className="ellipsis">{h.title ?? h.label}</span><span className="mono xs faint">{Number(h.score).toFixed(3)}</span></div><div className="xs muted clamp-2">{h.text}</div></div>)}</div>;
  }
  if ((tool === 'create_doc' || tool === 'append_doc' || tool === 'replace_doc') && result?.documentId) {
    return (
      <div className="col gap-6 small">
        <div>{tool === 'create_doc' ? '已新建' : '已写入'}《{result.title ?? result.documentId}》{result.blocks ? `，${result.blocks} 块` : ''}{result.url ? <> · <a href={result.url} target="_blank" rel="noreferrer">在飞书中打开 <ExternalLink size={11} /></a></> : null}</div>
        {result.shareError ? <Callout tone="warn">{result.shareError}</Callout> : null}
      </div>
    );
  }
  if ((tool === 'read_doc' || tool === 'read_kb_doc') && result?.markdown) {
    return <details><summary className="xs muted">《{result.title}》{result.chars} 字{result.truncated ? '（已截断）' : ''}</summary><div className="mt-4"><Markdown text={result.markdown} /></div></details>;
  }
  if (tool === 'list_docs' && result?.docs) {
    return <div className="col gap-4">{(result.docs as any[]).map((d) => <div key={d.documentId} className="xs"><b>{d.title}</b> <span className="faint mono">{d.documentId}</span></div>)}</div>;
  }
  return <div className="xs pre muted">{JSON.stringify(result, null, 1).slice(0, 800)}</div>;
}

// ======================================================================================
// 应用配置
// ======================================================================================
function ConfigTab({ config, onSaved }: { config: FeishuConfig | null; onSaved: (c: FeishuConfig) => void }) {
  const { toast, kbs } = useApp();
  const [form, setForm] = useState<FeishuConfig | null>(config);
  const [busy, setBusy] = useState<'save' | 'test' | 'syncNow' | 'subscribe' | null>(null);
  useEffect(() => { setForm(config); }, [config]);
  if (!form) return <div className="card"><div className="card-body"><Spinner label="加载中…" /></div></div>;
  const set = <K extends keyof FeishuConfig>(k: K, v: FeishuConfig[K]) => setForm({ ...form, [k]: v });
  const save = async () => {
    setBusy('save');
    try {
      onSaved(await feishuApi.saveConfig(form));
      toast('已保存', 'ok');
    } catch (e) {
      toast(errText(e), 'bad');
    } finally {
      setBusy(null);
    }
  };
  const test = async () => {
    setBusy('test');
    try {
      const r = await feishuApi.test();
      toast(r.ok ? `连接成功：文件夹里有 ${r.folder?.files ?? 0} 个文件` : `连接失败：${r.error}`, r.ok ? 'ok' : 'bad');
    } finally {
      setBusy(null);
    }
  };
  const syncNow = async () => {
    setBusy('syncNow');
    try {
      const r = await feishuApi.syncAutoNow();
      toast('skipped' in r ? '上一轮同步还没结束' : `检查 ${r.checked} 篇，更新 ${r.updated} 篇${r.failed ? `，失败 ${r.failed} 篇` : ''}`, 'skipped' in r || r.failed ? 'warn' : 'ok');
      onSaved(await feishuApi.config());
    } catch (e) {
      toast(errText(e), 'bad');
    } finally {
      setBusy(null);
    }
  };
  const subscribe = async () => {
    setBusy('subscribe');
    try {
      const r = await feishuApi.subscribeEvents();
      const failed = Object.values(r.failed);
      toast(`已订阅 ${r.subscribed} / ${r.total} 篇${failed.length ? `；失败：${failed[0]}` : ''}`, failed.length ? 'warn' : 'ok');
    } catch (e) {
      toast(errText(e), 'bad');
    } finally {
      setBusy(null);
    }
  };
  const eventUrl = new URL(`${API_BASE}/feishu/events`, window.location.origin).href;
  const last = config?.autoSync.last;
  const intervals = AUTO_SYNC_OPTIONS.some(([m]) => m === form.auto_sync_minutes) ? AUTO_SYNC_OPTIONS : [...AUTO_SYNC_OPTIONS, [form.auto_sync_minutes, `每 ${form.auto_sync_minutes} 分钟`] as [number, string]];
  return (
    <>
    <div className="card">
      <div className="card-header">
        <h3><PlugZap size={14} />飞书自建应用</h3>
        <div className="row gap-6">
          <Button size="sm" loading={busy === 'test'} onClick={test} disabled={!config?.configured}>测试连接</Button>
          <Button size="sm" variant="primary" loading={busy === 'save'} onClick={save}><Save size={13} />保存</Button>
        </div>
      </div>
      <div className="card-body col gap-12">
        <Callout>在 <a href="https://open.feishu.cn/app" target="_blank" rel="noreferrer">飞书开放平台</a> 创建企业自建应用 → 开通权限 <span className="mono">docx:document</span>、<span className="mono">drive:drive</span>、<span className="mono">wiki:wiki:readonly</span>、<span className="mono">docs:document.media:download</span> → 创建版本并发布；再把要用的文档 / 文件夹「添加文档应用」给它（知识库里的文档同样在文档右上角「…」里添加）。</Callout>
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
          <Field label="App ID"><input value={form.app_id} onChange={(e) => set('app_id', e.target.value)} placeholder="cli_xxx" /></Field>
          <Field label="App Secret" hint={form.app_secret === '********' ? '已保存；不改就保持原值' : undefined}><input type="password" value={form.app_secret} onChange={(e) => set('app_secret', e.target.value)} /></Field>
          <Field label="版本">
            <select value={form.domain} onChange={(e) => set('domain', e.target.value as FeishuConfig['domain'])}><option value="feishu">飞书（open.feishu.cn）</option><option value="lark">Lark 海外版（open.larksuite.com）</option></select>
          </Field>
          <Field label="文档链接前缀" hint="如 https://xxx.feishu.cn；留空用 feishu.cn"><input value={form.web_base} onChange={(e) => set('web_base', e.target.value)} /></Field>
          <Field label="默认文件夹" hint="新建文档放这里，也是「浏览文件夹」的默认目录；填文件夹 token"><input value={form.folder_token} onChange={(e) => set('folder_token', e.target.value)} /></Field>
          <Field label="智能体默认知识库" hint="页面上没选知识库时使用">
            <select value={form.default_kb_id} onChange={(e) => set('default_kb_id', e.target.value)}><option value="">（不指定）</option>{kbs.map((k) => <option key={k.id} value={k.id}>{k.name}</option>)}</select>
          </Field>
          <Field label="新建文档的链接分享" hint="应用新建的文档默认只有应用自己能看；打开后组织内拿到链接的人可以访问（需开通 docs:permission.setting:write_only 或 drive:drive）">
            <select value={form.new_doc_share} onChange={(e) => set('new_doc_share', e.target.value as FeishuConfig['new_doc_share'])}><option value="tenant_editable">组织内获得链接可编辑</option><option value="tenant_readable">组织内获得链接可阅读</option><option value="closed">不改（只有应用可见）</option></select>
          </Field>
          <Field label="智能体最多调用工具次数"><input type="number" min={1} max={12} value={form.agent_max_steps} onChange={(e) => set('agent_max_steps', Number(e.target.value) || 6)} /></Field>
        </div>
        <Field label="写入表格时用飞书官方 Markdown 转换（保留表格结构）" inline><Toggle on={form.use_convert_api} onChange={(v) => set('use_convert_api', v)} /></Field>
        <Field label="读取文档里的图片：OCR 识别图中文字，有多模态模型时再生成图片描述（截图 / 幻灯片为主的文档靠它才能检索和总结；需开通 docs:document.media:download）" inline><Toggle on={form.read_images} onChange={(v) => set('read_images', v)} /></Field>
        <Field label="智能体默认直接写飞书（关闭则新建 / 修改 / 删除需在页面上确认）" inline><Toggle on={form.agent_allow_write} onChange={(v) => set('agent_allow_write', v)} /></Field>
      </div>
    </div>

    <div className="card">
      <div className="card-header">
        <h3><BellRing size={14} />自动同步</h3>
        <div className="row gap-6">
          <Button size="sm" loading={busy === 'syncNow'} onClick={syncNow} disabled={!config?.configured}><RefreshCw size={13} />立即同步一轮</Button>
          <Button size="sm" variant="primary" loading={busy === 'save'} onClick={save}><Save size={13} />保存</Button>
        </div>
      </div>
      <div className="card-body col gap-12">
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
          <Field label="定时同步" hint={last ? `上次：${fmtTime(last.at)}，检查 ${last.checked} 篇，更新 ${last.updated} 篇${last.failed + last.errors.length ? `，失败 ${last.failed}` : ''}` : '按版本号增量同步所有知识库里接入的飞书文档，没改过的跳过'}>
            <select value={form.auto_sync_minutes} onChange={(e) => set('auto_sync_minutes', Number(e.target.value))}>{intervals.map(([m, label]) => <option key={m} value={m}>{label}</option>)}</select>
          </Field>
          <Field label="事件订阅 Verification Token" hint={form.event_verification_token === '********' ? '已保存；不改就保持原值' : '配置后飞书改了文档会立即推送，只同步那一篇'}>
            <input type="password" value={form.event_verification_token} onChange={(e) => set('event_verification_token', e.target.value)} />
          </Field>
          <Field label="事件订阅 Encrypt Key" hint={form.event_encrypt_key === '********' ? '已保存；不改就保持原值' : '飞书后台设置了 Encrypt Key 时填写，否则留空'}>
            <input type="password" value={form.event_encrypt_key} onChange={(e) => set('event_encrypt_key', e.target.value)} />
          </Field>
        </div>
        <Callout>
          事件订阅（可选，比定时同步更及时）：飞书开发者后台「事件与回调」→ 请求地址填 <span className="mono">{eventUrl}</span>（需公网可访问）→ 添加事件「文档编辑」<span className="mono">drive.file.edit_v1</span>。
          飞书只推送订阅过的文档：配置 Token 后新接入的文档会自动订阅，之前接入的点
          <Button size="sm" loading={busy === 'subscribe'} onClick={subscribe} disabled={!config?.configured || !config.event_verification_token}>为已接入文档订阅</Button>
        </Callout>
      </div>
    </div>
    </>
  );
}
