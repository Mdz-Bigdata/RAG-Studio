import { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, Bot, File as FileIcon, FileText, Images, Mic, MicOff, Network, Paperclip, Plus, Send, Settings2, Sparkles, Square, ThumbsDown, ThumbsUp, Trash2, User, X } from 'lucide-react';
import { api, streamChat } from '../api/client';
import { lakeApi } from '../api/lake';
import type { ChatMessage, ChatSession, CitationIssue, ImageHitView, RetrievalMode, RetrievalOverrides } from '../api/types';
import { ContextCard } from '../components/context-card';
import { IMAGE_ROUTE, ImageCard, ImageDetail } from '../components/image-card';
import { KbSelect } from '../components/kb-select';
import { Markdown } from '../components/markdown';
import { MODE_INFO, StrategyPanel } from '../components/strategy-panel';
import { Badge, Button, Callout, Drawer, Empty, Tabs } from '../components/ui';
import { formatDate } from '../lib/format';
import { useApp } from '../store/app-store';

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));
const IMAGE_EXT = ['png', 'jpg', 'jpeg', 'webp', 'gif'];
const TEXT_EXT = ['txt', 'md', 'markdown'];
const issueText = (c: CitationIssue) => {
  const s = `「${c.sentence.length > 24 ? `${c.sentence.slice(0, 24)}…` : c.sentence}」`;
  if (c.reason === 'out_of_range') return `[${c.index}] 超出本轮资料编号`;
  if (c.reason === 'number') return `${s}中的数值 ${c.value ?? ''} 在本轮资料里找不到`;
  return c.reason === 'mismatch' ? `[${c.index}]${s}与资料 ${c.index} 不符，更像资料 ${c.suggest}` : `[${c.index}]${s}在本轮资料里找不到依据`;
};
// 历史消息：图片随 trace 持久化，取出来放到消息上，和流式过程中的结构保持一致
const hydrate = (m: ChatMessage): ChatMessage => ({ ...m, images: m.images ?? m.trace?.images ?? [], imageCitations: m.imageCitations ?? m.trace?.imageCitations ?? [], queryImage: m.queryImage ?? m.trace?.queryImage ?? null });

export function ChatPage() {
  const { currentKb, kbs, status, toast, identity } = useApp();
  const [kbChoice, setKbChoice] = useState<string>(currentKb?.id ?? 'auto');
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [mode, setMode] = useState<RetrievalMode>('hybrid');
  const [overrides, setOverrides] = useState<RetrievalOverrides>({});
  const [showStrategy, setShowStrategy] = useState(false);
  const [image, setImage] = useState<string | null>(null);
  const [tempDoc, setTempDoc] = useState<{ name: string; content: string } | null>(null);
  const [parsing, setParsing] = useState(false);
  const [recording, setRecording] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);   // 选中的 assistant 消息下标
  const [activeCite, setActiveCite] = useState<number | null>(null);
  const [sideTab, setSideTab] = useState<'sources' | 'images' | 'trace'>('sources');
  const [detail, setDetail] = useState<ImageHitView | null>(null);
  const [feedback, setFeedback] = useState<Record<string, -1 | 0 | 1>>({});   // 回答反馈（写进 Agent 湖，点踩的回答不再作为历史参考）
  const endRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const recRef = useRef<any>(null);

  useEffect(() => { if (currentKb && kbChoice !== 'auto') setKbChoice(currentKb.id); }, [currentKb]); // eslint-disable-line
  const kbId = kbChoice === 'auto' ? null : kbChoice;

  const loadSessions = useCallback(async () => {
    try { setSessions(await api.chat.sessions(kbId)); } catch { /* ignore */ }
  }, [kbId]);
  useEffect(() => { loadSessions(); }, [loadSessions]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const openSession = async (id: string) => {
    try {
      const msgs = (await api.chat.messages(id)).map(hydrate);
      setSessionId(id);
      setMessages(msgs);
      setSelected(msgs.length ? msgs.length - 1 : null);
    } catch (e) { toast(errMsg(e), 'bad'); }
  };
  const newSession = () => { setSessionId(null); setMessages([]); setSelected(null); setActiveCite(null); };
  const deleteSession = async (id: string) => {
    try {
      await api.chat.deleteSession(id);
      if (id === sessionId) newSession();
      loadSessions();
    } catch (e) { toast(errMsg(e), 'bad'); }
  };

  // 语音输入（Web Speech API）
  useEffect(() => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) return;
    const rec = new SR();
    rec.continuous = true; rec.interimResults = true; rec.lang = 'zh-CN';
    rec.onresult = (ev: any) => {
      let final = '';
      for (let i = ev.resultIndex; i < ev.results.length; i++) if (ev.results[i].isFinal) final += ev.results[i][0].transcript;
      if (final) setInput((p) => p + final);
    };
    rec.onerror = () => setRecording(false);
    rec.onend = () => setRecording(false);
    recRef.current = rec;
    return () => { try { rec.stop(); } catch { /* ignore */ } };
  }, []);
  const toggleRecording = () => {
    if (!recRef.current) { toast('当前浏览器不支持语音输入（请使用 Chrome / Edge / Safari）', 'warn'); return; }
    if (recording) { recRef.current.stop(); setRecording(false); } else { try { recRef.current.start(); setRecording(true); } catch { /* ignore */ } }
  };

  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
    if (IMAGE_EXT.includes(ext)) {
      const r = new FileReader();
      r.onload = () => setImage(r.result as string);
      r.readAsDataURL(file);
    } else if (TEXT_EXT.includes(ext)) {
      setTempDoc({ name: file.name, content: await file.text() });
    } else {
      setParsing(true);
      try {
        const res = await api.documents.parseTemp(file);
        setTempDoc({ name: file.name, content: res.content });
      } catch (err) { toast(`临时解析失败：${errMsg(err)}`, 'bad'); } finally { setParsing(false); }
    }
  };

  // kbOverride：在建议的知识库里重新提问（setKbChoice 是异步的，这一次直接用指定的库）
  const send = async (text = input, kbOverride?: string) => {
    const message = text.trim();
    if ((!message && !image) || sending) return;   // 只发图片也可以：以图搜图 / 以图搜文
    if (kbs.length === 0) { toast('请先创建知识库并上传文档', 'warn'); return; }
    setInput('');
    setSending(true);
    setActiveCite(null);
    const now = new Date().toISOString();
    const userMsg: ChatMessage = { id: `u-${now}`, role: 'user', content: (message || (image ? '（图片检索）' : '')) + (tempDoc ? `\n\n[附件] ${tempDoc.name}` : ''), sources: [], relations: [], citations: [], trace: null, model: null, createdAt: now, queryImage: image };
    const botMsg: ChatMessage = { id: `a-${now}`, role: 'assistant', content: '', sources: [], relations: [], citations: [], trace: null, model: null, createdAt: now, streaming: true, images: [], imageCitations: [] };
    const base = messages.length;
    setMessages((m) => [...m, userMsg, botMsg]);
    setSelected(base + 1);
    const patch = (fn: (m: ChatMessage) => ChatMessage) => setMessages((list) => list.map((m, i) => (i === base + 1 ? fn(m) : m)));
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const img = image, doc = tempDoc;
    setImage(null); setTempDoc(null);
    try {
      await streamChat({ kbId: kbOverride ?? kbId ?? 'auto', sessionId, message, imageB64: img, tempDocContent: doc?.content ?? null, mode, overrides, user: identity }, {
        onMeta: (meta) => { setSessionId(meta.sessionId); patch((m) => ({ ...m, meta })); },
        onPlan: (trace) => patch((m) => ({ ...m, trace })),
        onSources: (sources) => patch((m) => ({ ...m, sources })),
        onRelations: (relations) => patch((m) => ({ ...m, relations })),
        onImages: (images) => patch((m) => ({ ...m, images })),
        onToken: (t) => patch((m) => ({ ...m, content: m.content + t })),
        onError: (error) => patch((m) => ({ ...m, error })),
        onDone: (d) => patch((m) => ({ ...m, id: (d as { messageId?: string | null }).messageId ?? m.id, streaming: false, citations: d.citations, imageCitations: d.imageCitations ?? [], model: d.model,
          trace: m.trace ? { ...m.trace, citationCheck: (d as { citationCheck?: CitationIssue[] }).citationCheck ?? [] } : m.trace })),
      }, ctrl.signal);
    } catch (e) {
      if ((e as any)?.name !== 'AbortError') patch((m) => ({ ...m, error: errMsg(e), content: m.content || '请求失败，请检查后端服务。' }));
    } finally {
      patch((m) => ({ ...m, streaming: false }));
      setSending(false);
      abortRef.current = null;
      loadSessions();
    }
  };

  const sel = selected !== null ? messages[selected] : null;
  const suggestions = ['年假可以跨年使用吗？', '订单 SO-8817 现在是什么状态？', 'MySQL 1062 有哪几种处理方案？', '给我看一下系统架构图'];
  const selImages = sel?.images ?? [];
  const openImage = (i: number, n: number) => {
    const im = messages[i]?.images?.find((x) => x.index === n);
    setSelected(i);
    setSideTab('images');
    if (im) setDetail(im);
  };

  return (
    <div className="layout-chat">
      {/* 会话列表 */}
      <div className="card col chat-sessions" style={{ minHeight: 0, gap: 0 }}>
        <div className="card-header"><h3>会话</h3><Button size="xs" variant="primary" onClick={newSession}><Plus size={12} />新对话</Button></div>
        <div className="scroll flex-1 col gap-4" style={{ padding: 8 }}>
          {sessions.length === 0 ? <div className="xs faint" style={{ padding: 12, textAlign: 'center' }}>暂无历史会话</div> : null}
          {sessions.map((s) => (
            <div key={s.id} className={`list-item ${s.id === sessionId ? 'active' : ''}`} style={{ padding: '8px 10px' }} onClick={() => openSession(s.id)}>
              <div className="flex-1"><div className="small ellipsis">{s.title || '未命名会话'}</div><div className="xs faint">{formatDate(s.updatedAt)} · {s.messageCount} 条</div></div>
              <button className="btn ghost icon xs" onClick={(e) => { e.stopPropagation(); deleteSession(s.id); }}><Trash2 size={11} /></button>
            </div>
          ))}
        </div>
      </div>

      {/* 对话区 */}
      <div className="card col chat-main" style={{ minHeight: 0, gap: 0 }}>
        <div className="card-header">
          <div className="row gap-8">
            <KbSelect allowAuto value={kbChoice} onChange={setKbChoice} />
            <Badge>{MODE_INFO[mode].icon}{MODE_INFO[mode].label}</Badge>
            {Object.keys(overrides).length ? <Badge tone="warn">已覆盖 {Object.keys(overrides).length} 组参数</Badge> : null}
          </div>
          <Button size="sm" onClick={() => setShowStrategy(true)}><Settings2 size={13} />检索策略</Button>
        </div>

        <div className="scroll flex-1 col gap-16" style={{ padding: 20 }}>
          {messages.length === 0 ? (
            <Empty icon={<Sparkles size={40} />} title="基于知识库的可追溯问答" desc={<>回答严格遵循协议：只依据资料、标注引用 [n]、证据不足时拒答、多版本冲突时指出。回答可以带图片（[图n]）；点回形针上传一张图片，即可以图搜图、以图搜文。{status?.runtime.llm.offline ? ' 当前未配置 LLM，将返回检索到的原文（离线抽取模式）。' : ''}</>} action={
              <div className="row gap-6 wrap" style={{ justifyContent: 'center', maxWidth: 560 }}>{suggestions.map((s) => <button key={s} className="btn sm chip" onClick={() => send(s)}>{s}</button>)}</div>
            } />
          ) : null}
          {messages.map((m, i) => (
            <div key={m.id} className={`msg ${m.role}`}>
              <div className="avatar">{m.role === 'user' ? <User size={15} /> : <Bot size={15} />}</div>
              <div className="col gap-6" style={{ minWidth: 0 }}>
                <div className={`bubble ${m.role === 'assistant' && selected === i && messages.filter((x) => x.role === 'assistant').length > 1 ? 'selected' : ''}`} onClick={() => m.role === 'assistant' && setSelected(i)}>
                  {m.role === 'assistant' && m.streaming && !m.content ? (
                    <span className="row gap-8 muted small"><span className="typing"><i /><i /><i /></span>{m.sources.length || m.images?.length ? '正在生成回答…' : m.trace ? '精排与上下文组装中…' : '查询变换与多路召回中…'}</span>
                  ) : m.role === 'assistant' ? (
                    <Markdown text={m.content} activeCite={selected === i ? activeCite : null} onImageCite={(n) => openImage(i, n)} onCite={(n) => { setSelected(i); setSideTab('sources'); setActiveCite(n); setTimeout(() => document.getElementById(`ctx-${n}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 50); }} />
                  ) : (
                    <div className="col gap-6">
                      {m.queryImage ? <img className="query-image" src={m.queryImage} alt="查询图片" /> : null}
                      {m.content ? <div className="pre">{m.content}</div> : null}
                    </div>
                  )}
                  {m.error ? <div className="xs mt-8 text-danger">{m.error}</div> : null}
                </div>
                {m.role === 'assistant' && !m.streaming && m.trace?.kbSuggestion ? (() => {
                  const sug = m.trace.kbSuggestion;
                  const question = messages[i - 1]?.role === 'user' ? messages[i - 1].content : '';
                  return (
                    <div className="callout" style={{ alignItems: 'center' }}>
                      <Sparkles size={14} />
                      <div className="row between wrap gap-8" style={{ flex: 1 }}>
                        <span className="small">当前知识库没有检索到资料，这个问题在「<b>{sug.name}</b>」里可能有答案{sug.fileName ? `（命中 ${sug.fileName}，检索分 ${sug.score.toFixed(2)}）` : ''}。</span>
                        <Button size="xs" variant="primary" disabled={sending || !question} onClick={() => { setKbChoice(sug.kbId); send(question, sug.kbId); }}>在「{sug.name}」里重新提问</Button>
                      </div>
                    </div>
                  );
                })() : null}
                {m.role === 'assistant' && !m.streaming && m.trace?.citationCheck?.length ? (
                  <Callout tone="warn" icon={<AlertTriangle size={14} />}>
                    <span className="small">引用校验：{m.trace.citationCheck.map(issueText).join('；')}。这几处可能沿用了上一轮的编号，或在本轮资料里找不到依据，请对照来源核实。</span>
                  </Callout>
                ) : null}
                {m.role === 'assistant' && m.images?.length ? (
                  <div className="img-strip">
                    {m.images.map((im) => <ImageCard key={im.imageId} image={im} compact highlighted={!!m.imageCitations?.includes(im.index ?? -1)} onOpen={() => { setSelected(i); setDetail(im); }} />)}
                  </div>
                ) : null}
                {m.role === 'assistant' && !m.streaming ? (
                  <div className="row gap-6 wrap xs faint">
                    {m.model ? <Badge mono>{m.model}</Badge> : null}
                    {m.trace?.strategy ? <Badge mono>{m.trace.strategy}</Badge> : null}
                    <span>{m.sources.length} 条资料 · 引用 {m.citations.length} 条{m.images?.length ? ` · 图片 ${m.images.length} 张` : ''}</span>
                    {m.trace?.routeFallback ? (
                      <Badge tone="warn" title={`自动路由先选了「${m.trace.routeFallback.fromName}」：${m.trace.routeFallback.reason}；改用「${m.trace.routeFallback.toName}」（精排最高分 ${m.trace.routeFallback.toScore.toFixed(2)}）`}>路由兜底：{m.trace.routeFallback.fromName} → {m.trace.routeFallback.toName}</Badge>
                    ) : m.meta?.route?.length ? <Badge tone="info">路由 → {m.meta.route[0].name}</Badge> : null}
                    {m.trace?.timingsMs?.total ? <span>检索 {m.trace.timingsMs.total}ms</span> : null}
                    {(m.trace as { experience?: unknown[] } | null)?.experience?.length ? <Badge tone="weaved" title="Agent 湖里相似的历史问答，作为参考信息交给模型（不作为引用来源）">历史问答参考 {(m.trace as { experience?: unknown[] }).experience!.length} 条</Badge> : null}
                    <span className="row gap-4" style={{ marginLeft: 'auto' }}>
                      {([1, -1] as const).map((v) => (
                        <button key={v} className={`btn icon xs ${feedback[m.id] === v ? (v > 0 ? 'primary' : 'danger') : 'ghost'}`} title={v > 0 ? '有帮助' : '没帮助（不再作为历史参考，并计入知识缺口）'}
                          onClick={(e) => { e.stopPropagation(); const next = feedback[m.id] === v ? 0 : v; lakeApi.feedback(m.id, next).then(() => setFeedback((f) => ({ ...f, [m.id]: next }))).catch((err) => toast(errMsg(err), 'bad')); }}>
                          {v > 0 ? <ThumbsUp size={11} /> : <ThumbsDown size={11} />}
                        </button>
                      ))}
                    </span>
                  </div>
                ) : null}
              </div>
            </div>
          ))}
          <div ref={endRef} />
        </div>

        {(image || tempDoc || parsing) ? (
          <div className="row gap-8" style={{ padding: '8px 16px', borderTop: '1px solid var(--border)' }}>
            {image ? <div style={{ position: 'relative' }}><img src={image} alt="附件" style={{ width: 52, height: 52, objectFit: 'cover', borderRadius: 8, border: '1px solid var(--border-strong)' }} /><button className="btn icon xs" style={{ position: 'absolute', top: -6, right: -6, padding: 2 }} onClick={() => setImage(null)}><X size={10} /></button></div> : null}
            {tempDoc ? <Badge tone="info"><FileIcon size={11} />{tempDoc.name}（{tempDoc.content.length} 字）<button className="btn ghost icon xs" style={{ padding: 0 }} onClick={() => setTempDoc(null)}><X size={10} /></button></Badge> : null}
            {parsing ? <Badge tone="info">正在解析附件…</Badge> : null}
          </div>
        ) : null}

        <form style={{ padding: 14, borderTop: '1px solid var(--border)' }} onSubmit={(e) => { e.preventDefault(); send(); }}>
          <div className="row gap-8" style={{ alignItems: 'flex-end' }}>
            <input ref={fileRef} type="file" hidden onChange={onFile} accept="image/*,.pdf,.txt,.md,.markdown,.docx,.pptx,.xlsx,.xls,.csv" />
            <Button type="button" icon title="上传图片（以图搜图 / 以图搜文 / 看图问答）或文档（临时分析，不入库）" onClick={() => fileRef.current?.click()}><Paperclip size={15} /></Button>
            <Button type="button" icon title="语音输入" onClick={toggleRecording} style={recording ? { color: 'var(--danger)', borderColor: 'var(--danger)' } : undefined}>{recording ? <Mic size={15} /> : <MicOff size={15} />}</Button>
            <textarea className="flex-1" rows={1} style={{ minHeight: 38, maxHeight: 140, resize: 'none' }} placeholder={recording ? '正在聆听…' : image ? '可以直接发送（以图搜图 / 以图搜文），或补充一句文字' : '输入问题，Enter 发送，Shift+Enter 换行'} value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(); } }} />
            {sending ? <Button type="button" variant="danger" onClick={() => abortRef.current?.abort()}><Square size={13} />停止</Button> : <Button type="submit" variant="primary" disabled={!input.trim() && !image}><Send size={14} />发送</Button>}
          </div>
          {recording ? <div className="row gap-8 mt-8 xs muted"><span className="voice-bars">{[0.1, 0.3, 0.5, 0.2, 0.4].map((d) => <i key={d} style={{ animationDelay: `${d}s` }} />)}</span>语音识别中，再次点击麦克风结束</div> : null}
        </form>
      </div>

      {/* 引用与链路 */}
      <div className="card col chat-side" style={{ minHeight: 0, gap: 0 }}>
        <div className="card-header">
          <Tabs value={sideTab} onChange={setSideTab} items={[{ value: 'sources', label: `来源${sel ? `（${sel.sources.length}）` : ''}`, icon: <FileText size={12} /> }, { value: 'images', label: `图片${sel ? `（${selImages.length}）` : ''}`, icon: <Images size={12} /> }, { value: 'trace', label: '链路', icon: <Network size={12} /> }]} />
        </div>
        <div className="scroll flex-1 col gap-8" style={{ padding: 12 }}>
          {!sel || sel.role !== 'assistant' ? <Empty icon={<FileText size={32} />} title="暂无内容" desc="发送问题后，这里显示本轮回答用到的资料与检索链路；点击回答里的 [n] 可定位到对应来源。" /> : sideTab === 'sources' ? (
            <>
              {sel.sources.length === 0 ? <div className="small faint" style={{ textAlign: 'center', padding: 20 }}>{sel.streaming ? '检索中…' : '未检索到资料（已按协议拒答）'}</div> : null}
              {sel.sources.map((s) => <ContextCard key={s.chunkId} ctx={s} index={s.index} compact highlighted={activeCite === s.index || (activeCite === null && sel.citations.includes(s.index))} onClick={() => setActiveCite(s.index)} />)}
              {sel.relations.length ? <div className="card-title mt-12" style={{ marginBottom: 4 }}><Network size={13} style={{ color: 'var(--c-weaved)' }} />UnWeaver 织入的术语</div> : null}
              {sel.relations.map((r, i) => <div key={i} className="chunk" style={{ borderLeft: '3px solid var(--c-weaved)' }}><div className="strong small" style={{ color: 'var(--c-weaved)' }}>{r.entity}</div><div className="xs muted">{r.definition}</div></div>)}
            </>
          ) : sideTab === 'images' ? (
            selImages.length === 0 ? <div className="small faint" style={{ textAlign: 'center', padding: 20 }}>{sel.streaming ? '检索中…' : '本轮回答没有相关图片。只有与命中段落关联、或相关性达到门槛的图片才会附上；想看图可以直接说"给我看××图"，或上传一张图片搜相似。'}</div> : (
              <div className="col gap-8">{selImages.map((im) => <ImageCard key={im.imageId} image={im} highlighted={!!sel.imageCitations?.includes(im.index ?? -1)} onOpen={() => setDetail(im)} />)}</div>
            )
          ) : sel.trace ? (
            <>
              <div className="chunk"><div className="xs faint mb-8">策略</div><Badge mono>{sel.trace.strategy}</Badge>{sel.meta ? <> <Badge>{sel.meta.language}</Badge></> : null}</div>
              <div className="chunk"><div className="xs faint mb-8">检索查询（{sel.trace.queries.length}）</div>{sel.trace.queries.map((q, i) => <div key={i} className="small">· {q}</div>)}</div>
              {sel.trace.hydeDocument ? <div className="chunk"><div className="xs mb-8" style={{ color: 'var(--c-hyde)' }}>HyDE 假设文档</div><div className="small muted pre">{sel.trace.hydeDocument}</div></div> : null}
              {sel.trace.hops.length > 1 ? <div className="chunk"><div className="xs faint mb-8">多跳轨迹</div>{sel.trace.hops.map((h) => <div key={h.hop} className="small">第 {h.hop} 跳：{h.query} <span className="faint xs">（{h.note}）</span></div>)}</div> : null}
              {'recall' in sel.trace.funnel ? <div className="chunk"><div className="xs faint mb-8">漏斗</div><div className="small mono">{sel.trace.funnel.recall} → {sel.trace.funnel.fused} → {sel.trace.funnel.reranked} → {sel.trace.funnel.contexts}</div><div className="xs faint">召回 → 融合 → 精排 → 上下文</div></div> : null}
              <div className="chunk"><div className="xs faint mb-8">耗时</div><div className="row gap-4 wrap">{Object.entries(sel.trace.timingsMs).map(([k, v]) => <Badge key={k} mono>{k} {v}ms</Badge>)}</div></div>
              {sel.trace.imageSearch ? (
                <div className="chunk">
                  <div className="xs faint mb-8">图片检索</div>
                  <div className="row gap-4 wrap"><Badge mono tone="weaved">{sel.trace.imageSearch.mode}</Badge>{sel.trace.imageSearch.routes.map((r) => <Badge key={r} mono tone={IMAGE_ROUTE[r]?.tone ?? ''}>{IMAGE_ROUTE[r]?.label ?? r}</Badge>)}</div>
                  {sel.trace.imageSearch.queryDescription ? <div className="xs muted mt-8 pre">查询图片识别：{sel.trace.imageSearch.queryDescription}</div> : null}
                  {sel.trace.imageSearch.notes.map((n, i) => <div key={i} className="xs muted mt-4">· {n}</div>)}
                </div>
              ) : null}
              {sel.trace.notes.length ? <div className="chunk"><div className="xs faint mb-8">说明</div>{sel.trace.notes.map((n, i) => <div key={i} className="xs muted">· {n}</div>)}</div> : null}
            </>
          ) : <div className="small faint" style={{ textAlign: 'center', padding: 20 }}>等待检索…</div>}
        </div>
      </div>

      {detail ? <ImageDetail image={detail} onClose={() => setDetail(null)} /> : null}
      {showStrategy ? (
        <Drawer title="检索策略（仅对后续提问生效）" subtitle="与检索实验室使用同一套参数" onClose={() => setShowStrategy(false)} footer={<Button variant="primary" onClick={() => setShowStrategy(false)}>完成</Button>}>
          <StrategyPanel mode={mode} onModeChange={setMode} value={overrides} onChange={setOverrides} />
        </Drawer>
      ) : null}
    </div>
  );
}
