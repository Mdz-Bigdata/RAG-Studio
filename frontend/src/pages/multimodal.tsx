// 图文检索实验室：文搜图 / 图搜图 / 图文混合 / 图搜文 / 文搜视频 / 图搜视频，每一路召回可见；任一图片、镜头、视频可看「相似 / 查重」，另有全库内容查重；右侧是能力状态与知识库图片库。
import { useCallback, useEffect, useRef, useState } from 'react';
import { Clapperboard, Clock, CopyCheck, FileSearch, Film, Image as ImageIcon, Images, Layers, ScanSearch, Search, Type, Upload, X, Zap } from 'lucide-react';
import { api } from '../api/client';
import type { ImageHitView, ImageView, MultimodalMode, MultimodalResponse, MultimodalStatus } from '../api/types';
import { ContextCard } from '../components/context-card';
import { DedupView } from '../components/dedup-view';
import { IMAGE_ROUTE, ImageCard, ImageDetail, titleOf } from '../components/image-card';
import { KbSelect } from '../components/kb-select';
import { SimilarPanel, type SimilarTarget } from '../components/similar-panel';
import { VideoResults } from '../components/video-results';
import { Badge, Button, Callout, Empty, ScoreBar, Spinner, Tabs } from '../components/ui';
import { formatScore } from '../lib/format';
import { useApp } from '../store/app-store';

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));

const MODES: Record<MultimodalMode, { label: string; icon: React.ReactNode; text: boolean; image: boolean; desc: string }> = {
  text2image: { label: '文搜图', icon: <Type size={13} />, text: true, image: false, desc: '文字 → 图片：图文向量（CLIP 文本塔）+ 图片描述的向量与 BM25 等多路 RRF 融合' },
  image2image: { label: '图搜图', icon: <ImageIcon size={13} />, text: false, image: true, desc: '图片 → 相似图片：图像塔编码后按余弦相似度检索，可设阈值过滤' },
  hybrid: { label: '图文混合', icon: <Layers size={13} />, text: true, image: true, desc: '文字 + 图片：score = α·z(text·I) + (1-α)·z(image·I)，在同一向量空间里统一检索（"像这张图，但要……"）' },
  image2text: { label: '图搜文', icon: <FileSearch size={13} />, text: false, image: true, desc: '图片 → 正文段落：图文向量直达 + 相似图片所在段落 + OCR / VLM 转文字再检索' },
  text2video: { label: '文搜视频', icon: <Film size={13} />, text: true, image: false, desc: '文字 → 视频片段：在视频关键帧上走文搜图的各路召回，命中的镜头按时间合并成片段，点击即从片段起点播放' },
  image2video: { label: '图搜视频', icon: <Clapperboard size={13} />, text: false, image: true, desc: '图片 → 视频片段：查询图与视频关键帧比视觉相似度，命中的镜头按时间合并成片段，点击即从片段起点播放' },
};
const isVideoMode = (md: MultimodalMode) => md === 'text2video' || md === 'image2video';
const EXAMPLES: Partial<Record<MultimodalMode, string[]>> = {
  text2image: ['各区域销售额柱状图', '系统架构图', '流程图', '产品外观照片'],
  text2video: ['海边日落', '有人在白板前讲解', '产品特写镜头', '幻灯片上的价格'],
};
const PLACEHOLDER: Partial<Record<MultimodalMode, string>> = {
  image2text: '可选：补充一句文字，和图片一起检索正文',
  hybrid: '用文字补充或修正图片，例如"同样的图表，但要 2024 年的"',
  text2image: '描述你要找的图片，例如"各区域销售额柱状图"',
  text2video: '描述你要找的画面，例如"海边日落""有人在白板前讲解"',
};
const CN_COUNT = ['零', '一', '两', '三', '四', '五', '六', '七', '八', '九'];
const ROUTE_HEAD: Partial<Record<MultimodalMode, string>> = { text2image: '文字 → 图片', image2text: '图片 → 正文段落', text2video: '文字 → 视频关键帧', image2video: '图片 → 视频关键帧' };

/** 说明按实际参与融合的召回路生成（多模态表开关、图文模型、权重不同，路数就不同）；状态还没返回时用默认说明。 */
function modeDesc(mode: MultimodalMode, status: MultimodalStatus | null): string {
  const keys = status?.routes?.[mode];
  const head = ROUTE_HEAD[mode];
  if (!head || !keys?.length) return MODES[mode].desc;
  const fused = keys.length > 1 ? `，${CN_COUNT[keys.length] ?? keys.length}路 RRF 融合` : '';
  return `${head}：${keys.map((k) => IMAGE_ROUTE[k]?.label ?? k).join(' + ')}${fused}${isVideoMode(mode) ? '，命中镜头按时间合并成片段' : ''}`;
}

function readAsDataUrl(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

function ImageInput({ value, onChange }: { value: string | null; onChange: (v: string | null) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const take = async (file?: File | null) => { if (file && file.type.startsWith('image/')) onChange(await readAsDataUrl(file)); };
  return (
    <div className={`img-drop ${over ? 'over' : ''}`} tabIndex={0} onClick={() => ref.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files?.[0]); }}
      onPaste={(e) => take(Array.from(e.clipboardData.files)[0])}>
      <input ref={ref} type="file" hidden accept="image/*" onChange={(e) => { take(e.target.files?.[0]); e.target.value = ''; }} />
      {value ? (
        <>
          <img src={value} alt="查询图片" />
          <button type="button" className="btn icon xs" style={{ position: 'absolute', top: 6, right: 6 }} aria-label="移除图片" onClick={(e) => { e.stopPropagation(); onChange(null); }}><X size={11} /></button>
        </>
      ) : (
        <div className="col gap-4" style={{ alignItems: 'center' }}><Upload size={18} /><span className="small">点击 / 拖拽 / 粘贴查询图片</span><span className="xs faint">也可以在右侧图片库里点一张图「以此图搜相似」</span></div>
      )}
    </div>
  );
}

export function MultimodalPage() {
  const { currentKb, toast } = useApp();
  const [mode, setMode] = useState<MultimodalMode>('text2image');
  const [query, setQuery] = useState('');
  const [image, setImage] = useState<string | null>(null);
  const [topK, setTopK] = useState(6);
  const [alpha, setAlpha] = useState(0.5);
  const [minScore, setMinScore] = useState('');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<MultimodalResponse | null>(null);
  const [tab, setTab] = useState<'results' | 'routes'>('results');
  const [status, setStatus] = useState<MultimodalStatus | null>(null);
  const [statusErr, setStatusErr] = useState<string | null>(null);
  const [gallery, setGallery] = useState<ImageView[]>([]);
  const [detail, setDetail] = useState<ImageView | ImageHitView | null>(null);
  const [view, setView] = useState<'search' | 'dedup'>('search');
  const [similar, setSimilar] = useState<SimilarTarget | null>(null);
  const kbId = currentKb?.id;
  const m = MODES[mode];

  const loadSeq = useRef(0);
  const load = useCallback(async () => {
    const seq = ++loadSeq.current;
    setStatus(null); setStatusErr(null);          // 切换知识库时先回到"加载中"，不残留上一个库的状态
    if (!kbId) { setGallery([]); return; }
    try {
      const [s, g] = await Promise.all([api.multimodal.status(kbId), api.multimodal.images(kbId)]);
      if (seq !== loadSeq.current) return;        // 已经切到别的知识库：丢掉这次迟到的结果
      setStatus(s); setGallery(g);
      setTopK(s.defaults.topK); setAlpha(s.defaults.alpha);
    } catch (e) {
      if (seq === loadSeq.current) { setStatusErr(errMsg(e)); toast(errMsg(e), 'bad'); }
    }
  }, [kbId, toast]);
  useEffect(() => { setData(null); setSimilar(null); load(); }, [load]);

  const run = async (over?: { mode?: MultimodalMode; query?: string; image?: string | null }) => {
    const md = over?.mode ?? mode;
    const q = (over?.query ?? query).trim();
    const img = over?.image === undefined ? image : over.image;
    if (!kbId) return;
    if (MODES[md].text && !q) { toast('请输入文字查询', 'warn'); return; }
    if (MODES[md].image && !img) { toast('请先提供一张查询图片', 'warn'); return; }
    setLoading(true);
    try {
      const res = await api.multimodal.search(kbId, { mode: md, query: MODES[md].text || md === 'image2text' ? q : undefined, imageB64: MODES[md].image ? img : undefined, topK, alpha: md === 'hybrid' ? alpha : undefined, minScore: md === 'image2image' && minScore !== '' ? Number(minScore) : undefined });
      setData(res);
      setTab('results');
    } catch (e) { toast(errMsg(e), 'bad'); } finally { setLoading(false); }
  };

  const searchSimilar = async (im: ImageView) => {
    try {
      const dataUrl = await readAsDataUrl(await (await fetch(im.url)).blob());
      const md = mode === 'image2video' ? 'image2video' : 'image2image';   // 图搜视频模式下就拿这张图去搜视频
      setImage(dataUrl);
      setMode(md);
      run({ mode: md, image: dataUrl });
    } catch (e) { toast(`读取图片失败：${errMsg(e)}`, 'bad'); }
  };

  const similarOf = (im: ImageView) => setSimilar({ imageId: im.imageId, label: titleOf(im) });
  const emb = status?.embedder;
  const hitIds = new Set(data?.images.map((h) => h.imageId) ?? []);

  const header = (
    <div className="row between wrap">
      <div className="row gap-8 wrap">
        <KbSelect />
        <Tabs value={view === 'dedup' ? 'dedup' : mode} onChange={(v) => { if (v === 'dedup') { setView('dedup'); return; } setView('search'); setMode(v); setData(null); }}
          items={[...(Object.keys(MODES) as MultimodalMode[]).map((k) => ({ value: k, label: MODES[k].label, icon: MODES[k].icon })), { value: 'dedup' as const, label: '内容查重', icon: <CopyCheck size={13} /> }]} />
      </div>
      <span className="xs faint">{view === 'dedup' ? '全库扫描重复的图片与视频：图片看向量相似度，视频看片段重合度' : modeDesc(mode, status)}</span>
    </div>
  );

  return (
    <div className="layout-mm">
      <div className="col gap-12" style={{ minHeight: 0 }}>
        {view === 'dedup' ? (
          <DedupView key={kbId} kbId={kbId} header={header} onOpenImage={setDetail} onSimilar={setSimilar} />
        ) : (
          <>
            <div className="card pad col gap-12">
              {header}
              <form className="row gap-12 top" onSubmit={(e) => { e.preventDefault(); run(); }}>
                {m.image ? <div style={{ width: 210, flexShrink: 0 }}><ImageInput value={image} onChange={setImage} /></div> : null}
                <div className="col gap-8 flex-1">
                  {m.text || mode === 'image2text' ? (
                    <div style={{ position: 'relative' }}>
                      <Search size={15} className="faint" style={{ position: 'absolute', left: 11, top: 10 }} />
                      <input style={{ paddingLeft: 34 }} disabled={!kbId} value={query} onChange={(e) => setQuery(e.target.value)}
                        placeholder={PLACEHOLDER[mode]} />
                    </div>
                  ) : null}
                  <div className="row gap-12 wrap small muted">
                    <label className="row gap-6">Top-K<input type="number" min={1} max={30} value={topK} onChange={(e) => setTopK(Math.max(1, Number(e.target.value) || 1))} style={{ width: 64, padding: '4px 8px', fontSize: 12 }} /></label>
                    {mode === 'hybrid' ? <label className="row gap-6" title="文本权重 α：两个模态的相似度各自标准化后按 α 加权">图片<input type="range" min={0} max={1} step={0.05} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} style={{ width: 130 }} />文本 <span className="mono">α={alpha.toFixed(2)}</span></label> : null}
                    {mode === 'image2image' ? <label className="row gap-6" title="余弦相似度低于该值的图片被过滤；留空则总返回 Top-K">相似度阈值<input type="number" min={0} max={1} step={0.05} placeholder="不限" value={minScore} onChange={(e) => setMinScore(e.target.value)} style={{ width: 72, padding: '4px 8px', fontSize: 12 }} /></label> : null}
                    <div className="flex-1" />
                    <Button variant="primary" type="submit" loading={loading} disabled={!kbId}><Zap size={14} />检索</Button>
                  </div>
                  {EXAMPLES[mode] ? <div className="row gap-6 wrap"><span className="xs faint">示例：</span>{EXAMPLES[mode]!.map((q) => <button type="button" key={q} className="btn xs chip" disabled={!kbId} onClick={() => { setQuery(q); run({ query: q }); }}>{q}</button>)}</div> : null}
                </div>
              </form>
            </div>

            <div className="scroll flex-1 col gap-12" style={{ paddingRight: 4 }}>
              {!kbId ? <div className="card"><Empty icon={<Images size={40} />} title="请先选择知识库" desc="在「知识库」页面上传含图片的 PDF / Word / PPT / Markdown，或直接上传图片 / 视频文件。" /></div> : !data && isVideoMode(mode) ? (
                <div className="card"><Empty icon={mode === 'image2video' ? <Clapperboard size={40} /> : <Film size={40} />} title={status && status.videos === 0 ? '这个知识库里还没有视频' : MODES[mode].label} desc={status && !status.video?.backend
                  ? '服务器上没有可用的视频解码器：安装 ffmpeg（推荐，支持字幕与 mkv / avi 预览转码）或 opencv-python 后即可上传视频。'
                  : status && status.videos === 0
                    ? '在「知识库」页面上传 mp4 / mov / webm / mkv 等视频。入库时按镜头切分，每个镜头取一张最清晰的关键帧，生成图文向量、OCR 与语义描述，并带上内嵌字幕。'
                    : mode === 'image2video'
                      ? '提供一张图片（截图、海报、某一帧画面），找到视频里画面相似的镜头：结果按视频分组，时间轴上高亮命中片段，点击即从片段起点播放。'
                      : '输入一句画面描述，找到视频里对应的镜头：结果按视频分组，时间轴上高亮命中片段，点击即从片段起点播放。'} /></div>
              ) : !data ? (
                <div className="card"><Empty icon={<ScanSearch size={40} />} title={status && status.images === 0 ? '这个知识库里还没有图片' : '图文检索实验室'} desc={status && status.images === 0
                  ? '上传含图片的 PDF / Word / PPT / Markdown，或直接上传 png / jpg 图片。入库时会抽取图片、记录它在文档里的位置，并生成 OCR 文字与语义描述。'
                  : '四种检索方式共用同一份图片索引：图文向量（图片与文字在同一空间）、图片描述的文本向量、图片描述的 BM25。选择方式后输入文字或提供图片。'} /></div>
              ) : (
                <>
                  {data.notes.length || data.queryDescription ? (
                    <Callout tone={data.images.length === 0 && data.contexts.length === 0 ? 'warn' : ''}>
                      {data.queryDescription ? <div><b>查询图片识别：</b>{data.queryDescription}</div> : null}
                      {data.notes.map((n, i) => <div key={i}>· {n}</div>)}
                    </Callout>
                  ) : null}
                  <div className="row between wrap">
                    <Tabs value={tab} onChange={setTab} items={[
                      { value: 'results', label: data.mode === 'image2text' ? `正文段落（${data.contexts.length}）· 相似图片（${data.images.length}）` : isVideoMode(data.mode) ? `视频片段（${data.segments.length}）` : `图片结果（${data.images.length}）` },
                      { value: 'routes', label: `各路召回（${Object.keys(data.routes).length} 路）` },
                    ]} />
                    <div className="row gap-6 wrap"><Clock size={12} className="faint" />{Object.entries(data.timingsMs).map(([k, v]) => <Badge key={k} mono tone={k === 'total' ? 'info' : ''}>{k} {v}ms</Badge>)}</div>
                  </div>

                  {tab === 'results' && isVideoMode(data.mode) ? (
                    data.segments.length ? <VideoResults segments={data.segments} onOpenFrame={setDetail} onSimilarFrame={similarOf}
                      onSimilarVideo={(docId) => setSimilar({ docId, label: data.segments.find((s) => s.docId === docId)?.fileName ?? docId })} /> : <Callout tone="warn">没有找到相关的视频片段。</Callout>
                  ) : tab === 'results' ? (
                    <>
                      {data.mode === 'image2text' ? (
                        data.contexts.length === 0 ? <Callout tone="warn">没有找到与这张图片相关的正文段落。</Callout> : <div className="col gap-8">{data.contexts.map((c, i) => <ContextCard key={c.chunkId} ctx={c} index={i + 1} />)}</div>
                      ) : null}
                      {data.images.length ? (
                        <>
                          {data.mode === 'image2text' ? <div className="section-label mt-8"><Images size={12} />知识库里的相似图片（它们所在的段落参与了上面的排序）</div> : null}
                          <div className="img-grid">{data.images.map((h) => <ImageCard key={h.imageId} image={h} onOpen={() => setDetail(h)} onSimilar={() => similarOf(h)} />)}</div>
                        </>
                      ) : data.mode !== 'image2text' ? <Callout tone="warn">没有符合条件的图片{data.mode === 'image2image' && minScore !== '' ? '（可以调低相似度阈值）' : ''}。</Callout> : null}
                    </>
                  ) : (
                    <div className="grid" style={{ gridTemplateColumns: `repeat(${Math.min(3, Math.max(1, Object.keys(data.routes).length))}, minmax(0, 1fr))` }}>
                      {Object.entries(data.routes).map(([route, items]) => {
                        const max = Math.max(...items.map((i) => i.score), 1e-9);
                        return (
                          <div key={route} className="card pad">
                            <div className="row between mb-8"><Badge tone={IMAGE_ROUTE[route]?.tone ?? ''} title={data.routeLabels[route]}>{IMAGE_ROUTE[route]?.label ?? route}</Badge><span className="xs faint">{items.length} 条</span></div>
                            <div className="xs faint mb-8">{data.routeLabels[route]}</div>
                            <div className="col gap-6">
                              {items.length === 0 ? <div className="faint small" style={{ padding: 12, textAlign: 'center' }}>无结果</div> : items.map((it) => (
                                <div key={it.id} className={`chunk ${hitIds.has(it.id) ? 'hl' : ''}`} style={{ padding: '8px 10px' }}>
                                  <div className="row gap-8">
                                    {it.url ? <img src={it.url} alt="" loading="lazy" style={{ width: 44, height: 34, objectFit: 'contain', background: '#fff', borderRadius: 4, border: '1px solid var(--border)', flexShrink: 0 }} /> : null}
                                    <div className="flex-1">
                                      <div className="chunk-head" style={{ marginBottom: 3 }}><span className="row gap-6" style={{ minWidth: 0 }}><span className="rank">#{it.rank}</span><span className="ellipsis" title={it.label}>{it.label}</span></span><span className="mono">{formatScore(it.score, it.score > 1 ? 2 : 3)}</span></div>
                                      <ScoreBar value={it.score} max={max} />
                                    </div>
                                  </div>
                                  <div className="chunk-text clamp-2 small" style={{ marginTop: 5 }}>{it.text}</div>
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </>
              )}
            </div>
          </>
        )}
      </div>

      <div className="scroll col gap-12" style={{ minHeight: 0, paddingRight: 4 }}>
        <div className="card pad col gap-8">
          <div className="card-title" style={{ marginBottom: 2 }}><ScanSearch size={14} />多模态能力</div>
          {!status ? (
            // 状态还没返回 / 读取失败：别把 null 当成"不可用"去显示兜底文案（会误报"未安装 OCR"之类）
            statusErr ? (
              <div className="chunk col gap-6"><div className="xs muted">模型状态读取失败：{statusErr}</div><div><Button size="xs" onClick={load}>重试</Button></div></div>
            ) : kbId ? <div className="chunk"><Spinner label="正在读取模型状态…" /></div> : <div className="xs faint">选择知识库后显示。</div>
          ) : (<>
          <div className="chunk">
            <div className="row between"><span className="xs faint">图文向量模型</span><span className="row gap-4">{emb?.videoNative ? <Badge tone="fused" title="视频片段整段编码进同一向量空间（WeMM）">全模态</Badge> : null}<Badge tone={emb?.crossModal ? 'ok' : 'warn'}>{emb?.crossModal ? '跨模态' : '仅图搜图'}</Badge></span></div>
            <div className="small strong mono ellipsis" title={emb?.model ?? ''}>{emb?.model ?? '—'}</div>
            <div className="xs muted">{emb?.crossModal ? `${emb.provider} · 文字、图片${emb.videoNative ? '与视频' : ''}在同一向量空间${emb.loaded ? '' : '（首次使用时加载）'}` : '离线感知特征：没有文本塔。文搜图只走图片描述两路；安装 torch + transformers 后自动启用 Chinese-CLIP。'}</div>
          </div>
          <div className="chunk">
            <div className="row between"><span className="xs faint">图片描述（VLM）</span><Badge tone={status?.vision ? 'ok' : 'warn'}>{status?.vision ? '可用' : '未配置'}</Badge></div>
            <div className="xs muted">{status?.vision ? `已为 ${status.described} / ${status.images} 张图生成语义描述` : '在「模型配置」激活多模态模型后，入库时为每张图生成语义描述，问答时模型可直接看图。'}</div>
          </div>
          <div className="chunk">
            <div className="row between"><span className="xs faint">视频检索</span><Badge tone={status.video?.backend && status.video.enabled ? 'ok' : 'warn'}>{!status.video?.enabled ? '已关闭' : status.video.backend ?? '不可用'}</Badge></div>
            <div className="xs muted">{!status.video?.enabled ? '策略配置 → 多模态 → video.enabled 打开后可上传视频。' : status.video.backend ? `${status.videos} 个视频 · ${status.videoFrames} 个镜头关键帧` : '安装 ffmpeg（推荐）或 opencv-python 后可上传视频：按镜头抽关键帧，文搜视频定位到片段。'}</div>
            {status.video?.enabled && emb?.videoNative ? <div className="xs muted">WeMM 视频向量：{status.video.vectors ?? status.indexEmbedder.video_vectors ?? 0} 个片段 / 其中 {status.video.jointVectors ?? status.indexEmbedder.video_joint_vectors ?? 0} 个带字幕</div> : null}
          </div>
          <div className="chunk">
            <div className="row between"><span className="xs faint">图中文字（OCR）</span><Badge tone={status?.ocr ? 'ok' : 'warn'}>{status?.ocr ?? '未安装'}</Badge></div>
            <div className="xs muted">{status?.ocr ? `${status.withOcr} / ${status.images} 张图识别出文字` : 'pip install rapidocr-onnxruntime 后重新解析文档：图表里的数值、标签即可被精确检索。'}</div>
          </div>
          </>)}
        </div>

        <div className="card pad col gap-8">
          <div className="row between"><div className="card-title" style={{ marginBottom: 0 }}><Images size={14} />图片库</div><span className="xs faint">{status ? `${status.images} 张 · ${status.documentsWithImages} 个文档` : ''}</span></div>
          {gallery.length === 0 ? <div className="xs faint" style={{ padding: '10px 0' }}>暂无图片。</div> : (
            <div className="img-grid dense">{gallery.slice(0, 60).map((g) => <ImageCard key={g.imageId} image={g} compact highlighted={hitIds.has(g.imageId)} onOpen={() => setDetail(g)} onSimilar={() => similarOf(g)} />)}</div>
          )}
          {gallery.length > 60 ? <div className="xs faint">仅显示前 60 张。</div> : null}
        </div>
      </div>

      {detail ? <ImageDetail image={detail} onClose={() => setDetail(null)} onSearchSimilar={(im) => { setView('search'); searchSimilar(im); }} onSimilar={similarOf} /> : null}
      {similar && kbId ? <SimilarPanel kbId={kbId} target={similar} onTarget={setSimilar} onClose={() => setSimilar(null)} /> : null}
    </div>
  );
}
