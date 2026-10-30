// 内容查重：全库扫描重复图片（图文向量余弦 ≥ 阈值）与重复视频（片段高度重合），按组列出；阈值留空时按图文向量模型自动取。
import { useState } from 'react';
import { CopyCheck, Film, Images, Play } from 'lucide-react';
import { api } from '../api/client';
import type { DuplicatesResponse, ImageView } from '../api/types';
import { ImageCard, titleOf } from './image-card';
import type { SimilarTarget } from './similar-panel';
import { Badge, Button, Callout, Empty } from './ui';
import { formatScore, pct } from '../lib/format';
import { useApp } from '../store/app-store';

type Kind = 'image' | 'video';
const KIND_LABEL: Record<Kind, string> = { image: '图片', video: '视频' };

/** header：页面顶部的知识库 / 模式切换行，和查重设置放在同一张卡片里。 */
export function DedupView({ kbId, header, onOpenImage, onSimilar }: { kbId: string | undefined; header: React.ReactNode; onOpenImage: (im: ImageView) => void; onSimilar: (t: SimilarTarget) => void }) {
  const { toast } = useApp();
  const [threshold, setThreshold] = useState('');
  const [kinds, setKinds] = useState<Record<Kind, boolean>>({ image: true, video: true });
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<DuplicatesResponse | null>(null);

  const run = async () => {
    const ks = (Object.keys(kinds) as Kind[]).filter((k) => kinds[k]);
    if (!kbId) return;
    if (!ks.length) { toast('至少选择一种内容', 'warn'); return; }
    setLoading(true);
    try {
      const res = await api.multimodal.duplicates(kbId, { threshold: threshold === '' ? null : Number(threshold), kinds: ks });
      setData(res);
      setThreshold(String(res.threshold));             // 回填实际用的阈值，方便在它基础上微调
    } catch (e) { toast(e instanceof Error ? e.message : String(e), 'bad'); } finally { setLoading(false); }
  };

  const count = (k: Kind) => data?.groups.filter((g) => g.kind === k).length ?? 0;
  return (
    <>
      <div className="card pad col gap-12">
        {header}
        <form className="row gap-12 wrap small muted" onSubmit={(e) => { e.preventDefault(); run(); }}>
          <label className="row gap-6" title="余弦相似度 ≥ 该值判为重复；留空按图文向量模型自动选取">判重阈值<input type="number" min={0} max={1} step={0.01} placeholder="自动" value={threshold} onChange={(e) => setThreshold(e.target.value)} style={{ width: 76, padding: '4px 8px', fontSize: 12 }} /></label>
          {(Object.keys(KIND_LABEL) as Kind[]).map((k) => (
            <label key={k} className="row gap-6"><input type="checkbox" checked={kinds[k]} onChange={(e) => setKinds({ ...kinds, [k]: e.target.checked })} />{KIND_LABEL[k]}</label>
          ))}
          <div className="flex-1" />
          <Button variant="primary" type="submit" loading={loading} disabled={!kbId}><CopyCheck size={14} />开始查重</Button>
        </form>
      </div>

      <div className="scroll flex-1 col gap-12" style={{ paddingRight: 4 }}>
        {!kbId ? <div className="card"><Empty icon={<Images size={40} />} title="请先选择知识库" /></div> : !data ? (
          <div className="card"><Empty icon={<CopyCheck size={40} />} title="内容查重" desc="扫描知识库里的全部图片与视频：图片按图文向量的余弦相似度判重；视频按片段重合度判重——大部分镜头都能在另一个视频里找到高度相似的镜头，就认为两个视频重复。" /></div>
        ) : (
          <>
            <Callout tone={data.groups.length ? 'warn' : ''}>
              扫描了 <b>{data.images}</b> 张图片 / <b>{data.videos}</b> 个视频，找到 <b>{data.groups.length}</b> 组重复{data.groups.length ? `（图片 ${count('image')} 组 · 视频 ${count('video')} 组）` : ''}。
              <span className="xs faint"> 判重阈值 {formatScore(data.threshold, 2)} · 视频片段重合度 ≥ {pct(data.overlap)}</span>
            </Callout>
            {data.groups.length === 0 ? <Callout>没有发现重复内容{threshold !== '' ? '（可以调低判重阈值再试）' : ''}。</Callout> : data.groups.map((g, i) => (
              <div key={i} className="card pad col gap-8">
                <div className="row between">
                  <span className="row gap-8">{g.kind === 'image' ? <Images size={14} /> : <Film size={14} />}<span className="small strong">{g.kind === 'image' ? `${g.items.length} 张重复图片` : `${g.items.length} 个重复视频`}</span></span>
                  <Badge mono tone="warn" title={g.kind === 'image' ? '组内最高的图片相似度' : '组内最高的片段重合度'}>{g.kind === 'image' ? `相似度 ${formatScore(g.score)}` : `片段重合度 ${pct(g.score)}`}</Badge>
                </div>
                {g.kind === 'image' ? (
                  <div className="img-strip">
                    {g.items.map((im) => <ImageCard key={im.imageId} image={im} compact onOpen={() => onOpenImage(im)} onSimilar={() => onSimilar({ imageId: im.imageId, label: titleOf(im) })} />)}
                  </div>
                ) : (
                  <div className="img-strip">
                    {g.items.map((v) => (
                      <div key={v.docId} className="video-seg col" style={{ width: 200, gap: 6 }} role="button" tabIndex={0} title="查看与它相似的视频（逐片段对照）"
                        onClick={() => onSimilar({ docId: v.docId, label: v.fileName ?? v.docId })} onKeyDown={(e) => { if (e.key === 'Enter') onSimilar({ docId: v.docId, label: v.fileName ?? v.docId }); }}>
                        <div className="video-seg-thumb" style={{ width: '100%' }}>
                          {v.cover ? <img src={v.cover} alt="" loading="lazy" /> : null}
                          <span className="video-seg-play"><Play size={14} /></span>
                        </div>
                        <div className="xs ellipsis row gap-4" title={v.fileName ?? v.docId}><Film size={10} style={{ flexShrink: 0 }} /><span className="ellipsis">{v.fileName ?? v.docId}</span></div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </>
        )}
      </div>
    </>
  );
}
