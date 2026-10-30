// 相似 / 查重抽屉：某张图（或视频关键帧）的相似图片，某个视频的相似视频（整体相似度 + 片段重合度 + 逐片段对照，点对照即播放对方视频的对应片段）。
import { useEffect, useMemo, useState } from 'react';
import { ArrowLeftRight, CopyCheck, Film, Images, Play } from 'lucide-react';
import { api } from '../api/client';
import type { ImageHitView, ImageView, SimilarImageView, SimilarResponse, SimilarVideoView } from '../api/types';
import { ImageCard, ImageDetail, spanOf, titleOf } from './image-card';
import { Timeline, useSegmentPlayback } from './video-results';
import { Badge, Callout, Drawer, ScoreBar, Spinner } from './ui';
import { formatScore, pct } from '../lib/format';

/** imageId：图片 / 视频关键帧；docId：视频。label 只用于抽屉标题。 */
export interface SimilarTarget { imageId?: string; docId?: string; label: string }

// 相似结果没有各路召回信息：补成命中视图，卡片 / 详情按"有分数"的样式显示
const asHit = (im: SimilarImageView): ImageHitView => ({ ...im, linked: false, routes: {} });
const DupBadge = () => <Badge tone="warn"><CopyCheck size={9} />重复</Badge>;

export function SimilarPanel({ kbId, target, onTarget, onClose }: { kbId: string; target: SimilarTarget; onTarget: (t: SimilarTarget) => void; onClose: () => void }) {
  const [data, setData] = useState<SimilarResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [detail, setDetail] = useState<ImageView | null>(null);

  useEffect(() => {
    let live = true;                                  // 连续切换目标时丢掉迟到的结果
    setData(null); setErr(null);
    api.multimodal.similar(kbId, { imageId: target.imageId, docId: target.docId, topK: 12 })
      .then((r) => { if (live) setData(r); }, (e) => { if (live) setErr(e instanceof Error ? e.message : String(e)); });
    return () => { live = false; };
  }, [kbId, target.imageId, target.docId]);

  const retarget = (im: ImageView) => onTarget({ imageId: im.imageId, label: titleOf(im) });
  const model = typeof data?.embedder?.model === 'string' ? data.embedder.model : null;
  const dupImages = data?.images.filter((im) => im.duplicate).length ?? 0;
  const dupVideos = data?.videos.filter((v) => v.duplicate).length ?? 0;

  return (
    <>
      {/* 详情弹窗开着时 Esc 只关弹窗，不连带关掉抽屉 */}
      <Drawer title={<span className="row gap-8"><CopyCheck size={15} />相似 / 查重</span>} onClose={() => { if (!detail) onClose(); }}
        subtitle={<span className="ellipsis" style={{ display: 'block', maxWidth: 640 }}>{target.label}{data ? ` · 判重阈值 ${formatScore(data.threshold, 2)}${model ? ` · ${model}` : ''}` : ''}</span>}>
        {err ? <Callout tone="bad">{err}</Callout> : !data ? <Spinner label="正在计算相似度…" /> : (
          <div className="col gap-16">
            {target.imageId ? (
              <div className="col gap-8">
                <div className="section-label" style={{ marginBottom: 0 }}><Images size={12} />相似图片 / 镜头（{data.images.length}）{dupImages ? ` · ${dupImages} 张达到判重阈值` : ''}</div>
                {data.images.length === 0 ? <Callout tone="warn">没有找到相似的图片。</Callout> : (
                  <div className="img-grid">
                    {data.images.map((im) => (
                      <ImageCard key={im.imageId} image={asHit(im)} onOpen={() => setDetail(asHit(im))} onSimilar={() => retarget(im)}
                        extra={im.duplicate ? <div><DupBadge /></div> : null} />
                    ))}
                  </div>
                )}
              </div>
            ) : null}
            {data.docId ? (
              <div className="col gap-8">
                <div className="section-label" style={{ marginBottom: 0 }}><Film size={12} />{target.imageId ? '与所在视频相似的视频' : '相似视频'}（{data.videos.length}）{dupVideos ? ` · ${dupVideos} 个判为重复` : ''}</div>
                {data.videos.length === 0 ? <Callout tone="warn">没有找到相似的视频。</Callout> : data.videos.map((v) => <SimilarVideo key={v.docId} video={v} onOpenFrame={setDetail} />)}
              </div>
            ) : null}
          </div>
        )}
      </Drawer>
      {detail ? <ImageDetail image={detail} onClose={() => setDetail(null)} onSimilar={retarget} /> : null}
    </>
  );
}

/** 一个相似视频：左边播放对方视频（时间轴高亮与源片段匹配上的片段），右边逐片段对照；点源帧看源片段，点对照行播放对方片段。 */
function SimilarVideo({ video, onOpenFrame }: { video: SimilarVideoView; onOpenFrame: (im: ImageView) => void }) {
  const spans = useMemo(() => video.pairs.map(({ match: m }) => ({ start: m.timeStart ?? m.time ?? 0, end: m.timeEnd ?? m.time ?? 0, time: m.time ?? 0 })), [video.pairs]);
  const p = useSegmentPlayback(spans);
  return (
    <div className="card pad col gap-12">
      <div className="row between wrap gap-8">
        <div className="card-title" style={{ marginBottom: 0, minWidth: 0 }}><Film size={14} /><span className="ellipsis">{video.fileName ?? video.docId}</span>{video.duplicate ? <DupBadge /> : null}</div>
        <div className="row gap-12 xs muted">
          <span className="row gap-6" title="两个视频整体的向量相似度">整体相似度<span style={{ width: 64 }}><ScoreBar value={video.score} /></span><span className="mono">{formatScore(video.score)}</span></span>
          <span title="一方视频里能在另一方找到高度相似片段的比例（取两个方向中较高的）；达到阈值即判为重复">片段重合度 <span className="mono">{pct(video.overlap)}</span></span>
        </div>
      </div>
      <div className="video-layout">
        <div className="col gap-8" style={{ minWidth: 0 }}>
          {p.failed
            ? <Callout tone="warn">浏览器无法播放这个视频（格式不受支持，且入库时没能转出预览——服务器需要 ffmpeg）。右侧片段对照仍可查看。</Callout>
            : <video className="video-view" src={video.videoUrl} poster={video.cover ?? undefined} {...p.videoProps} />}
          {spans.length ? <Timeline spans={spans} duration={p.duration} now={p.now} active={p.active} onPick={p.play} onSeek={p.seek} /> : null}
        </div>
        <div className="col gap-8 video-segs">
          {video.pairs.length === 0 ? <div className="xs faint">没有逐片段的对照。</div> : video.pairs.map((pr, i) => (
            <div key={`${pr.source.imageId}-${pr.match.imageId}`} className={`video-seg col ${i === p.active ? 'on' : ''}`} role="button" tabIndex={0} title="播放对方视频的这个片段"
              onClick={() => p.play(i)} onKeyDown={(e) => { if (e.key === 'Enter') p.play(i); }}>
              <div className="row gap-6">
                <div className="video-seg-thumb flex-1" style={{ width: 'auto' }} title="源片段：点击查看" onClick={(e) => { e.stopPropagation(); onOpenFrame(pr.source); }}>
                  <img src={pr.source.url} alt="" loading="lazy" />
                  <span className="img-ts">{spanOf(pr.source)}</span>
                </div>
                <ArrowLeftRight size={12} className="faint" style={{ flexShrink: 0 }} />
                <div className="video-seg-thumb flex-1" style={{ width: 'auto' }}>
                  <img src={pr.match.url} alt="" loading="lazy" />
                  <span className="video-seg-play"><Play size={14} /></span>
                  <span className="img-ts">{spanOf(pr.match)}</span>
                </div>
              </div>
              <div className="row gap-6"><span className="xs faint">源片段 ↔ 相似片段</span><div className="flex-1"><ScoreBar value={pr.score} /></div><span className="mono xs">{formatScore(pr.score)}</span></div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
