// 文搜视频结果：按视频分组，每个视频一个播放器 + 时间轴（命中片段高亮）；点片段从起点播放、到终点自动暂停。
import { useMemo, useRef, useState } from 'react';
import { Film, Maximize2, Play } from 'lucide-react';
import type { ImageHitView, VideoSegmentView } from '../api/types';
import { Badge, Callout, ScoreBar } from './ui';
import { IMAGE_ROUTE } from './image-card';
import { formatScore, formatTs } from '../lib/format';

export interface Span { start: number; end: number; time: number }

/** 片段播放：加载后停在第一个片段的代表帧；play(i) 从片段起点播到终点自动暂停（用户拖动进度条则不再自动暂停）。 */
export function useSegmentPlayback(spans: Span[], durationHint?: number | null) {
  const ref = useRef<HTMLVideoElement>(null);
  const stopAt = useRef<number | null>(null);
  const [duration, setDuration] = useState(durationHint || 0);
  const [now, setNow] = useState(0);
  const [active, setActive] = useState(0);
  const [failed, setFailed] = useState(false);

  const play = (i: number) => {
    const v = ref.current, s = spans[i];
    setActive(i);
    if (!v || !s) return;
    stopAt.current = s.end;
    v.currentTime = s.start;
    v.play().catch(() => {});
  };
  const seek = (t: number) => {
    stopAt.current = null;
    if (ref.current) ref.current.currentTime = t;
  };
  const videoProps = {
    ref, controls: true, playsInline: true, preload: 'metadata' as const,
    onLoadedMetadata: (e: React.SyntheticEvent<HTMLVideoElement>) => {
      const v = e.currentTarget;
      if (Number.isFinite(v.duration)) setDuration(v.duration);
      if (spans[active]) v.currentTime = spans[active].time;
    },
    onTimeUpdate: (e: React.SyntheticEvent<HTMLVideoElement>) => {
      const v = e.currentTarget;
      setNow(v.currentTime);
      if (stopAt.current !== null && v.currentTime >= stopAt.current) { v.pause(); stopAt.current = null; }
    },
    onSeeked: (e: React.SyntheticEvent<HTMLVideoElement>) => {
      // 用户自己拖到片段外：不再替他暂停
      const s = spans[active];
      if (stopAt.current !== null && s && (e.currentTarget.currentTime < s.start - 0.5 || e.currentTarget.currentTime > s.end)) stopAt.current = null;
    },
    onError: () => setFailed(true),
  };
  return { videoProps, duration, now, active, play, seek, failed };
}

export function Timeline({ spans, duration, now, active, onPick, onSeek }: { spans: Span[]; duration: number; now: number; active: number; onPick: (i: number) => void; onSeek: (t: number) => void }) {
  const d = duration || Math.max(1, ...spans.map((s) => s.end));
  const pct = (t: number) => `${Math.min(100, Math.max(0, (t / d) * 100))}%`;
  return (
    <div className="col gap-4">
      <div className="vtl" title="点击空白处跳转，点击高亮片段播放该片段"
        onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); onSeek(((e.clientX - r.left) / r.width) * d); }}>
        {spans.map((s, i) => (
          <button key={i} type="button" className={`vtl-seg ${i === active ? 'on' : ''}`} style={{ left: pct(s.start), width: `max(4px, calc(${pct(s.end)} - ${pct(s.start)} - 2px))` }}
            title={`${formatTs(s.start)}–${formatTs(s.end)}`} aria-label={`播放片段 ${formatTs(s.start)}–${formatTs(s.end)}`}
            onClick={(e) => { e.stopPropagation(); onPick(i); }} />
        ))}
        <div className="vtl-head" style={{ left: pct(now) }} />
      </div>
      <div className="row between xs faint mono"><span>{formatTs(now)}</span><span>{formatTs(d)}</span></div>
    </div>
  );
}

const frameOf = (s: VideoSegmentView) => s.frames.find((f) => f.time === s.time) ?? s.frames[0];
const briefOf = (f?: ImageHitView) => (f ? f.description || f.ocrText || f.contextText || f.caption : '') || '（这一帧没有描述文字：靠图文向量命中）';

export function VideoResults({ segments, onOpenFrame }: { segments: VideoSegmentView[]; onOpenFrame: (f: ImageHitView) => void }) {
  // 按视频分组；组的顺序 = 组内最好片段的名次（segments 已按分数排好）
  const groups = useMemo(() => {
    const m = new Map<string, VideoSegmentView[]>();
    for (const s of segments) m.set(s.docId, [...(m.get(s.docId) ?? []), s]);
    return [...m.values()];
  }, [segments]);
  return <div className="col gap-12">{groups.map((g) => <VideoGroup key={g[0].docId} segments={g} onOpenFrame={onOpenFrame} />)}</div>;
}

function VideoGroup({ segments, onOpenFrame }: { segments: VideoSegmentView[]; onOpenFrame: (f: ImageHitView) => void }) {
  const top = segments[0];
  const p = useSegmentPlayback(segments, top.duration);
  return (
    <div className="card pad col gap-12">
      <div className="row between wrap gap-8">
        <div className="card-title" style={{ marginBottom: 0, minWidth: 0 }}><Film size={14} /><span className="ellipsis">{top.fileName ?? top.docId}</span></div>
        <span className="xs faint">{segments.length} 个相关片段{top.duration ? ` · 时长 ${formatTs(top.duration)}` : ''}</span>
      </div>
      <div className="video-layout">
        <div className="col gap-8" style={{ minWidth: 0 }}>
          {p.failed
            ? <Callout tone="warn">浏览器无法播放这个视频（格式不受支持，且入库时没能转出预览——服务器需要 ffmpeg）。右侧关键帧仍可查看。</Callout>
            : <video className="video-view" src={top.videoUrl} poster={frameOf(top)?.url} {...p.videoProps} />}
          <Timeline spans={segments} duration={p.duration} now={p.now} active={p.active} onPick={p.play} onSeek={p.seek} />
        </div>
        <div className="col gap-8 video-segs">
          {segments.map((s, i) => {
            const f = frameOf(s);
            return (
              <div key={`${s.start}-${s.end}`} className={`video-seg ${i === p.active ? 'on' : ''}`} role="button" tabIndex={0} onClick={() => p.play(i)} onKeyDown={(e) => { if (e.key === 'Enter') p.play(i); }}>
                <div className="video-seg-thumb">
                  {f ? <img src={f.url} alt="" loading="lazy" /> : null}
                  <span className="video-seg-play"><Play size={14} /></span>
                  <span className="img-ts">{formatTs(s.start)}–{formatTs(s.end)}</span>
                </div>
                <div className="col gap-4 flex-1" style={{ minWidth: 0 }}>
                  <div className="row between"><span className="rank">#{s.rank}</span><span className="mono xs">{formatScore(s.score)}</span></div>
                  <ScoreBar value={s.score} />
                  <div className="xs muted clamp-2" title={briefOf(f)}>{briefOf(f)}</div>
                  <div className="row gap-4 wrap">
                    {s.frames.length > 1 ? <Badge>{s.frames.length} 个镜头</Badge> : null}
                    {f ? Object.entries(f.routes).slice(0, 3).map(([k, v]) => <Badge key={k} mono tone={IMAGE_ROUTE[k]?.tone ?? ''} title={`${IMAGE_ROUTE[k]?.label ?? k}：第 ${v.rank} 名`}>{IMAGE_ROUTE[k]?.label ?? k} #{v.rank}</Badge>) : null}
                    {f ? <button type="button" className="btn xs" style={{ marginLeft: 'auto' }} onClick={(e) => { e.stopPropagation(); onOpenFrame(f); }}><Maximize2 size={10} />关键帧</button> : null}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
