"""视频 → 关键帧：按镜头切分视频，每个片段取一张最清晰的帧进图片索引，帧带着片段的起止时间。

    视频 ── 按 sample_fps 抽帧（只用于分析：小图的颜色直方图 + 灰度缩略图）
        ├─ 镜头切换：和上一帧的差异 > scene_threshold（硬切；差异 = 颜色分布变化与"内容像素里变了多少"取大者）
        ├─ 画面漂移：和本片段第一帧的差异 > drift_threshold（推拉摇移、幻灯片翻页）
        └─ 片段过长：> max_segment_seconds 强制切开，之后画面没变的相邻片段再合并回去
    → 太短的片段（闪白 / 转场）并入前一段，连续的短片段（快速运动）合成一段 → 超过 max_frames 时反复合并最相似的相邻片段
    → 每段取中间部分最清晰（拉普拉斯方差最大）的一帧作为关键帧，另外均匀留 clip_frames 张小图（采样帧）

关键帧此后和文档里抽出的图片完全一样：CLIP 图文向量 + OCR + VLM 描述 + BM25，文搜图的各路召回直接可用。
图文向量模型能直接编码视频时（WeMM），采样帧把整个片段编码成"纯视频"与"视频 + 字幕"两种向量，文搜视频多两路召回。
文搜视频再把命中的关键帧按视频、按时间合并成连续片段（group_segments）。

解码优先用 ffmpeg 命令行（格式最全，还能读内嵌字幕），没有就用 opencv。都没有则视频不可入库。
浏览器放不了的封装格式（mkv / avi …）有 ffmpeg 时转一份 H.264 的预览 mp4 放在原文件旁边。
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import MultimodalConfig, VideoConfig
from .schemas import ImageHit, VideoSegment

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = frozenset({"mp4", "m4v", "mov", "webm", "mkv", "avi", "flv", "wmv", "mpg", "mpeg"})
BROWSER_PLAYABLE = frozenset({"mp4", "m4v", "webm", "mov"})
_PREVIEW_SUFFIX = ".preview.mp4"


# --------------------------------------------------------------------------------------
def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _ffprobe() -> str | None:
    return shutil.which("ffprobe")


def _cv2() -> Any | None:
    try:
        import cv2
    except Exception:
        return None
    return cv2


def video_backend() -> str | None:
    """可用的解码方式：ffmpeg | opencv | None。"""
    if _ffmpeg():
        return "ffmpeg"
    if _cv2() is not None:
        return "opencv"
    return None


def video_available() -> bool:
    return video_backend() is not None


def is_video(path: Path | str) -> bool:
    return Path(path).suffix.lower().lstrip(".") in VIDEO_EXTENSIONS


@dataclass
class VideoInfo:
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    backend: str = ""


def probe(path: Path) -> VideoInfo:
    """时长 / 分辨率 / 帧率。读不出来的字段为 0。"""
    if _ffprobe():
        try:
            out = subprocess.run([_ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,duration:format=duration",
                                  "-of", "json", str(path)], capture_output=True, text=True, timeout=60, check=True).stdout
            data = json.loads(out or "{}")
            st = (data.get("streams") or [{}])[0]
            num, _, den = str(st.get("avg_frame_rate") or "0/1").partition("/")
            fps = float(num) / float(den or 1) if float(den or 1) else 0.0
            dur = float(st.get("duration") or (data.get("format") or {}).get("duration") or 0)
            return VideoInfo(duration=dur, width=int(st.get("width") or 0), height=int(st.get("height") or 0), fps=fps, backend="ffmpeg")
        except Exception as exc:
            logger.warning("ffprobe 读取视频信息失败 %s：%s", path.name, exc)
    cv2 = _cv2()
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            return VideoInfo(duration=n / fps if fps else 0.0, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=fps, backend="opencv")
        finally:
            cap.release()
    return VideoInfo()


def _scaled(w: int, h: int, max_side: int) -> tuple[int, int]:
    if not w or not h:
        return 0, 0
    s = min(1.0, max_side / max(w, h))
    # 偶数尺寸：部分编码 / 像素格式要求
    return max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2)


def iter_frames(path: Path, *, fps: float, max_side: int, max_seconds: float, info: VideoInfo | None = None) -> Iterator[tuple[float, Any]]:
    """按 fps 均匀抽帧，产出 (时刻秒, PIL.Image RGB)，最长边不超过 max_side。"""
    from PIL import Image

    info = info or probe(path)
    if _ffmpeg() and info.width and info.height:
        w, h = _scaled(info.width, info.height, max_side)
        cmd = [_ffmpeg(), "-v", "error", "-nostdin", "-t", f"{max_seconds:.3f}", "-i", str(path), "-an", "-sn",
               "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        size = w * h * 3
        n = 0
        try:
            assert proc.stdout is not None
            while True:
                buf = proc.stdout.read(size)
                if len(buf) < size:
                    break
                yield n / fps, Image.frombytes("RGB", (w, h), buf)
                n += 1
        finally:
            proc.stdout.close() if proc.stdout else None
            proc.kill()
            proc.wait()
        if n:
            return
        logger.warning("ffmpeg 没有解出任何帧（%s），改用 opencv", path.name)
    cv2 = _cv2()
    if cv2 is None:
        raise RuntimeError("没有可用的视频解码器：请安装 ffmpeg（推荐）或 opencv-python")
    cap = cv2.VideoCapture(str(path))
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = 1.0 / fps
        next_t, i = 0.0, 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            t = i / src_fps
            i += 1
            if t > max_seconds:
                break
            if t + 1e-6 < next_t:
                continue
            next_t += step
            ok, frame = cap.retrieve()
            if not ok:
                continue
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side))
            yield t, im
    finally:
        cap.release()


# --------------------------------------------------------------------------------------
@dataclass
class _Feat:
    hist: np.ndarray       # HSV 直方图（归一化）
    thumb: np.ndarray      # 320×180 灰度缩略图（0~1）
    ink: np.ndarray        # "有内容"的像素：与整帧灰度中位数相差明显（白底上的字、画面主体）
    sharp: float           # 拉普拉斯方差：越大越清晰
    detail: float          # 灰度标准差：几乎为 0 的是纯色帧（黑场 / 白场）


_DIFF = 0.15               # 灰度差超过它算"不同"（0~1）
_BLANK = 0.02              # 灰度标准差低于它算纯色帧


def frame_features(im: Any) -> _Feat:
    small = im.resize((320, 180))
    hsv = np.asarray(small.resize((64, 36)).convert("HSV"), dtype=np.int32)
    hist = np.concatenate([np.bincount(hsv[..., 0].ravel() // 8, minlength=32), np.bincount(hsv[..., 1].ravel() // 16, minlength=16),
                           np.bincount(hsv[..., 2].ravel() // 16, minlength=16)]).astype(np.float32)
    hist /= hist.sum() or 1.0
    thumb = np.asarray(small.convert("L"), dtype=np.float32) / 255.0
    g = np.asarray(im.resize((256, max(2, int(256 * im.size[1] / max(1, im.size[0]))))).convert("L"), dtype=np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return _Feat(hist=hist, thumb=thumb, ink=np.abs(thumb - np.median(thumb)) > _DIFF, sharp=float(lap.var()) if lap.size else 0.0, detail=float(g.std() / 255.0))


def frame_distance(a: _Feat, b: _Feat) -> float:
    """两帧的差异（0~1），取两种信号的大者：
    - 颜色分布的变化（HSV 直方图总变差）：换场景、换光照；
    - 内容像素里变了的比例：分母只算两帧里"有内容"的像素（两帧 ink 的并集），而不是整幅画面。
      白底幻灯片翻页时变的只是文字那一小块，按整幅画面算不到 5%，按文字像素算是大半；
      自然画面里几乎处处都是内容，这一项就退化成普通的"变了的面积占比"。"""
    hist = 0.5 * float(np.abs(a.hist - b.hist).sum())      # H / S / V 三段直方图合起来归一化，结果在 0~1
    content = a.ink | b.ink
    n = int(content.sum())
    if n < 50:                 # 两帧都几乎是纯色
        return hist
    changed = float(((np.abs(a.thumb - b.thumb) > _DIFF) & content).sum()) / n
    return max(hist, changed)


@dataclass
class _Segment:
    start: float
    end: float
    shot: int = 0          # 镜头序号：只在硬切处递增，长镜头因画面漂移 / 超长切出的片段属于同一个镜头
    frames: list[tuple[float, Any, _Feat]] = field(default_factory=list)
    key: tuple[float, Any, _Feat] | None = None
    clip: list[tuple[float, bytes]] = field(default_factory=list)   # 采样帧（时刻, JPEG）

    def pick_key(self, clip_frames: int = 0, clip_side: int = 512) -> None:
        """片段中间 60% 里最清晰的一帧（避开转场的拖影和淡入淡出）；另外均匀留 clip_frames 张采样帧（JPEG 小图）。"""
        from .wemm import jpeg_bytes

        n = len(self.frames)
        lo, hi = int(n * 0.2), max(int(n * 0.2) + 1, int(np.ceil(n * 0.8)))
        self.key = max(self.frames[lo:hi], key=lambda f: f[2].sharp)
        if clip_frames > 0:
            idx = sorted({int(i) for i in np.linspace(0, n - 1, min(n, clip_frames)).round()})
            self.clip = [(self.frames[i][0], jpeg_bytes(self.frames[i][1], clip_side)) for i in idx]
        self.frames = []          # 只留关键帧，释放内存


def segment_video(frames: Iterator[tuple[float, Any]], cfg: VideoConfig, *, step: float) -> list[_Segment]:
    segs: list[_Segment] = []
    cur: _Segment | None = None
    prev: _Feat | None = None
    shot = 0
    for t, im in frames:
        f = frame_features(im)
        if cur is not None and prev is not None:
            cut = frame_distance(prev, f) > cfg.scene_threshold
            if cut or frame_distance(cur.frames[0][2], f) > cfg.drift_threshold or t - cur.start >= cfg.max_segment_seconds:
                cur.pick_key(cfg.clip_frames, cfg.clip_frame_side)
                segs.append(cur)
                cur = None
                shot += cut
        if cur is None:
            cur = _Segment(start=t, end=t + step, shot=shot)
        cur.frames.append((t, im, f))
        cur.end = t + step
        prev = f
    if cur is not None and cur.frames:
        cur.pick_key(cfg.clip_frames, cfg.clip_frame_side)
        segs.append(cur)
    out = _merge(segs, cfg)
    for s in out:              # 合并过的片段采样帧变多了，均匀抽回 clip_frames 张
        if len(s.clip) > cfg.clip_frames:
            s.clip = [s.clip[int(i)] for i in np.linspace(0, len(s.clip) - 1, cfg.clip_frames).round()]
    return out


def _merge(segs: list[_Segment], cfg: VideoConfig) -> list[_Segment]:
    if not segs:
        return segs
    # 1) 太短的片段：连续一串短片段（快速运动 / 摇镜头时帧帧都在"切"）合起来够长就自成一段，取其中最清晰的一帧；
    #    不够长的（闪白 / 转场）并入前一段。逐个并入前一段的话，一整段运动镜头会被吞进上一个镜头里
    out: list[_Segment] = []
    run: list[_Segment] = []

    def flush() -> None:
        if not run:
            return
        if run[-1].end - run[0].start >= cfg.min_segment_seconds or not out:
            out.append(_Segment(start=run[0].start, end=run[-1].end, shot=run[0].shot, key=max((r.key for r in run), key=lambda k: k[2].sharp),  # type: ignore[arg-type, return-value, union-attr]
                                clip=[c for r in run for c in r.clip]))
        else:
            out[-1].end = run[-1].end
            out[-1].clip += [c for r in run for c in r.clip]
        run.clear()

    for s in segs:
        if s.end - s.start < cfg.min_segment_seconds:
            run.append(s)
            continue
        flush()
        out.append(s)
    flush()
    if len(out) > 1 and out[0].end - out[0].start < cfg.min_segment_seconds:
        out[1].start = out[0].start
        out[1].clip = out[0].clip + out[1].clip
        out.pop(0)
    # 纯色片段（黑场 / 白场 / 淡入淡出）没有可检索的内容：不出关键帧，也不占 max_frames 的名额
    out = [s for s in out if s.key[2].detail >= _BLANK]  # type: ignore[index]

    def joinable(a: _Segment, b: _Segment) -> float:
        return frame_distance(a.key[2], b.key[2])  # type: ignore[index]

    def join(i: int) -> None:
        a, b = out[i], out[i + 1]
        a.end = b.end
        a.clip += b.clip
        if b.key[2].sharp > a.key[2].sharp and b.end - b.start > a.end - a.start:  # type: ignore[index]
            a.key = b.key
        out.pop(i + 1)

    # 2) 强制切开的长镜头：画面没变的相邻片段合并回去
    i = 0
    while i < len(out) - 1:
        if joinable(out[i], out[i + 1]) < cfg.drift_threshold / 2:
            join(i)
        else:
            i += 1
    # 3) 超过上限：反复合并最相似的相邻片段
    if len(out) > cfg.max_frames:
        d = [joinable(out[k], out[k + 1]) for k in range(len(out) - 1)]
        while len(out) > cfg.max_frames:
            k = int(np.argmin(d))
            join(k)
            d.pop(k)
            if k < len(d):
                d[k] = joinable(out[k], out[k + 1])
            if k > 0:
                d[k - 1] = joinable(out[k - 1], out[k])
    return out


# --------------------------------------------------------------------------------------
def read_subtitles(path: Path, max_seconds: float) -> list[tuple[float, float, str]]:
    """内嵌字幕轨（mkv / mp4 的 srt / mov_text / ass）→ [(开始, 结束, 文字)]。没有字幕或没有 ffmpeg 返回空。"""
    if not _ffmpeg():
        return []
    try:
        out = subprocess.run([_ffmpeg(), "-v", "error", "-nostdin", "-i", str(path), "-map", "0:s:0", "-t", f"{max_seconds:.3f}", "-f", "srt", "pipe:1"],
                             capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return []
    return parse_srt(out)


def parse_srt(text: str) -> list[tuple[float, float, str]]:
    import re

    def sec(ts: str) -> float:
        h, m, rest = ts.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)

    cues = []
    for block in re.split(r"\n\s*\n", (text or "").replace("\r", "")):
        lines = [x for x in block.strip().splitlines() if x.strip()]
        for k, line in enumerate(lines):
            m = re.match(r"(\d+:\d{2}:\d{2}[,.]\d+)\s*-->\s*(\d+:\d{2}:\d{2}[,.]\d+)", line)
            if m:
                body = " ".join(re.sub(r"<[^>]+>|\{[^}]*\}", "", x).strip() for x in lines[k + 1:])
                if body:
                    cues.append((sec(m.group(1)), sec(m.group(2)), body))
                break
    return cues


def ensure_browser_playable(path: Path) -> Path | None:
    """浏览器能直接播放的格式返回原文件；否则（mkv / avi …）有 ffmpeg 时转一份 H.264 预览 mp4。失败返回 None。"""
    if path.suffix.lower().lstrip(".") in BROWSER_PLAYABLE:
        return path
    preview = preview_path(path)
    if preview.is_file():
        return preview
    if not _ffmpeg():
        return None
    tmp = preview.with_name(preview.name + ".part")
    try:
        subprocess.run([_ffmpeg(), "-v", "error", "-nostdin", "-y", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0?", "-vf", "scale='min(1280,iw)':-2",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
                        "-movflags", "+faststart", "-f", "mp4", str(tmp)], capture_output=True, timeout=3600, check=True)
        tmp.replace(preview)
        return preview
    except Exception as exc:
        logger.warning("视频预览转码失败 %s：%s", path.name, exc)
        tmp.unlink(missing_ok=True)
        return None


def preview_path(path: Path) -> Path:
    return path.with_name(path.stem + _PREVIEW_SUFFIX)


def playable_path(path: Path) -> Path | None:
    """已有的可播放文件（不触发转码）。"""
    if path.suffix.lower().lstrip(".") in BROWSER_PLAYABLE:
        return path if path.is_file() else None
    p = preview_path(path)
    return p if p.is_file() else None


# --------------------------------------------------------------------------------------
def extract_video_frames(path: Path, cfg: MultimodalConfig) -> list[Any]:
    """视频 → ExtractedImage 列表（每个片段一张关键帧，带 time / time_start / time_end，字幕放进 context）。"""
    from .extract import ExtractedImage, pixel_sha1

    vc = cfg.video
    if not vc.enabled:
        return []
    info = probe(path)
    max_seconds = vc.max_duration_minutes * 60
    step = 1.0 / vc.sample_fps
    segs = segment_video(iter_frames(path, fps=vc.sample_fps, max_side=cfg.max_side, max_seconds=max_seconds, info=info), vc, step=step)
    duration = min(info.duration or (segs[-1].end if segs else 0.0), max_seconds)
    if segs and duration:
        segs[-1].end = max(segs[-1].start, min(segs[-1].end, duration))
    subs = read_subtitles(path, max_seconds)
    out = []
    for s in segs:
        t, im, _f = s.key  # type: ignore[misc]
        said = " ".join(text for a, b, text in subs if a < s.end and b > s.start)
        out.append(ExtractedImage(image=im, sha1=pixel_sha1(im), origin="video", time=round(t, 2), time_start=round(s.start, 2), time_end=round(s.end, 2), shot=s.shot,
                                  context=(f"台词 / 字幕：{said}" if said else "")[: cfg.context_chars], meta={"duration": round(duration, 2)},
                                  clip=[jpg for _, jpg in s.clip], clip_times=[round(ct - s.start, 2) for ct, _ in s.clip], subtitle=said))
    try:
        ensure_browser_playable(path)
    except Exception as exc:   # 预览只影响播放，不影响入库
        logger.warning("视频预览准备失败 %s：%s", path.name, exc)
    logger.info("%s：时长 %.1fs，切出 %d 个片段（%s 解码，字幕 %d 条）", path.name, duration, len(out), info.backend or video_backend(), len(subs))
    return out


# --------------------------------------------------------------------------------------
def group_segments(hits: list[ImageHit], *, gap: float, top_k: int, min_ratio: float = 0.35, join_ratio: float = 0.8) -> list[VideoSegment]:
    """命中的关键帧 → 片段。默认一帧一段（它所在镜头的那一截）；只有同一个镜头里时间相接（间隔 ≤ gap 秒）、
    且得分不低于段内最高分 join_ratio 的命中才连成一段——跨硬切的相邻镜头即使都"有点相关"也不合并：
    融合后的相对相关度挤在一个窄区间里（实测"饼图"一问，同一视频四个镜头 0.76~0.94），只按分数和时间连，
    短视频会整段连成一个"片段"，等于没有定位。
    得分低于全局最高分 min_ratio 的帧不出片段。片段分 = 段内最高帧分。"""
    hits = [h for h in hits if h.record.is_video]
    if not hits:
        return []
    best = max(h.score for h in hits)
    keep = [h for h in hits if h.score >= best * min_ratio]
    by_doc: dict[str, list[ImageHit]] = {}
    for h in keep:
        by_doc.setdefault(h.record.doc_id, []).append(h)
    segs: list[VideoSegment] = []
    for doc_id, items in by_doc.items():
        items.sort(key=lambda h: _start(h))
        group: list[ImageHit] = []
        for h in items:
            if group and not (h.record.shot is not None and h.record.shot == group[-1].record.shot
                              and _start(h) - max(_end(x) for x in group) <= gap and h.score >= join_ratio * max(x.score for x in group)):
                segs.append(_segment(doc_id, group))
                group = []
            group.append(h)
        if group:
            segs.append(_segment(doc_id, group))
    segs.sort(key=lambda s: s.score, reverse=True)
    for n, s in enumerate(segs[:top_k]):
        s.rank = n + 1
    return segs[:top_k]


def _start(h: ImageHit) -> float:
    r = h.record
    return r.time_start if r.time_start is not None else (r.time or 0.0)


def _end(h: ImageHit) -> float:
    r = h.record
    return r.time_end if r.time_end is not None else (r.time or 0.0)


def _segment(doc_id: str, group: list[ImageHit]) -> VideoSegment:
    top = max(group, key=lambda h: h.score)
    return VideoSegment(doc_id=doc_id, start=min(_start(h) for h in group), end=max(_end(h) for h in group), time=top.record.time or _start(top), score=top.score, hits=group)
