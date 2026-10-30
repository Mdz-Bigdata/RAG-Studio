"""视频检索：镜头切分 → 关键帧进图片索引 → 文搜视频定位到片段 → 视频文件可按 Range 播放。

视频用 PIL 画帧、ffmpeg 现场编码（带一条内嵌字幕轨）；没有 ffmpeg 的环境整个模块跳过。
向量用离线感知特征，文字信号来自字幕——不依赖 CLIP / OCR / VLM。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from prism.config import MultimodalConfig, Settings, VideoConfig
from prism.ingest import IngestPipeline
from prism.ingest.parsers import ParserRegistry
from prism.multimodal import MultimodalRetriever
from prism.multimodal import video as video_mod
from prism.multimodal.schemas import ImageHit, ImageRecord, format_ts
from prism.multimodal.video import extract_video_frames, group_segments, parse_srt

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="需要 ffmpeg")

W, H, FPS = 640, 360, 10
SRT = """1
00:00:00,500 --> 00:00:03,500
这是各区域的销售额柱状图

2
00:00:08,500 --> 00:00:13,000
傍晚的海边日落
"""


def _bar(i: int) -> Image.Image:
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    for k, h in enumerate([120, 220, 170, 90]):
        d.rectangle([80 + k * 120, 320 - h, 160 + k * 120, 320], fill=(40, 90, 200))
    return im


def _pie(i: int) -> Image.Image:
    im = Image.new("RGB", (W, H), (250, 250, 240))
    d = ImageDraw.Draw(im)
    d.pieslice([200, 40, 480, 320], 0, 130, fill=(220, 50, 50))
    d.pieslice([200, 40, 480, 320], 130, 360, fill=(250, 190, 40))
    return im


def _sunset(i: int) -> Image.Image:
    """同一个镜头里太阳慢慢移动：不应被切成好几段。"""
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    for y in range(H):
        d.line([0, y, W, y], fill=(255, int(120 + y * 0.3), int(40 + y * 0.2)))
    x = 100 + i * 6
    d.ellipse([x, 80, x + 120, 200], fill=(255, 240, 120))
    d.rectangle([0, 280, W, H], fill=(20, 40, 90))
    return im


def _slide(text: str):
    font = ImageFont.load_default(size=36)

    def draw(i: int) -> Image.Image:
        im = Image.new("RGB", (W, H), "white")
        ImageDraw.Draw(im).multiline_text((40, 60), text, fill="black", font=font, spacing=16)
        return im
    return draw


def _black(i: int) -> Image.Image:
    return Image.new("RGB", (W, H), "black")


def _encode(out: Path, scenes: list[tuple], srt: str | None = None) -> Path:
    frames = out.parent / f"{out.stem}_frames"
    frames.mkdir(parents=True, exist_ok=True)
    n = 0
    for draw, seconds in scenes:
        for i in range(int(seconds * FPS)):
            draw(i).save(frames / f"{n:05d}.png")
            n += 1
    cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(frames / "%05d.png")]
    if srt:
        (frames / "subs.srt").write_text(srt, encoding="utf-8")
        cmd += ["-i", str(frames / "subs.srt"), "-c:s", "mov_text"]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


@pytest.fixture(scope="module")
def videos(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("videos")
    demo = _encode(root / "demo.mp4", [(_bar, 4), (_pie, 4), (_sunset, 6), (_slide("Quarterly revenue +15%"), 4)], SRT)
    slides = _encode(root / "slides.mp4", [(_slide("Chapter 1\nProduct overview"), 5), (_slide("Chapter 2\nSystem architecture"), 5), (_black, 2),
                                           (_slide("Chapter 3\nPricing 99 per month"), 5), (_slide("Chapter 4\nContact sales"), 5)])
    mkv = root / "demo.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(demo), "-c", "copy", "-c:s", "srt", str(mkv)], check=True)
    return {"demo": demo, "slides": slides, "mkv": mkv}


def spans(frames) -> list[tuple[float, float]]:
    return [(f.time_start, f.time_end) for f in frames]


# --------------------------------------------------------------------------------------
def test_shots_and_subtitles(videos):
    frames = extract_video_frames(videos["demo"], MultimodalConfig())
    assert spans(frames) == [(0.0, 4.0), (4.0, 8.0), (8.0, 14.0), (14.0, 18.0)]   # 慢慢移动的太阳仍是一个镜头
    assert [f.shot for f in frames] == [0, 1, 2, 3]
    long_shot = extract_video_frames(videos["demo"], MultimodalConfig(video=VideoConfig(max_segment_seconds=2, drift_threshold=0.2)))
    sun = [f for f in long_shot if f.time_start >= 8 and f.time_end <= 14]
    assert len(sun) > 1 and len({f.shot for f in sun}) == 1                        # 长镜头切出的几段属于同一个镜头
    assert all(f.origin == "video" and f.time_start <= f.time < f.time_end for f in frames)
    assert "销售额柱状图" in frames[0].context and "日落" in frames[2].context and not frames[1].context   # 字幕按时间归到镜头


def test_slide_flips_and_black_frames(videos):
    """白底幻灯片只是文字换了也要切开；黑场不出关键帧。"""
    frames = extract_video_frames(videos["slides"], MultimodalConfig())
    assert spans(frames) == [(0.0, 5.0), (5.0, 10.0), (12.0, 17.0), (17.0, 22.0)]


def test_max_frames_merges_most_similar(videos):
    cfg = MultimodalConfig(video=VideoConfig(max_frames=3))
    frames = extract_video_frames(videos["slides"], cfg)
    assert len(frames) == 3 and frames[0].time_start == 0.0 and frames[-1].time_end == 22.0


def test_mkv_gets_browser_preview(videos):
    extract_video_frames(videos["mkv"], MultimodalConfig())
    preview = video_mod.preview_path(videos["mkv"])
    assert preview.is_file() and video_mod.playable_path(videos["mkv"]) == preview
    assert video_mod.playable_path(videos["demo"]) == videos["demo"]   # mp4 直接播放，不转码


@pytest.mark.skipif(video_mod._cv2() is None, reason="需要 opencv")
def test_opencv_fallback(videos, monkeypatch):
    monkeypatch.setattr(video_mod, "_ffmpeg", lambda: None)
    monkeypatch.setattr(video_mod, "_ffprobe", lambda: None)
    assert video_mod.video_backend() == "opencv"
    frames = extract_video_frames(videos["demo"], MultimodalConfig())
    assert spans(frames) == [(0.0, 4.0), (4.0, 8.0), (8.0, 14.0), (14.0, 18.0)]


def test_parse_srt():
    cues = parse_srt(SRT.replace("傍晚的海边日落", "<i>傍晚</i>的海边{\\an8}日落"))
    assert cues == [(0.5, 3.5, "这是各区域的销售额柱状图"), (8.5, 13.0, "傍晚的海边日落")]
    assert format_ts(65.4) == "01:05" and format_ts(3725) == "1:02:05"


def test_group_segments():
    def hit(doc: str, start: float, end: float, score: float, shot: int) -> ImageHit:
        rec = ImageRecord(image_id=f"{doc}_{start}", doc_id=doc, file="x.png", sha1="x", width=1, height=1, origin="video", time=start + 0.5, time_start=start, time_end=end, shot=shot)
        return ImageHit(record=rec, score=score)

    hits = [hit("a", 10, 14, 0.9, shot=2), hit("a", 14, 20, 0.8, shot=2),   # 同一镜头、分数相近 → 连成一段
            hit("a", 20, 26, 0.85, shot=3),                                   # 紧挨着但跨了硬切 → 单独一段
            hit("a", 26, 30, 0.5, shot=3),                                    # 同镜头但分数差得多 → 单独一段
            hit("b", 0, 5, 0.8, shot=0), hit("a", 60, 65, 0.1, shot=9)]       # 0.1 分的远端命中不成段
    segs = group_segments(hits, gap=2.0, top_k=10)
    assert [(s.doc_id, s.start, s.end, s.rank) for s in segs] == [("a", 10, 20, 1), ("a", 20, 26, 2), ("b", 0, 5, 3), ("a", 26, 30, 4)]
    assert segs[0].time == 10.5 and segs[0].score == 0.9 and len(segs[0].hits) == 2


# --------------------------------------------------------------------------------------
def test_ingest_and_text_to_video(videos, tmp_path):
    settings = Settings.lite(index_dir=tmp_path / "index")
    assert "mp4" in ParserRegistry(settings).supported_extensions()
    pipeline = IngestPipeline(settings)
    report = pipeline.ingest_paths([videos["demo"]])
    assert not report.failed, report.failed
    assert report.parsers == {"video": 1}
    index = pipeline.index
    recs = sorted(index.images.records.values(), key=lambda r: r.order)
    assert [(r.time_start, r.time_end, r.shot) for r in recs] == [(0.0, 4.0, 0), (4.0, 8.0, 1), (8.0, 14.0, 2), (14.0, 18.0, 3)]
    assert recs[2].label().endswith("00:08–00:14") and "[视频画面" in recs[2].search_text()
    # 正文：每个镜头一段带时间的文字，图片挂在自己那一段上
    parent = index.store.get(recs[2].anchor_chunk_id)
    assert parent is not None and "视频画面 00:08–00:14" in parent.text and "日落" in parent.text

    r = MultimodalRetriever(settings, index)
    res = r.text_to_video("海边日落")
    assert res.mode == "text2video" and res.segments
    top = res.segments[0]
    assert (top.start, top.end) == (8.0, 14.0) and top.start <= top.time < top.end and top.rank == 1
    assert res.images and res.images[0].record.time_start == 8.0
    # 限定文档：没有视频可搜时给出说明而不是报错
    empty = r.text_to_video("海边日落", doc_ids=["not-a-doc"])
    assert not empty.segments and any("没有可检索的视频" in n for n in empty.notes)


def test_video_api(videos, tmp_path_factory):
    from fastapi.testclient import TestClient

    data_dir = tmp_path_factory.mktemp("video_app")
    from app import config as app_config_mod

    cfg = app_config_mod.app_config
    saved = (cfg.data_dir, cfg.upload_dir, cfg.index_root)
    cfg.data_dir, cfg.upload_dir, cfg.index_root = data_dir, data_dir / "uploads", data_dir / "indexes"
    for d in (cfg.upload_dir, cfg.index_root):
        d.mkdir(parents=True, exist_ok=True)
    from app.main import app

    try:
        with TestClient(app) as c:
            r = c.put("/api/v1/settings/", json={"overrides": {"multimodal": {"embedding": {"provider": "perceptual"}, "ocr": "none", "vlm_caption": False}}})
            assert r.status_code == 200, r.text
            assert "mkv" in c.get("/api/v1/documents/supported-formats").json()["extensions"]
            kb_id = c.post("/api/v1/knowledge-bases/", json={"name": "视频库"}).json()["id"]
            with videos["demo"].open("rb") as f:
                r = c.post(f"/api/v1/documents/upload?kbId={kb_id}", files={"file": ("海边.mp4", f, "video/mp4")})
            assert r.status_code == 201, r.text
            doc = c.get(f"/api/v1/documents/{r.json()['id']}").json()
            assert doc["status"] == "COMPLETED", doc

            st = c.get(f"/api/v1/multimodal/{kb_id}/status").json()
            assert st["videos"] == 1 and st["videoFrames"] == 4 and st["video"]["backend"] == "ffmpeg"
            res = c.post(f"/api/v1/multimodal/{kb_id}/search", json={"mode": "text2video", "query": "海边日落"}).json()
            seg = res["segments"][0]
            assert (seg["start"], seg["end"], seg["fileName"], seg["duration"]) == (8.0, 14.0, "海边.mp4", 18.0)
            assert seg["frames"][0]["timeStart"] == 8.0 and seg["frames"][0]["videoUrl"] == seg["videoUrl"]

            full = c.get(seg["videoUrl"])
            assert full.status_code == 200 and full.headers["content-type"] == "video/mp4"
            part = c.get(seg["videoUrl"], headers={"Range": "bytes=0-99"})
            assert part.status_code == 206 and len(part.content) == 100   # 播放器拖动 / 跳转依赖 Range
            assert c.get(f"/api/v1/multimodal/{kb_id}/videos/not-a-doc/file").status_code == 404
            c.post("/api/v1/settings/reset")    # 测试库是整个会话共用的
    finally:
        cfg.data_dir, cfg.upload_dir, cfg.index_root = saved
