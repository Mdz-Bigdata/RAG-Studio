"""WeMM 接入：视频片段整段编码（纯视频 / 视频 + 字幕双向量）、文搜视频 / 图搜视频的视频向量路、换模型重算、
内容查重与相似推荐、多模态记忆、vLLM 接口的请求格式。

大部分测试用一个"能直接编码视频"的假模型（图片 = 感知特征，文字 = 字词哈希，视频 = 帧特征均值，
视频 + 字幕 = 视频向量加字幕的文字向量），不依赖权重；本机缓存了 WeMM-Embedding-2B 时，
test_real_wemm_* 用真实模型验证单次前向双向量与跨模态检索（WEMM_TEST=1 时才跑，CPU / MPS 上要几分钟）。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from prism.config import Settings
from prism.index import Index
from prism.index.embeddings import HashEmbedder
from prism.ingest import IngestPipeline
from prism.multimodal import MultimodalRetriever
from prism.multimodal.dedup import ContentDeduper
from prism.multimodal.embedders import BaseImageEmbedder, PerceptualImageEmbedder, _normalize
from prism.multimodal.wemm import VideoClip, WeMMAPIEmbedder, sample_clip

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="需要 ffmpeg")


class FakeOmni(BaseImageEmbedder):
    """假的全模态向量模型：图片 / 帧 → 感知特征（前 448 维），文字 → 字词哈希（后 448 维），视频 = 帧特征均值，
    视频 + 字幕 = 视频向量 + 字幕文字向量。文字与画面在不同的子空间：文字查询只能经"视频 + 字幕"向量（或描述）找到画面。"""

    name = "fake-omni"
    cross_modal = True
    video_native = True
    dedup_threshold = 0.95

    def __init__(self, model: str = "fake-omni-v1") -> None:
        self.model_name = model
        self._img = PerceptualImageEmbedder()
        self._txt = HashEmbedder(dim=448)
        self.dim = 896
        self.video_calls = 0

    def embed_images(self, images):
        v = self._img.embed_images(images)
        return _normalize(np.hstack([v, np.zeros_like(v)]))

    def embed_texts(self, texts):
        t = self._txt.embed_documents(texts)
        return _normalize(np.hstack([np.zeros_like(t), t]))

    def embed_videos(self, clips, *, with_text=True):
        self.video_calls += 1
        video = _normalize(np.vstack([self.embed_images(c.frames).mean(axis=0) for c in clips]))
        joint = _normalize(np.vstack([video[k] + (self.embed_texts([c.text])[0] if c.text.strip() else 0) for k, c in enumerate(clips)]))
        return video, (joint if with_text else None)


def _settings(tmp_path: Path) -> Settings:
    s = Settings.lite(index_dir=tmp_path / "index")
    s.multimodal.ocr = "none"
    s.multimodal.vlm_caption = False
    return s


@pytest.fixture(scope="module")
def videos(tmp_path_factory) -> dict[str, Path]:
    if shutil.which("ffmpeg") is None:
        pytest.skip("需要 ffmpeg")
    from test_video import SRT, _bar, _encode, _pie, _slide, _sunset

    root = tmp_path_factory.mktemp("wemm_videos")
    demo = _encode(root / "demo.mp4", [(_bar, 4), (_pie, 4), (_sunset, 6), (_slide("Quarterly revenue +15%"), 4)], SRT)
    copy = root / "demo_reupload.mp4"        # 同一段视频重新压缩、缩小分辨率后再上传
    import subprocess

    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(demo), "-map", "0", "-vf", "scale=480:-2", "-c:v", "libx264", "-crf", "33", "-pix_fmt", "yuv420p", "-c:s", "mov_text", str(copy)], check=True)
    slides = _encode(root / "slides.mp4", [(_slide("Chapter 1\nProduct overview"), 5), (_slide("Chapter 2\nSystem architecture"), 5)])
    return {"demo": demo, "copy": copy, "slides": slides, "sunset": _sunset(5)}


@pytest.fixture()
def kb(videos, tmp_path):
    settings = _settings(tmp_path)
    emb = FakeOmni()
    pipeline = IngestPipeline(settings, index=Index(settings, image_embedder=emb))
    report = pipeline.ingest_paths([videos["demo"], videos["copy"], videos["slides"]])
    assert not report.failed, report.failed
    return settings, pipeline.index, emb


# --------------------------------------------------------------------------------------
@needs_ffmpeg
def test_clip_frames_and_video_vectors(kb):
    settings, index, emb = kb
    demo = [r for r in index.images.records.values() if r.doc_title.startswith("demo") and "reupload" not in r.doc_title]
    assert len(demo) == 4 and all(r.clip for r in demo)
    first = min(demo, key=lambda r: r.time_start)
    assert 4 <= len(first.clip) <= settings.multimodal.video.clip_frames and all((index.directory / f).is_file() for f in first.clip)
    assert first.clip_times[0] == 0.0 and first.clip_times[-1] < first.time_end - first.time_start
    assert "销售额" in first.subtitle
    # 每个片段一条纯视频向量、一条"视频 + 字幕"向量；没有字幕的片段两条相同（demo 两段字幕 × 两个视频）
    n_frames = sum(1 for r in index.images.records.values() if r.clip)
    assert len(index.images.video) == n_frames and len(index.images.video_joint) == n_frames
    joint, video = index.images.video_joint.get_vectors([r.image_id for r in demo]), index.images.video.get_vectors([r.image_id for r in demo])
    assert sum(not np.allclose(joint[r.image_id], video[r.image_id]) for r in demo) == 2
    assert index.images.stats()["video_vectors"] == n_frames


@needs_ffmpeg
def test_text_and_image_to_video_routes(kb, videos):
    settings, index, _ = kb
    r = MultimodalRetriever(settings, index)
    res = r.text_to_video("各区域的销售额")          # 只出现在字幕里：画面向量找不到，"视频 + 字幕"向量能找到
    assert {"video", "video_joint"} <= set(res.routes)
    top_joint = index.images.get(res.routes["video_joint"][0].chunk_id)
    assert top_joint is not None and top_joint.time_start == 0.0 and "销售额" in top_joint.subtitle
    assert res.segments and res.segments[0].start == 0.0

    res = r.image_to_video(videos["sunset"])        # 拍一张"日落"找视频里的那一段
    assert {"visual", "video:image→video"} <= set(res.routes)
    assert res.segments and (res.segments[0].start, res.segments[0].end) == (8.0, 14.0)
    assert res.mode == "image2video"


@needs_ffmpeg
def test_reembed_from_clip_frames_on_model_switch(kb, tmp_path):
    settings, index, emb = kb
    index.save()
    other = FakeOmni(model="fake-omni-v2")
    loaded = Index.load(settings, index.directory, image_embedder=other)
    loaded.images.ensure_fresh()                    # 换了模型：用磁盘上的采样帧重算，不用重新解析视频
    assert other.video_calls >= 1 and len(loaded.images.video) == len(index.images.video)
    plain = Index.load(settings, index.directory, image_embedder=PerceptualImageEmbedder())
    plain.images.ensure_fresh()                     # 不能编码视频的模型：视频向量清空（不在同一空间）
    assert len(plain.images.video) == 0 and len(plain.images.video_joint) == 0


@needs_ffmpeg
def test_video_dedup_and_similar(kb):
    _, index, _ = kb
    d = ContentDeduper(index.images)
    docs = {r.doc_title: r.doc_id for r in index.images.records.values()}
    demo, copy, slides = docs["demo"], docs["demo_reupload"], docs["slides"]
    sims = d.similar_videos(demo)
    assert sims[0].doc_id == copy and sims[0].duplicate and sims[0].overlap == 1.0 and len(sims[0].pairs) == 4
    assert not next(s for s in sims if s.doc_id == slides).duplicate
    groups = d.duplicate_groups(("video",))
    assert [sorted(g.ids) for g in groups] == [sorted([demo, copy])]
    frame = next(r for r in index.images.of_document(demo) if r.time_start == 8.0)
    top = d.similar_images(frame.image_id, top_k=3)  # 同一视频的其他片段不算，最像的是重传视频里的同一段
    assert index.images.get(top[0][0]).doc_id == copy and index.images.get(top[0][0]).time_start == 8.0


def test_image_dedup_groups(tmp_path):
    settings = _settings(tmp_path)
    paths = []
    for name, color, size in (("a.png", (40, 90, 200), 600), ("a_small.png", (40, 90, 200), 300), ("b.png", (220, 50, 50), 600)):
        im = Image.new("RGB", (size, size * 2 // 3), "white")
        d = ImageDraw.Draw(im)
        for k, h in enumerate([0.3, 0.6, 0.45, 0.2] if name.startswith("a") else [0.2, 0.25, 0.7, 0.9]):
            d.rectangle([size * (0.1 + k * 0.2), size * 2 // 3 * (1 - h), size * (0.25 + k * 0.2), size * 2 // 3], fill=color)
        im.save(tmp_path / name)
        paths.append(tmp_path / name)
    pipeline = IngestPipeline(settings, index=Index(settings, image_embedder=FakeOmni()))
    assert not pipeline.ingest_paths(paths).failed
    groups = ContentDeduper(pipeline.index.images).duplicate_groups(("image",))
    titles = [sorted(pipeline.index.images.get(i).doc_title for i in g.ids) for g in groups]
    assert titles == [["a", "a_small"]]


# --------------------------------------------------------------------------------------
def test_sample_clip_even_and_bounded():
    frames = [Image.new("RGB", (64, 64), (i, i, i)) for i in range(3)]
    c = sample_clip(VideoClip(frames=frames, times=[0, 0.5, 1.0]), 16)
    assert len(c.frames) == 4 and c.times[-1] == 1.0          # 至少 4 帧、偶数（时间 patch 两帧一组）
    c = sample_clip(VideoClip(frames=frames * 20, times=[i * 0.5 for i in range(60)]), 16)
    assert len(c.frames) == 16 and c.times[0] == 0 and c.times[-1] == 29.5


def test_wemm_api_request_shape(monkeypatch):
    sent = []

    class Resp:
        def __init__(self, n):
            self.n = n

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"index": 0, "embedding": [float(self.n), 1.0, 0.0, 0.0]}]}

    def post(url, headers, json, timeout):
        sent.append((url, headers, json))
        return Resp(len(sent))

    import httpx

    monkeypatch.setattr(httpx, "post", post)
    emb = WeMMAPIEmbedder("tencent/WeMM-Embedding-2B", base_url="http://gpu:8001/v1/", api_key="k", dim=2)
    frames = [Image.new("RGB", (640, 360), "red")] * 5
    video, joint = emb.embed_videos([VideoClip(frames=frames, times=[0, 1, 2, 3, 4], text="台词"), VideoClip(frames=frames[:2], times=[0, 1])])
    assert video.shape == (2, 2) and joint.shape == (2, 2) and np.allclose(np.linalg.norm(video, axis=1), 1)   # Matryoshka 截到 2 维再归一化
    assert np.allclose(joint[1], video[1])                     # 没有字幕的片段：联合向量就是纯视频向量（不多发请求）
    assert len(sent) == 3 and sent[0][0] == "http://gpu:8001/v1/embeddings" and sent[0][1] == {"Authorization": "Bearer k"}
    part = sent[0][2]["messages"][0]["content"][0]
    assert part["type"] == "video_url" and part["video_url"]["url"].startswith("data:video/jpeg;base64,") and part["video_url"]["url"].count(",") == 6
    assert sent[2][2]["messages"][0]["content"][1] == {"type": "text", "text": "字幕 / 台词：台词"}
    q = emb.embed_query("海边日落", "video")
    assert sent[-1][2]["messages"][0]["content"][0]["text"] == "Find the video that best matches the given text: 海边日落" and q.shape == (2,)


# --------------------------------------------------------------------------------------
def test_multimodal_memory(tmp_path):
    pytest.importorskip("lancedb")
    import io

    from prism.config import MemoryConfig
    from prism.memory import LanceMemoryStore

    def png(color) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (200, 120), color).save(buf, format="PNG")
        return buf.getvalue()

    store = LanceMemoryStore(tmp_path / "lancedb", HashEmbedder(dim=64), MemoryConfig(min_score=0.0), mm_embedder=FakeOmni())
    car, _ = store.add("这是我的车", user_id="u1", media=(png((200, 30, 30)), "png"))
    car2, how = store.add("这是我的车", user_id="u1", media=(png((30, 30, 200)), "png"))
    assert how == "created" and car2.id != car.id                      # 配了不同照片的同一句话是两条记忆
    store.add("用户喜欢喝美式咖啡", user_id="u1")
    store.add("别人的车", user_id="u2", media=(png((200, 30, 30)), "png"))
    assert store.stats()["multimodal"]["media"] == 3

    hits = store.search("", user_id="u1", image=Image.new("RGB", (300, 180), (205, 35, 30)))   # 以图召回：红车那条排第一
    assert hits[0]["record"].id == car.id and all(h["record"].user_id == "u1" for h in hits)
    hits = store.search("美式咖啡", user_id="u1")
    assert hits[0]["record"].content == "用户喜欢喝美式咖啡"
    path = store.media.media_path(car.id)
    assert path is not None and path.is_file()
    assert store.delete(car.id) and not path.exists()
    store.clear("u2")
    assert store.media.count()["media"] == 1

    # 换图文向量模型：整表重算多模态向量，媒体文件保留
    again = LanceMemoryStore(tmp_path / "lancedb", HashEmbedder(dim=64), MemoryConfig(min_score=0.0), mm_embedder=FakeOmni(model="fake-omni-v2"))
    assert again.media.count() == {"rows": 2, "media": 1}
    with pytest.raises(ValueError):
        LanceMemoryStore(tmp_path / "plain", HashEmbedder(dim=64)).add("x", media=(png("red"), "png"))


# --------------------------------------------------------------------------------------
def _wemm_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache("tencent/WeMM-Embedding-2B", "config.json"), str)
    except Exception:
        return False


real = pytest.mark.skipif(not (os.environ.get("WEMM_TEST") and _wemm_cached()), reason="WEMM_TEST=1 且本机缓存了 tencent/WeMM-Embedding-2B 才跑")


@pytest.fixture(scope="module")
def wemm():
    from prism.multimodal.wemm import WeMMEmbedder

    return WeMMEmbedder("tencent/WeMM-Embedding-2B", device=os.environ.get("WEMM_DEVICE"), video_max_frames=8, image_max_pixels=256 * 32 * 32, video_max_pixels=64 * 32 * 32)


@real
def test_real_wemm_single_pass_equals_two_pass(wemm, videos):
    """单次前向的第一个 <embedding> 只看得到视频 token：和单独编码"视频 + <embedding>"应当几乎一样；
    末尾的联合向量与分两次前向的"视频 + 字幕"向量高度一致。"""
    from test_video import _bar

    frames = [_bar(i) for i in range(8)]
    clip = VideoClip(frames=frames, times=[i * 0.5 for i in range(8)], text="这是各区域的销售额柱状图")
    wemm.dual_pass = True
    v1, j1 = wemm.embed_videos([clip])
    wemm.dual_pass = False
    v2, j2 = wemm.embed_videos([clip])
    wemm.dual_pass = True
    assert float(v1[0] @ v2[0]) > 0.9 and float(j1[0] @ j2[0]) > 0.9
    assert float(v1[0] @ j1[0]) < 0.999                         # 两个向量确实不同：字幕改变了联合向量


@real
def test_real_wemm_cross_modal(wemm, videos):
    from test_video import _bar, _pie, _sunset

    imgs = [_bar(0), _pie(0), _sunset(0)]
    iv = wemm.embed_images(imgs)
    tv = np.vstack([wemm.embed_query(q, "image") for q in ("a bar chart", "a pie chart", "a sunset over the sea")])
    assert (tv @ iv.T).argmax(axis=1).tolist() == [0, 1, 2]
    clips = [VideoClip(frames=[f(i) for i in range(8)], times=[i * 0.5 for i in range(8)]) for f in (_bar, _sunset)]
    vv, _ = wemm.embed_videos(clips, with_text=False)
    q = wemm.embed_query("海边日落", "video")
    assert float(q @ vv[1]) > float(q @ vv[0])
