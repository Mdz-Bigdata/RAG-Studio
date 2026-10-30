"""WeMM-Embedding（腾讯，Qwen3.5 底座）：文本 / 图片 / 视频 / 图文 / 视频 + 字幕映射到同一个向量空间。

和 CLIP 双塔的区别：一个解码器同时看所有模态，向量取序列末尾 <embedding> 位置的隐状态（L2 归一化）。
- 文搜图、图搜图、图搜文与 CLIP 用法相同（BaseImageEmbedder 接口），中文和图中文字（报表 / 扫描件）理解强得多；
- 视频是原生输入：一个片段的若干帧整体编码，表达的是"这段视频在发生什么"，而不是某一帧长什么样；
- 单次前向双向量：因果注意力下，<embedding> 的隐状态只取决于它之前的 token。把一个 <embedding> 放在视频 token 之后、
  字幕之前，另一个放在序列末尾，一次前向同时得到"纯视频向量"和"视频 + 字幕向量"（技术报告 §3.1）；
  dual_pass=False 时改为分两次前向（结果等价于各自单独编码，慢一倍）；
- Matryoshka：取前 dim 维再归一化即可降维（2B 全长 2048，256 维保留约 98.7% 效果）；
- 不支持音频：视频的声音不参与编码，台词要靠字幕轨 / ASR 文本进"视频 + 字幕"向量。

两种运行方式：
- WeMMEmbedder：本地 transformers（trust_remote_code），CUDA 最快；Apple 芯片可用 MPS（与其他本地模型共用 prism/gpu.py 的锁）；
- WeMMAPIEmbedder：vLLM 部署的 OpenAI 兼容 /embeddings（vllm serve <model> --runner pooling），一次请求一条 messages；
  vLLM 只取末尾向量，"视频 + 字幕"要再发一次请求。
"""

from __future__ import annotations

import io
import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..gpu import auto_device, mps_guard
from .embedders import BaseImageEmbedder, _normalize, image_to_base64

logger = logging.getLogger(__name__)

EMBEDDING_TOKEN = "<embedding>"
_FACTOR = 32                 # patch 16 × merge 2：送入模型的边长必须是它的整数倍

# 检索指令（写在内容之后，与官方评测一致）。文本查询按目标模态给不同的指令；图片 / 视频作为文档不加指令，
# 这样图片既能当查询（图搜图）也能当文档，纯视频向量和"视频 + 字幕"向量的前缀完全一致（单次前向双向量的前提）
DEFAULT_INSTRUCTIONS: dict[str, str] = {
    "image": "Find the image that best matches the given text: ",
    "video": "Find the video that best matches the given text: ",
    "text": "",
    "subtitle": "字幕 / 台词：",
}


@dataclass
class VideoClip:
    """一个视频片段：按时间顺序的帧（PIL.Image）与各帧相对片段开头的时刻（秒），text 是这段的字幕 / 台词。"""

    frames: list[Any]
    times: list[float] = field(default_factory=list)
    text: str = ""


def _fit(image: Any, max_pixels: int, min_pixels: int = 4 * _FACTOR * _FACTOR) -> Any:
    """等比缩放到像素数不超过 max_pixels、边长取 32 的整数倍。"""
    im = image.convert("RGB")
    w, h = im.size
    scale = 1.0
    if w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
    elif w * h < min_pixels:
        scale = math.sqrt(min_pixels / (w * h))
    nw = max(_FACTOR, int(w * scale) // _FACTOR * _FACTOR)
    nh = max(_FACTOR, int(h * scale) // _FACTOR * _FACTOR)
    return im if (nw, nh) == (w, h) else im.resize((nw, nh))


def sample_clip(clip: VideoClip, max_frames: int) -> VideoClip:
    """均匀抽到不超过 max_frames 帧，且为偶数、至少 4 帧（模型按相邻两帧一组做时间 patch）。"""
    frames, times = list(clip.frames), list(clip.times) or [i * 0.5 for i in range(len(clip.frames))]
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames, times = [frames[i] for i in idx], [times[i] for i in idx]
    while len(frames) < 4 or len(frames) % 2:
        frames.append(frames[-1])
        times.append(times[-1])
    return VideoClip(frames=frames, times=times, text=clip.text)


class _WeMMBase(BaseImageEmbedder):
    cross_modal = True
    video_native = True
    # 2B 实测（合成图表 / 幻灯片 / 风景）：无关图片之间 0.33~0.53，只换了文字的同版式幻灯片 0.66；
    # 同一张图缩小一半 + JPEG q40 + 轻微模糊后 0.87~0.94，同一镜头里稍后的一帧 0.97
    similar_threshold = 0.8      # 图搜文："相似图片"门槛
    dedup_threshold = 0.85       # 查重：翻拍 / 转码 / 压缩过的同一张图

    def __init__(self, *, dim: int | None, instructions: dict[str, str] | None, video_max_frames: int) -> None:
        self.out_dim = dim
        self.instructions = {**DEFAULT_INSTRUCTIONS, **(instructions or {})}
        self.video_max_frames = max(4, video_max_frames)

    def _post(self, arr: np.ndarray) -> np.ndarray:
        """Matryoshka 截断 + 归一化。"""
        arr = np.asarray(arr, dtype=np.float32)
        if self.out_dim and arr.ndim == 2 and arr.shape[1] > self.out_dim:
            arr = arr[:, : self.out_dim]
        arr = _normalize(arr) if arr.size else arr.reshape(0, self.dim or (self.out_dim or 0))
        if arr.size:
            self.dim = arr.shape[1]
        return arr

    def embed_query(self, text: str, target: str = "image") -> np.ndarray:
        return self.embed_queries([text], target)[0]

    def embed_queries(self, texts: list[str], target: str = "image") -> np.ndarray:
        prefix = self.instructions.get(target, "")
        return self.embed_texts([prefix + (t or " ") for t in texts])

    def embed_text(self, text: str) -> np.ndarray:
        # 检索器里 embed_text 用于文本查询（文搜图 / 中性基线），embed_texts 用于正文块（图搜文的文档侧，不加指令）
        return self.embed_query(text, "image")

    def subtitle_text(self, text: str) -> str:
        return self.instructions.get("subtitle", "") + " ".join((text or "").split())

    def signature(self) -> dict[str, Any]:
        return {**super().signature(), "video": True}


class WeMMEmbedder(_WeMMBase):
    """本地 transformers 推理。需要 transformers ≥ 5.2（Qwen3.5）与 torch；权重约 5.4GB（2B，bf16）。"""

    name = "wemm"

    def __init__(self, model: str = "tencent/WeMM-Embedding-2B", *, device: str | None = None, batch_size: int = 4, dim: int | None = None,
                 instructions: dict[str, str] | None = None, image_max_pixels: int = 1024 * _FACTOR * _FACTOR, video_max_pixels: int = 256 * _FACTOR * _FACTOR,
                 video_max_frames: int = 16, dual_pass: bool = True, image_min_pixels: int = 256 * _FACTOR * _FACTOR) -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("WeMM-Embedding 需要 torch + transformers≥5.2：pip install 'rag-agent-studio[multimodal]'") from exc
        super().__init__(dim=dim, instructions=instructions, video_max_frames=video_max_frames)
        self._torch = torch
        self.device = device or auto_device()
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.processor = transformers.AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "right"
        self.model = transformers.AutoModel.from_pretrained(model, trust_remote_code=True, dtype=dtype, attn_implementation="sdpa")
        self._lock = threading.Lock()
        with mps_guard(self.device):
            self.model.to(self.device).eval()
        self.model_name = model
        self.batch_size = max(1, batch_size)
        self.image_max_pixels = image_max_pixels
        # 小图放大到至少约 256 个视觉 token：同一张图的缩略图 / 转发压缩版和原图才在同一个分辨率上比较
        # （实测 149px 的压缩小图与原图余弦 0.79 → 0.86，与无关图片仍在 0.37~0.56）
        self.image_min_pixels = min(image_min_pixels, image_max_pixels)
        self.video_max_pixels = video_max_pixels
        self.dual_pass = dual_pass
        self.emb_id = self.processor.tokenizer.convert_tokens_to_ids(EMBEDDING_TOKEN)
        hidden = int(getattr(getattr(self.model.config, "text_config", self.model.config), "hidden_size", 0) or 0)
        self.dim = min(hidden, dim) if dim and hidden else (dim or hidden)

    # --- 输入构造 ---------------------------------------------------------------------------
    def _render(self, content: list[dict[str, Any]]) -> str:
        return self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=False)

    def _forward(self, texts: list[str], *, images: list[Any] | None = None, videos: list[VideoClip] | None = None) -> list[np.ndarray]:
        """一批序列前向一次，返回每条序列里所有 <embedding> 位置的向量 (k, hidden)，按出现顺序。"""
        torch = self._torch
        kw: dict[str, Any] = {}
        if images:
            kw["images"] = images
        if videos:
            kw["videos"] = [[f for f in v.frames] for v in videos]
            # 帧已经按片段抽好：不让处理器再抽帧；fps / 帧序号只用来生成每组帧前的 "<t seconds>" 时间标记
            kw["video_metadata"] = [{"fps": 2.0, "frames_indices": [int(round(t * 2.0)) for t in v.times], "total_num_frames": len(v.frames)} for v in videos]
            kw["do_sample_frames"] = False
        inputs = self.processor(text=texts, padding=True, return_tensors="pt", **kw)
        ids = inputs["input_ids"]
        with self._lock, mps_guard(self.device), torch.no_grad():
            self.model.model.rope_deltas = None
            out = self.model.model(**{k: v.to(self.device) for k, v in inputs.items()})
            hidden = out.last_hidden_state
            res = []
            for row in range(ids.shape[0]):
                pos = (ids[row] == self.emb_id).nonzero(as_tuple=True)[0]
                if not len(pos):          # 处理器版本差异没有自动追加 <embedding> 时，退回到最后一个有效位置
                    pos = inputs["attention_mask"][row].sum().reshape(1) - 1
                res.append(hidden[row, pos.to(hidden.device)].float().cpu().numpy())
        return res

    def _run(self, texts: list[str], **kw: Any) -> np.ndarray:
        return np.vstack([v[-1:] for v in self._forward(texts, **kw)])

    # --- 接口 -------------------------------------------------------------------------------
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = [self._render([{"type": "text", "text": t if t.strip() else " "}]) for t in texts[i:i + self.batch_size]]
            vecs.append(self._run(batch))
        return self._post(np.vstack(vecs) if vecs else np.zeros((0, self.dim), dtype=np.float32))

    def embed_images(self, images: list[Any]) -> np.ndarray:
        vecs = []
        for i in range(0, len(images), self.batch_size):
            batch = [_fit(im, self.image_max_pixels, self.image_min_pixels) for im in images[i:i + self.batch_size]]
            vecs.append(self._run([self._render([{"type": "image"}])] * len(batch), images=batch))
        return self._post(np.vstack(vecs) if vecs else np.zeros((0, self.dim), dtype=np.float32))

    def embed_image_texts(self, pairs: list[tuple[Any, str]]) -> np.ndarray:
        """图 + 文 联合向量（商品图 + 标题、记忆里的图片 + 说明）。"""
        vecs = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            texts = [self._render([{"type": "image"}, {"type": "text", "text": t or " "}]) for _, t in batch]
            vecs.append(self._run(texts, images=[_fit(im, self.image_max_pixels, self.image_min_pixels) for im, _ in batch]))
        return self._post(np.vstack(vecs) if vecs else np.zeros((0, self.dim), dtype=np.float32))

    def embed_videos(self, clips: list[VideoClip], *, with_text: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        """视频片段 → (纯视频向量, 视频 + 字幕向量)。没有字幕的片段，联合向量那一行就是纯视频向量本身。
        with_text=False 只算纯视频向量（第二项返回 None）。"""
        prepared = [VideoClip([_fit(f, self.video_max_pixels) for f in c.frames], c.times, c.text) for c in (sample_clip(c, self.video_max_frames) for c in clips)]
        video_only, joint = [], []
        bs = max(1, self.batch_size // 2)
        for i in range(0, len(prepared), bs):
            batch = prepared[i:i + bs]
            if not with_text or self.dual_pass:
                texts = [self._dual_text(c) if with_text else self._render([{"type": "video"}]) for c in batch]
                for c, vecs in zip(batch, self._forward(texts, videos=batch)):
                    video_only.append(vecs[0])
                    joint.append(vecs[-1])
            else:
                video_only.extend(self._run([self._render([{"type": "video"}])] * len(batch), videos=batch))
                joint.extend(self._run([self._render([{"type": "video"}] + ([{"type": "text", "text": self.subtitle_text(c.text)}] if c.text.strip() else [])) for c in batch], videos=batch))
        v = self._post(np.vstack(video_only)) if video_only else np.zeros((0, self.dim), dtype=np.float32)
        return v, (self._post(np.vstack(joint)) if with_text and joint else None)

    def _dual_text(self, clip: VideoClip) -> str:
        """视频 token 之后插一个 <embedding>（纯视频向量），字幕接在后面，末尾的 <embedding>（视频 + 字幕向量）由分词器追加。
        没有字幕时只有末尾一个，两个向量取同一个。"""
        if not clip.text.strip():
            return self._render([{"type": "video"}])
        text = self._render([{"type": "video"}, {"type": "text", "text": self.subtitle_text(clip.text)}])
        marker = self.processor.video_token + self.processor.vision_end_token
        k = text.find(marker)
        return text if k < 0 else text[: k + len(marker)] + EMBEDDING_TOKEN + text[k + len(marker):]


class WeMMAPIEmbedder(_WeMMBase):
    """vLLM 部署的 WeMM：POST {base_url}/embeddings，body = {model, messages}（OpenAI 兼容的 chat 形式 embedding 请求）。
    视频以 data:video/jpeg;base64,<帧1>,<帧2>… 传逐帧 JPEG（vLLM 的视频 data URL 格式），不需要在本机重新编码 mp4。"""

    name = "wemm_api"

    def __init__(self, model: str, *, base_url: str, api_key: str | None = None, dim: int | None = None, instructions: dict[str, str] | None = None,
                 video_max_frames: int = 16, image_max_side: int = 1024, video_max_side: int = 512, timeout: float = 120.0) -> None:
        if not base_url:
            raise RuntimeError("WeMM API 需要 base_url（vLLM 服务地址，如 http://gpu-host:8000/v1）")
        super().__init__(dim=dim, instructions=instructions, video_max_frames=video_max_frames)
        self.model_name = model
        self.url = base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.timeout = timeout
        self.image_max_side = image_max_side
        self.video_max_side = video_max_side

    def _embed(self, content: list[dict[str, Any]]) -> np.ndarray:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = httpx.post(self.url, headers=headers, timeout=self.timeout,
                       json={"model": self.model_name, "messages": [{"role": "user", "content": content}], "encoding_format": "float"})
        r.raise_for_status()
        data = r.json().get("data") or []
        if not data:
            raise RuntimeError("WeMM API 没有返回向量")
        return np.asarray(data[0]["embedding"], dtype=np.float32)

    def _image_part(self, image: Any) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_to_base64(image, max_side=self.image_max_side)}}

    def _video_part(self, clip: VideoClip) -> dict[str, Any]:
        frames = ",".join(image_to_base64(f, max_side=self.video_max_side) for f in clip.frames)
        return {"type": "video_url", "video_url": {"url": "data:video/jpeg;base64," + frames}}

    def _many(self, contents: list[list[dict[str, Any]]]) -> np.ndarray:
        return self._post(np.vstack([self._embed(c) for c in contents])) if contents else np.zeros((0, self.dim), dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self._many([[{"type": "text", "text": t or " "}] for t in texts])

    def embed_images(self, images: list[Any]) -> np.ndarray:
        return self._many([[self._image_part(im)] for im in images])

    def embed_image_texts(self, pairs: list[tuple[Any, str]]) -> np.ndarray:
        return self._many([[self._image_part(im), {"type": "text", "text": t or " "}] for im, t in pairs])

    def embed_videos(self, clips: list[VideoClip], *, with_text: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        clips = [sample_clip(c, self.video_max_frames) for c in clips]
        video_only = self._many([[self._video_part(c)] for c in clips])
        if not with_text:
            return video_only, None
        joint = [self._embed([self._video_part(c), {"type": "text", "text": self.subtitle_text(c.text)}]) if c.text.strip() else None for c in clips]
        rows = [self._post(j[None, :])[0] if j is not None else video_only[n] for n, j in enumerate(joint)]
        return video_only, (np.vstack(rows) if rows else video_only)


def wemm_available() -> bool:
    """本地 WeMM 需要 transformers 里有 Qwen3.5（≥ 5.2）。"""
    try:
        import transformers
    except ImportError:
        return False
    return hasattr(transformers, "Qwen3_5ForConditionalGeneration")


def jpeg_bytes(image: Any, max_side: int = 512) -> bytes:
    """视频片段的采样帧在内存里以 JPEG 暂存（几百帧也只占几十 MB）。"""
    im = image.convert("RGB")
    if max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def from_jpeg(data: bytes) -> Any:
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")

