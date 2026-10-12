"""图文统一嵌入：把图片和文本映射到同一个向量空间。

- CLIPEmbedder：HuggingFace transformers 的 CLIP / Chinese-CLIP 双塔模型（文本塔 + 图像塔，对比学习对齐）。
  原版 CLIP 以英文训练，中文查询请用 Chinese-CLIP（默认 OFA-Sys/chinese-clip-vit-base-patch16）。
- APIImageEmbedder：Jina 兼容的多模态 /embeddings 端点（jina-clip-v2 等），input 为 [{"text": ..}, {"image": <base64>}]。
- PerceptualImageEmbedder：离线兜底。颜色直方图 + 灰度缩略图 + 梯度方向直方图，只反映"看起来像不像"，
  能做图搜图 / 去重；没有文本塔（cross_modal=False），文搜图此时只走"描述文本"两路。

所有向量都做 L2 归一化：内积 = 余弦相似度。
"""

from __future__ import annotations

import base64
import importlib.util
import io
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..config import ImageEmbeddingConfig
from ..gpu import mps_guard

logger = logging.getLogger(__name__)


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class BaseImageEmbedder(ABC):
    name: str = "base"
    model_name: str | None = None
    dim: int = 0
    cross_modal: bool = False            # True：有文本塔，文本向量可直接与图片向量比较
    similar_threshold: float = 0.9       # 图-图余弦达到多少才算"相似"。CLIP 类模型无关图片也有 0.7~0.85

    @abstractmethod
    def embed_images(self, images: list[Any]) -> np.ndarray:
        """images: PIL.Image 列表 → (n, dim) 已归一化。"""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} 没有文本塔，不支持跨模态文本向量")

    def embed_image(self, image: Any) -> np.ndarray:
        return self.embed_images([image])[0]

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def signature(self) -> dict[str, Any]:
        return {"embedder": self.name, "model": self.model_name, "dim": self.dim, "cross_modal": self.cross_modal}


class PerceptualImageEmbedder(BaseImageEmbedder):
    """无模型的感知特征：三段各自归一化后加权拼接，余弦 = 三段余弦的加权和。"""

    name = "perceptual"
    model_name = "perceptual-v1"
    cross_modal = False
    similar_threshold = 0.7
    _W_THUMB, _W_COLOR, _W_EDGE = 0.5, 0.3, 0.2

    def __init__(self) -> None:
        self.dim = 256 + 64 + 128

    def embed_images(self, images: list[Any]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._features(im) for im in images]).astype(np.float32)

    def _features(self, image: Any) -> np.ndarray:
        rgb = image.convert("RGB")
        small = np.asarray(rgb.resize((64, 64)), dtype=np.float32)
        gray = small @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

        # 1) 16×16 灰度缩略图（去均值）：整体构图
        thumb = gray.reshape(16, 4, 16, 4).mean(axis=(1, 3)).reshape(-1)
        thumb = thumb - thumb.mean()

        # 2) 4×4×4 RGB 联合直方图（开方压缩）：配色
        q = np.minimum((small // 64).astype(np.int32), 3)
        idx = (q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]).reshape(-1)
        color = np.sqrt(np.bincount(idx, minlength=64).astype(np.float32))
        color = color - color.mean()

        # 3) 4×4 网格 × 8 方向的梯度直方图：线条 / 纹理
        gy, gx = np.gradient(gray)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
        bins = np.minimum((ang * 8).astype(np.int32), 7)
        edge = np.zeros((4, 4, 8), dtype=np.float32)
        for cy in range(4):
            for cx in range(4):
                sl = (slice(cy * 16, cy * 16 + 16), slice(cx * 16, cx * 16 + 16))
                edge[cy, cx] = np.bincount(bins[sl].reshape(-1), weights=mag[sl].reshape(-1), minlength=8)
        edge = np.sqrt(edge.reshape(-1))

        parts = []
        for vec, w in ((thumb, self._W_THUMB), (color, self._W_COLOR), (edge, self._W_EDGE)):
            n = float(np.linalg.norm(vec))
            parts.append((vec / n if n > 0 else vec) * np.sqrt(w))
        return _normalize(np.concatenate(parts))[0]


class CLIPEmbedder(BaseImageEmbedder):
    """CLIP / Chinese-CLIP（transformers）。需要：pip install 'rag-agent-studio[multimodal]'（torch + transformers）。
    国内下载权重可设置 HF_ENDPOINT=https://hf-mirror.com。"""

    name = "clip"
    cross_modal = True

    def __init__(self, model: str, *, device: str | None = None, batch_size: int = 16, max_text_tokens: int | None = None) -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("图文向量模型需要 torch + transformers：pip install 'rag-agent-studio[multimodal]'") from exc
        self._torch = torch
        chinese = "chinese-clip" in model.lower() or "chinese_clip" in model.lower()
        if chinese:
            self.model = transformers.ChineseCLIPModel.from_pretrained(model)
            self.processor = transformers.ChineseCLIPProcessor.from_pretrained(model)
        else:
            self.model = transformers.CLIPModel.from_pretrained(model)
            self.processor = transformers.CLIPProcessor.from_pretrained(model)
        # 默认不用 Apple GPU（MPS）：PyTorch 的 MPS 后端不是线程安全的，Web 服务的请求线程与 MinerU（同样用 torch）
        # 并发使用 GPU 时会触发 Metal 断言、整个进程 abort（实测）。CLIP base 在 CPU 上每张图几十毫秒，足够用；
        # 确实要用 MPS 就显式设置 multimodal.embedding.device: mps（此时搬权重 / 推理与向量、重排模型、MinerU 共用
        # prism/gpu.py 的全局锁）。
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()   # 同一模型实例的推理串行化，防止多个请求线程同时进同一个模型
        with mps_guard(self.device):
            self.model.to(self.device).eval()
        self.model_name = model
        self.batch_size = batch_size
        self.dim = int(getattr(self.model.config, "projection_dim", 0) or 0)
        limit = int(getattr(self.model.config.text_config, "max_position_embeddings", 77) or 77)
        self.max_text_tokens = min(limit, max_text_tokens or (64 if chinese else 77))

    def _features(self, out: Any, kind: str) -> np.ndarray:
        torch = self._torch
        if not torch.is_tensor(out):  # 个别 transformers 版本返回 ModelOutput
            out = getattr(out, f"{kind}_embeds", None) if getattr(out, f"{kind}_embeds", None) is not None else getattr(out, "pooler_output", None)
        if out is None:
            raise RuntimeError("无法从模型输出中取得特征向量")
        out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return out.detach().cpu().float().numpy()

    def embed_images(self, images: list[Any]) -> np.ndarray:
        torch = self._torch
        vecs: list[np.ndarray] = []
        with self._lock, mps_guard(self.device), torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = [im.convert("RGB") for im in images[i:i + self.batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                vecs.append(self._features(self.model.get_image_features(**inputs), "image"))
        arr = np.vstack(vecs) if vecs else np.zeros((0, self.dim), dtype=np.float32)
        self.dim = arr.shape[1] if arr.size else self.dim
        return arr

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        vecs: list[np.ndarray] = []
        with self._lock, mps_guard(self.device), torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = [t if t.strip() else " " for t in texts[i:i + self.batch_size]]
                inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_tokens).to(self.device)
                vecs.append(self._features(self.model.get_text_features(**inputs), "text"))
        return np.vstack(vecs) if vecs else np.zeros((0, self.dim), dtype=np.float32)


class APIImageEmbedder(BaseImageEmbedder):
    """Jina 兼容的多模态向量 API：POST {base_url}/embeddings，input 里文本与图片可混排。"""

    name = "api"
    cross_modal = True

    def __init__(self, model: str, *, base_url: str, api_key: str | None, batch_size: int = 8, timeout: float = 60.0) -> None:
        if not base_url:
            raise RuntimeError("多模态向量 API 需要 base_url")
        self.model_name = model
        self.url = base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout

    def _post(self, inputs: list[dict[str, str]]) -> np.ndarray:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        vecs: list[list[float]] = []
        for i in range(0, len(inputs), self.batch_size):
            r = httpx.post(self.url, headers=headers, json={"model": self.model_name, "input": inputs[i:i + self.batch_size]}, timeout=self.timeout)
            r.raise_for_status()
            data = sorted(r.json().get("data", []), key=lambda d: d.get("index", 0))
            vecs.extend(d["embedding"] for d in data)
        arr = _normalize(np.asarray(vecs, dtype=np.float32)) if vecs else np.zeros((0, self.dim), dtype=np.float32)
        self.dim = arr.shape[1] if arr.size else self.dim
        return arr

    def embed_images(self, images: list[Any]) -> np.ndarray:
        return self._post([{"image": image_to_base64(im, max_side=768)} for im in images])

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self._post([{"text": t or " "} for t in texts])


def image_to_base64(image: Any, *, max_side: int = 768, fmt: str = "JPEG") -> str:
    """PIL.Image → 裸 base64（默认缩小并转 JPEG，控制传输体积）。"""
    im = image.convert("RGB")
    if max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format=fmt, quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def clip_available() -> bool:
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


def build_image_embedder(cfg: ImageEmbeddingConfig) -> BaseImageEmbedder:
    """auto：能加载 CLIP 就用 CLIP，否则回退离线感知特征（并记录原因）。"""
    if cfg.provider == "perceptual":
        return PerceptualImageEmbedder()
    if cfg.provider == "api":
        return APIImageEmbedder(cfg.model, base_url=cfg.base_url or "", api_key=cfg.api_key or os.environ.get(cfg.api_key_env), batch_size=min(cfg.batch_size, 8))
    if cfg.provider == "clip":
        return CLIPEmbedder(cfg.model, device=cfg.device, batch_size=cfg.batch_size, max_text_tokens=cfg.max_text_tokens)
    if clip_available():
        try:
            return CLIPEmbedder(cfg.model, device=cfg.device, batch_size=cfg.batch_size, max_text_tokens=cfg.max_text_tokens)
        except Exception as exc:
            logger.warning("图文向量模型 %s 加载失败（%s），回退离线感知特征", cfg.model, exc)
    return PerceptualImageEmbedder()
