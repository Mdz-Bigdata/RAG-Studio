"""向量模型抽象。

- HashEmbedder：无模型、无网络、确定性（token 哈希桶），用于测试 / 冒烟 / 跑通结构
- SentenceTransformerEmbedder：BGE-M3 等（推荐；中英双语、8192 上下文、稠密 + 稀疏多功能）
- OpenAICompatibleEmbedder：DashScope text-embedding-v3、vLLM、Ollama 等 OpenAI 兼容端点

嵌入模型选型提醒：排行榜高 ≠ 业务效果好，一定用真实业务问题做 A/B；维度不是越高越好。
"""

from __future__ import annotations

import hashlib
import os
import re
from abc import ABC, abstractmethod

import numpy as np

from ..config import EmbeddingConfig
from ..gpu import auto_device, mps_guard

_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-\.]*|[一-鿿]")


class BaseEmbedder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]


class HashEmbedder(BaseEmbedder):
    """token（中文单字 + 双字、英文词）哈希到固定维度并 L2 归一化。只反映词面重叠，不含真实语义。"""

    name = "hash"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        toks = [t.lower() for t in _TOKEN.findall(text)]
        bigrams = [a + b for a, b in zip(toks, toks[1:]) if len(a) == 1 and len(b) == 1]
        return toks + bigrams

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in self._tokens(t):
                h = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder(BaseEmbedder):
    name = "sentence_transformers"

    def __init__(self, model: str, *, batch_size: int = 32, normalize: bool = True, query_instruction: str = "", device: str | None = None) -> None:
        # 测试 / 不想在这台机器上加载 2GB 模型时整体关掉：调用方会回退到哈希向量
        if os.environ.get("RAG_DISABLE_LOCAL_EMBEDDER") == "1":
            raise RuntimeError("本地向量模型已被环境变量 RAG_DISABLE_LOCAL_EMBEDDER=1 禁用")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("需要 sentence-transformers：pip install 'rag-agent-studio[models]'") from exc
        # 先在 CPU 上加载，再持 GPU 锁搬到目标设备：搬权重也要用 MPS stream，不能和别的线程的推理重叠（见 prism/gpu.py）
        target = device or auto_device()
        self.model = SentenceTransformer(model, device="cpu")
        if target != "cpu":
            with mps_guard(target):
                self.model.to(target)
        self.model_name = model
        self.batch_size = batch_size
        self.normalize = normalize
        self.query_instruction = query_instruction
        # sentence-transformers 6 把 get_sentence_embedding_dimension 改名为 get_embedding_dimension
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim() or 0)

    _LOCK_SLICE = 64   # 每次持 GPU 锁最多编码这么多条：大文档入库时，查询能插在两段之间，不用等整篇编完

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts = []
        for i in range(0, len(texts), self._LOCK_SLICE):
            with mps_guard(self.model.device):
                parts.append(self.model.encode(texts[i:i + self._LOCK_SLICE], batch_size=self.batch_size, normalize_embeddings=self.normalize, convert_to_numpy=True, show_progress_bar=False))
        return np.asarray(np.vstack(parts), dtype=np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        if self.query_instruction:
            texts = [self.query_instruction + t for t in texts]
        return self.embed_documents(texts)


class OpenAICompatibleEmbedder(BaseEmbedder):
    name = "openai_compatible"

    def __init__(self, model: str, *, base_url: str | None, api_key: str | None, batch_size: int = 32, normalize: bool = True) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("需要 openai 包：pip install 'rag-agent-studio[llm]'") from exc
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self.model_name = model
        self.batch_size = batch_size
        self.normalize = normalize
        self.dim = 0

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(model=self.model_name, input=texts[i:i + self.batch_size])
            vecs.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        arr = np.asarray(vecs, dtype=np.float32)
        if self.normalize and arr.size:
            arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
        self.dim = arr.shape[1] if arr.ndim == 2 else self.dim
        return arr


def build_embedder(cfg: EmbeddingConfig) -> BaseEmbedder:
    if cfg.provider == "hash":
        return HashEmbedder(dim=cfg.dim)
    if cfg.provider == "sentence_transformers":
        return SentenceTransformerEmbedder(cfg.model, batch_size=cfg.batch_size, normalize=cfg.normalize, query_instruction=cfg.query_instruction, device=cfg.device)
    if cfg.provider == "openai_compatible":
        return OpenAICompatibleEmbedder(cfg.model, base_url=cfg.base_url, api_key=os.environ.get(cfg.api_key_env), batch_size=cfg.batch_size, normalize=cfg.normalize)
    raise ValueError(f"未知 embedding provider: {cfg.provider}")
