"""精排层：Reranker 的五种实现。

Bi-encoder（向量检索）各自编码再比距离，看不到 query 与文档词语之间的交互；
Cross-encoder 把 (query, chunk) 拼成一对整体打分，精度高一个档次，但每个候选都要跑一次，
所以只对召回后的候选集做精排——候选数必须 ≥ 50（20 候选 R@5 = 0.458，50 候选 = 0.826，100 候选 = 0.888）。

- CrossEncoderReranker：**默认**，BAAI/bge-reranker-v2-m3（中英双语，开源首选，本地推理；直接用 transformers 加载）
- CohereReranker：Cohere Rerank v2（rerank-v4.0-pro / rerank-v4.0-fast / rerank-v3.5 …）
- APIReranker：Jina / SiliconFlow 等 `/rerank` 端点（{model, query, documents, top_n} → results[{index, relevance_score}]）
- LLMReranker：大模型批量打分（0-100 分 JSON），没有重排模型时的替代方案（慢：一次查询十秒量级）
- LexicalReranker：确定性词频 + 位置加权，离线兜底

`provider: auto` 的选择顺序：显式配置的 API → 本地交叉编码器（装了 torch + transformers）→ LLM → 词法。
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..config import RerankConfig
from ..gpu import MPS_LOCK, mps_guard, on_mps

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
COHERE_API_URL = "https://api.cohere.com/v2"
COHERE_DEFAULT_MODEL = "rerank-v4.0-pro"
COHERE_MODELS = ("rerank-v4.0-pro", "rerank-v4.0-fast", "rerank-v3.5", "rerank-multilingual-v3.0", "rerank-english-v3.0")


class BaseReranker(ABC):
    name = "base"

    @abstractmethod
    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
        """candidates: [(id, text)] → [(id, score)] 降序，score 归一化到 0-1。"""

    def pop_note(self) -> str | None:
        """上一次 rerank 有没有需要让调用方知道的事（比如模型还在下载、这次用的是兜底精排）。取走即清空。"""
        return None


class NoopReranker(BaseReranker):
    """不重排：保持融合顺序，按位置给出递减分数。"""

    name = "none"

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
        n = max(1, len(candidates))
        return [(cid, 1.0 - i / n) for i, (cid, _) in enumerate(candidates[:top_k])]


class LexicalReranker(BaseReranker):
    """本地确定性精排：query 关键词在候选中的词频与首次出现位置加权，再与原排名融合。"""

    name = "lexical"

    _TOKEN = re.compile(r"[一-龥]{1,4}|[A-Za-z0-9][A-Za-z0-9\-\._]*")

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
        if not candidates:
            return []
        terms = self._terms(query)
        n = len(candidates)
        scored: list[tuple[str, float]] = []
        for rank, (cid, text) in enumerate(candidates):
            low = text.lower()
            term_score = 0.0
            for t in terms:
                idx = low.find(t)
                if idx >= 0:
                    freq = low.count(t)
                    term_score += (2.0 if len(t) > 1 else 0.5) * (1.0 + min(freq, 5) * 0.2)
                    if idx < 120:
                        term_score += 0.5
            position_score = (n - rank) / n              # 保留融合层的排序信号
            scored.append((cid, term_score + position_score))
        hi = max(s for _, s in scored) or 1.0
        scored = [(cid, s / hi) for cid, s in scored]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _terms(self, query: str) -> list[str]:
        toks = [t.lower() for t in self._TOKEN.findall(query)]
        # 中文再补双字 n-gram，提升短语匹配
        extra: list[str] = []
        for t in toks:
            if re.fullmatch(r"[一-龥]{3,4}", t):
                extra.extend(t[i:i + 2] for i in range(len(t) - 1))
        return list(dict.fromkeys(toks + extra))


# --------------------------------------------------------------------------------------
# 本地交叉编码器（默认 BAAI/bge-reranker-v2-m3）
# --------------------------------------------------------------------------------------
_LOADED: dict[tuple[str, str], tuple[Any, Any, str]] = {}   # (模型名, 设备) → (tokenizer, model, device)：换配置不重复加载 2GB 权重
_LOAD_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()                               # 非 MPS 设备上推理串行：单卡上并发没有收益，还会互相挤显存
                                                             # （MPS 上用全进程共享的 MPS_LOCK，与向量模型 / MinerU 互斥，见 prism/gpu.py）


def cross_encoder_available() -> bool:
    """本地交叉编码器只需要 torch + transformers（装了 `mineru[torch]` 或 `rag-agent-studio[models]` 就有）。
    环境变量 RAG_DISABLE_LOCAL_RERANKER=1 可以整体关掉（测试 / 不想在这台机器上加载 2GB 模型时）。"""
    if os.environ.get("RAG_DISABLE_LOCAL_RERANKER") == "1":
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def _auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cross_encoder_cached(model: str) -> bool:
    """权重是否已在本机（本地目录，或 Hugging Face 缓存里有 config + 权重文件）。不联网。"""
    if Path(model).is_dir():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    def hit(filename: str) -> bool:
        return isinstance(try_to_load_from_cache(model, filename), str)
    return hit("config.json") and (hit("model.safetensors") or hit("pytorch_model.bin"))


def _load_cross_encoder(model: str, device: str | None, local_only: bool) -> tuple[Any, Any, str]:
    """加载 tokenizer + 序列分类模型。local_only=True 时绝不联网（缓存里没有就抛错）。"""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = device or _auto_device()
    key = (model, dev)
    with _LOAD_LOCK:
        if key in _LOADED:
            return _LOADED[key]
        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_only)
        net = AutoModelForSequenceClassification.from_pretrained(model, local_files_only=local_only)
        net.eval()
        if dev.startswith("cuda"):
            net.half()
        with mps_guard(dev):
            net.to(dev)
        logger.info("已加载重排模型 %s（%s，%.1fs）", model, dev, time.perf_counter() - t0)
        _LOADED[key] = (tokenizer, net, dev)
        return _LOADED[key]


def _cross_encoder_logits(bundle: tuple[Any, Any, str], query: str, texts: list[str], max_length: int, batch_size: int) -> list[float]:
    """(query, text) 逐对打分，返回 logit。按长度排序后分批，减少 padding。"""
    import torch

    tokenizer, net, dev = bundle
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [0.0] * len(texts)
    with (MPS_LOCK if on_mps(dev) else _INFER_LOCK), torch.inference_mode():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = tokenizer([query] * len(idx), [texts[i] for i in idx], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(dev)
            logits = net(**enc).logits.view(-1).float().cpu().tolist()
            for i, v in zip(idx, logits):
                out[i] = float(v)
    return out


class CrossEncoderReranker(BaseReranker):
    """本地交叉编码器。构造不读权重，第一次 rerank 才加载（本机已有缓存时约 1 秒）。

    权重不在本机时：
    - 没给 fallback（CLI / 显式选择）：按 Hugging Face 的常规方式同步下载后使用；
    - 给了 fallback（Web 服务）：不让一次查询卡在几个 GB 的下载上——`auto_download` 时后台下载，
      下载 / 加载期间先用 fallback 精排，并通过 pop_note() 说明这次用的是谁。
    """

    name = "cross_encoder"

    def __init__(
        self,
        model: str = DEFAULT_RERANK_MODEL,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
        fallback: BaseReranker | None = None,
        auto_download: bool = True,
    ) -> None:
        if not cross_encoder_available():
            raise RuntimeError("本地重排模型需要 torch + transformers：uv pip install 'rag-agent-studio[models]'（或已包含它们的 'mineru[torch]'）")
        self.model_name = model
        self.device = device
        self.batch_size = max(1, batch_size)
        self.max_length = max_length
        self.fallback = fallback
        self.auto_download = auto_download
        self._bundle: tuple[Any, Any, str] | None = None
        self._state = "idle"                  # idle | loading | ready | failed
        self._error: str | None = None
        self._note: str | None = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    def status(self) -> dict[str, Any]:
        return {"state": self._state, "cached": cross_encoder_cached(self.model_name), "device": self._bundle[2] if self._bundle else self.device, "error": self._error}

    def pop_note(self) -> str | None:
        note, self._note = self._note, None
        return note

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
        if not candidates:
            return []
        if not self._ready():
            assert self.fallback is not None
            return self.fallback.rerank(query, candidates, top_k)
        logits = _cross_encoder_logits(self._bundle, query, [text for _, text in candidates], self.max_length, self.batch_size)
        scored = [(cid, _logit_to_score(v)) for (cid, _), v in zip(candidates, logits)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ----------------------------------------------------------------------------------
    def _ready(self) -> bool:
        if self._state == "ready":
            return True
        with self._lock:
            if self._state == "ready":
                return True
            if self.fallback is None:
                self._load(local_only=False)          # 同步：需要的话下载；失败直接抛给调用方
                return True
            via = getattr(self.fallback, "name", "fallback")
            if self._state == "loading":
                self._note = f"重排模型 {self.model_name} 正在下载 / 加载，本次用 {via} 精排"
                return False
            if self._state == "failed":
                self._note = f"重排模型 {self.model_name} 不可用（{self._error}），本次用 {via} 精排"
                return False
            if cross_encoder_cached(self.model_name):
                try:
                    self._load(local_only=True)
                    return True
                except Exception as exc:
                    self._fail(exc)
                    self._note = f"重排模型 {self.model_name} 加载失败（{self._error}），本次用 {via} 精排"
                    return False
            if not self.auto_download:
                self._note = f"重排模型 {self.model_name} 尚未下载（执行 `rag download-reranker`），本次用 {via} 精排"
                return False
            self._spawn_loader()
            self._note = f"重排模型 {self.model_name} 首次使用，已开始后台下载，本次用 {via} 精排"
            return False

    def start_background_load(self) -> None:
        """后台下载 / 加载权重（幂等）。失败后可以再次调用重试。"""
        with self._lock:
            if self._state not in ("ready", "loading"):
                self._spawn_loader()

    def _spawn_loader(self) -> None:
        self._state, self._error = "loading", None
        threading.Thread(target=self._background_load, name="reranker-download", daemon=True).start()

    def _load(self, local_only: bool) -> None:
        self._bundle = _load_cross_encoder(self.model_name, self.device, local_only)
        self._state, self._error = "ready", None

    def _fail(self, exc: Exception) -> None:
        self._state, self._error = "failed", f"{type(exc).__name__}: {exc}"[:300]
        logger.warning("重排模型 %s 不可用：%s", self.model_name, self._error)

    def _background_load(self) -> None:
        try:
            bundle = _load_cross_encoder(self.model_name, self.device, False)
        except Exception as exc:
            self._fail(exc)
            return
        self._bundle, self._state, self._error = bundle, "ready", None


def download_cross_encoder(model: str = DEFAULT_RERANK_MODEL) -> str:
    """把重排模型权重拉到本地 Hugging Face 缓存（阻塞），返回快照目录。国内网络可先设 HF_ENDPOINT=https://hf-mirror.com。"""
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])


# --------------------------------------------------------------------------------------
# HTTP 重排接口
# --------------------------------------------------------------------------------------
class APIReranker(BaseReranker):
    """Jina / SiliconFlow / Cohere 兼容的 rerank HTTP 接口：POST {base}/rerank，Bearer 鉴权。"""

    name = "api"

    def __init__(
        self,
        api_url: str,
        api_key: str | None,
        model: str,
        timeout: float = 20.0,
        *,
        extra_payload: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        max_retries: int = 2,
        max_documents: int = 1000,
    ) -> None:
        url = api_url.rstrip("/")
        if not url:
            raise ValueError("rerank API 需要 base URL")
        if not url.endswith("/rerank"):
            url += "/rerank"
        self.api_url = url
        self.api_key = api_key
        self.model_name = model
        self.timeout = timeout
        self.extra_payload = dict(extra_payload or {})
        self.extra_headers = dict(extra_headers or {})
        self.max_retries = max(0, max_retries)
        self.max_documents = max(1, max_documents)

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
        if not candidates:
            return []
        out: list[tuple[str, float]] = []
        # 相关度分数是逐对算的，分批请求后可以直接合并排序
        for start in range(0, len(candidates), self.max_documents):
            batch = candidates[start:start + self.max_documents]
            data = self._post({"model": self.model_name, "query": query, "documents": [t for _, t in batch], "top_n": min(top_k, len(batch)), **self.extra_payload})
            for item in data.get("results") or data.get("data") or []:
                idx = int(item.get("index", -1))
                if 0 <= idx < len(batch):
                    out.append((batch[idx][0], float(item.get("relevance_score", item.get("score", 0.0)))))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_k]

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "Accept": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps(payload).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.api_url, data=body, method="POST", headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = _http_error_detail(exc)
                last = RuntimeError(f"Rerank API 返回 {exc.code}：{detail}")
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise last from exc
                time.sleep(_retry_after(exc, attempt))
            except urllib.error.URLError as exc:
                last = RuntimeError(f"Rerank API 调用失败：{exc.reason}")
                if attempt == self.max_retries:
                    raise last from exc
                time.sleep(0.5 * (attempt + 1))
        raise last or RuntimeError("Rerank API 调用失败")


class CohereReranker(APIReranker):
    """Cohere Rerank v2：`POST https://api.cohere.com/v2/rerank`。

    请求 {model, query, documents, top_n, max_tokens_per_doc}，响应 results[{index, relevance_score}]（0~1）。
    单次最多建议 1000 篇文档；超过模型上下文的文档由服务端按 max_tokens_per_doc（默认 4096）截断。
    API Key：参数 → 环境变量 COHERE_API_KEY → CO_API_KEY。
    """

    name = "cohere"

    def __init__(self, api_key: str | None = None, model: str | None = None, api_url: str | None = None, *, max_tokens_per_doc: int | None = None, timeout: float = 20.0) -> None:
        key = api_key or os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
        if not key:
            raise ValueError("Cohere Rerank 需要 API Key：在「模型配置」里添加 Cohere，或设置环境变量 COHERE_API_KEY")
        super().__init__(
            api_url or COHERE_API_URL,
            key,
            model or COHERE_DEFAULT_MODEL,
            timeout,
            extra_payload={"max_tokens_per_doc": int(max_tokens_per_doc)} if max_tokens_per_doc else None,
            extra_headers={"X-Client-Name": "prism"},
        )


def is_cohere_url(url: str | None) -> bool:
    host = (urlparse(url or "").hostname or "").lower()
    return host == "cohere.com" or host.endswith(".cohere.com") or host.endswith(".cohere.ai")


def build_api_reranker(api_url: str, api_key: str | None, model: str) -> APIReranker:
    """按 base URL 选实现：Cohere 的地址走 CohereReranker，其余走通用 /rerank 协议。"""
    if is_cohere_url(api_url):
        return CohereReranker(api_key, model, api_url)
    return APIReranker(api_url, api_key, model)


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return exc.reason or ""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            err = data.get("error")
            msg = data.get("message") or data.get("detail") or (err.get("message") if isinstance(err, dict) else err)
            if msg:
                return str(msg)[:300]
    except ValueError:
        pass
    return raw[:300] or str(exc.reason)


def _retry_after(exc: urllib.error.HTTPError, attempt: int) -> float:
    try:
        return min(10.0, max(0.0, float(exc.headers.get("Retry-After", ""))))
    except (TypeError, ValueError):
        return 0.8 * (2 ** attempt)


class LLMReranker(BaseReranker):
    """大模型批量打分：每批 N 个文档，输出 [{"id": n, "score": 0-100}]；解析失败回退词法精排。"""

    name = "llm"

    _PROMPT = """你是搜索精排助手。请根据用户问题，评估下列每个文档片段的相关度（0-100）：
90-100 精准解答问题；70-89 高度相关、提供核心背景；40-69 部分相关；0-39 不相关。
只输出一个 JSON 数组，不要任何解释，格式：[{{"id": 1, "score": 95}}, {{"id": 2, "score": 40}}]

用户问题：{query}

待评估文档：
{docs}"""

    def __init__(self, llm, batch_size: int = 20) -> None:
        self.llm = llm
        self.batch_size = max(1, batch_size)
        self._fallback = LexicalReranker()

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
        if not candidates:
            return []
        scores: dict[str, float] = {}
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start:start + self.batch_size]
            docs = "\n".join(f"【{i + 1}】{text[:800]}\n---" for i, (_, text) in enumerate(batch))
            try:
                raw = self.llm.complete(self._PROMPT.format(query=query, docs=docs), max_tokens=800)
                parsed = _parse_json_array(raw)
                for item in parsed:
                    i = int(item.get("id", 0)) - 1
                    if 0 <= i < len(batch):
                        scores[batch[i][0]] = float(item.get("score", 0.0)) / 100.0
            except Exception as exc:
                logger.warning("LLM 精排失败，回退词法精排：%s", exc)
                return self._fallback.rerank(query, candidates, top_k)
        if not scores:
            return self._fallback.rerank(query, candidates, top_k)
        n = len(candidates)
        scored = [(cid, scores.get(cid, max(0.0, 0.3 - i / n))) for i, (cid, _) in enumerate(candidates)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def _parse_json_array(raw: str) -> list[dict]:
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("no json array")
    data = json.loads(m.group(0))
    if isinstance(data, dict):
        data = data.get("results") or []
    return [x for x in data if isinstance(x, dict)]


def _logit_to_score(x: float) -> float:
    """交叉编码器输出 logit → 0~1 相关度。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def build_reranker(cfg: RerankConfig, llm=None) -> BaseReranker:
    """按配置构建 reranker（CLI / 测试用；Web 服务在 app/services/model_service.py 里还会看模型配置中心）。"""
    if not cfg.enabled or cfg.provider == "none":
        return NoopReranker()
    soft = LLMReranker(llm, batch_size=cfg.llm_batch_size) if llm is not None else LexicalReranker()
    provider = cfg.provider
    auto = provider == "auto"
    if auto:
        if cfg.api_url:
            provider = "cohere" if is_cohere_url(cfg.api_url) else "api"
        elif cross_encoder_available():
            provider = "cross_encoder"
        elif llm is not None:
            provider = "llm"
        else:
            provider = "lexical"
    if provider == "cross_encoder":
        # auto：权重不在本机时不阻塞也不后台下载（CLI 进程活不到下载完），退到 LLM / 词法并提示；显式选择则同步下载
        return CrossEncoderReranker(cfg.model, device=cfg.device, fallback=soft if auto else None, auto_download=False)
    if provider == "cohere":
        return CohereReranker(cfg.api_key, cfg.cohere_model, cfg.api_url if is_cohere_url(cfg.api_url) else None)
    if provider == "api":
        if not cfg.api_url:
            raise ValueError("api reranker 需要 retrieval.rerank.api_url")
        return build_api_reranker(cfg.api_url, cfg.api_key, cfg.model)
    if provider == "llm":
        if llm is None:
            logger.warning("llm reranker 未提供 LLM，回退词法精排")
            return LexicalReranker()
        return LLMReranker(llm, batch_size=cfg.llm_batch_size)
    if provider == "lexical":
        return LexicalReranker()
    raise ValueError(f"未知 reranker provider: {cfg.provider}")
