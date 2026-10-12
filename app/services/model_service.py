"""模型运行时：把"模型配置中心"里激活的配置解析成 prism 的 embedder / LLM / reranker 实例。

解析优先级：数据库激活配置 → .env 环境变量（旧版兼容）→ config.yaml → 离线兜底
（哈希向量 + 词法精排 + 抽取式回答）。所有实例按配置签名缓存；配置变更后 invalidate()。
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from prism.config import EmbeddingConfig, LLMConfig, RerankConfig
from prism.generation.llm import AnthropicLLM, BaseLLM, OpenAICompatibleLLM
from prism.index.embeddings import BaseEmbedder, HashEmbedder, OpenAICompatibleEmbedder, SentenceTransformerEmbedder
from prism.ingest.parsers.mineru_parser import mineru_available
from prism.multimodal import APIImageEmbedder, BaseImageEmbedder, CLIPEmbedder, PerceptualImageEmbedder, available_ocr, build_image_embedder, clip_available
from prism.retrieval.rerank import (
    BaseReranker,
    CohereReranker,
    CrossEncoderReranker,
    LexicalReranker,
    LLMReranker,
    NoopReranker,
    build_api_reranker,
    cross_encoder_available,
    is_cohere_url,
)

from ..config import app_config
from ..db import ModelConfig, Repository, SessionLocal
from .settings_service import settings_service

logger = logging.getLogger(__name__)


def _sig(cfg: ModelConfig | None) -> str:
    return f"{cfg.id}:{cfg.updated_at.isoformat() if cfg.updated_at else ''}" if cfg else "none"


class ModelService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._embedder: tuple[str, BaseEmbedder, dict] | None = None
        self._llm: dict[str, tuple[str, BaseLLM | None, dict]] = {}
        self._reranker: tuple[str, BaseReranker, dict] | None = None
        self._image_embedder: tuple[str, BaseImageEmbedder, dict] | None = None

    def invalidate(self) -> None:
        with self._lock:
            self._embedder = None
            self._image_embedder = None
            self._llm = {}
            self._reranker = None

    def _active(self, category: str) -> ModelConfig | None:
        with SessionLocal() as db:
            cfg = Repository(db).get_active_model_config(category)
            if cfg:
                db.expunge(cfg)
            return cfg

    # --- Embedding ------------------------------------------------------------------
    def embedder(self) -> BaseEmbedder:
        return self.embedder_info()[0]

    def embedder_info(self) -> tuple[BaseEmbedder, dict]:
        cfg = self._active("embedding")
        key = _sig(cfg) + ":" + settings_service.get().embedding.model_dump_json()
        with self._lock:
            if self._embedder and self._embedder[0] == key:
                return self._embedder[1], self._embedder[2]
            emb, info = self._build_embedder(cfg)
            self._embedder = (key, emb, info)
            return emb, info

    def _build_embedder(self, cfg: ModelConfig | None) -> tuple[BaseEmbedder, dict]:
        base = settings_service.get().embedding
        if cfg is not None:
            try:
                if cfg.provider_type == "local":
                    emb = SentenceTransformerEmbedder(cfg.model_name, batch_size=base.batch_size, normalize=True, device=base.device)
                    return emb, {"source": "config", "provider": cfg.provider_name, "model": cfg.model_name, "offline": False}
                emb = OpenAICompatibleEmbedder(cfg.model_name, base_url=cfg.base_url, api_key=cfg.api_key, batch_size=base.batch_size, normalize=True)
                return emb, {"source": "config", "provider": cfg.provider_name, "model": cfg.model_name, "offline": False}
            except Exception as exc:
                logger.warning("激活的 embedding 配置不可用（%s），回退", exc)
        if app_config.openai_api_key:
            emb = OpenAICompatibleEmbedder(app_config.embedding_model, base_url=app_config.openai_api_base, api_key=app_config.openai_api_key, batch_size=base.batch_size)
            return emb, {"source": "env", "provider": "OpenAI-compatible (.env)", "model": app_config.embedding_model, "offline": False}
        if base.provider == "sentence_transformers":
            try:
                emb = SentenceTransformerEmbedder(base.model, batch_size=base.batch_size, normalize=base.normalize, query_instruction=base.query_instruction, device=base.device)
                return emb, {"source": "config.yaml", "provider": "sentence-transformers", "model": base.model, "offline": False}
            except Exception as exc:
                logger.info("本地向量模型不可用（%s），使用离线哈希向量", exc)
        if base.provider == "openai_compatible" and base.base_url:
            try:
                emb = OpenAICompatibleEmbedder(base.model, base_url=base.base_url, api_key=base.api_key, batch_size=base.batch_size)
                return emb, {"source": "config.yaml", "provider": "openai_compatible", "model": base.model, "offline": False}
            except Exception as exc:
                logger.warning("config.yaml 的 openai_compatible embedding 不可用：%s", exc)
        return HashEmbedder(dim=512), {"source": "fallback", "provider": "hash", "model": "hash-512", "offline": True}

    # --- 图文向量（CLIP 类双塔模型）-----------------------------------------------------
    def _image_key(self, cfg: ModelConfig | None) -> str:
        return _sig(cfg) + ":" + settings_service.get().multimodal.embedding.model_dump_json()

    def image_embedder(self) -> BaseImageEmbedder:
        """真正加载模型（可能下载权重）。只在图片入库 / 图片检索时调用；状态页用 image_embedder_status()。"""
        cfg = self._active("image_embedding")
        key = self._image_key(cfg)
        with self._lock:
            if self._image_embedder and self._image_embedder[0] == key:
                return self._image_embedder[1]
            emb, info = self._build_image_embedder(cfg)
            self._image_embedder = (key, emb, info)
            return emb

    def _build_image_embedder(self, cfg: ModelConfig | None) -> tuple[BaseImageEmbedder, dict]:
        base = settings_service.get().multimodal.embedding
        if cfg is not None:
            try:
                if cfg.provider_type == "local":
                    emb: BaseImageEmbedder = CLIPEmbedder(cfg.model_name, device=base.device, batch_size=base.batch_size, max_text_tokens=base.max_text_tokens)
                else:
                    emb = APIImageEmbedder(cfg.model_name, base_url=cfg.base_url or "", api_key=cfg.api_key)
                return emb, {"source": "config", "provider": cfg.provider_name, "model": cfg.model_name, "offline": False, "crossModal": True, "loaded": True}
            except Exception as exc:
                logger.warning("激活的图文向量配置不可用（%s），回退", exc)
        try:
            emb = build_image_embedder(base)
        except Exception as exc:
            logger.warning("config.yaml 的图文向量模型不可用（%s），使用离线感知特征", exc)
            emb = PerceptualImageEmbedder()
        if emb.cross_modal:
            return emb, {"source": "config.yaml", "provider": emb.name, "model": emb.model_name, "offline": False, "crossModal": True, "loaded": True}
        return emb, {"source": "fallback", "provider": emb.name, "model": emb.model_name, "offline": True, "crossModal": False, "loaded": True}

    def image_embedder_status(self) -> dict:
        """不加载模型：已加载则报告实际状态，否则报告"将会使用什么"。"""
        cfg = self._active("image_embedding")
        with self._lock:
            if self._image_embedder and self._image_embedder[0] == self._image_key(cfg):
                return self._image_embedder[2]
        base = settings_service.get().multimodal.embedding
        if cfg is not None:
            return {"source": "config", "provider": cfg.provider_name, "model": cfg.model_name, "offline": False, "crossModal": True, "loaded": False}
        if base.provider == "api" or base.provider == "clip" or (base.provider == "auto" and clip_available()):
            return {"source": "config.yaml", "provider": "api" if base.provider == "api" else "clip", "model": base.model, "offline": False, "crossModal": True, "loaded": False}
        return {"source": "fallback", "provider": "perceptual", "model": "perceptual-v1", "offline": True, "crossModal": False, "loaded": False}

    # --- LLM ------------------------------------------------------------------------
    def llm(self, *, aux: bool = False, multimodal: bool = False) -> BaseLLM | None:
        return self.llm_info(aux=aux, multimodal=multimodal)[0]

    def llm_info(self, *, aux: bool = False, multimodal: bool = False) -> tuple[BaseLLM | None, dict]:
        category = "multimodal" if multimodal else "text"
        cfg = self._active(category)
        if cfg is None and multimodal:
            cfg = self._active("text")
        key = f"{category}:{aux}:{_sig(cfg)}:{settings_service.get().generation.llm.model_dump_json()}"
        with self._lock:
            cached = self._llm.get(key)
            if cached and cached[0] == key:
                return cached[1], cached[2]
            llm, info = self._build_llm(cfg, aux=aux)
            fast = self._aux_model(llm) if aux else None
            if fast:
                info = {**info, "model": fast, "mainModel": llm.model}
                llm.model = fast
            self._llm[key] = (key, llm, info)
            return llm, info

    def _aux_model(self, llm: BaseLLM | None) -> str | None:
        """generation.llm.aux_model：查询改写 / HyDE / text-to-SQL / 路由等辅助调用换成同一端点上更快的模型
        （DeepSeek 实测 deepseek-flash 比 deepseek-v4-pro 快约 2 倍，检索质量不降）。只有端点的模型列表里
        确实有它才换：换了供应商后留下的旧名字不会让辅助调用全部失败。目前只对 OpenAI 兼容端点生效。"""
        name = (settings_service.get().generation.llm.aux_model or "").strip()
        if not name or not isinstance(llm, OpenAICompatibleLLM) or name == llm.model:
            return None
        try:
            available = {m.id for m in llm.client.with_options(timeout=5.0).models.list().data}
        except Exception as exc:
            logger.warning("取不到模型列表（%s），辅助调用仍用 %s", exc, llm.model)
            return None
        if name not in available:
            logger.warning("辅助模型 %s 不在当前端点的模型列表里，辅助调用仍用 %s", name, llm.model)
            return None
        return name

    def _build_llm(self, cfg: ModelConfig | None, *, aux: bool) -> tuple[BaseLLM | None, dict]:
        base = settings_service.get().generation.llm
        if cfg is not None:
            try:
                if cfg.provider_type == "anthropic":
                    lc = base.model_copy(update={"provider": "anthropic", "model": cfg.model_name})
                    llm = AnthropicLLM(lc, api_key=cfg.api_key, base_url=cfg.base_url or None, effort=(base.aux_effort if aux else None))
                else:
                    lc = base.model_copy(update={"provider": "openai_compatible", "model": cfg.model_name, "base_url": cfg.base_url})
                    llm = OpenAICompatibleLLM(lc, api_key=cfg.api_key, base_url=cfg.base_url, aux=aux)
                return llm, {"source": "config", "provider": cfg.provider_name, "providerType": cfg.provider_type, "model": cfg.model_name, "offline": False}
            except Exception as exc:
                logger.warning("激活的 LLM 配置不可用（%s），回退", exc)
        if app_config.anthropic_api_key and base.provider in ("anthropic", "none"):
            lc = base.model_copy(update={"provider": "anthropic"})
            return AnthropicLLM(lc, api_key=app_config.anthropic_api_key, effort=(base.aux_effort if aux else None)), {"source": "env", "provider": "Anthropic (.env)", "providerType": "anthropic", "model": lc.model, "offline": False}
        if app_config.openai_api_key:
            lc = base.model_copy(update={"provider": "openai_compatible", "model": app_config.llm_model, "base_url": app_config.openai_api_base})
            return OpenAICompatibleLLM(lc, api_key=app_config.openai_api_key, base_url=app_config.openai_api_base, aux=aux), {"source": "env", "provider": "OpenAI-compatible (.env)", "providerType": "openai_compatible", "model": app_config.llm_model, "offline": False}
        if base.provider == "anthropic" and _anthropic_env_available():
            try:
                return AnthropicLLM(base, effort=(base.aux_effort if aux else None)), {"source": "config.yaml", "provider": "Anthropic", "providerType": "anthropic", "model": base.model, "offline": False}
            except Exception as exc:
                logger.info("Anthropic 凭据不可用：%s", exc)
        if base.provider == "openai_compatible" and base.base_url and (base.api_key or _env(base.api_key_env)):
            try:
                return OpenAICompatibleLLM(base, aux=aux), {"source": "config.yaml", "provider": "openai_compatible", "providerType": "openai_compatible", "model": base.model, "offline": False}
            except Exception as exc:
                logger.info("openai_compatible LLM 不可用：%s", exc)
        return None, {"source": "fallback", "provider": None, "providerType": None, "model": None, "offline": True}

    def vision_llm(self, *, aux: bool = False) -> BaseLLM | None:
        """看得了图的模型：激活的多模态配置，或文本模型本身就是 Claude（原生支持图片）。
        只配了纯文本 OpenAI 兼容模型（如 DeepSeek）时返回 None——把图片发给它只会报错。"""
        # aux=True：图片描述这类小任务用关闭思考的实例（与查询改写等一致）；看图回答用 aux=False
        if self._active("multimodal") is not None:
            return self.llm(aux=aux, multimodal=True)
        llm = self.llm(aux=aux)
        return llm if isinstance(llm, AnthropicLLM) else None

    # --- Reranker -------------------------------------------------------------------
    def reranker(self) -> BaseReranker:
        return self.reranker_info()[0]

    def reranker_info(self) -> tuple[BaseReranker, dict]:
        cfg = self._active("rerank")
        rc: RerankConfig = settings_service.get().retrieval.rerank
        llm_key = self.llm_info(aux=True)[1]
        key = f"{_sig(cfg)}:{rc.model_dump_json()}:{llm_key.get('model')}"
        with self._lock:
            if self._reranker and self._reranker[0] == key:
                return self._reranker[1], self._reranker[2]
            rr, info = self._build_reranker(cfg, rc)
            self._reranker = (key, rr, info)
            return rr, info

    def _build_reranker(self, cfg: ModelConfig | None, rc: RerankConfig) -> tuple[BaseReranker, dict]:
        """优先级：模型配置中心激活的 rerank → config.yaml 的 provider（auto：API → 本地交叉编码器 → LLM → 词法）。
        默认的本地模型是 BAAI/bge-reranker-v2-m3；权重还没就绪时由 soft（LLM / 词法）顶上，不让查询卡住。"""
        if not rc.enabled or rc.provider == "none":
            return NoopReranker(), {"source": "settings", "provider": "none", "model": None, "offline": True}
        llm = self.llm(aux=True)
        soft: BaseReranker = LLMReranker(llm, batch_size=rc.llm_batch_size) if llm is not None else LexicalReranker()
        if cfg is not None:
            try:
                if cfg.provider_type == "local":
                    rr: BaseReranker = CrossEncoderReranker(cfg.model_name, device=rc.device, fallback=soft, auto_download=rc.auto_download)
                else:
                    rr = build_api_reranker(cfg.base_url or "", cfg.api_key, cfg.model_name)   # Cohere 的地址自动走 CohereReranker
                return rr, {"source": "config", "provider": cfg.provider_name, "kind": rr.name, "model": cfg.model_name, "offline": False}
            except Exception as exc:
                logger.warning("激活的 rerank 配置不可用（%s），回退", exc)
        provider = rc.provider
        if provider == "auto":
            if rc.api_url:
                provider = "cohere" if is_cohere_url(rc.api_url) else "api"
            else:
                provider = "cross_encoder" if cross_encoder_available() else ("llm" if llm is not None else "lexical")
        try:
            if provider == "cross_encoder":
                rr = CrossEncoderReranker(rc.model, device=rc.device, fallback=soft, auto_download=rc.auto_download)
                return rr, {"source": "config.yaml", "provider": "cross_encoder", "kind": rr.name, "model": rc.model, "offline": False}
            if provider == "cohere":
                rr = CohereReranker(rc.api_key, rc.cohere_model, rc.api_url if is_cohere_url(rc.api_url) else None)
                return rr, {"source": "config.yaml", "provider": "cohere", "kind": rr.name, "model": rr.model_name, "offline": False}
            if provider == "api" and rc.api_url:
                rr = build_api_reranker(rc.api_url, rc.api_key, rc.model)
                return rr, {"source": "config.yaml", "provider": "api", "kind": rr.name, "model": rc.model, "offline": False}
            if provider == "llm" and llm is not None:
                return LLMReranker(llm, batch_size=rc.llm_batch_size), {"source": "config.yaml", "provider": "llm", "kind": "llm", "model": getattr(llm, "model", None), "offline": False}
        except Exception as exc:
            logger.warning("reranker 构建失败（%s），回退词法精排", exc)
        return LexicalReranker(), {"source": "fallback", "provider": "lexical", "kind": "lexical", "model": None, "offline": True}

    def reranker_status(self) -> dict[str, Any]:
        """运行时信息 + 本地模型的实时状态（权重是否在本机 / 是否已加载 / 是否在下载）。"""
        rr, info = self.reranker_info()
        out = dict(info)
        if isinstance(rr, CrossEncoderReranker):
            st = rr.status()
            out.update({"state": st["state"], "cached": st["cached"], "device": st["device"], "error": st["error"], "fallback": getattr(rr.fallback, "name", None)})
            if not st["cached"] and st["state"] != "ready":
                out["offline"] = True      # 还没法用本地模型：界面上按"降级"显示，实际由 fallback 精排
        return out

    def download_reranker(self) -> dict[str, Any]:
        """后台下载当前本地重排模型的权重（幂等：已在下载 / 已就绪时直接返回状态）。"""
        rr, _ = self.reranker_info()
        if not isinstance(rr, CrossEncoderReranker):
            raise ValueError("当前生效的重排不是本地模型，无需下载")
        if rr.state not in ("ready", "loading"):
            rr.start_background_load()
        return self.reranker_status()

    # --- 状态 -----------------------------------------------------------------------
    def warmup(self) -> None:
        """后台预热：把向量 / 重排 / 图文向量模型在启动时就加载好，而不是让重启后的第一个提问等 20 秒。
        图文向量模型（CLIP）只在确实有知识库存了图片时才加载。任何一步失败都只记日志。"""
        import time

        t0 = time.perf_counter()
        steps: list[str] = []
        for name, fn in (("embedding", self.embedder), ("rerank", self.reranker)):
            try:
                fn()
                steps.append(name)
            except Exception as exc:
                logger.warning("预热 %s 失败：%s", name, exc)
        try:
            from ..config import app_config

            if any(app_config.index_root.glob("*/image_index/images.jsonl")):
                emb = self.image_embedder()
                if emb.cross_modal:
                    emb.embed_texts(["预热"])   # 触发一次推理，完成底层初始化
                steps.append("image_embedding")
        except Exception as exc:
            logger.warning("预热图文向量模型失败：%s", exc)
        logger.info("模型预热完成（%s），%.1fs", ", ".join(steps) or "无", time.perf_counter() - t0)

    def status(self) -> dict[str, Any]:
        _, emb = self.embedder_info()
        _, llm = self.llm_info()
        _, mm = self.llm_info(multimodal=True)
        rr = self.reranker_status()
        s = settings_service.get()
        return {
            "embedding": emb,
            "llm": llm,
            "multimodal": mm,
            "rerank": rr,
            "imageEmbedding": self.image_embedder_status(),
            "vision": self.vision_llm() is not None,
            "ocr": available_ocr(s.multimodal.ocr),
            "multimodalEnabled": s.multimodal.enabled,
            "mineruAvailable": mineru_available(),
            "mineruEnabled": s.parsing.mineru.enabled,
            "mineruTier": s.parsing.mineru.tier,
            "popoEnabled": s.parsing.popo.enabled,
            "crossEncoderAvailable": cross_encoder_available(),
            "vectorBackend": s.vector_store.backend,
            "offlineMode": bool(emb.get("offline")) and llm.get("offline", True),
        }


def _env(name: str | None) -> str | None:
    import os

    return os.environ.get(name) if name else None


def _anthropic_env_available() -> bool:
    import os

    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


model_service = ModelService()
