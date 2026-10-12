"""模型配置中心：供应商预设、配置 CRUD、激活、拉取模型列表、连接测试。"""

from __future__ import annotations

import time

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from prism.retrieval.rerank import BaseReranker, CrossEncoderReranker, build_api_reranker, is_cohere_url

from ..db import ModelConfig, Repository, get_db
from ..schemas import ModelConfigResponse, ModelConfigSave, ModelFetchRequest, ModelTestRequest
from ..services import index_service, model_service, reindex_service

router = APIRouter()

PROVIDERS = [
    {"name": "Anthropic", "type": "anthropic", "baseUrl": "https://api.anthropic.com", "categories": ["text", "multimodal"], "models": {"text": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"], "multimodal": ["claude-opus-5-5", "claude-sonnet-5-5"]}},
    {"name": "SiliconFlow", "type": "openai_compatible", "baseUrl": "https://api.siliconflow.cn/v1", "categories": ["text", "embedding", "multimodal", "rerank"], "models": {"text": ["Qwen/Qwen3-32B", "deepseek-ai/DeepSeek-V3"], "embedding": ["BAAI/bge-m3", "BAAI/bge-large-zh-v1.5"], "multimodal": ["Qwen/Qwen2.5-VL-72B-Instruct"], "rerank": ["BAAI/bge-reranker-v2-m3"]}},
    {"name": "DeepSeek", "type": "openai_compatible", "baseUrl": "https://api.deepseek.com/v1", "categories": ["text", "multimodal"], "models": {"text": ["deepseek-chat", "deepseek-reasoner"], "multimodal": ["deepseek-flash"]}},  # 实测 deepseek-v4-pro 收到图片显示 [Unsupported Image]，看不到图
    {"name": "OpenAI", "type": "openai_compatible", "baseUrl": "https://api.openai.com/v1", "categories": ["text", "embedding", "multimodal"], "models": {"text": ["gpt-4o-mini", "gpt-4o"], "embedding": ["text-embedding-3-small", "text-embedding-3-large"], "multimodal": ["gpt-4o"]}},
    {"name": "阿里百炼 (DashScope)", "type": "openai_compatible", "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1", "categories": ["text", "embedding", "multimodal"], "models": {"text": ["qwen3.5-plus", "qwen-plus", "qwen-max"], "embedding": ["text-embedding-v3"], "multimodal": ["qwen-vl-max", "qwen-vl-plus"]}},
    {"name": "智谱 AI (GLM)", "type": "openai_compatible", "baseUrl": "https://open.bigmodel.cn/api/paas/v4", "categories": ["text", "embedding", "multimodal"], "models": {"text": ["glm-4-plus", "glm-4-flash"], "embedding": ["embedding-3"], "multimodal": ["glm-4v"]}},
    {"name": "月之暗面 (Kimi)", "type": "openai_compatible", "baseUrl": "https://api.moonshot.cn/v1", "categories": ["text"], "models": {"text": ["moonshot-v1-8k", "moonshot-v1-32k"]}},
    {"name": "火山引擎 (Ark)", "type": "openai_compatible", "baseUrl": "https://ark.cn-beijing.volces.com/api/v3", "categories": ["text", "multimodal"], "models": {"text": ["doubao-pro-32k"], "multimodal": ["doubao-vision-pro-32k"]}},
    {"name": "Gemini", "type": "openai_compatible", "baseUrl": "https://generativelanguage.googleapis.com/v1beta/openai", "categories": ["text", "multimodal", "embedding"], "models": {"text": ["gemini-2.0-flash"], "multimodal": ["gemini-2.0-flash"], "embedding": ["text-embedding-004"]}},
    {"name": "Ollama (本地)", "type": "openai_compatible", "baseUrl": "http://localhost:11434/v1", "categories": ["text", "embedding", "multimodal"], "models": {"text": ["qwen3:8b", "llama3.1"], "embedding": ["bge-m3", "nomic-embed-text"], "multimodal": ["llava"]}},
    {"name": "vLLM / 自托管", "type": "openai_compatible", "baseUrl": "http://localhost:8001/v1", "categories": ["text", "embedding", "multimodal"], "models": {"text": ["Qwen3-VL-4B-Instruct"], "embedding": ["BAAI/bge-m3"], "multimodal": ["Qwen3-VL-4B-Instruct"]}},
    {"name": "Cohere Rerank", "type": "api", "baseUrl": "https://api.cohere.com/v2", "categories": ["rerank"], "models": {"rerank": ["rerank-v4.0-pro", "rerank-v4.0-fast", "rerank-v3.5", "rerank-multilingual-v3.0", "rerank-english-v3.0"]}},
    {"name": "Jina Rerank API", "type": "api", "baseUrl": "https://api.jina.ai/v1", "categories": ["rerank"], "models": {"rerank": ["jina-reranker-v3", "jina-reranker-v2-base-multilingual", "jina-reranker-m0", "jina-colbert-v2"]}},
    {"name": "本地 sentence-transformers", "type": "local", "baseUrl": None, "categories": ["embedding", "rerank"], "models": {"embedding": ["BAAI/bge-m3", "BAAI/bge-large-zh-v1.5"], "rerank": ["BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-base"]}},
    {"name": "本地 CLIP（transformers）", "type": "local", "baseUrl": None, "categories": ["image_embedding"], "models": {"image_embedding": ["OFA-Sys/chinese-clip-vit-base-patch16", "OFA-Sys/chinese-clip-vit-large-patch14", "openai/clip-vit-base-patch32"]}},
    {"name": "Jina 多模态向量 API", "type": "api", "baseUrl": "https://api.jina.ai/v1", "categories": ["image_embedding"], "models": {"image_embedding": ["jina-clip-v2"]}},
    {"name": "自定义", "type": "openai_compatible", "baseUrl": "", "categories": ["text", "embedding", "multimodal", "rerank"], "models": {}},
]


def _mask(key: str | None) -> str:
    if not key:
        return ""
    return key[:4] + "********" + key[-4:] if len(key) > 8 else "********"


def _resp(c: ModelConfig) -> ModelConfigResponse:
    return ModelConfigResponse(id=c.id, providerName=c.provider_name, providerType=c.provider_type, baseUrl=c.base_url, apiKeyMasked=_mask(c.api_key), category=c.category, modelName=c.model_name, isActive=c.is_active or 0, createdAt=c.created_at, updatedAt=c.updated_at)


@router.get("/providers")
def providers():
    return PROVIDERS


@router.get("/configs", response_model=list[ModelConfigResponse])
def list_configs(db: Session = Depends(get_db)):
    return [_resp(c) for c in Repository(db).list_model_configs()]


@router.post("/configs", response_model=ModelConfigResponse)
def save_config(body: ModelConfigSave, db: Session = Depends(get_db)):
    repo = Repository(db)
    api_key = None if (body.apiKey and "********" in body.apiKey) else body.apiKey
    if not api_key and body.keyFromConfigId:
        donor = repo.get_model_config(body.keyFromConfigId)
        api_key = donor.api_key if donor else None
    cfg = repo.upsert_model_config(body.id, provider_name=body.providerName, provider_type=body.providerType, base_url=body.baseUrl, api_key=api_key, category=body.category, model_name=body.modelName)
    if cfg.is_active:
        model_service.invalidate()
        index_service.invalidate()
        reindex_service.request("模型配置变更")
    return _resp(cfg)


@router.post("/configs/{config_id}/activate")
def activate(config_id: str, db: Session = Depends(get_db)):
    repo = Repository(db)
    cfg = repo.get_model_config(config_id)
    if not cfg:
        raise HTTPException(404, "配置不存在")
    repo.activate_model_config(cfg)
    model_service.invalidate()
    index_service.invalidate()
    reindex_service.request("模型配置变更")
    return {"message": f"已激活 {cfg.model_name}", "category": cfg.category}


@router.post("/configs/{config_id}/deactivate")
def deactivate(config_id: str, db: Session = Depends(get_db)):
    repo = Repository(db)
    cfg = repo.get_model_config(config_id)
    if not cfg:
        raise HTTPException(404, "配置不存在")
    repo.deactivate_model_config(cfg)
    model_service.invalidate()
    index_service.invalidate()
    reindex_service.request("模型配置变更")
    return {"message": f"已停用 {cfg.model_name}"}


@router.delete("/configs/{config_id}")
def delete_config(config_id: str, db: Session = Depends(get_db)):
    repo = Repository(db)
    cfg = repo.get_model_config(config_id)
    if not cfg:
        raise HTTPException(404, "配置不存在")
    was_active = bool(cfg.is_active)
    repo.delete_model_config(cfg)
    if was_active:
        model_service.invalidate()
        index_service.invalidate()
        reindex_service.request("模型配置变更")
    return {"message": "配置已删除"}


def _categorize(model_ids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"text": [], "embedding": [], "multimodal": [], "rerank": [], "image_embedding": []}
    for mid in model_ids:
        low = mid.lower()
        if "rerank" in low or "colbert" in low:
            out["rerank"].append(mid)
        elif "clip" in low or "siglip" in low:
            out["image_embedding"].append(mid)
        elif any(x in low for x in ("embed", "bge-", "gte-", "/gte", "e5-", "m3e", "text2vec", "conan-")):
            out["embedding"].append(mid)
        elif any(x in low for x in ("vl", "vision", "multimodal", "gpt-4o", "claude", "llava", "4v", "gemini", "omni")):
            out["multimodal"].append(mid)
            out["text"].append(mid)
        else:
            out["text"].append(mid)
    return out


_WEIGHTS = (".safetensors", ".bin", ".pt", ".onnx")


def _local_models() -> dict:
    """扫描本机 HuggingFace 缓存，列出能直接加载的本地模型（不联网），再并上本地预设。

    归类依据：带 config_sentence_transformers.json / modules.json → 文本嵌入；名字含 rerank → 重排；
    含 clip / siglip → 图文向量。没有权重文件的目录（下载中断）不算"已在本机"。
    """
    from pathlib import Path

    try:
        from huggingface_hub import constants

        root = Path(constants.HF_HUB_CACHE)
    except Exception:
        root = Path.home() / ".cache" / "huggingface" / "hub"
    models = _categorize([])
    cached: list[str] = []
    for d in sorted(root.glob("models--*")) if root.is_dir() else []:
        files: set[str] = set()
        for snap in d.glob("snapshots/*"):
            files |= {f.name for f in snap.iterdir()}
        if not any(f.endswith(_WEIGHTS) for f in files):
            continue
        mid = d.name[len("models--"):].replace("--", "/")
        low = mid.lower()
        if "rerank" in low:
            models["rerank"].append(mid)
        elif "clip" in low or "siglip" in low:
            models["image_embedding"].append(mid)
        elif {"config_sentence_transformers.json", "modules.json"} & files or "embed" in low or "bge-" in low:
            models["embedding"].append(mid)
        else:
            continue
        cached.append(mid)
    for p in PROVIDERS:
        if p["type"] == "local":
            for cat, names in p["models"].items():
                models[cat] = list(dict.fromkeys(models[cat] + names))
    return {"source": "local", "total": len(cached), "models": models, "cached": cached}


@router.post("/fetch-list")
def fetch_list(body: ModelFetchRequest, db: Session = Depends(get_db)):
    """用 Base URL + API Key 拉取服务商的全部可用模型并按用途归类。

    - Anthropic：GET /v1/models（x-api-key）
    - 其余（OpenAI 兼容 / Rerank API）：GET {baseUrl}/models（Bearer）
    - 本地（sentence-transformers / CLIP）：扫描本机 HuggingFace 缓存，已下载的排在前面并标记 cached
    服务商没有模型列表接口或请求失败时，返回该供应商的预设列表并附带 error，前端据此提示。
    """
    if body.providerType == "local":
        return _local_models()
    if not body.apiKey and body.keyFromConfigId:
        donor = Repository(db).get_model_config(body.keyFromConfigId)
        if donor and donor.api_key:
            body = body.model_copy(update={"apiKey": donor.api_key})
    # 同一 Base URL 可能对应多个预设（如 Jina 的重排与多模态向量），合并它们的模型表
    matched = [p for p in PROVIDERS if (p["type"] == "anthropic" == body.providerType) or (p["baseUrl"] and body.baseUrl and body.providerType != "anthropic" and body.baseUrl.rstrip("/").startswith(p["baseUrl"].rstrip("/")))]
    preset: dict = {"models": {}}
    for p in matched:
        for cat, names in p["models"].items():
            preset["models"][cat] = list(dict.fromkeys(preset["models"].get(cat, []) + names))
    empty = _categorize([])
    try:
        if body.providerType == "anthropic":
            base = (body.baseUrl or "https://api.anthropic.com").rstrip("/")
            url = base + ("/models" if base.endswith("/v1") else "/v1/models")
            r = httpx.get(url, params={"limit": 1000}, headers={"x-api-key": body.apiKey or "", "anthropic-version": "2023-06-01"}, timeout=10.0)
        elif is_cohere_url(body.baseUrl):
            # Cohere：GET /v1/models?endpoint=rerank → {"models": [{"name": ...}]}（重排接口本身在 /v2/rerank）
            r = httpx.get("https://api.cohere.com/v1/models", params={"endpoint": "rerank", "page_size": 200}, headers={"Authorization": f"Bearer {body.apiKey or ''}"}, timeout=10.0)
        else:
            headers = {"Authorization": f"Bearer {body.apiKey}"} if body.apiKey else {}
            r = httpx.get(body.baseUrl.rstrip("/") + "/models", headers=headers, timeout=10.0)
        r.raise_for_status()
        payload = r.json()
        items = (payload.get("data") or payload.get("models")) if isinstance(payload, dict) else payload
        ids = sorted({str(m.get("id") or m.get("name") or "") for m in (items or []) if isinstance(m, dict)} - {""})
        if "api.jina.ai" in (body.baseUrl or ""):
            # Jina 的列表带组织前缀（jina-ai/jina-reranker-v3），而调用接口时用不带前缀的名字
            ids = sorted({i.split("/", 1)[1] if i.startswith("jina-ai/") else i for i in ids})
        if not ids:
            raise ValueError("服务商返回了空的模型列表")
        models = _categorize(ids)
        if body.providerType == "anthropic":
            models["multimodal"] = list(models["text"])
        elif not models["multimodal"] and any("multimodal" in p["categories"] for p in matched):
            # 预设声明支持多模态、但模型名里看不出 vision / vl 的服务商（如 DeepSeek）：文本模型即多模态候选
            models["multimodal"] = list(models["text"])
        # 供应商只做某一类用途（如 Jina Rerank）时，把预设里的模型补进去，避免分类规则漏掉
        for cat, names in ((preset or {}).get("models") or {}).items():
            models.setdefault(cat, [])
            models[cat] = list(dict.fromkeys(models[cat] + [n for n in names if n in ids]))
        return {"source": "api", "total": len(ids), "models": models}
    except Exception as exc:
        msg = f"HTTP {exc.response.status_code}" if isinstance(exc, httpx.HTTPStatusError) else str(exc)
        return {"source": "presets", "error": msg[:300], "total": 0, "models": {**empty, **((preset or {}).get("models") or {})}}


@router.post("/test-connection")
def test_connection(body: ModelTestRequest, db: Session = Depends(get_db)):
    start = time.time()
    api_key, base_url, ptype = body.apiKey, body.baseUrl, body.providerType
    if body.configId:
        saved = Repository(db).get_model_config(body.configId)
        if saved:
            api_key = saved.api_key if (not api_key or "********" in api_key) else api_key
            base_url = base_url or saved.base_url
            ptype = saved.provider_type
    try:
        if ptype == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=api_key, base_url=base_url or None)
            client.messages.create(model=body.modelName, max_tokens=8, messages=[{"role": "user", "content": "ping"}])
        elif body.category == "image_embedding":
            from PIL import Image

            from prism.multimodal import APIImageEmbedder, CLIPEmbedder, clip_available

            if ptype == "local":
                if not clip_available():
                    raise RuntimeError("未安装 torch + transformers：pip install 'rag-agent-studio[multimodal]'")
                emb = CLIPEmbedder(body.modelName)
            else:
                emb = APIImageEmbedder(body.modelName, base_url=base_url or "", api_key=api_key)
            sim = float(emb.embed_text("一张纯红色的图片") @ emb.embed_image(Image.new("RGB", (64, 64), (220, 30, 30))))
            return {"success": True, "latency": int((time.time() - start) * 1000), "message": f"连接成功（图文相似度自检 {sim:.3f}，dim={emb.dim}）"}
        elif ptype == "local":
            if body.category == "embedding":
                from prism.index.embeddings import SentenceTransformerEmbedder

                SentenceTransformerEmbedder(body.modelName).embed_documents(["ping"])
            else:
                # 本地重排直接用 transformers 加载（不需要 sentence-transformers）；权重不在本机时这里会同步下载
                return {"success": True, "latency": int((time.time() - start) * 1000), "message": _rerank_selfcheck(CrossEncoderReranker(body.modelName))}
        elif body.category == "rerank" or ptype == "api":
            return {"success": True, "latency": int((time.time() - start) * 1000), "message": _rerank_selfcheck(build_api_reranker(base_url or "", api_key, body.modelName))}
        else:
            from openai import OpenAI

            client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url or None, timeout=15.0)
            if body.category == "embedding":
                client.embeddings.create(model=body.modelName, input=["ping"])
            else:
                client.chat.completions.create(model=body.modelName, messages=[{"role": "user", "content": "ping"}], max_tokens=5)
        return {"success": True, "latency": int((time.time() - start) * 1000), "message": "连接成功"}
    except Exception as exc:
        return {"success": False, "latency": int((time.time() - start) * 1000), "message": f"连接失败：{exc}"[:500]}


def _rerank_selfcheck(reranker: BaseReranker) -> str:
    """用一相关一无关两条文档打分：既验证连通，也验证返回的分数方向正确。"""
    ranked = dict(reranker.rerank("年假可以跨年使用吗", [("hit", "当年未休完的年假可以跨年使用，最晚在次年 3 月 31 日前休完。"), ("miss", "K8s Ingress 返回 502 时先检查 upstream 健康检查。")], 2))
    hit, miss = ranked.get("hit", 0.0), ranked.get("miss", 0.0)
    verdict = "排序正确" if hit > miss else "排序异常：无关文档得分更高"
    return f"连接成功（相关 {hit:.3f} / 无关 {miss:.3f}，{verdict}）"


@router.get("/reranker/status")
def reranker_status():
    """当前生效的重排：provider / 模型；本地模型另有 cached（权重是否在本机）与 state（idle / loading / ready / failed）。"""
    return model_service.reranker_status()


@router.post("/reranker/download")
def reranker_download():
    """后台下载当前本地重排模型的权重（BAAI/bge-reranker-v2-m3 约 2.3GB）。重复调用是安全的。"""
    try:
        return model_service.download_reranker()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
