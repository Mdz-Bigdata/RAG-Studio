import time
import uuid
import requests
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from openai import OpenAI
from app.repository import get_db, SQLiteRepository
from app.model import ModelConfigModel
from app.schema import ModelConfigSaveRequest, ModelConfigResponse, ModelFetchRequest, ModelTestConnectionRequest

router = APIRouter()

# 服务商模型预设兜底 (当接口调用失败或处于离线本地测试时使用)
PRESET_MODELS = {
    "text": [
        "gpt-4o-mini", "gpt-4o", "deepseek-chat", "deepseek-reasoner",
        "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp",
        "qwen-turbo", "qwen-max", "qwen-plus", "glm-4", "glm-4-flash",
        "doubao-pro-4k", "doubao-pro-32k", "doubao-lite-4k",
        "moonshot-v1-8k", "moonshot-v1-32k", "llama3", "llama-3.1-8b-instruct"
    ],
    "embedding": [
        "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002",
        "text-embedding-v2", "bge-large-zh-v1.5", "bge-m3", "bge-small-zh-v1.5", "bge-large-en-v1.5"
    ],
    "multimodal": [
        "gpt-4o", "gpt-4o-mini", "gemini-1.5-flash", "gemini-1.5-pro", 
        "qwen-vl-max", "qwen-vl-plus", "glm-4v", "claude-3-5-sonnet"
    ],
    "rerank": [
        "jina-rerank-v2-base-multilingual", "bge-reranker-large", "bge-reranker-base", "cohere-rerank-v3"
    ]
}

@router.get("/configs", response_model=list[ModelConfigResponse])
def list_model_configs(db: Session = Depends(get_db)):
    """
    列出所有已保存的模型配置 (脱敏)
    """
    repo = SQLiteRepository(db)
    configs = repo.list_model_configs()
    # Pydantic 自动处理，且我们在 Response 实体里不会泄漏明文的 API Key (虽然我们这里库内会存)
    # 为了彻底安全，我们可以在 repository / api 层把 apiKey 用星号脱敏
    out = []
    for c in configs:
        masked_key = ""
        if c.apiKey:
            masked_key = c.apiKey[:4] + "********" + c.apiKey[-4:] if len(c.apiKey) > 8 else "********"
        out.append(
            ModelConfigResponse(
                id=c.id,
                providerName=c.providerName,
                baseUrl=c.baseUrl,
                category=c.category,
                modelName=c.modelName,
                isActive=c.isActive,
                createdAt=c.createdAt,
                updatedAt=c.updatedAt
            )
        )
    return out

@router.post("/configs", response_model=ModelConfigResponse)
def save_model_config(req: ModelConfigSaveRequest, db: Session = Depends(get_db)):
    """
    创建或更新大模型配置
    """
    repo = SQLiteRepository(db)
    
    config_id = req.id if req.id else str(uuid.uuid4())
    
    # 如果是更新，且 apiKey 传空或为脱敏星号，我们沿用旧的 Key
    api_key_to_save = req.apiKey
    if req.id:
        old_config = repo.get_model_config(req.id)
        if old_config and (not req.apiKey or "********" in req.apiKey):
            api_key_to_save = old_config.apiKey

    config = ModelConfigModel(
        id=config_id,
        providerName=req.providerName,
        baseUrl=req.baseUrl,
        apiKey=api_key_to_save,
        category=req.category,
        modelName=req.modelName,
        isActive=0  # 保存时默认不激活，让用户显式触发激活或者保留原有 isActive 状态
    )
    
    # 继承原有的激活状态
    if req.id:
        old_config = repo.get_model_config(req.id)
        if old_config:
            config.isActive = old_config.isActive
            
    saved = repo.create_or_update_model_config(config)
    return saved

@router.post("/configs/{config_id}/activate", response_model=dict)
def activate_model_config(config_id: str, db: Session = Depends(get_db)):
    """
    激活特定模型配置
    """
    repo = SQLiteRepository(db)
    config = repo.get_model_config(config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model configuration not found"
        )
        
    success = repo.activate_model_config(config_id, config.category)
    if not success:
         raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to activate model configuration"
        )
         
    return {"message": f"Model configuration {config.modelName} activated successfully"}

@router.delete("/configs/{config_id}", response_model=dict)
def delete_model_config(config_id: str, db: Session = Depends(get_db)):
    """
    删除模型配置
    """
    repo = SQLiteRepository(db)
    success = repo.delete_model_config(config_id)
    if not success:
         raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model configuration not found"
        )
    return {"message": "Model configuration deleted successfully"}

@router.post("/fetch-list")
def fetch_provider_models(req: ModelFetchRequest):
    """
    接收 baseUrl 和 apiKey，尝试通过 `/models` 接口自动提取该服务商下的可用模型列表。
    自动按照文本、向量嵌入、多模态、重排序分类。
    """
    # 默认分类容器
    categorized = {
        "text": [],
        "embedding": [],
        "multimodal": [],
        "rerank": []
    }
    
    headers = {}
    if req.apiKey:
        headers["Authorization"] = f"Bearer {req.apiKey}"
        
    # 去除斜杠结尾
    base_url = req.baseUrl.rstrip("/")
    models_url = f"{base_url}/models"
    
    try:
        response = requests.get(models_url, headers=headers, timeout=6)
        if response.status_code == 200:
            data = response.json()
            models_data = data.get("data", [])
            
            for m in models_data:
                model_id = m.get("id")
                if not model_id:
                    continue
                    
                model_id_lower = model_id.lower()
                
                # 智能归类逻辑
                if "embedding" in model_id_lower:
                    categorized["embedding"].append(model_id)
                elif "rerank" in model_id_lower:
                    categorized["rerank"].append(model_id)
                elif any(x in model_id_lower for x in ["vl", "vision", "multimodal", "gpt-4o", "claude"]):
                    categorized["multimodal"].append(model_id)
                    # 多模态通常也可以作为文本生成
                    categorized["text"].append(model_id)
                else:
                    # 默认归为文本模型
                    categorized["text"].append(model_id)
                    
            # 如果某分类为空，依然拉入预设兜底
            for cat in categorized:
                if not categorized[cat]:
                    categorized[cat] = PRESET_MODELS[cat]
            
            return {
                "source": "api",
                "models": categorized
            }
    except Exception as e:
        print(f"Failed to fetch models from API: {e}. Falling back to presets.")
        
    # 当 API 不通或超时，直接返回预设列表
    return {
        "source": "presets",
        "models": PRESET_MODELS
    }

@router.post("/test-connection")
def test_model_connection(req: ModelTestConnectionRequest):
    """
    测试配置连接的可用性与接口响应延迟
    """
    start_time = time.time()
    
    try:
        if req.category == "text":
            client = OpenAI(api_key=req.apiKey, base_url=req.baseUrl)
            client.chat.completions.create(
                model=req.modelName,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                timeout=8
            )
        elif req.category == "embedding":
            client = OpenAI(api_key=req.apiKey, base_url=req.baseUrl)
            client.embeddings.create(
                model=req.modelName,
                input=["ping"],
                timeout=8
            )
        elif req.category == "rerank":
            # Jina / Cohere 格式重排握手
            headers = {
                "Authorization": f"Bearer {req.apiKey}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": req.modelName,
                "query": "ping",
                "documents": ["pong"],
                "top_n": 1
            }
            
            # 智能补全 rerank 路径
            rerank_url = req.baseUrl
            if not rerank_url.endswith("/rerank") and not rerank_url.endswith("/reranked"):
                # 如果 baseUrl 是 https://api.jina.ai/v1，自动加上 /rerank
                rerank_url = rerank_url.rstrip("/") + "/rerank"
                
            res = requests.post(rerank_url, headers=headers, json=payload, timeout=8)
            res.raise_for_status()
        elif req.category == "multimodal":
            # 多模态一般也支持 chat.completions 普通 ping
            client = OpenAI(api_key=req.apiKey, base_url=req.baseUrl)
            client.chat.completions.create(
                model=req.modelName,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                timeout=8
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported model category for connection test: {req.category}"
            )
            
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "success": True,
            "latency": latency_ms,
            "message": "Connection handshake successful"
        }
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "latency": latency_ms,
            "message": f"Connection failed: {str(e)}"
        }
