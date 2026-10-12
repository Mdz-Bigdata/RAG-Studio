import hashlib
import numpy as np
from sqlalchemy.orm import Session
from openai import OpenAI
from app.config import settings

class EmbeddingService:
    """
    向量化嵌入服务。支持从数据库动态读取活跃的嵌入模型配置，并缓存客户端实例。
    """
    def __init__(self):
        self.dimension = 1536  # 默认维度
        self.clients_cache = {}  # 缓存 config_id -> client

    def _get_client_and_model(self, db: Session = None) -> tuple[OpenAI | None, str]:
        """
        动态获取大模型客户端和模型名称。
        优先从 SQLite 读取 active 状态的嵌入模型配置，并进行缓存复用。
        """
        if db:
            try:
                from app.repository import SQLiteRepository
                repo = SQLiteRepository(db)
                active_config = repo.get_active_model_config("embedding")
                if active_config:
                    cid = active_config.id
                    # 缓存客户端以防频繁实例化
                    if cid not in self.clients_cache:
                        # 有 Key 则实例化，否则视为 Mock 模式
                        if active_config.apiKey:
                            self.clients_cache[cid] = OpenAI(
                                api_key=active_config.apiKey,
                                base_url=active_config.baseUrl
                            )
                        else:
                            self.clients_cache[cid] = None
                    return self.clients_cache[cid], active_config.modelName
            except Exception as e:
                print(f"Error querying active embedding config: {e}")

        # 降级到环境变量配置
        if settings.OPENAI_API_KEY:
            if "default" not in self.clients_cache:
                self.clients_cache["default"] = OpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    base_url=settings.OPENAI_API_BASE
                )
            return self.clients_cache["default"], settings.EMBEDDING_MODEL

        return None, settings.EMBEDDING_MODEL

    def get_embedding(self, text: str, db: Session = None) -> list[float]:
        """
        获取单条文本的向量表示 (支持动态 DB 模型)
        """
        client, model_name = self._get_client_and_model(db)
        
        if client:
            try:
                response = client.embeddings.create(
                    input=[text],
                    model=model_name
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Embedding API failed: {e}. Falling back to deterministic Mock Embedding.")
                return self._generate_mock_vector(text)
        else:
            return self._generate_mock_vector(text)

    def get_embeddings_bulk(self, texts: list[str], db: Session = None) -> list[list[float]]:
        """
        批量获取文本的向量表示 (支持动态 DB 模型)
        """
        if not texts:
            return []
            
        client, model_name = self._get_client_and_model(db)
        
        if client:
            try:
                response = client.embeddings.create(
                    input=texts,
                    model=model_name
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                print(f"Bulk Embedding API failed: {e}. Falling back to deterministic Mock Embeddings.")
                return [self._generate_mock_vector(t) for t in texts]
        else:
            return [self._generate_mock_vector(t) for t in texts]

    def _generate_mock_vector(self, text: str) -> list[float]:
        """
        确定性离线 Mock 向量生成器。
        使用文本 MD5 散列值作为随机数种子，对于相同文本产出完全相同的 1536 维单位向量。
        """
        seed_num = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed_num)
        
        vector = rng.standard_normal(self.dimension)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector.tolist()
