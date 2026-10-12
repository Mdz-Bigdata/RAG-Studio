from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from app.config import settings

class QdrantRepository:
    """
    Qdrant 向量仓储类，支持内存模式、本地存储模式和服务器模式
    """
    def __init__(self):
        # 根据配置初始化客户端
        if settings.QDRANT_URL:
            # 远程服务器模式
            self.client = QdrantClient(url=settings.QDRANT_URL)
        elif settings.QDRANT_PATH:
            # 本地持久化文件模式
            self.client = QdrantClient(path=settings.QDRANT_PATH)
        else:
            # 内存模式
            self.client = QdrantClient(":memory:")
            
    def init_collection(self, collection_name: str, vector_size: int = 1536) -> bool:
        """
        初始化 Collection，如果不存在则创建。
        默认向量维度为 1536 (OpenAI text-embedding-3-small)，距离度量使用 Cosine。
        """
        try:
            if not self.client.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=vector_size,
                        distance=rest_models.Distance.COSINE
                    )
                )
            return True
        except Exception as e:
            # TODO: 接入 logging
            print(f"Error initializing Qdrant collection: {e}")
            return False
            
    def upsert_vectors(self, collection_name: str, points: list) -> bool:
        """
        向指定的 Collection 批量插入向量数据点
        """
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            return True
        except Exception as e:
            print(f"Error upserting vectors to Qdrant: {e}")
            return False
            
    def search_vectors(self, collection_name: str, query_vector: list[float], limit: int = 10) -> list:
        """
        在 Collection 中进行密集向量相似度检索
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return results
        except Exception as e:
            print(f"Error searching vectors in Qdrant: {e}")
            return []
            
    def delete_vectors(self, collection_name: str, point_ids: list[str]) -> bool:
        """
        根据 ID 列表批量删除向量数据点
        """
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=rest_models.PointIdsList(
                    points=point_ids
                )
            )
            return True
        except Exception as e:
            print(f"Error deleting vectors from Qdrant: {e}")
            return False
            
# 实例化全局单例
qdrant_repository = QdrantRepository()
