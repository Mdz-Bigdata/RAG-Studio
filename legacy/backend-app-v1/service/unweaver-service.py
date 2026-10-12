import re
from sqlalchemy.orm import Session
from app.model import EntityRelationModel, ChunkModel

class UnWeaverService:
    """
    UnWeaver 实体关系织布机服务。
    用于在文档切片中挖掘专有名词、实体定义，并在混合检索后，将这些实体关系的上下游 Chunk 进行动态“编织”扩展。
    """
    
    def __init__(self):
        # 常见定义匹配正则，例如: **RAG**是指检索增强生成... 或者 向量数据库是...
        self.define_patterns = [
            re.compile(r"\*\*([^\*]{2,20})\*\*[:：\s]*(?:是|是指|定义为|指|为)([^。\n]+)"),
            re.compile(r"([a-zA-Z0-9\u4e00-\u9fa5\s_]{2,20})\s+(?:is|defined as)\s+([^.\n]+)"),
        ]

    def extract_and_store_entities(self, db: Session, kb_id: str, chunk_id: str, chunk_text: str) -> int:
        """
        从切片文本中自动提取专有名词及释义，存储至 SQLite 中形成关系图
        """
        count = 0
        relations = []
        
        # 使用正则提取
        for pattern in self.define_patterns:
            matches = pattern.findall(chunk_text)
            for entity, definition in matches:
                entity = entity.strip()
                definition = definition.strip()
                if len(entity) > 1 and len(definition) > 5:
                    relation = EntityRelationModel(
                        entityName=entity,
                        relationType="DEFINED_IN",
                        definition=definition[:500],  # 截断以保证长度合理
                        chunkId=chunk_id,
                        knowledgeBaseId=kb_id
                    )
                    relations.append(relation)
                    count += 1
                    
        if relations:
            # 批量写入
            db.add_all(relations)
            db.commit()
            
        return count

    def weave_context(self, db: Session, kb_id: str, retrieved_chunks: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        对已检索到的 Chunks 进行实体编织。
        如果 A chunk 中提到了实体 E，但定义 E 的 B chunk 没被召回，UnWeaver 会自动拉取 B chunk 并将其作为“编织节点”并入结果中。
        
        返回: (最终扩展后的 Chunks 列表, 编织出的实体图关系列表)
        """
        if not retrieved_chunks:
            return [], []
            
        # 记录已召回的 chunk ID
        retrieved_ids = {c["id"] for c in retrieved_chunks}
        
        # 拼接全部文本，用于实体匹配
        combined_text = "\n".join([c["content"] for c in retrieved_chunks])
        
        # 获取该知识库下的所有实体定义
        all_entities = db.query(EntityRelationModel).filter(
            EntityRelationModel.knowledgeBaseId == kb_id,
            EntityRelationModel.relationType == "DEFINED_IN"
        ).all()
        
        weaved_chunks = []
        weaved_relations = []
        weaved_entity_ids = set()
        
        for entity_rel in all_entities:
            # 如果已召回 Chunks 中提到了该实体，但该实体所在的原始 chunk 并没有被召回
            # (避免重复拉取)
            if entity_rel.entityName in combined_text and entity_rel.chunkId not in retrieved_ids:
                if entity_rel.entityName not in weaved_entity_ids:
                    # 查找这个定义关联的原始 Chunk
                    orig_chunk = db.query(ChunkModel).filter(ChunkModel.id == entity_rel.chunkId).first()
                    if orig_chunk:
                        weaved_chunks.append({
                            "id": orig_chunk.id,
                            "documentId": orig_chunk.documentId,
                            "knowledgeBaseId": orig_chunk.knowledgeBaseId,
                            "content": f"【UnWeaver 编织定义: {entity_rel.entityName}】\n{orig_chunk.content}",
                            "chunkIndex": orig_chunk.chunkIndex,
                            "score": 0.8,  # 给定一个固定的编织权重分
                            "isWeaved": True,
                            "weavedEntity": entity_rel.entityName
                        })
                        weaved_relations.append({
                            "entity": entity_rel.entityName,
                            "definition": entity_rel.definition,
                            "sourceChunkId": orig_chunk.id,
                            "weavedFrom": entity_rel.entityName
                        })
                        weaved_entity_ids.add(entity_rel.entityName)
                        
        # 将编织出的 chunks 追加到原有召回列表后
        final_chunks = list(retrieved_chunks) + weaved_chunks
        return final_chunks, weaved_relations
