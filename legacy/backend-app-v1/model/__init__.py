import importlib

# 动态加载 kebab-case 的 model 模块
sqlite_model = importlib.import_module(".sqlite-model", __package__)

Base = sqlite_model.Base
KnowledgeBaseModel = sqlite_model.KnowledgeBaseModel
DocumentModel = sqlite_model.DocumentModel
ChunkModel = sqlite_model.ChunkModel
EntityRelationModel = sqlite_model.EntityRelationModel
ModelConfigModel = sqlite_model.ModelConfigModel
