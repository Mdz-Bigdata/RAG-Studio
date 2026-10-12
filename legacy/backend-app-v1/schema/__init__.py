import importlib

# 动态加载 kebab-case 的 schema 模块
kb_schema_mod = importlib.import_module(".kb-schema", __package__)
doc_schema_mod = importlib.import_module(".document-schema", __package__)
chat_schema_mod = importlib.import_module(".chat-schema", __package__)
model_schema_mod = importlib.import_module(".model-schema", __package__)

# 导出 schemas
KBCreate = kb_schema_mod.KBCreate
KBResponse = kb_schema_mod.KBResponse

DocumentResponse = doc_schema_mod.DocumentResponse
DocumentEditRequest = doc_schema_mod.DocumentEditRequest

ChatHistoryItem = chat_schema_mod.ChatHistoryItem
ChatRequest = chat_schema_mod.ChatRequest
SearchPlaygroundRequest = chat_schema_mod.SearchPlaygroundRequest
SearchPlaygroundResponse = chat_schema_mod.SearchPlaygroundResponse

ModelConfigSaveRequest = model_schema_mod.ModelConfigSaveRequest
ModelConfigResponse = model_schema_mod.ModelConfigResponse
ModelFetchRequest = model_schema_mod.ModelFetchRequest
ModelTestConnectionRequest = model_schema_mod.ModelTestConnectionRequest
