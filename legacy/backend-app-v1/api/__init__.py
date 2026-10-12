import importlib

# 动态加载 kebab-case 的 API router 模块
kb_router_mod = importlib.import_module(".kb-router", __package__)
doc_router_mod = importlib.import_module(".document-router", __package__)
chat_router_mod = importlib.import_module(".chat-router", __package__)
model_router_mod = importlib.import_module(".model-router", __package__)

# 导出 routers
kb_router = kb_router_mod.router
document_router = doc_router_mod.router
chat_router = chat_router_mod.router
model_router = model_router_mod.router
