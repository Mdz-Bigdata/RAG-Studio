import importlib

# 动态加载 kebab-case 的 service 模块
parser_service_mod = importlib.import_module(".parser-service", __package__)
embedding_service_mod = importlib.import_module(".embedding-service", __package__)
retrieval_service_mod = importlib.import_module(".retrieval-service", __package__)
rerank_service_mod = importlib.import_module(".rerank-service", __package__)
llm_service_mod = importlib.import_module(".llm-service", __package__)
unweaver_service_mod = importlib.import_module(".unweaver-service", __package__)

# 导出服务实例或服务类
DocumentParserService = parser_service_mod.DocumentParserService
EmbeddingService = embedding_service_mod.EmbeddingService
RetrievalService = retrieval_service_mod.RetrievalService
RerankService = rerank_service_mod.RerankService
LLMService = llm_service_mod.LLMService
UnWeaverService = unweaver_service_mod.UnWeaverService

# 实例化全局单例
parser_service = DocumentParserService()
embedding_service = EmbeddingService()
retrieval_service = RetrievalService()
rerank_service = RerankService()
llm_service = LLMService()
unweaver_service = UnWeaverService()
