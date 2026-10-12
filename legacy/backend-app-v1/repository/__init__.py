import importlib

# 动态加载 kebab-case 的 repository 模块
sqlite_repo_mod = importlib.import_module(".sqlite-repository", __package__)
qdrant_repo_mod = importlib.import_module(".qdrant-repository", __package__)

SQLiteRepository = sqlite_repo_mod.SQLiteRepository
init_db = sqlite_repo_mod.init_db
get_db = sqlite_repo_mod.get_db
SessionLocal = sqlite_repo_mod.SessionLocal
qdrant_repository = qdrant_repo_mod.qdrant_repository
