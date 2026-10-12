import os
os.environ["QDRANT_PATH"] = ":memory:"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.model import Base, ChunkModel, EntityRelationModel, ModelConfigModel
from app.service import retrieval_service, unweaver_service
from app.repository import SQLiteRepository

def test_custom_tokenizer():
    """
    测试轻量化中英文分词器是否工作正常
    """
    text = "Antigravity RAG 系统的性能非常 Strong！它能解析 PDF 并提取出 **MinerU** 数据。"
    tokens = retrieval_service.tokenize(text)
    
    # 验证是否包含中文字符和英文单词
    assert "antigravity" in tokens
    assert "rag" in tokens
    assert "pdf" in tokens
    assert "mineru" in tokens
    assert "系" in tokens
    assert "能" in tokens
    # 标点符号应被剔除
    assert "！" not in tokens
    assert "**" not in tokens

def test_reciprocal_rank_fusion():
    """
    测试 RRF (Reciprocal Rank Fusion) 倒数排名融合算法的有效性与去重
    """
    dense = [
        {"id": "doc_a", "score": 0.9, "source": "dense"},
        {"id": "doc_b", "score": 0.8, "source": "dense"}
    ]
    bm25 = [
        {"id": "doc_c", "score": 12.5, "source": "bm25"},
        {"id": "doc_a", "score": 8.2, "source": "bm25"}
    ]
    hyde = [
        {"id": "doc_b", "score": 0.85, "source": "hyde"},
        {"id": "doc_d", "score": 0.7, "source": "hyde"}
    ]
    
    # 运行融合 (k = 60)
    # RRF Score (doc_a) = 1/(60+1) + 1/(60+2) = 0.01639 + 0.01612 = 0.03251
    # RRF Score (doc_b) = 1/(60+2) + 1/(60+1) = 0.03251
    # RRF Score (doc_c) = 1/(60+1) = 0.01639
    # RRF Score (doc_d) = 1/(60+2) = 0.01612
    # 最终 doc_a 和 doc_b 排名应该位于前两位
    fused = retrieval_service._reciprocal_rank_fusion(dense, bm25, hyde, k=60)
    
    assert len(fused) == 4
    top_ids = [item["id"] for item in fused]
    assert "doc_a" in top_ids[:2]
    assert "doc_b" in top_ids[:2]
    assert fused[-1]["id"] == "doc_d"  # 倒数排名得分最低
    
    # 验证 sources 字段是否正确合并
    doc_a_item = next(item for item in fused if item["id"] == "doc_a")
    assert "dense" in doc_a_item["sources"]
    assert "bm25" in doc_a_item["sources"]

def test_unweaver_entity_extraction():
    """
    测试 UnWeaver 正则解析引擎是否能从 Markdown 段落中精准提取名词及其解释
    """
    # 模拟内存 SQLite 数据库
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)
    db = Session()
    Base.metadata.create_all(bind=engine)
    
    text = (
        "在这个分布式系统中，**Qdrant** 是指一个高性能的向量数据库。\n"
        "同时我们需要知道，**RRF** 是一种倒数排名融合算法，它不需要参数配置。"
    )
    
    kb_id = "kb_test_123"
    chunk_id = "chunk_test_456"
    
    count = unweaver_service.extract_and_store_entities(db, kb_id, chunk_id, text)
    
    # 验证是否成功提取并存库 2 个实体
    assert count == 2
    
    relations = db.query(EntityRelationModel).all()
    assert len(relations) == 2
    
    entities = [r.entityName for r in relations]
    assert "Qdrant" in entities
    assert "RRF" in entities
    
    qdrant_rel = next(r for r in relations if r.entityName == "Qdrant")
    assert "高性能的向量数据库" in qdrant_rel.definition
    assert qdrant_rel.chunkId == chunk_id
    
    db.close()

def test_model_config_crud():
    """
    测试模型配置表在 SQLite 仓储层下的增删改查、排他性激活和级联状态
    """
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)
    db = Session()
    Base.metadata.create_all(bind=engine)
    
    repo = SQLiteRepository(db)
    
    # 1. 创建配置 A
    config_a = ModelConfigModel(
        id="config_1",
        providerName="SiliconFlow",
        baseUrl="https://api.siliconflow.cn/v1",
        apiKey="sk-test-123",
        category="text",
        modelName="qwen-turbo",
        isActive=0
    )
    repo.create_or_update_model_config(config_a)
    
    # 2. 查询列表验证
    configs = repo.list_model_configs()
    assert len(configs) == 1
    assert configs[0].modelName == "qwen-turbo"
    
    # 3. 激活配置 A
    repo.activate_model_config("config_1", "text")
    active = repo.get_active_model_config("text")
    assert active is not None
    assert active.modelName == "qwen-turbo"
    assert active.isActive == 1
    
    # 4. 创建同类别的配置 B 并激活它 (验证原 A 配置是否被排他置为 0)
    config_b = ModelConfigModel(
        id="config_2",
        providerName="DeepSeek",
        baseUrl="https://api.deepseek.com/v1",
        apiKey="sk-test-456",
        category="text",
        modelName="deepseek-chat",
        isActive=0
    )
    repo.create_or_update_model_config(config_b)
    repo.activate_model_config("config_2", "text")
    
    active_now = repo.get_active_model_config("text")
    assert active_now.modelName == "deepseek-chat"
    assert active_now.isActive == 1
    
    old_active = repo.get_model_config("config_1")
    assert old_active.isActive == 0
    
    # 5. 删除配置 A 并验证列表数量
    repo.delete_model_config("config_1")
    assert len(repo.list_model_configs()) == 1
    
    db.close()

def test_parse_temp_document():
    """
    测试文档临时解析接口 (/api/v1/documents/parse-temp) 提取纯文本逻辑
    """
    from fastapi.testclient import TestClient
    from app.main import app
    
    client = TestClient(app)
    
    # 模拟文本上传
    file_payload = {
        "file": ("temp_test.txt", b"Antigravity Hybrid RAG ad-hoc text", "text/plain")
    }
    
    res = client.post("/api/v1/documents/parse-temp", files=file_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["content"] == "Antigravity Hybrid RAG ad-hoc text"
    assert data["fileName"] == "temp_test.txt"
    
    # 模拟上传不支持的格式 (如 .exe) 触发 400 校验拦截
    file_payload_err = {
        "file": ("temp_test.exe", b"binary data...", "application/octet-stream")
    }
    res_err = client.post("/api/v1/documents/parse-temp", files=file_payload_err)
    assert res_err.status_code == 400
    assert "Unsupported file format" in res_err.json()["detail"]
