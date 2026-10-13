from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from prism import __version__ as core_version
from prism.ingest.parsers import ParserRegistry
from prism.ingest.parsers.mineru_parser import MINERU_EXTENSIONS, mineru_available
from prism.ingest.parsers.mineru_tiers import check_connection, tier_overview, verify_models

from ..config import app_config
from ..db import Repository, get_db
from ..services import index_service, model_service, settings_service

router = APIRouter()


@router.get("/status")
def status(db: Session = Depends(get_db)):
    repo = Repository(db)
    kbs = repo.list_kbs()
    docs = sum(len(repo.list_documents(kb.id)) for kb in kbs)
    chunks = 0
    for kb in kbs:
        meta = index_service.stats(kb.id)
        chunks += int(meta.get("children") or 0)
    s = settings_service.get()
    return {
        "project": app_config.project_name,
        "version": app_config.version,
        "coreVersion": core_version,
        "runtime": model_service.status(),
        "counts": {"knowledgeBases": len(kbs), "documents": docs, "chunks": chunks, "modelConfigs": len(repo.list_model_configs())},
        "pipeline": {
            "chunking": s.chunking.model_dump(mode="json"),
            "funnel": {"recallTopK": s.retrieval.top_n_per_route, "candidates": s.retrieval.rerank.candidates, "rerankTopN": s.retrieval.rerank.keep_n, "contextTopN": s.retrieval.rerank.top_k},
            "fusion": s.retrieval.fusion.model_dump(mode="json"),
            "queryTransform": s.retrieval.query_transform.model_dump(mode="json"),
            "diversify": s.retrieval.diversify.model_dump(mode="json"),
            "compression": s.retrieval.compression.model_dump(mode="json"),
            "graph": s.retrieval.graph.model_dump(mode="json"),
            "granularity": s.retrieval.granularity.model_dump(mode="json"),
            "rerank": s.retrieval.rerank.model_dump(mode="json", exclude={"api_key"}),
            "sparse": s.sparse.model_dump(mode="json"),
        },
    }


@router.get("/parsing")
def parsing():
    """解析能力矩阵：每种扩展名由哪个解析器处理；MinerU 档位与安装指引；MinerU-Popo 状态。"""
    s = settings_service.get()
    reg = ParserRegistry(s)
    matrix = []
    for ext in sorted(reg.supported_extensions() | set(MINERU_EXTENSIONS)):
        try:
            name = reg.get_parser(app_config.data_dir / f"x.{ext}").name
        except ValueError:
            name = None
        matrix.append({"ext": ext, "parser": name, "viaMineru": name == "mineru", "needsMineru": name is None})
    overview = tier_overview(s.parsing.mineru, mineru_available())
    popo_dir = s.paths.popo_dir
    trees = sorted((popo_dir / "outputs" / "build_tree" / "mineru").glob("*.json")) if popo_dir.exists() else []
    return {
        "matrix": matrix,
        "mineru": {
            "available": mineru_available(),
            "enabled": s.parsing.mineru.enabled,
            "tier": s.parsing.mineru.tier,
            "ocrMode": s.parsing.mineru.ocr_mode,
            "imageAnalysis": s.parsing.mineru.image_analysis,
            "apiUrl": s.parsing.mineru.api_url,
            "modelSource": s.parsing.mineru.model_source,
            "tiers": overview["tiers"],
            "hardware": overview["hardware"],
            "recommended": overview["recommended"],
            "remote": overview["remote"],
            "source": overview["source"],
            "mode": overview["mode"],
            "effectiveTier": overview["effectiveTier"],
            "install": [
                "uv pip install -U mineru                        # 基础包：ONNX CPU 小模型 + llama.cpp VLM，开箱即用",
                'uv pip install -U "mineru[torch]>=4.0,<5"      # Torch 小模型（Apple Silicon / 有 GPU 推荐）',
                'uv pip install -U "mineru[full]>=4.0,<5"       # vLLM（Linux）/ LMDeploy（Windows）高性能推理',
                "export MINERU_MODEL_SOURCE=modelscope           # 国内走 ModelScope 镜像（或在策略配置里选择模型下载源）",
                "mineru-kit models download --tier standard      # basic ≈0.8GB；standard / advanced ≈2GB",
                "mineru-kit models verify                        # 校验模型包完整性",
                "mineru-kit api-server --host 0.0.0.0 --port 8000 --tier standard   # 可选：服务化后在策略配置里选「自托管服务」并填地址",
                "# 不想在本机跑模型：策略配置 → 文档解析 → 解析方式选「云端 API」（mineru.net，可填 API 密钥；文档会上传到 MinerU 服务器）",
            ],
        },
        "popo": {
            "enabled": s.parsing.popo.enabled,
            "dir": str(popo_dir),
            "trees": len(trees),
            "subtasks": ["文本截断恢复（跨页段落续接）", "表格截断恢复（跨页表格合并）", "标题层级重建（全文开放层级）", "图文关联（caption ↔ 图表 ↔ 章节）"],
            "steps": [
                "知识库 → 文档 →「导出 Popo 输入」（MinerU 解析 + 标签归一化，bbox → 0~1 xyxy）",
                "POPO_REPO=/path/to/MinerU-Popo POPO_DIR=" + str(popo_dir) + " bash scripts/run_popo.sh（推理 + 建树，需 GPU）",
                "策略配置打开 parsing.popo.enabled，文档点「重新解析」即用文档树替换页面级结构",
            ],
        },
    }


@router.get("/mineru/tiers")
def mineru_tiers():
    """MinerU 全部解析档位及其在当前环境的状态（本机安装 / 云端 API 或自托管服务的 /v1/tiers）；策略配置页的档位下拉由此加载。"""
    s = settings_service.get()
    return {"available": mineru_available(), "enabled": s.parsing.mineru.enabled, "tier": s.parsing.mineru.tier, "apiUrl": s.parsing.mineru.api_url, **tier_overview(s.parsing.mineru, mineru_available())}


@router.post("/mineru/test")
def mineru_test():
    """测试已保存的远程解析配置（云端 API / 自托管服务）：能否连通、密钥是否有效、提供哪些档位、身份级别与限额。"""
    return check_connection(settings_service.get().parsing.mineru)


@router.post("/mineru/verify")
def mineru_verify(tier: str | None = None):
    """运行 `mineru-kit models verify --tier <档位>`，确认该档位所需模型包已完整下载（advanced 与 standard 共用）。"""
    return verify_models(tier)
