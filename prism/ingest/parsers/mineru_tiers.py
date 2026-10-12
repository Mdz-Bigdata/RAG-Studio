"""MinerU 4.x 解析档位：目录、可用性探测、按硬件推荐。

4.0 用四级解析质量（tier）取代了 3.x 的 pipeline / vlm / hybrid 后端，按"质量—速度—成本"取舍：
PDF 与图片支持全部四级；Office / OpenDocument / EPUB / OFD / HTML / CSV 在 flash 层由 DocVortex 原生解析。

档位列表的来源（按优先级）：
1. 配置了 `parsing.mineru.api_url` → 向自托管服务 `GET /v1/tiers` 询问它实际提供哪些档位；
2. 本机装了 mineru → 四档都可选：flash 不需要模型；其余用 MinerU 自己的模型注册表逐档核对
   （`mineru.model.registry.model_repos_for_tier` + `verify_model_repo`，advanced 与 standard 共用模型包），
   并检查该档位运行时依赖是否齐全（`mineru.parser.tier.missing_modules_for_tier`）；
3. 没装 → 四档仍然列出（可以先选好，装上即生效），状态标为"未安装"。
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from typing import Any

from ...config import MinerUConfig

logger = logging.getLogger(__name__)

# 资源数字来自 MinerU 4.0 发布说明：basic（ONNX）≈0.8GB 模型 / 2GB 内存；standard / advanced（ONNX + llama.cpp）≈2GB / 8GB；
# torch + vLLM ≈3GB / 16GB。
TIER_CATALOG: list[dict[str, Any]] = [
    {
        "tier": "flash", "label": "原生文本 + 版面 / OCR 小模型", "quality": "最低", "speed": "最快",
        "use": "发现、预览、建索引；Office 等数字文档（DocVortex 原生解析，不转 PDF）",
        # 实测（mineru 4.0.10）：PDF 在 flash 档也会跑 PP-DocLayoutV2 版面检测与 PP-OCRv6 OCR，首次解析自动下载约 240MB
        "models": "版面检测 + OCR 小模型，不跑 VLM", "requires": "无需 GPU；首次解析 PDF 自动下载约 240MB 小模型",
        "needsModels": False, "modelGb": 0.24, "minMemoryGb": 1,
        "formats": "PDF / 图片 / Office / OpenDocument / EPUB / OFD / HTML / CSV",
    },
    {
        "tier": "basic", "label": "小模型 OCR / 公式 / 表格", "quality": "基础", "speed": "快",
        "use": "扫描件、表格、公式的低资源本地解析",
        "models": "ONNX / Torch 小模型，无 VLM", "requires": "可纯 CPU；模型约 0.8GB，内存 ≥ 2GB",
        "needsModels": True, "modelGb": 0.8, "minMemoryGb": 2,
        "formats": "PDF / 图片",
    },
    {
        "tier": "standard", "label": "小模型 + VLM（默认质量）", "quality": "高", "speed": "中",
        "use": "复杂版面的主力档位：多栏、图文混排、跨栏表格",
        "models": "小模型 + VLM（llama.cpp / vLLM / LMDeploy）", "requires": "模型约 2GB，内存 ≥ 8GB；建议 Apple Silicon 或 8GB+ 显存",
        "needsModels": True, "modelGb": 2.0, "minMemoryGb": 8,
        "formats": "PDF / 图片",
    },
    {
        "tier": "advanced", "label": "最高质量，最慢", "quality": "最高", "speed": "最慢",
        "use": "疑难文档：投入更多推理算力换质量",
        "models": "与 standard 共用模型包，推理更充分", "requires": "同 standard，耗时更长",
        "needsModels": True, "modelGb": 2.0, "minMemoryGb": 8,
        "formats": "PDF / 图片",
    },
]
_KNOWN = {t["tier"] for t in TIER_CATALOG}

STATUS_LABEL = {
    "ready": "可用",
    "missing_deps": "缺少运行依赖",
    "needs_models": "需下载模型包",
    "not_installed": "未安装 MinerU",
    "remote": "由解析服务提供",
    "remote_missing": "解析服务未提供",
}


def detect_hardware() -> dict[str, Any]:
    """粗略识别加速器：Apple Silicon / NVIDIA（显存）/ 无。只用于给出档位建议，不影响可选项。"""
    info: dict[str, Any] = {"system": platform.system(), "machine": platform.machine(), "accelerator": "none", "gpuMemoryGb": None}
    if info["system"] == "Darwin" and info["machine"] in ("arm64", "aarch64"):
        info["accelerator"] = "apple_silicon"
        return info
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4).stdout
            mems = [float(x) for x in out.split() if x.replace(".", "", 1).isdigit()]
            if mems:
                info["accelerator"] = "nvidia"
                info["gpuMemoryGb"] = round(max(mems) / 1024, 1)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return info


def recommend_tier(hw: dict[str, Any]) -> dict[str, str]:
    """官方硬件建议：Apple Silicon → standard + torch；NVIDIA 8GB+ → standard + full；4GB+ → basic + torch；否则 basic。"""
    if hw.get("accelerator") == "apple_silicon":
        return {"tier": "standard", "extra": "torch", "reason": "Apple Silicon：standard 档 + mineru[torch]"}
    mem = hw.get("gpuMemoryGb") or 0
    if hw.get("accelerator") == "nvidia" and mem >= 8:
        return {"tier": "standard", "extra": "full", "reason": f"NVIDIA {mem}GB 显存：standard 档 + mineru[full]（vLLM / LMDeploy）"}
    if hw.get("accelerator") == "nvidia" and mem >= 4:
        return {"tier": "basic", "extra": "torch", "reason": f"NVIDIA {mem}GB 显存：basic 档 + mineru[torch]"}
    return {"tier": "basic", "extra": "", "reason": "没有现代加速器：basic 档（ONNX 小模型，纯 CPU）；只做数字文档建索引用 flash"}


def local_tier_status(tier: str) -> dict[str, Any]:
    """本机某一档是否就绪：模型包是否下载完整 + 运行依赖是否齐全。探测失败时保守地报告"需下载模型包"。"""
    if tier == "flash":
        return {"status": "ready", "repos": [], "missingModules": []}
    model_tier = "standard" if tier == "advanced" else tier   # advanced 与 standard 共用模型包
    try:
        from mineru.model.download import verify_model_repo
        from mineru.model.registry import model_repos_for_tier
        from mineru.parser.tier import missing_modules_for_tier

        repos = []
        for repo in model_repos_for_tier(model_tier):
            r = verify_model_repo(repo)
            repos.append({"name": repo.name, "ready": bool(r.ready), "root": str(r.root), "missing": len(r.missing_paths)})
        missing_modules = list(missing_modules_for_tier(model_tier))
    except Exception as exc:  # 版本差异 / 配置损坏：不让状态页挂掉
        logger.info("MinerU 档位 %s 状态探测失败：%s", tier, exc)
        return {"status": "needs_models", "repos": [], "missingModules": [], "error": str(exc)[:200]}
    status = "missing_deps" if missing_modules else ("ready" if all(r["ready"] for r in repos) else "needs_models")
    return {"status": status, "repos": repos, "missingModules": missing_modules}


def remote_tiers(api_url: str, api_key: str = "", timeout: float = 3.0) -> dict[str, Any]:
    """询问自托管 `mineru-kit api-server`：/v1/health + /v1/tiers。返回 {reachable, tiers: {name: available}, error}。"""
    import httpx

    base = api_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    out: dict[str, Any] = {"url": base, "reachable": False, "tiers": {}, "error": None}
    try:
        out["reachable"] = httpx.get(f"{base}/v1/health", headers=headers, timeout=timeout).status_code < 500
        r = httpx.get(f"{base}/v1/tiers", headers=headers, timeout=timeout)
        r.raise_for_status()
        out["reachable"] = True
        out["tiers"] = _parse_remote_tiers(r.json())
    except Exception as exc:
        out["error"] = str(exc)[:200]
    return out


def _parse_remote_tiers(payload: Any) -> dict[str, bool]:
    """响应格式宽松解析：["flash", ...] / [{"tier"|"name"|"id": .., "available"|"ready"|"enabled": ..}] / {"tiers": [...]} / {"flash": {...}}。"""
    if isinstance(payload, dict):
        for key in ("tiers", "data", "items"):
            if key in payload:
                return _parse_remote_tiers(payload[key])
        return {str(k): _truthy(v) for k, v in payload.items()}
    result: dict[str, bool] = {}
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                result[item] = True
            elif isinstance(item, dict):
                name = item.get("tier") or item.get("name") or item.get("id")
                if name:
                    result[str(name)] = _truthy(item)
    return result


def _truthy(v: Any) -> bool:
    if isinstance(v, dict):
        for key in ("available", "ready", "enabled", "loaded"):
            if key in v:
                return bool(v[key])
        return True
    return bool(v) if isinstance(v, bool) else True


def tier_overview(cfg: MinerUConfig, available: bool) -> dict[str, Any]:
    """档位全景：每一档的说明 + 在当前环境里的状态。所有档位都返回、都可选。"""
    hw = detect_hardware()
    rec = recommend_tier(hw)
    remote = remote_tiers(cfg.api_url, cfg.api_key) if cfg.api_url else None
    served = (remote or {}).get("tiers") or {}
    tiers: list[dict[str, Any]] = []
    catalog = list(TIER_CATALOG)
    # 解析服务报告了目录之外的档位（未来版本）：也列出来供选择
    for name in served:
        if name not in _KNOWN:
            catalog.append({"tier": name, "label": "解析服务提供的档位", "quality": "—", "speed": "—", "use": "由自托管解析服务定义", "models": "—", "requires": "—", "needsModels": False, "modelGb": 0.0, "minMemoryGb": 0, "formats": "—"})
    for t in catalog:
        name = t["tier"]
        if remote is not None and remote["reachable"] and served:
            status = "remote" if served.get(name) else "remote_missing"
        elif remote is not None:
            status = "remote"            # 服务不可达 / 没有 /v1/tiers：无法判断，按服务端启动参数为准
        local: dict[str, Any] = {}
        if remote is not None and remote["reachable"] and served:
            pass
        elif remote is not None:
            pass
        elif not available:
            status = "not_installed"
        else:
            local = local_tier_status(name)
            status = local["status"]
        tiers.append({
            **t,
            "modelRepos": local.get("repos", []),
            "missingModules": local.get("missingModules", []),
            "status": status,
            "statusLabel": STATUS_LABEL[status],
            "selectable": True,
            "current": name == cfg.tier,
            "recommended": name == rec["tier"],
            "download": f"mineru-kit models download --tier {'standard' if name == 'advanced' else name}" if t["needsModels"] else None,
        })
    return {"tiers": tiers, "hardware": hw, "recommended": rec, "remote": remote, "source": "remote" if remote is not None else ("local" if available else "catalog")}


def verify_models(tier: str | None = None, timeout: float = 120.0) -> dict[str, Any]:
    """`mineru-kit models verify [--tier basic|standard]`：校验本机模型包是否完整。"""
    exe = shutil.which("mineru-kit")
    if exe is None:
        return {"ok": False, "ran": False, "output": "未找到 mineru-kit：先安装 MinerU（uv pip install -U \"mineru[torch]>=4.0,<5\"）"}
    try:
        model_tier = {"advanced": "standard", "standard": "standard", "basic": "basic"}.get(tier or "")
        p = subprocess.run([exe, "models", "verify", *(["--tier", model_tier] if model_tier else [])], capture_output=True, text=True, timeout=timeout)
        return {"ok": p.returncode == 0, "ran": True, "output": ((p.stdout or "") + (p.stderr or "")).strip()[-2000:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "ran": True, "output": f"校验超时（>{int(timeout)}s）"}
    except OSError as exc:
        return {"ok": False, "ran": False, "output": str(exc)}
