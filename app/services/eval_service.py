"""评测服务：黄金集 → 分阶段消融（dense / sparse / fused / final）→ 指标 + 错误归因，报告入库；
语料准备：意图空间 × 知识空间覆盖分析（高匹配 / 部分匹配 / 无匹配 + 无关 / 过期文档）。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from prism.eval import CoverageAnalyzer, EvalRunner, GoldenExample, load_golden_set
from prism.training import MiningConfig

from ..config import app_config
from ..db import Repository, SessionLocal
from .index_service import index_service
from .model_service import model_service
from .retrieval_service import retrieval_service


class EvalService:
    def sample(self) -> list[dict[str, Any]]:
        p = app_config.data_dir / "eval" / "golden_qa.jsonl"
        return [e.model_dump() for e in load_golden_set(p)] if p.exists() else []

    def mine_hard_negatives(self, kb_id: str, examples: list[dict[str, Any]], *, held_out: list[dict[str, Any]], config: MiningConfig) -> dict[str, Any]:
        """已标注训练问题 → 多路难负例候选；结果供复核 / 导出，不修改索引或启动训练。"""
        from prism.training import HardNegativeMiner

        golden = [GoldenExample.model_validate(e) for e in examples]
        reserved = [GoldenExample.model_validate(e) for e in held_out]
        with index_service.reading(kb_id):
            retriever = retrieval_service.retriever(kb_id)
            retriever.transformer.llm = None
            before = _rerank_state(retriever.reranker)
            # 改写负例用辅助文本模型；召回阶段仍不调用 LLM
            llm = model_service.llm(aux=True) if config.synthetic_negatives else None
            report = HardNegativeMiner(retriever, config, llm=llm).run(golden, held_out=reserved)
            data = report.to_dict()
            data["runtimeModels"] = _model_info(retriever, before)
        data["kb_id"] = kb_id
        return data

    def run(self, kb_id: str, examples: list[dict[str, Any]], *, ks: list[int], name: str | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        golden = [GoldenExample.model_validate(e) for e in examples]
        with index_service.reading(kb_id):
            retriever = retrieval_service.retriever(kb_id, overrides)
            retriever.transformer.llm = None  # 评测默认关闭 LLM 查询变换，保证可复现
            before = _rerank_state(retriever.reranker)
            report = EvalRunner(retriever, ks=ks, ablation=True).run(golden)
            models = _model_info(retriever, before)
        models["overrides"] = overrides or {}
        models["questionSet"] = _question_set(golden)
        previous = next(iter(self.reports(kb_id)), None)
        data = {
            "n": report.n, "ks": report.ks, "stageMetrics": report.stage_metrics, "categoryMetrics": report.category_metrics,
            "errorCounts": report.error_counts, "refusalTotal": report.refusal_total, "refusalCorrect": report.refusal_correct,
            "seconds": round(report.seconds, 2), "examples": [asdict(e) for e in report.examples],
            "name": name, "models": models,
        }
        data["comparison"] = _compare_eval(previous, data) if previous else None
        data["markdown"] = _markdown(report.to_markdown(), models, data["comparison"])
        with SessionLocal() as db:
            row = Repository(db).add_eval_report(kb_id, name, data)
            data["id"] = row.id
            data["createdAt"] = row.created_at.isoformat()
        return data

    def reports(self, kb_id: str) -> list[dict[str, Any]]:
        import json

        with SessionLocal() as db:
            rows = Repository(db).list_eval_reports(kb_id)
        out = []
        for r in rows:
            d = json.loads(r.report_json)
            d["id"] = r.id
            d["createdAt"] = r.created_at.isoformat()
            out.append(d)
        return out

    # --- 语料准备：意图 × 知识 覆盖分析 ---------------------------------------------------
    def coverage(self, kb_id: str, intents: list[Any], *, high: float = 0.6, partial: float = 0.35) -> dict[str, Any]:
        with index_service.reading(kb_id):
            retriever = retrieval_service.retriever(kb_id)
            retriever.transformer.llm = None  # 覆盖分析看的是语料本身，不让查询改写"帮忙"
            before = _rerank_state(retriever.reranker)
            report = CoverageAnalyzer(retriever, high=high, partial=partial).run(intents)
            models = {**_model_info(retriever, before), "scoring": report.scoring}
        names = retrieval_service.doc_names(kb_id)
        data = report.to_dict()
        for d in data["docs"]:
            d["file_name"] = names.get(d["doc_id"])
        for r in data["intents"]:
            for t in r["top"]:
                t["file_name"] = names.get(t["doc_id"])
        data["kb_id"] = kb_id
        data["models"] = models
        data["createdAt"] = datetime.now().isoformat(timespec="seconds")
        previous = self.last_coverage(kb_id)
        data["comparison"] = _compare(previous, data) if previous else None
        data["markdown"] = _markdown(report.to_markdown(), models, data["comparison"])
        with SessionLocal() as db:
            Repository(db).set_setting(f"coverage:{kb_id}", data)
        return data

    def last_coverage(self, kb_id: str) -> dict[str, Any] | None:
        with SessionLocal() as db:
            return Repository(db).get_setting(f"coverage:{kb_id}")


# --- 覆盖分析的模型记录与可比性 ------------------------------------------------------------
# 同一批意图的匹配分只有在"嵌入模型 + 实际生效的精排 + 打分公式 + 阈值"都相同时才能前后比较：
# 例如精排模型下载中时会回退到词法精排，打分公式随之切换，分数整体偏低，看起来像"语料变差了"。

    # --- 多模态评测 -----------------------------------------------------------------------
    def multimodal_samples(self, kb_id: str) -> list[dict[str, Any]]:
        from prism.eval import build_samples

        from .index_service import index_service

        idx = index_service.get(kb_id)
        return [e.model_dump() for e in build_samples(idx.images, idx.store)]

    def multimodal_run(self, kb_id: str, examples: list[dict[str, Any]], *, ks: list[int], name: str | None = None) -> dict[str, Any]:
        from datetime import datetime

        from prism.eval import MultimodalEvalRunner, MultimodalExample

        from .multimodal_service import multimodal_service

        hybrid = retrieval_service.retriever(kb_id)
        hybrid.transformer.llm = None   # 与文本评测一致：关闭 LLM 查询变换，保证可复现
        runner = MultimodalEvalRunner(multimodal_service.retriever(kb_id), hybrid, ks=ks)
        report = runner.run([MultimodalExample.model_validate(e) for e in examples])
        data = report.to_dict()
        data.update({"name": name, "kb_id": kb_id, "createdAt": datetime.utcnow().isoformat()})
        with SessionLocal() as db:
            repo = Repository(db)
            history = repo.get_setting(f"mm_eval:{kb_id}", []) or []
            repo.set_setting(f"mm_eval:{kb_id}", ([data] + history)[:10])
        return data

    def multimodal_reports(self, kb_id: str) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            return Repository(db).get_setting(f"mm_eval:{kb_id}", []) or []


_MODEL_KEYS = {"embedderModel": "嵌入模型", "rerankerEffective": "实际生效的精排", "scoring": "打分公式"}


def _rerank_state(rr) -> str | None:
    status = getattr(rr, "status", None)
    try:
        return status().get("state") if callable(status) else None
    except Exception:
        return None


def _model_info(retriever, state_before: str | None) -> dict[str, Any]:
    emb, rr = retriever.index.embedder, retriever.reranker
    emb_model = getattr(emb, "model_name", None) or emb.name
    rr_name, rr_model = getattr(rr, "name", "none"), getattr(rr, "model_name", None)
    state_after = _rerank_state(rr)
    # 交叉编码器"下载 / 加载中 / 失败"时由兜底精排打分。idle 不算：权重在本机时第一次调用会同步加载，打分仍来自交叉编码器
    fallback = (state_after is not None and state_after != "ready") or state_before in ("loading", "failed")
    effective = f"{rr_name}:{rr_model}" if rr_model else rr_name
    if fallback:
        effective += "（未就绪，兜底精排）"
    import hashlib

    retrieval_cfg = hashlib.sha1(retriever.cfg.model_dump_json().encode("utf-8")).hexdigest()[:12]
    return {
        "embedder": emb.name, "embedderModel": emb_model, "embedderOffline": emb.name == "hash",
        "reranker": rr_name, "rerankerModel": rr_model, "rerankerState": state_after, "rerankerFallback": fallback,
        "rerankerEffective": effective,
        "retrievalConfig": retrieval_cfg,          # 检索策略（融合 / 精排阈值 / 查询变换…）指纹：变了仍可比，但要提示
    }


def _strategy_notes(pm: dict[str, Any] | None, cm: dict[str, Any]) -> list[str]:
    if pm and pm.get("retrievalConfig") and pm.get("retrievalConfig") != cm.get("retrievalConfig"):
        return ["检索策略配置与上次不同：指标变化可能来自策略调整，而不只是语料"]
    return []


def _compare(prev: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    pm = dict(prev.get("models") or {})
    pm.setdefault("scoring", prev.get("scoring", ""))
    reasons: list[str] = []
    if not prev.get("models"):
        reasons.append("上次分析没有记录所用模型（早于该功能），无法确认可比")
    else:
        for key, label in _MODEL_KEYS.items():
            if pm.get(key) != cur["models"].get(key):
                reasons.append(f"{label}不同：{pm.get(key) or '—'} → {cur['models'].get(key) or '—'}")
    if prev.get("scoring") and prev.get("scoring") != cur.get("scoring") and not any(r.startswith("打分公式") for r in reasons):
        reasons.append(f"打分公式不同：{prev.get('scoring')} → {cur.get('scoring')}")
    if prev.get("thresholds") != cur.get("thresholds"):
        reasons.append(f"匹配阈值不同：{prev.get('thresholds')} → {cur.get('thresholds')}")
    same_intents = {r["intent"] for r in prev.get("intents", [])} == {r["intent"] for r in cur.get("intents", [])}
    return {
        "comparable": not reasons,
        "reasons": reasons,
        "notes": _strategy_notes(prev.get("models"), cur["models"]),
        "sameIntents": same_intents,
        "previous": {k: prev.get(k) for k in ("createdAt", "n", "buckets", "coverage_rate", "answerable_rate", "scoring", "thresholds")} | {"models": prev.get("models")},
    }


def _question_set(golden: list[GoldenExample]) -> str:
    """评估集指纹：问题 + 标注一起算，换了题或改了标注都视为不同的评估集。"""
    import hashlib
    import json

    raw = json.dumps(sorted((g.model_dump_json() for g in golden)), ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _eval_summary(r: dict[str, Any]) -> dict[str, Any]:
    final = (r.get("stageMetrics") or {}).get("final") or {}
    return {"createdAt": r.get("createdAt"), "name": r.get("name"), "n": r.get("n"), "hit@1": final.get("hit@1"), "mrr": final.get("mrr"),
            "final": final, "refusalCorrect": r.get("refusalCorrect"), "refusalTotal": r.get("refusalTotal"), "models": r.get("models")}


def _compare_eval(prev: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    """评测报告的可比性：嵌入模型、实际生效的精排、按请求的参数覆盖、评估集、K 值都相同才能比较指标变化。"""
    pm, cm = prev.get("models"), cur["models"]
    reasons: list[str] = []
    if not pm:
        reasons.append("上次评测没有记录所用模型（早于该功能），无法确认可比")
    else:
        for key, label in (("embedderModel", "嵌入模型"), ("rerankerEffective", "实际生效的精排")):
            if pm.get(key) != cm.get(key):
                reasons.append(f"{label}不同：{pm.get(key) or '—'} → {cm.get(key) or '—'}")
        if pm.get("overrides") != cm.get("overrides"):
            reasons.append("按请求的检索参数覆盖不同")
        if pm.get("questionSet") != cm.get("questionSet"):
            reasons.append("评估集（问题或标注）不同")
    if prev.get("ks") != cur.get("ks"):
        reasons.append(f"K 值不同：{prev.get('ks')} → {cur.get('ks')}")
    return {"comparable": not reasons, "reasons": reasons, "notes": _strategy_notes(pm, cm), "previous": _eval_summary(prev)}


def _markdown(md: str, models: dict[str, Any], comparison: dict[str, Any] | None) -> str:
    head, _, rest = md.partition("\n")
    rr = models["rerankerEffective"]
    lines = [head, "", f"- 模型：嵌入 {models['embedderModel']}{'（离线哈希向量，无真实语义）' if models['embedderOffline'] else ''} · 精排 {rr}"]
    if comparison and not comparison["comparable"]:
        lines.append(f"- ⚠ 与上一份报告（{comparison['previous'].get('createdAt') or '时间未知'}）不可直接比较：" + "；".join(comparison["reasons"]))
    elif comparison and not comparison.get("sameIntents", True):     # 评测报告的评估集差异已在 reasons 里
        lines.append("- 注意：与上次分析的意图集合不同，总体比例不可直接比较")
    return "\n".join(lines) + rest


eval_service = EvalService()
