"""评测执行器：分阶段消融（dense / sparse / fused / final）+ 分类指标 + 错误归因。

错误归因（比指标更能指导优化方向）：
- 知识缺失 / 切块失败：任何一路 top-N 都没有相关块 → 先查语料与切片
- 语义漂移：BM25 找到了、向量没找到 → 换 embedding / 加面包屑 / 上下文摘要
- 关键词缺失：向量找到了、BM25 没找到 → 分词词典 / 精确 token 保护
- 融合失败：某路找到了、融合后掉出候选集 → 调 RRF 权重 / 候选数
- 排序失败：候选集里有、精排后掉出 top-K → 换 reranker / 候选数 ≥ 50
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from ..retrieval.retriever import HybridRetriever
from ..schemas import Chunk
from .dataset import GoldenExample
from .metrics import hit_rate, mrr, ndcg, recall_at_k

STAGES = ("dense", "sparse", "fused", "final")


@dataclass
class ExampleResult:
    question: str
    category: str
    per_stage: dict[str, dict[str, float]]
    error_type: str | None = None
    final_top: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    n: int = 0                                   # 参与检索指标的样例数（不含"无答案"样例）
    ks: list[int] = field(default_factory=list)
    stage_metrics: dict[str, dict[str, float]] = field(default_factory=dict)      # stage -> metric -> mean
    category_metrics: dict[str, dict[str, float]] = field(default_factory=dict)   # category -> metric(final) -> mean
    error_counts: dict[str, int] = field(default_factory=dict)
    examples: list[ExampleResult] = field(default_factory=list)
    refusal_total: int = 0                       # expected_refusal 样例数
    refusal_correct: int = 0                     # 其中系统确实没有给出上下文的数量
    seconds: float = 0.0

    def to_markdown(self) -> str:
        lines = [f"# 检索评测报告（n={self.n}，{self.seconds:.1f}s）", ""]
        metrics = _sorted_metrics({m for d in self.stage_metrics.values() for m in d})
        lines.append("## 分阶段消融")
        lines.append("| stage | " + " | ".join(metrics) + " |")
        lines.append("|---|" + "---|" * len(metrics))
        for stage in STAGES:
            if stage in self.stage_metrics:
                lines.append(f"| {stage} | " + " | ".join(f"{self.stage_metrics[stage].get(m, 0):.3f}" for m in metrics) + " |")
        lines.append("")
        lines.append("## 按问题类型（final）")
        cats = sorted(self.category_metrics)
        if cats:
            cmetrics = _sorted_metrics({m for d in self.category_metrics.values() for m in d})
            lines.append("| category | " + " | ".join(cmetrics) + " |")
            lines.append("|---|" + "---|" * len(cmetrics))
            for c in cats:
                lines.append(f"| {c} | " + " | ".join(f"{self.category_metrics[c].get(m, 0):.3f}" for m in cmetrics) + " |")
        lines.append("")
        if self.refusal_total:
            lines.append("## 拒答")
            lines.append(f"- 无答案样例 {self.refusal_total} 条，正确拒答（未给出上下文）{self.refusal_correct} 条；"
                         "未设置 retrieval.rerank.min_score 时系统总会返回上下文，需靠阈值 + 兜底话术实现拒答")
            lines.append("")
        lines.append("## 错误归因")
        if not self.error_counts:
            lines.append("- 无失败样例")
        for k, v in sorted(self.error_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)


def _sorted_metrics(names: set[str]) -> list[str]:
    def key(m: str) -> tuple[str, int]:
        base, _, k = m.partition("@")
        return (base, int(k) if k.isdigit() else 0)

    return sorted(names, key=key)


class EvalRunner:
    def __init__(self, retriever: HybridRetriever, ks: list[int] | None = None, ablation: bool = True) -> None:
        self.retriever = retriever
        self.ks = ks or [1, 3, 5, 10]
        self.ablation = ablation
        self.store = retriever.index.store

    def run(self, examples: list[GoldenExample]) -> EvalReport:
        t0 = time.perf_counter()
        report = EvalReport(ks=self.ks)
        sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        cat_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        cat_n: dict[str, int] = defaultdict(int)
        errors: dict[str, int] = defaultdict(int)
        max_k = max(self.ks)

        for ex in examples:
            if ex.expected_refusal:
                # "无答案"样例不参与检索指标，单独统计拒答正确率
                result = self.retriever.retrieve(ex.question, top_k=max_k)
                refused = not result.contexts
                report.refusal_total += 1
                report.refusal_correct += int(refused)
                err = None if refused else "不该答却答"
                if err:
                    errors[err] += 1
                report.examples.append(ExampleResult(question=ex.question, category=ex.category, per_stage={}, error_type=err, final_top=[c.chunk.chunk_id for c in result.contexts[:max_k]]))
                continue
            if not ex.has_labels():
                continue
            total_relevant = self._count_relevant_parents(ex)
            stage_ids: dict[str, list[str]] = {}
            if self.ablation:
                stage_ids["dense"] = [cid for cid, _ in self.retriever.retrieve_dense(ex.question, top_n=max(max_k, 50))]
                stage_ids["sparse"] = [cid for cid, _ in self.retriever.retrieve_sparse(ex.question, top_n=max(max_k, 50))]
                stage_ids["fused"] = [cid for cid, _ in self.retriever.retrieve_fused(ex.question)]
            result = self.retriever.retrieve(ex.question, top_k=max_k)
            stage_ids["final"] = [ctx.chunk.chunk_id for ctx in _by_score(result)]

            per_stage: dict[str, dict[str, float]] = {}
            for stage, ids in stage_ids.items():
                rel = self._relevance(ex, ids, dedupe_parent=(stage != "final"))
                m = self._metrics(rel, total_relevant)
                per_stage[stage] = m
                for name, v in m.items():
                    sums[stage][name] += v
            cat_n[ex.category] += 1
            for name, v in per_stage["final"].items():
                cat_sums[ex.category][name] += v

            err = None
            if per_stage["final"].get(f"hit@{max_k}", 0) == 0:
                err = self._classify_error(ex, stage_ids, result)
            if err:
                errors[err] += 1
            report.examples.append(ExampleResult(question=ex.question, category=ex.category, per_stage=per_stage, error_type=err, final_top=stage_ids["final"][:max_k]))
            report.n += 1

        for stage, d in sums.items():
            report.stage_metrics[stage] = {k: v / report.n for k, v in d.items()} if report.n else {}
        for cat, d in cat_sums.items():
            report.category_metrics[cat] = {k: v / cat_n[cat] for k, v in d.items()}
        report.error_counts = dict(errors)
        report.seconds = time.perf_counter() - t0
        return report

    # ----------------------------------------------------------------------------------
    def _relevance(self, ex: GoldenExample, ids: list[str], dedupe_parent: bool = True) -> list[bool]:
        """按 parent 粒度计分：每个检索结果折算成它覆盖的 parent（child / 句子级 → 所属 parent；章节级 → 其中的各 parent），
        每个相关 parent 只计一次分；覆盖的 parent 全部已出现过的结果直接跳过（同一 parent 的多个 child 不重复计）。
        否则多粒度索引下"章节 + 其中的段落"会被重复计分，NDCG 超过 1、Recall 被截断而失真。
        dedupe_parent 保留为兼容参数：各阶段统一按上述规则计分。"""
        seen: set[str] = set()
        out: list[bool] = []
        for cid in ids:
            c = self.store.get(cid)
            if c is None:
                continue
            if c.kind == "section":
                covered = [p for p in self.store.get_many(c.child_ids) if p.kind == "parent"] or [c]
            else:
                covered = [self.store.parent_of(cid) or c]
            new = [p for p in covered if p.chunk_id not in seen]
            if not new:
                continue
            seen.update(p.chunk_id for p in new)
            # 标注可能精确到 child（relevant_chunk_ids 里是 child id），所以命中的块本身也要看
            out.append(any(ex.is_relevant(p) for p in new) or (c.kind != "section" and ex.is_relevant(c)))
        return out

    def _count_relevant_parents(self, ex: GoldenExample) -> int:
        n = 0
        for c in self.store.chunks.values():
            if c.kind == "parent" and ex.is_relevant(c):
                n += 1
        return max(n, 1)

    def _metrics(self, rel: list[bool], total_relevant: int) -> dict[str, float]:
        m: dict[str, float] = {}
        for k in self.ks:
            m[f"hit@{k}"] = hit_rate(rel, k)
            m[f"recall@{k}"] = recall_at_k(rel, k, total_relevant)
            m[f"ndcg@{k}"] = ndcg(rel, k, total_relevant)
        m["mrr"] = mrr(rel)
        return m

    def _classify_error(self, ex: GoldenExample, stage_ids: dict[str, list[str]], result) -> str:
        def found(stage: str) -> bool:
            return any(self._relevance(ex, stage_ids.get(stage, []), dedupe_parent=True))

        in_dense, in_sparse = found("dense"), found("sparse")
        in_fused_candidates = any(ex.is_relevant(self.store.parent_of(sc.chunk_id) or self.store.get(sc.chunk_id)) for sc in result.trace.fused if self.store.get(sc.chunk_id))
        if not in_dense and not in_sparse and not in_fused_candidates:
            return "知识缺失/切块失败"
        if in_fused_candidates:
            return "排序失败(rerank)"
        if in_sparse and not in_dense:
            return "语义漂移(向量未召回)"
        if in_dense and not in_sparse:
            return "关键词缺失(BM25未召回)"
        return "融合失败"


def _by_score(result):
    return sorted(result.contexts, key=lambda c: c.score, reverse=True)
