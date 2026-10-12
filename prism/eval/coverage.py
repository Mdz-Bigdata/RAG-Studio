"""语料准备阶段：意图空间 × 知识空间 覆盖分析。

在调任何检索参数之前先回答两个问题——用户会问什么（意图空间）、知识库能答什么（知识空间），
然后把每条意图与现有语料的匹配程度分到三个区间，决定"补知识"还是"拿掉无关材料"：

    高匹配   意图有明确对应语料                         → 可直接进入评测 / 上线
    部分匹配 语料存在但缺章节 / 版本旧 / 只命中部分关键词   → 继续准备知识（补章节、更新版本、导出结构化表格）
    无匹配   高频意图完全没有对应语料                     → 列入知识补充计划，同时配置拒答话术

反过来看知识空间：没有被任何意图命中的文档是"与 RAG 目的不相关的材料"候选（噪音、权重虚高），
已过期的文档应下线或补全生效区间。

匹配分（0~1）由三类信号组成，全部来自一次真实检索，所以结论与线上行为一致：
- term_coverage：意图的关键词（按语料 IDF 加权，语料里从未出现的词权重最高）在最优上下文中的覆盖率；
- dense_sim：向量路 top-1 余弦相似度（绝对值，可跨意图比较）；
- rerank_score：交叉编码器 / API / LLM 精排的 top-1 相关度（仅这三类是绝对分数；词法精排是相对分数，不参与）。
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from ..index.bm25_store import PRECISE_TOKEN
from ..retrieval.retriever import HybridRetriever

BUCKETS = ("high", "partial", "none")
BUCKET_LABEL = {"high": "高匹配", "partial": "部分匹配", "none": "无匹配"}
BUCKET_ACTION = {
    "high": "语料已覆盖：纳入黄金评估集，作为回归基线",
    "partial": "继续准备知识：补充缺失章节 / 更新版本 / 把表格导出为结构化数据后重新入库",
    "none": "知识缺口：列入知识补充计划；补齐之前为该类问题配置拒答与转人工话术",
}

# 只表达提问方式、不承载知识点的词：不参与覆盖率计算
_FUNCTION_WORDS = {
    "能", "不能", "可以", "能否", "是否", "有", "没有", "多少", "多久", "多长", "几", "几种", "几点", "几天", "哪", "哪个", "哪里",
    "想", "问", "想问", "那", "啥", "来", "来着", "吗", "呢", "请", "帮", "帮忙", "告诉", "知道", "需要", "应该", "要", "会", "用", "使用",
    "处理", "办", "办理", "做", "写", "发", "买", "坐", "看", "查", "找", "不", "退不退", "有没有", "是不是", "为什么", "区别", "不同",
}
_ABSOLUTE_RERANKERS = {"cross_encoder", "api", "llm"}


@dataclass
class IntentResult:
    intent: str
    category: str
    score: float
    bucket: str
    bucket_label: str
    term_coverage: float
    dense_sim: float
    rerank_score: float | None
    missing_terms: list[str]
    top: list[dict[str, Any]]
    action: str
    note: str = ""


@dataclass
class DocCoverage:
    doc_id: str
    title: str
    source: str
    version: str | None
    valid_to: str | None
    expired: bool
    chunks: int
    hits: int                       # 被多少条意图作为 top 上下文命中（高匹配 / 部分匹配）
    hit_intents: list[str]
    status: str                     # core / supporting / unused / expired
    action: str


@dataclass
class CoverageReport:
    n: int = 0
    thresholds: dict[str, float] = field(default_factory=dict)
    buckets: dict[str, int] = field(default_factory=dict)
    coverage_rate: float = 0.0      # 高匹配占比
    answerable_rate: float = 0.0    # 高匹配 + 部分匹配占比
    by_category: dict[str, dict[str, int]] = field(default_factory=dict)
    intents: list[IntentResult] = field(default_factory=list)
    docs: list[DocCoverage] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    scoring: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [f"# 意图 × 知识 覆盖分析（{self.n} 条意图，{self.seconds:.1f}s）", ""]
        lines.append(f"- 高匹配 {self.buckets.get('high', 0)} · 部分匹配 {self.buckets.get('partial', 0)} · 无匹配 {self.buckets.get('none', 0)}")
        lines.append(f"- 覆盖率（高匹配）{self.coverage_rate:.0%}；可回答率（高 + 部分）{self.answerable_rate:.0%}")
        lines.append(f"- 打分方式：{self.scoring}")
        lines += ["", "## 意图空间", "| 意图 | 区间 | 匹配分 | 关键词覆盖 | 缺失关键词 | 最优语料 |", "|---|---|---|---|---|---|"]
        for r in sorted(self.intents, key=lambda x: x.score):
            top = r.top[0]["breadcrumb"] if r.top else "—"
            lines.append(f"| {r.intent} | {r.bucket_label} | {r.score:.2f} | {r.term_coverage:.0%} | {'、'.join(r.missing_terms) or '—'} | {top} |")
        lines += ["", "## 知识空间", "| 文档 | 版本 | 命中意图数 | 状态 | 处置 |", "|---|---|---|---|---|"]
        for d in self.docs:
            lines.append(f"| {d.title} | {d.version or '—'} | {d.hits} | {d.status} | {d.action} |")
        lines += ["", "## 建议"] + [f"- {x}" for x in self.recommendations]
        return "\n".join(lines)


class CoverageAnalyzer:
    """对一组意图跑真实检索，给出匹配区间、缺失关键词，以及每份文档被意图使用的情况。"""

    def __init__(self, retriever: HybridRetriever, *, high: float = 0.6, partial: float = 0.35, top_k: int = 3) -> None:
        if not 0 < partial < high <= 1:
            raise ValueError("阈值需满足 0 < partial < high ≤ 1")
        self.retriever = retriever
        self.index = retriever.index
        self.high = high
        self.partial = partial
        self.top_k = top_k
        self.absolute_rerank = getattr(retriever.reranker, "name", "") in _ABSOLUTE_RERANKERS
        self._idf, self._max_idf = self._build_idf()

    # ----------------------------------------------------------------------------------
    def run(self, intents: list[str | dict[str, Any]], on_date: date | None = None) -> CoverageReport:
        t0 = time.perf_counter()
        on_date = on_date or date.today()
        report = CoverageReport(thresholds={"high": self.high, "partial": self.partial}, buckets={b: 0 for b in BUCKETS})
        report.scoring = "0.6 × 精排相关度 + 0.4 × 关键词覆盖" if self.absolute_rerank else "0.7 × 关键词覆盖（IDF 加权）+ 0.3 × 向量相似度（离线 / 词法精排模式）"
        doc_hits: dict[str, list[str]] = defaultdict(list)
        by_cat: dict[str, Counter] = defaultdict(Counter)

        seen: set[str] = set()
        for item in intents:
            text, category = (item, "未分类") if isinstance(item, str) else (str(item.get("question") or item.get("intent") or ""), str(item.get("category") or "未分类"))
            text = text.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            r = self._analyze(text, category, on_date)
            report.intents.append(r)
            report.buckets[r.bucket] += 1
            by_cat[category][r.bucket] += 1
            if r.bucket != "none" and r.top:
                doc_hits[r.top[0]["doc_id"]].append(text)

        report.n = len(report.intents)
        if report.n:
            report.coverage_rate = report.buckets["high"] / report.n
            report.answerable_rate = (report.buckets["high"] + report.buckets["partial"]) / report.n
        report.by_category = {c: {b: cnt.get(b, 0) for b in BUCKETS} for c, cnt in by_cat.items()}
        report.docs = self._docs(doc_hits, on_date)
        report.recommendations = self._recommend(report)
        report.seconds = time.perf_counter() - t0
        return report

    # ----------------------------------------------------------------------------------
    def _analyze(self, intent: str, category: str, on_date: date) -> IntentResult:
        result = self.retriever.retrieve(intent, top_k=self.top_k, use_transform=False, on_date=on_date)
        contexts = sorted(result.contexts, key=lambda c: c.score, reverse=True)
        terms = self._terms(intent)

        # 取关键词覆盖最高的那条上下文（并列时取排名靠前的）
        coverage, missing = 0.0, [t for t, _ in terms]
        for ctx in contexts:
            cov, miss = self._coverage(terms, (ctx.chunk.breadcrumb_text + "\n" + ctx.chunk.text).lower())
            if cov > coverage:
                coverage, missing = cov, miss
        dense = result.trace.routes.get("dense:q0") or []
        dense_sim = max(0.0, float(dense[0].score)) if dense else 0.0
        rerank = float(result.trace.reranked[0].score) if (self.absolute_rerank and result.trace.reranked) else None

        if rerank is not None:
            score = 0.6 * rerank + 0.4 * coverage
        else:
            score = 0.7 * coverage + 0.3 * min(1.0, dense_sim / 0.5)
        if not contexts:
            score = 0.0
        score = round(min(1.0, score), 4)
        bucket = "high" if score >= self.high else "partial" if score >= self.partial else "none"

        note = ""
        if bucket != "high" and self.retriever.cfg.filter_expired:
            # 只被过期版本覆盖：这不是"没有知识"，而是"知识没更新"
            stale = self.retriever.retrieve(intent, top_k=1, use_transform=False, on_date=on_date, overrides={"filter_expired": False})
            if stale.contexts:
                md = self.index.store.docs.get(stale.contexts[0].chunk.doc_id)
                if md is not None and not md.is_effective(on_date):
                    cov, _ = self._coverage(terms, (stale.contexts[0].chunk.breadcrumb_text + "\n" + stale.contexts[0].chunk.text).lower())
                    if cov > coverage + 0.15:
                        note = f"仅过期版本《{md.title or md.source}》覆盖该意图：需要更新现行版本"

        top = [
            {"chunk_id": c.chunk.chunk_id, "doc_id": c.chunk.doc_id, "breadcrumb": c.chunk.breadcrumb_text, "score": round(c.score, 4), "snippet": c.chunk.text[:160]}
            for c in contexts[: self.top_k]
        ]
        return IntentResult(
            intent=intent, category=category, score=score, bucket=bucket, bucket_label=BUCKET_LABEL[bucket], term_coverage=round(coverage, 4),
            dense_sim=round(dense_sim, 4), rerank_score=None if rerank is None else round(rerank, 4), missing_terms=missing[:6], top=top, action=BUCKET_ACTION[bucket], note=note,
        )

    # ----------------------------------------------------------------------------------
    def _build_idf(self) -> tuple[dict[str, float], float]:
        n = len(self.index.bm25.corpus_tokens)
        df: Counter = Counter()
        for toks in self.index.bm25.corpus_tokens:
            df.update(set(toks))
        idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        return idf, math.log(1.0 + (n + 0.5) / 0.5) if n else 1.0

    def _terms(self, intent: str) -> list[tuple[str, float]]:
        """意图 → [(关键词, 权重)]。精确 token（编号 / 错误码 / 型号）权重加倍；单字与功能词降权 / 剔除。"""
        precise = {m.group(0).lower() for m in PRECISE_TOKEN.finditer(intent)}
        out: dict[str, float] = {}
        for t in self.index.bm25.tokenize(intent):
            if t in _FUNCTION_WORDS or t in out:
                continue
            w = self._idf.get(t, self._max_idf)
            if t in precise:
                w *= 2.0
            elif len(t) == 1 and _is_cjk(t):
                w *= 0.3
            out[t] = w
        return list(out.items())

    @staticmethod
    def _coverage(terms: list[tuple[str, float]], text: str) -> tuple[float, list[str]]:
        total = sum(w for _, w in terms)
        if total <= 0:
            return 0.0, []
        got, missing = 0.0, []
        for t, w in terms:
            if t in text:
                got += w
            elif len(t) >= 3 and _is_cjk(t) and any(t[i:i + 2] in text for i in range(len(t) - 1)):
                got += 0.5 * w            # 分词边界不一致（"货运费" vs "运费"）给一半
            else:
                missing.append(t)
        return got / total, missing

    def _docs(self, doc_hits: dict[str, list[str]], on_date: date) -> list[DocCoverage]:
        chunk_n: Counter = Counter(c.doc_id for c in self.index.store.chunks.values() if c.kind == "child")
        out: list[DocCoverage] = []
        for doc_id, md in self.index.store.docs.items():
            hits = doc_hits.get(doc_id, [])
            expired = not md.is_effective(on_date)
            if expired:
                status, action = "expired", "已过期：下线或确认生效区间；如仍有效请修正 valid_to"
            elif len(hits) >= 3:
                status, action = "core", "核心语料：优先保证解析与切片质量，纳入回归评测"
            elif hits:
                status, action = "supporting", "辅助语料：保留"
            else:
                status, action = "unused", "未被任何意图命中：与 RAG 目的可能不相关，确认后移出语料（或补充对应意图）"
            out.append(DocCoverage(
                doc_id=doc_id, title=md.title or md.source.rsplit("/", 1)[-1], source=md.source, version=md.version, valid_to=md.valid_to.isoformat() if md.valid_to else None,
                expired=expired, chunks=chunk_n.get(doc_id, 0), hits=len(hits), hit_intents=hits[:5], status=status, action=action,
            ))
        order = {"unused": 0, "expired": 1, "supporting": 2, "core": 3}
        out.sort(key=lambda d: (order[d.status], -d.hits))
        return out

    def _recommend(self, report: CoverageReport) -> list[str]:
        rec: list[str] = []
        none = [r for r in report.intents if r.bucket == "none"]
        partial = [r for r in report.intents if r.bucket == "partial"]
        if none:
            gaps = Counter(t for r in none for t in r.missing_terms)
            rec.append(f"{len(none)} 条意图无匹配，知识缺口关键词：{'、'.join(t for t, _ in gaps.most_common(8)) or '—'} → 补充对应文档后重新入库")
        if partial:
            rec.append(f"{len(partial)} 条意图部分匹配 → 检查对应章节是否完整、版本是否最新、表格是否被正确解析")
        stale = [r for r in report.intents if r.note]
        if stale:
            rec.append(f"{len(stale)} 条意图只被过期版本覆盖 → 更新现行版本的对应章节")
        unused = [d for d in report.docs if d.status == "unused"]
        if unused:
            rec.append(f"{len(unused)} 份文档未被任何意图命中（{'、'.join(d.title for d in unused[:5])}）→ 确认与业务目标无关后移出语料")
        expired = [d for d in report.docs if d.status == "expired"]
        if expired:
            rec.append(f"{len(expired)} 份文档已过期（{'、'.join(d.title for d in expired[:5])}）→ 检索时已自动过滤，建议定期下线")
        if report.n and report.n < 30:
            rec.append(f"当前仅 {report.n} 条意图：建议从工单 / 搜索日志收集 ≥ 100 条真实提问（常见 50% / 长尾 30% / 多跳 20%）")
        if not rec:
            rec.append("意图与语料匹配良好：把这批意图标注答案后沉淀为黄金评估集")
        return rec


def _is_cjk(t: str) -> bool:
    return bool(re.fullmatch(r"[一-鿿]+", t))
