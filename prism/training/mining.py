"""用生产索引挖掘难负例，先排除已知相关片段，再做同尺度教师过滤。

这里只使用原问题的 dense/BM25 child 召回，不运行生成、查询改写、上下文扩展。
词法和排名分不能当绝对语义分；教师缺席或降级时必须人工审核。
synthetic_negatives > 0 时另让文本模型把正例改写成“似是而非”的负例，单独存放，人工筛选后才导出。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..eval.dataset import GoldenExample
from ..retrieval.fusion import rrf_fusion
from ..retrieval.retriever import HybridRetriever
from ..schemas import Chunk


class MiningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    candidate_pool: int = Field(100, ge=10, le=1000)
    num_negatives: int = Field(4, ge=1, le=16)
    rank_min: int = Field(1, ge=1, le=1000)
    rank_max: int = Field(100, ge=1, le=1000)
    max_per_doc: int = Field(2, ge=1, le=16)
    exclude_same_document: bool = True
    use_teacher: bool = True
    min_positive_score: float = Field(0.5, ge=0, le=1)
    max_negative_score: float = Field(0.8, ge=0, le=1)
    # 低于该教师分的候选属于简单负例，不进训练集；0 表示不过滤
    min_negative_score: float = Field(0.0, ge=0, le=1)
    # 每个问题让文本模型改写正例生成的负例数；0 表示不生成
    synthetic_negatives: int = Field(0, ge=0, le=4)
    absolute_margin: float = Field(0.1, ge=0, le=1)
    relative_margin: float = Field(0.05, ge=0, lt=1)
    dev_fraction: float = Field(0.2, ge=0, lt=1)
    seed: int = Field(42, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def valid_ranks(self) -> "MiningConfig":
        if self.rank_min > self.rank_max:
            raise ValueError("rank_min 不能大于 rank_max")
        if self.rank_max > self.candidate_pool:
            raise ValueError("rank_max 不能大于 candidate_pool")
        if self.min_negative_score > self.max_negative_score:
            raise ValueError("min_negative_score 不能大于 max_negative_score")
        return self


MAX_SUSPECTED = 5  # 每个问题在报告里保留的疑似漏标正例上限

# 只改同一事实的数值 / 日期得到的是“错误答案”，检索模型无从分辨，所以要求换成相邻的对象或环节。
SYNTHETIC_PROMPT = """下面是一个用户问题，以及知识库里能回答它的一段原文。

<问题>
{query}
</问题>
<原文>
{text}
</原文>

请参照原文写出 {n} 段“似是而非”的文字，用来训练检索模型区分“看起来相似、其实不回答问题”的内容。要求：
- 主题领域、用词、句式和长度都贴近原文，读起来像同一份文档里的另一段
- 讲的是相邻的另一件事：换成相近的对象、适用人群、业务环节或另一类问题，使它回答的是另一个相近的问题
- 不能回答上面的问题，也不能包含回答该问题所需的关键信息
- 不要照搬原文只替换其中一个名词，不要只改原文里的数字、日期或个别字词，也不要写成原文的否定或反驳
{avoid}- {n} 段之间互不相同

只输出 JSON 数组，每个元素是一段文字：["...", "..."]"""


def _parse_passages(raw: str) -> list[str]:
    match = re.search(r"\[.*\]", re.sub(r"```(?:json)?", "", raw), re.S)
    data = json.loads(match.group(0)) if match else None
    if not isinstance(data, list):
        raise ValueError("no json array")
    return [x.strip() for x in data if isinstance(x, str) and x.strip()]


def normalize_text(text: str) -> str:
    """同一个问题的 Unicode、大小写、空白、标点变体必须留在同一组。"""
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def _rerank_text(chunk: Chunk) -> str:
    return (f"{chunk.breadcrumb_text}\n" if chunk.breadcrumbs else "") + chunk.text


def _in_scope(example: GoldenExample, chunk: Chunk) -> bool:
    if not example.relevant_doc_ids and not example.relevant_sources:
        return True
    source = str(chunk.metadata.get("source") or "")
    return chunk.doc_id in example.relevant_doc_ids or any(
        source.endswith(s) or Path(source).name == Path(s).name
        for s in example.relevant_sources if s.strip()
    )


def _positive(example: GoldenExample, chunk: Chunk) -> bool:
    # 训练比检索评测更严格：精确标签优先；关键词限定到指定来源，且必须在 child 自身。
    if example.relevant_chunk_ids:
        if chunk.chunk_id in example.relevant_chunk_ids:
            return True
        if chunk.parent_id not in example.relevant_chunk_ids:
            return False
        keywords = [k for k in example.answer_keywords if k.strip()]
        return not keywords or any(k.casefold() in chunk.text.casefold() for k in keywords)
    return _in_scope(example, chunk) and any(
        keyword.strip() and keyword.casefold() in chunk.text.casefold()
        for keyword in example.answer_keywords
    )


def _similar(left: str, right: str) -> bool:
    a, b = normalize_text(left), normalize_text(right)
    if not a or not b:
        return True  # 空正文、纯标点不能提供有效监督
    if a == b:
        return True
    # 重叠切片可能是包含关系，尤其跨 parent 时。长度短于 20 的子串不够可靠。
    if min(len(a), len(b)) >= 20 and (a in b or b in a):
        return True
    if min(len(a), len(b)) / max(len(a), len(b)) < 0.9:
        return False
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    return matcher.quick_ratio() >= 0.9 and matcher.ratio() >= 0.9


@dataclass
class MiningReport:
    config: dict[str, Any]
    stats: dict[str, int]
    items: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    models: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.config, "stats": self.stats, "items": self.items,
                "skipped": self.skipped, "warnings": self.warnings, "models": self.models}


class HardNegativeMiner:
    def __init__(self, retriever: HybridRetriever, config: MiningConfig | None = None, llm: Any = None) -> None:
        self.retriever = retriever
        self.index = retriever.index
        self.config = config or MiningConfig()
        self.llm = llm  # 仅用于 synthetic_negatives；召回阶段不使用

    def run(self, examples: list[GoldenExample], *, held_out: list[GoldenExample] | None = None) -> MiningReport:
        cfg = self.config
        report = MiningReport(config=cfg.model_dump(), stats={})
        held_queries = {normalize_text(e.question) for e in (held_out or [])}
        groups: dict[str, list[GoldenExample]] = {}
        for example in examples:
            groups.setdefault(normalize_text(example.question), []).append(example)
        allowed = self.retriever._allowed_ids(None, None, self.retriever.cfg)
        children = sorted((c for c in self.index.store.chunks.values()
                           if c.kind == "child" and (allowed is None or c.chunk_id in allowed)),
                          key=lambda c: c.chunk_id)
        teacher = self.retriever.reranker
        report.models = {**self.index.signature(), "teacher": teacher.name,
                         "teacherModel": getattr(teacher, "model_name", None),
                         "teacherRequested": cfg.use_teacher, "teacherFiltered": 0,
                         "syntheticModel": getattr(self.llm, "model", None) if cfg.synthetic_negatives else None,
                         "retrievalConfig": hashlib.sha256(self.retriever.cfg.model_dump_json().encode()).hexdigest()[:12]}
        if self.index.embedder.name == "hash":
            report.warnings.append("哈希向量仅适合流程自检；难度排序需要真实嵌入模型验证。")
        if cfg.synthetic_negatives and self.llm is None:
            report.warnings.append("没有可用的文本模型，本次未生成改写负例。")
        if held_out:
            report.warnings.append("独立评测问题仅按 Unicode、大小写、空白和标点归一化排除；同义改写仍需人工分组。")
        for key, labels in groups.items():
            query = labels[0].question.strip()
            reason = None
            if not key:
                reason = "empty_question"
            elif key in held_queries:
                reason = "held_out_question"
            elif any(e.expected_refusal for e in labels):
                reason = "expected_refusal"
            elif not any(e.relevant_chunk_ids or any(k.strip() for k in e.answer_keywords) for e in labels):
                reason = "precise_labels_required"
            if reason:
                report.skipped.append({"question": query, "reason": reason})
                continue
            positives = [c for c in children if any(_positive(e, c) for e in labels)]
            if not positives:
                report.skipped.append({"question": query, "reason": "no_positive_in_active_index"})
                continue
            positive_parents = {c.parent_id or c.chunk_id for c in positives}
            positive_docs = {c.doc_id for c in positives}
            # 关键词在 parent 内命中时，即使不在某个 child 内，也不能把那个 child 当负例。
            for c in children:
                parent = self.index.store.parent_of(c.chunk_id)
                if parent is not None and any(_positive(e, parent) for e in labels):
                    positive_parents.add(parent.chunk_id)
            dense = self.retriever.retrieve_dense(query, cfg.candidate_pool)
            sparse = self.retriever.retrieve_sparse(query, cfg.candidate_pool)
            fusion = self.retriever.cfg.fusion
            fused = rrf_fusion([(dense, fusion.weight_dense), (sparse, fusion.weight_sparse)], k=fusion.rrf_k)
            ranks = {cid: rank for rank, (cid, _) in enumerate(fused, 1)}
            positives.sort(key=lambda c: (ranks.get(c.chunk_id, cfg.candidate_pool * 2 + 1), c.chunk_id))
            selected_pos = positives
            candidate_chunks: list[Chunk] = []
            for rank, (cid, _) in enumerate(fused[:cfg.rank_max], 1):
                if rank < cfg.rank_min:
                    continue
                c = self.index.store.get(cid)
                if c is None or c.kind != "child" or (allowed is not None and cid not in allowed):
                    continue
                if (c.parent_id or cid) in positive_parents or (cfg.exclude_same_document and c.doc_id in positive_docs):
                    continue
                if any(_similar(c.text, p.text) for p in positives):
                    continue
                candidate_chunks.append(c)
            scores = self._teacher_scores(query, selected_pos + candidate_chunks, report)
            if scores is not None:
                selected_pos = [c for c in selected_pos if scores[c.chunk_id] >= cfg.min_positive_score]
                if not selected_pos:
                    report.skipped.append({"question": query, "reason": "low_positive_score"})
                    continue
                selected_pos.sort(key=lambda c: (-scores[c.chunk_id], ranks.get(c.chunk_id, cfg.candidate_pool * 2 + 1), c.chunk_id))
            selected_pos = selected_pos[:3]
            suspected: list[dict[str, Any]] = []
            easy = 0
            cutoff = None
            if scores is not None:
                # 对每个导出的 positive 都满足 margin，因此使用所选正例里的最低分。
                pos_score = min(scores[c.chunk_id] for c in selected_pos)
                cutoff = min(cfg.max_negative_score, pos_score - cfg.absolute_margin,
                             pos_score - abs(pos_score) * cfg.relative_margin)
                candidate_chunks.sort(key=lambda c: (-scores[c.chunk_id], ranks[c.chunk_id]))
                # 教师分贴近或超过正例的候选可能是漏标的正确答案：不当负例，留给人工补标。
                suspected = [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "score": scores[c.chunk_id],
                              "rank": ranks[c.chunk_id], "text": _rerank_text(c)}
                             for c in candidate_chunks if scores[c.chunk_id] > cutoff]
                easy = sum(scores[c.chunk_id] < cfg.min_negative_score for c in candidate_chunks)
                candidate_chunks = [c for c in candidate_chunks
                                    if cfg.min_negative_score <= scores[c.chunk_id] <= cutoff]
            negatives: list[Chunk] = []
            doc_counts: Counter[str] = Counter()
            seen_parents: set[str] = set()
            for c in candidate_chunks:
                pid = c.parent_id or c.chunk_id
                if pid in seen_parents or doc_counts[c.doc_id] >= cfg.max_per_doc:
                    continue
                if any(_similar(c.text, other.text) for other in negatives):
                    continue
                negatives.append(c)
                seen_parents.add(pid)
                doc_counts[c.doc_id] += 1
                if len(negatives) == cfg.num_negatives:
                    break
            synthetic, rejected = self._synthesize(query, labels, selected_pos, positives, negatives,
                                                   cutoff, report) if cfg.synthetic_negatives and self.llm is not None else ([], {})
            if not negatives and not synthetic:
                report.skipped.append({"question": query, "reason": "no_safe_negatives",
                                       "suspected_positives": suspected[:MAX_SUSPECTED], "easy_filtered": easy})
                continue
            split = "dev" if int(hashlib.sha256(f"{cfg.seed}:{key}".encode()).hexdigest()[:16], 16) / 2**64 < cfg.dev_fraction else "train"
            report.items.append({
                "query": query, "category": labels[0].category, "split": split,
                "pos": [c.embed_text() for c in selected_pos], "neg": [c.embed_text() for c in negatives],
                "rerank_pos": [_rerank_text(c) for c in selected_pos], "rerank_neg": [_rerank_text(c) for c in negatives],
                "positive_chunk_ids": [c.chunk_id for c in selected_pos], "negative_chunk_ids": [c.chunk_id for c in negatives],
                "positive_doc_ids": [c.doc_id for c in selected_pos], "negative_doc_ids": [c.doc_id for c in negatives],
                "pos_scores": [scores[c.chunk_id] for c in selected_pos] if scores is not None else [],
                "neg_scores": [scores[c.chunk_id] for c in negatives] if scores is not None else [],
                "negative_routes": [[name for name, route in (("dense", dense), ("sparse", sparse)) if any(cid == c.chunk_id for cid, _ in route)] for c in negatives],
                "negative_ranks": [ranks[c.chunk_id] for c in negatives],
                "requires_review": scores is None or any(not any(
                    c.chunk_id in e.relevant_chunk_ids or (
                        any(k.strip() and k.casefold() in c.text.casefold() for k in e.answer_keywords)
                        and _positive(e, c)
                    ) for e in labels) for c in selected_pos),
                "shortage": max(0, cfg.num_negatives - len(negatives)),
                "suspected_positives": suspected[:MAX_SUSPECTED],
                "suspected_positive_count": len(suspected), "easy_filtered": easy,
                "synthetic_neg": synthetic, "synthetic_rejected": rejected,
            })
        report.stats = {"input_examples": len(examples), "queries": len(groups), "items": len(report.items),
                        "negatives": sum(len(i["neg"]) for i in report.items),
                        "train": sum(i["split"] == "train" for i in report.items),
                        "dev": sum(i["split"] == "dev" for i in report.items),
                        "review_required": sum(i["requires_review"] for i in report.items),
                        "shortages": sum(i["shortage"] for i in report.items),
                        "suspected_positives": sum(i["suspected_positive_count"] for i in report.items),
                        "easy_filtered": sum(i["easy_filtered"] for i in report.items),
                        "synthetic_negatives": sum(len(i["synthetic_neg"]) for i in report.items),
                        "synthetic_rejected": sum(sum(i["synthetic_rejected"].values()) for i in report.items)}
        if report.stats["synthetic_negatives"]:
            report.warnings.append("改写负例是文本模型编写的，不是知识库原文，可能仍能回答问题或与事实相悖；逐条人工筛选后才能导出。")
        if report.stats["suspected_positives"]:
            report.warnings.append("部分候选的教师分贴近或超过正例，已从负例中剔除并列为疑似漏标正例；确实能回答问题的请补进 relevant_chunk_ids 后重新挖掘。")
        if report.stats["items"] and not report.stats["dev"] and cfg.dev_fraction:
            report.warnings.append("本次数据量小，稳定哈希划分没有产生 dev 样例；请补充独立验证数据。")
        report.warnings.append("教师分差过滤不能保证负例无误；请复核高分候选和关键词弱标注。train/dev 按问题分组，共享检索语料，未保证文档或同义问题隔离。")
        return report

    def _synthesize(self, query: str, labels: list[GoldenExample], selected_pos: list[Chunk], positives: list[Chunk],
                    negatives: list[Chunk], cutoff: float | None, report: MiningReport) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """改写教师分最高的正例；生成结果过关键词、近重复和同尺度教师过滤，仍需人工筛选。"""
        cfg = self.config
        source = selected_pos[0]
        rejected: Counter[str] = Counter()
        keywords = list(dict.fromkeys(k.strip() for e in labels for k in e.answer_keywords if k.strip()))
        avoid = f"- 下面这些词语是该问题答案的关键信息，写出的文字里不要出现：{'、'.join(keywords)}\n" if keywords else ""
        try:
            raw = self.llm.complete(SYNTHETIC_PROMPT.format(query=query, text=source.text[:1500], n=cfg.synthetic_negatives, avoid=avoid),
                                    max_tokens=300 + 900 * cfg.synthetic_negatives)
            passages = _parse_passages(raw)[:cfg.synthetic_negatives]
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            self._warn(report, f"部分问题的改写负例生成失败：{type(exc).__name__}" + (f"（HTTP {status}）" if isinstance(status, int) else ""))
            return [], {"generation_failed": 1}
        # 最末一级标题说的是正例自己的主题（FAQ 块里就是问题本身），套在改写文本上会让教师误判为相关。
        crumbs = source.breadcrumbs[:-1] if len(source.breadcrumbs) > 1 else source.breadcrumbs
        kept: list[Chunk] = []
        for body in passages:
            if any(k.casefold() in body.casefold() for k in keywords):
                rejected["contains_answer_keyword"] += 1
            elif any(_similar(body, c.text) for c in positives + negatives + kept):
                rejected["near_copy"] += 1
            else:
                # 去掉为正例生成的关键词 / 假设性问题，避免把正例的检索线索带给负例。
                kept.append(source.model_copy(update={
                    "chunk_id": f"synthetic:{len(kept)}", "text": body, "breadcrumbs": crumbs,
                    "child_ids": [], "keywords": [], "aliases": [],
                    "context_summary": None, "hypothetical_questions": [], "summary": None}))
        scores: dict[str, float] | None = None
        if cutoff is not None and kept:
            scores = self._teacher_scores(query, kept, report, count=False)
            if scores is None:  # 这一组的正负例已过教师过滤，不能混入未打分的生成文本
                rejected["teacher_unavailable"] += len(kept)
                kept = []
            for c in list(kept):
                reason = ("teacher_too_close" if scores[c.chunk_id] > cutoff
                          else "easy" if scores[c.chunk_id] < cfg.min_negative_score else None)
                if reason:
                    rejected[reason] += 1
                    kept.remove(c)
        return [{"text": c.embed_text(), "rerank_text": _rerank_text(c), "body": c.text,
                 "score": scores[c.chunk_id] if scores else None, "source_chunk_id": source.chunk_id}
                for c in kept], dict(rejected)

    def _teacher_scores(self, query: str, chunks: list[Chunk], report: MiningReport, *, count: bool = True) -> dict[str, float] | None:
        if not self.config.use_teacher:
            return None
        rr = self.retriever.reranker
        # LLMReranker 会静默回退到 lexical，不能确认本次分数来源。
        if rr.name not in {"cross_encoder", "api", "cohere"}:
            self._warn(report, "语义教师不可用：当前重排器不是可验证的 Cross-Encoder / Rerank API，候选需人工审核。")
            return None
        try:
            status = getattr(rr, "status", None)
            before = status() if callable(status) else {}
            pop_note = getattr(rr, "pop_note", None)
            if callable(pop_note):
                pop_note()  # 清掉此前查询的旧提示，只记录本次打分
            result = rr.rerank(query, [(c.chunk_id, _rerank_text(c)) for c in chunks], len(chunks))
            state = status() if callable(status) else {}
            report.models["teacherState"] = state.get("state")
            note = pop_note() if callable(pop_note) else None
            shared_fallback = rr.name == "cross_encoder" and getattr(rr, "fallback", None) is not None
            if note or (state and state.get("state") != "ready") or (shared_fallback and before.get("state") != "ready"):
                self._warn(report, f"语义教师降级，本次候选需人工审核：{note or state.get('state')}")
                return None
            scores = dict(result)
            expected = {c.chunk_id for c in chunks}
            if set(scores) != expected or len(result) != len(expected) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores.values()):
                self._warn(report, "语义教师未返回完整、有限的 0..1 分数，本次候选需人工审核。")
                return None
            report.models["teacherFiltered"] += count
            return scores
        except Exception as exc:
            self._warn(report, f"语义教师打分失败，本次候选需人工审核：{type(exc).__name__}")
            return None

    @staticmethod
    def _warn(report: MiningReport, warning: str) -> None:
        if warning not in report.warnings:
            report.warnings.append(warning)


def export_training(report: MiningReport, directory: Path, *, allow_unverified: bool = False,
                    include_synthetic: bool = False) -> dict[str, int]:
    """独立目录原子创建，避免覆盖已有数据；审核标记不会悄悄转成训练标签。

    改写负例只在 include_synthetic 时并入训练文件；只有改写负例的问题在此之前不会导出。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    counts: dict[str, int] = {}
    for split in ("train", "dev"):
        rows = [(i, i["synthetic_neg"] if include_synthetic else []) for i in report.items
                if i["split"] == split and (allow_unverified or not i["requires_review"])]
        rows = [(i, extra) for i, extra in rows if i["neg"] or extra]
        # 教师分随 FlagEmbedding 的 pos_scores / neg_scores 字段导出，供 --distill-weight 做 label 层蒸馏
        records = [{"query": i["query"], "pos": i["pos"], "neg": i["neg"] + [s["text"] for s in extra],
                    **({"pos_scores": i["pos_scores"], "neg_scores": i["neg_scores"] + [s["score"] for s in extra]}
                       if i["pos_scores"] else {})}
                   for i, extra in rows]
        pairs = [{"query": i["query"], "passage": text, "label": label}
                 for i, extra in rows
                 for label, texts in ((1, i["rerank_pos"]), (0, i["rerank_neg"] + [s["rerank_text"] for s in extra])) for text in texts]
        for filename, data in ((f"{split}.jsonl", records), (f"reranker.{split}.jsonl", pairs)):
            (directory / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")
            counts[filename] = len(data)
    manifest = {**report.to_dict(), "export": {"allow_unverified": allow_unverified, "include_synthetic": include_synthetic, "counts": counts}}
    (directory / "mining-report.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts
