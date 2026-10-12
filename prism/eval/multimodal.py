"""多模态评测：传统 Recall@K 只看文本块，衡量不了"图找得对不对""回答带的图对不对"。

四类任务：
- text2image   文搜图：Hit@K / MRR，并按召回路消融（图文向量 / 描述向量 / 描述 BM25 / 融合）
- image2image  图搜图：Hit@K / MRR（查询图默认是库里那张图缩小 + JPEG 压缩，模拟截图 / 转发后的图）
- image2text   图搜文：期望关键词出现在前 K 个正文段落里的比例
- answer       回答附图：附图的精确率 / 召回率；负例（与图无关的问题）不附图的比例

样例里指认图片的方式（任选）：image_id；图注片段；"文件名#p页码"（页码从 1 开始）。
查询图片：data URL / 裸 base64 / 本地路径 / "kb:<指认方式>"（引用库里的图，缩小压缩）/ "photo:<指认方式>"（模拟手机翻拍）。
"""

from __future__ import annotations

import io
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..multimodal.extract import decode_image_b64
from ..multimodal.retriever import MultimodalRetriever
from ..multimodal.schemas import ImageRecord

Task = Literal["text2image", "image2image", "image2text", "answer"]
TASK_LABEL = {"text2image": "文搜图", "image2image": "图搜图", "image2text": "图搜文", "answer": "回答附图"}
_FIG_PREFIX = re.compile(r"^\s*(?:图|圖|图表|Figure|Fig\.?|Chart)\s*[\dA-Za-z一二三四五六七八九十]+[\s.．:：、\-—]*", re.I)
NEGATIVE_QUESTIONS = ["公司的年假怎么休", "出差住宿可以报销多少", "请假需要提前几天申请"]


class MultimodalExample(BaseModel):
    task: Task = "text2image"
    question: str = ""
    query_image: str | None = None
    relevant_images: list[str] = Field(default_factory=list)
    relevant_keywords: list[str] = Field(default_factory=list)
    expect_no_image: bool = False
    category: str = ""


@dataclass
class MultimodalExampleResult:
    task: str
    question: str
    category: str
    ok: bool
    rank: int | None = None                      # 第一个相关结果的名次（1 起）；没命中为 None
    metrics: dict[str, float] = field(default_factory=dict)
    top: list[str] = field(default_factory=list) # 前几名的可读标签
    note: str = ""


@dataclass
class MultimodalEvalReport:
    n: int
    ks: list[int]
    task_metrics: dict[str, dict[str, float]]
    route_metrics: dict[str, dict[str, float]]   # 文搜图按召回路消融
    examples: list[MultimodalExampleResult]
    seconds: float
    embedder: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["markdown"] = self.to_markdown()
        return d

    def to_markdown(self) -> str:
        lines = [f"# 多模态评测（{self.n} 条，{self.seconds:.1f}s）", "", f"图文向量模型：{self.embedder.get('model') or self.embedder.get('embedder')}", ""]
        for task, m in self.task_metrics.items():
            lines.append(f"## {TASK_LABEL.get(task, task)}（{int(m.get('n', 0))} 条）")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("| --- | --- |")
            lines += [f"| {k} | {v:.3f} |" for k, v in m.items() if k != "n"]
            lines.append("")
        if self.route_metrics:
            lines += ["## 文搜图消融（各召回路单独排序）", "", "| 路 | " + " | ".join(f"hit@{k}" for k in self.ks) + " |", "| --- | " + " | ".join("---" for _ in self.ks) + " |"]
            for route, m in self.route_metrics.items():
                lines.append(f"| {route} | " + " | ".join(f"{m.get(f'hit@{k}', 0):.3f}" for k in self.ks) + " |")
            lines.append("")
        bad = [e for e in self.examples if not e.ok]
        if bad:
            lines += ["## 未通过的样例", ""] + [f"- [{TASK_LABEL.get(e.task, e.task)}] {e.question or '(图片)'} → {e.note}" for e in bad]
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
def match_image(rec: ImageRecord, spec: str) -> bool:
    spec = spec.strip()
    if not spec:
        return False
    if spec == rec.image_id:
        return True
    m = re.match(r"^(.*)#p(\d+)$", spec)
    if m:
        name, page = m.group(1).strip(), int(m.group(2)) - 1
        return rec.page == page and (not name or name in (rec.doc_title or "") or name in rec.doc_id)
    return spec in rec.caption or (len(spec) >= 6 and spec in rec.ocr_text)


def perturb(image: Any, kind: str = "light") -> Any:
    """light：缩小到 60% + JPEG 70——查询图不能和库里的像素完全一样，否则图搜图是在测"复制粘贴"。
    photo：模拟手机翻拍——桌面背景、透视倾斜、一侧偏暗的光照、噪声、轻微模糊，再走 light 的压缩。"""
    from PIL import Image, ImageFilter

    im = image.convert("RGB")
    if kind == "photo":
        import numpy as np

        w, h = im.size
        pad = Image.new("RGB", (int(w * 1.3), int(h * 1.3)), (120, 110, 100))
        pad.paste(im, (int(w * 0.15), int(h * 0.15)))
        W, H = pad.size
        src = [(0, 0), (W, 0), (W, H), (0, H)]
        dst = [(W * 0.06, H * 0.03), (W * 0.98, 0), (W, H), (W * 0.024, H * 0.94)]
        rows: list[list[float]] = []
        for (x, y), (u, v) in zip(src, dst):
            rows += [[u, v, 1, 0, 0, 0, -x * u, -x * v], [0, 0, 0, u, v, 1, -y * u, -y * v]]
        coeffs = np.linalg.solve(np.array(rows, float), np.array(src, float).reshape(8))
        arr = np.asarray(pad.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC)).astype(np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        arr = arr * (0.75 + 0.35 * (xx / W) * (1 - 0.3 * yy / H))[..., None] + np.random.default_rng(0).normal(0, 6, arr.shape)
        im = Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(0.9))
    im = im.resize((max(1, int(im.width * 0.6)), max(1, int(im.height * 0.6))))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


class MultimodalEvalRunner:
    def __init__(self, retriever: MultimodalRetriever, hybrid=None, *, ks: list[int] | None = None) -> None:
        self.r = retriever
        self.images = retriever.images
        self.hybrid = hybrid                       # HybridRetriever：answer 任务需要文本检索的上下文
        self.ks = sorted(set(ks or [1, 3, 5]))

    # ----------------------------------------------------------------------------------
    def run(self, examples: list[MultimodalExample]) -> MultimodalEvalReport:
        t0 = time.perf_counter()
        results: list[MultimodalExampleResult] = []
        route_hits: dict[str, list[int | None]] = {}
        for ex in examples:
            try:
                res = getattr(self, f"_{ex.task}")(ex, route_hits)
            except Exception as exc:  # 单条样例出错不影响整份报告
                res = MultimodalExampleResult(task=ex.task, question=ex.question, category=ex.category, ok=False, note=f"出错：{type(exc).__name__}: {exc}")
            results.append(res)
        task_metrics = {task: self._aggregate(task, [r for r in results if r.task == task]) for task in TASK_LABEL if any(r.task == task for r in results)}
        route_metrics = {route: {f"hit@{k}": sum(1 for x in ranks if x is not None and x <= k) / len(ranks) for k in self.ks} for route, ranks in route_hits.items() if ranks}
        return MultimodalEvalReport(n=len(results), ks=self.ks, task_metrics=task_metrics, route_metrics=route_metrics, examples=results, seconds=time.perf_counter() - t0, embedder=self.images.embedder.signature())

    def _aggregate(self, task: str, rows: list[MultimodalExampleResult]) -> dict[str, float]:
        n = len(rows)
        out: dict[str, float] = {"n": float(n)}
        if task in ("text2image", "image2image", "image2text"):
            for k in self.ks:
                out[f"hit@{k}"] = sum(1 for r in rows if r.rank is not None and r.rank <= k) / n
            out["mrr"] = sum(1.0 / r.rank for r in rows if r.rank) / n
        else:
            pos = [r for r in rows if "recall" in r.metrics]
            neg = [r for r in rows if "no_image_ok" in r.metrics]
            if pos:
                out["附图精确率"] = sum(r.metrics["precision"] for r in pos) / len(pos)
                out["附图召回率"] = sum(r.metrics["recall"] for r in pos) / len(pos)
            if neg:
                out["负例不附图率"] = sum(r.metrics["no_image_ok"] for r in neg) / len(neg)
        return out

    # ----------------------------------------------------------------------------------
    def _relevant(self, ex: MultimodalExample) -> set[str]:
        return {rec.image_id for rec in self.images.records.values() if any(match_image(rec, s) for s in ex.relevant_images)}

    def _label(self, image_id: str) -> str:
        rec = self.images.get(image_id)
        if rec is None:
            return image_id
        page = f" p{rec.page + 1}" if rec.page is not None else ""
        return (rec.caption or rec.brief(24) or rec.doc_title) + page

    def _query_image(self, ex: MultimodalExample) -> Any:
        src = (ex.query_image or "").strip()
        if not src:
            raise ValueError("缺少查询图片")
        if src.startswith(("kb:", "photo:")):
            kind, spec = src.split(":", 1)
            rec = next((r for r in self.images.records.values() if match_image(r, spec)), None)
            if rec is None:
                raise ValueError(f"库里找不到图片：{spec}")
            return perturb(self.images.load_image(rec), "photo" if kind == "photo" else "light")
        if src.startswith("data:") or len(src) > 260:
            return decode_image_b64(src)
        from PIL import Image

        return Image.open(src).convert("RGB")

    def _ranked(self, ex: MultimodalExample, ranking: list[str], relevant: set[str]) -> MultimodalExampleResult:
        rank = next((i + 1 for i, x in enumerate(ranking) if x in relevant), None)
        ok = rank is not None and rank <= self.ks[0]
        note = "" if ok else (f"第 {rank} 名才命中" if rank else "前 %d 名都没有命中" % len(ranking)) + f"（期望：{'、'.join(self._label(i) for i in relevant) or '未匹配到任何图片，检查样例'}）"
        return MultimodalExampleResult(task=ex.task, question=ex.question, category=ex.category, ok=ok, rank=rank, top=[self._label(i) for i in ranking[:3]], note=note)

    def _text2image(self, ex: MultimodalExample, route_hits: dict[str, list[int | None]]) -> MultimodalExampleResult:
        relevant = self._relevant(ex)
        res = self.r.text_to_image(ex.question, top_k=max(self.ks))
        for route, items in res.routes.items():   # 消融：每一路单独排序时第一个相关图的名次
            ids = [x.chunk_id for x in items]
            route_hits.setdefault(route, []).append(next((i + 1 for i, x in enumerate(ids) if x in relevant), None))
        route_hits.setdefault("fused", []).append(next((i + 1 for i, h in enumerate(res.images) if h.record.image_id in relevant), None))
        return self._ranked(ex, [h.record.image_id for h in res.images], relevant)

    def _image2image(self, ex: MultimodalExample, route_hits: dict[str, list[int | None]]) -> MultimodalExampleResult:
        relevant = self._relevant(ex)
        res = self.r.image_to_image(self._query_image(ex), top_k=max(self.ks), min_score=0.0)
        return self._ranked(ex, [h.record.image_id for h in res.images], relevant)

    def _image2text(self, ex: MultimodalExample, route_hits: dict[str, list[int | None]]) -> MultimodalExampleResult:
        res = self.r.image_to_text(self._query_image(ex), query=ex.question, top_k=max(self.ks))
        kws = [k for k in ex.relevant_keywords if k.strip()]
        rank = next((i + 1 for i, c in enumerate(res.contexts) if kws and all(k.lower() in c.chunk.text.lower() for k in kws)), None)  # 英文分词结果是小写
        ok = rank is not None and rank <= self.ks[0]
        top = [c.chunk.breadcrumb_text[-30:] for c in res.contexts[:3]]
        note = "" if ok else (f"第 {rank} 个段落才包含关键词" if rank else f"前 {len(res.contexts)} 个段落都不包含 {kws}") + (f"；{res.notes[-1]}" if res.notes else "")
        return MultimodalExampleResult(task=ex.task, question=ex.question, category=ex.category, ok=ok, rank=rank, top=top, note=note)

    def _answer(self, ex: MultimodalExample, route_hits: dict[str, list[int | None]]) -> MultimodalExampleResult:
        contexts = self.hybrid.retrieve(ex.question, use_transform=False).contexts if self.hybrid is not None else []
        hits, _ = self.r.images_for_answer(ex.question, contexts)
        attached = [h.record.image_id for h in hits]
        top = [self._label(i) for i in attached]
        if ex.expect_no_image:
            ok = not attached
            return MultimodalExampleResult(task=ex.task, question=ex.question, category=ex.category or "负例", ok=ok, metrics={"no_image_ok": float(ok)}, top=top, note="" if ok else f"不该附图却附了 {len(attached)} 张")
        relevant = self._relevant(ex)
        good = len(set(attached) & relevant)
        precision = good / len(attached) if attached else 0.0
        recall = good / len(relevant) if relevant else 0.0
        ok = recall > 0 and precision >= 0.5
        note = "" if ok else ("没有附图" if not attached else f"附了 {len(attached)} 张，其中相关 {good} 张") + f"（期望：{'、'.join(self._label(i) for i in relevant)}）"
        return MultimodalExampleResult(task=ex.task, question=ex.question, category=ex.category, ok=ok, metrics={"precision": precision, "recall": recall}, top=top, note=note)


# --------------------------------------------------------------------------------------
def build_samples(images, store, *, max_per_task: int = 20) -> list[MultimodalExample]:
    """从图片库自动生成样例：图注 / OCR 当文搜图问题，图片自身（加扰动）当图搜图 / 图搜文查询，
    再加几条与图无关的负例。属于"自检"——能发现解析、索引、阈值上的明显问题；真实效果仍要靠人工整理的问答对。"""
    from ..index.bm25_store import BM25Store

    out: list[MultimodalExample] = []
    recs = sorted(images.records.values(), key=lambda r: (r.doc_id, r.order))[:max_per_task]
    tokenize = BM25Store(images.sparse_cfg).tokenize
    # 关键词按"在多少个正文块里出现过"从少到多挑：英文文档里 "and""to" 这种词在所在段落里一定出现，
    # 拿它当期望关键词等于没有考题（CLIP 论文的自动样例实测选出过 ["and", "to"]）
    df = Counter(t for cid in store.child_ids() for t in set(tokenize(store.chunks[cid].text)))
    usable = lambda t: len(t) >= (3 if t.isascii() else 2) and not t.isdigit()  # noqa: E731
    for rec in recs:
        core = _FIG_PREFIX.sub("", rec.caption).strip()
        ref = rec.image_id
        if core:
            out.append(MultimodalExample(task="text2image", question=core, relevant_images=[ref], category="图注"))
            out.append(MultimodalExample(task="answer", question=f"{core}是怎样的？", relevant_images=[ref], category="图注"))
        elif rec.ocr_text:
            words = [t for t in tokenize(rec.ocr_text) if len(t) >= 2][:4]
            if words:
                out.append(MultimodalExample(task="text2image", question=" ".join(words), relevant_images=[ref], category="图中文字"))
        out.append(MultimodalExample(task="image2image", query_image=f"kb:{ref}", relevant_images=[ref], category="同图扰动"))
        out.append(MultimodalExample(task="image2image", query_image=f"photo:{ref}", relevant_images=[ref], category="模拟翻拍"))
        anchor = store.get(rec.anchor_chunk_id) if rec.anchor_chunk_id else None
        if anchor is not None:
            # 关键词优先取"这张图自己的"词（图注 / 图中文字）且在所在段落里出现过，避免几张图共用一个段落时样例全都一样
            text = anchor.text.lower()
            own = sorted(dict.fromkeys(t for t in tokenize(f"{core} {rec.ocr_text}") if usable(t) and t in text), key=lambda t: df.get(t, 0))
            kws = own[:2] or sorted(dict.fromkeys(t for t in tokenize(anchor.text) if usable(t)), key=lambda t: df.get(t, 0))[:2]
            if kws:
                out.append(MultimodalExample(task="image2text", query_image=f"kb:{ref}", relevant_keywords=kws, category="所在段落"))
    for q in NEGATIVE_QUESTIONS:
        out.append(MultimodalExample(task="answer", question=q, expect_no_image=True, category="负例"))
    return out
