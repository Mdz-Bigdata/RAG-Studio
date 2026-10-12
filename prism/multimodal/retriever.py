"""多模态检索器：文搜图 / 图搜图 / 图文混合 / 图搜文。

    文搜图   text ─┬─ 图文向量路   CLIP 文本塔 → 图片向量            （语义："一张柱状图"）
                   ├─ 描述向量路   文本向量 → 图片文本摘要向量        （VLM 描述 / 图注 / 上下文）
                   └─ 描述 BM25 路 关键词 → 图片文本摘要               （OCR 文字 / 编号 / 数值精确匹配）
                   → RRF 融合
    图搜图   image ── 图像塔 → 图片向量（余弦相似度，可设阈值）
    图文混合 text + image → score = α·z(text·I) + (1-α)·z(image·I) 在图文空间统一检索，再与描述两路按 α 加权合并
    图搜文   image ─┬─ 图文向量直达  图像塔 → 正文块的 CLIP 文本向量
                    ├─ 相似图片带出  相似图片所在的段落
                    └─ 描述再检索    OCR / VLM 把查询图转成文字 → 正文的向量 + BM25
                    → RRF 融合 → Small-to-Big
    LanceDB 多模态表（multimodal.lance_table，图片与正文同一张表）：文搜图多一路表内图片文字的全文检索（ICU 分词），
                    图搜文多一路正文行上"查询图 CLIP 向量 + 图中文字"的原生混合检索，都参与上面的 RRF

没有跨模态模型时（离线感知特征）：图文向量路与直达路自动跳过，其余各路照常工作。
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import numpy as np

from ..config import MultimodalConfig, Settings
from ..index import Index
from ..index.bm25_store import PRECISE_TOKEN
from ..retrieval.fusion import rrf_fusion
from ..retrieval.granularity import expand_contexts
from ..schemas import Context, ScoredChunk
from .describe import ImageDescriber
from .schemas import ImageHit, ImageRecord, MultimodalResult, RouteHit

logger = logging.getLogger(__name__)

# 显式"要图"的提问：此时不做相关性门槛过滤，直接返回融合后的 top-K
_WANTS_IMAGE = re.compile(r"图片|图表|图像|照片|截图|示意图|流程图|架构图|柱状图|折线图|饼图|插图|配图|出示|长什么样|什么样子|找.{0,6}图|有.{0,4}图|\bimages?\b|\bpictures?\b|\bphotos?\b|\bcharts?\b|\bfigures?\b|\bdiagrams?\b|\bscreenshots?\b", re.I)

Ranking = list[tuple[str, float]]
_ID_LIKE = re.compile(r"^[A-Za-z]{1,10}[-_]?\d")          # 编号形态（字母开头 + 数字），不含纯数字 / 年份
_MEANINGFUL = re.compile(r"[A-Za-z0-9一-鿿]")
_NUM_CODE = re.compile(r"(?<![A-Za-z0-9_.\-/])(?:\d+(?:-\d+)+|\d+(?:/\d+)+|\d+(?:\.\d+){2,})(?![A-Za-z0-9_.\-/])")   # 3-11、2025-0931、1.2.3（只认 ASCII 边界：\w 会把汉字也当成边界字符；单个小数点是小数，不算编号）
# 中性基线：通用词 + 与任何图都无关的泛问句。只用"图片 / 文档"这类词做基线时，带文字的图表对任何中文问句
# 都有 +0.05 左右的"虚高"（实测报告图表上无关问题最高 +0.089）；加入泛问句后无关问题最高 +0.064。
_NEUTRAL_PROMPTS = ("图片", "一张图片", "文档", "这是什么", "请问这个怎么办理", "公司的规定是什么", "我想了解一下相关情况", "有什么需要注意的")
# 请求用语：说明"要看图"，但不说明"哪张图相关"，不参与关键词证据
_REQUEST_WORDS = {"出示", "展示", "显示", "给我", "看看", "看一下", "一下", "相关", "图表", "图片", "图像", "截图", "照片", "图", "请", "帮", "帮我", "多少", "哪个", "哪些", "什么",
                  "问题", "出了", "怎么", "怎么样", "怎样", "是怎样", "如何", "是否", "可以", "吗", "呢", "情况", "有没有", "是什么", "为什么", "作用"}
# 图号 / 表号："Figure 20""Fig. 3""Table 11""图 3""表2"。分词后只剩"figure""20"两个普通词，BM25 和向量都认不出它是指哪张图
_LABEL_QUERY = re.compile(r"(?:\b(figure|fig\.?|table|tab\.?)\s*|(图|表)\s*)(\d{1,3})(?![\d.\-])", re.I)
_LABEL_CAPTION = re.compile(r"^\s*(?:(figure|fig\.?|table|tab\.?)\s*|(图|表)\s*)(\d{1,3})(?![\d\-])", re.I)
# 图注前面混进了图表自己的标题（"Zero-Shot CLIP vs. Linear Probe Figure 5. …"，MinerU standard 档实测）：
# 编号后紧跟句点 / 冒号的才是本图编号，"compared with Figure 5" 这类引用不算
_LABEL_CAPTION_INNER = re.compile(r"(?:\b(figure|fig\.|table)\s*|(图|表)\s*)(\d{1,3})\s*[.:：．](?!\d)", re.I)
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")
_LATIN_CHAR = re.compile(r"[A-Za-z]")
_TRANSLATIONS: dict[str, str] = {}      # (模型, 目标语言, 问题) → 译文；进程内缓存，检索器每次请求都会新建
_TRANSLATE_PROMPT = "把下面这句检索用的话翻译成{target}，用于在{target}资料里做关键词检索。保留专有名词、编号、数字；只输出译文。\n{query}"
_LABEL_KIND = {"figure": "figure", "fig": "figure", "fig.": "figure", "图": "figure", "table": "table", "tab": "table", "tab.": "table", "表": "table"}


def _zscore(x: Any) -> Any:
    x = np.asarray(x, dtype=np.float32)
    std = float(x.std())
    return (x - float(x.mean())) / std if std > 1e-8 else np.zeros_like(x)


def wants_images(query: str) -> bool:
    return bool(_WANTS_IMAGE.search(query or ""))


def cjk_share(text: str) -> float:
    """文本里中文的占比（按信息量粗算：4 个拉丁字母约等于 1 个汉字）。"""
    cjk, latin = len(_CJK_CHAR.findall(text or "")), len(_LATIN_CHAR.findall(text or ""))
    return cjk / (cjk + latin / 4) if cjk or latin else 0.0


def figure_labels(text: str, *, caption: bool = False) -> set[tuple[str, int]]:
    """文本里提到的图号 / 表号：{("figure", 3), ("table", 11)}。caption=True 时只认图注开头的那一个（图注正文里
    "compared with Figure 5" 这类引用不算这张图的编号）。"""
    if caption:
        m = _LABEL_CAPTION.match(text or "") or _LABEL_CAPTION_INNER.search(text or "")
        return {(_LABEL_KIND[(m.group(1) or m.group(2)).lower()], int(m.group(3)))} if m else set()
    return {(_LABEL_KIND[(m.group(1) or m.group(2)).lower()], int(m.group(3))) for m in _LABEL_QUERY.finditer(text or "")}


class MultimodalRetriever:
    def __init__(self, settings: Settings, index: Index, *, vlm=None, describer: ImageDescriber | None = None, lance=None, llm=None) -> None:
        self.settings = settings
        self.cfg: MultimodalConfig = settings.multimodal
        self.index = index
        self.images = index.images
        self.describer = describer or ImageDescriber(self.cfg, vlm)
        self.lance = lance                       # LanceDB 多模态表路（prism.lance.multimodal.MultimodalLanceRoute），None = 不接入
        self.llm = llm                           # 辅助文本模型：跨语言补译（multimodal.cross_lingual），None = 不补译
        self._neutral: Any = None
        self._df: tuple[int, dict[str, int]] | None = None
        self._vocab: tuple[int, set[str]] | None = None
        self._corpus_lang: tuple[int, float] | None = None

    # --- 文搜图 ---------------------------------------------------------------------------
    def text_to_image(self, query: str, *, top_k: int | None = None, doc_ids: list[str] | None = None) -> MultimodalResult:
        t0 = time.perf_counter()
        res = self._result("text2image", query=query)
        if not self._ready(res):
            return res
        allowed = self._allowed(doc_ids)
        rankings, weights = self._text_routes(query, allowed, res)
        pinned = self._label_hits(query, allowed, res) or self._exact_id_hits(query, res)
        # 关键词路几乎匹配不上查询（权重被覆盖率压低）时，它的第一名只是噪声，不能拿来和描述向量路"互相印证"
        if not pinned and weights.get("desc_sparse", 0.0) >= 0.5 * self.cfg.weight_desc_sparse:
            pinned = self._text_agreement(rankings, res)
        self._finish_images(res, rankings, weights, top_k, pinned=pinned)
        res.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        return res

    @staticmethod
    def _text_agreement(rankings: dict[str, Ranking], res: MultimodalResult) -> list[str]:
        """描述向量路和描述 BM25 路的第一名是同一张图：语义和字面都指向它，排最前。
        CLIP 文本塔对细粒度信息（"13-5" 和 "2-5"、具体的产品 / 区域组合）不敏感，等权 RRF 时它一路的噪声能推翻
        另外两路的一致结论（300 张图压测：按完整标题文搜图 7/60 因此排错）。"""
        dense, sparse = rankings.get("desc_dense") or [], rankings.get("desc_sparse") or []
        if dense and sparse and dense[0][0] == sparse[0][0]:
            top = rankings.get("clip") or []
            if top and top[0][0] != dense[0][0]:
                res.notes.append("描述的语义与关键词两路一致指向同一张图，优先于图文向量路")
            return [dense[0][0]]
        return []

    def _label_hits(self, query: str, allowed: set[str] | None, res: MultimodalResult | None = None) -> list[str]:
        """查询点名了图号 / 表号（"Figure 20""表 2"），图注以这个编号开头的图片排最前。
        CLIP 论文实测：按"Figure 20"文搜图三路都排不进前 5（它和 Figure 2 / 12 / 22 在词面和语义上几乎一样）。"""
        want = figure_labels(query)
        if not want:
            return []
        hits = [i for i, r in self.images.records.items() if (allowed is None or i in allowed) and figure_labels(r.caption, caption=True) & want]
        if hits and res is not None:
            res.notes.append(f"图号 / 表号精确命中：图注以 {'、'.join(f"{'图' if k == 'figure' else '表'} {n}" for k, n in sorted(want))} 开头的 {len(hits)} 张图排在最前")
        return hits

    def _exact_id_hits(self, query: str, res: MultimodalResult) -> list[str]:
        """查询里有编号时，图里确实写着这些编号的图片排最前。两类编号：
        - 字母 + 数字（SO-8817、RMA-5422、PUMP-07）：按分词结果匹配；
        - 数字分段码（3-11、2025-0931、1.2.3）：分词会把它拆成单个数字、丢掉顺序（"3-11" 与 "11-3" 无法区分），
          改为在图片文本（图注 / 描述 / 图中文字 / 上下文）里按完整字符串、带边界匹配。
        RRF 对这种查询不友好：300 张图压测里，正确的工单卡片在 BM25 路次次第一，却被"几路都中等靠前"的
        相似卡片反超，按编号文搜图只有 32% 命中第一；1000 张图里按"3-11"这类标题也有 3/60 排错。"""
        ids = {m.group(0).lower() for m in PRECISE_TOKEN.finditer(query) if _ID_LIKE.match(m.group(0))}
        codes = set(_NUM_CODE.findall(query))
        if not ids and not codes:
            return []
        texts = {r.image_id: r.search_text() for r in self.images.records.values()} if codes else {}
        hits = []
        for image_id, toks in zip(self.images.bm25.ids, self.images.bm25.corpus_tokens):
            if ids and not ids <= set(toks):
                continue
            if codes and not all(re.search(rf"(?<![\d.\-/]){re.escape(c)}(?![\d.\-/])", texts.get(image_id, "")) for c in codes):
                continue
            hits.append(image_id)
        if hits and len(hits) < len(self.images.bm25.ids):
            res.notes.append(f"编号精确命中：{'、'.join(sorted({*(x.upper() for x in ids), *codes}))} 出现在 {len(hits)} 张图里，排在最前")
            return hits
        return []

    def _identify(self, img_vec: Any, ocr_text: str, allowed: set[str] | None) -> tuple[list[tuple[str, float]], dict[str, Ranking], bool]:
        """查询图片是库里的哪一张：视觉相似度 + 图中文字（OCR → 图片描述 BM25）。
        视觉找得到"同类图"，文字才说得清"是哪一张"——300 张同风格图表、模拟手机翻拍的压测里，
        只看视觉 top1 30/60，只看图中文字 top1 60/60。返回 [(image_id, 余弦)] 按认定顺序排列、各路排名、是否用了文字。"""
        n = self.cfg.top_n_per_route
        visual = self.images.visual.search(img_vec, n, allowed)
        routes: dict[str, Ranking] = {"visual": visual}
        if len(_MEANINGFUL.findall(ocr_text or "")) < self.cfg.ocr_identify_min_chars or not len(self.images.bm25.ids):
            return visual, routes, False
        by_text = self.images.bm25.search(ocr_text, n, allowed)
        if not by_text:
            return visual, routes, False
        routes["ocr_sparse"] = by_text
        fused = [i for i, _ in rrf_fusion([(visual, 1.0), (by_text, self.cfg.weight_ocr_identify)], k=self.cfg.rrf_k)]
        # 图中文字明显指向某一张（BM25 第一名领先第二名 30% 以上）：直接认定，不让视觉上"更像"的同类图抢走
        if len(by_text) == 1 or by_text[0][1] >= 1.3 * by_text[1][1]:
            fused = [by_text[0][0]] + [i for i in fused if i != by_text[0][0]]
        cos = dict(visual)
        missing = [i for i in fused if i not in cos]
        if missing:
            cos.update({i: float(v @ img_vec) for i, v in self.images.visual.get_vectors(missing).items()})
        return [(i, cos.get(i, 0.0)) for i in fused], routes, True

    # --- 图搜图 ---------------------------------------------------------------------------
    def image_to_image(self, image: Any, *, top_k: int | None = None, min_score: float | None = None, doc_ids: list[str] | None = None, exclude_ids: set[str] | None = None) -> MultimodalResult:
        t0 = time.perf_counter()
        res = self._result("image2image", has_image=True)
        if not self._ready(res):
            return res
        top_k = top_k or self.cfg.top_k
        vec = self.images.embedder.embed_image(image)
        t = time.perf_counter()
        ocr_text = self.describer.ocr.read(image) if self.cfg.ocr_identify else ""
        res.timings_ms["ocr"] = (time.perf_counter() - t) * 1000
        ranking, routes, used_text = self._identify(vec, ocr_text, self._allowed(doc_ids))
        if used_text:
            res.query_description = f"图中文字：{ocr_text}"
            res.notes.append("图中有文字：视觉相似度 + 图中文字一起认定是哪张图（分数仍显示视觉余弦相似度）")
        if exclude_ids:
            ranking = [(i, s) for i, s in ranking if i not in exclude_ids]
        threshold = self.cfg.min_score if min_score is None else min_score
        if threshold is not None:
            kept = [(i, s) for i, s in ranking if s >= threshold]
            if len(kept) < len(ranking):
                res.notes.append(f"相似度阈值 {threshold}：{len(ranking)} → {len(kept)} 张")
            ranking = kept
        for route, items in routes.items():
            self._trace(res, route, items)
        pos = {route: {i: (k + 1, s) for k, (i, s) in enumerate(items)} for route, items in routes.items()}
        # 分数用视觉余弦相似度（便于设阈值 / 判重）；顺序按认定结果
        res.images = [ImageHit(record=self.images.records[i], score=float(s), rank=n + 1, routes={r: RouteHit(rank=p[i][0], score=float(p[i][1])) for r, p in pos.items() if i in p})
                      for n, (i, s) in enumerate(ranking[:top_k]) if i in self.images.records]
        res.fused = [ScoredChunk(chunk_id=h.record.image_id, score=h.score, rank=h.rank, route="identify") for h in res.images]
        res.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        return res

    # --- 图文混合 -------------------------------------------------------------------------
    def hybrid(self, query: str, image: Any, *, alpha: float | None = None, top_k: int | None = None, doc_ids: list[str] | None = None) -> MultimodalResult:
        """查询同时包含文本和图片（"像这张图，但要……"）。α 是文本权重。"""
        t0 = time.perf_counter()
        res = self._result("hybrid", query=query, has_image=True)
        if not self._ready(res):
            return res
        alpha = self.cfg.hybrid_alpha if alpha is None else min(1.0, max(0.0, alpha))
        allowed = self._allowed(doc_ids)
        emb = self.images.embedder
        n = self.cfg.top_n_per_route
        img_vec = emb.embed_image(image)
        rankings: dict[str, Ranking] = {}
        weights: dict[str, float] = {}
        if emb.cross_modal:
            # 图文空间融合：score = α·z(text·I) + (1-α)·z(image·I)。
            # 不能直接把两个查询向量相加——CLIP 有"模态间隙"：图-图余弦 0.7~1.0，文-图只有 0.3~0.5，
            # 原始相加时图片一侧哪怕只占 10% 权重也会压过文本。先在候选图片上各自标准化，α 才是真正的权重。
            ids = list(self.images.visual.ids)
            matrix = self.images.visual.matrix
            combined = alpha * _zscore(matrix @ emb.embed_text(query)) + (1 - alpha) * _zscore(matrix @ img_vec)
            order = np.argsort(-combined)
            rankings["clip_fused"] = [(ids[i], float(combined[i])) for i in order if allowed is None or ids[i] in allowed][:n]
            weights["clip_fused"] = self.cfg.weight_clip
            res.notes.append(f"图文空间融合：score = {alpha:.2f}·z(text·I) + {1 - alpha:.2f}·z(image·I)（各自标准化以消除模态间隙）")
        else:
            rankings["visual"] = self.images.visual.search(img_vec, n, allowed)
            weights["visual"] = self.cfg.weight_clip * 2 * (1 - alpha)
            res.notes.append("当前图文向量模型没有文本塔：改为晚融合（视觉相似路 + 描述文本两路，按 α 加权）")
        text_rankings, text_weights = self._desc_routes(query, allowed, res)
        rankings.update(text_rankings)
        weights.update({k: w * 2 * alpha for k, w in text_weights.items()})
        # 混合检索按"加权相对相关度"排序：RRF 只看排名，体现不出 α 的连续变化
        self._finish_images(res, rankings, weights, top_k, order_by="relevance")
        res.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        return res

    # --- 图搜文 ---------------------------------------------------------------------------
    def image_to_text(self, image: Any, *, query: str = "", top_k: int | None = None, describe: bool = True, recognized: tuple[str, str] | None = None, doc_ids: set[str] | None = None) -> MultimodalResult:
        """recognized：这张查询图已经识别过时的 (图中文字, 图片描述)——例如问答里自动路由换库后重做以图搜文。
        图中文字、图片描述与知识库无关，直接沿用，不再跑 OCR、不再调多模态模型；不给时照常识别。
        doc_ids：本轮可见的文档（过期 / 权限过滤后）；None 时只按 filter_expired 过滤过期文档。"""
        t0 = time.perf_counter()
        res = self._result("image2text", query=query, has_image=True)
        rcfg = self.settings.retrieval
        top_k = top_k or rcfg.rerank.top_k
        n = rcfg.top_n_per_route
        store = self.index.store
        if doc_ids is not None:
            allowed = store.filter_child_ids(lambda c, md: c.doc_id in doc_ids)
        else:
            allowed = store.effective_child_ids() if rcfg.filter_expired else None
        rankings: dict[str, Ranking] = {}
        weights: dict[str, float] = {}

        emb = self.images.embedder
        self.images.ensure_fresh()
        img_vec = emb.embed_image(image)

        # 路 1：图像塔 → 正文块的图文空间文本向量（直达，不经过任何文字中介）
        if emb.cross_modal and len(store):
            t = time.perf_counter()
            added, remaining = self.images.sync_chunk_vectors([store.chunks[cid] for cid in store.child_ids()], limit=self.cfg.chunk_sync_per_query)
            if remaining:
                res.notes.append(f"图文直达路还在补建：本次新建 {added} 个正文块的图文向量，还差 {remaining} 个，后续查询继续补齐（期间这一路只覆盖已建好的部分）")
            elif added:
                res.notes.append(f"图文直达路：为 {added} 个正文块补建了图文向量")
            rankings["clip:image→chunk"] = self.images.chunk_clip.search(img_vec, n, allowed)
            weights["clip:image→chunk"] = self.cfg.weight_clip
            res.timings_ms["clip_chunks"] = (time.perf_counter() - t) * 1000

        # 图中文字先认出来：既用于"认定是库里哪张图"，也用于路 3 的转文字检索
        if recognized is not None:
            ocr_text = recognized[0] if (describe or self.cfg.ocr_identify) else ""
            res.notes.append("沿用这张图已有的识别结果（图中文字 / 图片描述），不重复识别")
        else:
            t = time.perf_counter()
            ocr_text = self.describer.ocr.read(image) if (describe or self.cfg.ocr_identify) else ""
            res.timings_ms["ocr"] = (time.perf_counter() - t) * 1000
        res.ocr_text = ocr_text

        # 路 2：相似图片 → 它们所在的段落
        similar: list[tuple[str, float]] = []
        used_text = False
        if len(self.images):
            similar, _, used_text = self._identify(img_vec, ocr_text if self.cfg.ocr_identify else "", None)
        near_threshold = self.cfg.neighbor_min_score if self.cfg.neighbor_min_score is not None else emb.similar_threshold
        near = [(i, s) for i, s in similar if s >= near_threshold]
        if not near and used_text and similar and similar[0][1] >= self.cfg.neighbor_floor:
            near = [similar[0]]
            res.notes.append(f"按图中文字 + 视觉认定是库里的「{self.images.records[similar[0][0]].brief(20)}」（视觉相似度 {similar[0][1]:.2f}）")
        if not near and similar and similar[0][1] >= self.cfg.neighbor_floor:
            # 手机翻拍 / 截图转发的图和原图的余弦只有 0.67~0.88（实测），过不了 0.9 的"相似"门槛；
            # 但它明显领先第二名（实测领先 0.21~0.32，而同风格的无关图表之间往往扎堆）时，可以认定就是这张
            runner_up = similar[1][1] if len(similar) > 1 else 0.0
            if similar[0][1] - runner_up >= self.cfg.neighbor_margin:
                near = [similar[0]]
                res.notes.append(f"相似图片按「明显领先」认定：{similar[0][1]:.2f}，领先第二名 {similar[0][1] - runner_up:.2f}")
        neighbor: Ranking = []
        seen: set[str] = set()
        for image_id, score in near:
            rec = self.images.records.get(image_id)
            parent = store.get(rec.anchor_chunk_id) if rec and rec.anchor_chunk_id else None
            for cid in (parent.child_ids if parent else []):
                if cid not in seen and (allowed is None or cid in allowed):
                    seen.add(cid)
                    neighbor.append((cid, score))
        if neighbor:
            rankings["neighbor:image→image→chunk"] = neighbor
            weights["neighbor:image→image→chunk"] = 1.0
        res.images = [ImageHit(record=self.images.records[i], score=float(s), rank=k + 1, routes={"visual": RouteHit(rank=k + 1, score=float(s))}) for k, (i, s) in enumerate(near[: self.cfg.top_k]) if i in self.images.records]

        # 路 3：OCR / VLM 把查询图片转成文字，再走正文的向量 + BM25
        if describe:
            t = time.perf_counter()
            caption = recognized[1] if recognized is not None else self.describer.caption(image, _sha1(image), hint=ocr_text[:200])
            res.caption = caption
            res.query_description = "\n".join(x for x in (caption, f"图中文字：{ocr_text}" if ocr_text else "") if x)
            if recognized is None:
                res.timings_ms["describe"] = (time.perf_counter() - t) * 1000
        text_query = "\n".join(x for x in (query, res.query_description) if x).strip()
        if text_query:
            rankings["caption:dense"] = self.index.vectors.search(self.index.embedder.embed_query(text_query), n, allowed)
            rankings["caption:sparse"] = self.index.bm25.search(text_query, n, allowed)
            weights["caption:dense"] = rcfg.fusion.weight_dense
            weights["caption:sparse"] = rcfg.fusion.weight_sparse

        # 路 4：LanceDB 多模态表——正文行上"查询图的 CLIP 向量 + 图中文字 / 图片转文字"的原生混合检索。
        # 没有文字时它与直达路完全相同，不重复计票
        if self.lance is not None and self.cfg.weight_lance_mm > 0 and emb.cross_modal and text_query:
            t = time.perf_counter()
            rankings["lance_mm:image→text"] = self.lance.image_to_text(img_vec, text_query, n, allowed, res.notes)
            weights["lance_mm:image→text"] = self.cfg.weight_lance_mm
            res.timings_ms["lance_mm"] = (time.perf_counter() - t) * 1000

        for route, r in rankings.items():
            self._trace(res, route, r)
        fused = rrf_fusion([(r, weights[k]) for k, r in rankings.items() if r], k=rcfg.fusion.rrf_k)
        norm = sum(w / (rcfg.fusion.rrf_k + 1) for k, w in weights.items() if rankings.get(k)) or 1.0
        keep = max(rcfg.rerank.keep_n, top_k)
        scored = [ScoredChunk(chunk_id=cid, score=s / norm, rank=i + 1, route="fused") for i, (cid, s) in enumerate(fused[:keep])]
        own: set[str] = set()
        first = self.images.records.get(near[0][0]) if near else None
        if neighbor and first is not None and not query.strip() and scored:
            # 查询图已认定是库里的某张图、又没有附带问题："这张图讲的是什么"最好的答案就是它所在的段落。
            # RRF 里这一路只占一票，CLIP 论文实测常被其他几路共同偏好的泛泛段落（附录、方法综述）挤下去，
            # 甚至在上下文预算里被整段裁掉——所以要在展开成父块、按预算取舍之前就把它提到最前。
            # 只认第一张：同风格图表之间余弦常在 0.9 以上，"相似"的图有好几张，它们的段落全顶上来反而挤掉正确的那段
            parent = store.get(first.anchor_chunk_id) if first.anchor_chunk_id else None
            own = set(parent.child_ids) if parent else set()
            top = scored[0].score
            mine = [sc.model_copy(update={"score": max(sc.score, top)}) for sc in scored if sc.chunk_id in own]
            if mine:
                scored = mine + [sc for sc in scored if sc.chunk_id not in own]
                for i, sc in enumerate(scored):
                    sc.rank = i + 1
                res.notes.append(f"查询图认定为库里的「{first.brief(20)}」：它所在的段落排在最前")
        res.fused = scored
        contexts = expand_contexts(scored, store, rcfg.granularity)
        res.contexts = sorted(contexts, key=lambda c: (bool(own & set(c.matched_child_ids)), c.score), reverse=True)[:top_k]
        if not rankings:
            caps = self.describer.capabilities
            res.notes.append("图搜文没有可用信号：" + "；".join(x for x in (
                "" if emb.cross_modal else "图文向量模型没有文本塔（安装 torch + transformers 启用 CLIP）",
                "OCR 没有在图片里识别出文字" if caps["ocr"] else "未安装 OCR（pip install rapidocr-onnxruntime）",
                "" if caps["vlm"] else "未配置多模态模型（VLM）",
                "" if near else "知识库里没有足够相似的图片",
            ) if x))
        res.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        return res

    # --- 问答：决定一轮回答带哪些图 ----------------------------------------------------------
    def images_for_answer(self, query: str, contexts: list[Context], *, limit: int | None = None, doc_ids: set[str] | list[str] | None = None) -> tuple[list[ImageHit], MultimodalResult]:
        """端到端 RAG 的图片侧——决定这一轮回答带哪些图。

        文本检索总会返回 top-K，文搜图也总会返回 top-K，但"有结果"不等于"相关"：
        不设门槛的话每个回答下面都会挂几张不相干的图。三条准入规则：
        1) 图文关联：图片所在段落被文本检索命中，且图片描述与问题有词面 / 语义证据（门槛 × 0.75）；
        2) 文搜图高置信：排名前 3，且关键词覆盖率或图文向量相似度达到门槛；
        3) 显式要图（"给我看架构图"）：取第一名，以及相对相关度与第一名接近（≥ 85%）的命中。
        证据的优先级：只要有任何一张图在关键词上达标（问题里的实体 / 编号 / 数值确实写在图里），
        其余图片就只看关键词证据——CLIP 对"都是图表"的图普遍给分偏高，此时它的判断只会带进无关图。"""
        limit = self.cfg.chat_images if limit is None else limit
        if doc_ids is not None and not doc_ids:
            # 文档过滤（过期 / 权限）后一篇都不剩：不能把空集当成"不过滤"
            return [], self._result("text2image", query=query)
        docs = sorted(doc_ids) if doc_ids is not None else None
        res = self.text_to_image(query, top_k=max(limit, self.cfg.top_k), doc_ids=docs) if query.strip() else self._result("text2image", query=query)
        if limit <= 0 or not len(self.images) or not res.images:
            return [], res
        q_tokens = set(self.images.bm25.tokenize(query)) - _REQUEST_WORDS
        anchors = {c.chunk.chunk_id for c in contexts if not c.weaved}
        explicit = wants_images(query)
        clip_base = self._clip_query_baseline(query, res)
        labeled = set(self._label_hits(query, self._allowed(docs)))
        scored = [(hit, *self._evidence_parts(hit, q_tokens, clip_base=clip_base)) for hit in res.images]
        translated = self._translate(query)
        if translated:
            # 跨语言：原问题的词在图片文字里本来就找不到，覆盖率另用译文的词算一遍，取较高者
            t_tokens = set(self.images.bm25.tokenize(translated)) - _REQUEST_WORDS
            scored = [(hit, max(lex, self._evidence_parts(hit, t_tokens)[0]), clip_ev) for hit, lex, clip_ev in scored]
        # 点名的图号 / 表号（"图 3 说明了什么"）本身就是最强的词面证据
        scored = [(hit, max(lex, 1.0) if hit.record.image_id in labeled else lex, clip_ev) for hit, lex, clip_ev in scored]
        lexical_hit = any(lex >= 1.0 for _, lex, _ in scored)
        out: list[ImageHit] = []
        for hit, lex, clip_ev in scored:
            linked = hit.record.anchor_chunk_id in anchors
            evidence = lex if lexical_hit else max(lex, clip_ev)
            if linked and evidence >= 0.75:
                out.append(hit.model_copy(update={"linked": True}))
            elif evidence >= 1.0 and hit.rank <= 3:
                out.append(hit)
            elif explicit and (hit.rank == 1 or hit.score >= 0.85 * res.images[0].score):
                out.append(hit)
        if len(out) < len(res.images):
            res.notes.append(f"回答附图：{len(res.images)} 张候选中 {len(out)} 张通过相关性门槛" + ("（显式要图）" if explicit else ""))
        out = out[:limit]
        for i, h in enumerate(out):
            h.rank = i + 1
        return out, res

    def _evidence(self, hit: ImageHit, q_tokens: set[str]) -> float:
        """图片与问题相关的证据强度：1.0 = 达到门槛。取关键词覆盖率与图文向量相似度中较强的一项。"""
        return max(self._evidence_parts(hit, q_tokens))

    def _evidence_parts(self, hit: ImageHit, q_tokens: set[str], *, clip_base: float = 0.0) -> tuple[float, float]:
        """(关键词证据, 图文向量证据)，都以 1.0 为达标。clip_base 见 _clip_query_baseline。"""
        lexical = clip_ev = 0.0
        clip = hit.routes.get("clip")
        if clip is not None and self.cfg.chat_clip_margin > 0:
            # 这张图对"中性提示词"的平均相似度是它的基线；问题比基线高出多少才是证据
            vec = self.images.visual.get_vectors([hit.record.image_id]).get(hit.record.image_id)
            if vec is not None:
                clip_ev = (clip.score - float(vec @ self._neutral_vector()) - clip_base) / self.cfg.chat_clip_margin
        if "desc_sparse" in hit.routes and q_tokens and self.cfg.chat_term_coverage > 0:
            ids = self.images.bm25.ids
            row = ids.index(hit.record.image_id) if hit.record.image_id in ids else -1
            doc_tokens = set(self.images.bm25.corpus_tokens[row]) if row >= 0 else set()
            # 按 IDF 加权的覆盖率："可以""使用"这类在正文里到处都是的词几乎不算证据，"华东区""PUMP-07"才算
            idf = self._idf(q_tokens)
            total = sum(idf.values()) or 1.0
            lexical = sum(w for t, w in idf.items() if t in doc_tokens) / total / self.cfg.chat_term_coverage
        return lexical, clip_ev

    def _clip_query_baseline(self, query: str, res: MultimodalResult) -> float:
        """问题一侧的基线：这个问题对库里"典型图片"高出中性基线多少（各图超出量的中位数）。
        中性基线只校正了"这张图对什么问题都偏高"；英文论文的图表遇到学术类提问时，所有图一起偏高——
        CLIP 论文实测，"作者来自哪个机构"这类与图无关的问题，每张图都比中性基线高 1.3~1.6 个门槛，回答下挂满无关图。
        减去中位数后只剩"这张图比别的图更贴近问题"的部分。图片太少时中位数就是候选本身，不做这一步。"""
        if not query.strip() or len(self.images) < 8 or not self.images.embedder.cross_modal or "clip" not in res.routes:
            return 0.0
        matrix = self.images.visual.matrix
        excess = matrix @ self.images.embedder.embed_text(query) - matrix @ self._neutral_vector()
        return max(0.0, float(np.median(excess)))

    def _sparse_coverage(self, query: str) -> float:
        """查询的信息量（按 IDF 加权）有多少落在图片文本的词表里。中文提问、图片描述是英文时，BM25 只能靠
        "CLIP"这种每张图都有的词打分，排名纯属噪声（CLIP 论文实测：等权参与 RRF 时文搜图 hit@3 从 0.81 掉到 0.62）。"""
        tokens = set(self.images.bm25.tokenize(query)) - _REQUEST_WORDS
        if not tokens:
            return 1.0
        corpus = self.images.bm25.corpus_tokens
        if self._vocab is None or self._vocab[0] != len(corpus):
            self._vocab = (len(corpus), {t for toks in corpus for t in toks})
        idf = self._idf(tokens)
        return sum(w for t, w in idf.items() if t in self._vocab[1]) / (sum(idf.values()) or 1.0)

    def _idf(self, tokens: set[str]) -> dict[str, float]:
        """IDF 取自正文 + 图片描述的 BM25 语料。语料里没出现过的查询词按最稀有处理（它们也不在这张图的描述里，
        理应拉低覆盖率——否则"年假"被切成"年 / 假"、只剩"年"能匹配时，覆盖率会虚高到 100%）。"""
        corpus = self.index.bm25.corpus_tokens + self.images.bm25.corpus_tokens
        if self._df is None or self._df[0] != len(corpus):
            df: dict[str, int] = {}
            for toks in corpus:
                for t in set(toks):
                    df[t] = df.get(t, 0) + 1
            self._df = (len(corpus), df)
        n, df = self._df
        return {t: float(np.log((n + 1) / (df.get(t, 0) + 0.5))) for t in tokens}

    def _neutral_vector(self) -> Any:
        if self._neutral is None:
            self._neutral = self.images.embedder.embed_texts(list(_NEUTRAL_PROMPTS)).mean(axis=0)
        return self._neutral

    # --- 内部 -----------------------------------------------------------------------------
    def _result(self, mode: str, **kw: Any) -> MultimodalResult:
        return MultimodalResult(mode=mode, **kw)  # type: ignore[arg-type]

    def _ready(self, res: MultimodalResult) -> bool:
        if not len(self.images):
            res.notes.append("知识库里还没有图片：上传含图片的 PDF / Word / PPT / Markdown，或直接上传图片文件")
            return False
        self.images.ensure_fresh()
        res.embedder = self.images.embedder.signature()
        return True

    def _allowed(self, doc_ids: list[str] | set[str] | None) -> set[str] | None:
        """None = 不过滤；空列表 = 一篇都不可见（过期 / 权限过滤后什么都不剩），不能当成不过滤。"""
        if doc_ids is None:
            return None
        want = set(doc_ids)
        return {i for i, r in self.images.records.items() if r.doc_id in want}

    def _translate(self, query: str, res: MultimodalResult | None = None) -> str:
        """问题和图片文字不是同一种语言、关键词路又几乎匹配不上时，把问题译成图片文字的语言；否则返回空串。
        只在真需要时调用模型：同语言、或者原问题的词已经能在图片文字里找到一半信息量，都不译。"""
        if not (self.cfg.cross_lingual and self.llm is not None and query.strip() and len(self.images)):
            return ""
        corpus = self._corpus_cjk_share()
        if abs(cjk_share(query) - corpus) < 0.5 or self._sparse_coverage(query) >= 0.5:
            return ""
        target = "英文" if corpus < 0.5 else "简体中文"
        key = f"{getattr(self.llm, 'model', '')}|{target}|{query.strip()}"
        if key not in _TRANSLATIONS:
            t = time.perf_counter()
            try:
                text = self.llm.chat([self.llm.user_message(_TRANSLATE_PROMPT.format(target=target, query=query.strip()))], max_tokens=200).text.strip()
            except Exception as exc:
                logger.warning("跨语言补译失败：%s", exc)
                return ""
            if res is not None:
                res.timings_ms["translate"] = (time.perf_counter() - t) * 1000
            if len(_TRANSLATIONS) > 2000:
                _TRANSLATIONS.clear()
            _TRANSLATIONS[key] = "" if text == query.strip() else text.splitlines()[0][:300] if text else ""
        if _TRANSLATIONS[key] and res is not None:
            res.notes.append(f"问题与图片文字语言不同：译为{target}补充检索——{_TRANSLATIONS[key]}")
        return _TRANSLATIONS[key]

    def _corpus_cjk_share(self) -> float:
        """图片文字（图注 + 描述 + 图中文字）的中文占比；按图片数缓存。"""
        if self._corpus_lang is None or self._corpus_lang[0] != len(self.images):
            text = " ".join(f"{r.caption} {r.description} {r.ocr_text}" for r in self.images.records.values())
            self._corpus_lang = (len(self.images), cjk_share(text))
        return self._corpus_lang[1]

    def _desc_routes(self, query: str, allowed: set[str] | None, res: MultimodalResult | None = None) -> tuple[dict[str, Ranking], dict[str, float]]:
        n = self.cfg.top_n_per_route
        translated = self._translate(query, res)
        if translated:
            query = f"{query}\n{translated}"
        rankings = {
            "desc_dense": self.images.desc.search(self.images.text_embedder.embed_query(query), n, allowed),
            "desc_sparse": self.images.bm25.search(query, n, allowed),
        }
        # 哈希向量只反映词面重叠，与 BM25 重复且排序噪声大：离线模式下降权，别让它盖过图文向量路
        dense_w = self.cfg.weight_desc_dense * (0.3 if self.images.text_embedder.name == "hash" else 1.0)
        # 关键词路按覆盖率降权：查询一半以上的信息量能在图片文本里找到时满权重
        sparse_w = self.cfg.weight_desc_sparse * min(1.0, self._sparse_coverage(query) / 0.5) if rankings["desc_sparse"] else self.cfg.weight_desc_sparse
        return rankings, {"desc_dense": dense_w, "desc_sparse": sparse_w}

    def _text_routes(self, query: str, allowed: set[str] | None, res: MultimodalResult) -> tuple[dict[str, Ranking], dict[str, float]]:
        rankings: dict[str, Ranking] = {}
        weights: dict[str, float] = {}
        emb = self.images.embedder
        if emb.cross_modal:
            rankings["clip"] = self.images.visual.search(emb.embed_text(query), self.cfg.top_n_per_route, allowed)
            weights["clip"] = self.cfg.weight_clip
        else:
            res.notes.append("当前图文向量模型没有文本塔（离线感知特征）：文搜图只走图片描述的向量 + BM25 两路；安装 torch + transformers 后自动启用 CLIP 图文向量路")
        d_rank, d_w = self._desc_routes(query, allowed, res)
        rankings.update(d_rank)
        weights.update(d_w)
        if self.lance is not None and self.cfg.weight_lance_mm > 0:
            # LanceDB 多模态表：表内图片文字（图注 / 描述 / OCR / 上下文）的全文检索，与 jieba BM25 是两套分词，互为补充。
            # 同属词面证据，和描述 BM25 路一样按覆盖率降权（中文提问对英文图注时只会命中到处都有的词）
            t = time.perf_counter()
            rankings["lance_mm"] = self.lance.text_to_image(query, self.cfg.top_n_per_route, allowed, res.notes)
            weights["lance_mm"] = self.cfg.weight_lance_mm * min(1.0, self._sparse_coverage(query) / 0.5)
            res.timings_ms["lance_mm"] = (time.perf_counter() - t) * 1000
        return rankings, weights

    def _finish_images(self, res: MultimodalResult, rankings: dict[str, Ranking], weights: dict[str, float], top_k: int | None, order_by: str = "rrf", pinned: list[str] | None = None) -> None:
        top_k = top_k or self.cfg.top_k
        for route, r in rankings.items():
            self._trace(res, route, r)
        active = {k: r for k, r in rankings.items() if r and weights[k] > 0}
        # 排序用 RRF（只看排名，对各路分数尺度不敏感）
        fused = rrf_fusion([(r, weights[k]) for k, r in active.items()], k=self.cfg.rrf_k)
        # 展示分用"相对相关度"：每一路得分 / 该路最高分，按权重平均，没上榜的路记 0。
        # RRF 原始分彼此只差 1/61 与 1/62，看不出强弱；相对相关度能看出"第一名遥遥领先"还是"都差不多"。
        pos = {route: {i: (n + 1, s) for n, (i, s) in enumerate(r)} for route, r in rankings.items()}
        top = {route: max(r[0][1], 1e-9) for route, r in active.items()}
        total_w = sum(weights[k] for k in active) or 1.0

        def relevance(image_id: str) -> float:
            return sum(weights[k] * max(0.0, pos[k][image_id][1]) / top[k] for k in active if image_id in pos[k]) / total_w

        if order_by == "relevance":
            fused = sorted(((i, relevance(i)) for i, _ in fused), key=lambda x: x[1], reverse=True)
        if pinned:
            pin = set(pinned)
            fused = [x for x in fused if x[0] in pin] + [x for x in fused if x[0] not in pin]
        hits: list[ImageHit] = []
        for n, (image_id, _rrf) in enumerate(fused[:top_k]):
            rec = self.images.records.get(image_id)
            if rec is None:
                continue
            routes = {route: RouteHit(rank=p[image_id][0], score=float(p[image_id][1])) for route, p in pos.items() if image_id in p}
            hits.append(ImageHit(record=rec, score=min(1.0, relevance(image_id)), rank=n + 1, routes=routes))
        res.images = hits
        res.fused = [ScoredChunk(chunk_id=i, score=s, rank=n + 1, route=order_by) for n, (i, s) in enumerate(fused[: self.cfg.top_n_per_route])]

    @staticmethod
    def _trace(res: MultimodalResult, route: str, ranking: Ranking) -> None:
        res.routes[route] = [ScoredChunk(chunk_id=i, score=float(s), rank=n + 1, route=route) for n, (i, s) in enumerate(ranking)]


def _sha1(image: Any) -> str:
    from .extract import pixel_sha1

    return pixel_sha1(image)


def image_record_view(rec: ImageRecord) -> dict[str, Any]:
    """CLI / 调试用的紧凑视图。"""
    return {"image_id": rec.image_id, "label": rec.label(), "brief": rec.brief()}
